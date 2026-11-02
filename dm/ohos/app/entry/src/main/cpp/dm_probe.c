// 三个真机实验的实现。见 dm_probe.h。
#define _GNU_SOURCE

#include "dm_probe.h"

#include <dlfcn.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <hilog/log.h>
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>
#include <poll.h>              // 子进程隔离跑构造函数（等审计管道）
#include <sys/wait.h>
#include <signal.h>
#include <errno.h>

#include "dm.h"                 // module_t
#include "map_image_ohos.h"     // 鸿蒙版 map_image —— 实验 D 的被测对象
#include "dm_fs_ohos.h"         // 最小 nn::fs（romfs → 宿主目录；导入按符号名分流到这里）
#include "dm_sdk_ohos.h"        // 通用 SDK 胶水层（vi/oe 等轻量语义；与游戏无关）
#include "dm_libc_ohos.h"       // 通用 libc 胶水层（memcpy/str*/math 按符号名直连宿主）
#include "dm_svc_sync.h"        // 通用 svc 同步原语（事件/仲裁锁/进程级 key/多对象等待）
#include "dm_nvn_state_ohos.h"
#include "dm_nvn_dev_ohos.h"
#include "dm_nvn_cb_ohos.h"
#include "dm_nvn_tex_ohos.h"
#include "dm_nvn_exec_ohos.h"
#include "dm_nvn_ohos.h"        // 通用 NVN 胶水层（设备族等平台中立语义）
#include "dm_nvn_vk.h"         // 通用 NVN 内存池/缓冲胶水层（UMA 直通）
#include "dm_main_layout.h"     // 旷野之息 main 的布局常数（fixups 表头生成）
#include "dm_xmods_gen.h"       // 额外模块装载表（gen_embed --xmods-header 生成；换游戏只换它）
#include "dm_main_gates.h"      // 旷野之息 main 的门表（93 个 svc 门，实验 H 登记）

// np113：旧的"任务状态 1→2 补推"开关。默认 0（关闭）——它是"记录全零时代"的拐杖，
// libc 修好后真机证明它会把任务直接送进销毁流程（详见 hook_e8 里的注释）。
#ifndef NP_STATE_HACK
#define NP_STATE_HACK 0
#endif

// 旷野之息 main 的嵌入段（dm_images.S）：装载验证要把「映射区」与「嵌入段」逐字节对拍
extern const unsigned char __start_dmmainx0[];
extern const unsigned char __stop_dmmainx0[];
extern const unsigned char __start_dmmainw[];
extern const unsigned char __stop_dmmainw[];
extern const unsigned char __start_dmmainx1[];
extern const unsigned char __stop_dmmainx1[];

// ───────────────────────── 输出小工具 ─────────────────────────

// 每写一行，同时打进 hilog —— 万一界面没显示出来（或进程中途挂了），
// 也能从设备日志里看到跑到哪一步了。
// ⚠ np211c（实验性，可回滚）：OH_LOG_Print **临时禁用**。
//   依据：np211b 寄存器快照实测——runner（主线程）789/789 样本停在 musl ldso 的
//   dso 符号查找（musl+0x14a6f4，x0=NULL/x1=栈/x3=8/lr 在 musl 内）。
//   特征 = 某宿主侧导入经 musl lazy PLT 反复解析失败（musl 对失败不缓存）。
//   头号嫌疑就是本函数：klog 每一行都经它调 OH_LOG_Print。若禁掉后主线程 PC
//   立即改变/游戏前进 ⇒ 实锤；若不是它，下一轮按同样办法逐个排除。
//   恢复方法：把 g_hilog_off 改回 0（或删掉这个 if）。
static int g_hilog_off = 0;   // np211c 实验已结束（OH_LOG_Print 排除，已恢复）；改 1 可再禁用
static void logline(const char *s) {
    if (g_hilog_off) return;
    OH_LOG_Print(LOG_APP, LOG_INFO, 0xD001, "dmprobe", "%{public}s", s);
}

static void app(char *out, size_t cap, size_t *used, const char *s) {
    logline(s);
    size_t n = strlen(s);
    if (*used + n + 1 > cap) n = (cap > *used + 1) ? cap - *used - 1 : 0;
    memcpy(out + *used, s, n);
    *used += n;
    out[*used] = 0;
}

static void appf(char *out, size_t cap, size_t *used, const char *fmt, ...) {
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    app(out, cap, used, line);
}

// ──────────── 实验 C 的素材：一个自定义 ELF 段 ────────────
// 段名必须是合法的 C 标识符，链接器才会为它生成 __start_<段名> / __stop_<段名>。
// used 防止被 --gc-sections 丢掉。

__attribute__((section("dmprobesec"), used))
const char dm_probe_marker[] = "DM_PROBE_MARKER_0123456789ABCDEF";

extern const char __start_dmprobesec[];
extern const char __stop_dmprobesec[];

// 实验 B 用：本文件里的一个符号，用来反查本模块 .so 的路径
static void probe_self_symbol(void) {}

// ─────────────────────── 实验 A ───────────────────────

// 执行测试用的信号兜底：页不可执行时 CPU 抛 SIGSEGV/SIGBUS，
// 处理器跳回来，进程不至于挂掉。
static sigjmp_buf g_probe_jmp;
static volatile sig_atomic_t g_probe_jmp_armed;

static void probe_segv_handler(int sig) {
    if (g_probe_jmp_armed) {
        g_probe_jmp_armed = 0;
        siglongjmp(g_probe_jmp, 1);
    }
    _exit(128 + sig);   // 没武装：交回默认处理
}

static void exp_a(char *out, size_t cap, size_t *used) {
    size_t pg = (size_t)sysconf(_SC_PAGESIZE);

    app(out, cap, used, "【实验 A】匿名内存能否拿到可执行权限\n");
    app(out, cap, used, "  预期：失败（鸿蒙自 5.0.0(12) 起禁止匿名内存可执行）\n");
    appf(out, cap, used, "  页大小 = %zu 字节\n", pg);

    void *p = mmap(NULL, pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        appf(out, cap, used, "  ✗ mmap 匿名 RW 就失败了：%s\n", strerror(errno));
        app(out, cap, used, "  结论：连匿名内存都拿不到，情况比预期更严\n\n");
        return;
    }
    appf(out, cap, used, "  ✓ mmap 匿名 RW：成功，地址 %p\n", p);

    // 顺序要紧：先把指令写进去，再 mprotect 成 R+X。
    // 反过来（先 mprotect 再写）会往只读页写 → 当场 SIGSEGV（第一次跑就死在这儿）。
    memcpy(p, "\xc0\x03\x5f\xd6", 4);   // ARM64: ret

    if (mprotect(p, pg, PROT_READ | PROT_EXEC) != 0) {
        appf(out, cap, used, "  ✓ mprotect 成 R+X：失败（%s）\n", strerror(errno));
        app(out, cap, used, "  结论：符合预期。匿名可执行内存被禁 → 桌面版 map_image() 的写法"
                            "在鸿蒙上行不通，必须走文件映射路线。\n\n");
        munmap(p, pg);
        return;
    }

    // mprotect 说成功 —— 但那不一定代表页真的可执行（有些系统在缺页时才拦）。
    // 真执行一次才算数。
    //
    // 隔离方式：不用 fork()（实测在鸿蒙 App 里 fork 会把整个进程带崩），
    // 改用信号处理器 + siglongjmp：真崩了就跳回来，进程继续活。
    app(out, cap, used, "  ⚠ mprotect 成 R+X：成功  ← 与预期不符！再验一下是否真能执行\n");

    struct sigaction sa, old_segv, old_bus;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = probe_segv_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER;
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS, &sa, &old_bus);

    if (sigsetjmp(g_probe_jmp, 1) == 0) {
        g_probe_jmp_armed = 1;
        ((void (*)(void))p)();
        g_probe_jmp_armed = 0;
        app(out, cap, used, "  ⚠⚠ 匿名内存真的可执行！桌面版写法可用，方案可以简化\n");
    } else {
        app(out, cap, used, "  ✓ 执行时崩了（SIGSEGV/SIGBUS）→ 页其实不可执行，"
                            "mprotect 只是没当场拦\n");
    }
    sigaction(SIGSEGV, &old_segv, NULL);
    sigaction(SIGBUS, &old_bus, NULL);

    app(out, cap, used, "\n");
    munmap(p, pg);
}

// ─────────────────────── 实验 B ───────────────────────

static void exp_b(const char *files_dir, char *out, size_t cap, size_t *used) {
    size_t pg = (size_t)sysconf(_SC_PAGESIZE);

    app(out, cap, used, "【实验 B】mmap 自己的 .so 文件能否拿到可执行权限\n");
    app(out, cap, used, "  预期：成功 —— 这是整个装载方案的基石\n");

    Dl_info info;
    if (!dladdr((void *)probe_self_symbol, &info) || !info.dli_fname) {
        app(out, cap, used, "  ✗ dladdr 取不到本模块路径\n\n");
        return;
    }
    appf(out, cap, used, "  本模块 .so 路径：%s\n", info.dli_fname);

    int fd = open(info.dli_fname, O_RDONLY);
    if (fd < 0) {
        appf(out, cap, used, "  ✗ 打不开：%s\n\n", strerror(errno));
        return;
    }

    // B-1：最简单的形式 —— 文件页直接映射成 R+X
    void *q = mmap(NULL, pg, PROT_READ | PROT_EXEC, MAP_PRIVATE, fd, 0);
    int b1_ok = (q != MAP_FAILED);
    if (!b1_ok) {
        appf(out, cap, used, "  ✗ B-1 mmap bundle 里的 .so 成 R+X：失败（%s）\n", strerror(errno));
    } else {
        appf(out, cap, used, "  ✓ B-1 mmap bundle 里的 .so 成 R+X：成功，地址 %p\n", q);
        munmap(q, pg);

        // B-2：完整机制演练 —— 先预留连续地址，再把文件页 MAP_FIXED 进去
        //     这正是替代 iOS 的 vm_allocate + vm_remap 的做法
        size_t total = pg * 4;
        void *res = mmap(NULL, total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (res == MAP_FAILED) {
            appf(out, cap, used, "  ✗ B-2 预留地址失败：%s\n", strerror(errno));
        } else {
            void *want = (char *)res + pg;
            void *got = mmap(want, pg, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_FIXED, fd, 0);
            if (got == MAP_FAILED)
                appf(out, cap, used, "  ✗ B-2 MAP_FIXED 进预留区：失败（%s）\n", strerror(errno));
            else if (got != want)
                appf(out, cap, used, "  ✗ B-2 落在了别处：想要 %p，得到 %p\n", want, got);
            else
                app(out, cap, used, "  ✓ B-2 预留 + MAP_FIXED：成功 —— "
                                    "「按原相对位置摆回去」的机制可用\n");
            munmap(res, total);
        }
    }
    close(fd);

    // B-3：换一个「应用数据目录」里的文件再试一次。
    // 用来分辨：禁令是针对 bundle 目录（.so 所在处），还是所有应用文件都不许可执行映射。
    // 这决定了方案是"换个地方放镜像"就行，还是要整个换路线。
    int b3_ok = 0, b3_tried = 0;
    if (files_dir && *files_dir) {
        b3_tried = 1;
        char tpath[1024];
        snprintf(tpath, sizeof tpath, "%s/probe_b3.bin", files_dir);
        int tf = open(tpath, O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (tf < 0) {
            appf(out, cap, used, "  ✗ B-3 数据目录建文件失败（%s）：%s\n", tpath, strerror(errno));
        } else {
            char buf[4096];
            memset(buf, 0, sizeof buf);
            memcpy(buf, "\xc0\x03\x5f\xd6", 4);   // ARM64: ret
            if (write(tf, buf, sizeof buf) != (ssize_t)sizeof buf) {
                appf(out, cap, used, "  ✗ B-3 写测试文件失败：%s\n", strerror(errno));
            } else {
                void *r = mmap(NULL, pg, PROT_READ | PROT_EXEC, MAP_PRIVATE, tf, 0);
                if (r == MAP_FAILED) {
                    appf(out, cap, used, "  ✗ B-3 mmap 数据目录文件成 R+X：失败（%s）\n",
                         strerror(errno));
                } else {
                    b3_ok = 1;
                    appf(out, cap, used, "  ✓ B-3 mmap 数据目录文件成 R+X：成功，地址 %p\n", r);
                    munmap(r, pg);
                }
            }
            close(tf);
        }
    }

    // 汇总
    if (b1_ok) {
        app(out, cap, used, "  结论：文件映射路线成立，装载方案按原设计推进。\n");
    } else if (b3_ok) {
        app(out, cap, used, "  结论：bundle 目录不许可执行映射，但数据目录可以 →\n"
                            "        把镜像放到应用数据目录再映射即可，方案仍然成立。\n");
    } else if (b3_tried) {
        app(out, cap, used, "  结论：bundle 与数据目录都不许 → 文件映射路线整体不可用，\n"
                            "        必须换方案（候选：镜像放进 .so 自己的可执行段，\n"
                            "        用链接脚本强制三块在文件里相邻，不搬位置）。\n");
    } else {
        app(out, cap, used, "  结论：B-3 未跑（没拿到数据目录路径），无法判断禁令范围。\n");
    }
    app(out, cap, used, "\n");
}

// ─────────────────────── 实验 C ───────────────────────

static void exp_c(char *out, size_t cap, size_t *used) {
    app(out, cap, used, "【实验 C】自定义 ELF 段能否生成 __start_/__stop_ 符号\n");
    app(out, cap, used, "  预期：能（lld 的标准行为）\n");

    size_t n = (size_t)(__stop_dmprobesec - __start_dmprobesec);
    appf(out, cap, used, "  __start_dmprobesec = %p\n", (const void *)__start_dmprobesec);
    appf(out, cap, used, "  __stop_dmprobesec  = %p\n", (const void *)__stop_dmprobesec);
    appf(out, cap, used, "  段长度 = %zu 字节\n", n);

    if (n == 0) {
        app(out, cap, used, "  ✗ 段长度为 0 → 链接器没生成符号，或段被丢掉了\n");
        app(out, cap, used, "  结论：要用退路（运行时解析 ELF 头，或链接脚本强制布局）\n");
    } else if (n >= 32 && memcmp(__start_dmprobesec, "DM_PROBE_MARKER", 15) == 0) {
        appf(out, cap, used, "  段内容前 16 字节：%.16s\n", __start_dmprobesec);
        app(out, cap, used, "  ✓ 符号可用，内容与标记匹配\n");
        app(out, cap, used, "  结论：嵌入用「自定义段 + 链接器符号」可行，"
                            "不必自己解析 ELF 头。\n");
    } else {
        app(out, cap, used, "  ⚠ 段有长度，但内容对不上标记 —— 需要细看\n");
    }
    app(out, cap, used, "\n");
}

// ─────────── 实验 D：完整链路（嵌入 → 定位 → 映射 → 执行）───────────
// 用 gen_embed_ohos.py 生成的 test_image.S 当"游戏映像"：
//   偏移 0x0000  代码 mov w0,#0x1234 ; ret
//   偏移 0x4000  数据 "DM_OHOS_IMAGE_OK"
//   TEXT_END 0x2000 / 键页 0x4000 / PATCH_OFF 0x8000 / 共 0xC000
// 走的是真正的 dm_ohos_map_image_alloc/seal（dm/ohos/map_image_ohos.c）。
static void exp_d(char *out, size_t cap, size_t *used) {
    app(out, cap, used, "【实验 D】完整链路：嵌入 → 定位 → 映射 → 执行\n");
    app(out, cap, used, "  用 gen_embed_ohos.py 生成的测试映像走真正的 map_image\n");

    module_t m;
    memset(&m, 0, sizeof m);
    strncpy(m.name, "probe", sizeof m.name - 1);   // 测试映像（test_image.S，段名 dmprobe*）；真实模块是 main（实验 G/H）
    m.text_end = 0x2000;
    m.patch_off = 0x8000;
    m.key_page_off = 0x4000;
    m.key_page_size = 0x4000;

    if (dm_ohos_map_image_alloc(&m)) {
        app(out, cap, used, "  ✗ alloc 失败（见 hilog）\n\n");
        return;
    }
    appf(out, cap, used, "  ✓ 定位 + 拷贝：base %p，size %zu\n", (void *)m.base, m.size);

    // 封代码区之前，先确认数据段读得到（w 段落在映像偏移 0x4000）
    const char *magic = (const char *)(m.base + 0x4000);
    int data_ok = (memcmp(magic, "DM_OHOS_IMAGE_OK", 16) == 0);
    appf(out, cap, used, "  %s 数据段：\"%.16s\"\n", data_ok ? "✓" : "✗", magic);

    if (dm_ohos_map_image_seal(&m)) {
        app(out, cap, used, "  ✗ seal 失败（见 hilog）\n\n");
        return;
    }
    app(out, cap, used, "  ✓ 代码区已封为只读+可执行\n");

    // 真执行映像偏移 0 处的代码，应返回 0x1234
    int r = ((int (*)(void))m.base)();
    int code_ok = (r == 0x1234);
    appf(out, cap, used, "  %s 执行映像代码：返回 0x%x（期望 0x1234）\n",
         code_ok ? "✓" : "✗", (unsigned)r);

    if (code_ok && data_ok)
        app(out, cap, used, "  结论：整条链路通了 —— 嵌入/定位/映射/执行全部成立。\n\n");
    else
        app(out, cap, used, "  结论：链路有问题，需细看（见 hilog）。\n\n");
}

// ───────── 实验 E：musl 的线程槽布局（决定改写规则怎么改）─────────
// 整套改写方案只依赖一条等式：线程槽地址 = tpidrro_el0 + 键号 × 8
// （苹果的 pthread 把 TSD 放在 tpidrro_el0 指向的结构里）。
// 这里把 musl 的真实情况测出来：两个线程指针寄存器各是什么，
// pthread_setspecific 写的值落在 tpidr_el0 的哪个偏移上，键号是不是线性 ×8。
static void exp_e(char *out, size_t cap, size_t *used) {
    app(out, cap, used, "【实验 E】musl 的线程槽布局\n");
    app(out, cap, used, "  要验的等式：槽地址 = tpidrro_el0 + 键号×8（苹果 pthread 的布局）\n");

    uint64_t tp_el0 = 0, tp_ro = 0;
    __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp_el0));
    __asm__ volatile("mrs %0, tpidrro_el0" : "=r"(tp_ro));
    appf(out, cap, used, "  tpidr_el0   = 0x%llx\n", (unsigned long long)tp_el0);
    appf(out, cap, used, "  tpidrro_el0 = 0x%llx%s\n", (unsigned long long)tp_ro,
         tp_ro ? "" : "   ← 为 0！苹果方案靠它取基址，这里取不到");

    #define NK 4
    pthread_key_t k[NK];
    const uint64_t marker[NK] = { 0x1111111111111111ull, 0x2222222222222222ull,
                                  0x3333333333333333ull, 0x4444444444444444ull };
    for (int i = 0; i < NK; i++) {
        if (pthread_key_create(&k[i], NULL)) {
            appf(out, cap, used, "  ✗ 建第 %d 个键失败\n\n", i);
            return;
        }
        pthread_setspecific(k[i], (void *)(uintptr_t)marker[i]);
    }
    appf(out, cap, used, "  建了 %d 个键，键号：%lu %lu %lu %lu\n", NK,
         (unsigned long)k[0], (unsigned long)k[1], (unsigned long)k[2], (unsigned long)k[3]);

    // 在 tpidr_el0 附近扫，看每个 marker 落在哪个偏移。
    // 整段扫描包在一个信号兜底里：万一越界，进程不至于挂。
    struct sigaction sa, old_segv, old_bus;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = probe_segv_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER;
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS, &sa, &old_bus);

    long off[NK] = { 0 };
    int faulted = 0;
    if (sigsetjmp(g_probe_jmp, 1) == 0) {
        g_probe_jmp_armed = 1;
        for (int i = 0; i < NK; i++) {
            off[i] = 0;
            for (long o = -0x2000; o < 0x8000; o += 8) {
                uint64_t v = *(const uint64_t *)(uintptr_t)(tp_el0 + (uint64_t)o);
                if (v == marker[i]) { off[i] = o; break; }
            }
        }
        g_probe_jmp_armed = 0;
    } else {
        faulted = 1;
    }
    sigaction(SIGSEGV, &old_segv, NULL);
    sigaction(SIGBUS, &old_bus, NULL);

    if (faulted) {
        app(out, cap, used, "  ⚠ 扫描时越界，结果不完整\n\n");
        return;
    }

    for (int i = 0; i < NK; i++) {
        if (off[i]) {
            appf(out, cap, used, "  键号 %-4lu → 值落在 tpidr_el0 %+ld（0x%lx）\n",
                 (unsigned long)k[i], off[i], (unsigned long)off[i]);
        } else {
            appf(out, cap, used, "  键号 %-4lu → 在 tpidr_el0-0x2000..+0x8000 内没找到\n",
                 (unsigned long)k[i]);
        }
    }

    // 键号相邻的两个槽，偏移差应该是 8（说明是数组）
    if (off[0] && off[1]) {
        long d = off[1] - off[0];
        long dk = (long)k[1] - (long)k[0];
        appf(out, cap, used, "  相邻键的偏移差 %ld，键号差 %ld → 每键 %s\n",
             d, dk, (dk && d == dk * 8) ? "8 字节（是数组）" : "不是 8 字节，要细看");
        if (dk && d == dk * 8) {
            appf(out, cap, used, "  结论：槽地址 = tpidr_el0 + %ld + 键号×8\n",
                 off[0] - (long)k[0] * 8);
            app(out, cap, used, "        → 改写器把 tpidrro_el0 换成 tpidr_el0，\n"
                                "          键页里写 (上面那个常数 + 键号×8) 即可。\n");
        }
    }
    app(out, cap, used, "\n");
}

// ───── 实验 E2：找 musl 的 tsd 指针在 pthread 结构里的偏移 ─────
// 实验 E 已说明：槽值不在 tpidr_el0 + 键号×8 处，也不是 tpidr_el0 附近的数组。
// musl 的做法是在 struct pthread 里放一个 void **tsd，指向单独分配的槽数组：
//     槽地址 = *(void**)(tpidr_el0 + tsd_off) + 键号 × 8
// 这里扫 pthread 结构，找那个 tsd_off。
static int safe_read64(uint64_t addr, uint64_t *out) {
    if (sigsetjmp(g_probe_jmp, 1) == 0) {
        g_probe_jmp_armed = 1;
        *out = *(const uint64_t *)(uintptr_t)addr;
        g_probe_jmp_armed = 0;
        return 1;
    }
    return 0;   // 越界被信号兜底接住了
}

static void exp_e2(char *out, size_t cap, size_t *used) {
    app(out, cap, used, "【实验 E2】找 musl 的 tsd 指针偏移\n");

    uint64_t tp = 0;
    __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp));
    appf(out, cap, used, "  tpidr_el0 = 0x%llx\n", (unsigned long long)tp);

    pthread_key_t k1, k2;
    if (pthread_key_create(&k1, NULL) || pthread_key_create(&k2, NULL)) {
        app(out, cap, used, "  ✗ 建键失败\n\n");
        return;
    }
    const uint64_t m1 = 0x5a5cafe123456789ull;
    const uint64_t m2 = 0x0f1e2d3c4b5a6978ull;
    pthread_setspecific(k1, (void *)(uintptr_t)m1);
    pthread_setspecific(k2, (void *)(uintptr_t)m2);
    appf(out, cap, used, "  键号 %lu → 0x%llx，%lu → 0x%llx\n",
         (unsigned long)k1, (unsigned long long)m1,
         (unsigned long)k2, (unsigned long long)m2);

    // 扫 pthread 结构：找偏移 o，使 *(pthread+o) 是个指针 p，且 p[k1]==m1、p[k2]==m2
    // 扫描会碰到未映射地址，必须先装信号兜底（safe_read64 靠 siglongjmp 回来）
    struct sigaction sa, old_segv, old_bus;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = probe_segv_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER;
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS, &sa, &old_bus);

    long found_off = 0;
    for (long o = -0x400; o < 0x2000; o += 8) {
        uint64_t p = 0;
        if (!safe_read64(tp + (uint64_t)o, &p)) continue;
        if (p < 0x10000 || p > 0x80000000ull) continue;    // 不像用户态指针
        uint64_t v1 = 0, v2 = 0;
        if (!safe_read64(p + (uint64_t)k1 * 8, &v1)) continue;
        if (v1 != m1) continue;
        if (!safe_read64(p + (uint64_t)k2 * 8, &v2)) continue;
        if (v2 != m2) continue;
        found_off = o;
        break;
    }
    sigaction(SIGSEGV, &old_segv, NULL);
    sigaction(SIGBUS, &old_bus, NULL);

    if (!found_off) {
        app(out, cap, used, "  ✗ 没找到（可能是两级以上间接，或 tsd 数组另存）\n\n");
        return;
    }
    appf(out, cap, used, "  ✓ 找到了：pthread + %ld 是指针 p，且 p[%lu]==marker1、p[%lu]==marker2\n",
         found_off, (unsigned long)k1, (unsigned long)k2);
    appf(out, cap, used, "  → 完整公式：槽地址 = *(void**)(tpidr_el0 + %ld) + 键号 × 8\n", found_off);
    app(out, cap, used, "  → 改写器改用 tpidr_el0 作基址，插入一次「取 tsd 指针」的 ldr；\n"
                        "    键页仍写 键号×8，不动。\n\n");
}

// ───── 实验 F：改写器生成的指令序列，在 musl 上真的能读到槽吗 ─────
// 把 patch_nso.py 的 emit_slot_addr 在 OHOS 下生成的那串指令原样执行一遍：
//     mrs T, tpidr_el0 → sub T,T,#272 → ldr T,[T] → add T,T,key*8 → ldr S,[T]
// 值由 pthread_setspecific 写进去（改写后的游戏代码就是这么读的），
// 看它能不能读回来、和 pthread_getspecific 是否一致。
#if defined(__aarch64__)
#define DM_SLOT_READ(dst, off) __asm__ volatile ( \
    "mov x9, %1 \n"                  /* x9  = 键页里存的值（键号×8） */ \
    "mrs x10, tpidr_el0 \n"          /* x10 = pthread 指针 */ \
    "sub x10, x10, #272 \n"          /* x10 = &pthread->tsd */ \
    "ldr x10, [x10] \n"              /* x10 = musl 的 tsd 数组 */ \
    "add x9, x10, x9 \n"             /* x9  = 槽地址 */ \
    "ldr %0, [x9] \n"                /* dst = 槽值 */ \
    : "=r"(dst) : "r"((uint64_t)(off)) : "x9", "x10", "memory")
#else
#define DM_SLOT_READ(dst, off) do { (dst) = (uint64_t)(uintptr_t)pthread_getspecific(0); } while (0)
#endif

static void exp_f(char *out, size_t cap, size_t *used) {
    app(out, cap, used, "【实验 F】改写器生成的指令序列能否读到宿主写的槽\n");
    app(out, cap, used, "  原样执行 emit_slot_addr 在 OHOS 下生成的五条指令\n");

    pthread_key_t k1, k2;
    if (pthread_key_create(&k1, NULL) || pthread_key_create(&k2, NULL)) {
        app(out, cap, used, "  ✗ 建键失败\n\n");
        return;
    }
    const uint64_t v1 = 0xA5A5DEAD00000001ull;
    const uint64_t v2 = 0xA5A5DEAD00000002ull;
    pthread_setspecific(k1, (void *)(uintptr_t)v1);
    pthread_setspecific(k2, (void *)(uintptr_t)v2);

    uint64_t r1 = 0, r2 = 0;
    DM_SLOT_READ(r1, (uint64_t)k1 * 8);
    DM_SLOT_READ(r2, (uint64_t)k2 * 8);
    uint64_t g1 = (uint64_t)(uintptr_t)pthread_getspecific(k1);
    uint64_t g2 = (uint64_t)(uintptr_t)pthread_getspecific(k2);

    appf(out, cap, used, "  键号 %-4lu：原生序列读回 0x%llx，宿主读回 0x%llx，写入 0x%llx  %s\n",
         (unsigned long)k1, (unsigned long long)r1, (unsigned long long)g1,
         (unsigned long long)v1, (r1 == g1 && r1 == v1) ? "✓" : "✗");
    appf(out, cap, used, "  键号 %-4lu：原生序列读回 0x%llx，宿主读回 0x%llx，写入 0x%llx  %s\n",
         (unsigned long)k2, (unsigned long long)r2, (unsigned long long)g2,
         (unsigned long long)v2, (r2 == g2 && r2 == v2) ? "✓" : "✗");

    if (r1 == g1 && r2 == g2 && r1 == v1 && r2 == v2)
        app(out, cap, used, "  结论：改写器生成的代码序列在 musl 上读得到宿主写的槽，"
                            "且与 pthread_getspecific 一致。\n\n");
    else
        app(out, cap, used, "  结论：读不到或不一致，需要细看。\n\n");
}

// ─────────────────────── 入口 ───────────────────────

// ───────── 探针桩：门回路只走 svc 38 的占位路径，下列服务不会被触及 ─────────
// hle.c / hle_svc.c 还引用导入解析、线程与同步、私有宿主调用（都是"宿主服务按游戏提供"的边界，
// 正式装载由 loader.c/hle_thread.c/hle_tact.c 提供）。这里给惰性实现：链接闭合、行为可见。
module_t g_mod[MAX_MOD]; int g_nmod = 0;                 // 空模块表：导入解析路径不会走到
struct dm_flags g_flag;                                   // 调试开关：全 0
module_t *module_of(const void *p) { (void)p; return NULL; }
uint64_t guest_ns(void) {                                 // 客体时钟：单调钟直通（探针不走 DM_TICK_HOLD）
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
void dm_backtrace(uint64_t fp) { (void)fp; }
uint64_t dm_entropy64(void) { static uint64_t s; return s += 0x9E3779B97F4A7C15ull; }
void hlecount_svc(int num) { (void)num; }
void *hlecount_wrap(const char *a, const char *b, void *fn) { (void)a; (void)b; return fn; }
uint64_t svc_host_call(int num, uint64_t *regs) { (void)num; (void)regs; return 0; }
int svc_thread(int num, uint64_t *regs) { (void)num; (void)regs; return 0; }   // 0 = 未处理 → 落到占位返回
void gate_register(uint64_t ret_abs, const char *kind, int arg, uint64_t orig);   // hle.c
static void gate_probe_entry(void) { logline("✓ 到达 KGATE 入口（门块执行 + 槽读取路径成立）"); }
extern void dm_gate(void);                                // entry.S：门公共入口
void dm_thread_returned(void);                            // entry.S：客人线程要求退出时落到这里（noreturn）
void dm_thread_returned(void) {                           // 探针桩：门回路不触发线程退出；真触发就走 _exit 让它可见
    fprintf(stderr, "[dm] dm_thread_returned 被触发（不应发生在门回路验证里）\n"); _exit(0);
}
// 强制接管表的空段：hle.c 会遍历 dmswhle（正式装载由 HLE_FORCE 产出），探针没有条目
__attribute__((used, section("dmswhle"))) static const char dm_swhle_probe[4] = {0};

// 实验 H/K 的错误兜底：SIGSEGV/SIGBUS 时记下**完整现场**，跳回探针（进程活着，继续报）。
// 为什么连寄存器一起记：追崩链时"我们以为的调用路径"会和日志对不上（实测：分配器明明给了非 0
// 指针，被调方却看到 x0=0）——**崩点现场是唯一不会骗人的证人**。
// 异步信号安全：处理器里只做搬运，打印与帧链回溯留到 siglongjmp 之后（见 gate_dump_fault）。
#define GATE_REG_N 31
// 2026-10-22：改**线程局部**。看门狗把每个构造函数放在 worker 线程里跑，崩溃也在那边发生；
// 全局 jmp_buf 会被"上一个卡住的线程"和"当前线程"互相踩（longjmp 到别人的栈 = 灾难）。
// 信号处理器在**出错线程**上执行，所以 TLS 正好对上。
static __thread sigjmp_buf g_gate_jmp;
static __thread volatile sig_atomic_t g_gate_armed;
static volatile uint64_t g_gate_fault_pc;
static volatile int g_gate_fault_sig;
static uint64_t g_gate_regs[GATE_REG_N];
static uint64_t g_gate_sp;
static volatile uint64_t g_gate_fault_addr;   // ★ si_addr：真正访问失败的地址（缺了它，"崩在哪条指令"会骗人）
static volatile int g_gate_have_regs;
// ── Task #5 越障（np60）：崩点 +0xA05620 的定点修复 ──
// 崩点本质：worker 0xa051c4 用"解析结果"（我们给的全零替身）拼出 0 尺寸的命令编码，
// 最后 `str x8,[x0]` 时 x0=0。正解是让解析拿到真数据（§37.5 方案 B），这里只是**诊断越障**：
// 把写目标换成一块暂存页再放行，看下一层需求是什么。
static uint64_t g_self_base;            // np82：本 .so 的装载基址（dladdr 取，用于识别"外"调用者）
static int g_nnmain_entered;      // np86：nnMain 段是否已进入（工厂分支的迟到防护）
static volatile sig_atomic_t g_fixup_n;
static volatile uint64_t g_fixup_last[8];     // 最近 8 次拦截的 PC 偏移（信号里不打印，崩后统一报）
static uint8_t *g_fixup_page;                 // 安装期 calloc（不在信号上下文里分配——np58 教训）
static uintptr_t g_fixup_base;                // 模块基址（处理器里不能用 g_k_m，它在文件后面才声明）
static pthread_t g_probe_thr;                 // np115：探针线程的 id（setjmp 上下文只属于它）
static int g_probe_thr_set;
static void death_note(const char *why, uint64_t pc, uint64_t addr, int sig);   // np163 定义在下方

static void gate_fault_handler(int sig, siginfo_t *si, void *uc_) {
    // 字段写法与 dm/host/crash.c 的鸿蒙分支一致：mcontext_t = { regs[31], sp, pc, pstate }
    ucontext_t *uc = (ucontext_t *)uc_;
    if (g_fixup_page && g_fixup_base && g_fixup_n < 400) {
        uint64_t off = (uint64_t)uc->uc_mcontext.pc - (uint64_t)g_fixup_base;
        // worker 0xa051c4..0xa05fa0 是"命令编码"大函数：多处 `stp/str ...,[x0]` 用同一个
        // 写游标，游标来源（解析结果链条）目前恒 0。这里在整个函数范围内兜底：
        // 每次给一块新的暂存区（按 0x100 步进、0x4000 回绕），模拟"缓冲区在推进"。
        // np81：竞态崩点 +0xA011F4（`ldrb w10,[x9,#0x14]`，x9=0 ⇒ si_addr=0x2c）——
        //   "任务节点没注册 ⇒ 查找落空 x9=0 ⇒ 从 0x18 起遍历"。给 x9 一块**全零**暂存区：
        //   读到的字节是 0 ⇒ 该节点被跳过，循环继续（计数正常时很快结束）。
        if (off >= 0xA011E4 && off < 0xA01220 && (uint64_t)(uintptr_t)si->si_addr < 0x1000) {
            g_fixup_n++;
            uc->uc_mcontext.regs[9] = (uint64_t)(uintptr_t)g_fixup_page;   // x9 = 零页
            if (g_fixup_n <= 8) g_fixup_last[g_fixup_n - 1] = off;
            return;
        }
        if (off >= 0xA051C4 && off < 0xA05FA0 && (uint64_t)(uintptr_t)si->si_addr < 0x1000) {
            g_fixup_n++;
            uc->uc_mcontext.regs[0] = (uint64_t)(uintptr_t)(g_fixup_page + ((size_t)g_fixup_n % 64) * 0x100);
            if (g_fixup_n <= 8) g_fixup_last[g_fixup_n - 1] = off;   // 信号里只记账，崩后统一打印
            return;    // ★ 同步信号的处理函数返回 ⇒ 内核**重试**那条指令（这次写暂存页）
        }
    }
    g_gate_fault_sig = sig;
    g_gate_fault_pc = (uint64_t)uc->uc_mcontext.pc;   // aarch64 mcontext 的 pc
    g_gate_fault_addr = (uint64_t)(uintptr_t)si->si_addr;   // 出错的内存地址
    for (int i = 0; i < GATE_REG_N; i++) g_gate_regs[i] = (uint64_t)uc->uc_mcontext.regs[i];
    g_gate_sp = (uint64_t)uc->uc_mcontext.sp;
    g_gate_have_regs = 1;
    // ★ np115：**故障处理必须线程感知**。setjmp 的上下文只属于探针线程；
    //   客人自己的线程（Prepare / Vibration 等）出错时若也 siglongjmp，就是**跨线程跳栈**，
    //   会把探针线程的栈恢复出来、现场全乱（本轮真机：日志停在"止步"、nnMain 一行都没打出来）。
    //   对非探针线程只记现场然后退出（退出码 140），保证日志里留下真正的崩点。
    if (g_probe_thr_set && !pthread_equal(pthread_self(), g_probe_thr)) {
        LOG("【红】客人线程出错：pc=%#llx sig=%d si_addr=%#llx —— 不跨线程 longjmp，退出判读",
            (unsigned long long)g_gate_fault_pc, sig, (unsigned long long)g_gate_fault_addr);
        death_note("客人线程出错", g_gate_fault_pc, g_gate_fault_addr, sig);
        // ⭐ np164 实验：**只让这个工人线程停住，不杀进程**。
        //   理由：np163 实证死因是"工人线程跑任务时源指针空"（对象 +0x180 = 0）。
        //   若主线不依赖这个工人，进程就能继续走 ⇒ 用"worker 静默死亡"换主线存活，
        //   把"一个工人崩 = 整个游戏死"降级成"一个工人崩 = 一个任务丢"。
        //   风险：该线程若持着客体锁，主线程会卡在那把锁上（届时死亡笔记+trace 仍在）。
        for (;;) pause();
    }
    if (!g_gate_armed) {
        // 未武装段（尾跳补丁循环等）出段错误：先打日志再退出——PC 落在 hilog 里供定位
        LOG("【红】门回路：未武装段 SIGSEGV/SIGBUS pc=%#llx sig=%llx（退出码 139）", (unsigned long long)g_gate_fault_pc, (unsigned long long)sig);
        death_note("未武装段(_exit 139)", g_gate_fault_pc, g_gate_fault_addr, sig);
        _exit(139);
    }
    g_gate_armed = 0;
    siglongjmp(g_gate_jmp, 1);
}

// ───────── 实验 G：真实模块装载（旷野之息 main，改写器真实产物）─────────
// 与实验 D 的差别：D 用的是自造测试映像（49KB），G 用的是 **真实游戏模块**——
// patch_nso.py（DM_PATCH_TARGET=ohos）改写旷野之息 main 之后的 main.patched.bin（37MB），
// 经 gen_embed_ohos.py 内嵌成 dmmainx0/w/x1 三段。判据四条，**不做执行**
//（执行需要 93 个宿主门的接线，另立一步）：
//   ① alloc 成功（内含布局自检：x0+w == PATCH_OFF）
//   ② 三段**逐字节全等**：映射区 vs 嵌入段（37MB 全比，不止抽查）
//   ③ seal 后代码区 R+X 生效（/proc/self/maps 权限位）
//   ④ 布局常数与 fixups 一致（module_t 的常数来自 dm_main_layout.h，同源）
static void exp_g(char *out, size_t cap, size_t *used) {
    app(out, cap, used, "【实验 G】真实模块装载（旷野之息 main，37MB 改写产物）\n");
    app(out, cap, used, "  素材：patch_nso.py（DM_PATCH_TARGET=ohos）的 main.patched.bin\n");

    module_t m;
    memset(&m, 0, sizeof m);
    strncpy(m.name, "main", sizeof m.name - 1);
    m.text_end      = DM_IMG_MAIN_TEXT_END;
    m.patch_off     = DM_IMG_MAIN_PATCH_OFF;
    m.ro_off        = DM_IMG_MAIN_RO_OFF;
    m.data_off      = DM_IMG_MAIN_DATA_OFF;
    m.key_page_off  = DM_IMG_MAIN_KEY_PAGE_OFF;
    m.key_page_size = DM_IMG_MAIN_KEY_PAGE_SIZE;
    m.x0_end        = DM_IMG_MAIN_X0_END;      // 装载拷贝的精确范围（fixups），防链接器尾垫错位
    m.img_end       = DM_IMG_MAIN_END;
    appf(out, cap, used, "  布局：TEXT_END %#llx / PATCH_OFF %#llx / KEY_PAGE %#llx\n",
         (unsigned long long)m.text_end, (unsigned long long)m.patch_off,
         (unsigned long long)m.key_page_off);

    if (dm_ohos_map_image_alloc(&m)) {
        app(out, cap, used, "  ✗ alloc 失败（布局自检或 mmap，见 hilog）\n\n");
        return;
    }
    appf(out, cap, used, "  ✓ 定位 + 拷贝：base %p，size %zu\n", (void *)m.base, m.size);

    // 判据②：三段**全量** memcmp —— 映射区必须与嵌入段逐字节一致
    // 对拍长度按 **fixups 范围**（链接器会给 ELF 段尾垫对齐页，段大小比内容大 16K）
    const struct { const unsigned char *s; uint8_t *dst; size_t len; const char *nm; } regions[] = {
        { __start_dmmainx0, m.base,                            DM_IMG_MAIN_X0_END,             "x0(代码)" },
        { __start_dmmainw,  m.base + DM_IMG_MAIN_X0_END,       DM_IMG_MAIN_PATCH_OFF - DM_IMG_MAIN_X0_END, "w(数据)" },
        { __start_dmmainx1, m.base + DM_IMG_MAIN_PATCH_OFF,    DM_IMG_MAIN_END - DM_IMG_MAIN_PATCH_OFF,    "x1(补丁区)" },
    };
    int cmp_ok = 1;
    for (int k = 0; k < 3; k++) {
        size_t len = regions[k].len;
        int same = (memcmp(regions[k].dst, regions[k].s, len) == 0);
        appf(out, cap, used, "  %s %s 段逐字节一致（%zu 字节）\n", same ? "✓" : "✗", regions[k].nm, len);
        if (!same) cmp_ok = 0;
    }
    if (cmp_ok)
        app(out, cap, used, "  ✅ 判据②：映射区与嵌入段**全量一致**（37MB 全比）\n");
    else
        app(out, cap, used, "  ❌ 判据②：映射内容与嵌入段不一致\n");

    if (dm_ohos_map_image_seal(&m)) {
        app(out, cap, used, "  ✗ seal 失败（见 hilog）\n\n");
        return;
    }
    app(out, cap, used, "  ✓ 代码区与补丁区已封 R+X\n");

    // 判据③：/proc/self/maps 里找 base 所在行，验权限位
    int map_ok = 0; char perms[8] = "?";
    {
        FILE *f = fopen("/proc/self/maps", "r");
        char line[512];
        while (f && fgets(line, sizeof line, f)) {
            unsigned long lo, hi;
            if (sscanf(line, "%lx-%lx %7s", &lo, &hi, perms) == 3) {
                if ((uintptr_t)m.base + 0x1000 >= lo && (uintptr_t)m.base + 0x1000 < hi) { map_ok = 1; break; }
            }
        }
        if (f) fclose(f);
    }
    if (map_ok) {
        int rx = (perms[0] == 'r' && perms[1] == '-' && perms[2] == 'x');
        appf(out, cap, used, "  %s 判据③：代码区权限 = %s（期望 r-x）\n", rx ? "✅" : "❌", perms);
        if (!rx) cmp_ok = 0;
    } else {
        app(out, cap, used, "  ❌ 判据③：/proc/self/maps 里找不到映射区\n");
        cmp_ok = 0;
    }

    appf(out, cap, used, "  结论：%s\n\n",
         (cmp_ok && map_ok) ? "**真实模块装载成立** —— 装载链吃下了第一个真实游戏模块" : "链路有问题，见上");
}

// ───────── 实验 H：门回路（真实模块的真实门块 → dm_gate → gate_dispatch → svc 38 → 返回）─────────
// 装载（实验 G）之后的一步：旷野之息 main 的 93 个门**全部是 svc 38**（WaitSynchronization，占位返回 0）。
// 做法：装载 → 建 8 个线程槽（KGATE = dm_gate）→ 装键页（实际键号×8）→ 登记 93 门 →
//       每个门块的尾跳（B 原指令+4）暂改为 RET（隔离：不执行门后的游戏代码）→ 逐块调用。
// 判据：93 块全部走完整回路且返回 0（dm_gate 回写 x0 = svc 占位返回值）；未识别的门会 die（可见）。
// 门回路专用的增量文件日志：逐门结果边跑边落盘，崩溃也保得住（filesDir/dm_gate_test.log）
static FILE *g_gate_log;
static void glog(const char *fmt, ...) {
    if (!g_gate_log) return;
    va_list ap; va_start(ap, fmt);
    vfprintf(g_gate_log, fmt, ap);
    va_end(ap);
    fflush(g_gate_log);
}
// 看门狗假设验证：二分 B 的门调度搬到专用线程上跑（槽在该线程重设）——与上游设计一致
// （上游门调用发生在客体线程，不在主线程）。主线程卡死保护可能是此前"无痕死亡"的真因。
typedef struct { uint64_t base; pthread_key_t key[8]; uint64_t rv; int faulted; } gb_ctx_t;
static gb_ctx_t g_gb;
static void *gb_thread(void *arg) {
    gb_ctx_t *c = arg;
    uint64_t tp = 0; __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp));      // 本线程的 tsd
    uint64_t tsd = *(uint64_t *)(tp - 272);
    const void *sv[8] = {
        (void *)(uintptr_t)0x1111111111111111ull, (void *)(uintptr_t)0x2222222222222222ull,
        (void *)(uintptr_t)0x3333333333333333ull, (void *)dm_gate,
        (void *)(uintptr_t)0x4400000000000004ull, (void *)(uintptr_t)0x4400000000000005ull,
        (void *)(uintptr_t)0x4400000000000006ull, (void *)(uintptr_t)0x4400000000000007ull,
    };
    for (int i = 0; i < 8; i++) *(uint64_t *)(tsd + (uint64_t)c->key[i] * 8) = (uint64_t)sv[i];
    uint64_t regs2[64]; memset(regs2, 0, sizeof regs2);
    regs2[0] = 0x80000000ull;
    g_gate_armed = 1; g_gate_fault_pc = 0;
    if (sigsetjmp(g_gate_jmp, 1) == 0) {
        g_gb.rv = gate_dispatch(c->base, regs2);
        g_gate_armed = 0;
    } else { g_gate_armed = 0; c->faulted = 1; }
    return NULL;
}

// 实验 I：真实执行——调用模块里的 3 个真实小函数（离线反汇编选出的纯叶子函数，
// 行为完全可预测），在专用线程上调用并核对返回值。这是执行链从"门回路验证"走向
// "客体代码真正跑起来"的第一站。
static uint64_t g_i_rv[3];
static module_t g_i_m;
static volatile int g_i_faulted;
static void *i_thread(void *arg) {
    (void)arg;
    uint64_t b = (uint64_t)g_i_m.base;
    g_gate_armed = 1; g_gate_fault_pc = 0;
    if (sigsetjmp(g_gate_jmp, 1) == 0) {
        g_i_rv[0] = ((uint64_t (*)(uint64_t))(b + 0x2eb80))(0x1000);   // add x0,x0,#0x14; ret
        g_i_rv[1] = ((uint64_t (*)(uint64_t))(b + 0x2f3ad4))(0x5000);  // add x0,x0,#0x80; ret
        g_i_rv[2] = ((uint64_t (*)(void))(b + 0x157a10))();            // mov w0,#-1; ret
        g_gate_armed = 0;
    } else {
        g_gate_armed = 0; g_i_faulted = 1;
        g_i_rv[0] = g_i_rv[1] = g_i_rv[2] = ~0ull;
    }
    return NULL;
}
static void exp_i(char *out, size_t cap, size_t *used) {
    app(out, cap, used, "【实验 I】真实执行：调用模块里 3 个真实函数并核对返回值\n");
    memset(&g_i_m, 0, sizeof g_i_m);
    strncpy(g_i_m.name, "main", sizeof g_i_m.name - 1);
    g_i_m.text_end      = DM_IMG_MAIN_TEXT_END;
    g_i_m.patch_off     = DM_IMG_MAIN_PATCH_OFF;
    g_i_m.ro_off        = DM_IMG_MAIN_RO_OFF;
    g_i_m.data_off      = DM_IMG_MAIN_DATA_OFF;
    g_i_m.key_page_off  = DM_IMG_MAIN_KEY_PAGE_OFF;
    g_i_m.key_page_size = DM_IMG_MAIN_KEY_PAGE_SIZE;
    g_i_m.x0_end        = DM_IMG_MAIN_X0_END;
    g_i_m.img_end       = DM_IMG_MAIN_END;
    if (dm_ohos_map_image_alloc(&g_i_m)) { app(out, cap, used, "  ✗ alloc 失败\n\n"); return; }
    if (dm_ohos_map_image_seal(&g_i_m)) { app(out, cap, used, "  ✗ seal 失败\n\n"); return; }
    appf(out, cap, used, "  ✓ 装载并封 R+X：base %p\n", (void *)g_i_m.base);

    struct sigaction sa, o1, o2, o3, o4;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = gate_fault_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &o1); sigaction(SIGBUS, &sa, &o2);
    sigaction(SIGILL, &sa, &o3);  sigaction(SIGTRAP, &sa, &o4);

    g_i_faulted = 0;
    pthread_t th;
    if (pthread_create(&th, NULL, i_thread, NULL) == 0) pthread_join(th, NULL);
    else app(out, cap, used, "  ✗ pthread_create 失败\n");

    sigaction(SIGSEGV, &o1, NULL); sigaction(SIGBUS, &o2, NULL);
    sigaction(SIGILL, &o3, NULL);  sigaction(SIGTRAP, &o4, NULL);

    if (g_i_faulted) {
        appf(out, cap, used, "  ✗ 执行崩（pc %#llx sig %d）\n", (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
        app(out, cap, used, "\n");
        return;
    }
    int ok = 0;
    appf(out, cap, used, "  %s f₁(0x1000)=%#llx（期望 0x1014）  f₁: add x0,x0,#0x14; ret\n",
         g_i_rv[0] == 0x1014 ? "✓" : "✗", (unsigned long long)g_i_rv[0]);
    if (g_i_rv[0] == 0x1014) ok++;
    appf(out, cap, used, "  %s f₂(0x5000)=%#llx（期望 0x5080）  f₂: add x0,x0,#0x80; ret\n",
         g_i_rv[1] == 0x5080 ? "✓" : "✗", (unsigned long long)g_i_rv[1]);
    if (g_i_rv[1] == 0x5080) ok++;
    uint64_t r2 = g_i_rv[2] & 0xffffffffull;
    appf(out, cap, used, "  %s f₃()=%#llx（期望 w0=0xffffffff）  f₃: mov w0,#-1; ret\n",
         r2 == 0xffffffffull ? "✓" : "✗", (unsigned long long)g_i_rv[2]);
    if (r2 == 0xffffffffull) ok++;
    appf(out, cap, used, "  %s 判据：%d / 3 个真实函数在宿主执行且返回值精确正确\n",
         ok == 3 ? "✅" : "❌", ok);
    app(out, cap, used, "  意义：客体代码可以在鸿蒙宿主上真正执行并正确返回——执行链地基成立。\n");
    app(out, cap, used, "\n");
}

// 实验 J：真实函数 + 真实门 串成一条链——调用含门的真实函数，命中门后由**我们的服务**应答，
// 并验证服务的返回值从真实函数的出口流回来。这是执行链从"能跑"走向"跑真实流程"的一站。
// 目标函数 0x140c9d8（28B，零调用零全局寻址，含门@0x140c9ec）：
//   cmp w2,#1 ; b.lt → 门 ; cmp w1,w2 ; b.gt → 门 ; b 0x1632d40（正常尾调）
//   门分支：命中门 → 返回值落在 x0 → 0x140c9f0: ret  ⇒ 函数返回门的返回值
static volatile int g_j_hits;
static uint64_t g_j_rv;
static volatile int g_j_faulted;
static module_t g_j_m;
static pthread_key_t g_j_key[8];
static uint64_t j_gate_stub(void) { g_j_hits++; return 0xABCD; }   // 我们的"服务"：记命中、返回已知值
static void *j_thread(void *arg) {
    (void)arg;
    uint64_t b = (uint64_t)g_j_m.base;
    uint64_t tp = 0; __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp));
    uint64_t tsd = *(uint64_t *)(tp - 272);
    const void *sv[8] = {
        (void *)(uintptr_t)0x1111111111111111ull, (void *)(uintptr_t)0x2222222222222222ull,
        (void *)(uintptr_t)0x3333333333333333ull, (void *)j_gate_stub,   // KGATE = 我们的服务
        (void *)(uintptr_t)0x4400000000000004ull, (void *)(uintptr_t)0x4400000000000005ull,
        (void *)(uintptr_t)0x4400000000000006ull, (void *)(uintptr_t)0x4400000000000007ull,
    };
    for (int i = 0; i < 8; i++) *(uint64_t *)(tsd + (uint64_t)g_j_key[i] * 8) = (uint64_t)sv[i];
    g_gate_armed = 1; g_gate_fault_pc = 0;
    if (sigsetjmp(g_gate_jmp, 1) == 0) {
        // w2=0 ⇒ 走门分支（cmp w2,#1 → b.lt）
        g_j_rv = ((uint64_t (*)(uint64_t, uint64_t, uint64_t))(b + 0x140c9d8))(0, 0, 0);
        g_gate_armed = 0;
    } else { g_gate_armed = 0; g_j_faulted = 1; }
    return NULL;
}
static void exp_j(char *out, size_t cap, size_t *used) {
    app(out, cap, used, "【实验 J】真实函数 + 真实门 串链：服务返回值从函数出口流出\n");
    memset(&g_j_m, 0, sizeof g_j_m);
    strncpy(g_j_m.name, "main", sizeof g_j_m.name - 1);
    g_j_m.text_end      = DM_IMG_MAIN_TEXT_END;
    g_j_m.patch_off     = DM_IMG_MAIN_PATCH_OFF;
    g_j_m.ro_off        = DM_IMG_MAIN_RO_OFF;
    g_j_m.data_off      = DM_IMG_MAIN_DATA_OFF;
    g_j_m.key_page_off  = DM_IMG_MAIN_KEY_PAGE_OFF;
    g_j_m.key_page_size = DM_IMG_MAIN_KEY_PAGE_SIZE;
    g_j_m.x0_end        = DM_IMG_MAIN_X0_END;
    g_j_m.img_end       = DM_IMG_MAIN_END;
    if (dm_ohos_map_image_alloc(&g_j_m)) { app(out, cap, used, "  ✗ alloc 失败\n\n"); return; }
    if (dm_ohos_map_image_seal(&g_j_m)) { app(out, cap, used, "  ✗ seal 失败\n\n"); return; }
    appf(out, cap, used, "  ✓ 装载并封 R+X：base %p\n", (void *)g_j_m.base);

    for (int i = 0; i < 8; i++)
        if (pthread_key_create(&g_j_key[i], NULL)) { app(out, cap, used, "  ✗ 建键失败\n\n"); return; }
    // 键页按实际键号写入（门块照此寻址）
    uint64_t *page = (uint64_t *)(g_j_m.base + DM_IMG_MAIN_KEY_PAGE_OFF);
    memset(page, 0, DM_IMG_MAIN_KEY_PAGE_SIZE);
    page[DM_IMG_MAIN_KEY_K18_IDX]       = (uint64_t)g_j_key[0] * 8;
    page[DM_IMG_MAIN_KEY_KTLS_IDX]      = (uint64_t)g_j_key[1] * 8;
    page[DM_IMG_MAIN_KEY_KTLR_IDX]      = (uint64_t)g_j_key[2] * 8;
    page[DM_IMG_MAIN_KEY_KGATE_IDX]     = (uint64_t)g_j_key[3] * 8;
    page[DM_IMG_MAIN_KEY_KRES0_IDX]     = (uint64_t)g_j_key[4] * 8;
    page[DM_IMG_MAIN_KEY_KRES1_IDX]     = (uint64_t)g_j_key[5] * 8;
    page[DM_IMG_MAIN_KEY_KASPECT_IDX]   = (uint64_t)g_j_key[6] * 8;
    page[DM_IMG_MAIN_KEY_KASPECTUI_IDX] = (uint64_t)g_j_key[7] * 8;

    struct sigaction sa, o1, o2, o3, o4;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = gate_fault_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &o1); sigaction(SIGBUS, &sa, &o2);
    sigaction(SIGILL, &sa, &o3);  sigaction(SIGTRAP, &sa, &o4);

    g_j_hits = 0; g_j_rv = 0; g_j_faulted = 0;
    pthread_t th;
    if (pthread_create(&th, NULL, j_thread, NULL) == 0) pthread_join(th, NULL);
    else app(out, cap, used, "  ✗ pthread_create 失败\n");

    sigaction(SIGSEGV, &o1, NULL); sigaction(SIGBUS, &o2, NULL);
    sigaction(SIGILL, &o3, NULL);  sigaction(SIGTRAP, &o4, NULL);

    if (g_j_faulted) {
        appf(out, cap, used, "  ✗ 执行崩（pc %#llx sig %d）\n\n", (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
        return;
    }
    appf(out, cap, used, "  %s 门被真实函数命中：%d 次（期望 1）\n", g_j_hits == 1 ? "✓" : "✗", g_j_hits);
    appf(out, cap, used, "  %s 服务返回值从函数出口流出：%#llx（期望 0xabcd）\n",
         g_j_rv == 0xABCD ? "✓" : "✗", (unsigned long long)g_j_rv);
    appf(out, cap, used, "  %s 判据：真实函数 → 命中门 → 我们的服务应答 → 值流回函数出口\n",
         (g_j_hits == 1 && g_j_rv == 0xABCD) ? "✅" : "❌");
    app(out, cap, used, "  意义：门与执行串成一条链——真实代码可以借我们的服务跑通一条完整路径。\n\n");
}

// 实验 K：真实流程首跑——从模块的启动链（映像里第一个函数 0x1c8）起跑，跑到哪里出事就记哪里。
// KGATE 槽指向记录型服务（记命中与返回点），所有日志落盘（崩溃也保得住）。
static FILE *g_k_log;
static char g_files_dir[512];   // np125：探针线程里也要落盘（trace），存一份目录

static module_t g_k_m;   // 前向（正式定义在实验 K 段）——death_note 要用 base
static const char *g_k_stage;   // 前向（正式定义紧随 g_files_dir）——death_note 要带阶段

// ═══ np163：死亡笔记（进程即将退出时的最后一份证据）═══
// 为什么需要：np161/np162 预算制下游戏越过旧停车点后**进程消失**（无 faultlog、无日志尾巴），
//   连"是不是我们 _exit 的"都判不了。故障处理器退出前必须留下 PC。
// 为什么独立实现（不用 klog）：退出时共享锁/stdio 缓冲都不可信（可能正被楔住的线程持有），
//   唯一可靠 = 开独立 fd + **单次 write** + 立即 close（几十字节，实测这类小写从未楔过）。
// 备份：同一行也打 hilog（logline 直调，不经 klog 的预算与锁）。
static void death_note(const char *why, uint64_t pc, uint64_t addr, int sig) {
    char line[4096];
    uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
    int n = snprintf(line, sizeof line,
        "why=%s pc=%#llx img+%#llx si_addr=%#llx sig=%d tid=%ld stage=%s sp=%#llx\n",
        why, (unsigned long long)pc,
        (unsigned long long)((pc >= base && pc < base + 0x2600000) ? pc - base : 0),
        (unsigned long long)addr, sig, (long)gettid(), g_k_stage ? g_k_stage : "?",
        (unsigned long long)g_gate_sp);
    // np163：把故障现场的全寄存器 + 栈一起写进笔记（单次 write，1–2KB 这类小写从未楔过）。
    // 判读：x30 = 调用者返回地址；栈上连续值对回 main 映像偏移即还原调用链。
    if (g_gate_have_regs && n > 0) {
        n += snprintf(line + n, sizeof line - (size_t)n,
            "regs: x0=%#llx x1=%#llx x2=%#llx x3=%#llx x8=%#llx x9=%#llx x19=%#llx x20=%#llx x21=%#llx x29=%#llx x30=%#llx\n",
            (unsigned long long)g_gate_regs[0], (unsigned long long)g_gate_regs[1],
            (unsigned long long)g_gate_regs[2], (unsigned long long)g_gate_regs[3],
            (unsigned long long)g_gate_regs[8], (unsigned long long)g_gate_regs[9],
            (unsigned long long)g_gate_regs[19], (unsigned long long)g_gate_regs[20],
            (unsigned long long)g_gate_regs[21], (unsigned long long)g_gate_regs[29],
            (unsigned long long)g_gate_regs[30]);
        // np657：pc 处指令码 16 字节 + sp 前 8 字节。np657 的 SEGV 现场里 pc（img+0xa05324）
        // 反汇编是条无害 strb [sp,#0x92c]、与 si_addr=0x8 矛盾 ⇒ 需要直接看故障指令字节来分辨
        // "pc 不可靠 / 基址错位 / 跳进数据段"三种假设。pc 在客体 main 内才转储（宿主地址不读，
        // 避免 np530 的 ptrprobe 读崩重演）。
        // np657: dump 16 bytes at pc + 8 stack words. The np657 SEGV note had pc=img+0xa05324
        //   disassembling to a harmless strb, contradicting si_addr=0x8 -- the instruction bytes tell
        //   "unreliable pc / base off / jumped into data" apart. Guest-main pc only (host addresses are
        //   not dereferenced, see the np530 ptrprobe lesson).
        if (pc >= base && pc < base + 0x2420000) {
            n += snprintf(line + n, sizeof line - (size_t)n, "code@pc=");
            for (int i = 0; i < 16; i++)
                n += snprintf(line + n, sizeof line - (size_t)n, "%02x",
                    *(const unsigned char *)(uintptr_t)(pc + i));
            n += snprintf(line + n, sizeof line - (size_t)n, "\n");
        }
        for (int i = 0; i < 64 && n > 0 && (size_t)n + 40 < sizeof line; i++) {
            uint64_t v = 0;
            if (g_gate_sp > 0x10000ull && g_gate_sp < 0x800000000000ull)
                v = *(const uint64_t *)(uintptr_t)(g_gate_sp + (uint64_t)i * 8);
            n += snprintf(line + n, sizeof line - (size_t)n, "sp+%#04x=%#llx%s\n", i * 8,
                (unsigned long long)v,
                (v >= base && v < base + 0x2600000) ? "(img)" : "");
        }
    }
    if (n > 0) logline(line);   // hilog 备份（系统缓冲）
    if (!g_files_dir[0]) return;
    char p[600];
    snprintf(p, sizeof p, "%s/dm_death.txt", g_files_dir);
    int fd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) return;
    (void)!write(fd, line, (size_t)n);
    close(fd);
}
// 阶段标签：实验 K 连跑三站（启动链 → 对象工厂 → nnMain），同一批插桩在三站都会响。
// 每条日志带上"当前在哪一站"，读日志时才分得清"这次命中属于哪一站"。
static const char *g_k_stage = "准备";
// 日志串行化用的递归锁（见 klog 里的理由）
static pthread_mutex_t g_klog_mu;
static pthread_once_t  g_klog_once = PTHREAD_ONCE_INIT;
static void klog_mu_init(void) {
    pthread_mutexattr_t a;
    pthread_mutexattr_init(&a);
    pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&g_klog_mu, &a);
    pthread_mutexattr_destroy(&a);
}
// ═══ np159/160/161：日志输出模式 + **预算制**（2026-10-20 深夜）═══
// 四连实证：每次"冻结"都停在**下一次日志输出**上——文件写（1.1MB trace 全量重写卡死、
// 单行 fflush 卡死）、hilog 打点（突发 70 行 dump 中途卡死）、两者并存时同样卡死。
// 静默轮（np161 前一轮）反而走得最远（游戏越过旧停车点后死掉——另一个问题）。
// ⇒ 结论：本平台在这进程的 I/O 压力下，文件写与 hilog 都会**阻塞式楔住**调用线程；
//   客体线程一旦堵在 klog 里，游戏看起来就是"冻结"。
// 修法 = **预算制**（沿用 ConvertToTimeSpan 的"全局截断"范式）：
//   文件通道 ≤400KB、hilog 通道 ≤4000 行；用完各打一条"预算用尽"然后静默。
//   诊断期（启动链/前 400KB）信息完整，之后游戏不再被我们的日志拖住。
//   模式：0=文件+hilog（默认） 1=只 hilog 2=全静默（只计数）。
// np172（2026-10-21）：窗口 1.5MB→6MB。原因：np171/np172 已把游戏推进到 res::System /
//   Resource Loading 三线程 / TextureArchiveWork（日志 1.5 万行），1.5MB 在"主线程停车点之前"
//   就用尽了 ⇒ 又是 np167 那类**观测窗口不足造出的假停滞**。仍然是逐行写+fflush，
//   没有大单写（楔死的是大单写，不是总量）。
#define DM_LOG_FILE_BUDGET (6144u * 1024u)
#define DM_LOG_HILOG_LINES 20000u
static int g_klog_mode;
// ═══ np221：**运行时开关文件**（免重构建做 A/B）═══
static void klog(const char *fmt, ...);   // 前向声明（下面马上要用，定义在本函数之后）
// 为什么加：np215–np220 五轮里每轮只能验一个假设（改代码→构建→装机→跑≈8 分钟），
//   而这一轮有三个可疑变量（键解析替身 / NVN getter / 手柄输入链）同时在场。
// 做法：读 `/data/local/tmp/dm_switch.txt` 第一行整数，按位解释；文件不存在则全 0（=当前行为）。
//   依据：真机日志里 `[fs] open 失败 ...（/data/local/tmp/romfs/...）errno=2` —— errno 是
//   2(ENOENT) 不是 13(EACCES)，说明应用**能进这个目录**，读一个文件应当可行。
//   若读不到，`[开关]` 那行会显示 0，也就是没生效（那时再退回改代码构建）。
//   位0 = 键解析替身（1=恢复旧替身"返全零对象"，0=放行原实现）
//   位1 = NVN getter 退化（1=所有 *Get* 真实现停用、落回通用桩返 0，即 np214 之前的行为）
int dm_probe_sw(int bit) {
    static int flags = -1;
    if (flags < 0) {
        int v = 0;
        FILE *f = fopen("/data/local/tmp/dm_switch.txt", "r");
        if (f) { if (fscanf(f, "%d", &v) != 1) v = 0; fclose(f); }
        flags = v;
        klog("[开关] /data/local/tmp/dm_switch.txt → %d（位0 键解析替身=%d，位1 getter退化=%d）%s\n",
             v, (v >> 0) & 1, (v >> 1) & 1, f ? "" : "（文件不存在，全 0）");
    }
    return (flags >> bit) & 1;
}

static void klog(const char *fmt, ...) {
    if (!g_k_log) return;
    static unsigned long n_lines;
    static size_t n_bytes;
    static int file_cut, hilog_cut;
    n_lines++;
    if (g_klog_mode == 2) return;   // 全静默：只计数不输出
    char body[1200];
    va_list ap; va_start(ap, fmt);
    vsnprintf(body, sizeof body, fmt, ap);
    va_end(ap);
    // 带线程 ID：游戏有多个线程（探针主线程 / sead 任务线程 / Havok 物理工作线程）都在打日志，
    // 没有 tid 分不清哪行是谁写的（2026-10-18 加：判读主线程卡在哪必需）。
    // 真线程后必须串行化（否则多线程 fprintf 会互相穿插）。**必须递归锁**：
    // 万一某线程在写日志时崩了、故障处理器（gate_dump_fault）又要写日志，
    // 非递归锁会把它自己锁死 ⇒ 整份日志从那一刻静默（2026-10-18 现场嫌疑之一）。
    if (g_klog_mode != 1 && !file_cut) {
        pthread_once(&g_klog_once, klog_mu_init);
        // ⚠ 2026-10-22（S3 血的教训）：**必须 trylock**。
        // 真机现象：subsdk0 的构造函数卡住时，日志"连超时那一行都打不出来"，主线程也一起不动
        // ⇒ 那个线程正卡在 fprintf/fflush 里**持着日志锁**（本平台"文件写会楔住调用线程"早有记录），
        // 于是所有后续 klog 全堵在拿锁上。看门狗本身生效了，但它的日志发不出去。
        // 改成 trylock：拿不到就跳过文件通道（改走 hilog），**日志永远不会把调用线程堵死**。
        if (pthread_mutex_trylock(&g_klog_mu) == 0) {
            size_t n = (size_t)fprintf(g_k_log, "[阶段 %s][T%ld] %s", g_k_stage, (long)gettid(), body);
            fflush(g_k_log);
            pthread_mutex_unlock(&g_klog_mu);
            n_bytes += n;
            if (n_bytes > DM_LOG_FILE_BUDGET) {
                file_cut = 1;
                if (pthread_mutex_trylock(&g_klog_mu) == 0) {
                    fprintf(g_k_log, "[日志] 文件通道预算用尽（%zu 字节）⇒ 后续只走 hilog\n", n_bytes);
                    fflush(g_k_log);
                    pthread_mutex_unlock(&g_klog_mu);
                }
            }
        }
        // else：别的线程卡在写日志里（fprintf/fflush 楔住）⇒ **本行直接丢弃**。
        // 别退到 hilog：hilog 同样会楔（本文档头部早有记录），退了就是二次堵死。
        else { return; }
    }
    if (!hilog_cut) {
        logline(body);
        if (n_lines > DM_LOG_HILOG_LINES) {
            hilog_cut = 1;
            logline("[日志] hilog 通道预算用尽 ⇒ 后续全静默（游戏不再被日志拖住）");
        }
    }
}

// ═══ np151：线程快照（内核级等待点取证）═══
// 背景（真机实证链）：np149/np150 证实主线程**硬阻塞**（25s vs 90s 日志逐项一致、
// 轨迹环不环绕）；np151 插桩分析——PLT 等待族 0 打点、svc 同步层 0 调用、svc 门 0 命中
// ⇒ 冻结点在**纯 guest 原生代码**里，宿主侧仪器全部失灵。
// 取证换成内核视角：/proc/self/task/*/stat（线程状态）+ wchan（内核等待点）+
// syscall（正在执行的系统调用号与参数）。同 uid ⇒ App 自己读一定可读；hdc shell
// 是另一个 uid 读不了沙箱内进程 ⇒ 必须借一个**活着的宿主线程**来读——宿主 pthread
// 永不被调度（仪器铁律 np26–29/np136）⇒ 借 ArkTS UI 线程 setInterval → NAPI 进来打。
// 用法：ArkTS 每 2 秒调一次；只在探针运行期（g_k_log 开着）打，运行期外静默。
int g_tick_tid;   // 帧线程 tid（首个 tick 记；快照里打 ←帧 标记）

// np219：持锁表 dump（定义在互斥量表之后）。动机：np218 上机实测**三个游戏线程全部
// 阻塞在 futex、进程 CPU 只有 0.4 秒**⇒ 硬死锁。最可疑的机制是「持锁线程已死/被 longjmp
// 跳过」——sdk 有 7 个构造函数崩了被 setjmp 兜走，若崩时正持着某个 HLE 互斥量，
// 那把锁永远没人放。快照里把**每个被持有的锁 + 持有者 tid + 该 tid 是否还活着**打出来，
// 一眼定谳。
static void mtx_dump_held(void);
static void hle_cur_dump(void);

void dm_probe_thread_probe(void) {
    if (!g_k_log) return;
    // 节流：最多每 100ms 一份（工人每 250ms 醒一次；np153–np158 教训：节流 400ms+
    // 时主线程停车常发生在节流窗口内 ⇒ 抓不到；100ms 保证停车后 1 次醒来内必落一份）
    static struct timespec last;
    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts);
    if (last.tv_sec && (now_ts.tv_sec - last.tv_sec) * 1000 +
        (now_ts.tv_nsec - last.tv_nsec) / 1000000 < 100) return;
    last = now_ts;
    klog("[线程快照] =====\n");
    DIR *d = opendir("/proc/self/task");
    if (!d) { klog("[线程快照] 打不开 /proc/self/task\n"); return; }
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *t = de->d_name;
        if (t[0] < '0' || t[0] > '9') continue;
        char p[128], line[2048];
        char state = '?';
        long cpu = 0;
        // ① stat：取 state 与 utime+stime（") " 后第 12/13 个 token = 整体第 14/15 字段）
        snprintf(p, sizeof p, "/proc/self/task/%s/stat", t);
        FILE *f = fopen(p, "r");
        if (f && fgets(line, sizeof line, f)) {
            char *rp = strrchr(line, ')');
            if (rp && rp[1] == ' ') {
                state = rp[2];
                long v[13]; int nv = 0;
                char *q = rp + 2;
                while (nv < 13) {
                    while (*q == ' ') q++;
                    if (*q < '0' || *q > '9') break;
                    v[nv++] = strtol(q, &q, 10);
                }
                if (nv == 13) cpu = v[11] + v[12];   // utime + stime（ticks）
            }
        }
        if (f) fclose(f);
        // ② wchan：内核等待点（线程睡在哪个内核函数上）
        char wc[80] = "(不可读)";
        snprintf(p, sizeof p, "/proc/self/task/%s/wchan", t);
        f = fopen(p, "r");
        if (f) {
            if (fgets(wc, sizeof wc, f)) {
                char *nl = strpbrk(wc, "\r\n"); if (nl) *nl = 0;
                if (!wc[0]) snprintf(wc, sizeof wc, "(空)");
            }
            fclose(f);
        }
        // ③ syscall：正在执行的系统调用（running 或 "N x0 x1 …"；睡在内核里时给出调用号与参数）
        char sc[200] = "(不可读)";
        snprintf(p, sizeof p, "/proc/self/task/%s/syscall", t);
        f = fopen(p, "r");
        if (f) {
            if (fgets(sc, sizeof sc, f)) {
                char *nl = strpbrk(sc, "\r\n"); if (nl) *nl = 0;
                if (!sc[0]) snprintf(sc, sizeof sc, "(空)");
            }
            fclose(f);
        }
        klog("[线程快照] tid=%s%s state=%c cpu=%ld轮 wchan=%s syscall=%s\n",
             t, (g_tick_tid && atoi(t) == g_tick_tid) ? "←帧线程" : "",
             state, cpu, wc, sc);
    }
    closedir(d);
    mtx_dump_held();
    hle_cur_dump();
}

// ── NVN 最小空设备 + 按名字记账桩（2026-10-17 深夜，Task #9）──
// 真机实证（np20）：游戏只从 nvnBootstrapLoader 取**两个**名字：nvnDeviceInitialize /
// nvnDeviceGetProcAddress；其余 NVN 函数全走 GetProcAddress 按名字取。
// 真机实证（np25）：GetProcAddress 被调上百次；但之前**全返同一个通用桩**
//   ⇒ 日志只知道"被取走多少次"，看不出游戏**先撞哪一族、先撞哪个函数** ⇒ 无法排期实现。
// 本轮改法：每个名字一个独立桩（表在 dm_nvn_stubs.inc，由 tools/gen_nvn_stubs.py 生成）。
//   桩一律 `return 0` —— 与改前"全返同一个通用桩"**完全等价**，纯观测、不改语义。
// 日志给出三样东西：
//   ① **首次真正调用的顺序**（这就是 NVN 523 个 API 的实现排期依据）
//   ② 每 2 秒一行的**族合计**（由游戏自己的帧线程打出，见 plt_stub 里的 ConvertToTimeSpan tick）
//   ③ 崩点现场附带一份非零计数汇总（见 gate_dump_fault → nvn_api_summary）
// 参考实现：仓库 dm/gfx/nvn/hle_nvn.c（macOS/Metal 版为「强制接管 + 名字表 + 计数跳板」）。
// 表本身（族枚举 + 名字表 + 523 个记账桩 + 桩地址表）由 tools/gen_nvn_stubs.py 生成：
//   python dm/ohos/tools/gen_nvn_stubs.py     # 输入 = 仓库 dm/gfx/nvn/names.txt（NVN 完整 API 面）
// ⚠ np26 纠正：早先以为"BotW 只用 120 个 NVN 函数"——那是旧日志只记前 120 次的产物。
//   真机实测查询 636 次、去重 ~520 个，正好等于 names.txt 的 523 个 ⇒ 一律以全表为准。
static uint64_t nvn_api_touch(int idx);       // 前置声明：.inc 里的桩体要调它（定义在 .inc 之后）
#include "dm_nvn_stubs.inc"

static uint64_t nvn_stub(void) { return 0; }  // 兜底桩：清单外的名字（按名记录一次，换游戏会遇到）
static int g_n_nvn_unknown;

static uint64_t nvn_api_touch(int idx) {
    nvn_api_t *a = &g_nvn_api[idx];
    uint64_t old = __atomic_fetch_add(&a->calls, 1, __ATOMIC_RELAXED);
    if (old == 0)
        klog("[NVN] 首次调用 %-40s (%-11s)\n", a->name, g_nvn_fam_name[a->fam]);
    else if ((old % 10000) == 0)
        klog("[NVN] 累计 %-40s (%-11s) = %llu 次\n", a->name, g_nvn_fam_name[a->fam],
             (unsigned long long)(old + 1));
    return 0;
}

// 汇总：只打非零项，按族分组。崩点现场与收尾都调它——这 523 个 API 里到底谁在干活，一眼看出。
static void nvn_api_summary(const char *where) {
    int nz = 0;
    for (int i = 0; i < NVN_API_N; i++) if (g_nvn_api[i].calls) nz++;
    if (!nz) { klog("[NVN] 汇总（%s）：%d 个 API 一个都没被真正调用过\n", where, NVN_API_N); return; }
    klog("[NVN] 汇总（%s）：%d/%d 个 API 被调用过\n", where, nz, NVN_API_N);
    for (int f = 0; f < NVN_FAM__N; f++) {
        uint64_t sum = 0; int cnt = 0;
        for (int i = 0; i < NVN_API_N; i++)
            if (g_nvn_api[i].fam == f && g_nvn_api[i].calls) { sum += g_nvn_api[i].calls; cnt++; }
        if (!cnt) continue;
        klog("[NVN]   %-11s %d 个函数 合计 %llu 次\n", g_nvn_fam_name[f], cnt,
             (unsigned long long)sum);
        for (int i = 0; i < NVN_API_N; i++)
            if (g_nvn_api[i].fam == f && g_nvn_api[i].calls)
                klog("[NVN]     %-40s %llu\n", g_nvn_api[i].name,
                     (unsigned long long)g_nvn_api[i].calls);
    }
}

// 观察窗之一：**族合计**（谁在干活）。只在有变化时打一行，族级不刷屏。
// ⚠ 谁来调很关键：np26/np27 真机用"宿主看护线程每 N 秒打一次"——结果**一行都没出**，
//   而同一时刻游戏线程的日志正常。说明我们起的宿主线程根本没被调度（见 nvn_host_thread_probe）。
//   ⇒ 改成由**游戏自己的帧线程**打：plt_stub 里 ConvertToTimeSpan 每帧一次，每 120 次 ≈ 2 秒。
static void nvn_family_totals(uint64_t out[NVN_FAM__N]) {
    for (int f = 0; f < NVN_FAM__N; f++) out[f] = 0;
    for (int i = 0; i < NVN_API_N; i++)
        out[g_nvn_api[i].fam] += __atomic_load_n(&g_nvn_api[i].calls, __ATOMIC_RELAXED);
}
static void nvn_family_report(const char *who) {
    static uint64_t prev[NVN_FAM__N];                 // 初值全 0 ⇒ 第一次调用就打全量
    uint64_t now[NVN_FAM__N];
    nvn_family_totals(now);
    char line[320];
    size_t used = 0;
    int changed = 0;
    line[0] = '\0';
    for (int f = 0; f < NVN_FAM__N; f++) {
        if (now[f] == prev[f]) continue;
        changed = 1;
        prev[f] = now[f];
        if (used + 40 >= sizeof line) continue;        // 装不下就跳过这一族（不截断上一次的输出）
        int n = snprintf(line + used, sizeof line - used, "%s=%llu ",
                         g_nvn_fam_name[f], (unsigned long long)now[f]);
        if (n > 0) used += (size_t)n;
    }
    if (changed) klog("[NVN] 族合计（%s）：%s\n", who, line);
}

// 观察窗之二：**宿主线程到底跑不跑**（一个只打心跳的最小实验）。
// np26/np27：宿主看护线程一行日志都没有，而 pthread_create 返回成功 ⇒ 怀疑被游戏线程饿死。
// 这里只在前 3 拍打心跳；打得出来说明宿主线程能跑，打不出来就是调度饥饿（本身就是重要结论）。
static void *nvn_host_thread_probe(void *arg) {
    (void)arg;
    for (int i = 1; i <= 3; i++) {
        // ⚠ 别用 usleep(10*1000*1000)：POSIX 只保证 <1e6，某些 libc 直接 EINVAL ⇒ 瞬间跑完不打一行。
        struct timespec ts;
        ts.tv_sec = 2;
        ts.tv_nsec = 0;
        while (nanosleep(&ts, &ts) != 0 && errno == EINTR) { }
        klog("[NVN] 宿主线程心跳 %d/3（能打出这行 ⇒ 宿主线程有被调度）\n", i);
    }
    return NULL;
}
static void nvn_host_thread_probe_start(void) {
    static int started;
    if (started) return;
    started = 1;
    pthread_t t;
    if (pthread_create(&t, NULL, nvn_host_thread_probe, NULL) == 0) pthread_detach(t);
    else klog("[NVN] 宿主线程起不来（不影响主线）\n");
}

static int g_inner_caught;      // np110：内层指针是否被写过
static size_t g_inner_ctor;
static uint64_t g_inner_ctorentry;
static uint64_t g_inner_val;
static uint64_t g_inner_init;
static int g_gate_caught;        // np107：门字节是否已变非零（抓住写入者用）
static size_t g_gate_ctor;       // np107：写入者的构造器序号
static uint64_t g_gate_ctorentry;
static uint64_t g_gate_obj;
static unsigned g_gate_val;
// ===== np111：RootTask::calc 的「启动阶段状态机」（P0-1 / P0-2）=====
// 离线结论（2026-10-19 晚，main.patched.bin 定向反汇编 + 字符串核实，**不是猜**）：
//   0x726838 = **RootTask::calc**：日志名串 0x1b788b4「RootTask::calc」；
//     虚表 0x215a160 的槽 22（+0xb0）= 该函数；构造器 0x7277ec（alloc 0x3d0，置 phase=0）。
//   阶段分发：`ldrb w8,[x19,#0x200]` → `cmp w8,#9` → `b.hi 返回` → 跳表 0x1c1f448（10 项）。
//   各阶段（静态读出）：
//     phase 0 → 建 "Error Viewer"、"Volante"、"KingSystem"（0x108 字节，存 [+0x358]）
//               并起 "SoundCalcThread"，末尾置 phase=1
//     phase 1 → 等 KingSystem 的 **[+0xd4] ∈ {3,4}**；满足才销毁它并置 phase=2
//     phase 2 → 建管理器 "StarterPackMgr"（0xdb9284，站点 0x726f60）→ 置 phase=3
//     phase 3 → 0xdb9af4 返 bit0 就继续等，否则置 4
//     phase 4 → 装 AIDef_Game.product.byml → 0x726f60 再建管理器 → 3；或置 5
//     phase 5 → 6 → 7（连跳）；phase 7 → 大块收尾 → 8；phase 8 → 等 0x731340 → 9
//   ⇒ **只要 phase 停在 0/1，0x726f60 永远到不了**，管理器不建 ⇒ 包不读 ⇒ 一切空。
// 本探针只读不改：抓 this、读 phase、读 phase1 的放行条件、数"建管理器"命中。
static int g_rt_calls;        // calc 进入次数
static uint64_t g_rt_this;    // RootTask 对象地址
static int g_rt_phase = -1;   // 上一次读到的阶段
static uint32_t g_rt_seen;    // 见过的阶段位图（bit n = phase n）
static int g_rt_trans;        // 阶段转移次数
static int g_rt_mgr;          // 0x726f60「建管理器」命中次数
static int g_rt_ctor;         // 0x7277ec 构造命中次数

static int rt_read_u32(uint64_t addr, uint32_t *out) {
    if (addr < 0x10000ull || addr >= 0x800000000000ull) return 0;
    *out = *(const uint32_t *)(uintptr_t)addr;
    return 1;
}
static void rt_dump(const char *why) {
    if (!g_rt_this) { klog("[RootTask] %s：还没拿到 this（calc 一次都没进）\n", why); return; }
    uint64_t t = g_rt_this;
    uint32_t ph = *(const uint8_t *)(uintptr_t)(t + 0x200);
    uint64_t ks = *(const uint64_t *)(uintptr_t)(t + 0x358);
    uint32_t st = 0;
    int ok = rt_read_u32(ks + 0xd4, &st);
    klog("[RootTask] %s：进入=%d phase=%u 见过=%#x 转移=%d 建管理器=%d 构造=%d\n",
         why, g_rt_calls, ph, g_rt_seen, g_rt_trans, g_rt_mgr, g_rt_ctor);
    klog("[RootTask] %s：+0x208=%#llx +0x210=%#llx +0x218=%#llx +0x220=%#llx +0x228=%#llx\n",
         why,
         (unsigned long long)*(const uint64_t *)(uintptr_t)(t + 0x208),
         (unsigned long long)*(const uint64_t *)(uintptr_t)(t + 0x210),
         (unsigned long long)*(const uint64_t *)(uintptr_t)(t + 0x218),
         (unsigned long long)*(const uint64_t *)(uintptr_t)(t + 0x220),
         (unsigned long long)*(const uint64_t *)(uintptr_t)(t + 0x228));
    klog("[RootTask] %s：KingSystem([+0x358])=%#llx 其[+0xd4]=%s%d（phase1 放行条件：3 或 4）\n",
         why, (unsigned long long)ks, ok ? "" : "读不到/", ok ? (int)st : 0);
}
// np122：这两个"元函数"必须是 static 的前置声明——bootstrap 的补充表要用它们。
//   教训（真机实证）：np118 只把它们挂进 nvnDeviceGetProcAddress 的分支里，bootstrap 查不到，
//   Volante 初始化仍然在 0x9fc59c 早退（NvnDeviceInitialize 拿到 0），NVN 函数表一整个没注册
//   ⇒ 建对象 worker 调 *(0x22d7428) 空指针崩。
static uint64_t nvn_dev_init(void *dev, const void *builder);
static uint64_t nvn_dev_gpa(void *dev, const char *name);

static void *nvn_real_lookup(const char *name) {
    if (!name) return NULL;
    if (!strcmp(name, "nvnDeviceInitialize"))     return (void *)(uintptr_t)&nvn_dev_init;
    if (!strcmp(name, "nvnDeviceGetProcAddress")) return (void *)(uintptr_t)&nvn_dev_gpa;
    void *r = dm_nvn_vk_real(name);         // 通用 NVN 池/缓冲层（L1，UMA 直通）
    if (r) return r;
    r = dm_nvn_real(name);                  // 通用 NVN 设备族胶水层（L1）
    // np101：池/缓冲族的 23 个真实现已搬去 dm_nvn_vk.c（上面的 dm_nvn_vk_real），
    //   探针里不再保留自己的表。设备初始化/GetProcAddress 两个仍在本文件（下面）。
    return r;
}

// np122：nvnBootstrapLoader 的宿主入口——**永远不返回 0**。
//   语义：框架层（dm_nvn_bootstrap）认识就给它，不认识就给记账桩。
//   为什么必须兜底：真机实证 np118 直接绑 dm_nvn_bootstrap，它查不到时返回 0，
//   于是 Volante 初始化在 0x9fc59c 早退（以为驱动没有驱动初始化），NVN 函数表整个没注册，
//   建对象 worker 随后 blr 空指针（pc=0）。原 plt_stub 的语义就是"取不到也给桩"，这里保持一致。
static uint64_t host_nvn_bootstrap(const char *name) {
    void *p = dm_nvn_bootstrap(name);
    if (p) return (uint64_t)(uintptr_t)p;
    static int n_un;
    if (++n_un <= 40)
        klog("[NVN] BootstrapLoader(「%s」) → 兜底桩\n", name ? name : "(null)");
    return (uint64_t)(uintptr_t)&nvn_stub;
}

static uint64_t nvn_dev_init(void *dev, const void *builder) {
    (void)builder;
    if (dev) *(uint64_t *)dev = 0x4E564E44ull;   // 'NVND' 魔数（不 memset：设备对象大小未知，防越界）
    klog("[NVN] nvnDeviceInitialize → 1（空设备）\n");
    nvn_host_thread_probe_start();                 // 只打 3 拍心跳，测"宿主线程能不能被调度"
    return 1;                                     // NVNboolean 真 = 初始化成功
}
static uint64_t nvn_dev_gpa(void *dev, const char *name) {
    (void)dev;
    if (!name) return (uint64_t)(uintptr_t)&nvn_stub;
    // 这两个是**真实现**，不能走记账桩：np26 真机日志里游戏在 DeviceInitialize 之后又
    // GetProcAddress("nvnDeviceInitialize") 取了一次（4416/4326 行）——若发记账桩（返 0），
    // 第二次初始化会被告知"失败"。同理 GetProcAddress 自己也可能被再取一次。
    if (!strcmp(name, "nvnDeviceInitialize"))    return (uint64_t)(uintptr_t)&nvn_dev_init;
    if (!strcmp(name, "nvnDeviceGetProcAddress")) return (uint64_t)(uintptr_t)&nvn_dev_gpa;
    // ★ 真实现优先（内存池/缓冲）：必须在记账桩表**之前**查，否则会拿回返 0 的桩。
    //   ⚠ 走真实现的函数不再进 g_nvn_api[].calls 计数（桩没被调），所以 §28 那张
    //   "14/523" 汇总表会把它们显示为 0——判读时以 `[NVN] 池/缓冲` 那几行日志为准。
    {
        void *real = nvn_real_lookup(name);
        if (real) {
            static int n_real_logged;
            if (++n_real_logged <= 60)
                klog("[NVN] GetProcAddress(「%s」) → **真实现**\n", name);
            return (uint64_t)(uintptr_t)real;
        }
    }
    for (int i = 0; i < NVN_API_N; i++) {
        if (!strcmp(g_nvn_api[i].name, name)) {
            static int n_logged;
            if (++n_logged <= 1200)
                klog("[NVN] GetProcAddress(「%s」) → 独立记账桩 #%d/%s\n",
                     name, i, g_nvn_fam_name[g_nvn_api[i].fam]);
            return (uint64_t)(uintptr_t)g_nvn_thunks[i];
        }
    }
    // 清单外的名字 = §18.3 那份 120 项清单漏了东西（换游戏时一定会遇到），记下来但别刷屏。
    // ⚠ np26 真机：上限 40 被**打满**了 ⇒ 说明原始清单本身就截断了（旧代码只记前 120 个
    //   GetProcAddress）⇒ 这里放到 400，一次抓全，再把清单补进 nvn-api-botw.txt。
    if (++g_n_nvn_unknown <= 3000)
        klog("[NVN] ⚠ 清单外名字「%s」→ 兜底桩（该补进 nvn-api-botw.txt）\n", name);
    return (uint64_t)(uintptr_t)&nvn_stub;
}
static volatile int g_k_hits;
static uint64_t g_k_hit_lr[16];
static volatile int g_k_faulted;
static uint64_t g_k_rv;
static module_t g_k_m;
static pthread_key_t g_k_key[8];
// np133：sdk 模块的装载状态（M1）——声明放前面，线程蹦床要用
static module_t g_k_s;
static pthread_key_t g_s_key[8];
struct sdk_sym { const char *name; uint64_t addr; };
// np135：512 → 16384。M1 第二步要做 PLT 自解析，导出表必须收**全量**——
//   离线实证 sdk 有 14,834 个导出（dynsym 14,849），原先只收 nn::os 一族不够用。
#define SDK_OS_MAX 40000   // 7 模块合计导出 ≈ 26,865（原 16384 只够 sdk 自己）
static struct sdk_sym g_sdk_os[SDK_OS_MAX];
static int g_n_sdk_os;
static uint64_t sdk_ret0(uint64_t a, uint64_t b) { (void)a; (void)b; return 0; }   // sdk 未识别导入的兜底

// 把崩点现场落进日志。**必须在 siglongjmp 之后调用**——不在信号处理器里碰 stdio。
// 帧链回溯要解引用可疑地址，所以临时换上探针自己的兜底处理器（probe_segv_handler + safe_read64），
// 走完再换回来；否则一次野指针会落到 gate_fault_handler 的"未武装"分支 _exit(139)，整份日志都丢。
static void gate_dump_fault(const char *where) {
    if (!g_gate_have_regs) return;
    klog("=== 崩点现场（%s）：pc=%#llx sp=%#llx sig=%d 出错地址(si_addr)=%#llx ===\n", where,
         (unsigned long long)g_gate_fault_pc, (unsigned long long)g_gate_sp, g_gate_fault_sig,
         (unsigned long long)g_gate_fault_addr);
    for (int i = 0; i < GATE_REG_N; i++)
        klog("  x%-2d=%#llx%s", i, (unsigned long long)g_gate_regs[i], (i % 2) ? "\n" : "  ");
    struct sigaction sa, o1, o2;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = probe_segv_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &o1);
    sigaction(SIGBUS, &sa, &o2);
    // 帧链：AArch64 约定 [x29] = 上一帧 x29、[x29+8] = 本帧返回地址。
    // 这一串返回地址就是**真实调用栈**，比任何"推断的链"都可信。
    klog("  帧链（fp → 本帧返回地址 / 上一帧 fp）:\n");
    uint64_t fp = g_gate_regs[29];
    for (int d = 0; d < 16; d++) {
        uint64_t lr = 0, next = 0;
        if (!fp || !safe_read64(fp + 8, &lr) || !safe_read64(fp, &next)) break;
        klog("    [%2d] fp=%#llx  返回地址=%#llx\n", d, (unsigned long long)fp, (unsigned long long)lr);
        if (next <= fp) break;
        fp = next;
    }
    sigaction(SIGSEGV, &o1, NULL);
    sigaction(SIGBUS, &o2, NULL);
    // 顺手把 NVN 记账汇总打出来：崩在哪一步 + 此时 NVN 用到什么程度，放在一起才看得出因果
    nvn_api_summary(where);
    klog("=== 现场结束 ===\n");
}
// ⭐ np177：**per-thread 门命中取证**。为什么要：np175/176 之后卡点变成"主线程在 guest/ sdk
//   代码里打转、该区间零 SDK 调用"，普通环形轨迹分不清"guest 自旋"还是"sdk 内部自旋"。
//   而 sdk 的内部同步（自解析、走 svc 门）若在原地等，一定会**踩门**——按线程记账就能分辨：
//   · 主线程行 n 高速增长 ⇒ sdk 内部自旋（门桩返回 0 = 立刻返回 ⇒ 等不到就重试）；
//   · 主线程行 n 不涨    ⇒ 纯 guest 代码自旋（下一步只能上采样器取 PC）。
typedef struct { uint32_t tid; uint32_t n; uint64_t lr[4]; } gate_thr_t;
static gate_thr_t g_gthr[24];
static uint32_t g_main_pthread;        // nnMain 所在线程（在"终站：nnMain"处登记）

static uint64_t k_gate_stub(void) {
    uint64_t lr; __asm__ volatile("mov %0, x30" : "=r"(lr));
    if (g_k_hits < 16) g_k_hit_lr[g_k_hits] = lr;
    g_k_hits++;
    // 无锁登记：并发下最多丢几条，够判"谁在踩门"（不进任何锁，避免把门本身变成卡点）
    uint32_t me = (uint32_t)(uintptr_t)pthread_self();
    for (int i = 0; i < 24; i++) {
        uint32_t t = __atomic_load_n(&g_gthr[i].tid, __ATOMIC_RELAXED);
        if (t == 0) {
            uint32_t expect = 0;
            if (__atomic_compare_exchange_n(&g_gthr[i].tid, &expect, me, 0,
                                            __ATOMIC_RELAXED, __ATOMIC_RELAXED)) t = me;
            else if (expect != me) continue;
        }
        if (t == me) {
            uint32_t k = g_gthr[i].n & 3u;
            g_gthr[i].lr[k] = lr;
            __atomic_store_n(&g_gthr[i].n, g_gthr[i].n + 1u, __ATOMIC_RELAXED);
            break;
        }
    }
    return 0;   // 首跑：先一律"无事发生"（0），看流程能走多远
}
// np136（批次 B）：门观测桩命中快照。挂在 nnMain 调用前 + trace 定时器里——
//   判互斥量实现是否踩 svc 门（ArbitrateLock 等）。命中 0 = 纯用户态自旋链路走通；
//   >0 则前 8 个返回点离线定位（注意 LR 可能落在 main 或 sdk 映像，按值域分辨）。
// ⭐ np178：**主线程 PC 采样器**——"guest 代码里自旋"唯一能取到证据的仪器。
//   为什么必须上它（np175–np177 实证）：主线程停在 `建/查#34「TextureArchiveWork」` 之后，
//   该区间 **零 SDK 调用**（轨迹无记录）、**零等待打点**、**门命中总数 0**（np177 实测）
//   ⇒ 它纯在 guest 指令流里打转；环形轨迹、门计数、等待打点全部看不见。
//   做法：给 **nnMain 所在线程**挂 SIGPROF 定时器（20ms），handler 只把中断点的 pc/fp
//   写进静态数组（async-signal-safe，不加锁不分配），由帧线程定期把直方图落到 dm_gate.txt。
//   判读：直方图里那一两个 pc 就是"原地打转"的位置；配上 fp 可离线回溯。
#define SMP_N 4096
static volatile uint32_t g_smp_n;
static uint64_t g_smp_pc[SMP_N];
static uint64_t g_smp_fp[SMP_N];
static uint32_t g_smp_tid[SMP_N];     // np211：样本所属线程（轮投全部游戏线程用）
// np211b：热点样本的寄存器快照（x0-x3/x30）。主线程 789/789 停在 musl 符号哈希查找里，
//   只有 LR（x30）能告诉我们"是谁调进来的"——静态考古 musl 太慢，一次采样定音。
static uint64_t g_smp_r0[SMP_N], g_smp_r1[SMP_N], g_smp_r2[SMP_N], g_smp_r3[SMP_N], g_smp_lr[SMP_N];
static uint64_t g_smp_sp[SMP_N];     // np211e：中断 SP（扫栈找 guest 调用点的基准）
static int g_smp_started;
static pthread_t g_main_pth;          // 主线程句柄（给 pthread_kill 定向投信号用）
static volatile uint32_t g_smp_kicks; // np185 实验 G：已投递的采样信号数（诊断"没样本"是没发还是没送达））

static void smp_handler(int sig, siginfo_t *si, void *uc_) {
    (void)sig; (void)si;
    ucontext_t *uc = (ucontext_t *)uc_;
    uint32_t i = __atomic_fetch_add(&g_smp_n, 1u, __ATOMIC_RELAXED);
    if (i < SMP_N) {
        g_smp_pc[i] = (uint64_t)uc->uc_mcontext.pc;
        g_smp_fp[i] = (uint64_t)uc->uc_mcontext.regs[29];
        g_smp_tid[i] = (uint32_t)(uintptr_t)pthread_self();
        g_smp_r0[i] = (uint64_t)uc->uc_mcontext.regs[0];
        g_smp_r1[i] = (uint64_t)uc->uc_mcontext.regs[1];
        g_smp_r2[i] = (uint64_t)uc->uc_mcontext.regs[2];
        g_smp_r3[i] = (uint64_t)uc->uc_mcontext.regs[3];
        g_smp_lr[i] = (uint64_t)uc->uc_mcontext.regs[30];
        g_smp_sp[i] = (uint64_t)uc->uc_mcontext.sp;
    }
}
// 只装 handler；**不开定时器**——NDK 的 sigevent 不暴露 SIGEV_THREAD_ID，
// 改成由帧线程 pthread_kill 定向投（样本一定落在主线程上，且不依赖宿主线程被调度）。
static void smp_arm(void) {
    if (g_smp_started) return;
    g_smp_started = 1;
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = smp_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGPROF, &sa, NULL) != 0) { klog("[采样] sigaction 失败\n"); return; }
    klog("[采样] ✔ SIGPROF handler 已装（由帧线程定向投样）\n");
}
// np211：实现移到 g_thr 定义之后（要遍历线程表）；这里只留前向声明。
static void smp_kick(void);
// np178：**maps 快照**。采样点落在映像外时，必须先知道它属于哪个 .so 才能反查符号。
//   只记可执行段（r-xp）：行数可控，且足以判定"某地址属于哪个库、偏移多少"。
static void dump_maps_once(void) {
    if (!g_files_dir[0]) return;
    FILE *fp = fopen("/proc/self/maps", "r");
    if (!fp) return;
    char p[600];
    snprintf(p, sizeof p, "%s/dm_gate.txt", g_files_dir);
    int fd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) { fclose(fp); return; }
    char line[512];
    int n = 0;
    (void)!write(fd, "=== maps(全部可执行+匿名，前 500 行) ===\n", 34);
    while (fgets(line, sizeof line, fp) && n < 500) {
        // np185 实验 F：匿名映射（客人映像 main/sdk 都是 mmap 匿名区）必须收，
        //   否则自旋 PC 归不了属。收 r-xp 全部 + 匿名 rw 前 200 行。
        if (strstr(line, "r-xp") || (n < 200 && strstr(line, "rw") && !strstr(line, "/"))) {
            (void)!write(fd, line, strlen(line)); n++;
        }
    }
    close(fd);
    fclose(fp);
}
// np185 实验 F：把任意地址归类到 main / sdk / probe(.so) / 外，返回简短标签写进 out。
static void smp_cls(uint64_t a, char *out, size_t cap) {
    uint64_t mb = (uint64_t)(uintptr_t)g_k_m.base;
    uint64_t sb = (uint64_t)(uintptr_t)g_k_s.base;
    if (mb && a >= mb && a < mb + DM_IMG_MAIN_END)
        snprintf(out, cap, "main+%#llx", (unsigned long long)(a - mb));
    else if (sb && a >= sb && a < sb + DM_IMG_SDK_END)
        snprintf(out, cap, "sdk+%#llx", (unsigned long long)(a - sb));
    else if (g_self_base && a >= g_self_base && a < g_self_base + 0x400000ull)
        snprintf(out, cap, "probe+%#llx", (unsigned long long)(a - g_self_base));
    else
        snprintf(out, cap, "外%#llx", (unsigned long long)a);
}
// np185 实验 F：对采样点走帧链（[fp]=上层fp、[fp+8]=返回地址），最多 8 层。
//   自旋点的调用链 = "谁在让主线程转圈"的直接证据。
static int smp_bt(uint64_t pc, uint64_t fp, char *out, size_t cap) {
    int len = 0;
    uint64_t pcs[8]; int n = 0;
    pcs[n++] = pc;
    uint64_t f = fp;
    for (int i = 0; i < 7 && f > 0x10000ull && f < 0x800000000000ull && (f & 15ull) == 0; i++) {
        uint64_t lr = *(uint64_t *)(uintptr_t)(f + 8);
        uint64_t nf = *(uint64_t *)(uintptr_t)f;
        if (lr > 0x10000ull && lr < 0x800000000000ull && n < 8) pcs[n++] = lr;
        if (nf <= f) break;               // 必须递增，否则链断了
        f = nf;
    }
    for (int i = 0; i < n; i++) {
        char cls[48];
        smp_cls(pcs[i], cls, sizeof cls);
        len += snprintf(out + len, cap - (size_t)len, "%s%s", i ? " ← " : "", cls);
    }
    return n;
}
// np217：smp_dump 的扫栈要用受保护读（k_read_ok 定义在后面）
static int k_read_ok(const void *p, void *dst, size_t n);

static void smp_dump(const char *tag) {
    if (!g_files_dir[0]) return;
    uint32_t n = __atomic_load_n(&g_smp_n, __ATOMIC_RELAXED);
    if (n > SMP_N) n = SMP_N;
    char p[600], buf[2400];
    snprintf(p, sizeof p, "%s/dm_gate.txt", g_files_dir);
    int fd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;
    // np211：先按 tid 分桶（最多 16 个线程），每桶内各自统计 Top3 pc。
    //   判读：12 个工人线程若都挤在同一个 pc（如 0xa26f80 之后的某处），即"集体自旋等一个条件"。
    uint32_t tids[16]; uint32_t tcnt[16] = {0}; int ntid = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t t = g_smp_tid[i];
        int k, hit = -1;
        for (k = 0; k < ntid; k++) if (tids[k] == t) { hit = k; break; }
        if (hit < 0) {
            if (ntid < 16) { hit = ntid++; tids[hit] = t; tcnt[hit] = 0; }
            else continue;
        }
        tcnt[hit]++;
    }
    int len = snprintf(buf, sizeof buf, "smp[%s] 样本=%u kick=%u 线程数=%d\n", tag, n,
                       __atomic_load_n(&g_smp_kicks, __ATOMIC_RELAXED), ntid);
    for (int b = 0; b < ntid; b++) {
        uint64_t top[3] = {0}; uint32_t cnt[3] = {0};
        for (uint32_t i = 0; i < n; i++) {
            if (g_smp_tid[i] != tids[b]) continue;
            uint64_t pc = g_smp_pc[i];
            int k, hit = -1;
            for (k = 0; k < 3; k++) if (top[k] == pc) { hit = k; break; }
            if (hit < 0) {
                int slot = -1;
                for (k = 0; k < 3; k++) if (cnt[k] == 0) { slot = k; break; }
                if (slot < 0) continue;
                hit = slot; top[hit] = pc; cnt[hit] = 0;
            }
            cnt[hit]++;
        }
        len += snprintf(buf + len, sizeof buf - (size_t)len, "  tid=%u n=%u:", tids[b], tcnt[b]);
        for (int rank = 0; rank < 3; rank++) {
            int best = -1;
            for (int k = 0; k < 3; k++) if (cnt[k] && (best < 0 || cnt[k] > cnt[best])) best = k;
            if (best < 0) break;
            char cls[48];
            smp_cls(top[best], cls, sizeof cls);
            len += snprintf(buf + len, sizeof buf - (size_t)len, " %u×%s", cnt[best], cls);
            cnt[best] = 0;
        }
        len += snprintf(buf + len, sizeof buf - (size_t)len, "\n");
    }
    (void)!write(fd, buf, strlen(buf));
    // 主线程热点帧链（原逻辑保留：全局第一名 + 帧链）
    uint64_t top[16] = {0}; uint32_t cnt[16] = {0}; int used = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t pc = g_smp_pc[i];
        int k, hit = -1;
        for (k = 0; k < used; k++) if (top[k] == pc) { hit = k; break; }
        if (hit < 0) {
            if (used < 16) { hit = used++; top[hit] = pc; cnt[hit] = 0; }
            else continue;
        }
        cnt[hit]++;
    }
    uint64_t first_pc = 0; uint64_t first_fp = 0; uint32_t first_cnt = 0;
    for (int rank = 0; rank < 6; rank++) {
        int best = -1;
        for (int k = 0; k < used; k++) if (cnt[k] && (best < 0 || cnt[k] > cnt[best])) best = k;
        if (best < 0) break;
        if (rank == 0) {
            first_pc = top[best]; first_fp = g_smp_fp[0]; first_cnt = cnt[best];
            for (uint32_t i = 0; i < n; i++) if (g_smp_pc[i] == top[best]) { first_fp = g_smp_fp[i]; break; }
        }
        cnt[best] = 0;
    }
    if (first_pc) {
        char bt[600];
        smp_bt(first_pc, first_fp, bt, sizeof bt);
        char l2[700];
        int l2n = snprintf(l2, sizeof l2, "smp-bt[%s] %u×%#llx：%s\n",
                           tag, first_cnt, (unsigned long long)first_pc, bt);
        (void)!write(fd, l2, (size_t)l2n);
        // np211d：寄存器快照
        for (uint32_t i = 0; i < n; i++) {
            if (g_smp_pc[i] != first_pc) continue;
            char l3[400];
            char c0[48], clr[48];
            smp_cls(g_smp_r0[i], c0, sizeof c0);
            smp_cls(g_smp_lr[i], clr, sizeof clr);
            int l3n = snprintf(l3, sizeof l3,
                               "smp-reg[%s] x0=%s x1=%#llx x2=%#llx x3=%#llx lr=%s\n",
                               tag, c0, (unsigned long long)g_smp_r1[i],
                               (unsigned long long)g_smp_r2[i],
                               (unsigned long long)g_smp_r3[i], clr);
            (void)!write(fd, l3, (size_t)l3n);
            // np211e：**扫栈找 guest 调用点**。帧链（fp）在 guest/musl 混合栈上不可靠
            //   （实测只有一层）。改用暴力扫：从中断 SP 向下 96 字，取**第一个落在
            //   main/sdk 映像内的值** = 最靠近热点的客体调用点（musl 是被它调进的）。
            {
                uint64_t mb = (uint64_t)(uintptr_t)g_k_m.base;
                uint64_t sb = (uint64_t)(uintptr_t)g_k_s.base;
                uint64_t sp = g_smp_sp[i];          // 中断 SP（被中断线程的栈）
                // np211f：范围 0x300→0x1000（musl 内部帧 + plt_stub 宿主帧 + guest 帧
                //   可能深达几百字节每层）；并把 probe .so 与匿名区也纳入识别——
                //   guest 的返回地址通常隔着一层我们的 plt_stub（宿主函数）。
                int found = 0;
                for (int k = 0; k < 512 && found != 1; k++) {
                    uint64_t v = *(const uint64_t *)(uintptr_t)(sp + (uint64_t)k * 8);
                    // np217：**只认"像返回地址"的值**——该地址 -4 处必须是 `bl`（0x94xxxxxx）。
                    //   为什么（np211f 教训）：栈上散布着数据指针（实测第一个命中的是
                    //   img+0x22d9198 = F_wait 锁的单例对象），那不是调用点。
                    //   真正的返回地址一定紧跟一条 bl。
                    int is_ret = 0;
                    if (v >= 0x10000ull && v < 0x800000000000ull) {
                        uint32_t ins = 0;
                        if (k_read_ok((const void *)(uintptr_t)(v - 4), &ins, 4))
                            is_ret = ((ins & 0xFC000000u) == 0x94000000u);
                    }
                    if (!is_ret) {
                        // 宿主帧也按同样规则认（probe 内的调用点）
                        if (g_self_base && v >= g_self_base && v < g_self_base + 0x400000ull) {
                            uint32_t ins = 0;
                            if (k_read_ok((const void *)(uintptr_t)(v - 4), &ins, 4) &&
                                ((ins & 0xFC000000u) == 0x94000000u)) {
                                int l4n = snprintf(l2, sizeof l2,
                                                   "smp-stk[%s] 热点+%#llx 的宿主帧 = probe+%#llx（sp+%#x）\n",
                                                   tag, (unsigned long long)first_pc,
                                                   (unsigned long long)(v - g_self_base), k * 8);
                                (void)!write(fd, l2, (size_t)l4n);
                            }
                        }
                        continue;
                    }
                    if (v >= mb && v < mb + DM_IMG_MAIN_END) {
                        // 顺便解出 bl 的目标（哪个函数调进来的）
                        uint32_t ins = 0;
                        uint64_t target = 0;
                        if (k_read_ok((const void *)(uintptr_t)(v - 4), &ins, 4)) {
                            int32_t imm = (int32_t)(ins & 0x03FFFFFFu);
                            if (imm & 0x02000000) imm -= 0x04000000;
                            target = (v - 4) + (int64_t)imm * 4;
                        }
                        int l4n = snprintf(l2, sizeof l2,
                                           "smp-stk[%s] 热点+%#llx 的 guest 调用点 = img+%#llx（sp+%#x）bl→img+%#llx\n",
                                           tag, (unsigned long long)first_pc,
                                           (unsigned long long)(v - mb), k * 8,
                                           (unsigned long long)((target >= mb && target < mb + DM_IMG_MAIN_END)
                                                                ? target - mb : target));
                        (void)!write(fd, l2, (size_t)l4n);
                        found = 1;
                    } else if (sb && v >= sb && v < sb + DM_IMG_SDK_END) {
                        int l4n = snprintf(l2, sizeof l2,
                                           "smp-stk[%s] 热点+%#llx 的 guest 调用点 = sdk+%#llx（sp+%#x）\n",
                                           tag, (unsigned long long)first_pc,
                                           (unsigned long long)(v - sb), k * 8);
                        (void)!write(fd, l2, (size_t)l4n);
                        found = 1;
                    }
                }
                if (found == 1) { /* 已写 */ }
                else if (found == 0) {
                    int l4n = snprintf(l2, sizeof l2,
                                       "smp-stk[%s] 热点+%#llx：sp+0..0x1000 内无 guest/宿主地址\n",
                                       tag, (unsigned long long)first_pc);
                    (void)!write(fd, l2, (size_t)l4n);
                }
            }
            break;
        }
    }
    close(fd);
}

static void k_gate_obs_dump(const char *tag) {
    klog("[门观测:%s] k_gate_stub 共命中 %d 次；前 8 个返回点：", tag, g_k_hits);
    for (int i = 0; i < 8 && i < g_k_hits; i++)
        klog(" %#llx", (unsigned long long)g_k_hit_lr[i]);
    klog("\n");
}
// np177：per-thread 门命中落 dm_gate.txt（绕过日志预算，任何时候都能拿到）。
//   判读：主线程（带 ←main 标记）的 n 若持续增长 ⇒ 它在 sdk 内部踩门（自旋）；不涨 ⇒ 纯 guest 自旋。
static void gate_thr_dump(const char *tag) {
    if (!g_files_dir[0]) return;
    char p[600], buf[1200];
    snprintf(p, sizeof p, "%s/dm_gate.txt", g_files_dir);
    int fd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;
    int len = snprintf(buf, sizeof buf, "gate[%s] 总=%d\n", tag, g_k_hits);
    for (int i = 0; i < 24; i++) {
        uint32_t t = g_gthr[i].tid, n = g_gthr[i].n;
        if (!t || !n) continue;
        if (len > (int)sizeof buf - 220) break;
        len += snprintf(buf + len, sizeof buf - (size_t)len, "  tid=%#x n=%u%s lr=%#llx %#llx\n",
                        t, n, (t == g_main_pthread) ? " ←main" : "",
                        (unsigned long long)g_gthr[i].lr[0],
                        (unsigned long long)g_gthr[i].lr[1]);
    }
    (void)!write(fd, buf, (size_t)len);
    close(fd);
}
// np143：单例 A/B/当前实例 创建时机观测（任意线程可调，帧线程上采到的是 nnMain 运行期）。
//   依据：构造器 0x103ba34 尾部 AddRef(NULL) 定谳——它要 A（指针变量 @mem[0x232db98]，经指针槽
//   0x2264e48 双重取址）与 B（@mem[0x232db88]，槽 0x2289aa0）；"当前实例"@mem[0x232db90]（槽 0x2289af8）。
//   按客人自己的取址方式读（先读槽得指针、再解引用），槽值一并打出，可交叉验证槽是否真指向预期变量。
//   顺带增量 dump 大分配（≥0x400）：[替换] 分配 的**打印上限只有前 8 次**（构造器期不能打日志），
//   靠 g_big 数组（上限 24）才能覆盖 nnMain 段里的 A(0x21a0)/B(0x2c068) 分配。
static void dump_big_allocs(void);        // np143：前置声明（实现在下面）
static void k_ab_obs(const char *tag) {
    uint8_t *b = (uint8_t *)g_k_m.base;
    uint64_t _sa = *(const uint64_t *)(uintptr_t)(b + 0x2264e48);
    uint64_t _sb = *(const uint64_t *)(uintptr_t)(b + 0x2289aa0);
    uint64_t _sc = *(const uint64_t *)(uintptr_t)(b + 0x2289af8);
    uint64_t _A = (_sa > 0x10000ull && _sa < 0x800000000000ull)
                ? *(const uint64_t *)(uintptr_t)_sa : 0;
    uint64_t _B = (_sb > 0x10000ull && _sb < 0x800000000000ull)
                ? *(const uint64_t *)(uintptr_t)_sb : 0;
    uint64_t _C = (_sc > 0x10000ull && _sc < 0x800000000000ull)
                ? *(const uint64_t *)(uintptr_t)_sc : 0;
    klog("[AB观测] %s：A槽=%#llx→A=%#llx ；B槽=%#llx→B=%#llx ；当前实例=%#llx\n", tag,
         (unsigned long long)_sa, (unsigned long long)_A,
         (unsigned long long)_sb, (unsigned long long)_B,
         (unsigned long long)_C);
    dump_big_allocs();   // np143：改成增量输出（见函数内说明），此处可反复调用
}
// 记录型 PLT 存根：x16 = 该槽地址（thunk 里 add x16,x16,#imm 后 br），据此认出是哪个 SDK 符号。
typedef struct { uint64_t slot; const char *name; } plt_ent_t;
static plt_ent_t g_plt[512];
static int g_nplt;
static int g_nimpl;      // 直连宿主的导入个数（nn::fs 等，见重定位循环）
static int g_nimpl_sdk;  // np133：解析到 sdk 模块的导入个数（M1 第二步）
static volatile int g_plt_calls;
static int g_plt_log_n;          // 通用 PLT 记名行数（截断计数，见 plt_stub 末尾）
#define DM_PLT_LOG_CAP 20000     // init 段约一千行 + 充分余量；循环段靠它挡住
// 宿主侧"真服务"表：启动链真的需要返回值的几个（不是一律返 0）。
// 依据：崩点反汇编显示 AllocateMemoryBlock 的返回值被当对象指针使用。
static char g_host_arg0[] = "main";
static char *g_host_argv[2];
// nn::os::GetCurrentThread 要返回"当前线程"的 ThreadType*。我们还没有真实的 ThreadType，
// 先给一个宿主侧的**影子对象**：清零、够大，让游戏"拿它当结构体读"这一步不撞墙。
// 依据（真机实证）：新崩点 0xa1d94c 就是拿返回值去读 [+0x58]（返回值若为 0 ⇒ 读 0x58 崩）。
// 字段语义待看清游戏怎么用之后再补；单线程深跑阶段一个影子对象够用（多线程时须按线程分开）。
#define DM_THREAD_SHADOW_QWORDS 0x40            // 0x200 字节，覆盖到 [+0x58] 及其后
static uint64_t g_thread_shadow[DM_THREAD_SHADOW_QWORDS];
// nn::os::GetTlsValue 的影子对象：游戏把返回值当"线程局部存储"用，其中 **[+0xb0] 被 0xa0e2f8
// 当作"当前分配器对象"取出**。真机实证（2026-10-17）：不实现它时 GetTlsValue 返回 0 ⇒
// 0xa0e2f8 走兜底读 [obj+8]，而那个"自指"值此时已被清成 0 ⇒ 整条 sead 单例查找返回 0。
// 这里让 [+0xb0] 指向注入的分配器对象（在 k_inject_heap 里接线），使这条分支拿到非 0。
// 注：槽号目前只有 0（注入的上下文对象 [0x88] 为 0），故一个影子对象够用。
static uint64_t g_tls_shadow[0x40];             // 0x200 字节，覆盖到 [+0xb0]
// np211g：分配器对象地址的"转发指针"。g_alloc_obj 定义在 5300 行（本函数之后），
//   而 k_sdk_tls_shadow_install 要给影子块 +0xb0 填它 ⇒ 用指针转发（k_inject_heap
//   在 nnMain 前设值，工人线程创建时必已就绪）。
static void *g_alloc_obj_ptr;

// ── 真实线程（2026-10-18）：之前 CreateThread/StartThread 完全没实现 ⇒ 游戏所有"线程"在
//    单线程上内联跑 ⇒ **子系统工作线程压根不存在 ⇒ 它们的任务永远不产生 ⇒ sead 主循环空转**
//    （真机实证：28811 行日志只有两个 tid；主循环回调 633 万次却无事可做）。
//    这里用 pthread 真建线程：ThreadType* ↔ 线程槽 映射表，StartThread 用蹦床调 entry(arg)。
//    ⚠ 副作用：真线程后，互斥量必须是真的、TLS 必须按线程分开、日志必须串行化。
#define DM_THREAD_MAX 16
// np180：线程体返回记账要用（定义在后面）
static void note_append(const char *name, const char *line);
typedef struct {
    uint64_t game;              // 游戏侧 ThreadType*（0 = 空槽）
    pthread_t tid;              // 宿主线程
    void (*entry)(void *);
    void *arg;
    volatile int started;
    int prio, core_mask;
    char name[24];              // SetThreadName 存的线程名（pthread_setname_np 用）
    uint64_t id;                // 游戏侧线程号（GetThreadId 用；稳定、不复用）
} dm_thread_t;
static dm_thread_t g_thr[DM_THREAD_MAX];
static pthread_mutex_t g_thr_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_thr_id_next = 1;
static int thr_find_by_game(uint64_t game) {
    for (int i = 0; i < DM_THREAD_MAX; i++) if (g_thr[i].game == game) return i;
    return -1;
}
static int thr_find_by_self(void) {
    pthread_t me = pthread_self();
    for (int i = 0; i < DM_THREAD_MAX; i++)
        if (g_thr[i].game && g_thr[i].started && pthread_equal(g_thr[i].tid, me)) return i;
    return -1;
}
// np211：**轮投全部游戏线程**（原来只投主线程）。动机：12 个工人线程 R 态 cpu=0，
//   静态追查全堵死（入口 4 个 PLT 全非阻塞、0xa26f80 是 popcount），必须拿 PC。
//   ⚠ 别投自己——样本会落在 musl 的 `pthread_kill+0xa4`（svc 返回点），np178 首轮
//   4096/4096 就这么废掉的。主线程也由这里投（帧线程 ≠ 主线程，不冲突）。
static void smp_kick(void) {
    if (!g_smp_started || !g_main_pth) return;
    uint32_t me = (uint32_t)(uintptr_t)pthread_self();
    __atomic_fetch_add(&g_smp_kicks, 1u, __ATOMIC_RELAXED);
    pthread_kill(g_main_pth, SIGPROF);
    for (int i = 0; i < DM_THREAD_MAX; i++) {
        if (!g_thr[i].started) continue;
        uint32_t t = (uint32_t)(uintptr_t)g_thr[i].tid;
        if (t == me) continue;
        pthread_kill(g_thr[i].tid, SIGPROF);
    }
}
// ───────── np135：sdk 每线程 TLS 影子槽（M1 第三步）─────────
// 背景：ohos 目标下，sdk 的线程槽跳床统一走 musl tsd（mrs tpidr_el0 → *(tp−272) → +键号×8），
//   由探针按线程写槽。此前 8 个槽全填哨兵值——KGATE（门表）= k_gate_stub 是对的，但
//   K18/KTLR 必须是"真影子"，否则 sdk 的 TSD 访问/互斥量名牌一执行就撞（np134 崩因）。
// 尺寸依据（2026-10-20 离线实证，反汇编 sdk 全部 4,973 个站点）：
//   K18（x18 虚拟寄存器及物化指针的落点）：369 处 [x18+disp] 最大 0x41e0、add/sub 物化
//     偏移最大 0xc80、存在 −0x10 负偏移 ⇒ 块 0x4500，槽值 = 块 + 0x10（负偏移也有落点）。
//   KTLR（TLR 指针）：tpidrro 站点后续偏移集合 {0x1d0,0x1d8,0x1f0,0x1f8} ⇒ 区 0x200；
//     其中 [0x1f8] = NN 线程对象（自旋锁经它读 +0x1b8 的 u32 属主名牌）⇒ 对象 0x1c0。
//   KTLS（mrs/msr tpidr_el0，2 读 3 写）：语义待定，维持哨兵观察。
// 名牌唯一性：全局自增 u32（0 保留作"无主"）。探针场景线程少，影子块不释放（泄漏可接受）。
static uint32_t g_sdk_tag_seq;
static void k_sdk_tls_shadow_install(void) {
    uint64_t tp = 0; __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp));
    uint64_t tsd = *(uint64_t *)(tp - 272);
    uint8_t *k18 = (uint8_t *)calloc(1, 0x4500);      // TSD 影子（前 0x10 留给负偏移）
    uint8_t *tlr = (uint8_t *)calloc(1, 0x200);       // TLR 影子区
    uint8_t *nn  = (uint8_t *)calloc(1, 0x1c0);       // NN 线程对象（+0x1b8 = u32 名牌）
    if (!k18 || !tlr || !nn) { klog("[sdk-tls] ✗ 影子块分配失败\n"); return; }
    uint32_t tag = ++g_sdk_tag_seq;
    *(uint32_t *)(nn + 0x1b8)  = tag;
    *(uint64_t *)(tlr + 0x1f8) = (uint64_t)(uintptr_t)nn;
    *(uint64_t *)(tsd + (uint64_t)g_s_key[DM_IMG_SDK_KEY_K18_IDX]  * 8) = (uint64_t)(uintptr_t)(k18 + 0x10);
    *(uint64_t *)(tsd + (uint64_t)g_s_key[DM_IMG_SDK_KEY_KTLR_IDX] * 8) = (uint64_t)(uintptr_t)tlr;
    // np211g（np211f 定谳的修复）：游戏经 `0xa0e2f8` 读「GetTlsValue(键) 返回块 +0xb0」
    //   当"当前线程分配器"（0xa0913c 的 cbz 门：为空就放弃建子对象 ⇒ X+0x128 永不挂接
    //   ⇒ F_wait 退出条件①永不满足 ⇒ 内容门打不开）。K18 键的块 = k18+0x10 ⇒ +0xb0 =
    //   k18+0xc0；KTLR 键的块 = tlr ⇒ +0xb0 = tlr+0xb0。**两处都填**（键号由 ctx+0x88
    //   决定，静态难定，成本为零）。此前只填了静态 g_tls_shadow[0xb0]——那数组从没接到
    //   任何 TSD 槽上，等于没填。
    if (g_alloc_obj_ptr) {
        *(uint64_t *)(k18 + 0xc0) = (uint64_t)(uintptr_t)g_alloc_obj_ptr;   // +0xb0（np211f 门）
        *(uint64_t *)(k18 + 0x70) = (uint64_t)(uintptr_t)g_alloc_obj_ptr;   // +0x60（np164 同一模式）
        *(uint64_t *)(tlr + 0xb0) = (uint64_t)(uintptr_t)g_alloc_obj_ptr;
    }
    klog("[sdk-tls] 影子槽已装：K18=%p（块 %p） KTLR=%p NN=%p 名牌=%u\n",
         (void *)(k18 + 0x10), (void *)k18, (void *)tlr, (void *)nn, (unsigned)tag);
}

static void *thr_trampoline(void *p) {
    dm_thread_t *t = (dm_thread_t *)p;
    // np133：给新线程也填 **sdk 的键页槽**（sdk 的门表指针在它自己的键号上）。
    //   不填的话，sdk 代码一旦执行到 svc 门，就会读到空门表（真机实证：si_addr=0x158、x8=0x158）。
    if (g_n_sdk_os) {
        uint64_t tp = 0; __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp));
        uint64_t tsd = *(uint64_t *)(tp - 272);
        const void *sv[8] = {
            (void *)(uintptr_t)0x1111111111111111ull, (void *)(uintptr_t)0x2222222222222222ull,
            (void *)(uintptr_t)0x3333333333333333ull, (void *)k_gate_stub,
            (void *)(uintptr_t)0x4400000000000004ull, (void *)(uintptr_t)0x4400000000000005ull,
            (void *)(uintptr_t)0x4400000000000006ull, (void *)(uintptr_t)0x4400000000000007ull,
        };
        for (int i = 0; i < 8; i++) *(uint64_t *)(tsd + (uint64_t)g_s_key[i] * 8) = (uint64_t)sv[i];
        // np135：K18/KTLR 换真影子（TSD 影子块 / TLR 影子区 + NN 线程对象 + 唯一名牌）
        k_sdk_tls_shadow_install();
    }
    t->entry(t->arg);                         // 游戏的线程体（void 返回）
    uint64_t rv = 0;
    // ⭐ np180：**线程体返回**要记账。死锁形态可能是"主线程等某个线程/任务完成，而那个线程体
    //   一启动就返回了"（例如它拿到 0 参数直接 exit）。原先没有任何地方记这件事。
    {
        char line[160];
        snprintf(line, sizeof line, "threxit name=%s entry=img+%#llx arg=%#llx rv=%#llx\n",
                 t->name[0] ? t->name : "?",
                 (unsigned long long)((uintptr_t)t->entry - (uintptr_t)g_k_m.base),
                 (unsigned long long)(uintptr_t)t->arg, (unsigned long long)rv);
        note_append("dm_gate.txt", line);
    }
    return NULL;
}

// ── 线程局部存储：**不做真槽**（2026-10-18 真机实证回退，见 plt_stub 里 TLS 三行的注释）──
// 旧语义：槽号恒 0、Set 忽略、Get 恒返影子对象。真线程共享这个影子可接受
// （它只被当作"当前上下文对象"用，[+0xb0] 是全局分配器）。

// ── nn::os 互斥量：**真 pthread_mutex，且必须递归** ──
// ⚠ 2026-10-18 真机实证（0xa01924 二次加锁挂死）：sead 的"锁"（0xa1d56c/0xa1d57c）其实就是
//   `add x0,#0x20; b nn::os::LockMutex/UnlockMutex`（PLT 槽 0x22636f0/0x22636f8）。
//   **nn::os::Mutex 是递归语义**（同一线程可重复加锁、靠计数解锁），于是同一线程先由
//   mgr 构造器加锁、再由池注册函数加锁是正常路径。非递归 pthread_mutex 在这里必死锁。
// ⚠ np128（真机实证的重大教训）：这张表原来是 **64 格**，而游戏在启动阶段就初始化了 **297 个互斥量**。
//   表满后 mtx_get 退化成"所有新锁都返回 &g_mtx_impl[0]"⇒ 两个"不同"的锁其实是同一把：
//   同一个线程先后锁两把（都很常见）就**自死锁**，而且是 CPU 0%、所有线程 state=S 的死法，
//   看起来像"卡在某个未知地方"，极难定位。现在放到 2048 格，并在满表时打红日志（绝不静默退化）。
#define DM_MUTEX_MAX 2048
static void *g_mtx_key[DM_MUTEX_MAX];
static pthread_mutex_t g_mtx_impl[DM_MUTEX_MAX];
static pthread_mutex_t g_mtx_reg = PTHREAD_MUTEX_INITIALIZER;   // 登记表自身的锁
static long g_mtx_owner[DM_MUTEX_MAX];   // np160：持锁者 tid（0=无主；进/出锁登记）。
// ★ 为什么加：np160 反汇编链坐实——主线程每次都停在 `nn::os::LockMutex`（PLT#92，
//   GOT 0x22636f0；调用点 0xa0e140 → add x0,#0x20 → b PLT）。旧打点只在"锁等到之后"
//   才打（ms>=5）⇒ 永久等待无声。新版 = 成对进入/离开 + 持锁者 tid：
//   **有进入无离开 = 卡在这把锁**；owner 字段 + 该 tid 的最后一条日志 = 元凶现形
//   （持锁不释放 / 持锁者自己又卡在别处 / 持锁线程已死）。
static pthread_mutexattr_t g_mtx_attr;
static pthread_once_t g_mtx_once = PTHREAD_ONCE_INIT;
static void mtx_attr_init(void) {
    pthread_mutexattr_init(&g_mtx_attr);
    pthread_mutexattr_settype(&g_mtx_attr, PTHREAD_MUTEX_RECURSIVE);
}
static pthread_mutex_t *mtx_get(void *m) {
    pthread_once(&g_mtx_once, mtx_attr_init);
    pthread_mutex_lock(&g_mtx_reg);
    for (int i = 0; i < DM_MUTEX_MAX; i++) if (g_mtx_key[i] == m) { pthread_mutex_unlock(&g_mtx_reg); return &g_mtx_impl[i]; }
    for (int i = 0; i < DM_MUTEX_MAX; i++) if (!g_mtx_key[i]) {
        g_mtx_key[i] = m;
        pthread_mutex_init(&g_mtx_impl[i], &g_mtx_attr);
        pthread_mutex_unlock(&g_mtx_reg);
        return &g_mtx_impl[i];
    }
    pthread_mutex_unlock(&g_mtx_reg);
    // np128：**绝不静默退化**。表满 = 多把锁变成同一把 = 迟早自死锁，必须立刻看得见。
    static int n_full;
    if (++n_full <= 8)
        klog("【红】互斥量表满（%d 格）！新锁 %p 退化为 impl[0] ⇒ 会自死锁（调大 DM_MUTEX_MAX）\n",
             DM_MUTEX_MAX, m);
    return &g_mtx_impl[0];      // 表满：退化（不该发生）
}

// np219：把"当前被持有的锁"连同**持有者是否还活着**一起打出来。
// 判读：owner 线程已不存在（/proc/self/task/<tid> 打不开）⇒ 锁永久泄漏 = 死锁源。
static void mtx_dump_held(void) {
    int shown = 0, dead = 0;
    for (int i = 0; i < DM_MUTEX_MAX && shown < 40; i++) {
        long ow = g_mtx_owner[i];
        if (!ow) continue;
        char p[128];
        snprintf(p, sizeof p, "/proc/self/task/%ld/stat", ow);
        FILE *f = fopen(p, "r");
        int alive = (f != NULL);
        if (f) fclose(f);
        if (!alive) dead++;
        klog("[持锁] idx=%d mutex=%p owner=%ld %s\n", i, g_mtx_key[i], ow,
             alive ? "（线程仍在）" : "★线程已不存在=永久泄漏");
        shown++;
    }
    if (!shown) klog("[持锁] 无任何被持有的互斥量\n");
    else klog("[持锁] 合计 %d 把被持有，其中 %d 把的持有者已死\n", shown, dead);
}

// np219：某线程名下还挂着几把锁（构造器崩了被 setjmp 兜走时调用——若 >0，
// 这些锁就永久泄漏，后面的线程全卡在上面）。返回把数并打印明细。
static int mtx_report_owned(long tid, const char *tag) {
    int n = 0;
    if (tid <= 0) return 0;
    for (int i = 0; i < DM_MUTEX_MAX; i++) {
        if (g_mtx_owner[i] != tid) continue;
        if (n < 8) klog("[持锁泄漏?] %s tid=%ld 仍持有 idx=%d mutex=%p\n", tag, tid, i, g_mtx_key[i]);
        n++;
    }
    if (n) klog("[持锁泄漏?] %s tid=%ld 崩溃时共持有 %d 把锁（这些锁将永久无人释放）\n", tag, tid, n);
    return n;
}

// np160：取互斥量在表中的下标（owner 登记用）。benign race： worst case 返回 -1 不登记。
static int mtx_idx(void *m) {
    for (int i = 0; i < DM_MUTEX_MAX; i++) if (g_mtx_key[i] == m) return i;
    return -1;
}
// 成对决策油门：前 100 次全记，之后每 50 次记一次（进/出用同一次决策 ⇒ 配对不破）
static int g_n_mtxlog;
static int mtx_log_ok(void) {
    int n = ++g_n_mtxlog;
    return n <= 100 || (n % 50) == 0;
}

// ── 消息队列（2026-10-18）──
// Prepare Thread 用 ReceiveMessageQueue 收命令。之前存根秒返 0 但不写消息 ⇒ 游戏可能读到
// 未初始化消息，且线程只能忙等（真机日志实证：Receive+Yield 空转 11 轮）。真实现：
// 游戏提供**值缓冲区**（InitializeMessageQueue 的 buf/count），我们维护头尾 + 条件变量，
// 支持阻塞收/阻塞发（Send/Receive）与非阻塞（TrySend/TryReceive，bool 返回：1=成功）。
#define DM_MQ_MAX 8
typedef struct {
    uint64_t *buf;                 // 游戏侧值缓冲区
    size_t cap, head, tail, n;
    pthread_mutex_t mu;
    pthread_cond_t cv_notempty;    // 收方等"有消息"
    pthread_cond_t cv_notfull;     // 发方等"有空位"
    // ⭐ np180：**配对记录**（谁在等 / 谁发过），用来抓"本该发出却失踪的 Send"。
    //   死的形态：主等工人完成、工人等消息、消息没人发 ⇒ 必须把"谁等哪个队列"与
    //   "谁给哪个队列发过"并排看，才能判断是"没发"还是"发到别的队列"。
    uint32_t wait_n, send_n;       // 等待次数 / 发送次数
    uint32_t wait_tid, send_tid;   // 最后等待者 / 最后发送者
    uint64_t wait_lr, send_lr;     // 对应客体调用点（映像内偏移或外部地址）
    uint64_t send_val;             // 最后发送的值
} dm_mq_t;
static void *g_mq_key[DM_MQ_MAX];
static dm_mq_t g_mq[DM_MQ_MAX];
static pthread_mutex_t g_mq_reg = PTHREAD_MUTEX_INITIALIZER;
static dm_mq_t *mq_get(void *q) {
    pthread_mutex_lock(&g_mq_reg);
    for (int i = 0; i < DM_MQ_MAX; i++) if (g_mq_key[i] == q) { pthread_mutex_unlock(&g_mq_reg); return &g_mq[i]; }
    for (int i = 0; i < DM_MQ_MAX; i++) if (!g_mq_key[i]) {
        g_mq_key[i] = q;
        memset(&g_mq[i], 0, sizeof g_mq[i]);
        pthread_mutex_init(&g_mq[i].mu, NULL);
        // np156：队列 cond 回到**默认 REALTIME**。曾改成 MONOTONIC（np154）治"超时永不
        // 触发"，但 np155/np156 实测唤醒只成功过一次 ⇒ OHOS musl 对 MONOTONIC 绝对
        // 超时支持不全（futex bitset）。REALTIME+REALTIME 是处处可用的经典组合，
        // 测试窗口内无跳时风险。
        pthread_cond_init(&g_mq[i].cv_notempty, NULL);
        pthread_cond_init(&g_mq[i].cv_notfull, NULL);
        pthread_mutex_unlock(&g_mq_reg);
        return &g_mq[i];
    }
    pthread_mutex_unlock(&g_mq_reg);
    return NULL;
}

// ⭐ np180：队列配对记录落盘（绕过日志预算）。判读三句话：
//   · 某队列 wait_n 很大而 send_n = 0 ⇒ **没人发**（就是"在等消息、消息不会来"）；
//   · send_n > 0 但 wait 的队列不是它  ⇒ **发到别的队列去了**（绑定/偏移错）；
//   · 两边都有但值不匹配 ⇒ 语义错（发的是控制字，收方当任务指针）。
static void mq_pair_dump(const char *tag) {
    if (!g_files_dir[0]) return;
    char p[600], buf[1200];
    snprintf(p, sizeof p, "%s/dm_gate.txt", g_files_dir);
    int fd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;
    uint64_t b = (uint64_t)(uintptr_t)g_k_m.base;
    int len = snprintf(buf, sizeof buf, "mq[%s]\n", tag);
    for (int i = 0; i < DM_MQ_MAX; i++) {
        if (!g_mq_key[i]) continue;
        if (len > (int)sizeof buf - 260) break;
        len += snprintf(buf + len, sizeof buf - (size_t)len,
                        "  #%d key=%#llx cap=%zu n=%zu wait=%u(tid=%#x lr=%#llx) send=%u(tid=%#x lr=%#llx val=%#llx)\n",
                        i, (unsigned long long)(uintptr_t)g_mq_key[i], g_mq[i].cap, g_mq[i].n,
                        g_mq[i].wait_n, g_mq[i].wait_tid,
                        (unsigned long long)(g_mq[i].wait_lr >= b && g_mq[i].wait_lr < b + DM_IMG_MAIN_END
                                             ? g_mq[i].wait_lr - b : g_mq[i].wait_lr),
                        g_mq[i].send_n, g_mq[i].send_tid,
                        (unsigned long long)(g_mq[i].send_lr >= b && g_mq[i].send_lr < b + DM_IMG_MAIN_END
                                             ? g_mq[i].send_lr - b : g_mq[i].send_lr),
                        (unsigned long long)g_mq[i].send_val);
    }
    (void)!write(fd, buf, (size_t)len);
    close(fd);
}

// 前置声明：TLS 的每线程影子要把 [+0xb0] 填成注入的分配器对象（定义在后面，C 的试探性定义合法）
static uint64_t g_alloc_obj[2];

static uint64_t *g_fake_heap;       // np103b 前置声明（定义在后）
// np125：全量调用时间线（环缓冲，见后面 ktrace_rec 的注释）。前置声明——plt_stub 比它早。
static void ktrace_plt(int idx, uint64_t a0, uint64_t a1);
// np175：绕过日志预算的专用写（定义在后面；plt_stub 要用它记"建-信号循环"的调用点）
static void note_append(const char *name, const char *line);

// np213：StandardAllocator 的外部状态表（this → 池），plt_stub 用（C 无嵌套函数）
static struct { uint64_t self, base, size, used; } g_sa[8];
static int g_sa_n;
static int sa_lookup(uint64_t self, int create) {
    for (int i = 0; i < g_sa_n; i++) if (g_sa[i].self == self) return i;
    if (create && g_sa_n < 8) {
        g_sa[g_sa_n].self = self; g_sa[g_sa_n].base = 0;
        g_sa[g_sa_n].size = 0; g_sa[g_sa_n].used = 0;
        return g_sa_n++;
    }
    return -1;
}

// np220：**每线程"最后进入的 HLE 函数"**。动机：np218/219 实测进程 CPU 仅 0.4 秒、
//   三个游戏线程全阻塞在 futex、且 HLE 互斥量表是空的 ⇒ 阻塞发生在别处。要判"谁卡在哪个
//   等待原语上"，就得知道每个线程最后一次进的是哪个导入函数（WaitEvent / ReceiveMessageQueue /
//   LockMutex / …）。plt_stub 每次调用记一条（TLS 缓存槽号，单次 store + 一次 vDSO 取时），
//   线程快照里连同"距今多少毫秒"一起打出来：**时间戳很旧 = 不是卡在这个 HLE 里**。
#define HLE_CUR_MAX 32
static struct { long tid; const char *nm; uint64_t ms; } g_hle_cur[HLE_CUR_MAX];
static __thread int g_hle_slot = -1;
static uint64_t hle_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}
static int hle_slot_claim(void) {
    long me = (long)gettid();
    for (int i = 0; i < HLE_CUR_MAX; i++) if (g_hle_cur[i].tid == me) return i;
    for (int i = 0; i < HLE_CUR_MAX; i++) if (!g_hle_cur[i].tid) { g_hle_cur[i].tid = me; return i; }
    return -1;
}
static void hle_cur_dump(void) {
    uint64_t now = hle_now_ms();
    int shown = 0;
    for (int i = 0; i < HLE_CUR_MAX; i++) {
        if (!g_hle_cur[i].tid) continue;
        const char *n = g_hle_cur[i].nm ? g_hle_cur[i].nm : "(无)";
        // 只留 mangled 名的尾部（可读部分），避免一行太长
        const char *p = n;
        if (n[0] == '_' && n[1] == 'Z') { const char *q = strrchr(n, '_'); if (q && q[1]) p = q + 1; }
        klog("[最近HLE] tid=%ld %s（距今 %llu ms）\n", g_hle_cur[i].tid, p,
             (unsigned long long)(now > g_hle_cur[i].ms ? now - g_hle_cur[i].ms : 0));
        shown++;
    }
    if (!shown) klog("[最近HLE] 无记录\n");
}

static uint64_t plt_stub(void) {
    uint64_t x16v, a0, a1, a2, a3;
    __asm__ volatile("mov %0, x16" : "=r"(x16v));
    __asm__ volatile("mov %0, x0" : "=r"(a0));
    __asm__ volatile("mov %0, x1" : "=r"(a1));
    __asm__ volatile("mov %0, x2" : "=r"(a2));
    __asm__ volatile("mov %0, x3" : "=r"(a3));   // np96：第 4 参（VSNPrintf 的 va_list 在这里）
    // np175：**必须在第一次 strcmp 之前**取 x30（= 客体调用点返回地址）；任何调用都会把它冲掉。
    uint64_t lr0; __asm__ volatile("mov %0, x30" : "=r"(lr0));
    const char *nm = NULL;
    int nidx = -1;
    for (int i = 0; i < g_nplt; i++) if (g_plt[i].slot == x16v) { nm = g_plt[i].name; nidx = i; break; }
    g_plt_calls++;
    // np175：同上——空闲环的这两个导入（参数全 0）不记账，免得把环冲掉
    if (!(a0 == 0 && a1 == 0 && nm &&
          (!strcmp(nm, "_ZN2nn2os17ConvertToTimeSpanENS0_4TickE") ||
           !strcmp(nm, "_ZN2nn2os11SleepThreadENS_8TimeSpanE"))))
        ktrace_plt(nidx, a0, a1);   // np125：先记账（无日志开销），分析在离线做
    if (nm) {
        // np220：登记"本线程最后进入的 HLE 函数"（线程快照判读阻塞点用）
        if (g_hle_slot < 0) g_hle_slot = hle_slot_claim();
        if (g_hle_slot >= 0) { g_hle_cur[g_hle_slot].nm = nm; g_hle_cur[g_hle_slot].ms = hle_now_ms(); }
        // ⭐ np175：定位"建-信号无界循环"的调用点。真机轨迹（过滤空转后）显示主线程末段
        //   只剩 `InitializeLightEvent → SignalLightEvent`，事件对象逐次落在**栈上新地址**
        //   （0x5fe21fbb88 → … → 0x5fe21fd6a0），注册表一路涨到 16384+ ⇒ 某个条件永远不满足。
        //   这里记下客体调用点（lr = `bl PLT` 的下一句），离线反汇编即知循环在哪个函数。
        //   写 dm_gate.txt 绕过日志预算，只记前 12 次。
        if (!strcmp(nm, "_ZN2nn2os20InitializeLightEventEPNS0_14LightEventTypeEbNS0_14EventClearModeE")) {
            static int n;
            if (n < 12) {
                char line[96];
                uint64_t b = (uint64_t)(uintptr_t)g_k_m.base;
                snprintf(line, sizeof line, "lightev#%d lr=img+%#llx ev=%#llx\n", ++n,
                         (unsigned long long)(lr0 - b), (unsigned long long)a0);
                note_append("dm_gate.txt", line);
            }
        }
        // ⭐ np179：**空转循环的调用点**。SIGPROF 采样在本进程里始终拿不到样本（疑被掩码），
        //   改用最直接的等价物：这两条导入是空闲环的"心跳"，把它们的**客体返回地址**
        //   按首次出现各记一条 ⇒ 直接得到"哪个函数在转这个环"。
        if (!strcmp(nm, "_ZN2nn2os17ConvertToTimeSpanENS0_4TickE") ||
            !strcmp(nm, "_ZN2nn2os11SleepThreadENS_8TimeSpanE")) {
            static uint64_t seen[8]; static int ns;
            uint64_t off = lr0 - (uint64_t)(uintptr_t)g_k_m.base;
            int dup = 0;
            for (int i = 0; i < ns; i++) if (seen[i] == off) { dup = 1; break; }
            if (!dup && ns < 8) {
                char line[128];
                seen[ns++] = off;
                snprintf(line, sizeof line, "loopcall#%d src=%s img+%#llx\n", ns,
                         (!strcmp(nm, "_ZN2nn2os17ConvertToTimeSpanENS0_4TickE")) ? "Convert" : "Sleep",
                         (unsigned long long)off);
                note_append("dm_gate.txt", line);
            }
        }
        // ⭐ np176：`WaitSystemEvent` **没有映射**（落到通用存根=立刻返回 ⇒ 游戏只能自旋）。
        //   主库只导入了 Wait/TryWait/TimedWait/DestroySystemEvent，**没有** Initialize/Signal
        //   ⇒ 这些对象是 **sdk 模块自己造的**（sdk 的 nn::os 自解析），我们看不到它的信号。
        //   本轮先**只观察不改行为**：把对象前 0x40 字节 dump 出来，用于判定布局与当前是否已置位。
        if (!strcmp(nm, "_ZN2nn2os15WaitSystemEventEPNS0_15SystemEventTypeE")) {
            static int n;
            if (n < 2 && a0 >= 0x1000) {
                char line[400];
                const uint64_t *o = (const uint64_t *)(uintptr_t)a0;
                int len = snprintf(line, sizeof line, "sysev#%d obj=%#llx", ++n,
                                   (unsigned long long)a0);
                for (int i = 0; i < 8; i++)
                    len += snprintf(line + len, sizeof line - (size_t)len, " [%02x]=%#llx", i * 8,
                                    (unsigned long long)o[i]);
                snprintf(line + len, sizeof line - (size_t)len, "\n");
                note_append("dm_gate.txt", line);
            }
        }
        // nn::os::AllocateMemoryBlock(size_t* out_size, size_t size) —— 真给内存
        if (!strcmp(nm, "_ZN2nn2os19AllocateMemoryBlockEPmm")) {
            // 必须清零（calloc）：游戏会把这块内存当自己的内存池，池管理器的空闲链头若读到
            // 未初始化垃圾，第一次"切对象"就会得到垃圾指针（本轮插桩实证：this=0xC0600000=请求大小）。
            void *p = calloc(1, (size_t)a1);
            // 证据：插桩读到游戏把"输出参数的值"当内存块基址用（this=0xC0600000=请求大小）
            // ⇒ 该输出参数要的是**地址**，不是大小。写地址。
            if (a0) *(uint64_t *)a0 = (uint64_t)(uintptr_t)p;
            klog("[服务] AllocateMemoryBlock(大小 %llu) → %p（写回地址）\n", (unsigned long long)a1, p);
            return (uint64_t)(uintptr_t)p;
        }
        // nn::os::GetHostArgc / GetHostArgv —— 给一对合理的启动参数
        if (!strcmp(nm, "_ZN2nn2os11GetHostArgcEv")) { klog("[服务] GetHostArgc → 1\n"); return 1; }
        if (!strcmp(nm, "_ZN2nn2os11GetHostArgvEv")) {
            g_host_argv[0] = g_host_arg0; g_host_argv[1] = NULL;
            {   // np94：谁在问启动参数？（x30 = 调用者返回地址）
                uint64_t lr; __asm__ volatile("mov %0, x30" : "=r"(lr));
                uint64_t b2 = (uint64_t)(uintptr_t)g_k_m.base;
                int in = lr >= b2 && lr < b2 + DM_IMG_MAIN_END;
                klog("[服务] GetHostArgv → argv[0]=main（调用者=%s%#llx）\n", in ? "+" : "(外)",
                     (unsigned long long)(in ? lr - b2 : lr));
            }
            return (uint64_t)(uintptr_t)g_host_argv;
        }
        // ═══ np213：nxnative 对齐的 SDK 服务桩（交叉比对出的 25 个精确缺口）═══
        // 依据：nxnative（同源分支，MIT）docs/agents/03-bringup.md 第 75 行坐实——
        //   BotW 1.9.0（SDK 20.5.22）的 nfp 一族在 sdk+0x3f2a78 读空指针崩。
        // 交叉方法：我们 expect_scan 的"返 0 走失败分支"高风险表 ∩ nxnative 真实现
        //   = 25 个符号；这 25 个在我们 plt_stub 里**一个都没有**（全落通用存根）。
        // 判读要点：多数是 Result 型（0 = RES_OK = 成功）——返 0 本身没错，
        //   错在**不填输出参数**：ListDevices 不写设备、GetDebugPadState 不清零、
        //   OpenAudioRenderer 不写 handle、GetState 该返 1、WorkBufferSize 该返真大小。
        //   游戏读到垃圾 ⇒ 走"无设备/未初始化"分支 ⇒ 初始化链断在看不见的地方。
        // ── nfp（amiibo）：宿主无 NFC 读卡器，报"一个设备但永远找不到 tag"（与真机无 amiibo 同）──
        if (!strcmp(nm, "_ZN2nn3nfp10InitializeEv")) { klog("[nfp] Initialize：一个 NFC 设备，找不到 tag\n"); return 0; }
        if (!strcmp(nm, "_ZN2nn3nfp8GetStateEv")) return 1;                 // 库状态 1 = 已初始化
        if (!strcmp(nm, "_ZN2nn3nfp11ListDevicesEPNS0_12DeviceHandleEPii")) {
            // (out, *n, max)：写一个设备（8 字节 {u32 id, u8[4] reserved}），*n = 1
            if (a0 && a2 >= 1) { memset((void *)(uintptr_t)a0, 0, 8);
                *(uint32_t *)(uintptr_t)a0 = 0x4e465030u; /* 'NFP0' */ }
            if (a1) *(int *)(uintptr_t)a1 = (a2 >= 1) ? 1 : 0;
            klog("[nfp] ListDevices → 1 个设备（max=%lld）\n", (long long)a2);
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn3nfp9GetNpadIdEPjRKNS0_12DeviceHandleE")) {
            if (a0) *(uint32_t *)(uintptr_t)a0 = 0;                          // 手柄 0（当前连接）
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn3nfp14GetDeviceStateERKNS0_12DeviceHandleE")) return 0;  // 0=已初始化 1=搜索中
        if (!strcmp(nm, "_ZN2nn3nfp14StartDetectionERKNS0_12DeviceHandleE")) return 0;
        if (!strcmp(nm, "_ZN2nn3nfp13StopDetectionERKNS0_12DeviceHandleE")) return 0;
        if (!strcmp(nm, "_ZN2nn3nfp19AttachActivateEventEPNS_2os15SystemEventTypeERKNS0_12DeviceHandleE") ||
            !strcmp(nm, "_ZN2nn3nfp21AttachDeactivateEventEPNS_2os15SystemEventTypeERKNS0_12DeviceHandleE") ||
            !strcmp(nm, "_ZN2nn3nfp29AttachAvailabilityChangeEventEPNS_2os15SystemEventTypeE")) {
            // 事件"已附着但永不触发"：BotW 是轮询 GetDeviceState，不阻塞等它（nxnative 同做法）
            static int n_ev; if (++n_ev <= 3) klog("[nfp] 附着事件 %p（永不触发，游戏轮询状态）\n", (void *)(uintptr_t)a0);
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn3nfp8FinalizeEv")) return 0;
        // 需要 tag 的操作：一律"找不到 tag"（Result = 115 | 97<<9）
        if (!strncmp(nm, "_ZN2nn3nfp", 10)) {
            static int n_nt; if (++n_nt <= 4) klog("[nfp] %s → 无 tag\n", nm + 10);
            return (uint64_t)(115u | (97u << 9));
        }
        // ── hid：调试手柄状态必须**清零**（每项 32 字节）──
        if (!strcmp(nm, "_ZN2nn3hid16GetDebugPadStateEPNS0_13DebugPadStateE")) {
            if (a0) memset((void *)(uintptr_t)a0, 0, 32);
            static int n_dp; if (++n_dp <= 3) klog("[hid] GetDebugPadState → 未连接（32 字节清零）\n");
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn3hid17GetDebugPadStatesEPNS0_13DebugPadStateEi")) {
            if (a0 && a1 > 0) memset((void *)(uintptr_t)a0, 0, (size_t)a1 * 32);
            static int n_dps; if (++n_dps <= 3) klog("[hid] GetDebugPadStates(%lld) → 全未连接\n", (long long)a1);
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn3hid21ShowControllerSupportERKNS0_20ControllerSupportArgE")) return 0;
        // ── audio：handle / 工作缓冲大小 / 通道数 必须给真值 ──
        if (!strcmp(nm, "_ZN2nn5audio17OpenAudioRendererEPNS0_19AudioRendererHandleEPNS_2os11SystemEventERKNS0_22AudioRendererParameterEPvm")) {
            static uint64_t s_ar_handle = 0x41554430ull;   // 'AUD0' 假句柄
            if (a0) *(uint64_t *)(uintptr_t)a0 = s_ar_handle;
            klog("[audio] OpenAudioRenderer → 假句柄 %#llx（参数 %p）\n",
                 (unsigned long long)s_ar_handle, (void *)(uintptr_t)a2);
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn5audio30GetAudioRendererWorkBufferSizeERKNS0_22AudioRendererParameterE") ||
            !strcmp(nm, "_ZN2nn5audio36GetAudioRendererConfigWorkBufferSizeERKNS0_22AudioRendererParameterE")) {
            static int n_wb; if (++n_wb <= 3) klog("[audio] WorkBufferSize → 0x40000\n");
            return 0x40000;                                  // 256KB：够 Wwise/sead 起混音器
        }
        if (!strcmp(nm, "_ZN2nn5audio21GetActiveChannelCountEv")) return 2;
        if (!strcmp(nm, "_ZN2nn5audio36AcquireAudioDeviceSwitchNotificationEPNS_2os11SystemEventE")) return 0;
        // ── diag：符号名/回溯必须写缓冲（不写 ⇒ 游戏读到垃圾当函数名）──
        if (!strcmp(nm, "_ZN2nn4diag13GetSymbolNameEPcmm")) {
            if (a0 && a1 > 0) ((char *)(uintptr_t)a0)[0] = 0;   // 空串
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn4diag12GetBacktraceEPmm")) {
            if (a1) *(uint64_t *)(uintptr_t)a1 = 0;              // 深度 0
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn4diag34InitializeApplicationAbortObserverEv")) return 0;
        // ── image：JPEG 编解码器的工作缓冲大小（返 0 ⇒ 游戏分配 0 字节 ⇒ 后续越界）──
        if (!strcmp(nm, "_ZNK2nn5image11JpegEncoder25GetAnalyzedWorkBufferSizeEv")) return 0x10000;
        if (!strcmp(nm, "_ZN2nn5image11JpegEncoder7AnalyzeEv") ||
            !strcmp(nm, "_ZN2nn5image11JpegDecoder7AnalyzeEv")) return 0;
        // ── nn::mem::StandardAllocator：**游戏自有分配器的底座，必须真实现** ──
        // 依据（np213）：nxnative 对它是从 sdk 模块解析**真实现**（加载了全模块）；
        //   我们只装 main ⇒ 8 个符号全落通用存根 ⇒ Initialize 什么都不做、
        //   Allocate 返 0 ⇒ 游戏 `operator new` 全拿 null ⇒ 对象构造静默跳过
        //   ⇒ 「某个对象永远不存在」类症状（X+0x128 无人设就是这一类）。
        // 实现：外部表记 this→{base,size,used}，Allocate 走 bump（不回收，够用）；
        //   **绝不碰 this 指向的内存**（布局未知），只用外部表。
        if (!strncmp(nm, "_ZN2nn3mem17StandardAllocator", 29)) {
            // Initialize(this, mem, size) / C1(this, mem, size)
            if (!strcmp(nm, "_ZN2nn3mem17StandardAllocator10InitializeEPvm") ||
                !strcmp(nm, "_ZN2nn3mem17StandardAllocatorC1EPvm")) {
                int i = sa_lookup(a0, 1);
                if (i >= 0) { g_sa[i].base = a1; g_sa[i].size = a2; g_sa[i].used = 0; }
                klog("[mem] StandardAllocator::Initialize(this=%p, 内存=%p, 大小=%#llx)\n",
                     (void *)(uintptr_t)a0, (void *)(uintptr_t)a1, (unsigned long long)a2);
                return 0;
            }
            if (!strcmp(nm, "_ZN2nn3mem17StandardAllocatorC1Ev")) {
                sa_lookup(a0, 1);
                klog("[mem] StandardAllocator 默认构造(this=%p)\n", (void *)(uintptr_t)a0);
                return 0;
            }
            // Allocate(this, size) / Allocate(this, size, align)
            if (!strcmp(nm, "_ZN2nn3mem17StandardAllocator8AllocateEm") ||
                !strcmp(nm, "_ZN2nn3mem17StandardAllocator8AllocateEmm")) {
                uint64_t align = (nm[34] == 'E' && nm[35] == 'm' && nm[36] == 'm') ? a2 : 16;
                if (align < 16) align = 16;
                int i = sa_lookup(a0, 1);
                if (i < 0) return 0;
                uint64_t p = (g_sa[i].base + g_sa[i].used + align - 1) & ~(align - 1);
                if (!g_sa[i].base || p + a1 > g_sa[i].base + g_sa[i].size) {
                    static int n_ovf; if (++n_ovf <= 5)
                        klog("[mem] ⚠ Allocate(%#llx) 超出池：base=%p size=%#llx used=%#llx ⇒ 退回 calloc\n",
                             (unsigned long long)a1, (void *)(uintptr_t)g_sa[i].base,
                             (unsigned long long)g_sa[i].size, (unsigned long long)g_sa[i].used);
                    return (uint64_t)(uintptr_t)calloc(1, (size_t)a1);   // 池没初始化：退回宿主堆
                }
                g_sa[i].used = (p - g_sa[i].base) + a1;
                return p;
            }
            if (!strcmp(nm, "_ZN2nn3mem17StandardAllocator4FreeEPv")) return 0;   // bump：不回收
            if (!strcmp(nm, "_ZN2nn3mem17StandardAllocator10ReallocateEPvm")) {
                // (this, ptr, size)：不知原大小 ⇒ 新分配 + 拷贝 min(旧,新)（旧大小未知，按新大小拷）
                int i = sa_lookup(a0, 1);
                if (i < 0) return 0;
                uint64_t p = (g_sa[i].base + g_sa[i].used + 15) & ~15ull;
                if (!g_sa[i].base || p + a2 > g_sa[i].base + g_sa[i].size)
                    return (uint64_t)(uintptr_t)calloc(1, (size_t)a2);
                g_sa[i].used = (p - g_sa[i].base) + a2;
                if (a1 && p) memcpy((void *)(uintptr_t)p, (const void *)(uintptr_t)a1, (size_t)a2);
                return p;
            }
            if (!strcmp(nm, "_ZN2nn3mem17StandardAllocator8FinalizeEv")) return 0;
        }
        // ═══ 线程 / TLS / 互斥（2026-10-18 改造：真 pthread）═══
        // nn::os::CreateThread(ThreadType*, entry, arg, stack, stack_size, prio)
        if (!strcmp(nm, "_ZN2nn2os12CreateThreadEPNS0_10ThreadTypeEPFvPvES3_S3_mi")) {
            pthread_mutex_lock(&g_thr_mu);
            int slot = -1;
            for (int i = 0; i < DM_THREAD_MAX; i++)
                if (!g_thr[i].game) { slot = i; break; }
            if (slot >= 0) {
                memset(&g_thr[slot], 0, sizeof g_thr[slot]);
                g_thr[slot].game = a0;
                g_thr[slot].entry = (void (*)(void *))(uintptr_t)a1;
                g_thr[slot].arg = (void *)(uintptr_t)a2;
                g_thr[slot].prio = 16;
                g_thr[slot].id = g_thr_id_next++;
            }
            pthread_mutex_unlock(&g_thr_mu);
            klog("[线程] CreateThread：类型=%#llx 入口=%#llx 参数=%#llx → 槽 %d\n",
                 (unsigned long long)a0, (unsigned long long)a1,
                 (unsigned long long)a2, slot);
            // np92：把**入口参数对象**的虚表与消息处理槽读出来——线程体 0xa1e30c 的消息循环是
            //   `x19 = 入口参数; x9 = [x19]; x9 = [x9+0x80]; blr x9`，所以
            //   **[参数+0] = 虚表，[虚表+0x80] = "收到消息后干什么"**。
            //   客体内存同地址空间可直接读，不用挂钩子（np84 的教训：挂钩子会扰动初始化）。
            if (a2 > 0x10000ull && a2 < 0x800000000000ull) {
                uint64_t vt = *(const uint64_t *)(uintptr_t)a2;
                if (vt > 0x10000ull && vt < 0x800000000000ull) {
                    uint64_t b2 = (uint64_t)(uintptr_t)g_k_m.base;
                    uint64_t h = *(const uint64_t *)(uintptr_t)(vt + 0x80);
                    int in = h >= b2 && h < b2 + DM_IMG_MAIN_END;
                    klog("[线程] 参数=%#llx 虚表=%#llx 消息处理槽[+0x80]=%s%#llx\n",
                         (unsigned long long)a2, (unsigned long long)vt, in ? "+" : "(外)",
                         (unsigned long long)(in ? h - b2 : h));
                }
            }
            return slot >= 0 ? 0 : 1;      // 0 = 成功
        }
        // nn::os::StartThread(ThreadType*) —— 真建宿主线程，蹦床调 entry(arg)
        if (!strcmp(nm, "_ZN2nn2os11StartThreadEPNS0_10ThreadTypeE")) {
            pthread_mutex_lock(&g_thr_mu);
            int slot = thr_find_by_game(a0);
            int ok = -1;
            if (slot >= 0 && !g_thr[slot].started) {
                g_thr[slot].started = 1;
                ok = pthread_create(&g_thr[slot].tid, NULL, thr_trampoline, &g_thr[slot]);
                if (ok != 0) g_thr[slot].started = 0;
            } else if (slot >= 0) ok = 0;      // 已启动：幂等
            pthread_mutex_unlock(&g_thr_mu);
            klog("[线程] StartThread：类型=%#llx 槽 %d → %s\n",
                 (unsigned long long)a0, slot, ok == 0 ? "已启动" : "失败");
            return ok == 0 ? 0 : 1;
        }
        // nn::os::SetThreadName(ThreadType*, const char*)
        if (!strcmp(nm, "_ZN2nn2os13SetThreadNameEPNS0_10ThreadTypeEPKc")) {
            pthread_mutex_lock(&g_thr_mu);
            int slot = thr_find_by_game(a0);
            if (slot >= 0 && a1) {
                snprintf(g_thr[slot].name, sizeof g_thr[slot].name, "%s", (const char *)(uintptr_t)a1);
                if (g_thr[slot].started) pthread_setname_np(g_thr[slot].tid, g_thr[slot].name);
            }
            pthread_mutex_unlock(&g_thr_mu);
            klog("[线程] SetThreadName：类型=%#llx → 「%s」\n",
                 (unsigned long long)a0, a1 ? (const char *)(uintptr_t)a1 : "?");
            return 0;
        }
        // nn::os::GetThreadId(const ThreadType*)
        if (!strcmp(nm, "_ZN2nn2os11GetThreadIdEPKNS0_10ThreadTypeE")) {
            pthread_mutex_lock(&g_thr_mu);
            int slot = thr_find_by_game(a0);
            uint64_t id = slot >= 0 ? g_thr[slot].id : 0;
            pthread_mutex_unlock(&g_thr_mu);
            return id ? id : 1;            // 主线程等未登记者给 1
        }
        // nn::os::GetCurrentThread —— 已启动的游戏线程回它自己的 ThreadType；其余回影子
        if (!strcmp(nm, "_ZN2nn2os16GetCurrentThreadEv")) {
            int slot = thr_find_by_self();
            if (slot >= 0) return g_thr[slot].game;
            return (uint64_t)(uintptr_t)&g_thread_shadow[0];
        }
        // nn::os::AllocateTlsSlot / SetTlsValue / GetTlsValue
        // 2026-10-18 最终形态：**每线程一份影子对象**（_Thread_local），首次取用时把 [+0xb0]
        //   填成注入的分配器对象。为什么不是"真槽 + 按线程存值"（那次崩在 0xa0e33c：某槽被
        //   存过 1，sead 的"设置分配器"助手拿 1 去读 [1+0xb0]）；为什么也不能共用一份
        //   （那会让两个真线程的"当前线程"身份撞车 ⇒ sead 互斥量重入判断失效 ⇒ 第二次
        //   加锁就挂住，真机实证 0xa01924 两连击后停住）。每线程影子两头都占：
        //   [+0xb0] 是全局分配器，其余字段各自独立。
        if (!strcmp(nm, "_ZN2nn2os15AllocateTlsSlotEPNS0_7TlsSlotEPFvmE")) {
            if (a0) *(uint32_t *)(uintptr_t)a0 = 0;      // 槽号恒 0（旧行为，见上）
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn2os11SetTlsValueENS0_7TlsSlotEm")) {
            return 0;                                     // 忽略（见上，防"存非指针值"崩）
        }
        if (!strcmp(nm, "_ZN2nn2os11GetTlsValueENS0_7TlsSlotE")) {
            static _Thread_local uint64_t tls_shadow[0x40];
            if (!tls_shadow[0xb0 / 8])
                tls_shadow[0xb0 / 8] = (uint64_t)(uintptr_t)&g_alloc_obj[0];
            // np172（2026-10-21）：**重新打开 [+0x60] 补口**。
            //   np167 关它的依据是"补了之后实体只到 #9"，但同一天已查明那是**日志预算 400KB
            //   截断造成的假象**（预算提到 1.5MB 后实体本来就有 #40）⇒ 依据作废。
            //   而 np171 的死亡笔记给了硬证据：`img+0x16339f0 si_addr=0 x0=0`，反汇编即
            //   `GetTlsValue → ldr x0,[x0,#0x60]` → `ldr x8,[x0]` → 虚表 +0x10 分配 0x28 字节
            //   ⇒ 缺 [+0x60] 岗位线程必崩（被停住后 PreInitializeApp 的工人就断了）。
            if (!tls_shadow[0x60 / 8])
                tls_shadow[0x60 / 8] = (uint64_t)(uintptr_t)&g_alloc_obj[0];
            return (uint64_t)(uintptr_t)&tls_shadow[0];
        }
        // 核掩码 / 栈信息 / 优先级 —— 影子语义，别让游戏拿到 0 而误判"没有可用核"
        if (!strcmp(nm, "_ZN2nn2os26GetThreadAvailableCoreMaskEv")) return 0xF;
        if (!strcmp(nm, "_ZN2nn2os20GetCurrentCoreNumberEv")) return 0;
        if (!strcmp(nm, "_ZN2nn2os19GetCurrentStackInfoEPmS1_")) {
            if (a0) *(uint64_t *)(uintptr_t)a0 = 0;
            if (a1) *(uint64_t *)(uintptr_t)a1 = 0x100000;   // 1 MB
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn2os17GetThreadPriorityEPKNS0_10ThreadTypeE")) return 16;
        if (!strcmp(nm, "_ZN2nn2os20ChangeThreadPriorityEPNS0_10ThreadTypeEi")) {
            pthread_mutex_lock(&g_thr_mu);
            int slot = thr_find_by_game(a0);
            if (slot >= 0) g_thr[slot].prio = (int)(int32_t)a1;
            pthread_mutex_unlock(&g_thr_mu);
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn2os17SetThreadCoreMaskEPNS0_10ThreadTypeEim")) {
            pthread_mutex_lock(&g_thr_mu);
            int slot = thr_find_by_game(a0);
            if (slot >= 0) g_thr[slot].core_mask = (int)(int32_t)a1;
            pthread_mutex_unlock(&g_thr_mu);
            return 0;
        }
        // nn::os 互斥量 —— 真 pthread_mutex
        if (!strcmp(nm, "_ZN2nn2os12TryLockMutexEPNS0_9MutexTypeE")) {
            pthread_mutex_t *mu = mtx_get((void *)(uintptr_t)a0);
            int r = pthread_mutex_trylock(mu) == 0 ? 1 : 0;
            if (r) {
                int idx = mtx_idx((void *)(uintptr_t)a0);
                if (idx >= 0) g_mtx_owner[idx] = (long)gettid();
            }
            return (uint64_t)r;
        }
        if (!strcmp(nm, "_ZN2nn2os9LockMutexEPNS0_9MutexTypeE")) {
            // np160：成对打点 + 持锁者登记（详见 g_mtx_owner 处注释）。
            //   **有进入无离开 = 卡在这把锁**——本轮"静默卡死"的根因定位器。
            pthread_mutex_t *mu = mtx_get((void *)(uintptr_t)a0);
            int idx = mtx_idx((void *)(uintptr_t)a0);
            int log_it = mtx_log_ok();
            if (log_it)
                klog("[锁] LockMutex 进入 mutex=%#llx idx=%d tid=%ld 当前owner=%ld\n",
                     (unsigned long long)a0, idx, (long)gettid(),
                     idx >= 0 ? g_mtx_owner[idx] : -1);
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            pthread_mutex_lock(mu);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            long ms = (long)(t1.tv_sec - t0.tv_sec) * 1000 + (long)(t1.tv_nsec - t0.tv_nsec) / 1000000;
            if (idx >= 0) g_mtx_owner[idx] = (long)gettid();
            if (log_it)
                klog("[锁] LockMutex 离开 mutex=%#llx tid=%ld 耗时=%ld ms（owner 置本线程）\n",
                     (unsigned long long)a0, (long)gettid(), ms);
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn2os11UnlockMutexEPNS0_9MutexTypeE")) {
            int idx = mtx_idx((void *)(uintptr_t)a0);
            long me = (long)gettid();
            long ow = idx >= 0 ? g_mtx_owner[idx] : -1;
            int log_it = mtx_log_ok();
            if (log_it)
                klog("[锁] UnlockMutex mutex=%#llx idx=%d tid=%ld owner=%ld%s\n",
                     (unsigned long long)a0, idx, me, ow,
                     (ow > 0 && ow != me) ? "（非持锁者解锁！可疑）" : "");
            if (idx >= 0) g_mtx_owner[idx] = 0;
            pthread_mutex_unlock(mtx_get((void *)(uintptr_t)a0));
            return 0;
        }
        // ═══ 消息队列（见 mq_get 处注释）═══
        if (!strcmp(nm, "_ZN2nn2os22InitializeMessageQueueEPNS0_16MessageQueueTypeEPmm")) {
            dm_mq_t *q = mq_get((void *)(uintptr_t)a0);
            if (q) {
                q->buf = (uint64_t *)(uintptr_t)a1;    // 游戏提供的值缓冲区
                q->cap = (size_t)a2;
                q->head = q->tail = q->n = 0;
                klog("[队列] 初始化：%#llx buf=%#llx 容量=%llu\n",
                     (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
                // ⭐ np181：**客体队列对象布局 dump**。假设：主线程等待环读的是"客体队列对象"
                //   里的字段（如 [queue+0x128]），而我们把队列状态只存在自己的 dm_mq_t 里、
                //   从没回写客体对象 ⇒ 客体检视者永远看到"空队列"。dump 出布局以便定位该字段
                //   （已知量：buf=a1、cap=a2 一定在对象里，靠它们当锚点）。
                {
                    static int nq;
                    if (++nq <= 4 && a0 > 0x10000ull) {
                        char d[520];
                        for (int r = 0; r < 3; r++) {
                            int len = snprintf(d, sizeof d, "qobj#%d[%02x]:", nq, r * 0x40);
                            for (int k = 0; k < 8; k++) {
                                uint64_t v = ((const uint64_t *)(uintptr_t)a0)[r * 8 + k];
                                len += snprintf(d + len, sizeof d - (size_t)len, " %#llx",
                                                (unsigned long long)v);
                            }
                            snprintf(d + len, sizeof d - (size_t)len, "\n");
                            note_append("dm_gate.txt", d);
                        }
                    }
                }
            }
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn2os19ReceiveMessageQueueEPmPNS0_16MessageQueueTypeE")) {
            dm_mq_t *q = mq_get((void *)(uintptr_t)a1);
            if (!q || !q->cap) return 0;               // 未初始化：别悬住（宁可空转）
            pthread_mutex_lock(&q->mu);
            if (q->n == 0) {
                // np180：记"谁在等哪个队列"（配对记录，用来找失踪的 Send）
                {
                    uint64_t lr; __asm__ volatile("mov %0, x30" : "=r"(lr));
                    q->wait_n++;
                    q->wait_tid = (uint32_t)(uintptr_t)pthread_self();
                    q->wait_lr = lr;
                }
                static int nw;
                if (++nw <= 20) {
                    // np82：谁在等？x30 = 调用者返回地址（plt_stub 里可读）
                    uint64_t lr; __asm__ volatile("mov %0, x30" : "=r"(lr));
                    uint64_t b2 = (uint64_t)(uintptr_t)g_k_m.base;
                    int in_self = g_self_base && lr >= g_self_base && lr < g_self_base + 0x400000ull;
                    klog("[队列] Receive(%#llx) 队列空 → 阻塞等（第 %d 次）调用者=%s%#llx\n",
                         (unsigned long long)a1, nw,
                         (lr >= b2 && lr < b2 + DM_IMG_MAIN_END) ? "映像+" :
                         (in_self ? "本so+" : "(外)"),
                         (unsigned long long)((lr >= b2 && lr < b2 + DM_IMG_MAIN_END) ? lr - b2 :
                                              (in_self ? lr - g_self_base : lr)));
                }
                while (q->n == 0) {
                    // np157：**轮询式醒来**（放锁 → nanosleep 250ms → 重锁 → 重扫）。
                    // 为什么改成这样（np153–np156 四连教训）：原想借"工人定时醒"当运行期
                    // 唯一的活着采样时钟（主线程=UI 线程，setInterval/宿主 pthread 在运行期
                    // 排不进队），但 pthread_cond_timedwait 的定时超时在本平台**根本没响**
                    // （MONOTONIC 混用不响；改 REALTIME+REALTIME 经典组合同样只醒过一次）
                    // ⇒ 不再依赖 cond 超时：放锁后 nanosleep（纯 syscall，os_sleep 一直在用）
                    // 再重锁重扫。语义不变（消息到达后重扫必然发现；最坏多 250ms 延迟）
                    // + 醒来时打一份线程快照（主线程停在内核里时，wchan 就在这里被抓到）。
                    pthread_mutex_unlock(&q->mu);
                    struct timespec ts = { 0, 250000000L };
                    nanosleep(&ts, NULL);
                    pthread_mutex_lock(&q->mu);
                    if (q->n == 0) dm_probe_thread_probe();
                }
            }
            uint64_t v = q->buf[q->head];
            q->head = (q->head + 1) % q->cap;
            q->n--;
            pthread_cond_signal(&q->cv_notfull);
            pthread_mutex_unlock(&q->mu);
            if (a0) *(uint64_t *)(uintptr_t)a0 = v;
            // np98：消息要交给线程体处理了——线程体 0xa1e30c 的循环是
            //   `x19 = 线程对象; x9 = [x19]; x9 = [x9+0x80]; blr x9`。
            //   队列在对象里的偏移是 +0x38 ⇒ **线程对象 = 队列地址 - 0x38**，
            //   于是 [对象+0]=虚表、[虚表+0x80]=消息处理函数。客体内存可直接读，不挂钩子。
            {
                uint64_t obj = (uint64_t)(uintptr_t)a1 - 0x38;
                static int nm2;
                if (++nm2 <= 6 && obj > 0x10000ull && obj < 0x800000000000ull) {
                    uint64_t vt = *(const uint64_t *)(uintptr_t)obj;
                    uint64_t h = (vt > 0x10000ull && vt < 0x800000000000ull)
                               ? *(const uint64_t *)(uintptr_t)(vt + 0x80) : 0;
                    uint64_t b2 = (uint64_t)(uintptr_t)g_k_m.base;
                    int in = h >= b2 && h < b2 + DM_IMG_MAIN_END;
                    klog("[队列] 收到消息=%#llx；线程对象=%#llx 虚表=%#llx 处理函数=%s%#llx\n",
                         (unsigned long long)v, (unsigned long long)obj,
                         (unsigned long long)vt, in ? "+" : "(外)",
                         (unsigned long long)(in ? h - b2 : h));
                    // np99：第二跳——0xa1df50 是 x0=[x9+0x100]; x10=[x0]; x3=[x10]; br x3
                    uint64_t tbl = *(const uint64_t *)(uintptr_t)(obj + 0x100);
                    uint64_t real = (tbl > 0x10000ull && tbl < 0x800000000000ull)
                                  ? *(const uint64_t *)(uintptr_t)tbl : 0;
                    real = (real > 0x10000ull && real < 0x800000000000ull)
                           ? *(const uint64_t *)(uintptr_t)real : 0;
                    {
                        int in2 = real >= b2 && real < b2 + DM_IMG_MAIN_END;
                        klog("[队列] 第二跳：表=%#llx 真正处理函数=%s%#llx\n",
                             (unsigned long long)tbl, in2 ? "+" : "(外)",
                             (unsigned long long)(in2 ? real - b2 : real));
                    {   // 第三跳：0xa02478 是 std::function 调用器——[对象+8] 才是真回调，
                        // 为 0 就"收到消息但什么都不做"（= 回调没被注册）
                        uint64_t fobj = *(const uint64_t *)(uintptr_t)tbl;
                        uint64_t cb = (fobj > 0x10000ull && fobj < 0x800000000000ull)
                                      ? *(const uint64_t *)(uintptr_t)(fobj + 8) : 0;
                        int in3 = cb >= b2 && cb < b2 + DM_IMG_MAIN_END;
                        klog("[队列] 第三跳：回调对象=%#llx 真回调=%s%#llx%s\n",
                             (unsigned long long)fobj, in3 ? "+" : "(外)",
                             (unsigned long long)(in3 ? cb - b2 : cb),
                             cb ? "" : "（空 ⇒ 消息白收）");
                    }
                    }
                }
            }
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn2os19TrySendMessageQueueEPNS0_16MessageQueueTypeEm")) {
            dm_mq_t *q = mq_get((void *)(uintptr_t)a0);
            if (!q || !q->cap) return 0;
            pthread_mutex_lock(&q->mu);
            if (q->n >= q->cap) { pthread_mutex_unlock(&q->mu); return 0; }   // 满：bool 假
            q->buf[q->tail] = a1;
            q->tail = (q->tail + 1) % q->cap;
            q->n++;
            pthread_cond_signal(&q->cv_notempty);
            pthread_mutex_unlock(&q->mu);
            q->send_n++; q->send_tid = (uint32_t)(uintptr_t)pthread_self();
            q->send_lr = lr0; q->send_val = a1;
            {   // np77：找"谁该给空队列发消息"——发送方打点（与被阻塞的 Receive 对应）
                static int ns;
                if (++ns <= 24) {
                    uint64_t lr; __asm__ volatile("mov %0, x30" : "=r"(lr));
                    uint64_t b2 = (uint64_t)(uintptr_t)g_k_m.base;
                    int in_self = g_self_base && lr >= g_self_base && lr < g_self_base + 0x400000ull;
                    klog("[队列] TrySend(%#llx) 值=%#llx 送入后 n=%zu 调用者=%s%#llx\n",
                         (unsigned long long)a0, (unsigned long long)a1, q->n,
                         (lr >= b2 && lr < b2 + DM_IMG_MAIN_END) ? "映像+" :
                         (in_self ? "本so+" : "(外)"),
                         (unsigned long long)((lr >= b2 && lr < b2 + DM_IMG_MAIN_END) ? lr - b2 :
                                              (in_self ? lr - g_self_base : lr)));
                }
            }
            return 1;                                   // bool 真 = 已送入
        }
        if (!strcmp(nm, "_ZN2nn2os16SendMessageQueueEPNS0_16MessageQueueTypeEm")) {
            dm_mq_t *q = mq_get((void *)(uintptr_t)a0);
            if (!q || !q->cap) return 0;
            pthread_mutex_lock(&q->mu);
            while (q->n >= q->cap) pthread_cond_wait(&q->cv_notfull, &q->mu);
            q->buf[q->tail] = a1;
            q->tail = (q->tail + 1) % q->cap;
            q->n++;
            pthread_cond_signal(&q->cv_notempty);
            pthread_mutex_unlock(&q->mu);
            q->send_n++; q->send_tid = (uint32_t)(uintptr_t)pthread_self();
            q->send_lr = lr0; q->send_val = a1;
            {   // np77：同上，阻塞型发送方打点
                static int ns; if (++ns <= 24)
                    klog("[队列] Send(%#llx) 值=%#llx 送入后 n=%zu\n",
                         (unsigned long long)a0, (unsigned long long)a1, q->n);
            }
            return 0;
        }
        if (!strcmp(nm, "_ZN2nn2os22TryReceiveMessageQueueEPmPNS0_16MessageQueueTypeE")) {
            dm_mq_t *q = mq_get((void *)(uintptr_t)a1);
            if (!q || !q->cap) return 0;
            pthread_mutex_lock(&q->mu);
            if (q->n == 0) { pthread_mutex_unlock(&q->mu); return 0; }
            uint64_t v = q->buf[q->head];
            q->head = (q->head + 1) % q->cap;
            q->n--;
            pthread_cond_signal(&q->cv_notfull);
            pthread_mutex_unlock(&q->mu);
            if (a0) *(uint64_t *)(uintptr_t)a0 = v;
            return 1;
        }
        // np131：**TryPeekMessageQueue** —— 看一眼有没有消息，**不取走**（取走是 TryReceive）。
        // 真机实证：它此前没实现（落通用桩恒返 0 =「没有消息」）⇒ 游戏线程一直在 Peek 上空转
        //   （日志尾部几千行 `[PLT] ...TryPeekMessageQueue`），而队列里的消息永远没人处理。
        // 返回值语义与 nn 一致：非 0 = 看到了（消息仍在队列里）。
        if (!strcmp(nm, "_ZN2nn2os19TryPeekMessageQueueEPmPKNS0_16MessageQueueTypeE")) {
            dm_mq_t *qq = mq_get((void *)(uintptr_t)a1);
            if (!qq || !qq->cap) return 0;
            pthread_mutex_lock(&qq->mu);
            int hit = qq->n != 0;
            uint64_t vv = hit ? qq->buf[qq->head] : 0;
            pthread_mutex_unlock(&qq->mu);
            if (hit && a0) *(uint64_t *)(uintptr_t)a0 = vv;   // 只写回、不移动 head
            return (uint64_t)hit;
        }
        // 让出/等待类：真线程下 Yield 用 sched_yield
        if (!strcmp(nm, "_ZN2nn2os11YieldThreadEv")) { sched_yield(); return 0; }
        // nvnBootstrapLoader —— NVN 图形驱动引导（图形栈大门，2026-10-18 到达）
        // 真机实证：游戏只取两个名字（nvnDeviceInitialize / nvnDeviceGetProcAddress），
        // 其余全走 GetProcAddress。⇒ 这里按名字分派到最小空设备（见 nvn_dev_init 注释）。
        if (!strcmp(nm, "nvnBootstrapLoader")) {
            const char *nvn = a0 ? (const char *)(uintptr_t)a0 : NULL;
            static int n_nvn;
            if (++n_nvn <= 40)
                klog("[NVN] nvnBootstrapLoader(「%s」)\n", nvn ? nvn : "(null)");
            if (nvn && !strcmp(nvn, "nvnDeviceInitialize")) return (uint64_t)(uintptr_t)&nvn_dev_init;
            if (nvn && !strcmp(nvn, "nvnDeviceGetProcAddress")) return (uint64_t)(uintptr_t)&nvn_dev_gpa;
            return (uint64_t)(uintptr_t)&nvn_stub;
        }
        // ★ C++ 静态局部对象的初始化守卫：**已迁到通用胶水层**（dm/ohos/dm_sdk_ohos.c）。
        //   迁走的原因（2026-10-20 np145 定谳）：这是**与游戏无关的 C++ ABI 语义**，属通用框架能力；
        //   而旧实现在这里是"恒返回 1 + release 空操作 + 从不写 guard 字节" ⇒
        //   游戏自己的 `ldarb`+`tbnz #0`"已初始化"判据**永不成立**
        //   ⇒ 25179 个静态局部初始化器**每次被调用都重跑一遍**（重复写字段/加解锁/分配）。
        //   现由 dm_sdk_ohos.c 的 cxa_guard_acquire/release 实现真正的字节状态机。
        // ===== 语义已知的服务：成批实现（"步子适度放大"：这些不必一个个试）=====
        // 真时基：返回 ns 级单调时间（返回 0 会让游戏的计时/节拍逻辑失真）
        if (!strcmp(nm, "_ZN2nn2os13GetSystemTickEv")) {
            struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
            static uint64_t base_ns;
            uint64_t now = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
            if (!base_ns) base_ns = now;
            return (now - base_ns) * 19200ull / 1000000000ull;   // Switch 时基 ≈ 19.2MHz
        }
        // C++ 纯虚调用兜底：落到这里就是"调了纯虚函数"，记名后返回（不无限循环，便于定位）
        if (!strcmp(nm, "__cxa_pure_virtual")) { klog("[服务] ⚠ __cxa_pure_virtual 被调用（游戏有纯虚误调）\n"); return 0; }
        // 通用 PLT 记名 + **全局截断**：init 段约一千行全是信息量；但游戏跑起来后
        // ConvertToTimeSpan/SleepThread 每帧各调一次，一轮能刷出 170MB 日志把别的线索淹掉
        // （2026-10-17 实证）。到上限就只记一条截断宣言，计数仍走 g_plt_calls。
        // ★ NVN 用量打点：**借游戏自己的帧线程**（ConvertToTimeSpan 每帧一次）。
        //   为什么不用宿主线程：np26–np29 三个回合，宿主看护线程一行日志都没打出来
        //   （pthread_create 返回成功、失败分支没触发 ⇒ 线程建了但从未被调度，见
        //   nvn_host_thread_probe 的注释）⇒ 观测必须挂在"确定会被调度"的线程上。
        // ⚠ 阈值别定大：np29 用了 120（帧），而日志里 ConvertToTimeSpan 约每 90 行才 1 次
        //   ⇒ 要 1 万行才打一次，跑测根本等不到。用 20。
        if (!strcmp(nm, "_ZN2nn2os17ConvertToTimeSpanENS0_4TickE")) {
            static unsigned tick;
            if (++tick % 20 == 0) nvn_family_report("帧线程");
            // np151：记下帧线程 tid（线程快照里打 ←帧线程 标记；主线程停车取证用）
            if (!g_tick_tid) g_tick_tid = (int)gettid();
            // np111：借游戏自己的帧线程定期 dump RootTask 的阶段（P0-2）。
            //   第 20 帧一次（早看），之后每 200 帧一次（看是否卡住）。
            if (tick == 20 || tick % 200 == 0) rt_dump("周期");
            // np136（批次 B）：同点位报门观测桩命中——宿主定时器线程永不被调度
            //   （仪器铁律），累计值看互斥量链路在主循环期踩不踩 svc 门。
            if (tick == 20 || tick % 200 == 0) k_gate_obs_dump("帧线程");
            // np177：per-thread 门命中（写 dm_gate.txt，绕过预算）。判"主线程在 sdk 里自旋"还是"纯 guest 自旋"。
            if (tick == 20 || tick % 1000 == 0) gate_thr_dump("帧线程");
            // np180：队列配对记录（谁在等 / 谁发过）
            if (tick == 20 || tick % 1000 == 0) mq_pair_dump("帧线程");
            // np178：主线程 PC 直方图（判"这在 guest 里哪一段打转"）——先投一枪再 dump
            if (tick % 5 == 0) smp_kick();
            if (tick == 20 || tick % 100 == 0) smp_dump("帧线程");
            // np143：nnMain 运行期采样单例 A/B —— 判"建 A/B 的初始化链"是否在 nnMain 段内跑。
            //   挂在帧线程（唯一确定会被调度、又确在 nnMain 运行期的线程）上。
            if (tick == 20 || tick % 200 == 0) k_ab_obs("帧线程");
            // np145：guard 语义修复后的统计摘要（acquire 总数 / 我初始化 / 不同 guard / 重跑）。
            //   ★ 借游戏自己的帧线程打（仪器铁律：观测绝不挂宿主线程）。
            //   判据：**"重跑"计数应恒为 0**；"我初始化"应 ≈ "不同 guard"。
            if (tick == 20 || tick % 200 == 0) dm_sdk_guard_report();
            // np150：svc 同步层"当前阻塞者"快照（有等待者才打）。
            //   背景：np150 证实主线程 S 态停车而 PLT/dm_sdk 等待族 0 打点 ⇒ 冻结点
            //   只可能在 svc 层（0x1A 锁仲裁 / 0x1C 进程级 key / 0x18 多对象）。
            if (tick == 20 || tick % 200 == 0) dm_svc_sync_wait_report();
            // np103（P0-8 修正读偏移，2026-10-20）：0x22dce70 是**带双层 guard 的静态局部对象**，
            //   真实布局（§1.7.2 符号级闭环，勿再改）：+0x00 外层 guard 字节 / +0x08 载荷（两个 u32）
            //   / +0x10 内层 guard 字节 / +0x48 条目数（代码写死的 2，**不是运行计数**）
            //   / +0x50 表项指针（初始化器写完指向 tbl+0x08）。
            //   ⚠ 旧版读 tbl+0x18 当"缓存指针"是读错偏移 ⇒ "内容全 0"的旧结论作废（§1.7.2）。
            //   guard 字节语义（P0-7 修复后）：0=FREE / 4=BUSY / 2=DONE ⇒ 2 ⇒ 初始化器确实跑过。
            if (tick % 400 == 100) {
                uint64_t b2 = (uint64_t)(uintptr_t)g_k_m.base;
                if (b2) {
                    const uint8_t *tbl = (const uint8_t *)(uintptr_t)(b2 + 0x22dce70);
                    uint8_t g_out = tbl[0x00], g_in = tbl[0x10];
                    uint32_t p0 = *(const uint32_t *)(uintptr_t)(tbl + 0x08);
                    uint32_t p1 = *(const uint32_t *)(uintptr_t)(tbl + 0x0c);
                    uint32_t cnt = *(const uint32_t *)(uintptr_t)(tbl + 0x48);
                    uint64_t item = *(const uint64_t *)(uintptr_t)(tbl + 0x50);
                    klog("[变体表] tick=%u guard[out=%u in=%u] 载荷={%#x,%#x} 条目=%u 表项指针=%#llx%s\n",
                         tick, g_out, g_in, p0, p1, cnt, (unsigned long long)item,
                         item == (uint64_t)(uintptr_t)(tbl + 0x08) ? "（自指 tbl+0x08，符合预期）" : "");
                    // np103b：字符串对象(0x22dcdf0)的虚表来自 *(0x22828b0)+0x10——
                    //   若那又是我们的假堆/替身虚表 ⇒ 虚调用全返 0 ⇒ 字符串永远空（根因实锤）。
                    uint64_t g28 = *(const uint64_t *)(uintptr_t)(b2 + 0x22828b0);
                    uint64_t svt = *(const uint64_t *)(uintptr_t)(b2 + 0x22dcdf0);
                    uint64_t m18 = (svt > 0x10000ull && svt < 0x800000000000ull)
                                 ? *(const uint64_t *)(uintptr_t)(svt + 0x18) : 0;
                    const char *who28 = "(?)";
                    if (g28 >= (uint64_t)(uintptr_t)g_fake_heap && g28 < (uint64_t)(uintptr_t)g_fake_heap + 0x4000) who28 = "假堆!";
                    klog("[变体表] *(0x22828b0)=%#llx%s 串对象虚表=%#llx 槽[+0x18]=%#llx\n",
                         (unsigned long long)g28, who28,
                         (unsigned long long)svt, (unsigned long long)m18);
                    // np105：**启动包装载的门**（0xd5b670 附近那段：Pack/Bootup_Graphics.pack）：
                    //   x8 = *(0x2276168); x8 = [x8]; if ([x8+0x419] != 0) ⇒ **跳过装载**
                    //   实机读这个标志 + 那个全局指向谁。
                    if (tick <= 400) {
                        uint64_t g76 = *(const uint64_t *)(uintptr_t)(b2 + 0x2276168);
                        uint8_t gate = 0xff;
                        if (g76 > 0x10000ull && g76 < 0x800000000000ull)
                            gate = *(const uint8_t *)(uintptr_t)(g76 + 0x419);
                        klog("[启动包门](旧单层读，仅参考) 外层=%#llx [+0x419]=%#x\n",
                             (unsigned long long)g76, gate);
                    }
                }
            }
        }
        // ═══ 通用 SDK 胶水层（L1，dm_sdk_ohos.c）═══
        // vi/oe 这类"轻量语义"已搬到独立模块：与游戏无关、换游戏一行不改。
        // 探针只做分派；表里没有的仍落下面的通用记名 + 返 0 路径。
        {
            uint64_t r = 0;
            if (dm_sdk_call(nm, a0, a1, a2, a3, &r)) return r;
        }
        if (g_plt_log_n < DM_PLT_LOG_CAP) {
            g_plt_log_n++;
            klog("[PLT] %s\n", nm);
        } else if (g_plt_log_n == DM_PLT_LOG_CAP) {
            g_plt_log_n++;
            klog("[PLT] === 截断：通用 PLT 记名已达 %d 行，后续不再逐条记（总数仍在 g_plt_calls）===\n",
                 DM_PLT_LOG_CAP);
        }
    }
    return 0;
}
// 宿主分配器：按游戏的对象布局喂堆。分配器 0x9f34d8 的路径：
//   *(0x226f0a0) → 持有者 → *持有者 = 「对象 A」→ (x1==0 时) 0xa0e2f8 取「当前分配器」
//   → [A+8] → *该对象 = 虚表 → 虚表[0x30] = 分配方法 → 虚调用 (x0=对象, x1=大小, w2=8)
// 所以造：A = { &虚表, &A }（+8 自指），虚表[6] = host_alloc；两个全局指向对应持有者。
static uint64_t g_alloc_calls;
static void *host_alloc(void *self, uint64_t size, uint64_t flags) {
    (void)self; (void)flags;
    g_alloc_calls++;
    // calloc 而非 malloc：游戏把这块当自己的对象/池用（与 AllocateMemoryBlock 同一理由——
    // 喂未初始化内存会让游戏拿到垃圾指针/垃圾空闲链，2026-10-16 实证过一次）。
    void *p = calloc(1, (size_t)size);
    klog("[堆] 分配 %llu 字节 → %p\n", (unsigned long long)size, p);
    return p;
}
// 虚表兜底：游戏会往这个"分配器替身"上翻到我们没印的页（真机实证：第 17 槽 +0x88 越界 ⇒ blr 0 崩）。
// 兜底里记**调用者返回地址**——槽号自己不知道，但由返回地址可离线反查是哪一页。
static uint64_t vtbl_fallback(void *self, void *arg) {
    (void)self; (void)arg;
    klog("[虚表兜底] 被调用：返回地址=%p（离线反查即知是哪个槽）\n", __builtin_return_address(0));
    return 0;
}
// 第 17 槽（+0x88）：调用点 0xa0e120 后紧跟 `tbz w0, #0` ⇒ 返回值当 bool，bit0=1 表示成功。
// ⭐ np158：武装期内取 x30 + guest 栈——第 4 次插入后的 3 个第17槽调用是**最后的客体
// 动作**，第 3 份即停车边界（其 x30 = 停车后第一段客体会执行的返回地址）。
static int g_s17_arm;
static uint64_t vtbl_slot17_ok(void *self, void *arg) {
    (void)self; (void)arg;
    if (g_s17_arm > 0) {
        g_s17_arm--;
        uint64_t sp, lr;
        __asm__ volatile("mov %0, sp" : "=r"(sp));
        __asm__ volatile("mov %0, x30" : "=r"(lr));
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        klog("[取栈] ★ 第17槽#%d：x30(返回客体)=%s+%#llx 客体SP≈%#llx\n",
             3 - g_s17_arm,
             (lr >= base && lr < base + 0x2600000) ? "映像+" : "外",
             (unsigned long long)((lr >= base && lr < base + 0x2600000) ? lr - base : lr),
             (unsigned long long)sp);
        for (int i = 0; i < 24; i++) {
            uint64_t v = *(const uint64_t *)(uintptr_t)(sp + (uint64_t)i * 8);
            int in = (v >= base && v < base + 0x2600000);
            klog("[取栈]   [sp+#%#04x] = %#llx%s\n", i * 8, (unsigned long long)v,
                 in ? " ←main 映像内" : "");
        }
    }
    klog("[虚表] 第17槽(+0x88) 被调用 → 返回 1（bool 成功）\n");
    return 1;
}
// 第 16 槽（+0x80）：调用点 0xa091ac ⇒ 返回值被 `and #~7`（8 字节对齐）后要求 ≥0x131，
// 随后作为尺寸交给第 6 槽去**分配**（真机实证：兜底返 0 ⇒ sead::ThreadMgr 查找失败 ⇒ 崩 0x9fe2ec）。
// 语义是"报这块对象需要多大"（造 sead::ThreadMgr 用的）⇒ 给宽裕值：大了只浪费，小了被写越界。
static uint64_t vtbl_slot16_size(void *self, void *arg) {
    (void)self; (void)arg;
    klog("[虚表] 第16槽(+0x80) 被调用 → 返回尺寸 0x1000\n");
    return 0x1000;
}
// np164：分配器虚表 +0x10（槽 2）= **按尺寸分配**。
// 依据（真机死亡笔记 + 反汇编）：0x16339e4 `GetTlsValue(slot)` → `[ret+0x60]`（空）
//   → 期望是"当前线程分配器"；0x16339f0 处 `ldr x8,[obj]; ldr x8,[x8,#0x10];
//   w1=0x28; blr x8` ⇒ 调虚表 +0x10 的 allocate(size)，返回值被当对象指针
//   （随后 `stp xzr,xzr,[x19,#0x18]` 清零对象头）。
//   旧表全铺 vtbl_fallback（返 0）⇒ allocate 返 0 ⇒ 写 [0+0x18] 二次崩。
static uint64_t vtbl_slot2_alloc(void *self, uint64_t size) {
    (void)self;
    size_t n = (size_t)size;
    if (n == 0 || n > 0x1000000) n = 0x40;      // 异常尺寸兜底
    void *p = calloc(1, n);
    static int nlog;
    if (++nlog <= 8)
        klog("[虚表] 第2槽(+0x10) allocate(%llu) → %p\n", (unsigned long long)size, p);
    return (uint64_t)(uintptr_t)p;
}
#define DM_ALLOC_VTBL_QWORDS 0x40          // 0x200 字节：够覆盖游戏翻到的 +0xd0
static uint64_t g_alloc_vtbl[DM_ALLOC_VTBL_QWORDS];   // 虚表：槽 6（+0x30）= host_alloc，槽 17（+0x88）见上
static uint64_t g_alloc_obj[2];       // 对象 A：{ &虚表, &A }
static uint64_t g_alloc_holder[1];    // 持有者：{ &A }
static uint64_t g_ctx_holder[1];      // 上下文持有者：{ &ctx }
static uint64_t g_ctx_obj[0x20];      // 上下文对象（[0x88] 处为 0 = 让 thunk 返回 0，走本线程路径）
static uint64_t g_mgr_obj[8];       // 管理器对象（工厂 0x1f8 入参影子）：+0x00 虚表 / +0x18 子对象链 / +0x20 名字串
static uint64_t g_mgr_sub[8];       // 子对象：+0x18 处放虚方法指针
static uint64_t g_mgr_vt[8];        // 管理器虚表桩：构造器 0x103b83c 只读 [vt+0x10] 一个标志字节 ⇒ 零缓冲即可
static char g_mgr_name[] = "";      // 名字串：strlen=0 ⇒ 拷贝长度 0 ⇒ 不去碰 NULL 拷贝源（契约见 k_inject_heap）
// ── *(0x227df98) 指向的 .bss 全局（模块偏移 0x22d9198）──
// 真机+离线双重确认：**模块代码从不写它**（连同我们已成功注入的 0x22d8a18/0x22d8a40 也一样）
// ⇒ 这些全局本由 Switch 外部运行时填，鸿蒙上没有 ⇒ 只能由宿主补影子。
// 用法（崩点 base+0x9FE68C → 尾调用 0xa21a78）：`ldr x0,[x9]`（x9=*(0x227df98)）得到对象，
// 0xa21a78 再 `ldr x1,[x0,#0x20]; cbz; br x1` —— 所以 [+0x20] 要放一个函数指针。
static uint64_t g_cb_obj[0x10];          // 影子对象（够覆盖 +0x20）
static uint64_t g_cb_calls;
static uint64_t cb_stub(void) {
    // 每 1024 次才记一行：真机上它被调了 136 万次，逐条打会把日志刷爆、反而看不见别的线索
    if ((++g_cb_calls & 0x3FFu) == 1u)
        klog("[回调] 0xa21a78 转调（第 %llu 次）\n", (unsigned long long)g_cb_calls);
    return 0;
}

// ── nnMain 路径的 sead::ThreadMgr（文档"旷野之息 main 专属项"之一）──
// 真机实证（2026-10-17）：崩点 base+0x9FE9A8 的 `ldr x8,[x19]`，si_addr=0。反汇编：
//   0x9fe98c  ldr x8, [x8, #0xac0]   ← x8 = *(0x227aac0)（这个槽**一直没注入**）
//   0x9fe990  ldr w9, [x8] ; cbz w9  ← 读到的标志是 0 ⇒ 走 x19 = 0
//   0x9fe9a8  ldr x8, [x19]          ← 读 [0] ⇒ 崩
// 所以这里补上：P=[0x227aac0] 要求 P[0]≠0（标志）、P[8]→对象；对象[0]=虚表，
// 虚表 +0x80 会以 (x0=对象, w1=8) 被调用（调用后接着查 "sead::SystemManagers"）。
static uint64_t g_tm_a[2];          // P：{ 标志, &ThreadMgr }
static uint64_t g_tm_obj[2];        // ThreadMgr：{ &虚表, … }
#define DM_TM_VTBL_QWORDS 0x40
static uint64_t g_tm_vtbl[DM_TM_VTBL_QWORDS];
static uint64_t tm_method(void *self, uint64_t arg) {
    (void)self;
    klog("[sead::ThreadMgr] 虚表 +0x80 被调用（w1=%#llx）\n", (unsigned long long)arg);
    return 0;
}
static uint64_t mgr_method(void *self) {
    klog("[管理器] 虚方法被调用（self=%p）—— 宿主存根应答\n", self);
    return 0;
}
// 日志对象 + 节点池：nnMain 开头调用的日志辅助 0xf35374 从 *(0x2288548) 取对象，
// 字段：[+0x10]=计数（0）、[+0x28]=上限（64）、[+0x18]=节点链表头。
// 节点：>=0x20 字节，[+0]=下一节点。链表**必须非空**（空表会让它往地址 0 写）。
#define LOG_POOL_N 64
// 每节点 0x40 字节（2026-10-18 从 0x20 扩容）：离线解码游戏日志添加函数 0xf35374 得到
// 节点布局 [0]=tick、[8]=消息串、[0x10]=链节点（0x9f3904 插入用，[link]=next/[link+8]=prev）、
// [0x18]=0 ⇒ 至少 0x20；留一倍余量，排除"节点写越界污染相邻节点"的嫌疑（真机上见过
// [node+8] 拿到低 32 位对、高 16 位 +0x200 的坏指针）。
static uint64_t g_log_pool[LOG_POOL_N][8];      // 每节点 0x40 字节
static uint64_t g_log_obj[8];
static void k_inject_log(uint8_t *img) {
    for (int i = 0; i < LOG_POOL_N; i++)
        g_log_pool[i][0] = (i + 1 < LOG_POOL_N) ? (uint64_t)(uintptr_t)&g_log_pool[i + 1][0] : 0;
    memset(g_log_obj, 0, sizeof g_log_obj);
    g_log_obj[0x10 / 8] = 0;                                              // 计数
    g_log_obj[0x28 / 8] = LOG_POOL_N;                                     // 上限
    g_log_obj[0x18 / 8] = (uint64_t)(uintptr_t)&g_log_pool[0][0];         // 链表头（非空）
    *(uint64_t *)(img + 0x2288548) = (uint64_t)(uintptr_t)&g_log_obj[0];
}
// 注：早期"把分配器入口直接 b 到宿主函数"的做法（k_patch_allocators）已删除——
// 它只能打 ±128MB 内的直接跳转，且随后就被下面的「替换式插桩」覆盖（同一批站点），
// 日志里的"已接到宿主 alloc"与实际执行路径不符，留着只会误导。

// ★ 数据导入补值（2026-10-18）：main 的动态符号表里有 13 个 nn::util 数学常量 + std::nothrow
//   是**未定义导入**（GLOB_DAT 重定位，type 1025）。Switch 上由运行时链接器从 SDK 模块
//   （sdk/subsdk*，本环境没加载）解析进槽；我们的探针对非 RELATIVE"只计数不解析"，
//   槽留文件原值 0。游戏代码的用法是 `ldr x8,[槽]; ldr s0,[x8]`——**槽 0 = 解引用空指针崩**。
//   实证：静态构造器 #184（`*(0x22d84a0)=2π`）就是这么崩的（读 nn::util::FloatPi 的槽）。
//   这里把**数学事实型标量**按 IEEE754 二进制位补上（槽指向宿主静态量）；std::nothrow 给
//   1 字节哑对象（只当标签传地址）。
//   ⚠ 仍留 0 的（登记在案的地雷，内容未知无法编造）：
//     - detail::SinCoefficients / CosCoefficients / AtanCoefficients（多项式系数表）
//     - detail::SinCosSampleTable（正弦采样表）、detail::AngleIndexHalfRound（表半周长）
//     - detail::FloatQuaternionEpsilon（四元数 epsilon，量级约 1e-5/1e-6）
//     - glslc* ×8（弱符号，语义上就该是 0——可选着色器编译器，客体自行空判）
//     - __rel_dyn_start/end、__rel_plt_start/end（链接器边界符号：填了反而可能诱使游戏
//       去"自遍历重定位表"重复应用重定位，保持 0 与既往实测行为一致）
static float g_c_pi, g_c_2pi, g_c_pi2, g_c_inv2pi, g_c_deg180, g_c_ulp;
static uint8_t g_c_nothrow;
static void k_fill_data_imports(uint8_t *img) {
    // 位模式（Python struct 核过）：π=0x40490FDB 2π=0x40C90FDB π/2=0x3FC90FDB
    // 1/(2π)=0x3E22F983 180.0=0x43340000 ULP(2^-23)=0x34000000
    uint32_t pi = 0x40490FDBu, pi2 = 0x3FC90FDBu, tpi = 0x40C90FDBu;
    uint32_t inv2pi = 0x3E22F983u, d180 = 0x43340000u, ulp = 0x34000000u;
    __builtin_memcpy(&g_c_pi, &pi, 4);      __builtin_memcpy(&g_c_2pi, &tpi, 4);
    __builtin_memcpy(&g_c_pi2, &pi2, 4);    __builtin_memcpy(&g_c_inv2pi, &inv2pi, 4);
    __builtin_memcpy(&g_c_deg180, &d180, 4); __builtin_memcpy(&g_c_ulp, &ulp, 4);
    g_c_nothrow = 0;
    struct { uint32_t slot; void *p; } tab[] = {
        { 0x2280648, &g_c_pi      },   // nn::util::FloatPi
        { 0x2280528, &g_c_pi      },   // nn::util::detail::FloatPi
        { 0x2280548, &g_c_2pi     },   // nn::util::detail::Float2Pi
        { 0x2280550, &g_c_pi2     },   // nn::util::detail::FloatPiDivided2
        { 0x2280530, &g_c_inv2pi  },   // nn::util::detail::Float1Divided2Pi
        { 0x2280460, &g_c_deg180  },   // nn::util::detail::FloatDegree180
        { 0x2280650, &g_c_ulp     },   // nn::util::detail::FloatUlp
        { 0x2264110, &g_c_nothrow },   // std::nothrow（1 字节哑对象）
    };
    for (unsigned i = 0; i < sizeof tab / sizeof tab[0]; i++)
        *(uint64_t *)(img + tab[i].slot) = (uint64_t)(uintptr_t)tab[i].p;
    klog("[数据导入] 补 %u 个标量常量槽（π/2π/π/2/1/(2π)/180/ULP + nothrow）\n",
         (unsigned)(sizeof tab / sizeof tab[0]));
}

// ===== 洞内插桩：安装器 =====
// 把 hook_stub.S 的桩整段复制进代码洞，填三个槽（原指令 / 回跳 / 宿主函数地址），
// 再把目标函数首指令改成 4 字节 b 跳进桩。全部在 seal 之前做。
//
// ⚠ 桩保存帧的**下标约定**（与 hook_stub.S 的 stp 偏移一一对应；改桩必须同步改这里）：
//     0x00..0x88 = x0..x17 → f[0..17]   0x90 = nzcv → f[18]   0x98 = 未写入（padding）
//     0xA0 = x29 → f[20]   0xA8 = x30 → f[21]   0xB0 = 站点地址 → f[22]
//   实测方式：clang -D__OHOS__ -c hook_stub.S + llvm-objdump -d（2026-10-16 实证；
//   此前按 "f[20]=x30" 读是错的——f[20] 是 x29，f[19] 根本没写）。
// 帧下标（= 偏移/8，布局见 hook_stub.S 顶部注释）。
// np125 起 x18–x28 也存了：想看"这个对象是谁"（x19/x23）不用再重编重跑。
enum { FR_X0 = 0, FR_X1 = 1, FR_X2 = 2, FR_X3 = 3, FR_X8 = 8,
       FR_NZCV = 18, FR_X29 = 20, FR_X30 = 21, FR_SITE = 22,
       FR_X18 = 23, FR_X19 = 24, FR_X20 = 25, FR_X21 = 26, FR_X22 = 27,
       FR_X23 = 28, FR_X24 = 29, FR_X25 = 30, FR_X26 = 31, FR_X27 = 32, FR_X28 = 33 };
// 桩长实测约 0x1A0（加了 x18–x28 的存档/恢复后；objdump 量），步长取 0x1C0 对齐。
// 洞容量：0x10000 / 0x1C0 = **146 格**（原来 227）。够用：当前记录式钩子约 80 个。
#define HOOK_STRIDE 0x1C0u
#define HOOK_MAX    (DM_IMG_MAIN_CAVE_SIZE / HOOK_STRIDE)
extern const char hook_stub_begin[], hook_stub_end[], hook_slot_orig[], hook_slot_back[], hook_slot_host[], hook_slot_site[];

// ───────────────────────── trace 环缓冲（np125） ─────────────────────────
// 为什么：一次上机只回答一个是非题，是今天最大的时间浪费（15 轮 / 111 分钟）。
// 做法：在几十个关键点挂"只写内存、不打日志"的钩子（复用同一个桩，宿主函数换成 ktrace_rec），
//   一次跑测带回**完整调用时间线**；跑完把环缓冲落成 dm_trace.bin，离线用 trace_view.py 分析。
// 为什么快：klog 每条 ~10 µs（格式化 + 文件锁 + 写盘），环缓冲一条 ~50 ns（一条原子加 + 三次写）。
//   同样的点位密度，日志方式会把游戏拖慢几个数量级，环缓冲几乎无感。
#define KTRACE_N   (1u << 16)     // 6.5 万条 ≈ 1.5 MB
typedef struct { uint32_t off; uint32_t tid; uint64_t a0; uint64_t a1; } ktrace_ent;

static ktrace_ent *g_tr;                 // 首次用时 calloc（不进信号上下文）
static unsigned g_tr_n;                  // 已写条数（原子递增；超容量后丢）
static unsigned g_tr_lost;
static void ktrace_maybe_snapshot(void); // 前置：水位落盘（定义在下面，ktrace_rec 先用到）

static uint64_t ktrace_rec(uint64_t *f) {
    // ⭐ np175（2026-10-21）：**空闲环过滤**。真机实证（np173/np174）：环 65536 格被单个空转线程
    //   占满（65526/65536），它循环的四条全是"参数为 0 的空闲调用"：
    //     `+0xa01d70` → `+0xa01fe0` → `ConvertToTimeSpan(0)` → `SleepThread(0)`
    //   ⇒ 主线程停车前的真实调用被**整体冲掉**，轨迹只能看到空转。
    //   过滤条件刻意收窄：**站点匹配 且 x0==0** 才丢，真业务调用（参数非 0）照记。
    {
        uint64_t site = f[FR_SITE] - (uint64_t)(uintptr_t)g_k_m.base;
        // np175b：实测这两条即使 a0≠0 也是空闲环（a0 = 任务堆管理器），直接按站点过滤。
        if (site == 0xa01d70ull || site == 0xa01fe0ull) return 0;
    }
    unsigned i = __atomic_fetch_add(&g_tr_n, 1u, __ATOMIC_RELAXED);
    if (i < KTRACE_N && g_tr) {
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        g_tr[i].off = (uint32_t)(f[FR_SITE] - base);
        g_tr[i].tid = (uint32_t)(uintptr_t)pthread_self();
        g_tr[i].a0  = f[FR_X0];
        g_tr[i].a1  = f[FR_X1];
    } else {
        __atomic_fetch_add(&g_tr_lost, 1u, __ATOMIC_RELAXED);
    }
    ktrace_maybe_snapshot();
    // np178b：**由任意"非主线程"的导入调用驱动采样**（主线程自己不投）。
    //   为什么：tick 路径的 ConvertToTimeSpan 其实是主线程自己在调（np178 实证：采样归零），
    //   而 worker 线程的 ReceiveMessageQueue 高频（万次级）⇒ 用它当采样时基。
    if (!(i & 31u)) smp_kick();
    return 0;
}

// np125b：**中途落盘**。原设计只在 nnMain 返回/崩溃时落盘，但本轮的卡点是"阻塞"——
//   游戏既不返回也不崩，数据永远拿不到（真机实证：脚本报"轨迹没抓到"）。
//   改成按水位落盘：每 +500 条落一次（后面的覆盖前面的）。
//   这样无论跑多久、是否阻塞，总能拿到"到目前最大"的那份。
// np138（2026-10-20 下午）：粒度 2000 → **500**。主线程静默卡死定位时，旧 2000 粒度
//   最后一滴可能离冻点上千条调用（live 跑冻点就在 4000 水位后十几行）；500 让冻点更近。
//   代价 = 落盘更频（每次 fopen/fwrite/fclose，burst 期约每几毫秒一次）——可接受：
//   冻点定位的主仪器是等待族打点（零扰动），轨迹只是保险。
static void ktrace_dump(const char *files_dir);
static void ktrace_maybe_snapshot(void) {
    static unsigned next = 500u;
    if (!g_files_dir[0]) return;
    unsigned n = __atomic_load_n(&g_tr_n, __ATOMIC_RELAXED);
    if (n < next || next > KTRACE_N) return;
    klog("[TRACE] 水位快照：已记录 %u 条（下一水位 %u）\n", n, next + 500u);
    next += 500u;
    ktrace_dump(g_files_dir);
}

// PLT 调用记一条：off 的高位置 1 表示"这是一次导入调用"，低 24 位是符号表下标
// （映像内偏移 < 0x2600000，不会撞）。离线用 dm_plt_names.txt 还原名字。
#define KTRACE_PLT_MARK 0x10000000u
static void ktrace_plt(int idx, uint64_t a0, uint64_t a1) {
    if (!g_tr) return;
    unsigned i = __atomic_fetch_add(&g_tr_n, 1u, __ATOMIC_RELAXED);
    if (!(i & 63u)) smp_kick();     // np178b：worker 线程的导入调用驱动主线程采样
    if (i < KTRACE_N) {
        g_tr[i].off = KTRACE_PLT_MARK | (uint32_t)(idx < 0 ? 0xFFFFFF : (idx & 0xFFFFFF));
        g_tr[i].tid = (uint32_t)(uintptr_t)pthread_self();
        g_tr[i].a0  = a0;
        g_tr[i].a1  = a1;
    } else {
        __atomic_fetch_add(&g_tr_lost, 1u, __ATOMIC_RELAXED);
    }
    ktrace_maybe_snapshot();
}

static void ktrace_reset(void) {
    if (!g_tr) g_tr = (ktrace_ent *)calloc(KTRACE_N, sizeof *g_tr);
    __atomic_store_n(&g_tr_n, 0u, __ATOMIC_RELAXED);
    __atomic_store_n(&g_tr_lost, 0u, __ATOMIC_RELAXED);
}

// np126：**guest 侧标记**。有些关键函数已经挂了记录式钩子（日志有上限、会静默），
//   在那些钩子里顺手记一条轨迹 ⇒ 同一个钩子既保留原日志，又能在离线时间线上看见"到过没、来几次"。
//   注意：不能再挂第二个钩子（同一站点会互相顶掉），所以走"钩子内部追加记一笔"这条路。
static void ktrace_mark(uint32_t off, uint64_t a0, uint64_t a1) {
    if (!g_tr) return;
    unsigned i = __atomic_fetch_add(&g_tr_n, 1u, __ATOMIC_RELAXED);
    if (i < KTRACE_N) {
        g_tr[i].off = off;
        g_tr[i].tid = (uint32_t)(uintptr_t)pthread_self();
        g_tr[i].a0  = a0;
        g_tr[i].a1  = a1;
    } else {
        __atomic_fetch_add(&g_tr_lost, 1u, __ATOMIC_RELAXED);
    }
    // np185 实验 G：**采样 kick 也走这里**。np175 把 G/runner 从 ktrace_rec 过滤掉后，
    //   空闲线程（唯一确定被调度的非主线程）不再产生 kick ⇒ 新进程里 SIGPROF 无人投、
    //   样本恒 0（20:13 轮实证）。G/runner 经 hook_e8/hook_e11 → ktrace_mark 入环，
    //   在这里补 kick 即恢复"空闲环当时基"的旧行为。
    if (!(i & 63u)) smp_kick();
}

// np127：**定时落盘**（宿主线程，每 10 秒一次，共 4 次）。
//   为什么需要：按"条数水位"落盘有个死角——游戏卡住后不再产生新记录，最后一个水位之前
//   那几百条恰好是"卡死前的最后动作"，却永远落不下来（本轮真机：文件停在 2000 条，
//   而实际卡在 2000~4000 之间）。定时落盘按**时间**取样，卡住也能拿到最新的一份。
//   宿主 pthread 能被调度（探针线程本身就是），这里只做文件写，不碰客体状态。
static void *ktrace_timer(void *arg) {
    (void)arg;
    // ⚠ np136 实证：宿主 pthread_create 的线程体从未被调度（仪器铁律，np26–np29 三回合
    //   零输出）⇒ 这个定时器实际不会跑。保留仅为将来若调度问题修复后自动生效；
    //   当前周期打点全部挂在游戏帧线程（plt_stub 服务分派里 ConvertToTimeSpan tick）。
    for (int i = 0; i < 4; i++) {
        struct timespec ts = { 10, 0 };
        nanosleep(&ts, NULL);
        if (g_files_dir[0]) ktrace_dump(g_files_dir);
    }
    return NULL;
}

// 落盘：头部 32 字节（魔数/条数/丢失数/基址），随后是原始条目。离线用 tools/trace_view.py 读。
//
// ⚠⚠ np147/np148 真机缺陷（2026-10-20 晚修复，两轮实证）：
//   原实现"每次 fopen('wb') 全量重写整个环"——np147 47001 条、np148 48000 条时，
//   线程进 ktrace_dump 后**卡死在 fwrite 里再没出来**（日志最后一行 = 落盘点、
//   文件 0 字节、mtime = fopen 截断时刻）。旁证：符号表 19KB 多次小写能成、
//   日志 460KB 增量写能成、唯独 trace 的 ~1.1MB 单次写卡死 ⇒ 大写入在此环境不可靠。
// 新版 = **增量 append**（对齐已被证明可行的小写入范式）：
//   ① 首次 dump：建文件 + 写 32B 头（n 字段先写 0）；
//   ② 每次水位：只 fwrite **新增条目**（≤500 条 ≈ 12KB）+ 在原地把头部 n 改成已写条数；
//   ③ **永不截断、永不 rename** ⇒ 即使中途卡死，已写入部分完整保留；hdc file recv
//      随时可抓、不与 dump 互斥；冻点永远在文件尾（ trace_view.py --tail 即得）。
//   ⚠ 环绕回（n > KTRACE_N）后停止追加：文件只覆盖前 65536 条（早期窗口），
//     真机冻结都发生在 48000 条内，够用；将来要滚动窗口再议。
static pthread_mutex_t g_ktrace_dump_mu = PTHREAD_MUTEX_INITIALIZER;
static FILE     *g_tr_fp;        // 常开句柄（首dump 建，之后只 append）
static unsigned  g_tr_dumped;    // 已写入文件的条数

static void ktrace_dump(const char *files_dir) {
    unsigned n = __atomic_load_n(&g_tr_n, __ATOMIC_RELAXED);
    unsigned lost = __atomic_load_n(&g_tr_lost, __ATOMIC_RELAXED);
    if (n > KTRACE_N) n = KTRACE_N;                      // 环就 65536 条，线性取前 N
    klog("[TRACE] 落盘：%u 条（环容量 %u，丢弃 %u）\n", n, KTRACE_N, lost);
    if (!files_dir || !*files_dir || !g_tr) return;
    pthread_mutex_lock(&g_ktrace_dump_mu);
    char p[600];
    snprintf(p, sizeof p, "%s/dm_trace.bin", files_dir);
    if (!g_tr_fp) {
        g_tr_fp = fopen(p, "wb");
        if (!g_tr_fp) {
            pthread_mutex_unlock(&g_ktrace_dump_mu);
            klog("【红】[TRACE] 打开 %s 失败\n", p);
            return;
        }
        uint64_t hdr[4] = { 0x45434152544B4441ull /* "ADKT RACE" */, 0, 0,
                            (uint64_t)(uintptr_t)g_k_m.base };
        if (fwrite(hdr, sizeof hdr, 1, g_tr_fp) != 1) {
            fclose(g_tr_fp); g_tr_fp = NULL;
            pthread_mutex_unlock(&g_ktrace_dump_mu);
            klog("【红】[TRACE] 写头部失败 %s\n", p);
            return;
        }
        g_tr_dumped = 0;
        klog("[TRACE] 建轨迹文件 %s（增量 append 模式）\n", p);
    }
    unsigned add = n - g_tr_dumped;
    if ((int)add < 0) add = 0;
    if (add) {
        // ① 头部 n 原地更新（8 字节小写，offset 8）——保持 trace_view.py 格式不变
        uint64_t nn = (uint64_t)(g_tr_dumped + add);
        if (fseek(g_tr_fp, 8, SEEK_SET) != 0 ||
            fwrite(&nn, sizeof nn, 1, g_tr_fp) != 1 ||
            // ② 追加新条目（走文件尾）
            fseek(g_tr_fp, 0, SEEK_END) != 0 ||
            fwrite(g_tr + g_tr_dumped, sizeof *g_tr, add, g_tr_fp) != add) {
            pthread_mutex_unlock(&g_ktrace_dump_mu);
            klog("【红】[TRACE] 追加失败（已写 %u + %u）\n", g_tr_dumped, add);
            return;
        }
        g_tr_dumped = n;
    }
    if (fflush(g_tr_fp) != 0) {
        pthread_mutex_unlock(&g_ktrace_dump_mu);
        klog("【红】[TRACE] fflush 失败（已写 %u 条）\n", g_tr_dumped);
        return;
    }
    pthread_mutex_unlock(&g_ktrace_dump_mu);
    if (add) klog("[TRACE] 追加 %u 条 → 共 %u 条已落盘\n", add, g_tr_dumped);
}

// 站点指令搬进洞：PC 相对指令按洞内位置重算；**adrp / adr / ldr(字面量) 物化**
//（绝对地址/常量装载期已知 ⇒ movz/movk 序列，最多占 3 条指令槽）；b.cond/cbz/tbz 等
// **超程**时改"取反 + 跳过 + 无条件跳"（imm26 射程 ±128MB，洞在映像内必达）。
// out[0..2] = 最多三条指令；返回 0 = 形态未知/仍超程（调用方跳过该站点，宁可少挂不可挂错）。
// ⚠ 2026-10-21 重写：旧版对 adrp/adr/ldr 字面量一律**拒绝挂载**（槽①只有 4 字节，
//   装不下物化序列）——历史上因此丢了 0xa01490 等一批站点。现在桩有 3 个指令槽
//   （hook_stub.S 的 slot_orig/orig2/orig3，默认 nop），物化序列放得下了。
static uint32_t enc_movz(uint32_t rd, uint16_t v, int hw) {   // movz xN, #v, lsl #(16*hw)
    return 0xD2800000u | ((uint32_t)hw << 21) | ((uint32_t)v << 5) | (rd & 31);
}
static uint32_t enc_movk(uint32_t rd, uint16_t v, int hw) {   // movk xN, #v, lsl #(16*hw)
    return 0xF2800000u | ((uint32_t)hw << 21) | ((uint32_t)v << 5) | (rd & 31);
}
// 64 位绝对地址 → movz/movk/movk 三条（高位为 0 时第三条是无害的 OR 0）
static void emit_addr64(uint64_t a, uint32_t rd, uint32_t out[3]) {
    out[0] = enc_movz(rd, (uint16_t)(a & 0xFFFF), 0);
    out[1] = enc_movk(rd, (uint16_t)((a >> 16) & 0xFFFF), 1);
    out[2] = enc_movk(rd, (uint16_t)((a >> 32) & 0xFFFF), 2);
}
// 32 位立即数 → movz/movk 两条（w 寄存器形态）
static void emit_val32(uint32_t v, uint32_t rd, uint32_t out[3]) {
    out[0] = 0x52800000u | ((uint32_t)(v & 0xFFFF) << 5) | (rd & 31);
    out[1] = 0x72A00000u | ((uint32_t)((v >> 16) & 0xFFFF) << 5) | (rd & 31);
    out[2] = 0xD503201Fu;                                     // nop
}
// 条件分支超程时的公共尾巴：out[0] 已放好"取反+跳过"，这里放 out[1] = b target
static int cond_ooc_tail(uint8_t *tgt, uint8_t *to0, uint32_t out[3]) {
    int64_t d = ((int64_t)(intptr_t)tgt - (int64_t)(intptr_t)(to0 + 4)) / 4;
    if (d < -(1 << 25) || d >= (1 << 25)) return 0;
    out[1] = 0x14000000u | ((uint32_t)d & 0x03FFFFFFu);
    return 1;
}
static int reloc_site(uint32_t w, uint8_t *from, uint8_t *to0, uint32_t out[3]) {
    out[1] = out[2] = 0xD503201Fu;                            // 默认后两槽 nop
    uint32_t op26 = w & 0xFC000000u;
    int32_t  imm; int64_t d;
    // ── adrp：目标页 = (from & ~0xFFF) + imm21<<12；物化成绝对页地址 ──
    if ((w & 0x9F000000u) == 0x90000000u) {
        uint32_t rd = w & 0x1Fu;
        int32_t imm21 = (int32_t)((((w >> 5) & 0x7FFFFu) << 2) | ((w >> 29) & 3u));
        if (imm21 & 0x100000) imm21 -= 0x200000;
        uint64_t page = ((uint64_t)(uintptr_t)from & ~0xFFFull) + ((int64_t)imm21 << 12);
        emit_addr64(page, rd, out);
        return 1;
    }
    // ── adr：目标 = from + imm21；物化成绝对地址 ──
    if ((w & 0x9F000000u) == 0x10000000u) {
        uint32_t rd = w & 0x1Fu;
        int32_t imm21 = (int32_t)((((w >> 5) & 0x7FFFFu) << 2) | ((w >> 29) & 3u));
        if (imm21 & 0x100000) imm21 -= 0x200000;
        emit_addr64((uint64_t)(uintptr_t)from + imm21, rd, out);
        return 1;
    }
    // ── ldr (literal)：值在映像字面量池里、.text 只读 ⇒ 装载期读出来直接物化成立即数 ──
    //   （SIMD 形态 sN/dN 搬不进 GPR，明确拒绝）
    if ((w & 0x3B000000u) == 0x18000000u) {
        uint32_t opc = (w >> 30) & 3u;
        uint32_t rd = w & 0x1Fu;
        if (opc >= 2u) return 0;                              // sN/dN：不接
        int32_t imm19 = (int32_t)((w >> 5) & 0x7FFFFu);
        if (imm19 & 0x40000) imm19 -= 0x80000;
        const uint8_t *lit = from + imm19 * 4;
        if (opc == 0u) {                                      // wN ← 32 位常量
            uint32_t v;
            memcpy(&v, lit, 4);
            emit_val32(v, rd, out);
        } else {                                              // xN ← 64 位常量
            uint64_t v;
            memcpy(&v, lit, 8);
            emit_addr64(v, rd, out);
        }
        return 1;
    }
    if (op26 == 0x14000000u || op26 == 0x94000000u) {                            // b / bl
        imm = (int32_t)(w & 0x03FFFFFFu);
        if (imm & 0x02000000) imm -= 0x04000000;
        d = ((int64_t)(intptr_t)from + (int64_t)imm * 4 - (int64_t)(intptr_t)to0) / 4;
        if (d < -(1 << 25) || d >= (1 << 25)) return 0;
        out[0] = op26 | ((uint32_t)d & 0x03FFFFFFu);
        return 1;
    }
    // ── 条件分支：先按原形态重算；超程改"取反 + 跳过 + 无条件跳" ──
    if ((w & 0xFF000010u) == 0x54000000u) {                    // b.cond
        imm = (int32_t)((w >> 5) & 0x7FFFFu);
        if (imm & 0x40000) imm -= 0x80000;
        uint8_t *tgt = from + imm * 4;
        d = ((int64_t)(intptr_t)tgt - (int64_t)(intptr_t)to0) / 4;
        if (d >= -(1 << 18) && d < (1 << 18)) {
            out[0] = (w & ~(0x7FFFFu << 5)) | (((uint32_t)d & 0x7FFFFu) << 5);
            return 1;
        }
        out[0] = 0x54000000u | (2u << 5) | ((w & 0xFu) ^ 1u);  // b.!cond +8
        return cond_ooc_tail(tgt, to0, out);
    }
    if ((w & 0x7E000000u) == 0x34000000u) {                    // cbz/cbnz
        imm = (int32_t)((w >> 5) & 0x7FFFFu);
        if (imm & 0x40000) imm -= 0x80000;
        uint8_t *tgt = from + imm * 4;
        d = ((int64_t)(intptr_t)tgt - (int64_t)(intptr_t)to0) / 4;
        if (d >= -(1 << 18) && d < (1 << 18)) {
            out[0] = (w & ~(0x7FFFFu << 5)) | (((uint32_t)d & 0x7FFFFu) << 5);
            return 1;
        }
        out[0] = ((w ^ 0x01000000u) & ~(0x7FFFFu << 5)) | (2u << 5);   // 取反（cbz↔cbnz）+8
        return cond_ooc_tail(tgt, to0, out);
    }
    if ((w & 0x7E000000u) == 0x36000000u) {                    // tbz/tbnz
        imm = (int32_t)((w >> 5) & 0x3FFFu);
        if (imm & 0x2000) imm -= 0x4000;
        uint8_t *tgt = from + imm * 4;
        d = ((int64_t)(intptr_t)tgt - (int64_t)(intptr_t)to0) / 4;
        if (d >= -(1 << 13) && d < (1 << 13)) {
            out[0] = (w & ~(0x3FFFu << 5)) | (((uint32_t)d & 0x3FFFu) << 5);
            return 1;
        }
        out[0] = ((w ^ 0x01000000u) & ~(0x3FFFu << 5)) | (2u << 5);   // 取反（tbz↔tbnz）+8
        return cond_ooc_tail(tgt, to0, out);
    }
    out[0] = w;
    return 1;
}
static int g_hook_n;
static uint32_t *slot_of(uint8_t *dst, const char *b0) {
    return (uint32_t *)(dst + (hook_slot_orig - b0));
}
static void k_install_hook(uint8_t *img, uint32_t site_off, void *host_fn, const char *what) {
    const char *b0 = hook_stub_begin;
    size_t n = (size_t)(hook_stub_end - hook_stub_begin);
    if (n > HOOK_STRIDE) { klog("[插桩] 桩太长(%zu > %u)，跳过 %s\n", n, HOOK_STRIDE, what); return; }
    if (g_hook_n >= (int)HOOK_MAX) { klog("[插桩] 洞已满(%d 格)，跳过 %s\n", g_hook_n, what); return; }
    uint8_t *dst = img + DM_IMG_MAIN_CAVE_OFF + (size_t)g_hook_n * HOOK_STRIDE;
    memcpy(dst, b0, n);
    // 槽①+①b+①c：原指令（1–3 条）。PC 相对的分支（bl/b/b.cond/cbz/tbz）搬进洞后必须按
    //   新位置重算位移；adrp/adr/ldr(字面量) 物化成 movz/movk 序列（2026-10-21 起不再拒绝）。
    {
        uint32_t w = *(uint32_t *)(img + site_off), nw[3] = {0, 0, 0};
        uint8_t *to0 = dst + (hook_slot_orig - b0);
        int rr = reloc_site(w, img + site_off, to0, nw);
        if (rr == 0) { klog("[插桩] %s 站点指令 %08x 重定位超程/形态未知，跳过\n", what, w); return; }
        *(uint32_t *)to0 = nw[0];
        *(uint32_t *)(to0 + 4) = nw[1];
        *(uint32_t *)(to0 + 8) = nw[2];
    }
    // 槽②：b site+4（相对编码）
    {
        uint8_t *p = dst + (hook_slot_back - b0);
        int64_t d = ((int64_t)(uintptr_t)(img + site_off + 4) - (int64_t)(uintptr_t)p) / 4;
        if (d < -(1 << 25) || d >= (1 << 25)) { klog("[插桩] %s 回跳超程，跳过\n", what); return; }
        *(uint32_t *)p = 0x14000000u | (uint32_t)(d & 0x03FFFFFFu);
    }
    // 槽③：宿主函数地址；槽④：站点地址（日志据此识别调用点）
    *(uint64_t *)(dst + (hook_slot_host - b0)) = (uint64_t)(uintptr_t)host_fn;
    *(uint64_t *)(dst + (hook_slot_site - b0)) = (uint64_t)(uintptr_t)(img + site_off);
    // 站点：b dst
    {
        int64_t d = ((int64_t)(uintptr_t)dst - (int64_t)(uintptr_t)(img + site_off)) / 4;
        if (d < -(1 << 25) || d >= (1 << 25)) { klog("[插桩] %s 站点跳超程，跳过\n", what); return; }
        *(uint32_t *)(img + site_off) = 0x14000000u | (uint32_t)(d & 0x03FFFFFFu);
    }
    klog("[插桩] %s：site %#x → 洞 %#zx（洞内首指令 %08x）\n", what, site_off,
         (size_t)(dst - img), *slot_of(dst, b0));
    g_hook_n++;
}
// 统一的桩日志：打印站点地址、入参（x0 = this）、以及 x29/x30（**x30 = "谁调我"**）。
// 帧下标见上面 FR_* 约定。每条目标各限 HOOK_LOG_CAP 行——崩点前的**最后一行**才是答案，
// 所以宁可多记也别截断；上限只是防止在环里刷爆日志。
#define HOOK_LOG_CAP 256
static void frame_log(const char *what, uint64_t *f, int *cnt) {
    if (++*cnt <= HOOK_LOG_CAP)
        klog("[桩] %s  站点 %#llx  x0=%#llx x1=%#llx x2=%#llx x29=%#llx x30=%#llx\n",
             what, (unsigned long long)f[FR_SITE], (unsigned long long)f[FR_X0],
             (unsigned long long)f[FR_X1], (unsigned long long)f[FR_X2],
             (unsigned long long)f[FR_X29], (unsigned long long)f[FR_X30]);
}
static int g_n_ctor1, g_n_ctor2, g_n_any, g_n_9fe24c, g_n_9fe2e8, g_n_1e484, g_n_1dd94, g_n_1d8e0, g_n_s1d8e0;
static uint64_t hook_ctor_log (uint64_t *f) { frame_log("构造器 0xa08ef4", f, &g_n_ctor1); return 0; }
static uint64_t hook_ctor2_log(uint64_t *f) { frame_log("构造器 0xa08e94", f, &g_n_ctor2); return 0; }
static uint64_t hook_any_site (uint64_t *f) { frame_log("站点→0xa08e94", f, &g_n_any);   return 0; }
// ★ 沿崩链逐点插桩。链（离线用 tools/dm_branch.py 追出，文档里此前没有这几层）：
//   函数 0x9fe24c ──cbz w9 走 0x9fe294 分支──▶ 0x9fe2e8 `bl 0xa1e484`（x0 = **(0x22783d8)，
//   正是我们注入的线程上下文对象）──▶ 0xa1e484 是 `b 0xa1dd94` 尾调用桩 ──▶
//   0xa1dd94 ──`bl 0x9f33f0` 分配 0x100 字节──▶ ──`bl 0xa1d8e0`──▶ 0xa1d8e0 ──`bl 0xa08e94`
//   （基类构造器，this 直接沿用 x0，中间一条指令都没改 x0）──▶ 崩在 0xa08eb0 的
//   `stp x8, xzr, [x19]`（x19 = x0 = this ⇒ this=0 就是往地址 0 写）。
//   逐点记 x0（上一层交下来的对象）与 x30（谁调我）⇒ **哪一层把 0 交下去，就是"该建而未建"的位置**。
static uint64_t hook_p9fe24c (uint64_t *f) { frame_log("链点 0x9fe24c", f, &g_n_9fe24c); return 0; }
static uint64_t hook_p9fe2e8 (uint64_t *f) { frame_log("站点→0xa1e484", f, &g_n_9fe2e8); return 0; }
static uint64_t hook_p1e484 (uint64_t *f) { frame_log("链点 0xa1e484", f, &g_n_1e484); return 0; }
static uint64_t hook_p1dd94 (uint64_t *f) { frame_log("链点 0xa1dd94", f, &g_n_1dd94); return 0; }
static uint64_t hook_p1d8e0 (uint64_t *f) { frame_log("链点 0xa1d8e0", f, &g_n_1d8e0); return 0; }
static uint64_t hook_s1d8e0 (uint64_t *f) { frame_log("站点→0xa1d8e0", f, &g_n_s1d8e0); return 0; }
// ★ sead 单例查找 0xa0913c 为什么返回 0？把它的几条关键分支路径都挂上点，命中哪条就是走了哪条。
//   （它是中心设施，103 个调用点；出口 0xa09418 返回 x20。这几条都是条件分支的**目标**，
//     挂目标安全；直接挂条件分支指令本身会让"被命中时又跳回站点"成死循环，不做。）
static int g_n_lkA, g_n_lkB, g_n_lkC, g_n_lkD, g_n_lkE;
static uint64_t hook_lk_a(uint64_t *f) { frame_log("查找A(x24==0)", f, &g_n_lkA); return 0; }
static uint64_t hook_lk_b(uint64_t *f) { frame_log("查找B(走分配)", f, &g_n_lkB); return 0; }
static uint64_t hook_lk_c(uint64_t *f) { frame_log("查找C(非2幂→返0)", f, &g_n_lkC); return 0; }
static uint64_t hook_lk_d(uint64_t *f) { frame_log("查找D(返0:x19==0)", f, &g_n_lkD); return 0; }
static uint64_t hook_lk_e(uint64_t *f) { frame_log("查找E(返0:分配败)", f, &g_n_lkE); return 0; }
// ★ 再细一层：0xa0e2f8 为什么返回 0？
//   它内部 `0xa0e32c ldr x0,[x8]` → `0xa0e330`；在 0xa0e330 挂点可同时看到
//   **读的地址(x8)** 与 **读出的值(x0=返回值)**。再在调用点 0xa09184 挂点看传进去的 obj 是谁。
static int g_n_dbA, g_n_dbB;
static uint64_t hook_dbA(uint64_t *f) {   // 站点 0xa09184（bl 0xa0e2f8）
    if (++g_n_dbA <= 4)
        klog("[诊断] 调 0xa0e2f8：obj=%#llx（期望 = 注入的 g_alloc_obj）\n",
             (unsigned long long)f[FR_X0]);
    return 0;
}
static uint64_t hook_dbB(uint64_t *f) {   // 指令 0xa0e330（0xa0e2f8 的取值点之后）
    if (++g_n_dbB <= 4)
        klog("[诊断] 0xa0e2f8 读 [x8]=%#llx → 值 x0=%#llx（=返回值）\n",
             (unsigned long long)f[8], (unsigned long long)f[FR_X0]);
    return 0;
}
// ★ 崩在 `blr` 野指针（0xb9402be814000001，sig=7 取指失败）。定位到：
//   0xa0c1ac add x0, x22, #0x98 ; 0xa0c1b0 bl 0xa1d57c（sead 助手）
// 助手会把入参当多态对象用（读 [obj] 当虚表、再读 [+0x80]）。而 x22 是个**栈地址**
// ⇒ 怀疑 [x22+0x98] 那个字段本该是指针、却是栈上垃圾。在 0xa0c1b0 挂点，
// 把递过去的 obj 与 *(obj)（助手眼里的"虚表"）都打出来。
// ★ 崩在 `blr` 野指针（0xb9402be814000001，sig=7 取指失败）。探点挂在 0xa0c1b0 时没命中，
//   且现场 x30=0xa0c1b0 与"bl 的 LR=PC+4"对不上 ⇒ 崩的那次调用不是那一处。
//   两处都留着：0xa0c1b0（bl 之前）与 0xa1d57c（sead 助手入口）。
static int g_n_dbC, g_n_dbD;
static uint64_t hook_dbC(uint64_t *f) {   // 站点 0xa0c1b0（bl 0xa1d57c 之前）
    if (++g_n_dbC <= 4) {
        uint64_t obj = f[FR_X0];
        uint64_t vt = 0;
        if (obj) vt = *(const uint64_t *)(uintptr_t)obj;
        klog("[诊断] 0xa0c1b0 调 sead 助手：obj=%#llx  *(obj)=%#llx\n",
             (unsigned long long)obj, (unsigned long long)vt);
    }
    return 0;
}
static uint64_t hook_dbD(uint64_t *f) {   // 指令 0xa1d57c（sead 助手入口）
    if (++g_n_dbD <= 6) {
        uint64_t obj = f[FR_X0];
        uint64_t vt = 0;
        if (obj) vt = *(const uint64_t *)(uintptr_t)obj;
        klog("[诊断] sead 助手 0xa1d57c 被调：obj=%#llx x1=%#llx  *(obj)=%#llx  x30=%#llx\n",
             (unsigned long long)obj, (unsigned long long)f[FR_X1],
             (unsigned long long)vt, (unsigned long long)f[FR_X30]);
    }
    return 0;
}
// ===== 钉子 #8：类型分发函数 0xa00a78 收到空类型标签 ⇒ 返 0 ⇒ 调用方 0xa01790 写 [x21+0xc8] 崩 =====
// 离线结论（2026-10-17，main.patched.bin 指令级核实 + 真机日志闭环，见进度文档第十三节）：
//   0xa00a78 按 [x0] 的低 32 位标签分发：1 → *(*(0x2280a20))（.bss 0x22d8718，全零 ⇒ cbz 返 0）；
//   2 → 直接调 结构体[8]（Path A 里 = *(0x22809c0) = 函数 0x9fe910，有效代码）；
//   3 → *(*(0x227b058))（.bss 0x22d8720，全零 ⇒ 返 0）；其他 → 返 0（复现崩溃）。
//   调用方 0xa01790 用 arg1 的 [0]/[8]/[0x80] 拼栈上结构体喂给它，两个直接调用点：
//   0xa01490（Path A：函数 0x9fe82c 把标签**硬编码为 2**）与 0xa01f1c（Path B：任务链表节点）。
//   **真机定性（2026-10-17 傍晚）**：x30=0xA01F1C ⇒ 走 Path B；标签=0 ⇒ 返 0 ⇒ x21=0 崩。
//   **根因**：池节点里的记录是 mgr 构造函数 0xa00afc 从 *(父+0x10) 拷来的**全零副本**
//   （0xa00cf8 memcpy + 0xa00d48 池注册）；直创路径（mgr 虚表第 23 槽 0x9fe82c）从未被
//   调用（e1 实测 0 命中）。任务身份铁证 = sead::ControllerMgr：标签 2 工厂 *(0x22809c0)
//   = 0x9fe910（alloc 0x198 + 0xa1f07c 构造，基类 0x9fd9fc 名字串 0x1ba6eab 且不读 pair）。
static int g_n_e1, g_n_e2, g_n_e3, g_n_e4, g_n_e5;
static uint64_t g_obs_task;   // np113：状态迁移观察者——上次看到的任务
static uint32_t g_obs_state;  // np113：该任务上次的状态
static int g_n_e2fix;      // 影子补丁触发次数（标签 0 ⇒ 补 {2, ControllerMgr 工厂}）
static int g_n_e6, g_n_e7, g_n_e8;   // 同轮探针：工厂/池注册/G 三点运行时状态
static int g_n_e8fix;                // 状态 1→2 补推次数（G 激活路径的放行条件）
static int g_n_e9, g_n_e10;          // runner 探针：激活检查 0xa01cf0 / 运行转发器 0x9fdccc
static int g_n_e11;                  // runner 入口链表遍历探针（0xa01fe0）
static uint64_t hook_e1(uint64_t *f) {   // Path A 函数入口 0x9fe82c（标签硬编码 2 的那个）
    if (++g_n_e1 <= 8)
        klog("[钉子8] PathA 0x9fe82c 入口：x0=%#llx x1=%#llx x30=%#llx\n",
             (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X1],
             (unsigned long long)f[FR_X30]);
    return 0;
}
// ═══ np158：停车边界取证（全寄存器 + guest 栈）═══
// 背景：主线程每次都在"第 4 次 TaskHeap 插入"附近停车（np149/150/156/158 一致），
// 而停车点在纯 guest 原生代码里（PLT/svc/等待族全 0 命中），外部采样时钟不可靠
// ⇒ 改成**在最后一个已知客体动作的钩子上直取现场**：
//   ① guest SP 重构：hook_stub 从客体 SP 往下开 0x320 帧，f 即帧基址
//      ⇒ 客体 SP = (uint64_t)f + 0x320；栈上就是客体调用链的返回地址/局部量，
//      离线（plt 名表 + 反汇编）可把每个映像内值对回函数；
//   ② x0–x28 + x29/x30 全打出（this/arg/对象指针都在）；
//   ③ 触发点：hook_e2（0xa01790）第 4 次命中 = 停车前最后一次 TaskHeap 插入。
#define HOOK_FRAME_SIZE 0x320u
static int g_stall_arm;   // >0 时随后每个 0xa00a78 命中都打一份现场

static void stall_dump(uint64_t *f, const char *tag) {
    uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
    uint64_t gsp = (uint64_t)(uintptr_t)f + HOOK_FRAME_SIZE;
    klog("[取栈] ★ %s：客体SP=%#llx\n", tag, (unsigned long long)gsp);
    for (int i = 0; i <= 17; i++)
        klog("[取栈]   x%d=%#llx\n", i, (unsigned long long)f[i]);
    klog("[取栈]   nzcv=%#llx x29=%#llx x30=%s+%#llx\n",
         (unsigned long long)f[FR_NZCV], (unsigned long long)f[FR_X29],
         (f[FR_X30] >= base && f[FR_X30] < base + 0x2600000) ? "映像+" : "外",
         (unsigned long long)((f[FR_X30] >= base && f[FR_X30] < base + 0x2600000)
                              ? f[FR_X30] - base : f[FR_X30]));
    for (int i = FR_X18; i <= FR_X28; i++)
        klog("[取栈]   x%d=%#llx\n", i - 5, (unsigned long long)f[i]);
    for (int i = 0; i < 36; i++) {
        uint64_t v = *(const uint64_t *)(uintptr_t)(gsp + (uint64_t)i * 8);
        int in = (v >= base && v < base + 0x2600000);
        klog("[取栈]   [sp+#%#04x] = %#llx%s\n", i * 8, (unsigned long long)v,
             in ? " ←main 映像内" : "");
    }
}

static uint64_t hook_e2(uint64_t *f) {   // 分发调用方入口 0xa01790
    if (++g_n_e2 <= 16) {
        uint64_t a1 = f[FR_X1];
        uint64_t tag = 0;
        if (a1) tag = *(const uint64_t *)(uintptr_t)a1;   // arg1[0]：低 32 位是类型标签
        klog("[钉子8] 0xa01790：this=%#llx arg1=%#llx arg1[0]=%#llx arg2=%#llx x30=%#llx\n",
             (unsigned long long)f[FR_X0], (unsigned long long)a1,
             (unsigned long long)tag, (unsigned long long)f[FR_X2],
             (unsigned long long)f[FR_X30]);
    }
    // np158：第 4 次插入 = 停车边界 ⇒ 取全寄存器 + guest 栈，并给随后每个 0xa00a78
    // 命中武装（3×第17槽 序列就挂在它后面）。
    if (0 && g_n_e2 == 4) {   // np167：停用（突发写风险，现场已由死亡笔记承担）
        g_stall_arm = 4;
        g_s17_arm = 3;   // 随后 3 个第17槽 = 最后的客体动作，第 3 份即停车边界
        stall_dump(f, "第4次TaskHeap插入入口 0xa01790");
    }
    // ★ 钉子 #8 影子补丁（2026-10-17 傍晚，根因见上方注释与进度文档第十三节）：
    //   记录全零 ⇒ 标签 0 ⇒ 分发器返 0 ⇒ x21=0 ⇒ str [x21+0xc8] 崩。标签 0 不可能是合法
    //   调用（合法标签只有 1/2/3，且 1/3 路在鸿蒙上也是返 0），只可能是"该填的没填"。
    //   宿主补影子：补成 {标签=2, 工厂=base+0x9fe910}——与 Path A 直创时 Record::Init
    //   ({2, *(0x22809c0)}) 的产物逐字节同构（0x9fe910 = alloc 0x198 + 0xa1f07c 构造
    //   sead::ControllerMgr；基类 0x9fd9fc 不读 pair，其余字段全零无害）。
    //   写在记录本体上（持久）：F 随后读到的就是修好的标签/工厂，任务创建后 node[0] 有值，
    //   G 不会再进惰性路径；正常记录（标签≠0）零影响。
    {
        uint64_t a1 = f[FR_X1];
        if (a1 && *(const uint64_t *)(uintptr_t)a1 == 0) {
            uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
            uint64_t fn = base + 0x9fe910;
            *(uint64_t *)(uintptr_t)a1 = 2;
            *(uint64_t *)(uintptr_t)(a1 + 8) = fn;
            if (++g_n_e2fix <= 4)
                klog("[钉子8] 影子：记录 %#llx 标签 0 ⇒ 补 {2, %#llx}（ControllerMgr 工厂）\n",
                     (unsigned long long)a1, (unsigned long long)fn);
        }
    }
    return 0;
}
static uint64_t hook_e3(uint64_t *f) {   // 分发函数入口 0xa00a78
    if (++g_n_e3 <= 16) {
        uint64_t st = f[FR_X0];
        uint64_t tag = 0;
        if (st) tag = *(const uint32_t *)(uintptr_t)st;   // 结构体[0] 低 32 位 = 标签
        klog("[钉子8] 0xa00a78：结构体=%#llx 标签=%llu x1=%#llx\n",
             (unsigned long long)st, (unsigned long long)(uint32_t)tag,
             (unsigned long long)f[FR_X1]);
        // np185 实验 A：kind=2 路径是 `x2=[结构体+8]; br x2`——[结构体+8] 就是**运行时工厂
        //   函数指针**（np185 离线定谳：任务就绪戳 [task+0x60] bit1|bit2 在全二进制无写入者，
        //   置位者只可能是这类运行时派发）。打出来换算映像偏移，离线即可反汇编找写入者。
        if (st) {
            uint64_t fn = *(const uint64_t *)(uintptr_t)(st + 8);
            uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
            char l[176];
            snprintf(l, sizeof l, "派发#%d 0xa00a78 标签=%llu 工厂指针=%s%#llx x1=%#llx\n",
                     g_n_e3, (unsigned long long)(uint32_t)tag,
                     (fn >= base && fn < base + DM_IMG_MAIN_END) ? "img+" : "外",
                     (unsigned long long)((fn >= base && fn < base + DM_IMG_MAIN_END) ? fn - base : fn),
                     (unsigned long long)f[FR_X1]);
            note_append("dm_gate.txt", l);
        }
    }
    if (g_stall_arm > 0) {   // np158：武装期内每次分发都取现场（最后一次 = 停车边界）
        g_stall_arm--;
        stall_dump(f, "武装期 0xa00a78");
    }
    return 0;
}
static uint64_t hook_e4(uint64_t *f) {   // tag==2 目标 0x9fe910 入口（记录式，只看不拦）
    if (++g_n_e4 <= 8)
        klog("[钉子8] tag2 0x9fe910 入口：pair=%#llx *pair=%#llx\n",
             (unsigned long long)f[FR_X0],
             (unsigned long long)(f[FR_X0] ? *(const uint64_t *)(uintptr_t)f[FR_X0] : 0));
    return 0;
}
static uint64_t hook_e5(uint64_t *f) {   // 0xa01814（bl 0xa00a78 之后）：x0 = 分发返回值
    if (++g_n_e5 <= 16)
        klog("[钉子8] 分发返回：x0=%#llx（0 ⇒ 就是要找的返 0；x21 即将拿到它）\n",
             (unsigned long long)f[FR_X0]);
    return 0;
}
// ===== 钉子 #8 同轮探针（验证"记录全零来自 *(父+0x10)"的根因推断，只读不改）=====
// 离线只能静态推断父对象身份（工厂 0xa00f8c 零静态引用，疑经 0xa0913c 名字解析间接调），
// 这三个钩子把运行时三点状态打出来：① 工厂入口的父对象与 *(父+0x10) 的记录内容；
// ② 池注册时拷进节点的记录源内容；③ G 取任务时看到的池/节点/记录状态。
static uint64_t hook_e6(uint64_t *f) {   // 管理器工厂入口 0xa00f8c：x0 = 父对象
    if (++g_n_e6 <= 8) {
        uint64_t p = f[FR_X0];
        uint64_t rec = p ? *(const uint64_t *)(uintptr_t)(p + 0x10) : 0;  // *(父+0x10) = 记录指针
        uint64_t r0 = 0, r8 = 0, r70 = 0;
        if (rec) {   // mgr 构造函数随后会 memcpy(mgr+0x58, rec, 0xb8)——能读是确定的
            r0  = *(const uint64_t *)(uintptr_t)rec;
            r8  = *(const uint64_t *)(uintptr_t)(rec + 8);
            r70 = *(const uint64_t *)(uintptr_t)(rec + 0x70);
        }
        klog("[钉子8] 工厂 0xa00f8c：父=%#llx 父[0x10]=%#llx 记录[0]=%#llx [8]=%#llx [0x70]=%#llx\n",
             (unsigned long long)p, (unsigned long long)rec,
             (unsigned long long)r0, (unsigned long long)r8, (unsigned long long)r70);
    }
    return 0;
}
static uint64_t hook_e7(uint64_t *f) {   // 池注册入口 0xa018f8：x0 = mgr，x1 = 记录源（真机实测 = mgr+0x130）
    if (++g_n_e7 <= 8) {
        uint64_t r = f[FR_X1];
        uint64_t q[8] = {0};
        if (r) for (int i = 0; i < 8; i++) q[i] = *(const uint64_t *)(uintptr_t)(r + (uint64_t)i * 8);
        klog("[钉子8] 池注册 0xa018f8：mgr=%#llx 记录源=%#llx（mgr+%#llx） x2=%#llx\n",
             (unsigned long long)f[FR_X0], (unsigned long long)r,
             (unsigned long long)(r - f[FR_X0]), (unsigned long long)f[FR_X2]);
        // 记录头 8 个 qword 全打——真机 5122 行那次 [0]/[8]/[0x70] 全是 0，
        // 这次看清整个头部：如果里面有"名字串指针"却没值，就能确认"该填没填"。
        klog("[钉子8]   记录头：%#llx %#llx %#llx %#llx %#llx %#llx %#llx %#llx\n",
             (unsigned long long)q[0], (unsigned long long)q[1], (unsigned long long)q[2],
             (unsigned long long)q[3], (unsigned long long)q[4], (unsigned long long)q[5],
             (unsigned long long)q[6], (unsigned long long)q[7]);
    }
    return 0;
}
static uint64_t hook_e8(uint64_t *f) {   // G 入口 0xa01d70：x0 = mgr（读法与 G 自己逐条一致）
    ktrace_mark(0xa01d70u, f[FR_X0], 0);   // np126：轨迹标记（日志会静默，轨迹不会）
    // 先取池状态一次，日志与补推共用。**必须先查计数**（与 G 自己的 cmp w9,#1 判据一致）：
    // 激活路径会把节点释放回空闲链（计数--），此后再读 pool[8] 可能拿到野值。
    uint64_t m8 = f[FR_X0];
    uint64_t pool8 = m8 ? *(const uint64_t *)(uintptr_t)(m8 + 0x118) : 0;
    uint64_t node8 = 0, task8 = 0;
    if (pool8 && *(const uint32_t *)(uintptr_t)(pool8 + 0x10) >= 1) {
        node8 = *(const uint64_t *)(uintptr_t)(pool8 + 8);
        if (node8) task8 = *(const uint64_t *)(uintptr_t)(node8 - 0xe0);   // 节点[0] = 任务指针
    }
    if (++g_n_e8 <= 12 || (g_n_e8 % 5000) == 0) {
        uint64_t lab = node8 ? *(const uint64_t *)(uintptr_t)(node8 - 0xe0 + 8) : 0;
        // caller = x30（G 的调用者返回地址）⇒ 减去模块基址就是模块偏移，能直接反查是哪个子系统
        // 在调用 G。np26 真机：主循环里 G 一直被调但池/记录全空 ⇒ 必须知道"谁在问"。
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        uint64_t lr_off = (f[FR_X30] >= base && f[FR_X30] < base + 0x2600000)
                          ? f[FR_X30] - base : 0;
        klog("[钉子8] G#%d 0xa01d70：mgr=%#llx 池=%#llx 首节点链=%#llx 节点[0]=%#llx 记录[0]=%#llx 调用者=+%#llx\n",
             g_n_e8, (unsigned long long)m8, (unsigned long long)pool8,
             (unsigned long long)node8, (unsigned long long)task8,
             (unsigned long long)lab, (unsigned long long)lr_off);
        // ★ Task #16：G 的惰性创建路径末尾会 `if (mgr[0x50]) mgr[0x50]->虚表[0x18](1,1)`——
        //   任务建好后通知"观察者"。真机上它恒 0，通知整条跳过。这里把 0x50 与 0x48（通知数组）
        //   一起打出来：确认到底是"观察者缺失"还是"数组也空"。
        if (m8) {
            uint64_t obs = *(const uint64_t *)(uintptr_t)(m8 + 0x50);
            uint64_t arr = *(const uint64_t *)(uintptr_t)(m8 + 0x48);
            klog("[任务]   mgr[0x50]观察者=%#llx  mgr[0x48]通知数组=%#llx\n",
                 (unsigned long long)obs, (unsigned long long)arr);
        }
        // 额外 dump 任务 +0x80..0xb8（销毁流的通知数组在 +0x88）：看它在激活前是否已填。
        if (task8) {
            uint64_t *a = (uint64_t *)(uintptr_t)(task8 + 0x80);
            klog("[钉子8] 任务数组 [+0x80..0xb0]：%#llx %#llx %#llx %#llx %#llx %#llx\n",
                 (unsigned long long)a[0], (unsigned long long)a[1], (unsigned long long)a[2],
                 (unsigned long long)a[3], (unsigned long long)a[4], (unsigned long long)a[5]);
        }
    }
    // np113：**状态迁移观察者**（只读）——不管是谁写的，任务状态一变就记一行。
    //   目的：拐杖摘掉后，看游戏自己会不会把 1 推走（谁来推、推到几）。
    if (task8) {
        uint32_t st = *(const uint32_t *)(uintptr_t)(task8 + 0xb8);
        if (task8 != g_obs_task) {
            g_obs_task = task8;
            g_obs_state = st;
            klog("[任务链] 首次看到任务 %#llx：状态=%u\n", (unsigned long long)task8, st);
        } else if (st != g_obs_state) {
            klog("[任务链] 任务 %#llx 状态 %u → %u（第 %d 次进入 G）\n",
                 (unsigned long long)task8, g_obs_state, st, g_n_e8);
            g_obs_state = st;
        }
    }
    // ★ 钉子 #8 续修：状态 1→2 补推（2026-10-18，根因见进度文档 14.5）。
    //   ⚠ 2026-10-19 晚 **默认关闭**（np113 真机实证）：这条补丁是"记录全零时代"的拐杖。
    //   实测链路：我们把它推到 2 → 框架的 prepare 通道把 2 当"待销毁"→ 置 4 →
    //   下一拍 TaskMgr::calc 里置 6 并调用任务虚表槽 17/18（析构两张表）→ 销毁 →
    //   销毁里 `blr [task]+0x68`（槽 13 = 收尾）→ 收尾要 RootTask::calc 阶段 7 才建的单例
    //   ⇒ 单例为 0 ⇒ `si_addr=0x180` 崩。
    //   也就是说：**这条补丁直接导致了"任务建出来就死"**。libc 修好后先摘掉它，看游戏自己怎么走。
#if NP_STATE_HACK
    if (task8 && *(const uint32_t *)(uintptr_t)(task8 + 0xb8) == 1) {
        *(uint32_t *)(uintptr_t)(task8 + 0xb8) = 2;
        // ★ flags 补位（2026-10-18，用户批准）：runner 的外层检查（ldrb [task+0x60]; tbz bit1）
        //   与激活检查（and #6 == 6）都要求 [task+0x60] 的 bit1|bit2。我们的任务这块全零、
        //   全代码找不到写入者（无 strb 直写 / 无模板拷贝 / 构造器只清零）⇒ runner 在
        //   0xa01cf0 之前就跳过任务。**按字节** OR 6（检查用 ldrb 读，别碰 +0x61 起的邻居字节）。
        *(uint8_t *)(uintptr_t)(task8 + 0x60) |= 6;
        if (++g_n_e8fix <= 4)
            klog("[钉子8] 状态提升：任务 %#llx 状态 1 ⇒ 2，flags [0x60] |= 6（runner 放行条件）\n",
                 (unsigned long long)task8);
    }
#endif
    return 0;
}
// ===== 钉子 #8 续修探针（runner 链，只读）：激活之后游戏仍自旋 ⇒ runner（0xa02320 →
// 尾跳 0xa01fe0）遍历 mgr+0x90 链表、过检查 0xa01cf0 后才调 task->虚表[0x88]（0x9fdccc，
// 转发给 task+0xd0 子对象）并把状态推 6。检查要求 [task+0x60]&6==6 且状态==4；
// 我们的任务状态=4 但 [task+0x60]=0 且静态分析找不到写入者。两个钩子回答：
// ① 检查函数拿到的"任务"到底是谁（x22 是任务本体还是链表节点？）+ flags/状态现场；
// ② 运行转发器到底有没有被走到。
static uint64_t hook_e9(uint64_t *f) {   // runner 激活检查 0xa01cf0 入口：x0 = mgr，x1 = 候选对象
    if (++g_n_e9 <= 8) {
        uint64_t t = f[FR_X1];
        uint64_t f60 = t ? *(const uint32_t *)(uintptr_t)(t + 0x60) : 0;
        uint64_t st = t ? *(const uint32_t *)(uintptr_t)(t + 0xb8) : 0;
        uint64_t v0 = t ? *(const uint64_t *)(uintptr_t)t : 0;   // 候选对象 [0]（虚表？）
        klog("[钉子8] 检查 0xa01cf0：mgr=%#llx 候选=%#llx [0]=%#llx [0x60]=%#llx(与6=%u) 状态=%llu\n",
             (unsigned long long)f[FR_X0], (unsigned long long)t, (unsigned long long)v0,
             (unsigned long long)f60, (unsigned)((f60 & 6) == 6),
             (unsigned long long)st);
    }
    return 0;
}
static uint64_t hook_e10(uint64_t *f) {  // 运行转发器 0x9fdccc 入口：x0 = 任务（→ +0xd0 子对象）
    if (++g_n_e10 <= 8)
        klog("[钉子8] 运行 0x9fdccc：任务=%#llx [+0xb8]=%#llx（被 runner 选中执行）\n",
             (unsigned long long)f[FR_X0],
             (unsigned long long)(f[FR_X0] ? *(const uint32_t *)(uintptr_t)(f[FR_X0] + 0xb8) : 0));
    return 0;
}
// ★ runner 入口链表遍历（2026-10-18）：上一轮实证 0xa01cf0 零调用 ⇒ 检查根本没走到。
//   离线已解码：0x9f3904 = 双向链表后插（[link+0]=next [link+8]=prev [link+0x10]=载荷指针），
//   runner 从 mgr[0x98]（=头.prev=尾）沿 prev 反向走，x22 = [link+0x10] = 任务本体，
//   然后查 [task+0x60] bit1 —— 我们的任务 flags=0，在 0xa01cf0 之前就被跳过。
//   这个钩子在 runner 入口用 C 沿链表走一遍，运行时坐实：链表里有没有任务、flags 多少。
static uint64_t hook_e11(uint64_t *f) {  // runner 0xa01fe0 入口：x0 = mgr
    ktrace_mark(0xa01fe0u, f[FR_X0], 0);   // np126
    // np113：加"第几次"与 [+0xe0] 表——判"收尾只跑一次"还是"每帧都跑"。
    uint64_t m0 = f[FR_X0];
    int n = ++g_n_e11;
    if (n > 8 && (n % 200) != 0) return 0;
    {
        uint64_t m = m0;
        uint64_t head = m ? m + 0x90 : 0;
        uint64_t cur = m ? *(const uint64_t *)(uintptr_t)(m + 0x98) : 0;
        uint64_t e0 = (m > 0x10000ull && m < 0x800000000000ull)
                    ? *(const uint64_t *)(uintptr_t)(m + 0xe0) : 0;
        klog("[钉子8] runner 第 %d 次：mgr=%#llx [+0xe0]=%#llx（mgr+%#llx）\n",
             n, (unsigned long long)m0, (unsigned long long)e0,
             (unsigned long long)((m0 && e0 > m0) ? e0 - m0 : 0));
        int i = 0;
        while (cur && cur != head && i < 8) {
            uint64_t task = *(const uint64_t *)(uintptr_t)(cur + 0x10);   // 载荷指针
            uint64_t f60 = task ? *(const uint32_t *)(uintptr_t)(task + 0x60) : 0;
            uint64_t st  = task ? *(const uint32_t *)(uintptr_t)(task + 0xb8) : 0;
            uint64_t v0  = task ? *(const uint64_t *)(uintptr_t)task : 0;
            klog("[钉子8] runner 链表[%d]：节点=%#llx 任务=%#llx [0]=%#llx [0x60]=%#llx 状态=%llu\n",
                 i, (unsigned long long)cur, (unsigned long long)task,
                 (unsigned long long)v0, (unsigned long long)f60, (unsigned long long)st);
            cur = *(const uint64_t *)(uintptr_t)(cur + 8);   // prev
            i++;
        }
        if (i == 0)
            klog("[钉子8] runner 链表：空（mgr[0x98]=%#llx，头=%#llx）\n",
                 (unsigned long long)cur, (unsigned long long)head);
    }
    return 0;
}
// ═══ np186 实验 I：runner 放行门 flags 补位（**单因干预**，默认关闭）═══
// 背景（np186 真机 8 轮定谳）：主线程卡在 F_wait 纯用户态空转，退出条件二选一
//   （[mgr+0x128] 惰建子对象 / [waiter+0x50]），二者都可能依赖"任务跑完"；而 runner
//   遍历活动表时第一个门就是 `ldrb w8,[x22+0x60]; tbz w8,#1`（0xa02030/0xa02034），
//   我们的任务 flags 恒 0 ⇒ 0xa01cf0 激活检查从未真命中、任务永不执行。
//   flags 置位器已全部找到（bit2=0xa00960 经 vt+0xa8；bit1=0x28e830/0xf5e248 大函数内联）
//   但从未被调——藏在游戏逻辑深处，静态追不动。
// 干预设计：挂在 **0xa02030**（`ldrb w8,[x22+0x60]` 站点，**不是**前面的 ldp 0xa0202c）：
//   桩的次序是"先调宿主、后执行原指令"——挂 ldp 站点的话，钩子看到的是 ldp **之前**的
//   旧 x22（2026-10-22 首轮实验就这么翻的车：补位补到了错误对象上，随后野指针崩）。
//   挂 ldrb 站点时 x22 已被上面的 ldp 装好（= 当前任务），钩子把 state==4 且
//   flags&6!=6 的任务**仅按字节 OR 6**；随后原 ldrb 读到的就是新值 ⇒ 外层门通过，
//   游戏自己的激活检查/执行逻辑原样跑。
//   **不推 state**——np113 的崩已归因于"状态 1→2 补推"（被 prepare 当待销毁），
//   flags 单因从未单独试过；本实验就是补上这一课。
// 判据：hook_e9（0xa01cf0）首次真命中 = 门过了；hook_e10（0x9fdccc）命中 = 任务真执行；
//   F_wait 退出（[mgr+0x128] 或 [waiter+0x50] 变非 0）⇒ 内容线可能整片解开；
//   崩 ⇒ 崩点即"任务执行还缺什么"的精确答案。
// ⚠ 纪律：编译期开关、**默认关**；实验轮临时改 1，跑完立即改回。
#ifndef NP186_FLAGS
#define NP186_FLAGS 0   // ⚠ 实验 I 已跑完（2026-10-22 凌晨，结论见日报 §1.20）：恢复默认关
#endif
static int g_n_flagsfix;
static uint64_t hook_flagsfix(uint64_t *f) {   // 0xa02030：x22 = ldp 已装好的当前任务
#if NP186_FLAGS
    uint64_t t = f[FR_X22];
    if (t > 0x10000ull && t < 0x800000000000ull) {
        uint32_t st = *(const uint32_t *)(uintptr_t)(t + 0xb8);
        uint8_t fl = *(const uint8_t *)(uintptr_t)(t + 0x60);
        if (st == 4 && (fl & 6u) != 6u) {
            *(uint8_t *)(uintptr_t)(t + 0x60) = (uint8_t)(fl | 6u);
            if (++g_n_flagsfix <= 16)
                klog("[flags] 补位：任务=%#llx state=%u flags %#x → %#x（第 %d 次）\n",
                     (unsigned long long)t, st, fl, fl | 6u, g_n_flagsfix);
        }
    }
#else
    (void)f;
#endif
    return 0;
}
// ★ 销毁入口探针（2026-10-18）：runner 跑完任务（虚表[0x88]）后把任务移进完成链（mgr+0xd8），
//   下一轮对完成链逐个调 0xa01be8 销毁：状态 <7 ⇒ 推 7、析构 task+0x20、再 memcpy(task+0x88, 5)
//   逐个调 指针->[0]->虚表[0x20]。本轮崩在这里：第一个指针（=task+0xf0）的 [0]=0。
//   这个钩子 dump 销毁时的状态、子任务链、以及那 5 个指针（含各自 [0]）——判定数组是不是垃圾。
static int g_n_e12;
static uint64_t hook_e12(uint64_t *f) {   // 销毁入口 0xa01be8：x0 = mgr，x1 = 任务
    if (++g_n_e12 <= 8) {
        uint64_t t = f[FR_X1];
        uint64_t st = t ? *(const uint32_t *)(uintptr_t)(t + 0xb8) : 0;
        uint64_t sub = t ? *(const uint64_t *)(uintptr_t)(t + 0x28) : 0;
        uint64_t a0 = t ? *(const uint64_t *)(uintptr_t)(t + 0x88) : 0;
        uint64_t a1 = t ? *(const uint64_t *)(uintptr_t)(t + 0x90) : 0;
        uint64_t a2 = t ? *(const uint64_t *)(uintptr_t)(t + 0x98) : 0;
        uint64_t a0v = a0 ? *(const uint64_t *)(uintptr_t)a0 : 0;
        klog("[钉子8] 销毁 0xa01be8：mgr=%#llx 任务=%#llx 状态=%llu 子链=%#llx "
             "数组[88]=%#llx(→[0]=%#llx) [90]=%#llx [98]=%#llx\n",
             (unsigned long long)f[FR_X0], (unsigned long long)t, (unsigned long long)st,
             (unsigned long long)sub, (unsigned long long)a0, (unsigned long long)a0v,
             (unsigned long long)a1, (unsigned long long)a2);
    }
    return 0;
}
// ★ 池注册体内 / mgr 构造尾部的定位钩（2026-10-18）：真机现场「日志停在池注册入口之后、
//   内部 memcpy 之前」两轮完全一致 ⇒ 在注册体入口与构造尾部各放一枪，把区间切成两段：
//   若 e13 到、e14 不到 ⇒ 卡在注册体内；若 e14 到 ⇒ 卡在更后面。
static int g_n_e13, g_n_e14;
static uint64_t hook_e13(uint64_t *f) {   // 注册体 0xa01928：x0 = mgr，x1 = 记录源
    if (++g_n_e13 <= 4)
        klog("[钉子8] 注册体 0xa01928：mgr=%#llx 记录源=%#llx\n",
             (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X1]);
    return 0;
}
static uint64_t hook_e14(uint64_t *f) {   // mgr 构造尾部 0xa00d4c（注册之后的解锁点）
    (void)f;                              // 该点 x19=this 不在帧里（桩只存 x0-x17/x29/x30）
    if (++g_n_e14 <= 4)
        klog("[钉子8] 构造尾 0xa00d4c 到达（注册已返回、准备解锁）\n");
    return 0;
}
// ★ 日志系统探针（2026-10-18）：崩点 = 日志遍历（0xf35400）拿节点 [node+8] 当串与 "main"
//   做 strcmp，而该串指针已被写坏（si_addr 恒 0x61fbXXXX5e，高 16 位一致 ⇒ 系统性写坏）。
//   离线已解码：日志对象 [0]=链头 [+0x10]=计数 [+0x18]=空闲链 [+0x28]=容量；
//   节点 [0]=tick [8]=消息串 [0x10]=链节点。两枪对夹：添加入口看"游戏传什么"，遍历入口看现场。
static int g_n_e15, g_n_e16;
static uint64_t hook_e15(uint64_t *f) {   // 日志添加 0xf35374：x0 = 消息（串？结构？）
    if (++g_n_e15 <= 10) {
        uint64_t a = f[FR_X0];
        uint64_t obj = *(const uint64_t *)(uintptr_t)((uint64_t)(uintptr_t)g_k_m.base + 0x2288548);   // 日志对象全局
        uint64_t cnt = obj ? *(const uint32_t *)(uintptr_t)(obj + 0x10) : 0;
        klog("[日志] 添加：参数=%#llx 日志对象=%#llx 计数=%llu\n",
             (unsigned long long)a, (unsigned long long)obj, (unsigned long long)cnt);
    }
    return 0;
}
static uint64_t hook_e16(uint64_t *f) {   // 日志遍历 0xf35400
    if (++g_n_e16 <= 10) {
        uint64_t obj = *(const uint64_t *)(uintptr_t)((uint64_t)(uintptr_t)g_k_m.base + 0x2288548);
        uint64_t head = obj ? *(const uint64_t *)(uintptr_t)obj : 0;
        uint64_t cnt = obj ? *(const uint32_t *)(uintptr_t)(obj + 0x10) : 0;
        uint64_t msg = head ? *(const uint64_t *)(uintptr_t)(head - 0x10 + 8) : 0;
        klog("[日志] 遍历：对象=%#llx 计数=%llu 链头=%#llx 首节点消息=%#llx\n",
             (unsigned long long)obj, (unsigned long long)cnt,
             (unsigned long long)head, (unsigned long long)msg);
    }
    return 0;
}
// ★ 注册体加锁点 0xa01924（2026-10-18）：e13（0xa01928 注册体第一条）从未触发 ⇒ 卡在它前一条
//   `bl 0xa1d56c`（给 mgr+8 上 sead 锁）。这里在锁调用前 dump 锁对象四个字，看 state/owner。
static int g_n_e17;
static uint64_t hook_e17(uint64_t *f) {   // 0xa01924 处：x0 = &mgr[8]（锁对象）
    if (++g_n_e17 <= 8) {
        uint64_t o = f[FR_X0];
        uint64_t w0 = o ? *(const uint64_t *)(uintptr_t)o : 0;
        uint64_t w1 = o ? *(const uint64_t *)(uintptr_t)(o + 8) : 0;
        klog("[钉子8] 注册体上锁前 0xa01924：锁对象=%#llx [0]=%#llx [8]=%#llx\n",
             (unsigned long long)o, (unsigned long long)w0, (unsigned long long)w1);
    }
    return 0;
}

// ===== Task #16：追「第一个任务由谁创建」（离线结论见进度文档 §20）=====
// 背景：sead 主循环每帧只做 lock → G(取任务) → runner(跑任务) → unlock → 睡。
// G 只有在池里**已经有登记项**时才会创建/激活任务；而池的唯一填充入口是 0xa018f8，
// 它的调用者里只有"创建路径"和"任务自建子任务"两条是真来源。
// 建任务的 API 只有两个：0xa0143c（立即，6 个调用点）和 0x9fe314（虚调用，0 个直接调用点）。
// 而 0x726xxx 那三个疑似"连建根任务"的站点，全映像没有任何 bl/b 能到达 ⇒ 只能运行时抓。
//
// 两个探针：
//   ① 0xa0143c 入口：谁调的（x30−基址）、mgr、参数结构体指针；
//   ② 0xa01490（`bl 0xa01790` 的下一条）：此时 **x0 = 刚创建出来的任务**，
//      顺手把 [task]（虚表指针）记下来，供离线反查类名（vtable−0x10 → typeinfo → 名字串）。
static int g_n_e18, g_n_e19;
// 从**我们自己 mmap 的映像**里读一段可打印 ASCII（最长 48 字符）。
// 为什么不用 RTTI 反查类名：实测映像里 `N4sead` 出现 0 次、`typeinfo for` 0 次
// ⇒ 零售版把 RTTI 名字剥了，vtable−0x10 那条路走不通。改读"创建参数里的名字串"。
// 安全性：只读 [base, base+DM_IMG_MAIN_END) 这一段（我们自己的映射，必然可读）。
static int k_img_str(uint64_t addr, char *out, size_t cap) {
    uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
    if (addr < base || addr >= base + DM_IMG_MAIN_END) return 0;
    const char *p = (const char *)(uintptr_t)addr;
    size_t n = 0;
    while (n < 48 && n + 1 < cap) {
        char c = p[n];
        if (!c) break;
        if (c < 0x20 || c > 0x7e) return 0;
        n++;
    }
    if (n < 3) return 0;
    memcpy(out, p, n);
    out[n] = '\0';
    return 1;
}
static uint64_t hook_e18(uint64_t *f) {   // 0xa0143c：CreateTaskImmediate(mgr, param)
    if (++g_n_e18 <= 12 || (g_n_e18 % 200) == 0) {
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        uint64_t lr = f[FR_X30];
        uint64_t lr_off = (lr >= base && lr < base + DM_IMG_MAIN_END) ? lr - base : 0;
        uint64_t par = f[FR_X1];
        klog("[任务] CreateTaskImmediate#%d 调用者=+%#llx mgr=%#llx 参数=%#llx\n",
             g_n_e18, (unsigned long long)lr_off,
             (unsigned long long)f[FR_X0], (unsigned long long)par);
        // 参数结构体（0xb0 字节）头 8 个 qword 里凡是"指向映像内的字符串"，都打出来——
        // 任务名几乎一定在里面（这是唯一不依赖 RTTI 的识别手段）。
        if (par) {
            for (int i = 0; i < 8; i++) {
                uint64_t v = *(const uint64_t *)(uintptr_t)(par + (uint64_t)i * 8);
                char s[64];
                if (k_img_str(v, s, sizeof s))
                    klog("[任务]   参数[%d]=%#llx 「%s」\n", i, (unsigned long long)v, s);
            }
        }
    }
    return 0;
}
static uint64_t hook_e19(uint64_t *f) {   // 0xa01490：刚建好的任务在 x0
    if (++g_n_e19 <= 12) {
        uint64_t t = f[FR_X0];
        uint64_t vt = t ? *(const uint64_t *)(uintptr_t)t : 0;
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        uint64_t vt_off = (vt >= base && vt < base + DM_IMG_MAIN_END) ? vt - base : 0;
        uint64_t st = t ? *(const uint32_t *)(uintptr_t)(t + 0xb8) : 0;
        klog("[任务] 建出任务=%#llx 虚表=+%#llx 状态=%llu\n",
             (unsigned long long)t, (unsigned long long)vt_off, (unsigned long long)st);
    }
    return 0;
}

// ===== np185 实验 A：CTA 派发点显形（纯观测，零行为变化；判据写 dm_gate.txt 绕过预算）=====
// 离线定谳（`离线分析-任务系统卡点定谳（np185）.md`）：主线程卡在 F_wait(0x9feb38) 等待环，
// 因 runner(0xa01fe0) 放行门 [task+0x60] bit1|bit2 在全二进制无写入者。置位者只可能是
// 运行时派发，本组把五个派发点的实际地址打到 dm_gate.txt，拿到即可离线反汇编：
//   ① 0xa00a78 kind=2 工厂指针（[结构体+8]）——已在 hook_e3 内补打（同一站点不能挂两次）
//   ② CTA 建好任务后的 [arg+0xb0] 回调        → hook_e31 @ 0xa014d8（mov x0,x21 之前）
//   ③ 任务的 vt+0x50 方法                     → hook_e32 @ 0xa01514（mov x0,x21 之前）
//   ④ 逐类目 0xa00804(task, i, entry)         → hook_e33 @ 0xa01570（bl 之前，安装器重算 bl）
//   ⑤ F_wait 的调用者（经虚表进来、无直接 bl） → hook_e34 @ 0x9feb3c（序言之后，打 x30）
// 站点均为安全指令（非 PC 相对 / 不改 SP；bl 站点由安装器按原站点 PC 重算 imm26）。
static int g_n_e31, g_n_e32, g_n_e33, g_n_e34;

// 0xa014d8：x8 = [arg+0xb0] 的"建好后回调"（arg 在 x20，一并记下便于对齐 CTA#N）
static uint64_t hook_e31(uint64_t *f) {
    if (++g_n_e31 <= 8) {
        uint64_t cb = f[FR_X8];
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        char l[176];
        snprintf(l, sizeof l, "派发#%d [arg+0xb0]回调=%s%#llx arg=%#llx\n",
                 g_n_e31,
                 (cb >= base && cb < base + DM_IMG_MAIN_END) ? "img+" : "外",
                 (unsigned long long)((cb >= base && cb < base + DM_IMG_MAIN_END) ? cb - base : cb),
                 (unsigned long long)f[FR_X20]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// 0xa01514：x8 = 任务虚表 +0x50 槽的方法（CTA 在 state=1 后调它）
static uint64_t hook_e32(uint64_t *f) {
    if (++g_n_e32 <= 8) {
        uint64_t m = f[FR_X8];
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        char l[176];
        snprintf(l, sizeof l, "派发#%d vt+0x50=%s%#llx arg=%#llx\n",
                 g_n_e32,
                 (m >= base && m < base + DM_IMG_MAIN_END) ? "img+" : "外",
                 (unsigned long long)((m >= base && m < base + DM_IMG_MAIN_END) ? m - base : m),
                 (unsigned long long)f[FR_X20]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// 0xa01570：bl 0xa00804 之前——x0=任务 x1=类目序号 x2=类目值（逐类目初始化）
static uint64_t hook_e33(uint64_t *f) {
    if (++g_n_e33 <= 16) {
        char l[128];
        snprintf(l, sizeof l, "派发#%d 类目 0xa00804：任务=%#llx 序号=%u 值=%#llx\n",
                 g_n_e33, (unsigned long long)f[FR_X0],
                 (unsigned)(uint32_t)f[FR_X1], (unsigned long long)f[FR_X2]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// 0x9feb3c（F_wait 序言之后）：x0=等待对象，x30=调用者返回地址 ⇒ 谁在等、等的哪个对象
// 受保护读：客体指针可能是垃圾，直接解引用会把探针自己读崩。
// 用门跳转缓冲（g_gate_jmp，已改 __thread）兜住：读失败返回 0，调用方据此跳过。
static int k_read_ok(const void *p, void *dst, size_t n) {
    if (!p) return 0;
    if (sigsetjmp(g_gate_jmp, 1) == 0) {
        g_gate_armed = 1;
        memcpy(dst, p, n);
        g_gate_armed = 0;
        return 1;
    }
    g_gate_armed = 0;
    return 0;
}

// ===== np210 实验 A：M3 挂接链验证（纯观测，零行为变化；判据写 dm_gate.txt）=====
// 离线定谳（`离线分析-M3内容门-惰性挂接定谳（np210）.md`）：
//   `[X+0x128]` 是**惰性挂接槽**——X 的类 ctor(0xa00afc) 在 0xa00d20 用 `str xzr` 清零，
//   由子系统初始化函数（含 0x9fc6e0 段）在 0x9fc8c4 挂上 0x60 字节子对象。F_wait 的退出
//   条件①就是它。本组钩子判定卡点假设 H1（挂接函数没被触发）/H2（半路断：接口表没填齐）/
//   H3（挂到别的对象上）：
//   ① hook_m3_enter @ 0x9fc6f4（函数体内首次分配之前）——函数跑没跑、来时的 x19/x20/x22/x24；
//   ② hook_m3_attach @ 0x9fc8c4（挂接点）——挂没挂、x19=管理器（与 F_wait 的 [waiter+0x38] 对号）；
//   ③ hook_e34（F_wait）首次进入时补打接口表九槽（0x227f000+0x6a0..0x6e8 → 0x22d73xx）——
//      静态映像里这些结构首成员全零=运行时才填；自旋时仍为 0 ⇒ 注册没发生（H2 判据）。
static int g_n_m3_enter, g_n_m3_attach;

// 接口表九槽 dump：槽值=结构指针，结构首成员=函数指针（运行时填）。
static void m3_dump_iface(uint64_t base) {
    static const uint32_t offs[] = {0x6a0,0x6a8,0x6b0,0x6b8,0x6c0,0x6c8,0x6d0,0x6d8,0x6e8};
    char l[224];
    for (unsigned i = 0; i < sizeof offs / sizeof offs[0]; i++) {
        uint64_t slot = 0, fn = 0;
        int ok1 = k_read_ok((const void *)(base + 0x227f000 + offs[i]), &slot, 8);
        int ok2 = (ok1 && slot) ? k_read_ok((const void *)slot, &fn, 8) : 0;
        snprintf(l, sizeof l, "  iface[+%#x]=%#llx → fn=%#llx（槽%s fn%s）\n",
                 offs[i], (unsigned long long)slot, (unsigned long long)fn,
                 ok1 ? "OK" : "败", ok2 ? "OK" : (ok1 && slot ? "败" : "-"));
        note_append("dm_gate.txt", l);
    }
}

// 0x9fc6f4：bl 0x9f33f0（首次分配）之前。此时 x19/x20/x22/x24 还是调用方传进来的值。
static uint64_t hook_m3_enter(uint64_t *f) {
    if (++g_n_m3_enter <= 8) {
        char l[256];
        snprintf(l, sizeof l, "M3挂接#%d 子系统初始化进入：x19=%#llx x20=%#llx x22=%#llx x24=%#llx\n",
                 g_n_m3_enter, (unsigned long long)f[FR_X19], (unsigned long long)f[FR_X20],
                 (unsigned long long)f[FR_X22], (unsigned long long)f[FR_X24]);
        note_append("dm_gate.txt", l);
        if (g_n_m3_enter == 1) {
            note_append("dm_gate.txt", "M3 接口表（挂接函数进入时）：\n");
            m3_dump_iface((uint64_t)(uintptr_t)g_k_m.base);
        }
    }
    return 0;
}

// 0x9fc8c4：str x23,[x19,#0x128] —— 惰性挂接点。x19=管理器 x23=新子对象。
static uint64_t hook_m3_attach(uint64_t *f) {
    if (++g_n_m3_attach <= 8) {
        uint64_t mgr = f[FR_X19], sub = f[FR_X23];
        char l[256];
        snprintf(l, sizeof l, "M3挂接点#%d [x19+0x128] <- %#llx（管理器 x19=%#llx）\n",
                 g_n_m3_attach, (unsigned long long)sub, (unsigned long long)mgr);
        note_append("dm_gate.txt", l);
        // 对号：管理器 +0 应是主 vt（img+0x21b5d40）；不是 ⇒ 挂到了别的对象上（H3）
        uint64_t vt = 0;
        if (k_read_ok((const void *)mgr, &vt, 8)) {
            uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
            int in = (vt >= base && vt < base + DM_IMG_MAIN_END);
            snprintf(l, sizeof l, "  管理器 vt=%s%#llx（期望 img+0x21b5d40）\n",
                     in ? "img+" : "外",
                     (unsigned long long)(in ? vt - base : vt));
            note_append("dm_gate.txt", l);
        }
    }
    return 0;
}

// np211g：0xc5274c 是 np211f 离线定谳的**真正挂接点**（str x0,[x19,#0x128]）。
//   宿主函数 0xc52664；x19 = 0xa0913c 经分配器协议（0xa0e2f8 → 影子块+0xb0 → vt+0x80/vt+0x30）
//   建出的对象。判读：修复前此钩子不响；修复后响且 X+0x128 非零 ⇒ F_wait 退出条件①满足。
static int g_n_m3_real;
static uint64_t hook_m3_real_attach(uint64_t *f) {
    if (++g_n_m3_real <= 8) {
        uint64_t mgr = f[FR_X19], sub = f[FR_X0];
        char l[256];
        snprintf(l, sizeof l, "M3真挂接#%d [x19+0x128] <- %#llx（对象 x19=%#llx）\n",
                 g_n_m3_real, (unsigned long long)sub, (unsigned long long)mgr);
        note_append("dm_gate.txt", l);
        uint64_t vt = 0;
        if (k_read_ok((const void *)mgr, &vt, 8)) {
            uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
            int in = (vt >= base && vt < base + DM_IMG_MAIN_END);
            snprintf(l, sizeof l, "  对象 vt=%s%#llx；子对象 vt=%s\n",
                     in ? "img+" : "外",
                     (unsigned long long)(in ? vt - base : vt),
                     k_read_ok((const void *)sub, &vt, 8) ? "(已读)" : "(读不到)");
            note_append("dm_gate.txt", l);
        }
    }
    return 0;
}

// ===== np211g 诊断链（np211f 根因链的四点验证）=====
// 0xc52664：挂接宿主函数入口（无静态调用者=虚派发）——进没进
static int g_n_h1;
static uint64_t hook_np211g_host(uint64_t *f) {
    if (++g_n_h1 <= 4) {
        char l[224];
        snprintf(l, sizeof l, "np211g宿主#%d 进入 x0=%#llx x1=%#llx x2=%#llx x3=%#llx\n",
                 g_n_h1, (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X1],
                 (unsigned long long)f[FR_X2], (unsigned long long)f[FR_X3]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}
// 0xa0913c：取子对象函数入口——打第三参 x2（非空就直接用，不进 0xa0e2f8）
static int g_n_h2;
static uint64_t hook_np211g_sub(uint64_t *f) {
    if (++g_n_h2 <= 4) {
        char l[224];
        snprintf(l, sizeof l, "np211g取子#%d 进入 x2(第三参)=%#llx x3=%#llx\n",
                 g_n_h2, (unsigned long long)f[FR_X2], (unsigned long long)f[FR_X3]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}
// 0xa0e2f8 入口：打 ctx 与键号（w0 = [ctx+0x88]）——定"读哪个 TSD 槽"
static int g_n_h3;
static uint64_t hook_np211g_tlsget(uint64_t *f) {
    if (++g_n_h3 <= 4) {
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        uint64_t ctxh = 0, slot = 0xFFFFFFFFu;
        if (k_read_ok((const void *)(base + 0x22783d8), &ctxh, 8) && ctxh)
            if (k_read_ok((const void *)ctxh, &ctxh, 8) && ctxh)
                k_read_ok((const void *)(ctxh + 0x88), &slot, 4);
        char l[224];
        snprintf(l, sizeof l, "np211g TLS取#%d arg=%#llx ctx+0x88(键号)=%u\n",
                 g_n_h3, (unsigned long long)f[FR_X0], (unsigned)slot);
        note_append("dm_gate.txt", l);
    }
    return 0;
}
// 0xa0e338：0xa0e2f8 返回点——打返回值（0 ⇒ 门就是在这断的）
static int g_n_h4;
static uint64_t hook_np211g_tlsret(uint64_t *f) {
    if (++g_n_h4 <= 4) {
        char l[224];
        snprintf(l, sizeof l, "np211g TLS返#%d x0=%#llx\n",
                 g_n_h4, (unsigned long long)f[FR_X0]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// np211i：三个新挂接候选（np211h 排除 0xc5274c 后的下一批）
// ① 0xa3581c：链表推进+挂接（x8=[x21+0x38] 当前节点，x23 挂到其 +0x128）
static int g_n_i1;
static uint64_t hook_np211i_list(uint64_t *f) {
    if (++g_n_i1 <= 8) {
        uint64_t node = 0, vt = 0;
        // x8 在站点已载入（FR_X8）；子对象 x23
        node = f[FR_X8];
        if (k_read_ok((const void *)node, &vt, 8)) {
            uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
            int in = (vt >= base && vt < base + DM_IMG_MAIN_END);
            char l[256];
            snprintf(l, sizeof l, "np211i链表挂接#%d [%#llx+0x128] <- %#llx；节点 vt=%s%#llx\n",
                     g_n_i1, (unsigned long long)node, (unsigned long long)f[FR_X23],
                     in ? "img+" : "外", (unsigned long long)(in ? vt - base : vt));
            note_append("dm_gate.txt", l);
        }
    }
    return 0;
}
// ② 0x523ec：惰建单例真入口（0xc5274c 那条链的门；判进没进/单例空不空）
static int g_n_i2;
static uint64_t hook_np211i_singleton(uint64_t *f) {
    if (++g_n_i2 <= 4) {
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        uint64_t holder = 0, inst = 0;
        if (k_read_ok((const void *)(base + 0x227a218), &holder, 8) && holder)
            k_read_ok((const void *)holder, &inst, 8);
        char l[224];
        snprintf(l, sizeof l, "np211i单例#%d 进入 arg=%#llx 持有者=%#llx 单例=%s\n",
                 g_n_i2, (unsigned long long)f[FR_X0], (unsigned long long)holder,
                 inst ? "已存在" : "空（将新建）");
        note_append("dm_gate.txt", l);
    }
    return 0;
}
// ③ 0x1187c74：str x21,[x8,#0x128]（x8 = [x19+8] 的子对象）
static int g_n_i3;
static uint64_t hook_np211i_sub2(uint64_t *f) {
    if (++g_n_i3 <= 8) {
        char l[224];
        snprintf(l, sizeof l, "np211i子对象挂接#%d [x8+0x128] <- x21=%#llx（x19=%#llx）\n",
                 g_n_i3, (unsigned long long)f[FR_X21], (unsigned long long)f[FR_X19]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// ===== np210 实验 B（第二轮）：门态 + 图形初始化入口 + 两个备选挂点 =====
// 第一轮判读（见 np210 文档）：挂接函数体（0x9fc6f4 起）从未执行；离线定谳表明它是
// **图形/显示初始化 0x9fc450**，唯一调用点 0x7355a0 之前有一道门
// （0x735574 `cbz x0`——x0 = 单例 getter 0x1074b50 的返回值，空则整个图形初始化被跳过）。
// 单例 getter 读两个句柄变量（img+0x232f9e8 / img+0x232f9f0，经静态句柄槽
// 0x2289e60/0x2289e68 间接），静态为 0、无静态写入者 ⇒ 运行时创建。
// 本轮四钩：
//   ① hook_m3_gate  @ 0x735574（cbz 站点，x0=getter 结果）——门到底跳不跳；
//      首次调用时把两个句柄变量 dump 出来（自旋时是否为 0）。
//   ② hook_m3_ginit @ 0x9fc450（图形初始化入口）——被进入过没有。
//   ③ hook_m3_alt1  @ 0x9fdaf8（兄弟类 ctor 里的 +0x128 写）——备选挂点跑没跑。
//   ④ hook_m3_alt2  @ 0x9ffbd8（另一处 +0x128 写）——备选挂点跑没跑。
static int g_n_m3_gate, g_n_m3_ginit, g_n_m3_alt1, g_n_m3_alt2;

// dump 两个单例句柄变量：*(0x2289e68) → var1；*(0x2289e60) → var2（句柄槽静态指向 bss 变量）
static void m3_dump_singletons(uint64_t base) {
    char l[256];
    const uint64_t slots[2] = {0x2289e68, 0x2289e60};
    for (int i = 0; i < 2; i++) {
        uint64_t h = 0, v = 0;
        int ok1 = k_read_ok((const void *)(base + slots[i]), &h, 8);
        int ok2 = (ok1 && h) ? k_read_ok((const void *)h, &v, 8) : 0;
        snprintf(l, sizeof l, "  单例句柄[%#x]=%#llx → 变量=%#llx（%s）\n",
                 slots[i], (unsigned long long)h, (unsigned long long)v,
                 ok2 ? (v ? "非空" : "空") : "读败");
        note_append("dm_gate.txt", l);
    }
}

// 0x735574：bl 0x1074b50 之后的 cbz——x0 = 单例 getter 的返回值（0 = 门跳走、图形初始化被跳过）
static uint64_t hook_m3_gate(uint64_t *f) {
    if (++g_n_m3_gate <= 8) {
        char l[224];
        snprintf(l, sizeof l, "M3门#%d getter 结果=%#llx（%s）\n",
                 g_n_m3_gate, (unsigned long long)f[FR_X0],
                 f[FR_X0] ? "非空⇒放行" : "空⇒跳过图形初始化");
        note_append("dm_gate.txt", l);
        if (g_n_m3_gate == 1) {
            note_append("dm_gate.txt", "M3 单例句柄（门处）：\n");
            m3_dump_singletons((uint64_t)(uintptr_t)g_k_m.base);
        }
    }
    return 0;
}

// 0x9fc450：图形/显示初始化入口（序言 stp x28,x27,[sp,#-0x60]!）
static uint64_t hook_m3_ginit(uint64_t *f) {
    if (++g_n_m3_ginit <= 4) {
        char l[224];
        snprintf(l, sizeof l, "M3图形初始化#%d 进入：x0=%#llx x1=%#llx x2=%#llx\n",
                 g_n_m3_ginit, (unsigned long long)f[FR_X0],
                 (unsigned long long)f[FR_X1], (unsigned long long)f[FR_X2]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// 0x9fdaf8：兄弟类 ctor 内的 +0x128 写（备选挂点一）
static uint64_t hook_m3_alt1(uint64_t *f) {
    if (++g_n_m3_alt1 <= 8) {
        char l[224];
        snprintf(l, sizeof l, "M3备选一#%d [x20+0x128] <- %#llx（对象 x20=%#llx）\n",
                 g_n_m3_alt1, (unsigned long long)f[FR_X8], (unsigned long long)f[FR_X20]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// 0x9ffbd8：另一处 +0x128 写（备选挂点二）
static uint64_t hook_m3_alt2(uint64_t *f) {
    if (++g_n_m3_alt2 <= 8) {
        char l[224];
        snprintf(l, sizeof l, "M3备选二#%d [x20+0x128] <- %#llx（对象 x20=%#llx）\n",
                 g_n_m3_alt2, (unsigned long long)f[FR_X24], (unsigned long long)f[FR_X20]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// ===== np210 实验 C（第三轮）：X 类槽30方法入口 + X 的工厂入口 =====
// 第二轮判读：X 的 +0x128 应由 X 类虚方法 0x9fd858（vt 0x21b5d40 槽30/+0xf0）初始化
// （其函数体内 0x9fdaf8 写内嵌子对象 vt）——round2 它对另外 3 个同类对象跑过，独独漏了 X。
// 注意 0x9fdaf8 是函数中段，若方法有分支提前返回，中段钩子会漏 ⇒ 本轮挂**入口**：
//   ① hook_m3_m30 @ 0x9fd858——每次调用打 x0（对象）+ x30（派发者），看 X 收没收到、谁派发；
//   ② hook_m3_fac  @ 0xa00f90——X 的工厂入口，打 x0（父对象）+ x30（创建者），看 X 谁建的。
// 两个地址都是函数序言（stp），安全可挂。
static int g_n_m3_m30, g_n_m3_fac;

static uint64_t hook_m3_m30(uint64_t *f) {
    if (++g_n_m3_m30 <= 16) {
        uint64_t lr = f[FR_X30];
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        char l[224];
        snprintf(l, sizeof l, "M3方法30#%d 对象=%#llx 派发者=%s%#llx\n",
                 g_n_m3_m30, (unsigned long long)f[FR_X0],
                 (lr >= base && lr < base + DM_IMG_MAIN_END) ? "img+" : "外",
                 (unsigned long long)((lr >= base && lr < base + DM_IMG_MAIN_END) ? lr - base : lr));
        note_append("dm_gate.txt", l);
    }
    return 0;
}

static uint64_t hook_m3_fac(uint64_t *f) {
    if (++g_n_m3_fac <= 6) {
        uint64_t lr = f[FR_X30];
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        char l[224];
        snprintf(l, sizeof l, "M3工厂#%d 父对象=%#llx 创建者=%s%#llx\n",
                 g_n_m3_fac, (unsigned long long)f[FR_X0],
                 (lr >= base && lr < base + DM_IMG_MAIN_END) ? "img+" : "外",
                 (unsigned long long)((lr >= base && lr < base + DM_IMG_MAIN_END) ? lr - base : lr));
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// np212：X+0x128 代填桩的虚方法——一律"无事发生"（返 0），与 k_gate_stub 同语义。
static int g_x128_stub_calls;
static uint64_t x128_stub_method(void) {
    if (++g_x128_stub_calls <= 8)
        klog("[np212] X+0x128 桩虚方法第 %d 次被调\n", g_x128_stub_calls);
    return 0;
}

static uint64_t hook_e34(uint64_t *f) {
    if (++g_n_e34 <= 8) {
        uint64_t lr = f[FR_X30];
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        uint64_t waiter = f[FR_X0];
        char l[256];
        snprintf(l, sizeof l, "等待#%d F_wait 进入：等待对象=%#llx 调用者=%s%#llx\n",
                 g_n_e34, (unsigned long long)waiter,
                 (lr >= base && lr < base + DM_IMG_MAIN_END) ? "img+" : "外",
                 (unsigned long long)((lr >= base && lr < base + DM_IMG_MAIN_END) ? lr - base : lr));
        note_append("dm_gate.txt", l);

        // np210：F_wait 首次自旋时打接口表九槽——与"挂接函数进入时"的 dump 对照，
        // 看注册到底发生在哪个时间点（H2 判据：自旋时仍全零 ⇒ 注册从未发生）。
        if (g_n_e34 == 1) {
            note_append("dm_gate.txt", "M3 接口表（F_wait 首次自旋时）：\n");
            m3_dump_iface(base);
        }

        // ★ 2026-10-22（M3 关键一问）：F_wait 的退出条件是 [waiter+0x38]→[+0x128] 或 [waiter+0x50]。
        //   **前两次进入**就把这两条读出来，并把 [waiter+0x38] 指向的对象头 dump 16 个字——
        //   目的是"定类"：它到底是哪一类对象，决定了 +0x128 该由谁写。
        if (g_n_e34 <= 2) {
            uint64_t sub = 0, v128 = 0;
            uint32_t v50 = 0;
            int ok_sub = k_read_ok((const void *)(uintptr_t)(waiter + 0x38), &sub, 8);
            int ok_50  = k_read_ok((const void *)(uintptr_t)(waiter + 0x50), &v50, 4);
            if (ok_sub && sub) ok_sub = k_read_ok((const void *)(uintptr_t)(sub + 0x128), &v128, 8);
            snprintf(l, sizeof l, "  → [waiter+0x38]=%#llx（读%s）  [[+0x38]+0x128]=%#llx  [waiter+0x50]=%#x（读%s）\n",
                     (unsigned long long)sub, ok_sub ? "成功" : "失败",
                     (unsigned long long)v128, v50, ok_50 ? "成功" : "失败");
            note_append("dm_gate.txt", l);
            if (sub) {
                uint64_t hdr[16];
                if (k_read_ok((const void *)(uintptr_t)sub, hdr, sizeof hdr)) {
                    snprintf(l, sizeof l,
                             "  obj@%#llx: %#llx %#llx %#llx %#llx | %#llx %#llx %#llx %#llx\n"
                             "              %#llx %#llx %#llx %#llx | %#llx %#llx %#llx %#llx\n",
                             (unsigned long long)sub,
                             (unsigned long long)hdr[0], (unsigned long long)hdr[1],
                             (unsigned long long)hdr[2], (unsigned long long)hdr[3],
                             (unsigned long long)hdr[4], (unsigned long long)hdr[5],
                             (unsigned long long)hdr[6], (unsigned long long)hdr[7],
                             (unsigned long long)hdr[8], (unsigned long long)hdr[9],
                             (unsigned long long)hdr[10], (unsigned long long)hdr[11],
                             (unsigned long long)hdr[12], (unsigned long long)hdr[13],
                             (unsigned long long)hdr[14], (unsigned long long)hdr[15]);
                    note_append("dm_gate.txt", l);
                }
            }
        }
    }
    // ★★★ np212 实验 A（用户 2026-10-23 02:33 批准）：**代填 X+0x128 让游戏冲过去** ★★★
    // 依据：np188 实验 I 已实证——强行置位 X+0x128 后任务生命周期全程跑通
    // （插入→激活→执行→完成→销毁），崩点在销毁路径（可另外伺候）。
    // 与其继续查"谁该设它"（已查 20+ 轮无果），不如先盖章让游戏前进：
    //   · 冲过去 ⇒ M3 真正的问题当场暴露（新阶段日志/新崩溃点）；
    //   · 崩了 ⇒ 崩溃点直接告诉我们这个字段该长什么样。
    // 桩对象：合法 vtable + 全部虚方法返 0（F_wait 只查非零；退出后游戏拿它走虚调用，
    // 返回 0 是最保守的"无事发生"语义，与 k_gate_stub 一致）。
    // 注意：本段在 g_n_e34<=8 限制**之外**——每轮 F_wait 都检查，直到填上。
    {
        static uint64_t stub_obj[0x40];
        static uint64_t stub_vt[0x40];
        static int stub_ready, fill_n;
        if (!stub_ready) {
            for (int i = 0; i < 0x40; i++) stub_vt[i] = (uint64_t)(uintptr_t)&x128_stub_method;
            stub_obj[0] = (uint64_t)(uintptr_t)&stub_vt[0];
            stub_obj[1] = (uint64_t)(uintptr_t)&stub_obj[0];   // 自指（与 g_alloc_obj 同构）
            stub_ready = 1;
        }
        uint64_t sub = 0, v128 = 0;
        if (k_read_ok((const void *)(uintptr_t)(f[FR_X0] + 0x38), &sub, 8) && sub) {
            if (k_read_ok((const void *)(uintptr_t)(sub + 0x128), &v128, 8) && v128 == 0) {
                *(uint64_t *)(uintptr_t)(sub + 0x128) = (uint64_t)(uintptr_t)&stub_obj[0];
                if (++fill_n <= 8) {
                    char l[224];
                    snprintf(l, sizeof l, "np212代填#%d X=%#llx +0x128 0 → 桩 %p（F_wait 放行）\n",
                             fill_n, (unsigned long long)sub, (void *)&stub_obj[0]);
                    note_append("dm_gate.txt", l);
                    klog("[np212] 代填 X+0x128：X=%p → 桩对象\n", (void *)sub);
                }
            }
        }
    }
    return 0;
}

// np185 实验 C v2：**每次都记**的路径计数器（bc6 的首次命中式回答不了"最后一次调用卡在哪"）。
// 站点表与安装处 cpts[] 一致；按站点偏移分桶计数，每命中一行（上限 64 行防刷爆）。
// 判读：日志末尾**最后出现的 C 点** = 主线程停车前最后一站；若某点计数比后点少 1 ⇒ 卡在两点之间。
static int g_cpt_n[20];
static uint64_t hook_cpt(uint64_t *f) {
    uint64_t off = f[FR_SITE] - (uint64_t)(uintptr_t)g_k_m.base;
    // np185 实验 E v2：主线程终卡点 = TextureArchiveWork 实体创建返回之后的路径。
    //   路径：0x107e824 存对象 → vt+0x70(0xa0b150 getter) → vt+0x78(0xa0b158 锁+链表)
    //   → 0x107e868 挂载(0xa112fc) → 0x107e890 通知(0x1086ed0 自身也加锁) → 0x107e894+
    //   每命中都计；**最后停者即卡点**。另计 sead 锁族进出（0xa1d56c 进 / 0xa1d57c 出）：
    //   进 > 出 ⇒ 卡在 SDK 的 nn::os::LockMutex（np133 白名单直连 sdk 实现，绕过宿主日志桩）。
    static const uint32_t sites[20] = { 0xa1d69c, 0xa1d77c, 0xa1d79c,
                                        0xa1d7b0, 0xa1d7d8, 0xa1d818,
                                        0xa091b0, 0xa091fc, 0xa0923c, 0x1086ac8,
                                        0xa1d56c, 0xa1d57c, 0xa0b188, 0x107e858,
                                        0x107e86c, 0x1086efc, 0x107e894,
                                        0, 0, 0 };
    for (int i = 0; i < 17; i++) {
        if (off == sites[i]) {
            // 高频站点（锁族，本轮实证 3500 万次级）粗抽样：前 32 条全记，之后每 65536 条一次；
            //   路径站点（<百次级）不受影响。判据是"最后一条 C 点是哪个"，不是精确总数。
            int n = ++g_cpt_n[i];
            int hot = (off == 0xa1d56c || off == 0xa1d57c);
            if (n <= 32 || (n % (hot ? 65536 : 256)) == 0) {
                char l[160];
                if (off == 0xa1d69c || off == 0x1086ac8) {   // 入口/返回点：附 x30 = 调用者返回地址（定位是哪个调用者）
                    uint64_t lr = f[FR_X30];
                    uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
                    snprintf(l, sizeof l, "C点 img+%#x 第 %d 次 调用者=%s%#llx\n",
                             (unsigned)off, g_cpt_n[i],
                             (lr >= base && lr < base + DM_IMG_MAIN_END) ? "img+" : "外",
                             (unsigned long long)((lr >= base && lr < base + DM_IMG_MAIN_END) ? lr - base : lr));
                } else {
                    snprintf(l, sizeof l, "C点 img+%#x 第 %d 次\n", (unsigned)off, g_cpt_n[i]);
                }
                note_append("dm_gate.txt", l);
            }
            break;
        }
    }
    return 0;
}

// np185 实验 D：KingSystem 落户观测。RootTask 周期 dump 读 [RootTask+0x358]（KingSystem）恒 0，
// 而 RootTask::calc 内 0x726afc `str x20,[x19+0x358]` 理应把它写进去（0x726adc 注册调用的返回后）。
// 本钩子挂在 0x726b00（存储已执行的安全指令）：打 x20（新对象）、写入后的 [x19+0x358]、
// 新对象 [+0xd4]（phase1 放行条件 3|4 的那个字段）与虚表——回答"对象建出来没有/落户没有/状态推没推"。
static int g_n_e36;
static uint64_t hook_e36(uint64_t *f) {
    if (++g_n_e36 <= 12) {
        uint64_t obj = f[FR_X20], ks = 0, d4 = 0, vt = 0;
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        // x19 不在帧里（桩只存 x0-x17/x29/x30 + x18-x28……实际有 x19-x28）——用 FR_X19 取
        uint64_t rt = f[FR_X19];
        if (rt > 0x10000ull) ks = *(const uint64_t *)(uintptr_t)(rt + 0x358);
        if (obj > 0x10000ull) {
            vt = *(const uint64_t *)(uintptr_t)obj;
            uint32_t v = *(const uint32_t *)(uintptr_t)(obj + 0xd4);
            d4 = v;
        }
        char l[224];
        snprintf(l, sizeof l, "KingSystem#%d 新对象=%#llx 虚表=%s%#llx [+0xd4]=%llu 落户[+0x358]=%#llx\n",
                 g_n_e36, (unsigned long long)obj,
                 (vt >= base && vt < base + DM_IMG_MAIN_END) ? "img+" : "外",
                 (unsigned long long)((vt >= base && vt < base + DM_IMG_MAIN_END) ? vt - base : vt),
                 (unsigned long long)d4, (unsigned long long)ks);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

static int g_n_e20, g_n_e21;
// 0xa0913c 是"按名字建/查管理器对象"的助手（x1 指向一个 {虚表?, 名字串} 结构）。
// 真机日志 5100 行那次唯一的方法树创建就用了它，名字来自 0x1ba6xxx 那张表——
// 那张表里全是 sead 的方法/管理器名：CalculateTask、RootCalc/SysCalc/AppCalc、
// RootDraw/SysDraw/AppDraw/AppDrawFinal、sead::MethodTreeMgr / ThreadMgr / SystemManagers …
// ⇒ 只要把"被请求的名字 + 拿到的结果"打出来，就知道哪一环在宿主上没建起来。
static uint64_t hook_e20(uint64_t *f) {   // 0xa0913c 入口
    if (++g_n_e20 <= 300) {
        uint64_t st = f[FR_X1];
        uint64_t nm = 0;
        char s[64] = {0};
        if (st) nm = *(const uint64_t *)(uintptr_t)(st + 8);
        if (nm && k_img_str(nm, s, sizeof s)) { /* 拿到名字 */ }
        klog("[名字] 建/查#%d 结构=%#llx 名字指针=%#llx 「%s」\n",
             g_n_e20, (unsigned long long)st, (unsigned long long)nm, s);
    }
    return 0;
}
static uint64_t hook_e21(uint64_t *f) {   // 0x9fe380：刚拿到 0xa0913c 的返回值（x0）
    if (++g_n_e21 <= 20) {
        klog("[名字] 方法树创建结果 x0=%#llx（0 = 没建起来）\n", (unsigned long long)f[FR_X0]);
    }
    return 0;
}
// ⚠ 真机教训（np32）：只挂了 0xa0913c（thunk：`and w5,w5,#1` 后 b 到实体）→ **一行都没打**。
//   全映像统计：`bl 0xa0913c` 102 处、`bl 0xa09144`（实体入口）26 处 ⇒ 这个 API 有**两个入口**，
//   只挂一个必然漏。两个都挂，才能拿到完整的"按名字建对象"序列。
static int g_n_e22;
static uint64_t hook_e22(uint64_t *f) {   // 0xa09144：同一 API 的实体入口
    if (++g_n_e22 <= 300) {
        uint64_t st = f[FR_X1];
        uint64_t nm = 0;
        char s[64] = {0};
        if (st) nm = *(const uint64_t *)(uintptr_t)(st + 8);
        if (nm) k_img_str(nm, s, sizeof s);
        {
            uint64_t lr = f[FR_X30];
            uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
            klog("[名字] 实体入口#%d 结构=%#llx 名字=%#llx 「%s」 调用者=%s%#llx\n",
                 g_n_e22, (unsigned long long)st, (unsigned long long)nm, s,
                 (lr >= base && lr < base + DM_IMG_MAIN_END) ? "img+" : "外",
                 (unsigned long long)((lr >= base && lr < base + DM_IMG_MAIN_END) ? lr - base : lr));
        }
    }
    return 0;
}
// 0x9fe984 是 `SystemManagers::initialize()` 里"去建 ThreadMgr"的那一步（0x9fe984 → 0x9fe24c）。
// 它命中 ⇒ 框架初始化确实走到这里；不命中 ⇒ 卡在更前面。
static int g_n_e23;
static uint64_t hook_e23(uint64_t *f) {
    if (++g_n_e23 <= 6) klog("[框架] 0x9fe984：SystemManagers 初始化里要建 ThreadMgr\n");
    return 0;
}
// 0xa112fc = 「往列表里塞 {节点, 名字}」的助手。7 棵方法树（RootCalc/SysCalc/AppCalc/
// RootDraw/SysDraw/AppDraw/AppDrawFinal）就是靠它连续挂 7 次（见 0x9ff398–0x9ff430）。
// 它被调几次、每次名字是什么，直接回答"方法树有没有挂上去"。
static int g_n_e24;
static uint64_t hook_e24(uint64_t *f) {
    if (++g_n_e24 <= 300) {
        uint64_t st = f[FR_X1];
        uint64_t nm = st ? *(const uint64_t *)(uintptr_t)(st + 8) : 0;
        char s[64] = {0};
        if (nm) k_img_str(nm, s, sizeof s);
        klog("[挂载] 0xa112fc#%d 目标=%#llx 名字指针=%#llx 「%s」\n",
             g_n_e24, (unsigned long long)f[FR_X0], (unsigned long long)nm, s);
    }
    return 0;
}
static int g_n_e25;
static uint64_t hook_e25(uint64_t *f) {   // 0x9ff3a0：7 棵树挂载段（纯 store，不喂寄存器）
    if (++g_n_e25 <= 4) klog("[框架] 0x9ff3a0：进入「挂 7 棵方法树」这一段\n");
    return 0;
}

// ===== np111：RootTask::calc 阶段状态机（P0-1/P0-2，见文件上方 g_rt_* 的离线结论）=====
static uint64_t hook_rt_calc(uint64_t *f) {   // 0x726838 入口：x0 = RootTask*
    uint64_t th = f[FR_X0];
    ktrace_mark(0x726838u, th, 0);   // np126：轨迹标记——"calc 到没到"在时间线上一眼可见
    g_rt_calls++;
    if (!g_rt_this && th > 0x10000ull && th < 0x800000000000ull) {
        uint64_t b0 = (uint64_t)(uintptr_t)g_k_m.base;
        int in = b0 && th >= b0 && th < b0 + DM_IMG_MAIN_END;
        g_rt_this = th;
        g_rt_phase = *(const uint8_t *)(uintptr_t)(th + 0x200);
        g_rt_seen |= 1u << (g_rt_phase & 31);
        klog("[RootTask] ★ 首次进入 calc：this=%#llx（映像内 %s%#llx）phase=%d x30=%#llx\n",
             (unsigned long long)th, in ? "+" : "(外)",
             (unsigned long long)(in ? th - b0 : th), g_rt_phase,
             (unsigned long long)f[FR_X30]);
        rt_dump("首帧");
    }
    if (th == g_rt_this) {   // 每帧读一次阶段，只在变化时记（不刷屏）
        int ph = *(const uint8_t *)(uintptr_t)(th + 0x200);
        if (ph != g_rt_phase) {
            int from = g_rt_phase;
            g_rt_phase = ph;
            g_rt_seen |= 1u << (ph & 31);
            if (++g_rt_trans <= 32)
                klog("[RootTask] 阶段 %d → %d（第 %d 次进入 calc）\n", from, ph, g_rt_calls);
        }
    }
    return 0;
}
// ⭐ np166：**缓冲源空 ⇒ 零页兜底**（影子数据范式，救活建任务的工人）。
// 依据（死亡笔记 + 反汇编）：0x11e7a20（缓冲读）从 `[obj+0x180]` 取源指针，经
//   csel 传给 0x11e8754（逐字节比较/拷贝循环，x0=目标、x1=源、w2=长度）；
//   源指针为 0 ⇒ `ldrb w15,[x1,x10]` 崩（si_addr=0）。
//   该工人是"建任务"的那个 ⇒ 它一死任务链永远空 ⇒ RootTask 不 calc ⇒ 不装载。
// 修法（与"缺内容=空串/零页"的既有影子范式一致）：钩 0x11e8754 入口，
//   源指针非法 ⇒ 换成静态零页、长度钳到零页大小 ⇒ 比较必然不相等（走"未命中"分支，
//   语义最保守），工人不崩、任务链能建。零页数据不代表内容正确——只为越过这一关。
static uint8_t g_zero_src[0x10000];
static int g_zsrc_n;
static uint64_t hook_bufcopy(uint64_t *f) {
    uint64_t src = f[FR_X1];
    if (!src || src < 0x10000ull) {
        f[FR_X1] = (uint64_t)(uintptr_t)g_zero_src;
        if (f[FR_X2] > (uint64_t)sizeof g_zero_src) f[FR_X2] = (uint64_t)sizeof g_zero_src;
        if (++g_zsrc_n <= 8) {
            klog("[零页] ★ 0x11e8754 源指针=%#llx 非法 ⇒ 换零页（长度钳 %llu，第 %d 次）\n",
                 (unsigned long long)src, (unsigned long long)f[FR_X2], g_zsrc_n);
            // 绕过日志预算的专用写（崩溃点在 400KB 预算之后，普通日志看不到）
            if (g_files_dir[0]) {
                char line[200], p[600];
                int n = snprintf(line, sizeof line, "零页#%d src=%#llx len=%llu\n", g_zsrc_n,
                                 (unsigned long long)src, (unsigned long long)f[FR_X2]);
                snprintf(p, sizeof p, "%s/dm_zero.txt", g_files_dir);
                int fd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0666);
                if (fd >= 0 && n > 0) { (void)!write(fd, line, (size_t)n); close(fd); }
            }
        }
    }
    return 0;
}

// ⭐ np184：任务"运行"虚方法 `0xa0078c` 的生命周期链（已实测它永不返回）：
//   `blr [vt+0x78]` → `blr [vt+0x80]` → `blr [vt+0x20](w1=0)` → `blr [vt+0x28](w1=0)` → `br [vt+0x60]`（尾调）。
//   这里把 5 个目标一次性记下来，离线逐个反汇编即可找出真正死循环的那个。
static int g_taskrun_n;
static uint64_t hook_taskrun(uint64_t *f) {   // 0xa00794（序言之后，安全指令）
    if (++g_taskrun_n <= 3) {
        char l[320];
        uint64_t t = f[FR_X0], vt = 0;
        uint64_t s[5] = {0, 0, 0, 0, 0};
        if (t > 0x10000ull) vt = *(const uint64_t *)(uintptr_t)t;
        if (vt > 0x10000ull)
            for (int i = 0; i < 5; i++)
                s[i] = *(const uint64_t *)(uintptr_t)(vt + (const uint32_t[]){0x78, 0x80, 0x20, 0x28, 0x60}[i]);
        uint64_t b = (uint64_t)(uintptr_t)g_k_m.base;
        snprintf(l, sizeof l, "run#%d task=%#llx vt=%#llx  78=%s%#llx 80=%s%#llx 20=%s%#llx 28=%s%#llx 60=%s%#llx\n",
                 g_taskrun_n, (unsigned long long)t, (unsigned long long)vt,
                 s[0] >= b && s[0] < b + DM_IMG_MAIN_END ? "img+" : "外", (unsigned long long)(s[0] >= b && s[0] < b + DM_IMG_MAIN_END ? s[0] - b : s[0]),
                 s[1] >= b && s[1] < b + DM_IMG_MAIN_END ? "img+" : "外", (unsigned long long)(s[1] >= b && s[1] < b + DM_IMG_MAIN_END ? s[1] - b : s[1]),
                 s[2] >= b && s[2] < b + DM_IMG_MAIN_END ? "img+" : "外", (unsigned long long)(s[2] >= b && s[2] < b + DM_IMG_MAIN_END ? s[2] - b : s[2]),
                 s[3] >= b && s[3] < b + DM_IMG_MAIN_END ? "img+" : "外", (unsigned long long)(s[3] >= b && s[3] < b + DM_IMG_MAIN_END ? s[3] - b : s[3]),
                 s[4] >= b && s[4] < b + DM_IMG_MAIN_END ? "img+" : "外", (unsigned long long)(s[4] >= b && s[4] < b + DM_IMG_MAIN_END ? s[4] - b : s[4]));
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// ⭐ np183b：**面包屑**——在 PreInitializeApp（0xdb9284..0xdb9ae8）内部布 8 个标记点，
//   每个只在首次命中时写一行 `bcrd img+<offset>`。跑一轮就能看出"它停在哪一段之后"。
//   站点都是安全指令（非 PC 相对/不动 SP）；cbz/bl 由安装器重算位移。
static uint64_t g_bcrd_seen[48];
static int g_bcrd_n;
static uint64_t hook_bcrd(uint64_t *f) {
    uint64_t off = f[FR_SITE] - (uint64_t)(uintptr_t)g_k_m.base;
    // ⭐ np184：`0xa01620` 是 `blr [vt+0x58]`（任务虚方法）之前的那条——已实测它**永不返回**。
    //   这里把任务对象、虚表、以及 0x58 槽的方法地址一起记下来，离线直接反汇编那个方法。
    if (off == 0xa01620ull) {
        char l[192];
        uint64_t task = f[FR_X21];
        uint64_t vt = 0, m = 0;
        if (task > 0x10000ull) {
            vt = *(const uint64_t *)(uintptr_t)task;
            if (vt > 0x10000ull) m = *(const uint64_t *)(uintptr_t)(vt + 0x58);
        }
        uint64_t b = (uint64_t)(uintptr_t)g_k_m.base;
        snprintf(l, sizeof l, "vcall58# task=%#llx vt=%s%#llx m=%s%#llx\n",
                 (unsigned long long)task,
                 (vt >= b && vt < b + DM_IMG_MAIN_END) ? "img+" : "外",
                 (unsigned long long)((vt >= b && vt < b + DM_IMG_MAIN_END) ? vt - b : vt),
                 (m >= b && m < b + DM_IMG_MAIN_END) ? "img+" : "外",
                 (unsigned long long)((m >= b && m < b + DM_IMG_MAIN_END) ? m - b : m));
        note_append("dm_gate.txt", l);
    }
    for (int i = 0; i < g_bcrd_n; i++) if (g_bcrd_seen[i] == off) return 0;
    if (g_bcrd_n < 48) {
        char l[96];
        g_bcrd_seen[g_bcrd_n++] = off;
        snprintf(l, sizeof l, "bcrd img+%#llx（第 %d 个）\n",
                 (unsigned long long)off, g_bcrd_n);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// ⭐ np183：**编排进度标记**。目的：回答"PreInitializeApp 到底停在哪一段"。
//   两个点都是安全指令（非 PC 相对、不动 SP）：
//     0xdb9ae8 `mov w26,wzr` —— 堆空时走的补支路（w26=0 那条）；
//     0xdb99e4 `mov x0,x20`  —— 打印 "StarterPackMgr" 前一句（StarterPackMgr 段入口，
//                              装载编排 0xdb9960 就在它附近）。
static int g_pm_early_n, g_pm_spm_n;
static uint64_t hook_pm_early(uint64_t *f) {   // 0xdb9ae8
    (void)f;
    if (++g_pm_early_n <= 4) {
        char l[96];
        snprintf(l, sizeof l, "pm_early#%d（堆空补支路）\n", g_pm_early_n);
        note_append("dm_gate.txt", l);
    }
    return 0;
}
static uint64_t hook_pm_spm(uint64_t *f) {     // 0xdb99e4
    (void)f;
    if (++g_pm_spm_n <= 4) {
        char l[96];
        snprintf(l, sizeof l, "pm_spm#%d（StarterPackMgr 段）\n", g_pm_spm_n);
        note_append("dm_gate.txt", l);
    }
    return 0;
}

// ⭐ np179：**主线程在等谁**。真机已定谳：主线程在 `ksys::PreInitializeApp` 里跑等待环
//   （0x9febac `ldr w8,[x19,#0x50]; cbnz → 退出`，否则 0x9febb4 `bl 0xa1db68` 边 pump 任务边睡），
//   `[this+0x50]` 永远为 0 ⇒ 无限等。这里把 this 的身份与关键字段记下来（只记前 3 次）。
static int g_wait50_n;
static uint64_t hook_wait50(uint64_t *f) {   // 0x9febac：等待环判据点
    uint64_t self = f[FR_X19];
    int n = ++g_wait50_n;
    if (n <= 3) {
        char line[320];
        uint64_t sub = 0, sub128 = 0, f50 = 0, vt = 0;
        if (self > 0x10000ull) {
            vt = *(const uint64_t *)(uintptr_t)self;
            sub = *(const uint64_t *)(uintptr_t)(self + 0x38);
            f50 = *(const uint32_t *)(uintptr_t)(self + 0x50);
            if (sub > 0x10000ull) sub128 = *(const uint64_t *)(uintptr_t)(sub + 0x128);
        }
        snprintf(line, sizeof line,
                 "wait50#%d this=%#llx vt=%#llx [0x38]=%#llx [0x38+0x128]=%#llx [0x50]=%#x\n",
                 n, (unsigned long long)self, (unsigned long long)vt,
                 (unsigned long long)sub, (unsigned long long)sub128, (unsigned)f50);
        note_append("dm_gate.txt", line);
        // np180：整对象 dump（只在第 1 次），用来判"这个等待者是什么东西"——
        //   前几槽通常是 vtable/自指，中段常含名字指针或状态字。
        if (n == 1 && self > 0x10000ull) {
            char d[400];
            for (int r = 0; r < 4; r++) {
                int len = snprintf(d, sizeof d, "  obj[%02x]:", r * 0x20);
                for (int k = 0; k < 4; k++)
                    len += snprintf(d + len, sizeof d - (size_t)len, " %#llx",
                                    (unsigned long long)((const uint64_t *)(uintptr_t)self)[r * 4 + k]);
                snprintf(d + len, sizeof d - (size_t)len, "\n");
                note_append("dm_gate.txt", d);
            }
        }
    }
    // ⭐ np181：**强制放行实验已做，结论为负**——真机把 `[this+0x50]` 写成 1 后，
    //   游戏**没有**继续推进（建/查仍止于 #34、0xdb9960 未命中、[fs] open 仍 0）。
    //   ⇒ 这个标志不是（唯一）阻塞点，或每轮被对象重建清掉。
    //   代码保留但**默认关闭**（`NP181_FORCE50=0`），避免把实验手段混进基线。
#define NP181_FORCE50 0
#if NP181_FORCE50
    if (n == 400 && self > 0x10000ull) {
        char l2[128];
        *(uint32_t *)(uintptr_t)(self + 0x50) = 1;
        snprintf(l2, sizeof l2, "force50 this=%#llx 置 [0x50]=1（第 %d 次）\n",
                 (unsigned long long)self, n);
        note_append("dm_gate.txt", l2);
    }
#else
    (void)n; (void)self;
#endif
    return 0;
}

// ⭐ np175：**"建-信号无界循环"溯源**。真机证据链（np173–np175）：
//   ① 过滤空转后，主线程末段只剩 `InitializeLightEvent → SignalLightEvent`；
//   ② 12 次初始化**全部**来自同一返回地址 `img+0x10795c8`，事件对象按 0x2F0 递增；
//   ③ 0x10795c8 是构造器 `0x1079574` 里 `bl 0xf3bfd4`（InitializeLightEvent 的本地 thunk）的下一句。
//   ⇒ 每轮都在构造一个新对象（内含 LightEvent @ +0x58）。抓**构造器的调用者**：
//     · LR 恒定 ⇒ 某函数在循环里反复构造（再看它上层的退出条件）；
//     · LR 各不相同 ⇒ 是**无限递归**（每层构造一个子对象）。
static int g_ltor_n;
static uint64_t hook_ctor2f0(uint64_t *f) {   // 0x1079574：内含 LightEvent 的对象构造器
    (void)f;
    // np175：前 3 次留完整帧链；之后每 64 次留一条，用来判定"重建池"是**慢**还是**无界**。
    ++g_ltor_n;
    if (g_ltor_n <= 3 || (g_ltor_n % 64) == 0) {
        char line[512];
        uint64_t b = (uint64_t)(uintptr_t)g_k_m.base;
        // **一次走完栈回溯**（np175b）：构造器是一层层派生下来的，逐轮挂一个钩子太慢；
        //   这里顺着帧指针链把 8 层返回地址全带回来，离线一把反汇编即可定位工厂/循环。
        //   帧链安全闸：fp 16 字节对齐、严格递增、lr 落在映像内才算有效帧。
        int n = 0;
        int len = snprintf(line, sizeof line, "ctor#%d site=img+%#llx lr=img+%#llx x0=%#llx\n",
                           g_ltor_n, (unsigned long long)(f[FR_SITE] - b),
                           (unsigned long long)(f[FR_X30] - b), (unsigned long long)f[FR_X0]);
        // 池初始化函数的帧链可能不规整：放宽为"只要求在映像内"，并带原始 fp 便于诊断
        uint64_t fp = f[FR_X29];
        for (int i = 0; i < 10 && fp >= 0x1000 && (fp & 7) == 0; i++) {
            uint64_t prev = *(uint64_t *)(uintptr_t)fp;
            uint64_t ret  = *(uint64_t *)(uintptr_t)(fp + 8);
            len += snprintf(line + len, sizeof line - (size_t)len, "  ↑#%d fp=%#llx ret=",
                            i + 1, (unsigned long long)fp);
            if (ret >= b && ret < b + DM_IMG_MAIN_END)
                len += snprintf(line + len, sizeof line - (size_t)len, "img+%#llx\n",
                                (unsigned long long)(ret - b));
            else
                len += snprintf(line + len, sizeof line - (size_t)len, "(外)%#llx\n",
                                (unsigned long long)ret);
            if (len >= (int)sizeof line - 80) break;
            if (prev == fp || prev < fp) break;
            fp = prev;
        }
        note_append("dm_gate.txt", line);
    }
    return 0;
}

static uint64_t hook_rt_mgr(uint64_t *f) {   // 0x726f60：phase2/phase4 的「建管理器」站点
    (void)f;
    if (++g_rt_mgr <= 4)
        klog("[RootTask] ★ 走到「建管理器」分支（站点 0x726f60，bl 0xdb9284）第 %d 次\n", g_rt_mgr);
    return 0;
}

// ⭐ np165：**装载器入口探针**（0xd5b694）——回答"启动包到底有没有被要求装载"。
// 依据（离线反汇编）：0xdb9284（StarterPackMgr 创建 + 装载编排）尾部依次调
//   0xd5b354 → 0xd5b4c8 → **0xd5b694（装载器）** → 0xd5b758 → …；
//   装载器一进来就读门：`x8 = *(0x2276168); x8 = [x8]; ldrb w8,[x8,#0x419];
//   cbnz w8 → 跳到 0xd5b744（**跳过装载**）`。
//   ⇒ 本钩子把门三层值打出来：外层指针 / 内层对象 / 门字节。判读：
//     · 钩子 0 次  ⇒ 编排没走到装载器（更早的分支/条件把它拦了）；
//     · 内层=0     ⇒ 门对象没建（会在装载器里崩，或已被前置判断拦）；
//     · 门字节≠0   ⇒ **主动跳过装载**（这就是 [fs] open 恒 0 的直接原因）。
// np168：0xdb9284 内部两点探针（判断"建管理器之后是否走到装载编排"）。
//   0xdb9960 = 装载编排入口；0xdb9a54 = 装载器调用点。两点都走**绕过预算**的专用写。
static void note_append(const char *name, const char *line) {
    if (!g_files_dir[0]) return;
    char p[600];
    snprintf(p, sizeof p, "%s/%s", g_files_dir, name);
    int fd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd >= 0) { (void)!write(fd, line, strlen(line)); close(fd); }
}
static int g_mgr_entry_n;
static uint64_t hook_mgr_entry(uint64_t *f) {   // 0xdb9284 入口：验证"钩子桩重放 bl"是否真把被调函数跑起来
    if (++g_mgr_entry_n <= 6) {
        char line[120];
        snprintf(line, sizeof line, "mgr入口#%d x0=%#llx x30=%#llx\n",
                 g_mgr_entry_n, (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X30]);
        klog("[编排] ★ 0xdb9284 入口 x0=%#llx\n", (unsigned long long)f[FR_X0]);
        note_append("dm_gate.txt", line);
    }
    return 0;
}
static int g_orch_n, g_ldcall_n;
static uint64_t hook_orch(uint64_t *f) {   // 0xdb9960：装载编排入口
    if (++g_orch_n <= 8) {
        char line[120];
        snprintf(line, sizeof line, "编排入口#%d x0=%#llx\n", g_orch_n, (unsigned long long)f[FR_X0]);
        klog("[编排] ★ 0xdb9960 进入 x0=%#llx\n", (unsigned long long)f[FR_X0]);
        note_append("dm_gate.txt", line);
    }
    return 0;
}
static uint64_t hook_ldcall(uint64_t *f) {   // 0xdb9a54：装载器调用点
    if (++g_ldcall_n <= 8) {
        char line[120];
        snprintf(line, sizeof line, "装载调用#%d x0=%#llx\n", g_ldcall_n, (unsigned long long)f[FR_X0]);
        klog("[编排] ★ 0xdb9a54 装载器调用点 x0=%#llx\n", (unsigned long long)f[FR_X0]);
        note_append("dm_gate.txt", line);
    }
    return 0;
}

static int g_ld_gate;
static uint64_t hook_loader(uint64_t *f) {
    if (++g_ld_gate <= 6) {
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        uint64_t outer = *(const uint64_t *)(uintptr_t)(base + 0x2276168);
        uint64_t inner = 0, gate = 0xff;
        if (outer > 0x10000ull && outer < 0x800000000000ull) {
            inner = *(const uint64_t *)(uintptr_t)outer;
            if (inner > 0x10000ull && inner < 0x800000000000ull)
                gate = *(const uint8_t *)(uintptr_t)(inner + 0x419);
        }
        char line[240];
        snprintf(line, sizeof line,
                 "装载器#%d x0=%#llx 门[外=%#llx 内=%#llx 字节=%#x]%s\n",
                 g_ld_gate, (unsigned long long)f[FR_X0], (unsigned long long)outer,
                 (unsigned long long)inner, (unsigned)gate,
                 (gate != 0 && gate != 0xff) ? " ⇒ 将跳过装载" : "");
        klog("[装载器] ★ %s", line);
        // 绕过日志预算的专用写（预算在 ~4100 行就截了，而装载发生在 5000+ 行）
        if (g_files_dir[0]) {
            char p[600];
            snprintf(p, sizeof p, "%s/dm_gate.txt", g_files_dir);
            int fd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0666);
            if (fd >= 0) { (void)!write(fd, line, strlen(line)); close(fd); }
        }
    }
    return 0;
}
static uint64_t hook_rt_ctor(uint64_t *f) {  // 0x7277ec：RootTask 构造（alloc 0x3d0）
    if (++g_rt_ctor <= 4)
        klog("[RootTask] 构造 0x7277ec 被调（第 %d 次）owner=%#llx x30=%#llx\n",
             g_rt_ctor, (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X30]);
    return 0;
}

static int g_rt_prepare;      // np113：RootTask::prepare 命中次数
static uint64_t hook_rt_prepare(uint64_t *f) {   // 0x7265b8 = RootTask::prepare（名串 0x1b78894）
    if (++g_rt_prepare <= 4)
        klog("[RootTask] prepare 0x7265b8 被调（第 %d 次）this=%#llx x30=%#llx\n",
             g_rt_prepare, (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X30]);
    return 0;
}

// ===== np114：把客体的日志函数（0xf35374）换成宿主实现 =====
// 真机实证（本轮）：prepare 在 **Prepare 线程** 上跑起来后，它与主线程同时调 0xf35374，
// 两边都从**同一个空闲节点链表**（日志对象 +0x18）里摘节点：
//   节点空闲时 [0] = next；在用时 [0] = 字符串指针。
//   两个线程同时摘同一个节点 ⇒ 后一个把 [0]（已是字符串指针）当成 next 写回链表头
//   ⇒ 链表头变成垃圾 ⇒ 下一次 `ldr x8,[x21]` 崩在 0xf353a8（si_addr 是垃圾串地址）。
// 换成宿主实现后：① 天然串行（klog 自带锁）；② 客体的日志原文转进我们的日志文件，调试更有用。
static int g_gamelog_n;
static uint64_t hook_game_log(uint64_t *f) {   // 0xf35374：参数 = 消息串指针
    const char *s = (const char *)(uintptr_t)f[FR_X0];
    uint64_t p = f[FR_X0];
    uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
    int in_img = base && p >= base && p < base + DM_IMG_MAIN_END;
    int in_host = p >= 0x10000ull && p < 0x800000000000ull && !in_img;
    if (s && (in_img || in_host) && g_gamelog_n < 400) {
        char buf[192];
        size_t i = 0;
        for (; i + 1 < sizeof buf; i++) { char c = s[i]; if (!c) break; buf[i] = c; }
        buf[i] = 0;
        if (i > 0) {
            g_gamelog_n++;
            klog("[游戏日志] %s\n", buf);
        }
    }
    return 0;
}

// ===== np116：阶段 0 的崩点 +0xa041e0（x19=0，读 [x19+0x4a]）=====
// 离线（本轮）：
//   崩点函数 0xa041d0 很小：`if (this->[0x4a]) { cb=*(0x22d7420); cb([this+0x10], w1..w4); [0x4a]=0 }`
//   它的唯一调用者 0x9fd604：`x0=[x19+0x128]; bl 0xa041d0` ⇒ **x19+0x128 是 0**。
//   x19 = 帧循环对象（构造器 0x9fc328，vtable 0x21b5a00，槽 34 = 0x9fd490 的帧循环）。
//   [x19+0x128] 只有两个人写：
//     ① 0x9fc8c4（在 0x9fc450 里）：alloc 0x60 + 0xa040b0 构造 ⇒ 存进去。
//        0x9fc450 内部调 nv::SetGraphicsAllocator / nv::InitializeGraphics / nn::vi::GetNativeWindow
//        —— 是 **Volante 的图形初始化**。但它没有任何 bl/adr/静态指针引用 ⇒ 经 blr 间接调用。
//     ② 0x9fdaf8（在 0x9fd9c4 里，0x9fc314 的尾调）：[x20+0x128] = *(0x2280990)+0x10。
//   ⇒ 两条路谁该先跑、跑没跑，用钩子一测便知。
static int g_vol_init;     // 0x9fc450 命中
static int g_gfx_post;     // 0x9fd9c4 命中
static int g_gfx_flush;    // 0xa041d0 命中
static uint64_t hook_vol_init(uint64_t *f) {   // 0x9fc450：Volante 图形初始化
    if (++g_vol_init <= 4)
        klog("[Volante] 初始化 0x9fc450 被调（第 %d 次）this=%#llx x30=%#llx\n",
             g_vol_init, (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X30]);
    return 0;
}
static uint64_t hook_gfx_post(uint64_t *f) {   // 0x9fd9c4：另一条写 +0x128 的路
    if (++g_gfx_post <= 4)
        klog("[Volante] 后置 0x9fd9c4 被调（第 %d 次）this=%#llx x30=%#llx\n",
             g_gfx_post, (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X30]);
    return 0;
}
static uint64_t hook_gfx_flush(uint64_t *f) {  // 0xa041d0：崩点函数本体
    if (++g_gfx_flush <= 4)
        klog("[Volante] ★ 收尾 0xa041d0：this=%#llx x30=%#llx（0 ⇒ 就是崩因）\n",
             (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X30]);
    return 0;
}

// ===== np121：NVN 函数表为什么是空的（0x22d7428 = 0 ⇒ worker blr 空指针）=====
// 离线（本轮）：0x9b2938 = **NVN 函数表注册**（Volante 初始化里 `bl 0x9b2938(0, nvnDeviceGetProcAddress)`），
//   里面是一长串 `x0=dev; x1="nvnProgramInitialize"; blr x19(解析器); [槽]=x0`，
//   把解析结果存进 0x22d7410 一族全局槽。我们的崩点槽 0x22d7428（经 0x227f7a0 访问）
//   **唯一的写者就是 0x9b3030 这条 `str x0,[x8]`**。
//   ⇒ 三个互斥可能，一次跑测就能分辨：
//     ① 0x9b2938 没跑（表初始化根本没被调）
//     ② 跑了但半路 bail（0x9b3030 不命中）
//     ③ 跑了且写进 0，即 GPA 返回了 0（0x9b3030 命中但 x0=0）
static int g_nvn_tab_init;   // 0x9b2938 命中
static int g_nvn_tab_store;  // 0x9b3030 命中
static uint64_t hook_nvn_tab_init(uint64_t *f) {
    if (++g_nvn_tab_init <= 4)
        klog("[NVN表] 注册函数 0x9b2938 被调（第 %d 次）dev=%#llx 解析器=%#llx x30=%#llx\n",
             g_nvn_tab_init, (unsigned long long)f[FR_X0],
             (unsigned long long)f[FR_X1], (unsigned long long)f[FR_X30]);
    return 0;
}
static uint64_t hook_nvn_tab_store(uint64_t *f) {   // 0x9b3030：str x0,[x8]
    if (++g_nvn_tab_store <= 8 || (g_nvn_tab_store % 64) == 0)
        klog("[NVN表] 写入第 %d 次：值=%#llx → 槽=%#llx\n",
             g_nvn_tab_store, (unsigned long long)f[FR_X0], (unsigned long long)f[8]);
    return 0;
}

// np123：确认"显示对象 [+0x128] 到底有没有建起来 + 谁在用它"。
//   0x9fc8c4 = `str x23,[x19,#0x128]`（Volante 初始化里建显示对象那一条，唯一正常的写法）
//   0x9fd490 = 帧循环入口（虚表槽 34），它用 [this+0x128]
static int g_gfx_set128, g_gfx_frame;
static uint64_t hook_gfx_set128(uint64_t *f) {   // 0x9fc8c4：str x23,[x19,#0x128]
    // ⚠ 桩只保存 x0–x17/nzcv/x29/x30，**x19/x23 读不到** ⇒ 这里只记"这条存不存在"。
    (void)f;
    if (++g_gfx_set128 <= 4)
        klog("[Volante] 显示对象建立点 0x9fc8c4 命中（第 %d 次）\n", g_gfx_set128);
    return 0;
}
static uint64_t hook_gfx_frame(uint64_t *f) {    // 帧循环入口
    uint64_t t = f[FR_X0], d = 0;
    ktrace_mark(0x9fd490u, t, 0);   // np126
    if (t > 0x10000ull && t < 0x800000000000ull) d = *(const uint64_t *)(uintptr_t)(t + 0x128);
    if (++g_gfx_frame <= 4)
        klog("[Volante] 帧循环 #%d：this=%#llx [this+0x128]=%#llx\n",
             g_gfx_frame, (unsigned long long)t, (unsigned long long)d);
    return 0;
}

// ===== np113：查"RootTask 建出来就立刻被销毁"的那条路径 =====
// 离线（本轮）：0x726768 = RootTask 虚表槽 13（+0x68）= 收尾/退出：
//   先 0x728088(**(0x227aaa0)+0x180) 拆掉某个子系统，再 0x1072adc 清单例，再关一串子系统。
//   它由 0xa01be8（任务销毁）里 `blr [task]+0x68` 调用 ⇒ 只要任务被销毁，它就必然跑。
// 而那个"单例"正是 **RootTask::calc 阶段 7** 用工厂 0x1072abc 建出来的（0x72722c 处读槽）
//   ⇒ 阶段 7 没跑过，单例就是 0 ⇒ 退出路径拿 0+0x180 当对象 ⇒ si_addr=0x180。
// 所以真正要回答的是：**任务为什么刚建好就被销毁？** 三个点足够定性：
//   ① 0xa01fe0（TaskMgr 的每帧处理）被调几次 —— 一次=收尾流程，多次=帧循环
//   ② 0xa01aec（链表插入）把任务插进哪张表 —— mgr+0xd8 还是别处
//   ③ 0x726768（退出）的调用者 x30 —— 谁在拆
static int g_list_ins;     // 0xa01aec 插入次数
static int g_task_exit;    // 0x726768 进入次数

static uint64_t hook_list_ins(uint64_t *f) {   // 0xa01aec：把任务插进某张有序表
    uint64_t m = f[FR_X0], lst = f[FR_X1], t = f[FR_X2];
    if (++g_list_ins <= 12) {
        uint32_t st = 0;
        if (t > 0x10000ull && t < 0x800000000000ull)
            st = *(const uint32_t *)(uintptr_t)(t + 0xb8);
        klog("[任务链] 链表插入 0xa01aec：mgr=%#llx 目标表=%#llx（mgr+%#llx） 任务=%#llx 状态=%u\n",
             (unsigned long long)m, (unsigned long long)lst,
             (unsigned long long)((m && lst > m) ? lst - m : 0),
             (unsigned long long)t, st);
    }
    return 0;
}
static uint64_t hook_task_exit(uint64_t *f) {  // 0x726768：RootTask 槽 13（退出/收尾）
    uint64_t t = f[FR_X0];
    if (++g_task_exit <= 6) {
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        uint64_t lr = f[FR_X30];
        int in = lr >= base && lr < base + DM_IMG_MAIN_END;
        uint64_t holder = *(const uint64_t *)(uintptr_t)(base + 0x227aaa0);
        uint64_t inst = (holder > 0x10000ull && holder < 0x800000000000ull)
                      ? *(const uint64_t *)(uintptr_t)holder : 0;
        uint32_t st = 0;
        if (t > 0x10000ull && t < 0x800000000000ull)
            st = *(const uint32_t *)(uintptr_t)(t + 0xb8);
        klog("[任务链] ★ 任务退出 0x726768：this=%#llx 状态=%u 调用者=%s%#llx 单例=%#llx\n",
             (unsigned long long)t, st, in ? "+" : "(外)",
             (unsigned long long)(in ? lr - base : lr), (unsigned long long)inst);
    }
    return 0;
}

// ===== 启动链追踪（Task #16 第三步）=====
// 思路换档：前几轮都是"猜一个点、挂一个钩"，一次只回答一个是非题，太慢。
// 这里把 **nnMain 里所有 bl 目标**按出现顺序一次性挂上，用同一个宿主函数（靠 f[FR_SITE] 区分站点）。
// 一次跑测就能得到"初始化走到第几步、在哪一步之后断掉"的完整顺序图。
// 站点顺序 = nnMain（0x735330）里 bl 的出现顺序（离线反汇编得到，见进度文档 §22/§23）。
static const uint32_t g_boot_sites[] = {
    0x9f3830, 0xf32228, 0xf326d4, 0x9fe234, 0xc4f3e4,
    0x8115d4, 0x811730, 0x815254, 0xdaab74, 0xc4f2f4,
    0xdaabc0, 0x9fc450, 0xa22508, 0xa2262c, 0xa03178,
    0xa03224, 0xa025f0, 0xa026a0, 0xa2469c, 0x9fce0c,
    0xa003f0, 0x9fe240, 0xa1e5f0, 0xa9cc4c,
};
static int g_n_boot;
// np68：设值函数 0xa2469c 的**替身**。原实现两条指令：`x8 = *(0x2280eb8); *(u64*)x8 = x0`。
// 真机怪象：启动链钩子（同一入口）读到槽是对的（base+0x22d9260），两条指令后 `ldr` 却拿到 0。
// 与其继续猜"谁在两指令间清零"，不如替它干活：槽正常就直接替它写，槽异常就记录并跳过。
// 这样既能绕开，又能把"槽异常"这一事实打在日志里。
// np88：解析器入口 0x9fa4c8（stp 开头，可挂）——把游戏正在解析的**键名**打出来。
// 这是配置/内容路径的正门：键名能告诉我们游戏在找什么（配置项？文件？命令？）。
static int g_n_parse;
static uint64_t hook_parse(uint64_t *f) {
    if (++g_n_parse <= 40) {
        uint64_t k = f[FR_X1];
        char buf[64]; size_t i = 0;
        if (k > 0x10000ull && k < 0x800000000000ull) {
            const char *key = (const char *)(uintptr_t)k;
            for (; i < sizeof buf - 1 && key[i]; i++) buf[i] = key[i];
        }
        buf[i] = 0;
        klog("[解析] #%d 键=\"%s\"\n", g_n_parse, buf);
    }
    return 0;
}
// np84：线程消息处理的虚调用点（0xa1e32c：mov x0,x19 在 blr x9 之前，此时 x9=处理函数、x1=消息）。
// 目的：看这条"唯一被处理的消息"到底调到哪里——若调进我们的零桩，就是"收了但什么都没做"。
static int g_n_msg;
static uint64_t hook_msg(uint64_t *f) {
    uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
    uint64_t fn = f[9], msg = f[1];
    int in_img = fn >= base && fn < base + DM_IMG_MAIN_END;
    if (++g_n_msg <= 12)
        klog("[消息] 处理：函数=%s%#llx 消息=%#llx\n", in_img ? "+" : "(外)",
             (unsigned long long)(in_img ? fn - base : fn), (unsigned long long)msg);
    return 0;
}
static int g_n_setter;
static uint64_t hook_setter(uint64_t *f) {
    uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
    uint64_t slot = *(const uint64_t *)(uintptr_t)(base + 0x2280eb8);
    uint64_t val = f[FR_X0];
    if (++g_n_setter <= 8)
        klog("[设值] 0xa2469c 替身 #%d：槽=%#llx 值(x0)=%#llx\n",
             g_n_setter, (unsigned long long)slot, (unsigned long long)val);
    if (slot > 0x10000ull && slot < 0x800000000000ull)
        *(uint64_t *)(uintptr_t)slot = val;      // 原函数干的就是这件事
    return 0;
}
static void *g_boot_last_disp;   // 诊断（2026-10-20）：启动链全程盯 SIGSEGV disposition 的上一个值
static uint64_t hook_boot(uint64_t *f) {    uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
    uint64_t site = f[FR_SITE];
    uint64_t so = (site >= base && site < base + DM_IMG_MAIN_END) ? site - base : site;
    uint64_t lr = f[FR_X30];
    uint64_t lo = (lr >= base && lr < base + DM_IMG_MAIN_END) ? lr - base : 0;
    if (++g_n_boot <= 80) {
        // np67：顺带盯 0x2280eb8——崩点 +0xA246A4 的间接目标。装载后/静态初始化后都正确，
        // 崩时却是 0 ⇒ 是 nnMain 流程里被"覆盖写"清掉的（代码里没有 #0xeb8 的直接 store）。
        // 逐站读它，看哪一段之后变 0。
        uint64_t slot = *(const uint64_t *)(uintptr_t)(base + 0x2280eb8);
        klog("[启动链] #%d 进入 +%#llx（返回地址 +%#llx）｜槽 0x2280eb8=%#llx\n",
             g_n_boot, (unsigned long long)so, (unsigned long long)lo,
             (unsigned long long)slot);
    }
    {   // 诊断（2026-10-20）：本轮崩点（PostCalc 收 NULL 句柄）时我方 handler 没拦到、
        //   进程被 DFX 收尸。逐站读回 SIGSEGV disposition，只在**变化时**打一行——
        //   变化时刻就是"谁换掉了它"（musl 信号链/游戏自装崩溃处理器）。
        struct sigaction rb;
        sigaction(SIGSEGV, NULL, &rb);
        if ((void *)rb.sa_sigaction != g_boot_last_disp) {
            g_boot_last_disp = (void *)rb.sa_sigaction;
            klog("[信号] 启动链 #%d：SIGSEGV disposition 变为 %p\n",
                 g_n_boot, (void *)rb.sa_sigaction);
        }
    }
    return 0;
}
// ★ 决定性挂点（2026-10-18）：nnMain 0x73557c 是 `mov x22, x0`——x0 就是后面那个
//   `cbz x22, 0x735650` 的判断依据。为 0 就**跳过 7 个初始化调用**直接进主循环（真机已证）。
//   x0 来自 0x1074b2c / 0x1074b50 两个"取单例 + 调其虚表[0x10]"的小函数，
//   单例槽在 0x2289e60 / 0x2289e68。这里把"取值结果 + 两个单例槽 + 对象的虚表"一起打出来。
static int g_n_e26;
static uint64_t hook_e26(uint64_t *f) {
    if (++g_n_e26 <= 8) {
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        uint64_t v = f[FR_X0];
        uint64_t s60 = *(const uint64_t *)(uintptr_t)(base + 0x2289e60);
        uint64_t s68 = *(const uint64_t *)(uintptr_t)(base + 0x2289e68);
        uint64_t vt = 0, fn10 = 0;
        if (s60) {
            vt = *(const uint64_t *)(uintptr_t)s60;
            if (vt >= base && vt < base + DM_IMG_MAIN_END)
                fn10 = *(const uint64_t *)(uintptr_t)(vt + 0x10);
        }
        klog("[关键] 0x73557c x0(=x22)=%#llx  单例[0x2289e60]=%#llx [0x2289e68]=%#llx 虚表=%#llx 虚表[0x10]=%#llx\n",
             (unsigned long long)v, (unsigned long long)s60, (unsigned long long)s68,
             (unsigned long long)vt, (unsigned long long)fn10);
    }
    return 0;
}

// ★ 对照实验（2026-10-18 01:00）：0x73557c 的钩子"装上却不命中"，而紧邻的 0x735580 命中了，
//   这是个矛盾——不解决它，"7 个初始化被跳过"这个结论就不可信。
//   0x7355a8 是 `cbz x22` **不跳**时才执行的第一条指令（mov x0, x22）：
//   · 它命中 ⇒ 那 7 个初始化**其实跑了**，之前的结论要推翻（钩子在本区域不可靠）；
//   · 它不命中 ⇒ x22 确实为 0，结论成立，且钩子本身没问题。
static int g_n_e27;
static uint64_t hook_e27(uint64_t *f) {
    if (++g_n_e27 <= 4) klog("[对照] 0x7355a8 命中 ⇒ cbz x22 **没有**跳，7 个初始化真的执行了\n");
    return 0;
}

// ===== 影子「当前堆」（2026-10-18 01:05，用户批准方案 A）=====
// 真机结论（进度文档 §24/§25）：nnMain 在 0x7355a4 `cbz x22, 0x735650`，
// 而 x22 来自两个"取当前堆"的 getter（0x1074b2c / 0x1074b50）：
//     x8 = *(0x2289e60) → base+0x232f9e8；x0 = *(base+0x232f9e8)；
//     if (!x0) return 0;  return (*(*(x0)+0x10))(x0);
// 这两个变量在文件里是 0、全映像没有任何代码写它们 ⇒ 宿主上恒 0 ⇒
// x22==0 ⇒ nnMain 跳过 7 个初始化（a22508/a2262c/a03178/a03224/a025f0/a026a0/a2469c）直接进主循环。
//
// 修法（方案 A）：造一个最小"堆对象"挂上去，让 getter 能返回非 0。
//   · 分配仍然走我们已替换的宿主分配器（0x9f34d8/0x9f3464/0x9f33f0），它们**忽略**这个上下文指针；
//   · 所以这里只需要"非空 + 虚表[0x10] 能返回非空"，不要求它是真堆。
// 风险与判据：后续构造器若解引用它的字段，可能崩在下一个钉子（有 si_addr 可判）；
//   判据 = `[启动链]` 出现 +0xa22508 … +0xa2469c 这 7 条且不崩。
// 槽 0x80（下标 16）：**返回"带头部开销的分配尺寸"，不是指针**。依据 0xa09144 反汇编：
//     x8 = [heap]; x8 = [x8+0x80]; blr x8(heap, w20=size)   → x0
//     x24 = x0 & ~7;  if (x24 < 0x131) → 直接返 0（失败）     ← ★ 小尺寸会被判失败
//     x8 = [heap+0x30]; blr x8(heap, x24, w21*w20)           → 真正的分配（槽 0x30）
// 真机 np42：返回 8 ⇒ 8 < 0x131 ⇒ 立刻失败（GameConfig 拿不到，x20=0 崩）。
// ⇒ 真堆这里显然会加上"头部/对齐开销"。这里按 size + 0x130 返回（稳稳越过 0x131 门槛）。
static uint64_t fake_heap_size(void *self, uint64_t size) {
    (void)self;
    uint64_t v = (size + 0x130 + 7) & ~7ull;
    static int n;
    if (++n <= 8) klog("[影子堆] 取尺寸 size=%llu → %llu（含 0x130 开销）\n",
                       (unsigned long long)size, (unsigned long long)v);
    return v;
}
// 槽 0x30（下标 6）：真分配。参数按 (self, size, count) 解读（反汇编：x1=x24 尺寸、w2=w21*w20 个数）。
static uint64_t fake_heap_alloc(void *self, uint64_t size, uint64_t count) {
    (void)self;
    static int n;
    uint64_t total = size * (count ? count : 1);
    if (++n <= 12)
        klog("[影子堆] 分配 size=%llu count=%llu 合计=%llu\n",
             (unsigned long long)size, (unsigned long long)count, (unsigned long long)total);
    if (total == 0 || total > (64u << 20)) {
        if (n <= 12) klog("[影子堆] 合计不合理，返 0\n");
        return 0;
    }
    void *p = calloc(1, (size_t)total);
    if (n <= 12) klog("[影子堆] 分配 → %p\n", p);
    return (uint64_t)(uintptr_t)p;
}
#define FAKE_VT_QWORDS   64          // 虚表 64 槽（覆盖偏移 0..0x1f8，防 blr 到表外）
#define FAKE_HEAP_QWORDS 512         // 堆对象 4KB：真实堆结构体字段可能很多，给足
// ★ Task #4（2026-10-18 修）：这两个对象**必须放独立 calloc 的内存，不能放 .so 的 .bss**。
//   原因：游戏会把"堆对象"当自己的结构体往里写字段；对象若在 .bss 里，越界写入会直接
//   踩坏 .bss 的其它变量（g_n_e28 计数器输出垃圾值 -247789891 即此症状，§32.2 附注）。
//   放 calloc 内存里，最坏也只是踩坏我们自己那一块，不波及仪器计数器。
static uint64_t *g_fake_vt;          // 假虚表（运行期 calloc）
static uint64_t *g_fake_heap;        // 假堆对象（首字 = 虚表指针；运行期 calloc）
// 虚表[0x10]（下标 2）：返回"堆"本身。签名按 (self) → uint64_t。
static uint64_t fake_heap_get(void *self) {
    return (uint64_t)(uintptr_t)self;
}
// 其余槽：真机 np39 教训——只填了槽 2，结果在"堆对象"上做了另一次虚调用 ⇒ 槽里是 0 ⇒
// `blr x8` 直接 pc=0 崩。现在把 32 个槽**全填上**（每个槽一个独立桩，带槽号日志），
// 返回值一律 0：既不会 pc=0 崩，又能从日志看出"游戏对堆对象调了哪个方法"。
static int g_n_vtstub;
static uint64_t fake_vt_ret0(int idx) {
    if (++g_n_vtstub <= 40) klog("[影子堆] 虚表[+%#x] 被调用 → 返 0\n", idx * 8);
    return 0;
}
#define VT_STUB(I) static uint64_t vt_s_##I(void *p) { (void)p; return fake_vt_ret0(I); }
VT_STUB(0)  VT_STUB(1)  VT_STUB(2)  VT_STUB(3)  VT_STUB(4)  VT_STUB(5)  VT_STUB(6)  VT_STUB(7)
VT_STUB(8)  VT_STUB(9)  VT_STUB(10) VT_STUB(11) VT_STUB(12) VT_STUB(13) VT_STUB(14) VT_STUB(15)
VT_STUB(16) VT_STUB(17) VT_STUB(18) VT_STUB(19) VT_STUB(20) VT_STUB(21) VT_STUB(22) VT_STUB(23)
VT_STUB(24) VT_STUB(25) VT_STUB(26) VT_STUB(27) VT_STUB(28) VT_STUB(29) VT_STUB(30) VT_STUB(31)
VT_STUB(32) VT_STUB(33) VT_STUB(34) VT_STUB(35) VT_STUB(36) VT_STUB(37) VT_STUB(38) VT_STUB(39)
VT_STUB(40) VT_STUB(41) VT_STUB(42) VT_STUB(43) VT_STUB(44) VT_STUB(45) VT_STUB(46) VT_STUB(47)
VT_STUB(48) VT_STUB(49) VT_STUB(50) VT_STUB(51) VT_STUB(52) VT_STUB(53) VT_STUB(54) VT_STUB(55)
VT_STUB(56) VT_STUB(57) VT_STUB(58) VT_STUB(59) VT_STUB(60) VT_STUB(61) VT_STUB(62) VT_STUB(63)
static const uint64_t g_fake_vt_tab[FAKE_VT_QWORDS] = {
    (uint64_t)(uintptr_t)&vt_s_0,  (uint64_t)(uintptr_t)&vt_s_1,
    (uint64_t)(uintptr_t)&fake_heap_get,   // 下标 2 = 偏移 0x10：取堆（必须返非 0）
    (uint64_t)(uintptr_t)&vt_s_3,  (uint64_t)(uintptr_t)&vt_s_4,
    (uint64_t)(uintptr_t)&vt_s_5,  (uint64_t)(uintptr_t)&fake_heap_alloc,  // 下标 6 = 偏移 0x30：真分配
    (uint64_t)(uintptr_t)&vt_s_7,  (uint64_t)(uintptr_t)&vt_s_8,
    (uint64_t)(uintptr_t)&vt_s_9,  (uint64_t)(uintptr_t)&vt_s_10,
    (uint64_t)(uintptr_t)&vt_s_11, (uint64_t)(uintptr_t)&vt_s_12,
    (uint64_t)(uintptr_t)&vt_s_13, (uint64_t)(uintptr_t)&vt_s_14,
    (uint64_t)(uintptr_t)&vt_s_15, (uint64_t)(uintptr_t)&fake_heap_size,  // 下标 16 = 偏移 0x80：取尺寸
    (uint64_t)(uintptr_t)&vt_s_17, (uint64_t)(uintptr_t)&vt_s_18,
    (uint64_t)(uintptr_t)&vt_s_19, (uint64_t)(uintptr_t)&vt_s_20,
    (uint64_t)(uintptr_t)&vt_s_21, (uint64_t)(uintptr_t)&vt_s_22,
    (uint64_t)(uintptr_t)&vt_s_23, (uint64_t)(uintptr_t)&vt_s_24,
    (uint64_t)(uintptr_t)&vt_s_25, (uint64_t)(uintptr_t)&vt_s_26,
    (uint64_t)(uintptr_t)&vt_s_27, (uint64_t)(uintptr_t)&vt_s_28,
    (uint64_t)(uintptr_t)&vt_s_29, (uint64_t)(uintptr_t)&vt_s_30,
    (uint64_t)(uintptr_t)&vt_s_31, (uint64_t)(uintptr_t)&vt_s_32,
    (uint64_t)(uintptr_t)&vt_s_33, (uint64_t)(uintptr_t)&vt_s_34,
    (uint64_t)(uintptr_t)&vt_s_35, (uint64_t)(uintptr_t)&vt_s_36,
    (uint64_t)(uintptr_t)&vt_s_37, (uint64_t)(uintptr_t)&vt_s_38,
    (uint64_t)(uintptr_t)&vt_s_39, (uint64_t)(uintptr_t)&vt_s_40,
    (uint64_t)(uintptr_t)&vt_s_41, (uint64_t)(uintptr_t)&vt_s_42,
    (uint64_t)(uintptr_t)&vt_s_43, (uint64_t)(uintptr_t)&vt_s_44,
    (uint64_t)(uintptr_t)&vt_s_45, (uint64_t)(uintptr_t)&vt_s_46,
    (uint64_t)(uintptr_t)&vt_s_47, (uint64_t)(uintptr_t)&vt_s_48,
    (uint64_t)(uintptr_t)&vt_s_49, (uint64_t)(uintptr_t)&vt_s_50,
    (uint64_t)(uintptr_t)&vt_s_51, (uint64_t)(uintptr_t)&vt_s_52,
    (uint64_t)(uintptr_t)&vt_s_53, (uint64_t)(uintptr_t)&vt_s_54,
    (uint64_t)(uintptr_t)&vt_s_55, (uint64_t)(uintptr_t)&vt_s_56,
    (uint64_t)(uintptr_t)&vt_s_57, (uint64_t)(uintptr_t)&vt_s_58,
    (uint64_t)(uintptr_t)&vt_s_59, (uint64_t)(uintptr_t)&vt_s_60,
    (uint64_t)(uintptr_t)&vt_s_61, (uint64_t)(uintptr_t)&vt_s_62,
    (uint64_t)(uintptr_t)&vt_s_63,
};
// 指针合法性快查：np54 实证游戏侧越界写会把 .bss 指针槽污染成 94 一类的小值，
// 用之前必须先验（否则解引用即崩）。合法区间：远大于 0、小于用户态上限。
static int k_ptr_sane(const void *p) {
    uintptr_t v = (uintptr_t)p;
    return p && v > 0x10000ull && v < 0x800000000000ull;
}
static void k_fake_heap_install(uint8_t *img) {
    // 自愈：指针槽若被污染（np54：g_fake_parse_obj 被写成 0x5e），重新分配再挂。
    if (!k_ptr_sane(g_fake_vt) || !k_ptr_sane(g_fake_heap)) {
        klog("[影子堆] ⚠ 指针槽被污染（vt=%p heap=%p），重新分配\n",
             (void *)g_fake_vt, (void *)g_fake_heap);
        g_fake_vt   = calloc(FAKE_VT_QWORDS, sizeof(uint64_t));
        g_fake_heap = calloc(FAKE_HEAP_QWORDS, sizeof(uint64_t));
    }
    if (!g_fake_vt) {
        g_fake_vt   = calloc(FAKE_VT_QWORDS, sizeof(uint64_t));
        g_fake_heap = calloc(FAKE_HEAP_QWORDS, sizeof(uint64_t));
    }
    memcpy(g_fake_vt, g_fake_vt_tab, FAKE_VT_QWORDS * sizeof(uint64_t));
    g_fake_heap[0] = (uint64_t)(uintptr_t)g_fake_vt;
    // 两个"当前堆"变量（模块偏移，来自 RELA addend 反查：0x2289e60→0x232f9e8、0x2289e68→0x232f9f0）
    *(uint64_t *)(img + 0x232f9e8) = (uint64_t)(uintptr_t)g_fake_heap;
    *(uint64_t *)(img + 0x232f9f0) = (uint64_t)(uintptr_t)g_fake_heap;
    klog("[影子堆] 已挂：堆=%p（calloc %dKB）虚表=%p（%d 槽，槽[0x10]=取堆）\n",
         (void *)g_fake_heap, FAKE_HEAP_QWORDS * 8 / 1024, (void *)g_fake_vt, FAKE_VT_QWORDS);
}

static int g_n_e28;
static uint64_t hook_e28(uint64_t *f) {
    // ★ 挂的是 worker **0xa051c4**（不是 0xa05110——后者只是"创建/取单例"：
    //   里面 bl 0x9f3464 分配 0x7c8 字节、bl 0xa08ef4 构造，然后返回那个单例）。
    //   崩点 0xa05324 在 worker 里：`mov x23, x2`（第 3 个参数）随后 `ldr w8,[x23,#8]`。
    //   这里把 4 个入参与调用者打出来，看"是谁把 0 传进来的"。
    // ⚠ np58 教训：钩子里**别在静态初始化期做 calloc/多行日志/解引用客体指针**——会卡死
    //   （真机卡在 ctor #385）。这里保持 np57 的形态：静态计数器 + 只多打两行映像内标量。
    if (++g_n_e28 <= 12) {
        uint64_t base = (uint64_t)(uintptr_t)g_k_m.base;
        uint64_t lr = f[FR_X30];
        uint64_t lo = (lr >= base && lr < base + DM_IMG_MAIN_END) ? lr - base : 0;
        // 帧下标：FR_X0=0/FR_X1=1/FR_X2=2，FR_NZCV=18/FR_X29=20/FR_X30=21/FR_SITE=22
        // ⇒ x3..x17 就是 3..17（没有专门宏，直接用下标）
        klog("[建对象] worker 0xa051c4#%d：x0=%#llx x1=%#llx **x2=%#llx** w3=%#llx 调用者=+%#llx\n",
             g_n_e28, (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X1],
             (unsigned long long)f[FR_X2], (unsigned long long)f[3],
             (unsigned long long)lo);
        // np58：后段崩点 (+0xA05620) 的关键标量——**只读映像内的槽、不碰客体指针**
        // （上一版在这里解引用客体指针 + 多打 6 行，真机卡死在静态初始化 #385，已回退）。
        {
            const uint8_t *im = (const uint8_t *)(uintptr_t)g_k_m.base;
            uint64_t sa = *(const uint64_t *)(im + 0x227f900);   // 槽 → .bss 变量（或运行期换成的堆对象）
            uint64_t sb = *(const uint64_t *)(im + 0x227f8e8);
            int sa_in = sa >= base && sa < base + DM_IMG_MAIN_END;
            int sb_in = sb >= base && sb < base + DM_IMG_MAIN_END;
            klog("[建对象] 槽A(0x227f900)=%#llx%s 槽B(0x227f8e8)=%#llx%s\n",
                 (unsigned long long)sa, sa_in ? "（映像内）" : "（非映像）",
                 (unsigned long long)sb, sb_in ? "（映像内）" : "（非映像）");
            if (sa_in) {
                uint64_t fn = *(const uint64_t *)(uintptr_t)sa;
                klog("[建对象] 槽A内容=%#llx%s\n", (unsigned long long)fn,
                     (fn >= base && fn < base + DM_IMG_MAIN_END) ? "（是映像内函数）" : "（非映像）");
            }
        }
    }
    return 0;
}

// 探针用：把一段客体内存打成十六进制（同进程同地址空间，可直接解引用）。
// 用途：看清"游戏传进来的对象/字符串"到底是什么——键名、字符串、小结构体都靠它认。
static void k_dump_hex(const char *tag, const void *p, size_t n) {
    char line[200];
    const unsigned char *b = (const unsigned char *)p;
    if (!p) { klog("%s（空指针）\n", tag); return; }
    size_t pos = (size_t)snprintf(line, sizeof line, "%s %p:", tag, p);
    for (size_t i = 0; i < n && pos + 4 < sizeof line; i++)
        pos += (size_t)snprintf(line + pos, sizeof line - pos, " %02x", b[i]);
    klog("%s\n", line);
}

// 金丝雀 + 计数器快照：np49 实证 .so 的 .bss 会被**点状污染**——g_n_e28/e29 变成垃圾值
//（-247770147），而 取尺寸/分配/虚表 等其它 .bss 计数器完全正常。后果很阴：
//"if (++g_n <= N) 才打印" 的守卫在计数器被写成大正数后**静默吞掉日志**，
// 伪装成"钩子没命中"（0x9fa4c8 探针 np49 零命中即此）。
// 金丝雀紧挨着声明，用来看污染的形状（写哪个、写多大范围）。
static int g_canary0, g_canary1, g_canary2, g_canary3;
static int g_canary4, g_canary5, g_canary6, g_canary7;
static int g_n_e29;
static void k_log_counters(const char *tag) {
    klog("[计数器%s] e28=%d e29=%d 金丝雀=%d %d %d %d %d %d %d %d\n",
         tag, g_n_e28, g_n_e29,
         g_canary0, g_canary1, g_canary2, g_canary3,
         g_canary4, g_canary5, g_canary6, g_canary7);
}

// 0x9fa4c8：通用「键:值」解析器（全映像 8 处调用）。当前崩点链的**最上游**：
//   0xa05fa4(经函数指针进来) 用它解析一个字符串 → 返回 0
//   → worker 0xa051c4 的 x2 = 0 → 崩在 0xa05324（`ldr w8,[x23,#8]`）。
// 静态追查到此断了（0xa05fa4 无直接调用者，只能经虚表/函数指针进来），
// 所以运行时取证：把两个入参对象 dump 出来，看清"在解析哪个键、键从哪来"。
// ★ 自 np50 起**无条件打印**：不再用 g_n_e29 计数抑制（.bss 计数器会被踩，见上方注释）。
// np50 首战：对象布局已认清——x1+0x20 = 字符串数据指针、x1+0x28 = 尺寸（本回合 16 字节）。
// 本轮把**串内容本身**打出来，直接回答"在解析哪个键"。
static uint64_t hook_e29(uint64_t *f) {
    const unsigned char *o = (const unsigned char *)(uintptr_t)f[FR_X1];
    klog("[键解析] 0x9fa4c8：x0=%#llx x1=%#llx 金丝雀地址=%p\n",
         (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X1], (void *)&g_canary0);
    k_dump_hex("[键解析] x0 对象", (const void *)(uintptr_t)f[FR_X0], 32);
    k_dump_hex("[键解析] x1 对象", o, 64);
    if (o) {
        uint64_t data = *(const uint64_t *)(o + 0x20);
        uint64_t sz   = *(const uint64_t *)(o + 0x28);
        klog("[键解析] 疑似字符串：数据=%#llx 尺寸字段=%#llx（低32=cap 高32=len？）\n",
             (unsigned long long)data, (unsigned long long)sz);
        // +0x28 实际是 {cap=0x1000, len=0x10} 两半（np51 实证），别拿它当整体尺寸卡条件
        if (data > 0x10000 && data < 0x800000000000ull)
            k_dump_hex("[键解析] 串内容", (const void *)(uintptr_t)data, 48);
    }
    k_log_counters("解析时");
    return 0;
}

// 0xa05fa4：解析派发 = worker 0xa051c4 的**调用者**（经函数指针进来，全映像无直接 bl）。
// 它的 x2 会喂给构造函数 0xa112fc（`mov x1, x2`）⇒ 极可能就是"要解析的字符串"本体。
// （np52 证实 0x9fa4c8 的 x1+0x20 只是二级指针：0x5b96a6d000 → {0x5b96a64600,…}，
//   继续跟指针链不如回到上游看 x2。）
static int g_n_e30;
static uint64_t hook_e30(uint64_t *f) {
    const unsigned char *s = (const unsigned char *)(uintptr_t)f[FR_X2];
    klog("[派发] 0xa05fa4：x0=%#llx x1=%#llx x2=%#llx\n",
         (unsigned long long)f[FR_X0], (unsigned long long)f[FR_X1],
         (unsigned long long)f[FR_X2]);
    k_dump_hex("[派发] x2 对象", s, 32);
    if (s) {
        uint64_t p0 = *(const uint64_t *)s;   // std::string 长形态首字 = 数据指针
        if (p0 > 0x10000 && p0 < 0x800000000000ull)
            k_dump_hex("[派发] x2[0] 指向", (const void *)(uintptr_t)p0, 48);
    }
    return 0;
}

// ★ 崩溃点仪器（2026-10-20）：CallbackemitterPostCalc_Basic_0（0x103b1f0）入口。
//   机理已坐实（faultlog ESR=读故障 + si_addr=0 + x0=0 ⇒ 第三条指令 `ldr x20,[x0]` 读 0 崩；
//   诊断串 EFF_ERR[CallbackemitterPostCalc_Basic_0]；兄弟 Initialize/Finalize 同构吃句柄）：
//   **调用方递了 NULL 句柄**。静态只确认"必走某对象 +0xc10 的 blr"（构造器 0x1037630 是
//   唯一静态引用、全映像无直接 bl），231 处同偏移加载静态无法区分 ⇒ 挂入口记
//   x0（句柄）与 x30（LR=调用方返回址）。钩子在入口、崩指令之前触发 ⇒ **致命调用必被捕获**；
//   klog 即 fflush ⇒ 崩前最后一条已落盘。离线：x30-4 反汇见 blr，其前若干条内的
//   `ldr x?,[x?,#0xc10]` 即调用者函数与对象来源。
//   ⚠ 只记地址、不解引用 *x0——非致命调用的 x0 可能是野指针，钩子自己解引用会污染现场。
static int g_n_pcalc;
static uint64_t hook_pcalc(uint64_t *f) { frame_log("PostCalc 0x103b1f0", f, &g_n_pcalc); return 0; }

// 0x9fa4c8 的**替身**（np54 起）：原实现是"建临时对象 → 按 ':' 切分 → 遍历已注册解析器"，
// 在宿主上一路缺数据、恒返回 0 ⇒ worker 的 x2 = 0 ⇒ 崩在 0xa05324（`ldr w8,[x23,#8]`）。
// 按 §八「影子对象」打法：宿主直接返回一个**合法的全零对象**（calloc 8KB），
// 让流程越过"解析失败"这道坎——每越一坎，下一个真实需求就会暴露（与影子堆同一逻辑）。
// ⚠ np54 教训：对象指针**不能存 .bss 静态变量**——np54 实证 `g_fake_parse_obj` 被游戏侧
//   越界写污染成 94(0x5e)，`if (!ptr)` 检查被绕过 ⇒ 把 0x5e 当对象返回 ⇒ 原样崩在同一点。
//   现在每次**局部变量现分配**（.bss 只留一份副本仅供比对，不参与判断）。
static void *g_fake_parse_obj;
// np215：键解析替换开关。0 = 让原实现跑（默认，见安装处注释）；1 = 退回全零对象替身。
static int g_parse_repl = 0;
// np215：原实现放行时的观察钩子——只打键名与参数，不改任何东西。
static int g_n_pw;
static uint64_t hook_parse_watch(uint64_t *f) {
    if (++g_n_pw <= 24) {
        uint64_t so = f[FR_X1];
        char buf[64]; size_t i = 0; const char *key = NULL;
        if (so > 0x10000ull && so < 0x800000000000ull) {
            uint64_t k = *(const uint64_t *)(uintptr_t)so;
            if (k > 0x10000ull && k < 0x800000000000ull) key = (const char *)(uintptr_t)k;
        }
        if (key) for (; i < sizeof buf - 1 && key[i]; i++) buf[i] = key[i];
        buf[i] = 0;
        char l[160];
        snprintf(l, sizeof l, "np215键解析#%d 键=\"%s\" 表=%#llx\n", g_n_pw, buf,
                 (unsigned long long)f[FR_X0]);
        note_append("dm_gate.txt", l);
    }
    return 0;
}
static uint64_t hook_rep_parse(uint64_t *f) {
    void *old = g_fake_parse_obj;                 // 旧值：0=从没写过；94=被污染（np54 实证）
    void *obj = calloc(1, 0x2000);                // 局部现分配，不依赖 .bss
    g_fake_parse_obj = obj;                       // 留档（便于事后比对），不用于判断
    {   // np88/np89：把**键名**打出来。注意 x1 不是字符串本身，而是**字符串对象**
        //   （sead::BufferedSafeString 布局：+0 数据指针、+4 长度、+8 容量）——真正的串在 [x1+0]。
        uint64_t so = f[FR_X1];
        char buf[64]; size_t i = 0; const char *key = NULL;
        if (so > 0x10000ull && so < 0x800000000000ull) {
            uint64_t k = *(const uint64_t *)(uintptr_t)so;      // [x1+0] = 数据指针
            if (k > 0x10000ull && k < 0x800000000000ull) key = (const char *)(uintptr_t)k;
        }
        if (key) for (; i < sizeof buf - 1 && key[i]; i++) buf[i] = key[i];
        buf[i] = 0;
        if (++g_n_parse <= 3 && so > 0x10000ull && so < 0x800000000000ull) {
            const uint64_t *q = (const uint64_t *)(uintptr_t)so;
            klog("[键解析·替身] 串对象布局：%016llx %016llx %016llx %016llx\n",
                 (unsigned long long)q[0], (unsigned long long)q[1],
                 (unsigned long long)q[2], (unsigned long long)q[3]);
        }
        klog("[键解析·替身] 键=\"%s\"（串对象=%#llx）旧值=%p 新对象=%p\n", buf,
             (unsigned long long)so, old, obj);
    }
    return (uint64_t)(uintptr_t)obj;
}

// ===== 替换式安装器：宿主替函数干活（返回值经 x0 带回），**原实现一条都不执行** =====
// ⚠ 早先这里把桩的"回跳槽"填成 `b 站点+4`——被替换的是**函数入口**，站点+4 就是原实现的
//   第二条指令，于是变成"宿主跑完、原实现接着跑"（双执行）：首条 stp 被跳过 ⇒ 栈帧少 0x20、
//   原实现里的 mov w19,w2 直接写坏 x19，最后返回的 x0 是原实现的结果（真机实证：x0 与 x23 双 0，
//   宿主的指针被丢掉）。现在桩自己 `ret`（回 x30 = 调用点下一条），安装器不再改回跳槽。
extern const char hook_rep_begin[], hook_rep_end[], hook_rep_host[], hook_rep_site[];
static void k_install_replace_hook(uint8_t *img, uint32_t site_off, void *host_fn, const char *what) {
    const char *b0 = hook_rep_begin;
    size_t n = (size_t)(hook_rep_end - hook_rep_begin);
    uint8_t *dst = img + DM_IMG_MAIN_CAVE_OFF + (size_t)g_hook_n * HOOK_STRIDE;
    if (n > HOOK_STRIDE) { klog("[替换] %s 桩过长(%zu > %u)，跳过\n", what, n, HOOK_STRIDE); return; }
    if (g_hook_n >= (int)HOOK_MAX) { klog("[替换] 洞已满(%d 格)，跳过 %s\n", g_hook_n, what); return; }
    memcpy(dst, b0, n);
    *(uint64_t *)(dst + (hook_rep_host - b0)) = (uint64_t)(uintptr_t)host_fn;
    *(uint64_t *)(dst + (hook_rep_site - b0)) = (uint64_t)(uintptr_t)(img + site_off);
    {   // 站点：b dst
        int64_t d = ((int64_t)(uintptr_t)dst - (int64_t)(uintptr_t)(img + site_off)) / 4;
        if (d < -(1 << 25) || d >= (1 << 25)) { klog("[替换] %s 站点超程，跳过\n", what); return; }
        *(uint32_t *)(img + site_off) = 0x14000000u | (uint32_t)(d & 0x03FFFFFFu);
    }
    klog("[替换] %s：site %#x → 洞 %#zx\n", what, site_off, (size_t)(dst - img));
    g_hook_n++;
}
// 宿主替分配器干活：x0 = 请求大小，返回真实内存（calloc 清零）
static uint64_t g_rep_n;
static int g_n_big;      // np106/np106b：大块分配计数（≥0x400，含管理器对象级别的）
static uint64_t g_big_sz[24];
static uint64_t g_big_lr[24];
static unsigned g_big_seq[24];
static uint64_t host_alloc_replace(uint64_t *f) {
    uint64_t size = f[FR_X0];
    void *p = calloc(1, (size_t)size);
    if (++g_rep_n <= 8)
        klog("[替换] 分配 %llu 字节 → %p（站点 %#llx 上下文 %#llx）\n",
             (unsigned long long)size, p,
             (unsigned long long)f[FR_SITE], (unsigned long long)f[FR_X1]);
    // np106：**启动包装载门**（[*(0x2276168)+0x419]）所在的管理器对象由这里分配；
    //   它在启动链前就已带 [+0x419]=4 ⇒ 分配后某段代码写了 4。记录所有 ≥0x400 的分配
    //   与站点，据尺寸（≥0x41c）与构造现场定位写入者。
    // np106b：**只记录、不打印**（np58 教训：分配器钩子里在构造器期打日志 ⇒ 卡死 ctor #385）。
    //   构造器跑完后由 dump_big_allocs() 统一输出。
    if (size >= 0x400 && g_n_big < 24) {
        uint64_t lr = f[FR_X30];
        g_big_sz[g_n_big] = size;
        g_big_lr[g_n_big] = lr;
        g_big_seq[g_n_big] = g_rep_n;      // 第几次分配（便于和日志行号对齐）
        g_n_big++;
    }
    return (uint64_t)(uintptr_t)p;
}

// np106b：大分配记录落地（构造器期之后调用，此时打日志才安全）
// np143：改成**增量输出**（只打新增条目）——现在会在多个采样点反复调用
//   （静态初始化后 / nnMain 运行期帧线程），每次都全量重打会淹日志。
static void dump_big_allocs(void) {
    static int done;                 // 已打印到第几条
    uint64_t b2 = (uint64_t)(uintptr_t)g_k_m.base;
    for (int i = done; i < g_n_big; i++) {
        int in = g_big_lr[i] >= b2 && g_big_lr[i] < b2 + DM_IMG_MAIN_END;
        klog("[大分配] #%d %#llx 字节（第 %d 次分配）；站点=%s%#llx\n", i,
             (unsigned long long)g_big_sz[i], (int)g_big_seq[i],
             in ? "+" : "(外)",
             (unsigned long long)(in ? g_big_lr[i] - b2 : g_big_lr[i]));
    }
    done = g_n_big;
}
// 批量挂点：扫全映像，找出所有指向 target 的 **bl（调用）与 b（尾调用）** 站点并全部挂上。
// 为什么连 b 也要挂：崩链上的调用可能是尾调用（不更新 x30），只扫 bl 会把它们漏掉。
// 为什么要扫两块：改写器把每个 svc 门点改成跳进**补丁区**的蹦床，所以补丁区是真代码——
// 只扫 text 会漏站点（离线用 tools/dm_branch.py 核实过）。代码洞本身不扫：里面是我们自己的桩。
static void k_hook_all_branchers(uint8_t *img, uint32_t target_off, void *host_fn, const char *what, int cap) {
    const uint32_t lo[2] = {0, DM_IMG_MAIN_PATCH_OFF};
    const uint32_t hi[2] = {DM_IMG_MAIN_TEXT_END, DM_IMG_MAIN_CAVE_OFF};
    uint32_t n = 0, nbl = 0, nb = 0;
    for (int r = 0; r < 2; r++) {
        for (uint32_t a = lo[r]; a + 4 <= hi[r] && n < (uint32_t)cap; a += 4) {
            uint32_t w = *(uint32_t *)(img + a);
            uint32_t op = w & 0xFC000000u;
            if (op != 0x94000000u && op != 0x14000000u) continue;      // bl / b
            int32_t imm = (int32_t)(w & 0x03FFFFFFu);
            if (imm & 0x02000000) imm -= 0x04000000;
            if ((int64_t)a + (int64_t)imm * 4 != (int64_t)target_off) continue;
            k_install_hook(img, a, host_fn, what);
            n++;
            if (op == 0x94000000u) nbl++; else nb++;
        }
    }
    klog("[插桩] %s：共挂 %u 个站点（bl %u / b 尾调用 %u，上限 %d）\n", what, n, nbl, nb, cap);
}
// ⭐ np171（2026-10-21）：**"当前堆"补口**——内容门 1/2 的真正开关。
// 依据（本轮真机 + 离线反汇编双向坐实；⚠ np169 曾把这处误判成"启动信息"，已纠正）：
//   ① `0x1074b50` / `0x1074b2c` = **取当前堆**（`x0 = *(*holder)`；非空则虚调用 [vt+0x10] 返回；
//      两个槽互为回退）。证据：全映像 **127 个调用点**、nnMain 自己也调，调用形态一律
//      `x = [this+8]; cbnz x,skip; bl getter`（"给了堆就用给的，没给取当前的"）。
//   ② 门 1 = `0xdb9334 ldr x20,[this+8]; cbnz → 继续; bl 0x1074b50; 0xdb9344 cbz x20 → 早退`，
//      拿到的 x20 立刻当堆用：`0xdb935c bl 0xf34f9c` → 内部 `w0=0x158, x1=x20, w2=8, bl 0x9f3464`
//      （分配 StarterPackMgr 本体 0x158 字节；0x9f3464 是我们替换的分配器，它只 calloc、不看堆参数）。
//   ③ ⭐ **门 2 是门 1 的函数**：w26 在 0xdb9284 里**只有一处写**（`0xdb9368 mov w26,#1`，位于门 1
//      通过之后），三个读点 0xdb93e8 / 0xdb977c / **0xdb980c** 全由它决定 ⇒ **堆非空 ⇒ 门 2 自动放行**。
//   ⇒ 真正的门只有一道：**当前堆有没有**。np169 把这里补成"零填充结构"（语义错），被当堆取走后
//      读虚表 +0x80 当场崩（si_addr=0x80，pc=img+0xa091a0）——这就是 np170 基线的早崩源。
// 修法：holder → (虚表 +0x10) → **影子分配器 g_alloc_obj**（64 槽虚表：+0x10/+0x30 分配、
//   +0x80 报尺寸、+0x88 当 bool），而不是零填充结构。
static uint64_t g_curheap_vt[8];       // "当前堆"持有者的虚表：+0x10 = 取堆
static uint64_t g_curheap_obj[2];      // 持有者对象 { &虚表, 0 }
static uint64_t g_curheap_holder[1];   // 全局槽指向它，再解一层
static uint64_t curheap_get(void *self) {
    (void)self;
    static int n;
    if (++n <= 4) klog("[当前堆] ★ 取当前堆（虚方法 +0x10，第 %d 次）→ 影子分配器 %p\n",
                       n, (void *)&g_alloc_obj[0]);
    return (uint64_t)(uintptr_t)&g_alloc_obj[0];
}

static void k_inject_heap(uint8_t *img) {
    for (int i = 0; i < DM_ALLOC_VTBL_QWORDS; i++)
        g_alloc_vtbl[i] = (uint64_t)(uintptr_t)&vtbl_fallback;   // 先全铺兜底（记录调用点、返回 0）
    g_alloc_vtbl[2]  = (uint64_t)(uintptr_t)&vtbl_slot2_alloc;    // +0x10 按尺寸分配（np164）
    g_alloc_vtbl[6]  = (uint64_t)(uintptr_t)&host_alloc;         // +0x30 分配内存
    g_alloc_vtbl[16] = (uint64_t)(uintptr_t)&vtbl_slot16_size;   // +0x80 报尺寸 ⇒ 0x1000
    g_alloc_vtbl[17] = (uint64_t)(uintptr_t)&vtbl_slot17_ok;     // +0x88 当 bool 用 ⇒ 返 1
    g_alloc_obj[0] = (uint64_t)(uintptr_t)&g_alloc_vtbl[0];
    g_alloc_obj[1] = (uint64_t)(uintptr_t)&g_alloc_obj[0];   // +8 自指：0xa0e2f8 的兜底路径取到同一对象
    // ⚠ 真机实证（2026-10-17）：跑到 0xa0e2f8 时这个自指值**已被清成 0**（游戏会写它），
    //   所以兜底路径不可靠——改由 GetTlsValue 的影子对象 [+0xb0] 顶上（见 g_tls_shadow）。
    g_alloc_holder[0] = (uint64_t)(uintptr_t)&g_alloc_obj[0];
    g_ctx_holder[0] = (uint64_t)(uintptr_t)&g_ctx_obj[0];
    // np171：当前堆补口（门 1 的唯一开关；门 2 的 w26 由门 1 决定）——见 g_curheap_vt 处注释
    g_curheap_vt[0x10 / 8] = (uint64_t)(uintptr_t)&curheap_get;
    g_curheap_obj[0] = (uint64_t)(uintptr_t)&g_curheap_vt[0];
    g_curheap_holder[0] = (uint64_t)(uintptr_t)&g_curheap_obj[0];
    *(uint64_t *)(img + 0x2289e68) = (uint64_t)(uintptr_t)&g_curheap_holder[0];
    *(uint64_t *)(img + 0x2289e60) = (uint64_t)(uintptr_t)&g_curheap_holder[0];
    g_tls_shadow[0xb0 / 8] = (uint64_t)(uintptr_t)&g_alloc_obj[0];   // 影子对象 [+0xb0] = 分配器对象
    g_alloc_obj_ptr = (void *)&g_alloc_obj[0];   // np211g：转发给 shadow_install 填影子块 +0xb0
    *(uint64_t *)(img + 0x226f0a0) = (uint64_t)(uintptr_t)&g_alloc_holder[0];   // 分配器持有者
    *(uint64_t *)(img + 0x22783d8) = (uint64_t)(uintptr_t)&g_ctx_holder[0];     // 线程上下文持有者
    // 工厂 0x1f8 的入参契约（np141 符号级定谳；客体构造器 0x103b83c 全函数反汇编封闭）：
    //   +0x00 虚表指针：构造器 ldr x8,[x20] 后读 [vt+0x10] 一个标志字节（0x103b994-0x103b99c）
    //   +0x18 子对象链：读 [sub+0x18] 虚方法，两次 blr（0x103b8b8/0x103b8d4，日志两次"[管理器] 虚方法被调用"）
    //   +0x20 名字串  ：其上做手工 strlen（分隔符全局指向 0x00；钳位 63）定拷贝长度，
    //                   拷贝源是 [+0x18+8]；空串 ⇒ 长度 0 ⇒ musl memcpy 不 deref NULL 源 ⇒ 安全通过
    // 构造器 0x103b9a0 之后只操作新对象、0x1f8 返回后亦然 ⇒ 契约就这三项，无隐藏字段。
    g_mgr_obj[0x00 / 8] = (uint64_t)(uintptr_t)&g_mgr_vt[0];
    g_mgr_sub[0x18 / 8] = (uint64_t)(uintptr_t)&mgr_method;
    g_mgr_obj[0x18 / 8] = (uint64_t)(uintptr_t)&g_mgr_sub[0];
    g_mgr_obj[0x20 / 8] = (uint64_t)(uintptr_t)g_mgr_name;
    klog("[mgr-契约] P0-5：+0x00 虚表=%p +0x18 链=%p +0x20 空串=%p（strlen=0）\n",
         (void *)&g_mgr_vt[0], (void *)&g_mgr_sub[0], (void *)g_mgr_name);
    // nnMain 路径的 sead::ThreadMgr（见 g_tm_a 处注释：这个槽此前漏注 ⇒ 崩 0x9FE9A8）
    for (int i = 0; i < DM_TM_VTBL_QWORDS; i++) g_tm_vtbl[i] = (uint64_t)(uintptr_t)&tm_method;
    g_tm_obj[0] = (uint64_t)(uintptr_t)&g_tm_vtbl[0];
    g_tm_a[0]   = 1;                                  // 标志位：非零才会往下走
    g_tm_a[1]   = (uint64_t)(uintptr_t)&g_tm_obj[0];
    *(uint64_t *)(img + 0x227aac0) = (uint64_t)(uintptr_t)&g_tm_a[0];
    // *(0x227df98) → 模块 .bss 的 0x22d9198；模块代码从不写它（见 g_cb_obj 处注释）
    g_cb_obj[0x20 / 8] = (uint64_t)(uintptr_t)&cb_stub;
    *(uint64_t *)(img + 0x22d9198) = (uint64_t)(uintptr_t)&g_cb_obj[0];
    klog("[注入自检] g_alloc_obj=%p [0]=%#llx [1]=%#llx；g_tls_shadow=%p [+0xb0]=%#llx\n",
         (void *)&g_alloc_obj[0], (unsigned long long)g_alloc_obj[0], (unsigned long long)g_alloc_obj[1],
         (void *)&g_tls_shadow[0], (unsigned long long)g_tls_shadow[0xb0 / 8]);
}

// ★ 跑模块自己的静态初始化（全局对象构造）。深跑直接跳进 nnMain ⇒ 跳过了这一步，
//   data 段里那些全局对象全是空的（真机实证：偏移 0x22D8AA0 的对象首字=0 ⇒ 崩在 blr 野指针）。
// 依据：MOD0 动态段里的 DT_INIT / DT_INIT_ARRAY / DT_INIT_ARRAYSZ（离线解析出的值）：
//   DT_INIT = 0x24；DT_INIT_ARRAY = 0x2293390；DT_INIT_ARRAYSZ = 0x21f8 ⇒ 1085 个构造器。
#define DM_IMG_MAIN_INIT_OFF        0x24u
#define DM_IMG_MAIN_INIT_ARRAY_OFF  0x2293390u
#define DM_IMG_MAIN_INIT_ARRAY_SIZE 0x21f8u
// 已知会崩的构造器（先跳过以验证因果，别让它们污染后面）：
//   #184 = `*(0x22d84a0)=2π`，崩因是读 nn::util::FloatPi 的 GLOB_DAT 槽（0x2280648）为空指针
//         （2026-10-18：槽已在 k_fill_data_imports 补上 ⇒ 放出来重跑，静观是否真过）。
//         旧注释"把 x20 当对象指针"是早前误判——反汇编显示它根本不碰 x20。
//   #206 待查（0xb2a370，一个注册表构造器）。
// 2026-10-18：#206（0xb2a370）**放行重试**。它此前"已知会崩"被跳过，但新的线索表明
// 它可能在填某个注册表/全局对象（用了 0x2264288/0x2264448 等类型全局，且扫一个找 0x3c 的表）。
// gate 会兜底：崩了自动跳过并打印 pc/si_addr，成功则可能顺带解开当前的解析失败。
// 若确认无用可改回 return i == 206。
static int k_skip_ctor(size_t i) { (void)i; return 0; }

// ───────── 通用：跑某个模块自己的 init_array（S1，2026-10-22）─────────
// 与 k_run_module_init 的区别：那个用**生成头里的 main 常量**，这个用模块动态段里的
// DT_INIT_ARRAY/DT_INIT_ARRAYSZ ⇒ 任何模块都能跑。逐个武装 + 逐个编号 + 崩了跳过继续，
// 这样一轮就能看到"能走多远、第一个崩的是谁"。
// Generic ctor runner driven by the module's own dynamic tags (works for any module).
// ── 看门狗：把"卡死"也变成一条可记账的记录 ──
// 为什么必须：sigsetjmp 只接**崩溃**，接不住**卡死**。而 SDK 里那些"连服务/连驱动"的构造函数
// 在宿主上不是崩、是**永远等下去**（subsdk0 #3 连 nvhost、subsdk3 #2 同类）。没有看门狗，
// 一个卡死就把整轮拖住，剩下的构造函数永远跑不到。
// 做法：每个构造函数单开一个线程，主线程带超时等；超时就记账并**放弃那个线程**（不 join，泄漏一个
// 阻塞线程，探针里可以接受）。崩溃仍由 worker 里的 sigsetjmp 接住。
// The watchdog turns a hang into a logged, skippable event: each ctor runs on its own thread and the
// main thread waits with a timeout; on timeout the thread is abandoned (leaked) and we move on.
#define CTOR_TIMEOUT_MS 1500
// ── 无锁审计（S3 关键手段）──
// 为什么不能走 klog：卡死的线程正卡在 fprintf/fflush 里持着日志锁，任何"正常日志"都会被堵死。
// 这里用**裸 write 到同一个日志文件的 fd**：不加锁、不经 stdio 缓冲、单次几十字节
// （"小单写"在本平台是可用的；楔住的是大单写与 fflush）。
static int g_audit_fd = -1;          // -1 = 用主日志文件的 fd；子进程里设成管道写端
static void audit(const char *fmt, ...) {
    if (!g_k_log) return;
    char b[256];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if (n > (int)sizeof b - 1) n = (int)sizeof b - 1;
    int fd = g_audit_fd >= 0 ? g_audit_fd : fileno(g_k_log);
    if (fd >= 0) { ssize_t w = write(fd, b, (size_t)n); (void)w; }
}

typedef struct {
    void (*fn)(int, char **, char **);
    volatile int done, faulted;
    pthread_mutex_t mu; pthread_cond_t cv;
} ctor_job;
static void *ctor_tramp(void *p) {
    ctor_job *j = (ctor_job *)p;
    if (sigsetjmp(g_gate_jmp, 1) == 0) {
        g_gate_armed = 1;
        j->fn(0, NULL, NULL);
        g_gate_armed = 0;
    } else {
        g_gate_armed = 0;
        j->faulted = 1;
    }
    pthread_mutex_lock(&j->mu);
    j->done = 1;
    pthread_cond_signal(&j->cv);
    pthread_mutex_unlock(&j->mu);
    return NULL;
}

static void k_run_ctors_isolated(const char *tag, module_t *m, int timeout_ms);   // 前置声明（实现在后面）
// 直接模式：**在调用线程上跑构造函数**，不新开 pthread。
// 2026-10-22 离线定谳：SDK 的临界区/互斥量用 `tpidrro_el0 → +0x1f8 → +0x1b8` 取"当前线程 id"当锁持有者；
// **新开的宿主 pthread 没有客体 TLS、也没在 nn::os 注册** ⇒ 持有者字段是垃圾 ⇒ 锁自旋永不结束。
// 三个卡点（sdk#61 / subsdk0#3 / subsdk3#2）第一次调用全是 InitializeMutex / NvOsMutexCreate，完全对上。
// fork 会继承**调用线程**的 TLS ⇒ 隔离要放在进程层，不是线程层。
static int g_ctors_direct;

static void k_run_ctors(const char *tag, module_t *m) {
    if (!m || !m->base || !m->init_array || !m->init_arraysz) {
        klog("[%s 构造] 没有 init_array（跳过）\n", tag);
        return;
    }
    uint8_t *b = m->base;
    typedef void (*ctor_t)(int, char **, char **);
    ctor_t *arr = (ctor_t *)(b + m->init_array);
    size_t n = m->init_arraysz / 8, ran = 0, skipped = 0, faulted = 0, hung = 0;
    klog("[%s 构造] DT_INIT_ARRAY=%#llx 共 %zu 个\n", tag,
         (unsigned long long)(uintptr_t)m->init_array, n);
    // 跳过名单：**卡死**的构造函数（setjmp 只能接崩溃，接不住卡死）。
    // 每跳一个都要有明确的"为什么"——它们是待办，不是永久豁免。
    // Skip list for constructors that HANG (setjmp cannot catch a hang). Each entry is a TODO, not a pardon.
    // 2026-10-22 真机：subsdk0 = **NVN/GPU 运行时**，它的构造函数序列就是驱动初始化（#3 卡在 +0x35C48
    // 的 NvRmGpu 一族，跳过 #3 后 #4 又卡）⇒ 在宿主的 NVN 层（M4）就位之前，**整个模块的构造函数都别跑**，
    // 一条条试没有意义。装载（映射/重定位/PLT）照做——符号要用。
    // subsdk0 = the NVN/GPU runtime: its ctor sequence is driver bring-up, so skip the whole module's
    // ctors until the host NVN layer (M4) exists. Loading (map/reloc/PLT) still happens: the symbols are needed.
    // 现在有看门狗了，跳过名单**清空**：卡死由超时自动记账（每条都会打印"超时 N ms 未返回"），
    // 不需要人工逐条维护。名单本身保留——将来遇到"卡死但超时后又活着、会污染后续状态"的才放进来。
    static const struct { const char *tag; size_t idx; const char *why; } kSkip[] = {
        { "", 0, "" },      // 占位（空名单时避免零长数组）
    };
    for (size_t i = 0; i < n; i++) {
        if (!arr[i]) { skipped++; continue; }
        int skip = 0;
        for (size_t k = 0; k < sizeof kSkip / sizeof kSkip[0]; k++)
            if (kSkip[k].tag[0] && !strcmp(kSkip[k].tag, tag) &&
                (kSkip[k].idx == i || kSkip[k].idx == (size_t)-1)) {
                klog("[%s 构造] #%zu 跳过：%s\n", tag, i, kSkip[k].why); skip = 1; break;
            }
        if (skip) { skipped++; continue; }
        audit("[构造审计] %s #%zu/%zu → %#llx\n", tag, i, n, (unsigned long long)(uintptr_t)arr[i]);
        if (g_ctors_direct) {
            // 直接跑：调用线程有客体 TLS。崩了接住继续；卡死由**进程层**看门狗（父进程 kill）兜底。
            if (sigsetjmp(g_gate_jmp, 1) == 0) {
                g_gate_armed = 1;
                arr[i](0, NULL, NULL);
                g_gate_armed = 0;
                ran++;
            } else {
                g_gate_armed = 0;
                faulted++;
                audit("[构造审计] %s #%zu 崩了（pc %#llx sig %d）—— 跳过继续\n", tag, i,
                      (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
                mtx_report_owned((long)gettid(), tag);
            }
            continue;
        }
        ctor_job *j = (ctor_job *)calloc(1, sizeof *j);      // 超时后 worker 还会写它 ⇒ 不能放栈上
        if (!j) { skipped++; continue; }
        j->fn = arr[i];
        pthread_mutex_init(&j->mu, NULL);
        pthread_cond_init(&j->cv, NULL);
        pthread_t th;
        if (pthread_create(&th, NULL, ctor_tramp, j) != 0) { skipped++; continue; }
        struct timespec dl;
        clock_gettime(CLOCK_REALTIME, &dl);
        dl.tv_sec += CTOR_TIMEOUT_MS / 1000;
        dl.tv_nsec += (long)(CTOR_TIMEOUT_MS % 1000) * 1000000L;
        if (dl.tv_nsec >= 1000000000L) { dl.tv_sec++; dl.tv_nsec -= 1000000000L; }
        pthread_mutex_lock(&j->mu);
        while (!j->done) {
            if (pthread_cond_timedwait(&j->cv, &j->mu, &dl) != 0) break;
        }
        int done = j->done, bad = j->faulted;
        pthread_mutex_unlock(&j->mu);
        if (!done) {
            hung++;                                          // 卡死：记账、放弃该线程、继续下一个
            audit("[构造审计] %s #%zu **超时 %d ms 未返回** ⇒ 放弃该线程继续（卡死：要宿主拦的服务/驱动）\n",
                  tag, i, CTOR_TIMEOUT_MS);
        } else if (bad) {
            faulted++;
            audit("[构造审计] %s #%zu 崩了（pc %#llx sig %d）—— 跳过继续\n", tag, i,
                  (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
            pthread_join(th, NULL);
            free(j);
        } else {
            ran++;
            pthread_join(th, NULL);
            free(j);
        }
    }
    audit("[构造审计] %s 跑完：成功 %zu / 空槽+跳过 %zu / 崩 %zu / **卡死 %zu**（共 %zu）\n",
          tag, ran, skipped, faulted, hung, n);
}

static void k_run_module_init(const module_t *m) {
    uint8_t *b = m->base;
    size_t n = DM_IMG_MAIN_INIT_ARRAY_SIZE / 8;
    klog("[静态初始化] DT_INIT=%#llx，DT_INIT_ARRAY=%#llx，共 %zu 个构造器\n",
         (unsigned long long)(uintptr_t)(b + DM_IMG_MAIN_INIT_OFF),
         (unsigned long long)(uintptr_t)(b + DM_IMG_MAIN_INIT_ARRAY_OFF), n);
    typedef void (*ctor_t)(int, char **, char **);
    ctor_t *arr = (ctor_t *)(b + DM_IMG_MAIN_INIT_ARRAY_OFF);
    size_t ran = 0, failed = 0;
    // 逐个武装 + 逐个编号：崩了记现场并**跳过继续跑**（先看整体能走多远，再逐个收拾）。
    // 真机实证：#184 崩在 `ldr x9,[x10]`（x10=0 ⇒ si_addr=0）；注意报告的 pc 常比真正访存的
    // 指令多 4，判读时 pc 与 pc-4 都要看。
    g_inner_init = *(const uint64_t *)(uintptr_t)((uint8_t *)b + 0x22fc760);
    for (size_t i = 0; i < n; i++) {
        if (!arr[i]) continue;
        if (k_skip_ctor(i)) { klog("[静态初始化] #%zu 跳过（已知会崩）\n", i); continue; }
        klog("[静态初始化] #%zu/%zu → %#llx\n", i, n, (unsigned long long)(uintptr_t)arr[i]);
        if (sigsetjmp(g_gate_jmp, 1) == 0) {
            g_gate_armed = 1;
            // np102 实验（给 argv 传零缓冲）已撤销：无效——#206 的崩点不在约定上，
            //   而在"查表拿到空指针"（见文档 §49：根因是内容没加载，表计数为 0）。
            arr[i](0, NULL, NULL);
            g_gate_armed = 0;
            ran++;
            // np110：盯**内层指针** [(0x22fc760)]——真门对象 = [[0x2276168]]，
            //   而 [0x2276168] 恒指向 +0x22fc760 的描述符表、其首字恒 0 ⇒ 该由某个
            //   构造器写这个双字。逐个检查，抓住"它本该被写"的那个构造器。
            if (!g_inner_caught) {
                uint64_t iv = *(const uint64_t *)(uintptr_t)((uint8_t *)b + 0x22fc760);
                if (iv) {
                    g_inner_caught = 1;
                    g_inner_ctor = i;
                    g_inner_ctorentry = (uint64_t)(uintptr_t)arr[i];
                    g_inner_val = iv;
                }
            }
            // np107：**逐个查宅基地**——门字节 [*(0x2276168)+0x419] 一旦变非零，
            //   刚才那个构造器就是写入者。只记录不打印（np58 教训），循环后统一输出。
            if (!g_gate_caught) {
                // np109：装载器是 **x8=[0x2276168]; x8=[x8]; ldrb [x8+0x419]**（二次解引用！）
                //   真正的门对象 = [[0x2276168]]。np108 实测 [0x2276168]→0x22fc760 的首字为 0
                //   ⇒ 内层指针还没建立。这里改判"内层指针首次非零"。
                uint64_t g = *(const uint64_t *)(uintptr_t)((uint8_t *)b + 0x2276168);
                uint64_t obj = (g > 0x10000ull && g < 0x800000000000ull)
                             ? *(const uint64_t *)(uintptr_t)g : 0;
                if (obj > 0x10000ull && obj < 0x800000000000ull) {
                    g_gate_caught = 1;
                    g_gate_ctor = i;
                    g_gate_ctorentry = (uint64_t)(uintptr_t)arr[i];
                    g_gate_obj = obj;
                    g_gate_val = *(const uint8_t *)(uintptr_t)(obj + 0x419);
                }
            }
        } else {
            g_gate_armed = 0;
            failed++;
            klog("=== 静态初始化 #%zu 崩：pc=%#llx sig=%d si_addr=%#llx（跳过，继续）===\n",
                 i, (unsigned long long)g_gate_fault_pc, g_gate_fault_sig,
                 (unsigned long long)g_gate_fault_addr);
            gate_dump_fault("静态初始化");
        }
    }
    klog("[静态初始化] 完成：成功 %zu 个，失败 %zu 个\n", ran, failed);
    // np66：槽体检第二站——静态初始化之后。装载后该槽是对的（base+0x22d9260），
    // 若这里变 0 ⇒ **某个构造器把它清零了**；若还是对的 ⇒ 是 nnMain 里的流程清零的。
    dump_big_allocs();   // np106b：大分配记录落地（此刻打日志才安全）
    // np107：门字节写入者（构造器序号 + 入口）
    if (g_gate_caught) {
        uint64_t b3 = (uint64_t)(uintptr_t)b;
        uint64_t e = g_gate_ctorentry;
        int in = e >= b3 && e < b3 + DM_IMG_MAIN_END;
        klog("[启动包门] 写入者 = 构造器 #%zu（入口 %s%#llx）；对象=%#llx 当时值=%#x\n",
             g_gate_ctor, in ? "+" : "(外)",
             (unsigned long long)(in ? e - b3 : e),
             (unsigned long long)g_gate_obj, g_gate_val);
        // np108：dump 门对象的内存，弄清它是什么（对象地址换算成映像内偏移）
        {
            const uint8_t *o = (const uint8_t *)(uintptr_t)g_gate_obj;
            int ino = g_gate_obj >= b3 && g_gate_obj < b3 + DM_IMG_MAIN_END;
            klog("[启动包门] 对象映像内偏移 = %s%#llx\n",
                 ino ? "+" : "(外)",
                 (unsigned long long)(ino ? g_gate_obj - b3 : g_gate_obj));
            char hx[96]; size_t u = 0;
            for (int k = 0; k < 24 && u + 3 < sizeof hx; k++)
                u += (size_t)snprintf(hx + u, sizeof hx - u, "%02x", o[k]);
            hx[u] = 0; klog("[启动包门] 头 24 字节：%s\n", hx);
            u = 0;
            for (int k = 0x410; k < 0x428 && u + 3 < sizeof hx; k++)
                u += (size_t)snprintf(hx + u, sizeof hx - u, "%02x", o[k]);
            hx[u] = 0; klog("[启动包门] +0x410..+0x428：%s\n", hx);
        }
    } else {
        klog("[启动包门] 构造器跑完时门仍为 0 ⇒ 4 是启动链/nnMain 阶段写的\n");
    }
    // np110：内层指针（+0x22fc760 的双字）
    if (g_inner_caught) {
        uint64_t b3 = (uint64_t)(uintptr_t)b;
        uint64_t e = g_inner_ctorentry;
        int in = e >= b3 && e < b3 + DM_IMG_MAIN_END;
        klog("[内层指针] 被构造器 #%zu（入口 %s%#llx）写成 %#llx\n", g_inner_ctor,
             in ? "+" : "(外)", (unsigned long long)(in ? e - b3 : e),
             (unsigned long long)g_inner_val);
    } else {
        klog("[内层指针] 构造器期始终未被写（初值 %#llx）⇒ 写在启动链/nnMain 阶段，或根本没有写\n",
             (unsigned long long)g_inner_init);
    }
    klog("[槽体检] 静态初始化后：0x2280eb8=%#llx（装载后应=%#llx）\n",
         (unsigned long long)*(const uint64_t *)(b + 0x2280eb8),
         (unsigned long long)((uintptr_t)b + 0x22d9260));
}

static void *k_thread(void *arg) {
    (void)arg;
    g_probe_thr = pthread_self();   // np115：先记下"谁拥有 setjmp 上下文"
    g_probe_thr_set = 1;
    uint64_t b = (uint64_t)g_k_m.base;
    uint64_t tp = 0; __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp));
    uint64_t tsd = *(uint64_t *)(tp - 272);
    const void *sv[8] = {
        (void *)(uintptr_t)0x1111111111111111ull, (void *)(uintptr_t)0x2222222222222222ull,
        (void *)(uintptr_t)0x3333333333333333ull, (void *)k_gate_stub,
        (void *)(uintptr_t)0x4400000000000004ull, (void *)(uintptr_t)0x4400000000000005ull,
        (void *)(uintptr_t)0x4400000000000006ull, (void *)(uintptr_t)0x4400000000000007ull,
    };
    for (int i = 0; i < 8; i++) *(uint64_t *)(tsd + (uint64_t)g_k_key[i] * 8) = (uint64_t)sv[i];
    // np133：**sdk 的键页槽也要填**（sdk 有自己的 8 个键号）。缺了它，sdk 代码一进 svc 门
    //   就会读空门表（真机实证：静态初始化期崩，si_addr=0xd0/0x158，x8=门表偏移）。
    if (g_n_sdk_os)
        for (int i = 0; i < 8; i++) *(uint64_t *)(tsd + (uint64_t)g_s_key[i] * 8) = (uint64_t)sv[i];
    // np135：K18/KTLR 换真影子（TSD 影子块 / TLR 影子区 + NN 线程对象 + 唯一名牌）
    if (g_n_sdk_os) k_sdk_tls_shadow_install();
    // 装机他判：把本次编译时间打进日志——跑完先看这一行与本地编译时间是否一致，
    // 不一致就说明设备上跑的是旧包（今天已被这个问题浪费三轮）。
    klog("=== 探针构建：%s %s ===\n", __DATE__, __TIME__);
    // ★ 影子「当前堆」：必须在静态初始化与 nnMain **之前**挂上——
    //   nnMain 一开始就要"取当前堆"，取不到就跳过 7 个初始化（见 §24/§25）。
    k_fake_heap_install((uint8_t *)g_k_m.base);
    // ★ 2026-10-22（通用化第一步）：先跑 **sdk 自己的 78 个构造函数**。
    //   以前只把 sdk 当"符号来源"，它的构造函数一次都没跑 ⇒ SDK 的系统（任务/内存/服务）
    //   从未初始化，这也是"任务 runner 的 flags 没人置位"最可能的解释。
    //   开关 NP_SDK_CTORS 默认开；要回到旧行为就设 0。
    // ⚠ 2026-10-22 真机：**就地跑**会被某个构造函数卡死并把主线程一起拖住（sdk #61 实测，
    // 连看门狗的超时都发不出来）⇒ 改成子进程隔离：卡死只害死子进程，主路径照常进 nnMain。
    // In-process runs can wedge the main thread (sdk #61), so run them in a forked child.
    if (getenv("NP_SDK_CTORS") == NULL || atoi(getenv("NP_SDK_CTORS")) != 0)
        k_run_ctors_isolated("sdk", &g_k_s, 60000);

    k_run_module_init(&g_k_m);   // ★ 先跑模块自己的静态初始化（否则 data 段的全局对象全是空的）
    {   // 崩点归属用：把本进程映射抄进日志（模块区 / 探针 .so / 宿主堆 各自范围）
        FILE *mp = fopen("/proc/self/maps", "r");
        if (mp) {
            char line[256]; int n = 0;
            klog("=== 进程映射（前 24 行）===\n");
            while (n++ < 24 && fgets(line, sizeof line, mp)) klog("  %s", line);
            fclose(mp);
        }
    }
    // np105b：启动包装载的门（0xd5b670：*(0x2276168) → 对象 → [+0x419] != 0 则跳过装载）。
    //   在三个关键时刻读它，看这个 4 是什么时候写进去的。
    #define GATE_LOG(tag) do { \
        uint64_t _g = *(const uint64_t *)(uintptr_t)(b + 0x2276168); \
        uint64_t _o = (_g > 0x10000ull && _g < 0x800000000000ull) \
                   ? *(const uint64_t *)(uintptr_t)_g : 0; \
        uint8_t _v = 0xff; \
        if (_o > 0x10000ull && _o < 0x800000000000ull) \
            _v = *(const uint8_t *)(uintptr_t)(_o + 0x419); \
        klog("[启动包门] %s：外层=%#llx 内层(真门对象)=%#llx [+0x419]=%#x\n", tag, \
             (unsigned long long)_g, (unsigned long long)_o, _v); \
    } while (0)
    GATE_LOG("启动链前");
    k_ab_obs("启动链前");
    g_k_stage = "启动链 0x1c8";
    klog("=== 起跑：入口 0x1c8（启动链）base %#llx ===\n", (unsigned long long)b);
    g_gate_armed = 1; g_gate_fault_pc = 0;
    if (sigsetjmp(g_gate_jmp, 1) == 0) {
        g_k_rv = ((uint64_t (*)(void))(b + 0x1c8))();
        g_gate_armed = 0;
        klog("=== 启动链返回：rv=%#llx ===\n", (unsigned long long)g_k_rv);
    } else {
        g_gate_armed = 0; g_k_faulted = 1;
        klog("=== 启动链崩：pc %#llx sig %d ===\n", (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
        gate_dump_fault("启动链 0x1c8");
        return NULL;
    }
    GATE_LOG("启动链后");
    k_ab_obs("启动链后");
    // ⭐ np161：**深跑 0x1f8 正式退役**（np143/144 早已决议"这条路在真实游戏里不存在"，
    //   但代码一直没删——每轮都主动深跑 ⇒ 每轮必崩 AddRef(NULL) ⇒ handler longjmp 把
    //   客体线程从调用现场传送走 ⇒ 客体锁状态被我们污染，后续 LockMutex 永久等待）。
    //   深跑是当年为定位 0x1f8 崩点插的探针，使命已完成（结论见 §1.6/§1.7），
    //   继续开着只有负价值：崩溃 + 状态污染 + 掩盖真实流程。
    // 如需复活做实验，取消下面注释即可。
#if 0   // np167 实验结束：恢复退役状态
    // 更深一站：模块的第一个真实功能函数 0x1f8（C++ 对象工厂：读数据槽 → 分配 0xaa0 → 构造）
    g_k_stage = "工厂 0x1f8";
    klog("=== 深跑：0x1f8（对象工厂，x0 = 注入的管理器对象）===\n");
    g_gate_armed = 1; g_gate_fault_pc = 0;
    if (sigsetjmp(g_gate_jmp, 1) == 0) {
        uint64_t r2 = ((uint64_t (*)(uint64_t))(b + 0x1f8))((uint64_t)(uintptr_t)&g_mgr_obj[0]);
        g_gate_armed = 0;
        klog("=== 0x1f8 返回：rv=%#llx ===\n", (unsigned long long)r2);
    } else {
        g_gate_armed = 0; g_k_faulted = 1;
        klog("=== 0x1f8 崩：pc %#llx sig %d ===\n", (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
        gate_dump_fault("工厂 0x1f8");
        // np86：**迟到防护**——本分支是 fall-through（设计上崩完要继续跑 nnMain）。但若
        //   nnMain 段已经进入过，说明这次 longjmp 是"后面阶段的崩倒流回本阶段的 setjmp"，
        //   继续 fall-through 就会"装影子堆→崩→回这里→又跑 nnMain"无限循环（np85 实测
        //   空转 94,108 次、日志 226 万行）。此时必须止步。
        if (g_nnmain_entered) {
            klog("【红】工厂分支被迟到触发（nnMain 已进入过）——止步，避免倒流成环\n");
            return NULL;
        }
    }
#endif
    // 终站：游戏的真正入口 nnMain（模块符号表 #506：st_value 0x735330，size 1008，GLOBAL FUNC）
    // 依据：rtld 的符号表里列出它要为主模块解析的入口名 nnMain（启动约定，非猜测）。
    GATE_LOG("nnMain前");
    k_ab_obs("nnMain前");
    g_k_stage = "nnMain";
    g_nnmain_entered = 1;   // np86：置位后，工厂分支就知道"别再 fall-through 了"
    klog("=== 终站：nnMain（0x735330）===\n");
    // np177：登记"主线程"身份，供 per-thread 门命中取证标记 ←main
    g_main_pthread = (uint32_t)(uintptr_t)pthread_self();
    g_main_pth = pthread_self();
    // np178：装 SIGPROF handler（采样由帧线程 tick 定向投给主线程）
    smp_arm();
    dump_maps_once();     // 采样点若落在映像外，靠这张表定位所属 .so
    // 再挂一次影子堆：万一静态初始化里有构造器把它清零，这里补回来（代价极小）。
    k_fake_heap_install((uint8_t *)b);
    // np72：**框架自证**——影子堆安装会写映像里的几个洞；这里确认它没有顺手清掉 0x2280eb8。
    // 若这一行读到 0，说明"槽变 0"是我们自己的注入造成的（**通用框架 bug**，影响任何游戏）。
    klog("[槽体检] 影子堆安装后：0x2280eb8=%#llx（应=%#llx）\n",
         (unsigned long long)*(const uint64_t *)(b + 0x2280eb8),
         (unsigned long long)((uintptr_t)b + 0x22d9260));
    g_gate_armed = 1; g_gate_fault_pc = 0;
    k_gate_obs_dump("nnMain 调用前");   // np136：静态初始化期（344 次 InitializeMutex 都在这）的门命中快照
    if (!g_fixup_page) g_fixup_page = calloc(1, 0x8000);   // 越障暂存区 32KB（64 块 × 0x100，见 gate_fault_handler）
    g_fixup_base = (uintptr_t)b;
    {   // 诊断（2026-10-20）：nnMain 之前 SIGSEGV 的 disposition 还是我们的吗？静态初始化期
        //   （模块 .init_array）有没有谁换掉它。线程掩码用"空集读回式"（不改掩码）——
        //   同步信号（SIGSEGV）在本线程被阻塞也会让 handler 不跑，是"没拦到"的候选原因。
        struct sigaction rb;
        sigaction(SIGSEGV, NULL, &rb);
        sigset_t empty, cur;
        sigemptyset(&empty);
        pthread_sigmask(SIG_BLOCK, &empty, &cur);
        klog("[信号] nnMain前：SIGSEGV disposition=%p｜本线程掩码%s\n",
             (void *)rb.sa_sigaction,
             sigismember(&cur, SIGSEGV) ? " 阻塞 SIGSEGV（★ 没拦到的候选原因）" : " 未阻塞");
    }
    if (sigsetjmp(g_gate_jmp, 1) == 0) {
        // 按"干净调用"准备寄存器：x21 = 模块基址（Switch SDK 约定），x0..x7 清零。
        register uint64_t r21 __asm__("x21") = b;
        register uint64_t r_x2 __asm__("x2") = 0; (void)r_x2;
        register uint64_t r_x3 __asm__("x3") = 0; (void)r_x3;
        register uint64_t r_x4 __asm__("x4") = 0; (void)r_x4;
        register uint64_t r_x5 __asm__("x5") = 0; (void)r_x5;
        register uint64_t r_x6 __asm__("x6") = 0; (void)r_x6;
        register uint64_t r_x7 __asm__("x7") = 0; (void)r_x7;
        __asm__ volatile("" : "+r"(r21), "+r"(r_x2), "+r"(r_x3), "+r"(r_x4), "+r"(r_x5), "+r"(r_x6), "+r"(r_x7));
        uint64_t r3 = ((uint64_t (*)(uint64_t, uint64_t))(b + 0x735330))(0, 0);
        g_gate_armed = 0;
        klog("=== nnMain 返回：rv=%#llx ===\n", (unsigned long long)r3);
        nvn_api_summary("nnMain 返回");
        ktrace_dump(g_files_dir);   // np125：把 trace 环缓冲落盘（离线分析）
    } else {
        g_gate_armed = 0; g_k_faulted = 1;
        klog("=== nnMain 崩：pc %#llx sig %d ===\n", (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
        // np73（§38.8 的决定性测试）：把 **pc 处的运行时指令字节**打出来，和文件对照。
        // 文件里 0xa2469c..0xa246a8 = 9000c2e8 / f9475d08 / f9000100 / d65f03c0
        //   （adrp / ldr x8,[x8,#0xeb8] / str x0,[x8] / ret）
        // 若运行时前 1-2 条被改写 ⇒ 那里有钩子桩（语义已变，不能按原指令推断崩因）。
        // np73（§38.8 的决定性测试）：把 **pc 处的运行时指令字节**打出来，和文件对照。
        //   ⚠ np119：pc 可能是 0（`blr` 了空函数指针）——读 pc-8 就是读 0xfffffff8，
        //   处理器里再崩一次 ⇒ 进程直接死、寄存器现场全丢（本轮实证：日志停在"pc 0"那一行）。
        //   只在 pc 落在模块映像内才读。
        if (g_gate_fault_pc >= b && g_gate_fault_pc < b + DM_IMG_MAIN_END) {
            const uint32_t *c = (const uint32_t *)(uintptr_t)(g_gate_fault_pc - 8);
            klog("[指令] pc-8..pc+8 运行时 = %08x %08x %08x %08x %08x\n",
                 c[0], c[1], c[2], c[3], c[4]);
            klog("[指令] 文件里同处应为   = d65f03c0 9000c2e8 f9475d08 f9000100 d65f03c0\n");
        } else {
            klog("[指令] pc=%#llx 不在模块映像内（多半是 blr 了空/野指针），不读指令\n",
                 (unsigned long long)g_gate_fault_pc);
        }
        // np64：槽体检——判"运行期槽是 0"到底是**游戏写回 0**还是**装载层 bss 清零洗掉了重定位**
        // （崩点 +0xA046A4 的存储目标 x9 = *(0x2280aa8) = 0；RELA 本该是 base+0x22d8790）
        {
            const uint8_t *im = (const uint8_t *)(uintptr_t)b;
            uint64_t s98 = *(const uint64_t *)(im + 0x2280a98);
            uint64_t sa0 = *(const uint64_t *)(im + 0x2280aa0);
            uint64_t sa8 = *(const uint64_t *)(im + 0x2280aa8);
            klog("[槽检] 0x2280a98=%#llx 0x2280aa0=%#llx 0x2280aa8=%#llx（基址=%#llx，应为 基址+0x22d8790=%#llx）\n",
                 (unsigned long long)s98, (unsigned long long)sa0, (unsigned long long)sa8,
                 (unsigned long long)(uintptr_t)b,
                 (unsigned long long)((uintptr_t)b + 0x22d8790));
            klog("[槽检] .bss 0x22d8790 内容=%#llx；0x22d8798 首字节=%#x（once 标志）\n",
                 (unsigned long long)*(const uint64_t *)(im + 0x22d8790),
                 (unsigned)*(const unsigned char *)(im + 0x22d8798));
            // np70：崩时把槽**前后各 4 个 qword** 也打出来——判"整片被清零（memset 越界）"
            //      还是"只有这一个槽被改（定向写/竞态）"。这是回答 §38.6 那个"两指令内变 0"的直证。
            {
                const uint64_t *p = (const uint64_t *)(im + 0x2280eb8);
                klog("[槽周边] 0x2280eb8 前4: %016llx %016llx %016llx %016llx\n",
                     (unsigned long long)p[-4], (unsigned long long)p[-3],
                     (unsigned long long)p[-2], (unsigned long long)p[-1]);
                klog("[槽周边] 自身/后3: %016llx %016llx %016llx %016llx\n",
                     (unsigned long long)p[0], (unsigned long long)p[1],
                     (unsigned long long)p[2], (unsigned long long)p[3]);
                const uint64_t *q = (const uint64_t *)(im + 0x22d9260);
                klog("[槽周边] 变量 0x22d9260 及其后: %016llx %016llx %016llx %016llx\n",
                     (unsigned long long)q[0], (unsigned long long)q[1],
                     (unsigned long long)q[2], (unsigned long long)q[3]);
            }
        }
        if (g_fixup_n)
            klog("[越障] 共拦截 %d 次；前 8 处 PC 偏移：%#llx %#llx %#llx %#llx %#llx %#llx %#llx %#llx\n",
                 (int)g_fixup_n,
                 (unsigned long long)g_fixup_last[0], (unsigned long long)g_fixup_last[1],
                 (unsigned long long)g_fixup_last[2], (unsigned long long)g_fixup_last[3],
                 (unsigned long long)g_fixup_last[4], (unsigned long long)g_fixup_last[5],
                 (unsigned long long)g_fixup_last[6], (unsigned long long)g_fixup_last[7]);
        k_log_counters("崩时");
        gate_dump_fault("nnMain");
        ktrace_dump(g_files_dir);   // np125：崩了也要把 trace 落盘——最后几十条往往就是崩因
    }
    k_gate_obs_dump("k_thread 收尾");   // np136：nnMain 返回/崩后才打得到，循环期靠定时器
    return NULL;
}
// ───────── np133：装载 sdk 模块并建导出表（M1 第二步）─────────
// 为什么：sdk 自带 340 个 nn::os 实现（离线普查已确认）。把 main 的 nn::os 导入解析到 sdk，
//   就能逐步删掉我们手写的那一层。但**不能一次全切**：CreateThread/TLS/队列都依赖宿主线程基建，
//   先只切"互斥量"——它只用到我们刚补的 ArbitrateLock/Unlock，不涉及句柄创建。
// 装载要做四件事（与 main 的装载逐条对应）：映射 → 写键页 → 应用 RELATIVE → 封 R+X。
// （状态声明在文件前面：g_k_s / g_s_key / g_sdk_os / g_n_sdk_os，线程蹦床要用）

// np135：导出表查找。**全量表**（np133 时只收 nn::os 340 个）——PLT 自解析与白名单
// 都要查它；线性扫描，装载期一次性调用（4,135 槽 × 14,834 条 ≈ 6,000 万次首字符比较，可接受）。
// ───────── 服务拦截表（M2，2026-10-22）─────────
// 为什么在这里拦：NX 的跨模块/同模块调用全走 **PLT 槽**，槽里放谁就是谁。与其去实现 CMIF 协议
// （要在 IPC 缓冲区里写 nn::sf 的应答头，一次写错就是崩/卡），不如在**填充槽时**把这几个服务
// 直接解析到宿主函数：语义确定（"成功但什么也没做"），也不再往 sdk 的 IPC 栈里钻。
// 实测依据：这 5 个服务的 IPC 一次都没发起（[IPC] 账本为空）⇒ sdk 那边本来就没跑通，
// 拦截不会"打断"任何已经工作的东西。
// Intercept table: NX calls go through PLT slots, so filling a slot with a host function is a
// legitimate (and much cheaper) way to serve a service than implementing the CMIF protocol.
static unsigned long g_svc_hits[16];
static int g_svc_ret0(uint64_t a, uint64_t b, uint64_t c, uint64_t d) { (void)a;(void)b;(void)c;(void)d; return 0; }
// account::OpenPreselectedUser(UserHandle *out)：**必须写出参数**，否则调用方拿未初始化句柄。
static int g_svc_open_user(uint64_t out, uint64_t b, uint64_t c, uint64_t d) {
    (void)b;(void)c;(void)d;
    if (out) *(uint64_t *)(uintptr_t)out = 0x1000000000000001ull;   // 假用户句柄（非 0，格式合法位）
    return 0;
}
static const struct { const char *sym; void *fn; const char *note; int slot; } g_svc_int[] = {
    { "_ZN2nn3hid14InitializeNpadEv",        (void *)g_svc_ret0,      "hid::InitializeNpad → 无手柄（成功）", 0 },
    { "_ZN2nn3hid18InitializeDebugPadEv",    (void *)g_svc_ret0,      "hid::InitializeDebugPad → 成功", 1 },
    { "_ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE",
                                             (void *)g_svc_ret0,      "hid::SetSupportedNpadStyleSet → 成功", 2 },
    { "_ZN2nn3hid22SetSupportedNpadIdTypeEPKjm", (void *)g_svc_ret0,  "hid::SetSupportedNpadIdType → 成功", 3 },
    { "_ZN2nn3hid15GetNpadStyleSetERKj",     (void *)g_svc_ret0,      "hid::GetNpadStyleSet → 0（无手柄）", 4 },
    { "_ZN2nn7account10InitializeEv",        (void *)g_svc_ret0,      "account::Initialize → 成功", 5 },
    { "_ZN2nn7account19OpenPreselectedUserEPNS0_10UserHandleE", (void *)g_svc_open_user, "account::OpenPreselectedUser → 假用户句柄", 6 },
    { "_ZN2nn4pctl10InitializeEv",           (void *)g_svc_ret0,      "pctl::Initialize → 成功", 7 },
    { "_ZN2nn3aoc10InitializeEv",            (void *)g_svc_ret0,      "aoc::Initialize → 成功", 8 },
};
#define N_SVC_INT ((int)(sizeof g_svc_int / sizeof g_svc_int[0]))

// 命中就返回宿主函数（并按符号记一次账）
static void *dm_svc_intercept(const char *nm) {
    for (int i = 0; i < N_SVC_INT; i++)
        if (!strcmp(g_svc_int[i].sym, nm)) {
            unsigned long c = ++g_svc_hits[g_svc_int[i].slot];
            if (c == 1) klog("[服务] 拦截 %s（第 1 次）\n", g_svc_int[i].note);
            return g_svc_int[i].fn;
        }
    return NULL;
}

static void *k_sdk_lookup(const char *nm) {
    for (int i = 0; i < g_n_sdk_os; i++)
        if (!strcmp(g_sdk_os[i].name, nm)) return (void *)(uintptr_t)g_sdk_os[i].addr;
    return NULL;
}

static int k_load_sdk(void) {
    memset(&g_k_s, 0, sizeof g_k_s);
    strncpy(g_k_s.name, "sdk", sizeof g_k_s.name - 1);
    g_k_s.text_end      = DM_IMG_SDK_TEXT_END;
    g_k_s.patch_off     = DM_IMG_SDK_PATCH_OFF;
    g_k_s.ro_off        = DM_IMG_SDK_RO_OFF;
    g_k_s.data_off      = DM_IMG_SDK_DATA_OFF;
    g_k_s.key_page_off  = DM_IMG_SDK_KEY_PAGE_OFF;
    g_k_s.key_page_size = DM_IMG_SDK_KEY_PAGE_SIZE;
    g_k_s.x0_end        = DM_IMG_SDK_X0_END;
    g_k_s.img_end       = DM_IMG_SDK_END;
    if (dm_ohos_map_image_alloc(&g_k_s)) { klog("[sdk] ✗ 映射失败\n"); return -1; }

    // 键页：sdk 有自己的 8 个宿主键（它的改写产物按自己的键号寻址，不能借用 main 的）
    for (int i = 0; i < 8; i++)
        if (pthread_key_create(&g_s_key[i], NULL)) { klog("[sdk] ✗ 建键失败\n"); return -1; }
    uint64_t *page = (uint64_t *)((uint8_t *)g_k_s.base + DM_IMG_SDK_KEY_PAGE_OFF);
    memset(page, 0, DM_IMG_SDK_KEY_PAGE_SIZE);
    page[DM_IMG_SDK_KEY_K18_IDX]       = (uint64_t)g_s_key[0] * 8;
    page[DM_IMG_SDK_KEY_KTLS_IDX]      = (uint64_t)g_s_key[1] * 8;
    page[DM_IMG_SDK_KEY_KTLR_IDX]      = (uint64_t)g_s_key[2] * 8;
    page[DM_IMG_SDK_KEY_KGATE_IDX]     = (uint64_t)g_s_key[3] * 8;
    page[DM_IMG_SDK_KEY_KRES0_IDX]     = (uint64_t)g_s_key[4] * 8;
    page[DM_IMG_SDK_KEY_KRES1_IDX]     = (uint64_t)g_s_key[5] * 8;
    page[DM_IMG_SDK_KEY_KASPECT_IDX]   = (uint64_t)g_s_key[6] * 8;
    page[DM_IMG_SDK_KEY_KASPECTUI_IDX] = (uint64_t)g_s_key[7] * 8;

    uint8_t *img = (uint8_t *)g_k_s.base;
    uint32_t mod0 = 0;
    for (uint32_t o = 0; o < 0x100; o += 4)
        if (*(uint32_t *)(img + o) == 0x30444F4Du) { mod0 = o; break; }
    if (!mod0) { klog("[sdk] ✗ 找不到 MOD0\n"); return -1; }
    uint64_t dyn = (uint64_t)(img + mod0 + *(uint32_t *)(img + mod0 + 4));
    uint64_t DT_RELA = 0, DT_RELASZ = 0, DT_SYMTAB = 0, DT_STRTAB = 0, DT_HASH = 0;
    for (int i = 0; i < 64; i++) {
        uint64_t tag = *(uint64_t *)(dyn + (uint64_t)i * 16);
        uint64_t val = *(uint64_t *)(dyn + (uint64_t)i * 16 + 8);
        if (!tag) break;
        if (tag == 7) DT_RELA = val;
        else if (tag == 8) DT_RELASZ = val;
        else if (tag == 6) DT_SYMTAB = val;
        else if (tag == 5) DT_STRTAB = val;
        else if (tag == 4) DT_HASH = val;
        // 25/27 = DT_INIT_ARRAY / DT_INIT_ARRAYSZ：**sdk 自己的 78 个构造函数**就挂在这里。
        // 以前只拿 sdk 当"符号来源"，从没跑过它的构造函数 —— 而 SDK 的系统（任务/内存/文件服务）
        // 正是在这些构造函数里初始化的。这正是"SDK 自己的系统跑起来"缺的那一步。
        else if (tag == 25) g_k_s.init_array = val;
        else if (tag == 27) g_k_s.init_arraysz = val;
    }
    uint64_t n_rel = 0;
    for (uint64_t off = 0; off + 24 <= DT_RELASZ; off += 24) {
        uint64_t r_off  = *(uint64_t *)(img + DT_RELA + off);
        uint64_t r_info = *(uint64_t *)(img + DT_RELA + off + 8);
        int64_t  r_add  = *(int64_t  *)(img + DT_RELA + off + 16);
        if ((uint32_t)r_info == 1027) { *(uint64_t *)(img + r_off) = (uint64_t)img + (uint64_t)r_add; n_rel++; }
    }

    // 导出表：**全量收集**（np135，M1 第二步 PLT 自解析的依据）。
    //   离线实证：sdk 的 4,135 个 PLT 导入里 4,126 个是本模块自己的导出——NX 动态链接
    //   约定"模块对自身导出的调用也走 PLT 导入槽"，装载器必须自解析；只收 nn::os 一族时
    //   这些内线全落到 sdk_ret0 空转（np134 崩链的根因之一）。
    g_n_sdk_os = 0;
    uint32_t nsym = DT_HASH ? *(uint32_t *)(img + DT_HASH + 4) : 0;
    for (uint32_t i = 0; DT_HASH && i < nsym && g_n_sdk_os < SDK_OS_MAX; i++) {
        uint64_t st = DT_SYMTAB + (uint64_t)i * 24;
        uint32_t name_off = *(uint32_t *)(img + st);
        uint16_t shndx = *(uint16_t *)(img + st + 6);
        uint64_t value = *(uint64_t *)(img + st + 8);
        if (!name_off || !shndx || !value) continue;
        g_sdk_os[g_n_sdk_os].name = (const char *)(img + DT_STRTAB + name_off);
        g_sdk_os[g_n_sdk_os].addr = (uint64_t)img + value;
        g_n_sdk_os++;
    }

    // ★ sdk 自己的 PLT 槽也要填。**np135 实证：它导入 4,135 个符号，其中 4,126 个是本模块
    //   自己的导出**（旧注释"只导入 13 个"是过期错注——漏数这 4,126 条内线，sdk 的内部协作
    //   曾全部落到空转桩）。不填的话，sdk 一调 malloc 就是 `blr x17` 跳到「未加基址的原始值」
    //   （真机实证：pc=0x33ff04 正是 sdk 的函数偏移）。
    //   填充优先级：malloc 族 → 宿主 libc；本模块导出 → 自解析（img + st_value）；
    //   剩余 4 个真外部符号（nnMain / nninitStartup /
    //   nnosGetDefaultUserExceptionStackForApplication / __nnDetailNintendoSdkRuntimeObjectFile）
    //   → sdk_ret0 暂桩并记账。
    uint64_t DT_JMPREL_S = 0, DT_PLTRELSZ_S = 0;
    for (int i = 0; i < 64; i++) {
        uint64_t tag = *(uint64_t *)(dyn + (uint64_t)i * 16);
        uint64_t val = *(uint64_t *)(dyn + (uint64_t)i * 16 + 8);
        if (!tag) break;
        if (tag == 0x17) DT_JMPREL_S = val;
        else if (tag == 2) DT_PLTRELSZ_S = val;
    }
    static uint64_t (*sdk_ret0p)(uint64_t, uint64_t) = sdk_ret0; (void)sdk_ret0p;
    uint64_t n_sdk_plt = 0, n_self = 0, n_libc = 0, n_ext = 0;
    for (uint64_t off = 0; off + 24 <= DT_PLTRELSZ_S; off += 24) {
        uint64_t r_off  = *(uint64_t *)(img + DT_JMPREL_S + off);
        uint64_t r_info = *(uint64_t *)(img + DT_JMPREL_S + off + 8);
        uint32_t symi   = (uint32_t)(r_info >> 32);
        uint64_t st     = DT_SYMTAB + (uint64_t)symi * 24;
        uint32_t name_off = *(uint32_t *)(img + st);
        const char *nm = (const char *)(img + DT_STRTAB + name_off);
        uint64_t target = 0;
        if      (!strcmp(nm, "malloc"))        { target = (uint64_t)(uintptr_t)malloc; n_libc++; }
        else if (!strcmp(nm, "calloc"))        { target = (uint64_t)(uintptr_t)calloc; n_libc++; }
        else if (!strcmp(nm, "realloc"))       { target = (uint64_t)(uintptr_t)realloc; n_libc++; }
        else if (!strcmp(nm, "free"))          { target = (uint64_t)(uintptr_t)free; n_libc++; }
        else if (!strcmp(nm, "aligned_alloc")) { target = (uint64_t)(uintptr_t)aligned_alloc; n_libc++; }
        else {
            void *p = dm_svc_intercept(nm);  // M2：服务拦截表优先（不经 IPC）
            if (!p) p = k_sdk_lookup(nm);     // 再自解析：本模块自己的导出
            if (p) { target = (uint64_t)(uintptr_t)p; n_self++; }
            else   { target = (uint64_t)(uintptr_t)sdk_ret0; n_ext++; }
        }
        *(uint64_t *)(img + r_off) = target;
        n_sdk_plt++;
    }
    if (dm_ohos_map_image_seal(&g_k_s)) { klog("[sdk] ✗ 封 R+X 失败\n"); return -1; }
    klog("[sdk] 已装载：base=%p RELATIVE %llu 条，导出 %d 个；PLT %llu 槽 = 自解析 %llu + libc %llu + 外部桩 %llu\n",
         (void *)g_k_s.base, (unsigned long long)n_rel, g_n_sdk_os,
         (unsigned long long)n_sdk_plt, (unsigned long long)n_self,
         (unsigned long long)n_libc, (unsigned long long)n_ext);
    return 0;
}


// ───────── S2（2026-10-22）：**通用模块装载** —— 把 rtld / subsdk0-3 也装进来 ─────────
// 为什么：用户定调"先通用化、让游戏跑起来"。nxnative 的做法是**装全部模块**，让 SDK 自己的
//   系统跑起来，宿主只留 svc/IPC/图形。勘察数据支持这一点：装全之后 main 的 418 条导入里
//   406 条能在模块内部解析（剩下的是 __rel_* 伪符号 + 8 个 glslc*）。
// 与 k_load_sdk 的区别：那个是给 sdk 定制的（键页/导出/PLT 都写死）；这里是**表驱动**：
//   一个描述符表 + 一个通用的 映射/重定位/收集导出 流程，任何模块都能装。
// Why table-driven: k_load_sdk was bespoke for sdk; this loads any module from a descriptor table.
// 容量上限：旷野之息用 5 个（rtld + subsdk0-3）；换游戏可能更多（subsdk4…），留余量。
// **实际装载哪些模块不写死在这里**——由生成的 dm_xmods_gen.h 决定（以 exefs 实际
// 内容为准）。超出 EXTRA_MAX 的部分忽略并告警（宁可少装，不能越界写内存）。
#define EXTRA_MAX 16
static module_t      g_k_x[EXTRA_MAX];
static pthread_key_t g_x_key[EXTRA_MAX][8];

typedef struct {
    const char *name; module_t *m;
    uint64_t text_end, patch_off, ro_off, data_off, key_page_off, key_page_size, x0_end, img_end;
} xmod_t;

static xmod_t        g_xmods[EXTRA_MAX];
static int           g_n_xm;      // 表里的模块数（= DM_XMODS_GEN_N，含装载失败的）
static int           g_n_x;       // 装载成功的个数（记账用）

// 从生成表初始化 g_xmods：**换游戏只重新生成 dm_xmods_gen.h，本文件不动**
// （以前这里是手写 5 行 DM_IMG_<名>_* 宏——缺一个 subsdk 就直接编译不过）。
// 表由 gen_embed_ohos.py --xmods-header 生成：除 main/sdk（那两个有定制装载路径：
// main 走 exp_k、sdk 走 k_load_sdk）外的全部模块，顺序 = 规范装载序
// （rtld 最先——0 导出；然后 subsdkN）——与手写表时代的旷野之息逐项一致。
static void k_init_xmods(void) {
    int n = DM_XMODS_GEN_N < EXTRA_MAX ? DM_XMODS_GEN_N : EXTRA_MAX;
    for (int i = 0; i < n; i++) {
        xmod_t *x = &g_xmods[i];
        x->name          = dm_xmods_gen[i].name;
        x->m             = &g_k_x[i];
        x->text_end      = dm_xmods_gen[i].text_end;
        x->patch_off     = dm_xmods_gen[i].patch_off;
        x->ro_off        = dm_xmods_gen[i].ro_off;
        x->data_off      = dm_xmods_gen[i].data_off;
        x->key_page_off  = dm_xmods_gen[i].key_page_off;
        x->key_page_size = dm_xmods_gen[i].key_page_size;
        x->x0_end        = dm_xmods_gen[i].x0_end;
        x->img_end       = dm_xmods_gen[i].img_end;
    }
    g_n_xm = n;
    if (DM_XMODS_GEN_N > EXTRA_MAX)
        klog("[xmods] ⚠ 游戏有 %d 个额外模块，超过 EXTRA_MAX %d——超出部分不装载\n",
             DM_XMODS_GEN_N, EXTRA_MAX);
    else if (n == 0)
        klog("[xmods] 本游戏只有 main/sdk，无额外模块（自动少装载）\n");
    else
        klog("[xmods] 额外模块 %d 个（首：%s）\n", n, g_xmods[0].name);
}

// 每个模块的动态段位置（装载时记下，填 PLT 时用）
static struct { uint64_t rela, relasz, jmprel, pltrelsz, symtab, strtab, hash, dyn; int nexp, nrel; } g_xs[EXTRA_MAX];

// MOD0 的位置**不固定**：main/sdk 在映像很前面，但 rtld 在 0x441C（踩过：按 0x100 扫会找不到）。
// 全映像搜"MOD0 + dynamic 偏移落在映像内"的第一处。
static uint64_t k_find_dyn(uint8_t *img, size_t size) {
    for (size_t o = 0; o + 8 <= size; o += 4) {
        if (*(uint32_t *)(img + o) != 0x30444F4Du) continue;      // "MOD0"
        uint32_t off = *(uint32_t *)(img + o + 4);
        if (off && o + off + 16 < size) return (uint64_t)(uintptr_t)(img + o + off);
    }
    return 0;
}

// 装载一个模块：映射 → 键页 → RELATIVE 重定位 → 收集导出（追加到全局表）→ 封 R+X。
// 不在这里填 PLT：等**全部模块的导出都收齐**再统一填（否则 subsdk3 引 subsdk0 时会漏）。
static int k_load_extra(xmod_t *x, int idx) {
    module_t *m = x->m;
    memset(m, 0, sizeof *m);
    strncpy(m->name, x->name, sizeof m->name - 1);
    m->text_end = x->text_end; m->patch_off = x->patch_off;
    m->ro_off = x->ro_off; m->data_off = x->data_off;
    m->key_page_off = x->key_page_off; m->key_page_size = x->key_page_size;
    m->x0_end = x->x0_end; m->img_end = x->img_end;
    if (dm_ohos_map_image_alloc(m)) { klog("[%s] ✗ 映射失败\n", x->name); return -1; }

    // 键页：每个模块有自己的 8 个宿主键（各模块的改写产物按自己的键号寻址）
    for (int i = 0; i < 8; i++)
        if (pthread_key_create(&g_x_key[idx][i], NULL)) { klog("[%s] ✗ 建键失败\n", x->name); return -1; }
    uint64_t *page = (uint64_t *)(m->base + m->key_page_off);
    memset(page, 0, (size_t)m->key_page_size);
    for (int i = 0; i < 8; i++) page[i] = (uint64_t)g_x_key[idx][i] * 8;

    uint8_t *img = m->base;
    uint64_t dyn = k_find_dyn(img, (size_t)m->img_end);
    if (!dyn) { klog("[%s] ✗ 找不到 MOD0\n", x->name); return -1; }
    uint64_t rela = 0, relasz = 0, symtab = 0, strtab = 0, hash = 0, jmprel = 0, pltrelsz = 0;
    for (int i = 0; i < 64; i++) {
        uint64_t tag = *(uint64_t *)(dyn + (uint64_t)i * 16);
        uint64_t val = *(uint64_t *)(dyn + (uint64_t)i * 16 + 8);
        if (!tag) break;
        if (tag == 7) rela = val; else if (tag == 8) relasz = val;
        else if (tag == 6) symtab = val; else if (tag == 5) strtab = val;
        else if (tag == 4) hash = val; else if (tag == 0x17) jmprel = val;
        else if (tag == 2) pltrelsz = val;
        else if (tag == 25) m->init_array = val;
        else if (tag == 27) m->init_arraysz = val;
    }
    uint64_t nrel = 0;
    for (uint64_t off = 0; off + 24 <= relasz; off += 24) {
        uint64_t r_off = *(uint64_t *)(img + rela + off);
        uint64_t r_info = *(uint64_t *)(img + rela + off + 8);
        int64_t r_add = *(int64_t *)(img + rela + off + 16);
        if ((uint32_t)r_info == 1027) { *(uint64_t *)(img + r_off) = (uint64_t)img + (uint64_t)r_add; nrel++; }
    }
    // 导出 → 追加到**全局**表（g_sdk_os）：后面装的模块、以及 main 的 PLT 都能查到
    int nexp = 0;
    uint32_t nsym = hash ? *(uint32_t *)(img + hash + 4) : 0;
    for (uint32_t i = 1; hash && i < nsym && g_n_sdk_os < SDK_OS_MAX; i++) {
        uint64_t st = symtab + (uint64_t)i * 24;
        uint32_t name_off = *(uint32_t *)(img + st);
        uint16_t shndx = *(uint16_t *)(img + st + 6);
        uint64_t value = *(uint64_t *)(img + st + 8);
        if (!name_off || !shndx || !value) continue;
        g_sdk_os[g_n_sdk_os].name = (const char *)(img + strtab + name_off);
        g_sdk_os[g_n_sdk_os].addr = (uint64_t)img + value;
        g_n_sdk_os++; nexp++;
    }
    if (dm_ohos_map_image_seal(m)) { klog("[%s] ✗ 封 R+X 失败\n", x->name); return -1; }
    g_xs[idx].rela = rela; g_xs[idx].relasz = relasz; g_xs[idx].jmprel = jmprel;
    g_xs[idx].pltrelsz = pltrelsz; g_xs[idx].symtab = symtab; g_xs[idx].strtab = strtab;
    g_xs[idx].hash = hash; g_xs[idx].dyn = dyn; g_xs[idx].nexp = nexp; g_xs[idx].nrel = (int)nrel;
    klog("[%s] 已装载：base=%p RELATIVE %d 条，导出 +%d（全局表累计 %d），构造 %llu 个\n",
         x->name, (void *)m->base, (int)nrel, nexp, g_n_sdk_os,
         (unsigned long long)(m->init_arraysz / 8));
    return 0;
}

// 统一填某个模块的 PLT：优先解析到**任何已装模块的导出**，其次宿主 libc，最后兜底桩。
static void k_fill_plts_extra(xmod_t *x, int idx) {
    module_t *m = x->m; uint8_t *img = m->base;
    uint64_t self = 0, cross = 0, libc = 0, stub = 0;
    for (uint64_t off = 0; off + 24 <= g_xs[idx].pltrelsz; off += 24) {
        uint64_t r_off = *(uint64_t *)(img + g_xs[idx].jmprel + off);
        uint64_t r_info = *(uint64_t *)(img + g_xs[idx].jmprel + off + 8);
        uint32_t symi = (uint32_t)(r_info >> 32);
        uint64_t st = g_xs[idx].symtab + (uint64_t)symi * 24;
        uint32_t name_off = *(uint32_t *)(img + st);
        const char *nm = (const char *)(img + g_xs[idx].strtab + name_off);
        void *p = dm_svc_intercept(nm);      // M2：服务拦截表优先
        if (!p) p = k_sdk_lookup(nm);
        uint64_t target;
        if (p) { target = (uint64_t)(uintptr_t)p; if ((uint8_t *)p >= img && (uint8_t *)p < img + m->img_end) self++; else cross++; }
        else if (!strcmp(nm, "malloc"))  { target = (uint64_t)(uintptr_t)malloc;  libc++; }
        else if (!strcmp(nm, "calloc"))  { target = (uint64_t)(uintptr_t)calloc;  libc++; }
        else if (!strcmp(nm, "realloc")) { target = (uint64_t)(uintptr_t)realloc; libc++; }
        else if (!strcmp(nm, "free"))    { target = (uint64_t)(uintptr_t)free;    libc++; }
        else { target = (uint64_t)(uintptr_t)sdk_ret0; stub++; }
        *(uint64_t *)(img + r_off) = target;
    }
    klog("[%s] PLT 填好：自解析 %llu / 跨模块 %llu / libc %llu / 桩 %llu\n",
         x->name, (unsigned long long)self, (unsigned long long)cross,
         (unsigned long long)libc, (unsigned long long)stub);
}

// ── 子进程隔离跑构造函数（S3 的正解）──
// 为什么：subsdk0/subsdk3 里有"连驱动/连服务"的构造函数，在宿主上会**卡死**，而且卡死会连锁
// （它会握住某个宿主锁，把主线程一起拖住——两次真机实测都是这样）。
// 隔离办法：fork 一个子进程专门跑它们，审计行通过管道回传；父进程只等固定时限，超时就 kill。
// 这样"卡死"最多损失一个子进程，主路径照常往下走，账本还是完整的。
// Isolation: run those ctors in a forked child; audit lines come back through a pipe; the parent
// waits a bounded time and then kills the child. A hang then costs nothing but the child.
static void k_run_ctors_isolated(const char *tag, module_t *m, int timeout_ms) {
    int fds[2];
    if (pipe(fds) != 0) { audit("[构造审计] %s 管道失败，改为就地跑\n", tag); k_run_ctors(tag, m); return; }
    pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        g_audit_fd = fds[1];                 // 子进程：审计写管道（裸 write，无锁）
        g_ctors_direct = 1;                  // ★ 在调用线程上跑（继承来的客体 TLS），不新开 pthread
        k_run_ctors(tag, m);
        close(fds[1]);
        _exit(0);
    }
    close(fds[1]);
    // 父：把子进程的审计行原样转进主日志；超时就杀
    int fd = g_audit_fd >= 0 ? g_audit_fd : fileno(g_k_log);
    char buf[512];
    uint64_t t0 = guest_ns() / 1000000ull;
    for (;;) {
        if (guest_ns() / 1000000ull - t0 > (uint64_t)timeout_ms) {
            kill(pid, SIGKILL);
            waitpid(pid, NULL, 0);
            char msg[160];
            int n = snprintf(msg, sizeof msg, "[构造审计] %s **子进程 %d ms 未跑完 ⇒ 已杀（卡死）**\n",
                             tag, timeout_ms);
            if (fd >= 0 && n > 0) { ssize_t w = write(fd, msg, (size_t)n); (void)w; }
            return;
        }
        struct pollfd pf = { fds[0], POLLIN, 0 };
        int pr = poll(&pf, 1, 100);
        if (pr > 0) {
            ssize_t r = read(fds[0], buf, sizeof buf);
            if (r <= 0) break;                       // 子进程收尾/退出
            if (fd >= 0) { ssize_t w = write(fd, buf, (size_t)r); (void)w; }
        } else if (pr < 0 && errno != EINTR) break;
    }
    close(fds[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    char msg[192];
    int n = snprintf(msg, sizeof msg, "[构造审计] %s 子进程结束：%s%s（status=%d）\n", tag,
                     WIFSIGNALED(st) ? "被信号终止 sig=" : "正常退出",
                     WIFSIGNALED(st) ? "" : "", WIFEXITED(st) ? WEXITSTATUS(st) : st);
    if (fd >= 0 && n > 0) { ssize_t w = write(fd, msg, (size_t)n); (void)w; }
}

// 总入口：装全部额外模块 → 统一填 PLT →（可选）跑它们的构造函数。
static void k_load_extras(void) {
    k_init_xmods();                 // 从生成表填 g_xmods（换游戏只换 dm_xmods_gen.h）
    g_n_x = 0;
    for (int i = 0; i < g_n_xm; i++)
        if (k_load_extra(&g_xmods[i], i) == 0) g_n_x++;
    for (int i = 0; i < g_n_xm; i++)
        if (g_xmods[i].m->base) k_fill_plts_extra(&g_xmods[i], i);
    // 跑构造函数：**默认开**（S3）。GUI 进程拿不到环境变量（`aa start` 不传 env），所以只能靠
    // 编译期默认；要回到"只装载不跑构造"，设 NP_EXTRA_CTORS=0 或把这行改成 == 判断。
    // 逐个 setjmp 保护：某个构造函数崩了只记账、继续下一个（先看整体能走多远）。
    if (getenv("NP_EXTRA_CTORS") == NULL || atoi(getenv("NP_EXTRA_CTORS")) != 0)
        for (int i = 0; i < g_n_xm; i++)
            if (g_xmods[i].m->base) k_run_ctors_isolated(g_xmods[i].name, g_xmods[i].m, 25000);
}

// 第一批白名单：只开互斥量（Initialize/Lock/Unlock/TryLock/Finalize）。
// 为什么只开它：它只用 ArbitrateLock/Unlock，不创建句柄、不碰线程；出问题也最容易回退。
//
// ⚠ np134（真机实证，暂关）：切到 sdk 的互斥量后崩在 sdk 自己的 TLS 分配器
//   （patch 区 0x7ac308：apple 形态的 `mrs x2,tpidrro_el0; add x2,x2,键页[K18]` bump 分配）。
//   原因：**宿主上 tpidrro_el0 是垃圾值**（读出来 = 0x8000000000000000 之类），
//   main 不踩这条路径（它的 TLS 帮助函数从未被调），sdk 一调就撞。
// ✅ np136（2026-10-20，批次 B 激活）：三块拼图（① ohos 重改写 ② PLT 自解析
//   ③ TLS 影子槽）已在批次 A 真机验证（行为零变化），**本批翻开白名单**——
//   过滤式 `strstr(nm,"Mutex")` 经 main 导入清单实证只会命中这 5 个
//   （Finalize/Lock/Unlock/Initialize/TryLockMutex；无 WaitConditionVariable 误伤，
//   它的 mangle 名虽含 9MutexType 但 main 没导入它）。判据：g_nimpl_sdk=5、
//   崩 0、InitializeMutex 的 344 次调用不再落宿主空转桩。
static int k_sdk_bind_allowed(const char *nm) {
    // np136 批次 B：只放行互斥量一族。误伤已排查（见上方注释：main 导入里含 "Mutex"
    //   的 nn::os 符号恰好 5 个）；k_sdk_lookup 未命中会打「白名单符号未在 sdk 找到」告警。
    return !strncmp(nm, "_ZN2nn2os", 9) && strstr(nm, "Mutex") != NULL;
}

static void exp_k(const char *files_dir, char *out, size_t cap, size_t *used) {
    app(out, cap, used, "【实验 K】真实流程首跑：从模块启动链（0x1c8）起跑\n");
    // np120：**每轮起跑前把跨轮状态清零**。这些是 .so 的全局，同一进程里跑第二轮时
    // 若不清零，"迟到防护"会用上一轮的 nnMain 标记把本轮直接掐死在工厂分支
    // （真机实证：日志只有"工厂崩 + 红色护栏"，nnMain 一行都没有）。
    g_nnmain_entered = 0;
    g_k_faulted = 0;
    // np159/160：日志模式——np161 轮起恢复**文件 + hilog 镜像**（默认 0）。
    // 保留开关备查：DM_LOG_MODE=hilog 只走 hilog。
    { const char *m = getenv("DM_LOG_MODE"); if (m && !strcmp(m, "hilog")) g_klog_mode = 1; }
    // np125：探针线程（k_thread）里也要用文件目录（trace 落盘），存一份全局
    snprintf(g_files_dir, sizeof g_files_dir, "%s", files_dir ? files_dir : "");
    if (files_dir && *files_dir) {
        char pth[512]; snprintf(pth, sizeof pth, "%s/dm_run_test.log", files_dir);
        g_k_log = fopen(pth, "w");
        dm_log_set_mirror(g_k_log);
    }
    memset(&g_k_m, 0, sizeof g_k_m);
    strncpy(g_k_m.name, "main", sizeof g_k_m.name - 1);
    g_k_m.text_end      = DM_IMG_MAIN_TEXT_END;
    g_k_m.patch_off     = DM_IMG_MAIN_PATCH_OFF;
    g_k_m.ro_off        = DM_IMG_MAIN_RO_OFF;
    g_k_m.data_off      = DM_IMG_MAIN_DATA_OFF;
    g_k_m.key_page_off  = DM_IMG_MAIN_KEY_PAGE_OFF;
    g_k_m.key_page_size = DM_IMG_MAIN_KEY_PAGE_SIZE;
    g_k_m.x0_end        = DM_IMG_MAIN_X0_END;
    g_k_m.img_end       = DM_IMG_MAIN_END;
    if (dm_ohos_map_image_alloc(&g_k_m)) { app(out, cap, used, "  ✗ alloc 失败\n\n"); return; }

    // ===== 两步：① 应用 RELATIVE 重定位；② 392 个 PLT 槽填「记录型存根」 =====
    // 为什么：映像只是"照抄图纸"，加载器该做的重定位没做——数据段里的指针全是原始值，
    // GOT 槽还是「懒绑定占位」（存根自身地址）⇒ 第一个 SDK 调用就跳到沟里（实验 K 首跑实证）。
    {
        uint8_t *img = (uint8_t *)g_k_m.base;
        uint32_t mod0 = 0;
        for (uint32_t o = 0; o < 0x100; o += 4)
            if (*(uint32_t *)(img + o) == 0x30444F4Du) { mod0 = o; break; }   // 'MOD0'
        uint64_t dyn = (uint64_t)(img + mod0 + (mod0 ? *(uint32_t *)(img + mod0 + 4) : 0));   // 动态段偏移是相对 MOD0 起点
        uint64_t DT_RELA = 0, DT_RELASZ = 0, DT_RELACOUNT = 0;
        uint64_t DT_JMPREL = 0, DT_PLTRELSZ = 0, DT_SYMTAB = 0, DT_STRTAB = 0;
        for (int i = 0; mod0 && i < 64; i++) {
            uint64_t tag = *(uint64_t *)(dyn + (uint64_t)i * 16);
            uint64_t val = *(uint64_t *)(dyn + (uint64_t)i * 16 + 8);
            if (tag == 0) break;
            if (tag == 7) DT_RELA = val;
            else if (tag == 8) DT_RELASZ = val;
            else if (tag == 0x6ffffff9ull) DT_RELACOUNT = val;
            else if (tag == 0x17) DT_JMPREL = val;
            else if (tag == 2) DT_PLTRELSZ = val;
            else if (tag == 6) DT_SYMTAB = val;
            else if (tag == 5) DT_STRTAB = val;
        }
        uint64_t n_rel = 0, n_other = 0;
        for (uint64_t off = 0; off + 24 <= DT_RELASZ; off += 24) {
            uint64_t r_off  = *(uint64_t *)(img + DT_RELA + off);
            uint64_t r_info = *(uint64_t *)(img + DT_RELA + off + 8);
            int64_t  r_add  = *(int64_t  *)(img + DT_RELA + off + 16);
            if ((uint32_t)r_info == 1027) { *(uint64_t *)(img + r_off) = (uint64_t)img + (uint64_t)r_add; n_rel++; }
            else n_other++;
        }
        k_fill_data_imports(img);   // ★ 补数据导入常量（nn::util 数学常量等，见函数注释）
        g_hook_n = 0;   // ★ 每轮从零开始：洞是映像内的固定空间，上一轮装的桩还占着格
        k_inject_heap(img);   // ★ 喂堆：把游戏分配器要的两个全局指向宿主对象（见 k_inject_heap）
        k_inject_log(img);    // ★ 喂日志对象 + 节点池（见 k_inject_log）
        // ★ 三个分配器用「替换式」：宿主直接给内存，跳过游戏自己的实现（绕开注入被改写/分支不确定）
        k_install_replace_hook(img, 0x9f34d8, (void *)host_alloc_replace, "分配器 0x9f34d8");
        k_install_replace_hook(img, 0x9f3464, (void *)host_alloc_replace, "分配器 0x9f3464");
        k_install_replace_hook(img, 0x9f33f0, (void *)host_alloc_replace, "分配器 0x9f33f0");
        // ★ np111：RootTask::calc 的阶段状态机（P0-1/P0-2）。放在最前面，保证一定装得上。
        k_install_hook(img, 0x726838, (void *)hook_rt_calc, "RootTask::calc 入口（阶段状态机）");
        k_install_hook(img, 0x726f60, (void *)hook_rt_mgr, "RootTask 建管理器站点 0x726f60");
        // np165：装载器入口探针（门三层值）——见 hook_loader 注释
        k_install_hook(img, 0xd5b694, (void *)hook_loader, "装载器入口 0xd5b694（启动包门）");
        // np166：缓冲源空 ⇒ 零页兜底（救活建任务的工人）——见 hook_bufcopy 注释
        k_install_hook(img, 0x11e8754, (void *)hook_bufcopy, "缓冲比较/拷贝 0x11e8754（源空兜底）");
        // ⚠ 不挂 0xdb9284 首指令：它是改 SP 的 pre-index stp，钩子桩恢复路径假设 SP 不变
        k_install_hook(img, 0xdb92a4, (void *)hook_mgr_entry, "建管理器函数内 0xdb92a4（mov x19,x0）");
        k_install_hook(img, 0xdb9960, (void *)hook_orch, "装载编排入口 0xdb9960");
        k_install_hook(img, 0xdb9a54, (void *)hook_ldcall, "装载器调用点 0xdb9a54");
        k_install_hook(img, 0x7277ec, (void *)hook_rt_ctor, "RootTask 构造 0x7277ec");
        // np175："建-信号无界循环"溯源——内含 LightEvent 的对象构造器
        // ⚠ 不挂 0x103e434 首指令：那是 pre-index 改 SP 的 `str x21,[sp,#-0x30]!`；
        //   改挂序言之后的 0x103e448（`ldr x19,[x19,#0xac0]`，非 PC 相对，SP 已调好）。
        k_install_hook(img, 0x103e448, (void *)hook_ctor2f0, "对象池初始化 0x103e448");
        // np179：主线程等待环的判据点（[this+0x50] 永远为 0 ⇒ 看清 this 是谁）
        // ⭐ np182：**实验——强制等待环退出**（比"写标志"更强：直接改判据指令）。
        //   原指令 `0x9febac ldr w8,[x19,#0x50]` 改成 `mov w8,#1` ⇒ 紧随的 `cbnz` 永远跳去退出。
        //   ⚠ 必须**在挂 0x9febac 钩子之前**改：钩子会把站点首指令搬进洞、再回跳，
        //     因此它搬到的就是这条 `mov`，行为与"改判据"一致，且日志照旧。
        //   判据：主线程是否越过它继续到 0xdb9960 装载编排、以及 [fs] open 是否变非 0。
#define NP182_SKIP_WAIT 0
#if NP182_SKIP_WAIT
        *(uint32_t *)(img + 0x9febac) = 0x52800028u;      // mov w8, #1
        klog("[实验] np182：0x9febac 改为 mov w8,#1（强制等待环退出）\n");
#endif
        k_install_hook(img, 0x9febac, (void *)hook_wait50, "等待环判据 0x9febac");
        // np183b：PreInitializeApp 内部面包屑（8 点，均匀铺开）
        {
            static const uint32_t bc[] = { 0xdb92c0, 0xdb9400, 0xdb9500, 0xdb9600,
                                           0xdb9700, 0xdb9800, 0xdb9900, 0xdb9a00 };
            for (unsigned i = 0; i < sizeof bc / sizeof bc[0]; i++)
                k_install_hook(img, bc[i], (void *)hook_bcrd, "面包屑");
        }
        // np183c：**0xdb7684（ResourceMgrTask 启动）内部面包屑**——上一轮已证明主线程停在该函数内，
        //   这层用来定位它到底停在"建任务/等完成/别的分支"哪一句。
        {
            static const uint32_t bc2[] = { 0xdb7704, 0xdb7780, 0xdb7800, 0xdb7880,
                                            0xdb7904, 0xdb7980, 0xdb7a04, 0xdb7a80,
                                            // np183d：再细一档——建任务之后的三条关键指令
                                            0xdb7a24,   // CreateTaskImmediate 返回点之后
                                            0xdb7a60,   // guard 判断之后
                                            0xdb7a78 }; // 对任务对象做虚调用（vtable+0x10）之前
            for (unsigned i = 0; i < sizeof bc2 / sizeof bc2[0]; i++)
                k_install_hook(img, bc2[i], (void *)hook_bcrd, "面包屑2");
        }
        // np183e：**CreateTaskImmediate（0xa0143c）内部面包屑**——上一层已证明主线程从不返回它，
        //   这一层直接定位它内部自旋/等待的那一句。
        {
            static const uint32_t bc3[] = {
                0xa01460, 0xa014a0, 0xa014e0, 0xa01520, 0xa01560, 0xa015a0, 0xa015e0,
                0xa01620, 0xa01660, 0xa016a0, 0xa016e0, 0xa01720, 0xa01760 };
            for (unsigned i = 0; i < sizeof bc3 / sizeof bc3[0]; i++)
                k_install_hook(img, bc3[i], (void *)hook_bcrd, "面包屑3");
        }
        // np184：任务运行链内部面包屑（4 个 blr 之后的返回点）——定位到底哪个虚调用不返回
        {
            static const uint32_t bc4[] = { 0xa007a8, 0xa007b8, 0xa007cc, 0xa007e0 };
            for (unsigned i = 0; i < sizeof bc4 / sizeof bc4[0]; i++)
                k_install_hook(img, bc4[i], (void *)hook_bcrd, "面包屑4");
        }
        // np184：任务 run 实现（0x708be4）内部面包屑——上一层已把"卡死"收窄到这条尾调，
        //   而三个候选里只有它不是 `ret`。
        {
            static const uint32_t bc5[] = { 0x708c28, 0x708c60, 0x708ca0, 0x708ce4,
                                            0x708d20, 0x708d60, 0x708da0, 0x708de0 };
            for (unsigned i = 0; i < sizeof bc5 / sizeof bc5[0]; i++)
                k_install_hook(img, bc5[i], (void *)hook_bcrd, "面包屑5");
        }
        // np185 实验 C v2：主线程终卡点定位（每次都记，不去重）。
        //   背景：本轮实证主线程最后活动 = 第 120 次分配（[大分配] 站点 +0xa1d768，即
        //   "注册缓冲区"函数 0xa1d67c 内 bl 0x9f3464 的返回点），此后再无分配/导入日志、
        //   线程快照 R 态。bc6（首次命中式）已证明该路径能完整跑通，但回答不了"最后一次调用"
        //   卡在哪——改用每次命中都计数的 C 点钩子：日志末尾最后出现的 C 点即卡点前最后一站。
        //   路径：0xa1d69c 入口 → 0xa1d77c 分 0x1c0 返回 → 0xa1d79c SDK 注册(0x1632920)返回
        //         → 0xa1d7b0 vt+0x18 返回 → 0xa1d7d8 SDK(0x1632930)+锁(+0x40)返回 → 0xa1d818 链表完
        //   若某 C 点的计数停在比后点少 1 ⇒ 卡在这两点之间（如 0xa1d7d8 不进 ⇒ 卡在 0x1632930
        //   或用户态互斥量自旋——M1 已知互斥量纯用户态自旋、不踩 svc 门）。
        {
            static const uint32_t cpts[] = { 0xa1d69c, 0xa1d77c, 0xa1d79c,
                                             0xa1d7b0, 0xa1d7d8, 0xa1d818,
                                             0xa091b0, 0xa091fc, 0xa0923c, 0x1086ac8,
                                             0xa1d56c, 0xa1d57c, 0xa0b188, 0x107e858,
                                             0x107e86c, 0x1086efc, 0x107e894 };
            for (unsigned i = 0; i < sizeof cpts / sizeof cpts[0]; i++)
                k_install_hook(img, cpts[i], (void *)hook_cpt, "np185 C点");
        }
        // np185 实验 D：KingSystem 落户观测（0x726b00 = RootTask::calc 内注册调用的存储后）。
        //   周期 dump 读 [RootTask+0x358] 恒 0 ⇒ 打"新对象/落户结果/对象的 +0xd4"三件套。
        k_install_hook(img, 0x726b00, (void *)hook_e36, "np185 KingSystem 落户 0x726b00");
        // np183：编排进度标记（早退补支路 / StarterPackMgr 段）
        k_install_hook(img, 0xdb9ae8, (void *)hook_pm_early, "编排 0xdb9ae8（堆空补支路）");
        k_install_hook(img, 0xdb99e4, (void *)hook_pm_spm, "编排 0xdb99e4（StarterPackMgr 段）");
        // np184：任务"运行"虚方法的生命周期链（0xa0078c 永不返回）
        k_install_hook(img, 0xa00794, (void *)hook_taskrun, "任务运行 0xa0078c 的 5 个虚槽");
        // np113：查"建出来就被销毁"（见 hook_list_ins/hook_task_exit 的注释）
        //   ⚠ 0xa01fe0 已由 hook_e11 挂着（同一站点不能挂两次），那边加了调用计数与 [+0xe0] 表。
        k_install_hook(img, 0xa01aec, (void *)hook_list_ins, "任务链 链表插入 0xa01aec");
        k_install_hook(img, 0x726768, (void *)hook_task_exit, "RootTask 退出 0x726768（槽 13）");
        k_install_hook(img, 0x7265b8, (void *)hook_rt_prepare, "RootTask::prepare 0x7265b8");
        // np116：阶段 0 内部崩点（见 hook_vol_init 注释）
        // ⚠ 0x9fd9c4 **不能挂**：它的首指令是 `b 0x16321a0`（尾调），不是函数入口——
        //   挂上去后桩"执行原指令 → 回站点+4"会把尾调变成"返回后继续跑函数体"，流程全乱
        //   （np117 真机实证：日志停在工厂分支 + 红色护栏）。要看它，只能挂它的**调用者**。
        k_install_hook(img, 0x9fc450, (void *)hook_vol_init, "Volante 初始化 0x9fc450");
        k_install_hook(img, 0xa041d0, (void *)hook_gfx_flush, "Volante 收尾 0xa041d0");
        // np121：NVN 函数表注册（见 hook_nvn_tab_init 注释）
        k_install_hook(img, 0x9b2938, (void *)hook_nvn_tab_init, "NVN 函数表 0x9b2938");
        k_install_hook(img, 0x9b3030, (void *)hook_nvn_tab_store, "NVN 函数表 写入 0x9b3030");
        // np123：显示对象有没有建起来 / 帧循环拿谁去用
        k_install_hook(img, 0x9fc8c4, (void *)hook_gfx_set128, "Volante 显示对象 0x9fc8c4");
        k_install_hook(img, 0x9fd490, (void *)hook_gfx_frame, "Volante 帧循环 0x9fd490");
        // 崩链上的两个构造器入口（记录 this）
        k_install_hook(img, 0xa08ef4, (void *)hook_ctor_log, "构造器 0xa08ef4 入口");
        k_install_hook(img, 0xa08e94, (void *)hook_ctor2_log, "构造器 0xa08e94 入口");
        // ★★ 沿崩链逐点插桩（链见上面 hook_p* 的注释）。先挂"信息量最大"的链点，
        //    再挂批量站点——洞只有 227 格（0x10000/0x120），要紧的放前面，满了也丢不了它们。
        k_install_hook(img, 0x9fe24c, (void *)hook_p9fe24c, "链点 0x9fe24c（链顶函数入口）");
        k_install_hook(img, 0x9fe2e8, (void *)hook_p9fe2e8, "站点→0xa1e484（交出注入对象处）");
        k_install_hook(img, 0xa1e484, (void *)hook_p1e484, "链点 0xa1e484（b 尾调用桩）");
        k_install_hook(img, 0xa1dd94, (void *)hook_p1dd94, "链点 0xa1dd94（分配 0x100 + 调构造器）");
        k_install_hook(img, 0xa1d8e0, (void *)hook_p1d8e0, "链点 0xa1d8e0（派生构造器入口）");
        // ★ 钉子 #8（当前卡点：0xa00a78 收到空标签 ⇒ 返 0 ⇒ 0xa01790 写 [x21+0xc8] 崩）：
        //   五个钩子都排在批量挂点之前，保证装得上（离线分析见 hook_e* 处注释）。
        k_install_hook(img, 0x9fe82c, (void *)hook_e1, "钉子8 PathA 0x9fe82c 入口");
        k_install_hook(img, 0xa01790, (void *)hook_e2, "钉子8 0xa01790 入口（标签+调用点）");
        k_install_hook(img, 0xa00a78, (void *)hook_e3, "钉子8 0xa00a78 入口（标签）");
        k_install_hook(img, 0x9fe910, (void *)hook_e4, "钉子8 tag2 目标 0x9fe910 入口");
        k_install_hook(img, 0xa01814, (void *)hook_e5, "钉子8 分发返回点 0xa01814");
        // ★ 钉子 #8 同轮探针（根因验证三连，只读）：工厂真入口 0xa00f8c（0xa00f90 是函数体
        //   第二条指令）、池注册 0xa018f8、G 真入口 0xa01d70（0xa01d88 同理是体内指令）。
        k_install_hook(img, 0xa00f8c, (void *)hook_e6, "钉子8 工厂 0xa00f8c 入口（父对象）");
        k_install_hook(img, 0xa018f8, (void *)hook_e7, "钉子8 池注册 0xa018f8 入口（记录源）");
        k_install_hook(img, 0xa01d70, (void *)hook_e8, "钉子8 G 0xa01d70 入口（池状态）");
        // ★ 钉子 #8 续修探针（runner 链，只读）：激活检查 + 运行转发器，见 hook_e9/e10 注释。
        k_install_hook(img, 0xa01cf0, (void *)hook_e9, "钉子8 runner 检查 0xa01cf0 入口");
        k_install_hook(img, 0x9fdccc, (void *)hook_e10, "钉子8 运行转发器 0x9fdccc 入口");
        k_install_hook(img, 0xa01fe0, (void *)hook_e11, "钉子8 runner 入口 0xa01fe0（链表遍历）");
        // np186 实验 I：flags 补位挂在 runner 循环体的 ldrb 站点（0xa02030）——
        //   挂 ldp(0xa0202c) 会读到旧 x22 补错对象（2026-10-22 首轮翻车实证）。
        //   开关默认关；实验轮把 NP186_FLAGS 改 1 重编，跑完改回。
        k_install_hook(img, 0xa02030, (void *)hook_flagsfix, "np186 flags 补位 0xa02030（runner 外层门 ldrb）");
        k_install_hook(img, 0xa01be8, (void *)hook_e12, "钉子8 销毁入口 0xa01be8（数组 dump）");
        // ★ 注册体/构造尾定位钩（见 hook_e13/e14 注释）
        k_install_hook(img, 0xa01928, (void *)hook_e13, "钉子8 注册体 0xa01928");
        k_install_hook(img, 0xa00d4c, (void *)hook_e14, "钉子8 构造尾 0xa00d4c");
        // ★ 日志系统探针（见 hook_e15/e16 注释）
        // ⚠ np114：0xf35374 改成**替换式**（宿主实现 + 串行化 + 转写），不再用记录式——
        //   记录式只是旁观，救不了"两线程争同一空闲链表"的竞态（真机已崩在 0xf353a8）。
        k_install_replace_hook(img, 0xf35374, (void *)hook_game_log, "客体日志 0xf35374 → 宿主（串行）");
        k_install_hook(img, 0xf35400, (void *)hook_e16, "日志 遍历 0xf35400");
        k_install_hook(img, 0xa01924, (void *)hook_e17, "钉子8 注册体上锁前 0xa01924");
        // ★ Task #16：第一个任务由谁创建（见 hook_e18/e19 注释；离线结论见进度文档 §20）
        k_install_hook(img, 0xa0143c, (void *)hook_e18, "任务 CreateTaskImmediate 0xa0143c 入口");
        k_install_hook(img, 0xa01490, (void *)hook_e19, "任务 建出任务回填 0xa01490（x0=新任务）");
        // ★ Task #16 第二轮：名字级追踪（见 hook_e20/e21 注释）
        k_install_hook(img, 0xa0913c, (void *)hook_e20, "名字 建/查管理器 0xa0913c 入口");
        k_install_hook(img, 0x9fe380, (void *)hook_e21, "名字 方法树创建结果 0x9fe380");
        k_install_hook(img, 0xa09144, (void *)hook_e22, "名字 建/查管理器 0xa09144 实体入口");
        k_install_hook(img, 0x9fe984, (void *)hook_e23, "框架 SystemManagers 建 ThreadMgr 0x9fe984");
        k_install_hook(img, 0xa112fc, (void *)hook_e24, "挂载 方法树列表 0xa112fc");
        // ⚠ np34 教训：原来挂在 0x9ff398（`adrp x8,#0x1ba6000`）——那是**给后续指令喂寄存器**的位置，
        //   宿主函数调用会破坏 caller-saved 的 x8，导致紧跟的 `add x8,x8,#0x891` 算出垃圾指针。
        //   改挂 0x9ff3a0（`str x21,[x23,#0x350]`，纯 store，不喂任何寄存器给后续指令）。
        k_install_hook(img, 0x9ff3a0, (void *)hook_e25, "框架 挂 7 棵方法树 0x9ff3a0（纯 store）");
        // ⚠ 2026-10-18：洞容量 227 格、已用 218 ⇒ 快满了。下面两条 all_branchers 最多各占
        //   64/200 格，且"找站点"的使命已经完成（钉子 #8 早已闭环）⇒ 暂时关闭，把格子让给
        //   **启动链追踪**（一次拿到 nnMain 的完整调用顺序，比逐个微探针性价比高得多）。
        // k_hook_all_branchers(img, 0xa1d8e0, (void *)hook_s1d8e0, "站点→0xa1d8e0", 64);
        // k_hook_all_branchers(img, 0xa08e94, (void *)hook_any_site, "站点→构造器 0xa08e94", 200);
        // 诊断 sead 单例查找 0xa0913c 的分支路径 —— 同样已完成使命，关掉腾格子。
        // k_install_hook(img, 0xa0919c, (void *)hook_lk_a, "查找路径 0xa0919c");
        // k_install_hook(img, 0xa091e4, (void *)hook_lk_b, "查找路径 0xa091e4");
        // k_install_hook(img, 0xa091dc, (void *)hook_lk_c, "查找路径 0xa091dc");
        // k_install_hook(img, 0xa092a0, (void *)hook_lk_d, "查找路径 0xa092a0");
        // k_install_hook(img, 0xa092a8, (void *)hook_lk_e, "查找路径 0xa092a8");
        // k_install_hook(img, 0xa09184, (void *)hook_dbA, "诊断 0xa09184（调 0xa0e2f8）");
        // k_install_hook(img, 0xa0e330, (void *)hook_dbB, "诊断 0xa0e330（取值点之后）");
        // k_install_hook(img, 0xa0c1b0, (void *)hook_dbC, "诊断 0xa0c1b0（调 sead 助手前）");
        // k_install_hook(img, 0xa1d57c, (void *)hook_dbD, "诊断 0xa1d57c（sead 助手入口）");
        // ★ 启动链追踪：nnMain 的 bl 目标按出现顺序全部挂上（一个宿主函数，靠 f[FR_SITE] 区分）。
        //   目的：拿到"初始化走到第几步、在哪一步之后断掉"的完整顺序图。
        for (size_t bi = 0; bi < sizeof g_boot_sites / sizeof g_boot_sites[0]; bi++)
            k_install_hook(img, g_boot_sites[bi], (void *)hook_boot, "启动链");
        // np68：设值函数换替身（覆盖上面那条记录式钩子——同一站点只能挂一个）。
        // ⚠ 真机：装了替身之后**确定性卡死在静态初始化 ctor #385**（同 np58 的症状）——
        //   说明 ctor #385 会调这个设值函数，而替身在那条路径上把它打断（也可能是同站点
        //   二次挂接和记录钩子的洞冲突）。**暂不启用**，留作线索；恢复后 np67 行为可复现。
        // k_install_replace_hook(img, 0xa2469c, (void *)hook_setter, "设值 0xa2469c 替身");
        // 决定性挂点：`mov x22, x0`（x22==0 ⇒ 跳过 7 个初始化直接进主循环）
        k_install_hook(img, 0x73557c, (void *)hook_e26, "关键 x22 取值 0x73557c");
        k_install_hook(img, 0x7355a8, (void *)hook_e27, "对照 cbz 未跳则命中 0x7355a8");
        k_install_hook(img, 0xa051c4, (void *)hook_e28, "建对象 worker 0xa051c4（看 x2 是谁传的 0）");
        // 崩点链最上游的解析器：原实现一路缺数据恒返 0 ⇒ worker 的 x2=0 ⇒ 崩 0xa05324。
        // np54 起换成**替身**：宿主直接返全零对象，越过这道坎看下一层需求（§八影子对象打法）。
        // ★ np215 实验 B：**取消键解析替换**（拐杖复审）。
        // np54 时代替换的理由：原实现"建临时对象 → 按 ':' 切分 → 遍历已注册解析器"在宿主上
        //   缺数据、恒返回 0 ⇒ worker 的 x2=0 ⇒ 崩在 0xa05324。于是返一个全零对象越过该坎。
        // 现在（np213 真机）证据表明拐杖有害：
        //   `[键解析·替身] 键=""` + `[settings] eq(lang=..., code="") → 0` 全部不匹配
        //   ⇒ 语言/区域判定失败（游戏读了 System/RegionLangMask.txt 也白读）。
        // 离线复核：原实现的依赖**都有效**——`*(0x2264200)`=0x1c429ee（分隔符数据）、
        //   `*(0x22644d8)`=0x206b0e8（vtable 非零）。np54 的"缺数据"现在可能已就绪
        //   （游戏跑到帧循环，注册表很可能已建）。
        // 做法：默认**不替换**（让原实现跑）；设 NP_PARSE_REPL=1 可退回旧替身（真机 GUI
        //   拿不到 env ⇒ 实际改代码里的 g_parse_repl 初值）。保留一个入口观察钩子打键名。
        //   np221：再叠一层运行时开关（/data/local/tmp/dm_switch.txt 位0），免重构建 A/B。
        if (g_parse_repl || dm_probe_sw(0))
            k_install_replace_hook(img, 0x9fa4c8, (void *)hook_rep_parse, "键解析·替身 0x9fa4c8（返全零对象）");
        else
            k_install_hook(img, 0x9fa4c8, (void *)hook_parse_watch, "键解析·观察 0x9fa4c8（原实现放行）");
        // 解析派发（worker 的调用者，函数指针进来）：直接看它喂给构造函数的 x2
        k_install_hook(img, 0xa05fa4, (void *)hook_e30, "派发 0xa05fa4（看 x2）");
        // ★ 崩溃点仪器（2026-10-20）：PostCalc 入口——本轮崩点就是它收到 NULL 句柄
        //   （ldr x20,[x0] 读 0）。记录 x0 + x30(LR=调用方返回址)，见 hook_pcalc 注释。
        k_install_hook(img, 0x103B1F0, (void *)hook_pcalc, "PostCalc 0x103b1f0 入口（NULL 句柄崩点）");
        // ★ np185 实验 A（纯观测）：CTA 派发点显形 + F_wait 调用者。判据落 dm_gate.txt。
        //   站点均为安全指令；0xa01570 是 bl（安装器按原站点 PC 重算 imm26）。
        k_install_hook(img, 0xa014d8, (void *)hook_e31, "np185 [arg+0xb0] 回调 0xa014d8");
        k_install_hook(img, 0xa01514, (void *)hook_e32, "np185 任务 vt+0x50 0xa01514");
        k_install_hook(img, 0xa01570, (void *)hook_e33, "np185 类目 0xa00804 调用点 0xa01570");
        k_install_hook(img, 0x9feb3c, (void *)hook_e34, "np185 F_wait 入口 0x9feb3c（打调用者 x30）");
        // np210 实验 A：M3 惰性挂接链验证（0x9fc6f4 函数体内首次分配前 / 0x9fc8c4 挂接点）。
        // 判据落 dm_gate.txt：enter 不响 = H1（没触发）；enter 响 attach 不响 = H2（半路断）；
        // attach 响但管理器 vt 不是 0x21b5d40 = H3（挂错对象）。接口表 dump 见 hook_e34。
        k_install_hook(img, 0x9fc6f4, (void *)hook_m3_enter, "np210 子系统初始化 0x9fc6f4（首次分配前）");
        k_install_hook(img, 0x9fc8c4, (void *)hook_m3_attach, "np210 挂接点 0x9fc8c4（[x19+0x128]<-x23）");
        // np211g：np211f 定谳的真正挂接点（0xa0913c 分配器协议链的终点）
        k_install_hook(img, 0xc5274c, (void *)hook_m3_real_attach, "np211g 真挂接点 0xc5274c（[x19+0x128]<-x0）");
        // np211g 诊断链：宿主函数 / 取子对象 / TLS 取的入口与返回
        k_install_hook(img, 0xc52664, (void *)hook_np211g_host, "np211g 宿主函数 0xc52664 入口");
        k_install_hook(img, 0xa0913c, (void *)hook_np211g_sub, "np211g 取子对象 0xa0913c 入口");
        k_install_hook(img, 0xa0e2f8, (void *)hook_np211g_tlsget, "np211g TLS取 0xa0e2f8 入口");
        k_install_hook(img, 0xa0e338, (void *)hook_np211g_tlsret, "np211g TLS取返回 0xa0e338");
        // np211i：排除 0xc5274c 后的三个新候选
        k_install_hook(img, 0xa3581c, (void *)hook_np211i_list, "np211i 链表挂接 0xa3581c");
        k_install_hook(img, 0xc523ec, (void *)hook_np211i_singleton, "np211i 惰建单例真入口 0xc523ec");
        k_install_hook(img, 0x1187c74, (void *)hook_np211i_sub2, "np211i 子对象挂接 0x1187c74");
        // np210 实验 B（第二轮）：门态 + 图形初始化入口 + 两个备选挂点。
        // 第一轮已定谳"挂接函数体没跑"；离线追出它 = 图形/显示初始化 0x9fc450，
        // 唯一调用点 0x7355a0 前有单例门（0x735574 cbz）。本轮看门跳不跳、单例空不空、
        // 图形初始化进没进、另外两个 +0x128 写点跑没跑。
        k_install_hook(img, 0x735574, (void *)hook_m3_gate, "np210 单例门 0x735574（cbz，x0=getter结果）");
        k_install_hook(img, 0x9fc450, (void *)hook_m3_ginit, "np210 图形初始化入口 0x9fc450");
        k_install_hook(img, 0x9fdaf8, (void *)hook_m3_alt1, "np210 备选挂点一 0x9fdaf8（兄弟类ctor +0x128）");
        k_install_hook(img, 0x9ffbd8, (void *)hook_m3_alt2, "np210 备选挂点二 0x9ffbd8（+0x128）");
        // np210 实验 C（第三轮）：X 类槽30方法入口（谁派发、X 收没收到）+ X 的工厂入口（X 谁建的）。
        k_install_hook(img, 0x9fd858, (void *)hook_m3_m30, "np210 X类槽30方法入口 0x9fd858");
        k_install_hook(img, 0xa00f90, (void *)hook_m3_fac, "np210 X工厂入口 0xa00f90");
        // np133：装载 sdk（M1）。失败不致命：装载不了就全走宿主实现，行为与之前一致。
        if (k_load_sdk() != 0)
            klog("[sdk] 未装载（继续用宿主实现，行为不变）\n");
        // S2：把 rtld / subsdk0-3 也装进来（NP_EXTRA=0 可关）。失败只记账，不影响已有链路。
        if (getenv("NP_EXTRA") == NULL || atoi(getenv("NP_EXTRA")) != 0) k_load_extras();
        // M4 自测四连（2026-10-22 真机档）：状态/设备/命令缓冲/纹理 四层的 Set→Get 往返。
        // 判据：四个失败数**全 0**。落独立文件，不受日志预算影响。
        {
            // 同时打 klog（主日志）+ note_append（独立文件）：上一次只写文件，结果"文件没出现"时
            // 完全看不出是没执行、还是执行到一半崩了。两条通道互相兜底（诊断通道冗余原则）。
            klog("[自测] 进入 NVN 四层自测\n");
            int a = dm_nvn_state_selftest();
            klog("[自测] state=%d\n", a);
            int b = dm_nvn_dev_selftest();
            klog("[自测] dev=%d\n", b);
            int c = dm_nvn_cb_selftest();
            klog("[自测] cb=%d\n", c);
            int d = dm_nvn_tex_selftest();
            klog("[自测] tex=%d\n", d);
            char l[192];
            snprintf(l, sizeof l, "[自测] NVN 四层：state=%d dev=%d cb=%d tex=%d（全 0 才算过；函数数 %d/%d/%d/%d）\n",
                     a, b, c, d,
                     dm_nvn_state_count(), dm_nvn_dev_count(),
                     dm_nvn_cb_count(), dm_nvn_tex_count());
            note_append("dm_selftest.txt", l);
            klog("[真执行] 进入合成冒烟\n");
            int st = dm_nvn_exec_smoketest();
            klog("[真执行] 冒烟返回 %d\n", st);
            // l 要够大：vk_backend_shader 的报告有 ~700 字节，判定行在**末尾**（截断就看不到结论）
            static char rep[1600];
            snprintf(rep, sizeof rep, "[真执行] 合成冒烟：%s（verdict=%d）\n",
                     dm_nvn_exec_report(), st);
            note_append("dm_selftest.txt", rep);
        }
        // np146：把 main 映像的 R+X 封区告知胶水层（写 guest 内存前的可写性判据）。
        //   ⚠ 必须在写任何 guard 字节之前——否则拿 /proc/self/maps 兜底，
        //   而本进程 maps 有上千行，缓冲区一截断就"查无此页"（np146 实测 27 个全被误判）。
        //   封区两段：[0, text_end) 与 [patch_off, end)（见 map_image_ohos.c 的 seal）。
        {
            const uint64_t ranges[][2] = {
                { 0,                    DM_IMG_MAIN_TEXT_END },
                { DM_IMG_MAIN_PATCH_OFF, DM_IMG_MAIN_END      },
            };
            dm_sdk_set_sealed_ranges((uint64_t)(uintptr_t)img, ranges, 2);
            klog("[封区] 已告知胶水层：[%#llx,%#llx) + [%#llx,%#llx)（基址 %p）\n",
                 (unsigned long long)0, (unsigned long long)DM_IMG_MAIN_TEXT_END,
                 (unsigned long long)DM_IMG_MAIN_PATCH_OFF,
                 (unsigned long long)DM_IMG_MAIN_END, (void *)img);
        }
        g_nplt = 0; g_plt_calls = 0;
        for (uint64_t off = 0; off + 24 <= DT_PLTRELSZ; off += 24) {
            uint64_t r_off  = *(uint64_t *)(img + DT_JMPREL + off);
            uint64_t r_info = *(uint64_t *)(img + DT_JMPREL + off + 8);
            uint32_t symi   = (uint32_t)(r_info >> 32);
            const char *nm  = (const char *)(img + DT_STRTAB + *(uint32_t *)(img + DT_SYMTAB + (uint64_t)symi * 24));
            if (g_nplt < 512) { g_plt[g_nplt].slot = (uint64_t)(img + r_off); g_plt[g_nplt].name = nm; g_nplt++; }
            // ★ 已实现的导入直连宿主函数（目前 = nn::fs 23 个 + libc 一族）；其余仍落记录型存根。
            //   分流按**符号名**，换游戏不用改表（只要 ABI 同名）。
            //   ⚠ np112：libc 一族（memcpy/memset/str*/math）此前全部落到存根 ⇒ memcpy 空操作
            //   ⇒ 所有结构体拷贝为 0 ⇒ 任务记录 tag=0 ⇒ RootTask 建不出来（2026-10-19 真机实证）。
            void *impl = dm_fs_impl(nm);
            const char *src = "fs";
            if (!impl) { impl = dm_libc_impl(nm); src = "libc"; }
            // np118/np122：NVN 的"前门"。游戏图形初始化第一步就是 nvnBootstrapLoader(名字) 拿函数指针；
            //   返回 0 ⇒ Volante 初始化早退 ⇒ NVN 函数表整个没注册 ⇒ 建对象 worker 空指针崩。
            //   ⚠ 必须走 host_nvn_bootstrap（查不到也给兜底桩），不能直接绑框架层（它查不到返回 0）。
            if (!impl && !strcmp(nm, "nvnBootstrapLoader")) { impl = (void *)host_nvn_bootstrap; src = "nvn"; }
            // np133：**M1 第二步** —— 白名单里的 nn::os 符号解析到 sdk 自己的实现。
            //   第一批只开互斥量（只用 ArbitrateLock/Unlock，不建句柄、不碰线程）。
            if (!impl && k_sdk_bind_allowed(nm)) {
                void *p = dm_svc_intercept(nm);   // M2：服务拦截表优先
                if (!p) p = k_sdk_lookup(nm);
                if (p) { impl = p; src = "sdk"; g_nimpl_sdk++; }
                else klog("[sdk] ⚠ 白名单符号未在 sdk 找到：%s（仍走宿主）\n", nm);
            }
            if (impl) { g_nimpl++; if (g_nimpl <= 80) klog("[导入] 直连 %s（%s）\n", nm, src); }
            *(uint64_t *)(img + r_off) = impl ? (uint64_t)(uintptr_t)impl
                                              : (uint64_t)(uintptr_t)&plt_stub;
        }
        // np125：trace 开局（清环）+ 落一份"符号下标 → 名字"表，供离线把 trace 里的调用还原成名字
        ktrace_reset();
        {   // np127：定时落盘线程（卡死时也能拿到最新轨迹）
            pthread_t t;
            if (pthread_create(&t, NULL, ktrace_timer, NULL) == 0) pthread_detach(t);
        }
        {
            char p[512];
            snprintf(p, sizeof p, "%s/dm_plt_names.txt", files_dir);
            FILE *fp = fopen(p, "w");
            if (fp) {
                for (int i = 0; i < g_nplt; i++) fprintf(fp, "%d\t%s\n", i, g_plt[i].name);
                fclose(fp);
                klog("[TRACE] 符号表已落盘：%s（%d 条）\n", p, g_nplt);
            }
        }
        {   // np82：记下本 .so 的装载基址，队列/fs 日志里的"外"调用者才能对上号
            Dl_info di;
            if (dladdr((void *)&plt_stub, &di) && di.dli_fbase) {
                g_self_base = (uint64_t)(uintptr_t)di.dli_fbase;
                klog("[自检] 本 .so 基址 = %#llx（%s）\n",
                     (unsigned long long)g_self_base, di.dli_fname ? di.dli_fname : "?");
            }
        }
        dm_fs_set_log(klog);
        dm_fs_set_base(img);   // np80：让 fs 日志能把调用者 LR 换算成映像内偏移
        dm_libc_set_log(klog);  // np112：libc 胶水层同此
        dm_svc_sync_set_log(klog);   // np132：svc 同步层（事件/仲裁锁/条件变量）
        dm_svc_sync_init();          // 每轮初始化本线程的等待条件变量
        dm_sdk_set_log(klog);   // 通用 SDK 胶水层也用同一份日志
        dm_nvn_set_log(klog);   // 通用 NVN 胶水层同此
        dm_nvn_set_extra(nvn_real_lookup);   // np118：把池/缓冲与设备两个特殊符号挂进 bootstrap
        dm_nvn_vk_set_log(klog);   // 池/缓冲层同此
        dm_fs_selftest();
        // np65 槽体检（装载后、游戏跑之前）：崩点 +0xA246A4 的 `str x0,[*(0x2280eb8)]` 里那个槽
        // 运行期是 0，但镜像里它有 RELA（addend 0x22d87e0）。这里在**游戏动手之前**读一次：
        //   = base+0x22d87e0 ⇒ 重定位是对的，是游戏后来把它写成 0（应用层问题）
        //   = 0            ⇒ 装载层漏了这条重定位（框架 bug，影响面大）
        klog("[槽体检] 装载后：0x2280eb8=%#llx（应=%#llx）｜对照 0x2280a98=%#llx 0x227f900=%#llx\n",
             (unsigned long long)*(const uint64_t *)(img + 0x2280eb8),
             (unsigned long long)((uintptr_t)img + 0x22d87e0),
             (unsigned long long)*(const uint64_t *)(img + 0x2280a98),
             (unsigned long long)*(const uint64_t *)(img + 0x227f900));
        klog("[重定位诊断] mod0=%#x dyn=%#llx DT_RELA=%#llx RELASZ=%#llx JMPREL=%#llx PLTRELSZ=%#llx SYMTAB=%#llx STRTAB=%#llx\n",
             mod0, (unsigned long long)dyn, (unsigned long long)DT_RELA, (unsigned long long)DT_RELASZ,
             (unsigned long long)DT_JMPREL, (unsigned long long)DT_PLTRELSZ,
             (unsigned long long)DT_SYMTAB, (unsigned long long)DT_STRTAB);
        klog("[重定位诊断] 映像前 16 字节: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
             img[0],img[1],img[2],img[3],img[4],img[5],img[6],img[7],img[8],img[9],img[10],img[11],img[12],img[13],img[14],img[15]);
        appf(out, cap, used, "  ✓ 重定位：RELATIVE %llu 条（共 %llu），非 RELATIVE %llu 条\n",
             (unsigned long long)n_rel, (unsigned long long)(DT_RELASZ / 24), (unsigned long long)n_other);
        appf(out, cap, used, "  ✓ SDK 槽打桩：%d 个 PLT 槽 → 记录型存根（调用即记名）\n", g_nplt);
        appf(out, cap, used, "  ✓ M1：%d 个导入解析到 sdk（白名单 = nn::os 互斥量）\n", g_nimpl_sdk);
    }
    if (dm_ohos_map_image_seal(&g_k_m)) { app(out, cap, used, "  ✗ seal 失败\n\n"); return; }
    appf(out, cap, used, "  ✓ 装载并封 R+X：base %p\n", (void *)g_k_m.base);

    for (int i = 0; i < 8; i++)
        if (pthread_key_create(&g_k_key[i], NULL)) { app(out, cap, used, "  ✗ 建键失败\n\n"); return; }
    uint64_t *page = (uint64_t *)(g_k_m.base + DM_IMG_MAIN_KEY_PAGE_OFF);
    memset(page, 0, DM_IMG_MAIN_KEY_PAGE_SIZE);
    page[DM_IMG_MAIN_KEY_K18_IDX]       = (uint64_t)g_k_key[0] * 8;
    page[DM_IMG_MAIN_KEY_KTLS_IDX]      = (uint64_t)g_k_key[1] * 8;
    page[DM_IMG_MAIN_KEY_KTLR_IDX]      = (uint64_t)g_k_key[2] * 8;
    page[DM_IMG_MAIN_KEY_KGATE_IDX]     = (uint64_t)g_k_key[3] * 8;
    page[DM_IMG_MAIN_KEY_KRES0_IDX]     = (uint64_t)g_k_key[4] * 8;
    page[DM_IMG_MAIN_KEY_KRES1_IDX]     = (uint64_t)g_k_key[5] * 8;
    page[DM_IMG_MAIN_KEY_KASPECT_IDX]   = (uint64_t)g_k_key[6] * 8;
    page[DM_IMG_MAIN_KEY_KASPECTUI_IDX] = (uint64_t)g_k_key[7] * 8;

    struct sigaction sa, o1, o2, o3, o4;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = gate_fault_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &o1); sigaction(SIGBUS, &sa, &o2);
    sigaction(SIGILL, &sa, &o3);  sigaction(SIGTRAP, &sa, &o4);
    {   // 诊断（2026-10-20）：我们的处理器真的装上了吗？OHOS musl 的 sigaction 包了一层
        //   "信号链"——真实 disposition 是它自己的 signal_chain_handler，各处理器在链里
        //   排队（hilog 见 MUSL-SIGCHAIN：signal_chain_handler 先跑两级、随后 DfxSignalHandler
        //   接管落盘）。读回来看：真实 disposition 是谁、装之前的旧主（o1）是谁。
        struct sigaction rb;
        sigaction(SIGSEGV, NULL, &rb);
        klog("[信号] 装后读回：SIGSEGV 真实 disposition=%p flags=%#x｜装之前的旧主=%p\n",
             (void *)rb.sa_sigaction, rb.sa_flags, (void *)o1.sa_sigaction);
    }

    g_k_hits = 0; g_k_faulted = 0; g_k_rv = 0;
    pthread_t th;
    if (pthread_create(&th, NULL, k_thread, NULL) == 0) pthread_join(th, NULL);
    else app(out, cap, used, "  ✗ pthread_create 失败\n");

    sigaction(SIGSEGV, &o1, NULL); sigaction(SIGBUS, &o2, NULL);
    sigaction(SIGILL, &o3, NULL);  sigaction(SIGTRAP, &o4, NULL);

    appf(out, cap, used, "  SDK 槽被真实调用 %d 次%s\n", g_plt_calls, g_plt_calls ? "（名见 dm_run_test.log）" : "");
    appf(out, cap, used, "  宿主堆被分配 %llu 次%s\n", (unsigned long long)g_alloc_calls,
         g_alloc_calls ? "（游戏分配器已通）" : "（未走到分配器）");
    appf(out, cap, used, "  崩链插桩命中：链点 0x9fe24c %d / 站点→0xa1e484 %d / 0xa1e484 %d / 0xa1dd94 %d / "
                         "0xa1d8e0 %d；构造器 0xa08e94 入口 %d 次、站点 %d 次（明细见 dm_run_test.log）\n",
         g_n_9fe24c, g_n_9fe2e8, g_n_1e484, g_n_1dd94, g_n_1d8e0, g_n_ctor2, g_n_any);
    appf(out, cap, used, "  钉子8 命中：PathA入口 %d / 0xa01790 %d / 0xa00a78 %d / 0x9fe910 %d / 分发返回 %d"
                         "（标签与返回值见 dm_run_test.log 的 [钉子8] 行）\n",
         g_n_e1, g_n_e2, g_n_e3, g_n_e4, g_n_e5);
    appf(out, cap, used, "  钉子8 影子：记录标签 0 补 {2, ControllerMgr 工厂} %d 次；"
                         "状态 1⇒2 补推 %d 次；同轮探针 工厂 %d / 池注册 %d / G %d\n",
         g_n_e2fix, g_n_e8fix, g_n_e6, g_n_e7, g_n_e8);
    appf(out, cap, used, "  钉子8 runner 链：激活检查 %d 次 / 运行转发器 %d 次 / "
                         "runner 入口 %d 次 / 销毁 %d 次（明细见日志）\n",
         g_n_e9, g_n_e10, g_n_e11, g_n_e12);
    appf(out, cap, used, "  PostCalc 崩点仪器：入口命中 %d 次（见 dm_run_test.log 的 [桩] PostCalc 行；"
                         "崩前最后一条的 x30 即调用方返回址，离线 x30-4 反汇定位调用者）\n", g_n_pcalc);
    {
        int n = 0;
        for (int i = 0; i < DM_THREAD_MAX; i++) if (g_thr[i].game) n++;
        appf(out, cap, used, "  线程：游戏建了 %d 个（明细见日志 [线程] 行）\n", n);
    }
    appf(out, cap, used, "  %s 启动链跑完并返回（rv=%#llx）；门被命中 %d 次%s\n",
         g_k_faulted ? "✗" : "✓", (unsigned long long)g_k_rv, g_k_hits,
         g_k_faulted ? "（中途崩，崩点见 dm_run_test.log）" : "");
    if (g_k_faulted)
        appf(out, cap, used, "  崩点：pc %#llx sig %d（离线对符号可定位到函数）\n",
             (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
    appf(out, cap, used, "  （明细落盘 %s/dm_run_test.log）\n", files_dir ? files_dir : "?");
    if (g_k_log) { fclose(g_k_log); g_k_log = NULL; }
    dm_log_set_mirror(NULL);
    app(out, cap, used, "\n");
}

static void exp_h(const char *files_dir, char *out, size_t cap, size_t *used) {
    app(out, cap, used, "【实验 H】门回路：93 个真实门块 → dm_gate → gate_dispatch → svc 38 → 返回\n");
    if (files_dir && *files_dir) {
        char pth[512]; snprintf(pth, sizeof pth, "%s/dm_gate_test.log", files_dir);
        g_gate_log = fopen(pth, "w");
        dm_log_set_mirror(g_gate_log);   // 宿主侧全部 LOG 镜像进同一文件（mem_init/FATAL/Break# 都看得见）
    }
    setenv("DM_GUEST_AS_MB", "512", 1);   // 探针进程 VA 有限：门回路不碰堆/别名/栈，512MB 预留够用
    // stderr 重定向到文件：crash.c 的崩溃处理器把完整现场（pc + 29 寄存器）打到 stderr，
    // 而 OHOS 应用把 stderr 丢弃——重定向后现场落盘（filesDir/dm_crash.log），门回路崩一次即得全部线索。
    if (files_dir && *files_dir) {
        char cpth[512]; snprintf(cpth, sizeof cpth, "%s/dm_crash.log", files_dir);
        if (!freopen(cpth, "w", stderr)) appf(out, cap, used, "  ⚠ stderr 重定向失败\n");
    }

    module_t m;
    memset(&m, 0, sizeof m);
    strncpy(m.name, "main", sizeof m.name - 1);
    m.text_end      = DM_IMG_MAIN_TEXT_END;
    m.patch_off     = DM_IMG_MAIN_PATCH_OFF;
    m.ro_off        = DM_IMG_MAIN_RO_OFF;
    m.data_off      = DM_IMG_MAIN_DATA_OFF;
    m.key_page_off  = DM_IMG_MAIN_KEY_PAGE_OFF;
    m.key_page_size = DM_IMG_MAIN_KEY_PAGE_SIZE;
    m.x0_end        = DM_IMG_MAIN_X0_END;
    m.img_end       = DM_IMG_MAIN_END;
    if (dm_ohos_map_image_alloc(&m)) {
        app(out, cap, used, "  ✗ alloc 失败（见 hilog）\n\n");
        return;
    }
    appf(out, cap, used, "  ✓ 装载：base %p，size %zu\n", (void *)m.base, m.size);

    // ① 线程槽：按 loader.c create_keys 的顺序建 8 键（musl 顺序发号），设值
    static const char *kn[8] = {"K18", "KTLS", "KTLR", "KGATE", "KRES0", "KRES1", "KASPECT", "KASPECTUI"};
    pthread_key_t key[8];
    int k_ok = 1;
    for (int i = 0; i < 8; i++)
        if (pthread_key_create(&key[i], NULL)) { appf(out, cap, used, "  ✗ 建键 %s 失败\n", kn[i]); k_ok = 0; break; }
    if (!k_ok) { app(out, cap, used, "\n"); return; }
    // 槽写入用【直写】：与门块的读取公式（tsd 基址 + 键号×8）同一条路，构造上保证一致。
    // （pthread_setspecific 写的位置与门块按公式读的位置不一致——二分定位已证实，2026-10-16 真机。）
    uint64_t tp_el0 = 0;
    __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp_el0));
    uint64_t tsd_base = *(uint64_t *)(tp_el0 - 272);        // musl：TSD 数组基址（-272 真机实测）
    const void *slot_val[8] = {
        (void *)(uintptr_t)0x1111111111111111ull, (void *)(uintptr_t)0x2222222222222222ull,
        (void *)(uintptr_t)0x3333333333333333ull, (void *)dm_gate,
        (void *)(uintptr_t)0x4400000000000004ull, (void *)(uintptr_t)0x4400000000000005ull,
        (void *)(uintptr_t)0x4400000000000006ull, (void *)(uintptr_t)0x4400000000000007ull,
    };
    for (int i = 0; i < 8; i++)
        *(uint64_t *)(tsd_base + (uint64_t)key[i] * 8) = (uint64_t)slot_val[i];

    // ② 键页：写实际键号×8（与 loader.c install_key_page 同款；槽位下标来自 fixups 的 KEY_*_IDX）
    uint64_t *page = (uint64_t *)(m.base + DM_IMG_MAIN_KEY_PAGE_OFF);
    memset(page, 0, DM_IMG_MAIN_KEY_PAGE_SIZE);
    page[DM_IMG_MAIN_KEY_K18_IDX]       = (uint64_t)key[0] * 8;
    page[DM_IMG_MAIN_KEY_KTLS_IDX]      = (uint64_t)key[1] * 8;
    page[DM_IMG_MAIN_KEY_KTLR_IDX]      = (uint64_t)key[2] * 8;
    page[DM_IMG_MAIN_KEY_KGATE_IDX]     = (uint64_t)key[3] * 8;
    page[DM_IMG_MAIN_KEY_KRES0_IDX]     = (uint64_t)key[4] * 8;
    page[DM_IMG_MAIN_KEY_KRES1_IDX]     = (uint64_t)key[5] * 8;
    page[DM_IMG_MAIN_KEY_KASPECT_IDX]   = (uint64_t)key[6] * 8;
    page[DM_IMG_MAIN_KEY_KASPECTUI_IDX] = (uint64_t)key[7] * 8;
    uint64_t kgate_off = page[DM_IMG_MAIN_KEY_KGATE_IDX];
    uint64_t slot_now  = *(uint64_t *)(tsd_base + kgate_off);
    appf(out, cap, used, "  诊断：tsd 基址 %p，键页 KGATE 项 = %llu（×8），KGATE 槽现值 = %p（期望 dm_gate = %p）\n",
         (void *)tsd_base, (unsigned long long)(kgate_off / 8), (void *)slot_now, (void *)dm_gate);
    app(out, cap, used, "  ✓ 线程槽 8 键直写就绪，KGATE = dm_gate；键页已按实际键号写入\n");

    // 信号兜底：登记与过门都套（崩了进程活着，出错 PC 进报告）
    struct sigaction sa, old_segv, old_bus;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = gate_fault_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS, &sa, &old_bus);

    g_gate_armed = 1;
    if (sigsetjmp(g_gate_jmp, 1) != 0) {
        g_gate_armed = 0;
        appf(out, cap, used, "  ❌ 登记阶段崩（pc %#llx，sig %d）—— 见 hilog\n",
             (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
        sigaction(SIGSEGV, &old_segv, NULL); sigaction(SIGBUS, &old_bus, NULL);
        return;
    }
    // ③ 登记 93 门（kind 码 → 字符串；返回点/原指令按模块基址换算成绝对地址）
    for (int i = 0; i < DM_GATES_MAIN_N; i++) {
        const char *ks = DM_GATES_MAIN[i].kind == 0 ? "svc" : DM_GATES_MAIN[i].kind == 1 ? "cntpct" : "cntvct";
        gate_register((uint64_t)(m.base + DM_GATES_MAIN[i].ret), ks, DM_GATES_MAIN[i].arg,
                      (uint64_t)(m.base + DM_GATES_MAIN[i].orig));
    }
    g_gate_armed = 0;
    appf(out, cap, used, "  ✓ 已登记 %d 个门（全部 svc，编号 38）\n", DM_GATES_MAIN_N);

    // ④ 尾跳暂改 RET：每个门块的最后一条是 B 原指令+4（在 返回点+4），改成 RET 让回路止于探针
    for (int i = 0; i < DM_GATES_MAIN_N; i++)
        *(uint32_t *)(m.base + DM_GATES_MAIN[i].ret + 4) = 0xD65F03C0;   // ret

    // ⑤ 封权限：门块要**执行**，代码区与补丁区必须先封成 R+X（实验 D 验证过这条路径）
    //   ——首轮真机就是漏了这一步：跳进 RW 页直接 SIGSEGV，进程死亡（faultlog 有档）。
    if (dm_ohos_map_image_seal(&m)) {
        app(out, cap, used, "  ✗ seal 失败（见 hilog）\n\n");
        return;
    }
    app(out, cap, used, "  ✓ 代码区与补丁区已封 R+X\n");

    // ⑤ 二分 B（线程版，见 gb_thread）：门调度搬专用线程——验证主线程卡死看门狗假设
    {
        glog("二分 B（线程版）开始：将在线程里直调 gate_dispatch(ret=%#llx)\n",
             (unsigned long long)(m.base + DM_GATES_MAIN[0].ret));
        memset(&g_gb, 0, sizeof g_gb);
        g_gb.base = (uint64_t)(m.base + DM_GATES_MAIN[0].ret);
        memcpy(g_gb.key, key, sizeof g_gb.key);
        pthread_t th;
        if (pthread_create(&th, NULL, gb_thread, &g_gb) == 0) {
            pthread_join(th, NULL);
            glog("二分 B（线程版）：线程返回 rv=%#llx faulted=%d\n",
                 (unsigned long long)g_gb.rv, g_gb.faulted);
            appf(out, cap, used, "  %s 二分 B（线程版）：返回 %#llx%s\n",
                 (!g_gb.faulted && g_gb.rv == 0) ? "✓" : "✗",
                 (unsigned long long)g_gb.rv, g_gb.faulted ? "（线程内崩，兜底接住）" : "");
        } else {
            app(out, cap, used, "  ✗ 二分 B：pthread_create 失败\n");
        }
    }

    // ⑤ 二分 C：直调 svc_dispatch(38, regs2, orig)（跳过哈希查找）——分离「查找」与「处理器」
    {
        uint64_t regs3[64]; memset(regs3, 0, sizeof regs3);
        regs3[0] = 0x80000000ull;
        g_gate_armed = 1; g_gate_fault_pc = 0;
        if (sigsetjmp(g_gate_jmp, 1) == 0) {
            uint64_t rv3 = svc_dispatch(38, regs3, (uint64_t)(m.base + DM_GATES_MAIN[0].orig));
            g_gate_armed = 0;
            glog("二分 C：svc_dispatch(38) 返回 %#llx\n", (unsigned long long)rv3);
            appf(out, cap, used, "  %s 二分 C：svc_dispatch(38) 直调返回 %#llx\n",
                 rv3 == 0 ? "✓" : "✗", (unsigned long long)rv3);
        } else {
            g_gate_armed = 0;
            glog("二分 C：崩 pc %#llx sig %d\n", (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
            appf(out, cap, used, "  ✗ 二分 C：svc_dispatch 直调崩（pc %#llx，sig %d）\n",
                 (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
        }
    }

    // ⑤′ 二分定位（必须在 seal 之后：门块要可执行）：门 0 的块先以「KGATE = 探针函数」过一次
    //    —— 验证「门块执行 + 槽读取」；通了再把 KGATE 指回 dm_gate，崩了就锁定在宿主侧入口。
    {
        uint32_t *p0 = (uint32_t *)(m.base + DM_GATES_MAIN[0].ret);
        void *blk0 = NULL;
        for (int b = 1; b <= 16; b++)
            if (p0[-b] == 0xA9BF7BFD) { blk0 = (uint8_t *)p0 - 4 * b; break; }
        if (blk0) {
            // 尾跳已在步骤④（seal 前）改为 RET——此处**不可再写**（seal 后补丁区只读，写了必 SIGSEGV）
            *(uint64_t *)(tsd_base + (uint64_t)key[3] * 8) = (uint64_t)gate_probe_entry;
            g_gate_armed = 1; g_gate_fault_pc = 0;
            glog("二分 A：门 0 开始（KGATE=探针函数，块起点 %p）\n", blk0);
            if (sigsetjmp(g_gate_jmp, 1) == 0) {
                ((void (*)(void))blk0)();
                g_gate_armed = 0;
                glog("二分 A：过门成功\n");
                app(out, cap, used, "  ✓ 二分 A：门 0 以探针入口过门成功（门块执行 + KGATE 槽读取成立）\n");
            } else {
                g_gate_armed = 0;
                glog("二分 A：崩 pc %#llx sig %d\n", (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
                appf(out, cap, used, "  ✗ 二分 A：门 0 以探针入口仍崩（pc %#llx）⇒ 问题在门块/槽读取本身\n",
                     (unsigned long long)g_gate_fault_pc);
            }
            *(uint64_t *)(tsd_base + (uint64_t)key[3] * 8) = (uint64_t)dm_gate;   // 恢复
        }
    }

    // ⑥ 逐门过回路：门块地址 = 从返回点向后扫 stp x29,x30,[sp,#-16]!（0xA9BF7BFD）
    int ok = 0, noblock = 0, faulted = 0;
    for (int i = 0; i < DM_GATES_MAIN_N; i++) {
        uint32_t *p = (uint32_t *)(m.base + DM_GATES_MAIN[i].ret);
        void *blk = NULL;
        for (int b = 1; b <= 16; b++)
            if (p[-b] == 0xA9BF7BFD) { blk = (uint8_t *)p - 4 * b; break; }
        if (!blk) { noblock++; continue; }
        g_gate_armed = 1; g_gate_fault_pc = 0;
        glog(">>> 门 %d（ret %#llx）开始\n", i, (unsigned long long)DM_GATES_MAIN[i].ret);
        if (sigsetjmp(g_gate_jmp, 1) == 0) {
            // 过门参数：svc 38 = Break(reason, ptr, size)。reason 的最高位 = 「仅通知，可继续」——
            // 处理器看到它就返回 0 而不 _exit(42)；ptr/size 置 0 免得处理器解引用垃圾指针。
            register uint64_t g_x0 __asm__("x0") = 0x80000000ull;
            register uint64_t g_x1 __asm__("x1") = 0;
            register uint64_t g_x2 __asm__("x2") = 0;
            __asm__ volatile("blr %3"
                             : "+r"(g_x0), "+r"(g_x1), "+r"(g_x2)
                             : "r"(blk)
                             : "x3","x4","x5","x6","x7","x8","x9","x10","x11","x12",
                               "x13","x14","x15","x16","x17","x30","memory");
            uint64_t rv = g_x0;   // dm_gate 回写 x0 = 处理器返回值
            g_gate_armed = 0;
            glog("门 %d 返回 %#llx\n", i, (unsigned long long)rv);
            // Break「仅通知」路径不改 x0（帧里的 x0 原样恢复）⇒ 期望值 = 我们喂的仅通知位
            if (rv == 0 || rv == 0x80000000ull) ok++;
            else appf(out, cap, used, "  ✗ 门 %d（ret %#llx）返回 %#llx（期望 0）\n",
                      i, (unsigned long long)DM_GATES_MAIN[i].ret, (unsigned long long)rv);
        } else {
            g_gate_armed = 0; faulted++;
            glog("门 %d 崩：pc %#llx sig %d\n", i, (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
            appf(out, cap, used, "  ✗ 门 %d（ret %#llx）崩：pc %#llx，sig %d\n",
                 i, (unsigned long long)DM_GATES_MAIN[i].ret,
                 (unsigned long long)g_gate_fault_pc, g_gate_fault_sig);
        }
    }
    sigaction(SIGSEGV, &old_segv, NULL); sigaction(SIGBUS, &old_bus, NULL);
    glog("=== 结束：ok=%d faulted=%d noblock=%d ===\n", ok, faulted, noblock);
    if (g_gate_log) { fclose(g_gate_log); g_gate_log = NULL; }
    dm_log_set_mirror(NULL);
    appf(out, cap, used, "  （逐门明细已落盘 %s/dm_gate_test.log）\n", files_dir ? files_dir : "?");
    appf(out, cap, used, "  %s 判据：%d / %d 个门块走完整回路且返回 0（Break 以仅通知位过门）；崩 %d（PC 见上）；块起点未找到 %d\n",
         (ok == DM_GATES_MAIN_N - noblock - faulted && faulted == 0) ? "✅" : "❌",
         ok, DM_GATES_MAIN_N, faulted, noblock);
    app(out, cap, used, "  说明：每个门块的尾跳已暂改 RET（不执行门后的游戏代码）；首次过门时 hilog 有"
                        " \"SVC 0x26 …（占位返回成功）\"\n\n");
}

int dm_probe_run(const char *files_dir, char *out, size_t cap) {
    if (!out || cap < 64) return -1;
    size_t used = 0;
    out[0] = 0;

    app(out, cap, &used, "D2R Mobile · 鸿蒙移植 · 第 1 站实验\n");
    app(out, cap, &used, "==================================================\n\n");
    if (files_dir && *files_dir) {
        appf(out, cap, &used, "应用数据目录：%s\n\n", files_dir);
    }

    // 这几条只进 hilog、不进报告：万一某一步把进程搞挂了，
    // 从日志里就能看出死在哪一步（有"进入"没"返回"）。
    // 顺序有讲究：A 的最后一步要真执行一段可能不可执行的内存，
    // 万一信号兜底没兜住、进程挂了，B/C 也已经跑完并写进日志了。
    // 2026-10-16 收窄：只跑「执行链探针」这一组（G 装载 / H 门回路 / I 真实执行 / J 门+执行 / K 启动链）。
    // D/E/E2/F 与 A/B/C 是早期探路实验（装载机制/图形链），已收官——函数留在代码里备查，但不再每轮跑，
    // 每轮省掉 mmap 试探与大量报告，迭代更快。
    // 2026-10-16 再收窄：G/H/I/J 均已验收（93/93、3/3、串链全绿），每轮不再重复跑，
    // 只跑当前焦点实验 K（启动链→nnMain→崩点）——每轮省约 40 秒；要回归时把它们放开即可。
    logline(">>> 实验 G/H/I/J 已验收，本轮跳过（要回归时放开）");
    (void)exp_g; (void)exp_h; (void)exp_i; (void)exp_j;
    logline(">>> 实验 K 开始"); exp_k(files_dir, out, cap, &used); logline(">>> 实验 K 返回");

    app(out, cap, &used, "==================================================\n");
    app(out, cap, &used, "把这份报告发回来，就能定下装载方案。\n");
    logline(">>> 执行链探针全部跑完（G/H/I/J/K）");
    return 0;
}
