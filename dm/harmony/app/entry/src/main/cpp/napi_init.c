// napi_init.c —— S2-a2 的最小 N-API 桥（np230）
//
// 只做一件事：**把 ArkUI 交给 XComponent 的 OHNativeWindow 接住，交给两层消费者**
//   ① 前端：dm_front_ohos_set_window()  → front_metal_layer() 的返回值
//   ② 上屏：dm_android_vk_set_window()  → vulkan_present.cpp 的交换链目标
// 并提供 probe() 让 ArkTS 把状态显示在页面上（真机一眼看出接没接上）。
//
// 刻意不做（留给后续）：
//   · 不起 guest（S2-b 才接宿主装载）
//   · 不调 dm_android_vk_initialize（它要求 .xgs 着色器目录；S3 才有）
//   · 不接触摸（BotW 用手柄，S2-c 走 dm_pad_set）
#include <napi/native_api.h>
#include <ace/xcomponent/native_interface_xcomponent.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <hilog/log.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "android/native_window.h"   // 上游 android_shim：ANativeWindow → OHNativeWindow
#include "libdeflate.h"   // np346：writeFileGz 解 gzip（libdeflate 已在 include 路径，hle_tact.c 同款）

// np539：dm/host/crash.c 里的"把崩溃现场直写进这个 fd"入口。crash.c 是 C，这里按 C 链接声明。
// np539: entry point in dm/host/crash.c that points the crash dump at this fd; crash.c is C, so declare it as C.
#ifdef __cplusplus
extern "C"
#endif
void crash_set_fd(int fd);

// 两层消费者的入口（见 dm/harmony/native/front_ohos_min.c 与 dm/android/vulkan_present.cpp）
void dm_front_ohos_set_window(void *window);
void dm_android_vk_set_window(ANativeWindow *window);
int dm_android_vk_has_window(void);
int front_screen_max_fps(void);

// 手柄：宿主的输入入口（dm/host/dm_input.h）。
// 键是位图（DM_BTN_*，与 Switch Pro 手柄一一对应），摇杆是 int32。
//   A=1<<0 B=1<<1 X=1<<2 Y=1<<3 LS=1<<4 RS=1<<5 L=1<<6 R=1<<7
//   ZL=1<<8 ZR=1<<9 PLUS=1<<10 MINUS=1<<11 LEFT=1<<12 UP=1<<13 RIGHT=1<<14 DOWN=1<<15
void dm_pad_set(uint64_t buttons, int32_t lx, int32_t ly, int32_t rx, int32_t ry);

// ── 起 guest（np237，照抄上游 napi_ohos.c 的 host_thread 形状）──
// 宿主的 main.c 在 App 库里以 -Dmain=dm_main 编（同上游安卓/iOS 的做法），
// 命令行 dmhost 才用原名 main。所以这里声明的是 dm_main。
int dm_main(int argc, char **argv);
int dm_android_vk_initialize(const char *shader_directory, char *error, size_t bytes);
// 公共读表接口（dm/host/dm_derived_rt.h）：鸿蒙由平台层设好 DM_DERIVED 后显式调；弱引用
__attribute__((weak)) int dm_derived_reload(void);

// ── 宿主 stderr → hilog 桥（np235）──
// 宿主的 LOG 全走 fprintf(stderr,...)。应用的 stderr 默认哪儿都不去 ⇒ 建一根管道，
// 把 fd 2 接到管道的写端，再起一个线程逐行 OH_LOG_Print。
// 这样**所有宿主输出都进 hilog**，hdc 随时能读（不用把 app 拉到前台、也不用读文件）。
#ifndef DM_LOG_DOMAIN
#define DM_LOG_DOMAIN 0x4D48   // "MH"，本工程自用域
#endif
static int g_pipe_wr = -1, g_pipe_rd = -1;
static void *hilog_pump(void *arg) {
    (void)arg;
    char buf[2048];
    size_t used = 0;
    for (;;) {
        ssize_t n = read(g_pipe_rd, buf + used, sizeof buf - 1 - used);
        if (n <= 0) break;
        used += (size_t)n;
        buf[used] = 0;
        char *line = buf, *nl;
        while ((nl = strchr(line, '\n'))) {
            *nl = 0;
            if (*line) OH_LOG_Print(LOG_APP, LOG_INFO, DM_LOG_DOMAIN, "dm-host", "%{public}s", line);   // %{public}：否则隐私策略会把内容打成 <private>
            line = nl + 1;
        }
        used = strlen(line);
        if (used) memmove(buf, line, used);
    }
    return NULL;
}
// 在宿主开跑前调用：返回 0 = 桥接成功
static int stderr_to_hilog(void) {
    int fds[2];
    if (pipe(fds)) return -1;
    g_pipe_rd = fds[0];
    g_pipe_wr = fds[1];
    if (dup2(g_pipe_wr, STDERR_FILENO) < 0) return -1;
    pthread_t th;
    if (pthread_create(&th, NULL, hilog_pump, NULL)) return -1;
    pthread_detach(th);
    return 0;
}

static char g_runtime[1200];   // 应用可写目录（start 的第一个参数；out/ 与 shaders/ 都在它下面）
static char g_log_path[1300];  // 宿主 stderr 重定向目标（start 的第二个参数；空 = 不重定向）
static char g_marker_path[1300];  // 诊断用：ArkTS 写的探针文件，原生侧试读它（start 的第三个参数）

static void *host_thread(void *arg) {
    (void)arg;
    // 宿主的 LOG 全走 stderr，而应用的 stderr 不进 hilog ⇒ 先重定向到文件（崩了也留得下现场）。
    // 用 "a" 追加：每次起 guest 的历史都留着，方便对比。
    // 二选一：hilog 桥优先（后台随时可读）；桥接失败才退回文件。
    // （两者都开会互相覆盖 fd 2——freopen 会把管道写端换掉。）
    if (stderr_to_hilog() == 0) {
        fprintf(stdout, "boot: 宿主 stderr 已桥接到 hilog（域 0x%x，标签 dm-host）\n", DM_LOG_DOMAIN);
    } else if (g_log_path[0]) {
        FILE *lf = freopen(g_log_path, "a", stderr);
        if (lf) setvbuf(lf, NULL, _IOLBF, 0);   // 行缓冲：die() 前那行一定落得下磁盘
        else fprintf(stdout, "boot: stderr 重定向失败：%s\n", g_log_path);
    }
    // 诊断：原生眼里的 runtime 目录长什么样（与 ArkTS 的 fs.listFileSync 对照）
    fprintf(stderr, "boot: 原生列目录 %s\n", g_runtime);
    DIR *d = opendir(g_runtime);
    if (!d) {
        fprintf(stderr, "boot: opendir 失败：%s\n", strerror(errno));
    } else {
        struct dirent *e;
        int n = 0;
        while ((e = readdir(d)) && n++ < 12) {
            if (strcmp(e->d_name, ".") && strcmp(e->d_name, ".."))
                fprintf(stderr, "boot:   %s\n", e->d_name);
        }
        closedir(d);
    }
    // 诊断：ArkTS 刚写的探针文件，原生能不能读到
    if (g_marker_path[0]) {
        FILE *m = fopen(g_marker_path, "rb");
        if (m) {
            char buf[64] = { 0 };
            size_t got = fread(buf, 1, sizeof buf - 1, m);
            fprintf(stderr, "boot: 探针文件读到 %zu 字节：%s\n", got, buf);
            fclose(m);
        } else {
            fprintf(stderr, "boot: 探针文件打不开：%s（%s）\n", g_marker_path, strerror(errno));
        }
    }
    char out[1200], shaders[1200], error[512] = { 0 };
    snprintf(out, sizeof out, "%s/out", g_runtime);          // 改写产物目录（main.patched.bin / *.tsv）
    snprintf(shaders, sizeof shaders, "%s/shaders", g_runtime);   // .xgs 目录
    // 先 Vulkan（失败就地报告，不进宿主）——照上游 napi_ohos.c:177 的次序
    int graphics_rc = dm_android_vk_initialize(shaders, error, sizeof error);
    if (graphics_rc) {
        fprintf(stderr, "boot: Vulkan 初始化失败 %d: %s\n", graphics_rc, error);
        return NULL;
    }
    // 宿主入口：argv[1]=产物目录，其后是模块名。
    // ⚠ 必须装**全部**模块（np241 离线实证）：BotW 把系统库拆成多个 subsdk——
    //   nn::vi::* 在 subsdk1、nv::InitializeGraphics / SetGraphicsAllocator 在 subsdk0、
    //   movie::Decoder 在 subsdk3。只装 main+sdk 时这 14 个强引用导入会落到"未实现桩"
    //   （调用恒返回 0），客体把它们当指针用即崩。装全后这 14 条缺口全部消失
    //   （剩下的只有弱引用 glslc* 与 rtld 提供的 __rel_* 链接符号，返回 0 是预期）。
    //   rtld 不装：宿主自己扮演 rtld（InitModules）。
    char *argv[] = { "dmhost", out, "main", "sdk", "subsdk0", "subsdk1", "subsdk2", "subsdk3", NULL };
    fprintf(stderr, "boot: dm_main %s（main+sdk+subsdk0-3）\n", out);
    int rc = dm_main(8, argv);
    fprintf(stderr, "boot: dm_main 返回 %d\n", rc);
    return NULL;
}

static void *g_window;      // 当前 OHNativeWindow（NULL = 还没有表面）
static unsigned g_w, g_h;

static void take_window(OH_NativeXComponent *c, void *window) {
    uint64_t w = 0, h = 0;
    if (window && OH_NativeXComponent_GetXComponentSize(c, window, &w, &h) != OH_NATIVEXCOMPONENT_RESULT_SUCCESS)
        w = h = 0;
    g_window = window; g_w = (unsigned)w; g_h = (unsigned)h;
    dm_front_ohos_set_window(window);                    // 前端侧（front_metal_layer 返回它）
    dm_android_vk_set_window((ANativeWindow *)window);   // 上屏侧（保留引用；NULL = 暂停呈现）
}

static void OnSurfaceCreatedCB(OH_NativeXComponent *c, void *window) { take_window(c, window); }
static void OnSurfaceChangedCB(OH_NativeXComponent *c, void *window) { take_window(c, window); }
static void OnSurfaceDestroyedCB(OH_NativeXComponent *c, void *window) {
    (void)c; (void)window;
    g_window = NULL; g_w = g_h = 0;
    dm_front_ohos_set_window(NULL);
    dm_android_vk_set_window(NULL);
}
static void DispatchTouchEventCB(OH_NativeXComponent *c, void *window) { (void)c; (void)window; }

// probe()：把"表面接上没"的判据一次性交给 ArkTS 显示；若起过 guest，顺带回显宿主日志尾部
// （宿主的 LOG 全走 stderr，而应用的 stderr 不进 hilog ⇒ 起 guest 前重定向到文件，崩了也好复盘）
static napi_value Probe(napi_env env, napi_callback_info info) {
    (void)info;
    char buf[2048];
    int off = snprintf(buf, sizeof buf, "window=%p size=%ux%u vk_has_window=%d screen_max_fps=%d",
                       g_window, g_w, g_h, dm_android_vk_has_window(), front_screen_max_fps());
    if (g_log_path[0]) {
        FILE *f = fopen(g_log_path, "rb");
        if (f) {
            // 只回显尾部 700 字节（屏暮放得下），行边界对齐交给阅读方
            static char tail[4000];
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            long skip = n > (long)sizeof tail - 1 ? n - ((long)sizeof tail - 1) : 0;
            fseek(f, skip, SEEK_SET);
            size_t got = fread(tail, 1, sizeof tail - 1, f);
            tail[got] = 0;
            fclose(f);
            off += snprintf(buf + off, sizeof buf - off, "\n--- host.log 尾 ---\n%s", tail);
        } else {
            off += snprintf(buf + off, sizeof buf - off, "\n--- host.log 打不开：%s ---", g_log_path);
        }
    }
    napi_value out = NULL;
    napi_create_string_utf8(env, buf, NAPI_AUTO_LENGTH, &out);
    return out;
}

// writeFile(path, data)：用原生 IO 把数据落盘，返回写入字节数。
// 为什么不用 ArkTS 的 fs.writeSync：实测 ArkTS 写进 filesDir 的文件**原生侧看不到**
// （探针文件能读是因为它写在 filesDir 根；子目录里的就没了）⇒ 宿主要读的素材必须由原生写。
// mkdir -p：父目录不存在时一级级建出来（fopen 不建目录）
static void mkdirs(const char *path) {
    char tmp[1300];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    // np345f 修复：上面循环只建"前缀"，**最后一个目录组件从来没建过**——
    // mkdirs("<files>/wtest") 只建了 <files>，wtest 本身不存在 ⇒ fopen 全线 ENOENT，
    // 而旧 WriteFile 把 0 字节当成功 ⇒ "romfs 解出 8190 文件"全是假账，完整 romfs 从未落地。
    mkdir(tmp, 0755);
}

static napi_value WriteFile(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 2) {
        napi_throw_error(env, NULL, "writeFile 需要 2 个参数：path, ArrayBuffer");
        return NULL;
    }
    char path[1300];
    size_t n = 0;
    napi_get_value_string_utf8(env, argv[0], path, sizeof path, &n);
    // 兼容两种入参：ArrayBuffer 直接取；TypedArray（getRawFileContentSync 实际返回的就是它）
    // 要连 byteOffset 一起算，否则会从缓冲头部读错位置。
    void *data = NULL;
    size_t len = 0;
    bool is_ab = false, is_ta = false;
    napi_is_arraybuffer(env, argv[1], &is_ab);
    napi_is_typedarray(env, argv[1], &is_ta);
    if (is_ab) {
        napi_get_arraybuffer_info(env, argv[1], &data, &len);
    } else if (is_ta) {
        napi_typedarray_type type;
        size_t esz = 0, length = 0;
        void *raw = NULL;
        napi_value ab = NULL;
        size_t off = 0;
        if (napi_get_typedarray_info(env, argv[1], &type, &length, &raw, &ab, &off) != napi_ok) {
            napi_throw_type_error(env, NULL, "TypedArray 信息取不到");
            return NULL;
        }
        napi_get_arraybuffer_info(env, ab, &data, &esz);
        data = (char *)data + off;
        len = length;
        (void)esz;
    } else {
        napi_throw_type_error(env, NULL, "第二个参数要 ArrayBuffer 或 TypedArray");
        return NULL;
    }
    size_t wrote = 0;
    // 先把父目录建出来（fopen 不建目录；ArkTS 的 mkdir 写出来的目录原生看不到）
    char dir[1300];
    snprintf(dir, sizeof dir, "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash && slash != dir) { *slash = 0; mkdirs(dir); }
    FILE *f = fopen(path, "wb");
    if (f) {
        wrote = fwrite(data, 1, len, f);
        fclose(f);
    } else {
        // np345：fopen 失败必须如实报负数。原先返回 wrote=0，extractRaw 把 >=0 都算成功，
        // "8190 个文件解出"全是假象——新建目录的写入一直静默失败，完整 romfs 从未落地
        // （真机上只有 10-24 旧引导子集的 3 个目录里那几个文件，guest 一直跑在 stub 上）。
        // 返回 -errno，ArkTS 侧打前 10 条失败路径时直接带出原因。
        napi_value v = NULL;
        napi_create_int32(env, -errno, &v);
        return v;
    }
    napi_value v = NULL;
    napi_create_uint32(env, (uint32_t)wrote, &v);
    return v;
}

// readLog(path)：读宿主日志尾部（起 guest 崩了之后，重启 app 时由 ArkTS 调它回看）。
// 单独一个接口而不是靠 probe()：probe() 只在**本进程**调过 start() 时才知道日志路径，
// 而崩溃后是新进程，路径得由 ArkTS 重新给。
static napi_value ReadLog(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1) {
        napi_throw_error(env, NULL, "readLog 需要 1 个参数：日志路径");
        return NULL;
    }
    char path[1300];
    size_t n = 0;
    napi_get_value_string_utf8(env, argv[0], path, sizeof path, &n);
    // 第二个参数可选：'head' 取开头（启动经过）、'tail' 取结尾（默认）、'all' 取全文（上限 24KB）
    // ⚠ argv 数组必须按"最多可能读到的参数个数"分配：N-API 的 argc 是 in-out，调用方传几个
    //   它就变成几，只按 1 分配却去读 argv[1] 会读到未初始化栈内存 → SIGSEGV（本轮实测）。
    int mode = 0;   // 0=tail 1=head 2=all
    if (argc >= 2) {
        char m[16] = { 0 };
        size_t mn = 0;
        napi_get_value_string_utf8(env, argv[1], m, sizeof m, &mn);
        if (!strcmp(m, "head")) mode = 1;
        else if (!strcmp(m, "all")) mode = 2;
    }
    static char buf[24001];
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(buf, sizeof buf, "（没有日志文件：%s）", path); }
    else if (mode == 2) {
        size_t got = fread(buf, 1, sizeof buf - 1, f);
        buf[got] = 0;
        fclose(f);
    } else {
        static char tail[4000];
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        long skip = sz > (long)sizeof tail - 1 ? sz - ((long)sizeof tail - 1) : 0;
        fseek(f, skip, SEEK_SET);
        size_t got = fread(tail, 1, sizeof tail - 1, f);
        tail[got] = 0;
        fclose(f);
        if (mode == 1) {
            // 开头：从文件头读 4000 字节
            FILE *g = fopen(path, "rb");
            size_t g2 = g ? fread(buf, 1, 4000, g) : 0;
            if (g) fclose(g);
            buf[g2] = 0;
        } else {
            snprintf(buf, sizeof buf, "%s", tail);
        }
    }
    napi_value v = NULL;
    napi_create_string_utf8(env, buf, NAPI_AUTO_LENGTH, &v);
    return v;
}

// logLine(s)：ArkTS 侧的逐步跟踪直接写 hilog（标签 dm-host）。
// 为什么不用文件：起 guest 崩了之后文件在应用沙箱里，hdc 读不到；hilog 后台随时可读。
static napi_value LogLine(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1) {
        napi_value und = NULL; napi_get_undefined(env, &und); return und;
    }
    char line[1024] = { 0 };
    size_t n = 0;
    napi_get_value_string_utf8(env, argv[0], line, sizeof line, &n);
    OH_LOG_Print(LOG_APP, LOG_INFO, DM_LOG_DOMAIN, "dm-host", "ark %{public}s", line);
    napi_value und = NULL;
    napi_get_undefined(env, &und);
    return und;
}

// listDir(path)：列目录内容（诊断用：解出到底成功没有、shaders 里有什么，一眼可见）
static napi_value ListDir(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1) {
        napi_throw_error(env, NULL, "listDir 需要 1 个参数：目录路径");
        return NULL;
    }
    char path[1300];
    size_t n = 0;
    napi_get_value_string_utf8(env, argv[0], path, sizeof path, &n);
    char out[1024];
    int off = snprintf(out, sizeof out, "%s：", path);
    DIR *d = opendir(path);
    if (!d) {
        off += snprintf(out + off, sizeof out - off, "打不开（%s）", strerror(errno));
    } else {
        int cnt = 0;
        struct dirent *e;
        while ((e = readdir(d)) && off < (int)sizeof out - 40) {
            if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
            off += snprintf(out + off, sizeof out - off, " %s", e->d_name);
            cnt++;
            if (cnt >= 20) { off += snprintf(out + off, sizeof out - off, " …"); break; }
        }
        if (!cnt) off += snprintf(out + off, sizeof out - off, "（空）");
        closedir(d);
    }
    napi_value v = NULL;
    napi_create_string_utf8(env, out, NAPI_AUTO_LENGTH, &v);
    return v;
}

// dirFileCount(path)：递归统计目录里的文件数；目录不存在/打不开返回 -1。
// 用途（d1 瘦包）：HAP 里不带 romfs 时，用**原生侧**判断沙箱里上一版 fat 包解出的 romfs
// 还在不在——ArkTS 的 fs.listFileSync/accessSync 对不存在的路径不抛异常（np220/np279 教训），
// 存在性判断必须用原生 stat/readdir。
static long long DirCountRec(const char *path) {
    DIR *d = opendir(path);
    if (!d) return -1;
    long long total = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char sub[1300];
        snprintf(sub, sizeof sub, "%s/%s", path, e->d_name);
        if (e->d_type == DT_DIR) {
            long long r = DirCountRec(sub);
            if (r > 0) total += r;
        } else {
            total++;
        }
    }
    closedir(d);
    return total;
}

static napi_value DirFileCount(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1) {
        napi_throw_error(env, NULL, "dirFileCount 需要 1 个参数：目录路径");
        return NULL;
    }
    char path[1300];
    size_t n = 0;
    napi_get_value_string_utf8(env, argv[0], path, sizeof path, &n);
    long long cnt = DirCountRec(path);
    napi_value v = NULL;
    napi_create_int64(env, cnt, &v);
    return v;
}

// writeFileGz(path, gzData)：解 gzip 后落盘（np346 着色器投递）。
// rawfile 里放 .xgs.gz（全量 728MB→约 215MB，装机时间从不可行变成一分钟级）；
// 原生侧用 libdeflate（已链接，hle_tact.c 同款）解出。成功返回写出的字节数，失败 -errno。
// gzip 尾部 ISIZE = 原始大小（我们的文件 <4GB，python gzip 写的是真值）。
static napi_value WriteFileGz(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 2) {
        napi_throw_error(env, NULL, "writeFileGz 需要 2 个参数：path, gzArrayBuffer");
        return NULL;
    }
    char path[1300];
    size_t n = 0;
    napi_get_value_string_utf8(env, argv[0], path, sizeof path, &n);
    void *gz = NULL; size_t gz_len = 0;
    bool is_ab = false;
    napi_is_arraybuffer(env, argv[1], &is_ab);
    if (!is_ab || napi_get_arraybuffer_info(env, argv[1], &gz, &gz_len) != napi_ok || gz_len < 20) {
        napi_throw_error(env, NULL, "writeFileGz 第二个参数要 ArrayBuffer");
        return NULL;
    }
    if (gz_len < 4) { napi_value v = NULL; napi_create_int32(env, -EINVAL, &v); return v; }
    uint32_t isize = 0;
    memcpy(&isize, (const char *)gz + gz_len - 4, 4);          // gzip ISIZE（小端）
    if (!isize || isize > (1u << 30)) { napi_value v = NULL; napi_create_int32(env, -EINVAL, &v); return v; }
    uint8_t *out = (uint8_t *)malloc(isize);
    if (!out) { napi_value v = NULL; napi_create_int32(env, -ENOMEM, &v); return v; }
    struct libdeflate_decompressor *d = libdeflate_alloc_decompressor();
    size_t actual = 0;
    enum libdeflate_result r = d ? libdeflate_gzip_decompress(d, gz, gz_len, out, isize, &actual) : LIBDEFLATE_BAD_DATA;
    if (d) libdeflate_free_decompressor(d);
    if (r != LIBDEFLATE_SUCCESS || actual != isize) {
        free(out);
        OH_LOG_Print(LOG_APP, LOG_ERROR, DM_LOG_DOMAIN, "dm-host",
                     "writeFileGz 解压失败 %{public}s r=%{public}d actual=%{public}zu isize=%{public}u",
                     path, (int)r, actual, isize);
        napi_value v = NULL;
        napi_create_int32(env, -EIO, &v);
        return v;
    }
    size_t wrote = 0;
    char dir[1300];
    snprintf(dir, sizeof dir, "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash && slash != dir) { *slash = 0; mkdirs(dir); }
    FILE *f = fopen(path, "wb");
    if (f) {
        wrote = fwrite(out, 1, actual, f);
        fclose(f);
    } else {
        OH_LOG_Print(LOG_APP, LOG_ERROR, DM_LOG_DOMAIN, "dm-host",
                     "writeFileGz fopen 失败 %{public}s errno=%{public}d", path, errno);
        free(out);
        napi_value v = NULL;
        napi_create_int32(env, -errno, &v);
        return v;
    }
    free(out);
    napi_value v = NULL;
    napi_create_uint32(env, (uint32_t)wrote, &v);
    return v;
}

// makeDir(path)：直接调 mkdir(path,0755) 并如实回报——0 成功、1 已存在、-errno 失败。
// np345d 诊断：沙箱里 writeFile 全线 ENOENT，mkdirs 静默忽略失败 ⇒ 用它直接拿 mkdir 的 errno。
static napi_value MakeDir(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1) {
        napi_throw_error(env, NULL, "makeDir 需要 1 个参数：目录路径");
        return NULL;
    }
    char path[1300];
    size_t n = 0;
    napi_get_value_string_utf8(env, argv[0], path, sizeof path, &n);
    int rc = mkdir(path, 0755);
    int out;
    if (rc == 0) out = 0;
    else if (errno == EEXIST) out = 1;
    else out = -errno;
    napi_value v = NULL;
    napi_create_int32(env, out, &v);
    return v;
}

// setPad(buttons, lx, ly, rx, ry)：ArkTS 把手柄状态推进宿主。
// 摇杆量程取 ±32767（int16 满程）——宿主侧具体读法要真机核（见文档待办）。
static napi_value SetPad(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value argv[5];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 5) {
        napi_throw_error(env, NULL, "setPad 需要 5 个参数：buttons, lx, ly, rx, ry");
        return NULL;
    }
    int64_t b = 0;
    int32_t v[4] = { 0, 0, 0, 0 };
    if (napi_get_value_int64(env, argv[0], &b) != napi_ok) { napi_throw_type_error(env, NULL, "buttons 要是整数"); return NULL; }
    for (int i = 0; i < 4; i++) {
        int64_t t = 0;
        if (napi_get_value_int64(env, argv[i + 1], &t) != napi_ok) { napi_throw_type_error(env, NULL, "轴值要是整数"); return NULL; }
        v[i] = (int32_t)t;
    }
    dm_pad_set((uint64_t)b, v[0], v[1], v[2], v[3]);
    napi_value und = NULL;
    napi_get_undefined(env, &und);
    return und;
}

// start(runtimeDir, logPath)：备好目录约定并起宿主线程跑 dm_main。
//   runtimeDir：<dir>/out/ 改写产物、<dir>/shaders/ 的 .xgs、<dir>/derived.tsv 平台层读表
//   logPath   ：宿主 stderr 的重定向目标（建议传应用 filesDir 下的 host.log——应用自己可写、
//               崩了重启 app 也能从 probe() 回看到）
static napi_value Start(napi_env env, napi_callback_info info) {
    static int started = 0;
    if (started) { napi_value und = NULL; napi_get_undefined(env, &und); return und; }
    size_t argc = 4;
    napi_value argv[4];
    if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1) {
        napi_throw_error(env, NULL, "start 需要参数：runtimeDir[, logPath[, markerPath[, envs]]]");
        return NULL;
    }
    size_t n = 0;
    napi_get_value_string_utf8(env, argv[0], g_runtime, sizeof g_runtime, &n);
    g_log_path[0] = 0;
    g_marker_path[0] = 0;
    if (argc >= 2) napi_get_value_string_utf8(env, argv[1], g_log_path, sizeof g_log_path, &n);
    if (argc >= 3) napi_get_value_string_utf8(env, argv[2], g_marker_path, sizeof g_marker_path, &n);
    // 第四个参数：环境变量串（空格分隔的 KEY=VALUE），在起宿主持前 setenv。
    // 用途：开 DM_IMPORTS / DM_SVC / DM_TICKS 等调试开关，不用为了改一个开关重编前端。
    if (argc >= 4) {
        char envs[1024] = { 0 };
        napi_get_value_string_utf8(env, argv[3], envs, sizeof envs, &n);
        // np450：把**原始** envs 串打一行，专治"某个 KEY 没进 environ 却查不出为什么"
        // （现象：DM_PROBE 缺失，而它前后的 KEY 都在 —— 只能先看串本身传过来是什么）。
        OH_LOG_Print(LOG_APP, LOG_INFO, DM_LOG_DOMAIN, "dm-host", "start: envs 原文=[%{public}s]", envs);
        char *p = envs;
        while (p && *p) {
            while (*p == ' ') p++;
            if (!*p) break;
            char *eq = strchr(p, '=');
            char *end = strchr(p, ' ');
            if (!eq || (end && eq > end)) { if (!end) break; p = end; continue; }
            if (end) *end = 0;
            *eq = 0;                    // ⚠ 必须在 '=' 处也截断：否则 setenv 会把 "KEY=VALUE" 整个当变量名
            int rc = setenv(p, eq + 1, 1);
            // np450：逐条打印，专治"串里有、environ 里没有"这种静默跳过（本轮 DM_PROBE 就这么丢的）
            OH_LOG_Print(LOG_APP, LOG_INFO, DM_LOG_DOMAIN, "dm-host",
                         "start: setenv %{public}s rc=%{public}d", p, rc);
            p = end ? end + 1 : NULL;
        }
    }
    // 诊断：环境变量到底有没有设进去（直接用 hilog，不依赖后面的 stderr 桥）
    OH_LOG_Print(LOG_APP, LOG_INFO, DM_LOG_DOMAIN, "dm-host", "%{public}s", "start: 环境变量已设置");
    extern char **environ;
    for (char **e = environ; e && *e; e++)
        if (!strncmp(*e, "DM_", 3)) {
            char kv[256]; snprintf(kv, sizeof kv, "%s", *e);
            OH_LOG_Print(LOG_APP, LOG_INFO, DM_LOG_DOMAIN, "dm-host", "start: env %{public}s", kv);
        }
    // 宿主 shell 的读表约定：设 DM_DERIVED 后显式调（弱引用，没有这个接口的构建跳过）
    char derived[1300];
    snprintf(derived, sizeof derived, "%s/derived.tsv", g_runtime);
    setenv("DM_DERIVED", derived, 1);
    if (dm_derived_reload) dm_derived_reload();
    // np539：崩溃现场**直写文件**（配合 dm/host/crash.c 的 crash_set_fd）。
    //   为什么：宿主 stderr 走"管道 → hilog 转发线程"，而信号 11 一到系统会冻结其它线程
    //   （hilog 里的 FREEZE_signo_11），转发线程不再被调度 ⇒ 崩溃当轮 on_crash 打的 FAULT 全烂在管道里。
    //   np538 真机就是这个现象：崩在 nvn-translate，一行现场都没有，只剩系统的 DFX 块。
    //   ① 先把文件里**上一轮**留下的现场回灌进 hilog；② 清空后把 fd 交给宿主，之后 on_crash 直接 write()。
    // np539: crash state goes straight to a file (see crash_set_fd in dm/host/crash.c). The host's stderr runs
    //   through a pipe into a forwarding thread, which the system freezes on SIGSEGV (FREEZE_signo_11), so every
    //   FAULT line written during the crash is lost -- exactly np538's device run. (1) echo the previous run's
    //   file into hilog; (2) truncate it and hand the fd to the host, so on_crash can write() directly.
    {
        char cf[1400];
        snprintf(cf, sizeof cf, "%s/crash.txt", g_runtime);
        int fd = open(cf, O_RDWR | O_CREAT, 0600);
        if (fd >= 0) {
            char prev[2048];
            ssize_t got = read(fd, prev, sizeof prev - 1);
            if (got > 0) {
                prev[got] = 0;
                OH_LOG_Print(LOG_APP, LOG_INFO, DM_LOG_DOMAIN, "dm-host", "上一轮崩溃现场：%{public}s", prev);
            }
            if (ftruncate(fd, 0) == 0) lseek(fd, 0, SEEK_SET);
            crash_set_fd(fd);
            OH_LOG_Print(LOG_APP, LOG_INFO, DM_LOG_DOMAIN, "dm-host", "start: 崩溃现场直写就绪 fd=%{public}d 文件=%{public}s", fd, cf);
        } else {
            OH_LOG_Print(LOG_APP, LOG_WARN, DM_LOG_DOMAIN, "dm-host", "start: crash.txt 打不开（%{public}s）", strerror(errno));
        }
    }
    started = 1;
    pthread_t th;
    if (pthread_create(&th, NULL, host_thread, NULL)) {
        started = 0;
        napi_throw_error(env, NULL, "起宿主线程失败");
        return NULL;
    }
    pthread_detach(th);
    napi_value und = NULL;
    napi_get_undefined(env, &und);
    return und;
}

static napi_value Init(napi_env env, napi_value exports) {
    // 注册 XComponent 生命周期（形状照抄项目里已验证的 napi_init.c）
    napi_value exportInstance = NULL;
    if (napi_get_named_property(env, exports, OH_NATIVE_XCOMPONENT_OBJ, &exportInstance) == napi_ok) {
        OH_NativeXComponent *xc = NULL;
        if (napi_unwrap(env, exportInstance, (void **)&xc) == napi_ok && xc) {
            static OH_NativeXComponent_Callback cb;
            cb.OnSurfaceCreated = OnSurfaceCreatedCB;
            cb.OnSurfaceChanged = OnSurfaceChangedCB;
            cb.OnSurfaceDestroyed = OnSurfaceDestroyedCB;
            cb.DispatchTouchEvent = DispatchTouchEventCB;
            OH_NativeXComponent_RegisterCallback(xc, &cb);
        }
    }
    napi_property_descriptor desc[] = {
        { "probe", NULL, Probe, NULL, NULL, NULL, napi_default, NULL },
        { "setPad", NULL, SetPad, NULL, NULL, NULL, napi_default, NULL },
        { "start", NULL, Start, NULL, NULL, NULL, napi_default, NULL },
        { "readLog", NULL, ReadLog, NULL, NULL, NULL, napi_default, NULL },
        { "listDir", NULL, ListDir, NULL, NULL, NULL, napi_default, NULL },
        { "dirFileCount", NULL, DirFileCount, NULL, NULL, NULL, napi_default, NULL },
        { "makeDir", NULL, MakeDir, NULL, NULL, NULL, napi_default, NULL },
        { "logLine", NULL, LogLine, NULL, NULL, NULL, napi_default, NULL },
        { "writeFile", NULL, WriteFile, NULL, NULL, NULL, napi_default, NULL },
        { "writeFileGz", NULL, WriteFileGz, NULL, NULL, NULL, napi_default, NULL },
    };
    napi_define_properties(env, exports, sizeof desc / sizeof desc[0], desc);
    return exports;
}

static napi_module g_dms2_module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = NULL,
    .nm_register_func = Init,
    .nm_modname = "dms2",
    .nm_priv = NULL,
    .reserved = { 0 },
};

__attribute__((constructor)) static void RegisterDms2Module(void) { napi_module_register(&g_dms2_module); }
