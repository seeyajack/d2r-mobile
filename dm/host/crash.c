#include <TargetConditionals.h>
// 崩溃现场：打印出错地址、pc、lr（换成模块内偏移）与寄存器；缺页钩子分发。
// 只放信号处理本身：诊断功能（探针 / UI 消息 / 采样）在 diag.c，本文件不依赖它们（图形单测只链本文件）。
// Crash state: prints the faulting address, pc, lr (converted to module-relative offsets) and the registers; dispatches page fault hooks.
// Only signal handling itself lives here: diagnostic features (probes / UI messages / sampling) are in diag.c, and this file does not depend on them (the graphics unit tests link only this file).
#include "dm.h"
#include <string.h>
#include "dm_uctx.h"
#include <signal.h>
#include <dlfcn.h>
#include <sys/ucontext.h>
#include <unistd.h>
#include <sys/mman.h>
#include <libkern/OSCacheControl.h>
#include <mach/mach.h>
#include <stdatomic.h>
#include <time.h>

static _Atomic uint64_t g_watch_last_ns;
static _Atomic uint64_t g_watch_count;
static _Atomic uint32_t g_watch_tick;
static _Atomic int g_watch_pregame = 1;
static _Atomic int g_watch_stage_idx;
static char g_watch_stage[2][256] = {"启动", "启动"};
static unsigned g_watch_stall;
static void (*g_dump_hook)(void);
static void (*g_wait_dump)(void);

static uint64_t watch_now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
uint64_t crash_watch_progress_count(void) { return atomic_load(&g_watch_count); }
// ---------- 本轮标识 run-id（np304） ----------
// 为什么：np300–np302 三轮的结论建立在"崩点分毫未变"上，而 hilog 是环形缓冲、带历史全部崩溃块 ——
// 一旦读到的其实是**上一轮**的块，就会得出"补丁在文件里、在内存里都对，执行流却按旧指令走"这种
// 按代码不可能的结论（np303 已证明那个地址静态不可达）。所以现场必须能自证"这是哪一轮"。
// 用法：每个崩溃块头部都带 [run=… pid=… 构建=…]；读日志只认**最后一块**且块内有本轮痕迹。
// ---------- Per-run id (np304) ----------
// Why: the np300-np302 conclusions rested on "the crash point never moved", but hilog is a ring buffer that
// also holds every historical crash block -- reading a block from the PREVIOUS run yields the "patch is in the
// file and in memory, yet the old instruction still executes" paradox (np303 proved that address is statically
// unreachable). So every dump must identify the run it came from: read only the LAST block, and only if its
// run tag matches this run.
static char g_run_tag[96];
static void run_tag_init(void) {
    static int done; if (done) return; done = 1;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    snprintf(g_run_tag, sizeof g_run_tag, "run=%ld.%03ld pid=%ld 构建=%s %s",
             (long)ts.tv_sec, (long)(ts.tv_nsec / 1000000), (long)getpid(), __DATE__, __TIME__);
}
const char *dm_run_tag(void) { run_tag_init(); return g_run_tag; }
void crash_watch_progress(uint32_t tick, const char *stage) {
    int next = 1 - atomic_load(&g_watch_stage_idx);
    if (stage && *stage) snprintf(g_watch_stage[next], sizeof g_watch_stage[next], "%s", stage);
    atomic_store(&g_watch_tick, tick);
    atomic_store(&g_watch_stage_idx, next);
    atomic_fetch_add(&g_watch_count, 1);
    atomic_store(&g_watch_last_ns, watch_now_ns());
}
void crash_watch_frame(void) { if (atomic_load(&g_watch_pregame)) crash_watch_progress(0, "出图帧"); }
void crash_watch_pregame_done(void) { atomic_store(&g_watch_pregame, 0); }
void crash_watch_pregame_again(void) { atomic_store(&g_watch_pregame, 1); }   // 从局里退回前端：重新按出图帧算进展（diag.c） | Back from a run to the front end: count progress by presented frames again (diag.c)
void crash_dump_now(const char *why);
static void *watchdog_main(void *arg) {
    (void)arg;
    for (;;) {
        usleep(250000);
        // 先读最后进展时刻、再读现在：反过来读时，两次读之间游戏线程刚好记了一次进展，last 会比 now 新，now − last 无符号下溢成「第 18446744073 秒起无进展」，
        // 当场误判卡死退出（accept inv_drag_out，本机负载 11、看门狗线程两次读之间被切走）。这样读 last ≤ now；万一时钟回绕也不判
        // Read the last progress time first, then read now: reading in the opposite order, if the game thread happens to record progress between the two reads, last is newer than now and now − last underflows as unsigned into 「第 18446744073 秒起无进展」(no progress since second 18446744073),
        // which is misjudged on the spot as a hang and exits (accept inv_drag_out, local load 11, the watchdog thread was switched out between the two reads). Reading this way gives last ≤ now; and if the clock ever wraps around it is not judged either
        uint64_t last = atomic_load(&g_watch_last_ns), now = watch_now_ns();
        if (last > now) continue;
        // 秒级卡顿（进展停了 1 秒以上、还没到判死的阈值）：不退出，把各线程在等什么导一份。撞到「每 2 秒才出一帧」的现场，15 秒阈值抓不到、日志里没有它在等什么。
        // 进局之后才导；每个进程最多 3 份、两份之间至少隔 10 秒（读取画面、暂停菜单也会停拍，别让它们刷屏）
        // Second-scale stall (progress has stopped for more than 1 second but has not yet reached the kill threshold): do not exit, dump what each thread is waiting on. When hitting a case of "only one frame every 2 seconds", the 15-second threshold cannot catch it and the log has nothing on what it is waiting for.
        // Dump only after entering a run; at most 3 dumps per process, at least 10 seconds between two dumps (the loading screen and the pause menu also stop ticks, do not let them flood the log)
        { static int dumps; static uint64_t last_dump, dumped_for;
          if (g_watch_stall && !atomic_load(&g_watch_pregame) && atomic_load(&g_watch_tick) > 0 && now - last >= 1000000000ull && now - last < (uint64_t)g_watch_stall * 1000000000ull
              && dumps < 3 && dumped_for != last && (!last_dump || now - last_dump >= 10000000000ull)) { dumps++; last_dump = now; dumped_for = last; crash_dump_now("进展停了 1 秒以上"); } }
        if (g_watch_stall && now - last >= (uint64_t)g_watch_stall * 1000000000ull) {
            uint64_t sec = (now - last) / 1000000000ull;
            int idx = atomic_load(&g_watch_stage_idx);
            LOG("【红】看门狗：第 %llu 秒起无进展，卡在 拍 %u / 阶段 %s", (unsigned long long)sec,
                atomic_load(&g_watch_tick), g_watch_stage[idx]);
            if (g_dump_hook) g_dump_hook();
            if (g_wait_dump) g_wait_dump();   // 图形侧所有阻塞等待点的状态（nvn_mtl.m 初始化时注册） | State of all blocking wait points on the graphics side (registered when nvn_mtl.m initializes)
#if TARGET_OS_OSX
            { const char *d = getenv("DM_WATCH_SAMPLE");   // Mac 诊断：卡住时对自己采样 1 秒，线程栈写到该路径 | Mac diagnostics: when stuck, sample itself for 1 second and write the thread stacks to this path
              if (d) { char c[1200]; snprintf(c, sizeof c, "/usr/bin/sample %d 1 -f '%s' >/dev/null 2>&1", (int)getpid(), d); if (system(c)) LOG("看门狗：采样失败"); else LOG("看门狗：线程栈已写 %s", d); } }
#endif
            _exit(128 + SIGALRM);
        }
    }
    return NULL;
}
void crash_off(const char *tag, uint64_t a) {   // diag.c 也用 | Also used by diag.c
#define off crash_off

    module_t *m = module_of((void *)a);
    Dl_info di;
    if (m) fprintf(stderr, " %s%s%s+%#llx", tag, *tag ? "=" : "", m->name, a - (uint64_t)m->base);
    else if (dladdr((void *)a, &di) && di.dli_sname) fprintf(stderr, " %s%s%s+%#lx", tag, *tag ? "=" : "", di.dli_sname, (unsigned long)(a - (uint64_t)di.dli_saddr));
    else fprintf(stderr, " %s%s%#llx", tag, *tag ? "=" : "", a);
}
// 帧指针回溯：x29 链 [fp]=上一帧 fp、[fp+8]=返回地址；只走 8 字节对齐、单调上升的帧
// Frame pointer backtrace: x29 chain [fp]=previous frame's fp, [fp+8]=return address; only 8-byte aligned, monotonically increasing frames are walked
void dm_backtrace(uint64_t fp) {
    fprintf(stderr, "[dm] bt:");
    for (int i = 0; i < 32 && fp && !(fp & 7); i++) {
        uint64_t nfp = ((uint64_t *)fp)[0], ra = ((uint64_t *)fp)[1];
        off("", ra);
        if (nfp <= fp) break;
        fp = nfp;
    }
    fputc('\n', stderr);
}
// ---------- 缺页钩子（写保护脏页跟踪） ----------
// on_crash 打印现场前按注册顺序调用；返回 1 = 已处理（直接返回、重执行出错指令），全部返回 0 才走崩溃路径。
// 钩子运行在信号处理上下文：只许查表 / 原子操作 / mprotect，不许分配内存、加锁。
// ---------- Page fault hooks (write-protect dirty page tracking) ----------
// Called in registration order before on_crash prints the crash state; returning 1 = handled (return directly and re-execute the faulting instruction); the crash path is taken only when all return 0.
// Hooks run in signal handler context: only table lookups / atomic operations / mprotect are allowed, no memory allocation and no locking.
#define MAX_FAULT_HOOK 4
static int (*g_fault_hook[MAX_FAULT_HOOK])(void *addr, int is_write, void *uc);
static void (*g_write_prep[MAX_FAULT_HOOK])(void *addr, size_t len);
static int g_nfault_hook, g_nwrite_prep;
void dm_fault_hook_register(int (*fn)(void *addr, int is_write, void *uc)) {
    if (g_nfault_hook >= MAX_FAULT_HOOK) die("缺页钩子过多");
    g_fault_hook[g_nfault_hook++] = fn;
}
static void (*g_res_report)(uint32_t tick, const char *why);   // 资源计数回调（图形层注册，diag.c 按拍 / 事件调用） | Resource count callback (registered by the graphics layer, called by diag.c per tick / event)
void dm_res_report_register(void (*fn)(uint32_t tick, const char *why)) { g_res_report = fn; }
void (*dm_res_report_cb(void))(uint32_t, const char *) { return g_res_report; }
void dm_guest_write_prep_register(void (*fn)(void *addr, size_t len)) {
    if (g_nwrite_prep >= MAX_FAULT_HOOK) die("宿主写前钩子过多");
    g_write_prep[g_nwrite_prep++] = fn;
}
// 宿主自己写客体内存（read() 直读进客体缓冲等）之前调用：写保护页不会对系统调用触发缺页、而是返回 EFAULT
// Called before the host itself writes guest memory (read() straight into a guest buffer, etc.): for a system call a write-protected page does not trigger a page fault but returns EFAULT
void dm_guest_write_prepare(void *addr, size_t len) { for (int i = 0; i < g_nwrite_prep; i++) g_write_prep[i](addr, len); }

// 诊断模块（diag.c 的探针）与线程现场（hle_thread.c）由各自注册进来：单测只链本文件时不依赖它们
// The diagnostics module (the probes in diag.c) and the thread state dump (hle_thread.c) each register themselves here: a unit test that links only this file does not depend on them
static int (*g_trap_hook)(ucontext_t *uc);
void crash_set_trap_hook(int (*fn)(ucontext_t *uc)) { g_trap_hook = fn; }
void crash_set_dump_hook(void (*fn)(void)) { g_dump_hook = fn; }
void crash_set_wait_dump(void (*fn)(void)) { g_wait_dump = fn; }
// 按需导出现场（不退出）：iOS 包没有无进展看门狗，冒烟判红时经 DM_LOG_PORT 发 dump 触发——各客体线程在等什么 + 图形侧等待点，和看门狗打的是同一份
// Dump the current state on demand (without exiting): the iOS app has no no-progress watchdog, so when the smoke test fails it sends dump through DM_LOG_PORT to trigger this -- what each guest thread is waiting on + the graphics-side wait points, the same thing the watchdog prints
void crash_dump_now(const char *why) { fprintf(stderr, "[dm] 现场导出（%s）：拍 %u / 阶段 %s\n", why, atomic_load(&g_watch_tick), g_watch_stage[atomic_load(&g_watch_stage_idx)]); if (g_dump_hook) g_dump_hook(); if (g_wait_dump) g_wait_dump(); }
unsigned long dm_guest_malloc_stat(unsigned long *fail);
// ---------- 空指针保护桩的"现场自证"（np307） ----------
// 离线补丁（vk-spike/s3/fixup.py）把每个保护桩里"this 为空"的那一格从 `ret` 换成
// `brk #0xB0xx`（xx = 桩序号+1）。命中时在这里打印**是哪一个桩、寄存器、以及 lr（= 调用点）**，
// 然后把 PC 设成 LR —— 与原来那条 `ret` 语义完全一致，游戏照常往下走。
// 为什么值得这么做：np296–np303 追"this 为什么是 0"追了三天，最大的代价是**现场里没有"谁传的"**。
// 现在命中即带调用点，下一次真机日志里直接就有答案。
// ---------- Self-identifying null guards (np307) ----------
// The offline patch turns the "this is null" cell of every guard stub from `ret` into `brk #0xB0xx`
// (xx = guard index + 1). On a hit we print which guard, the registers, and lr (= the call site), then set
// PC to LR -- exactly what the old `ret` did, so the game keeps running. Worth it because np296-np303 chased
// "why is this null" for three days and the most expensive part was that the dump never said WHO passed it.
#define NULL_GUARD_BRK 0xB000u
// 是"我们的 brk"就返回 id，否则 0
// Return the id when this word is one of our brk instructions, 0 otherwise
static unsigned our_brk_id(uintptr_t pc) {
    if (pc < 0x1000 || (pc & 3)) return 0;
    uint32_t insn = *(volatile uint32_t *)pc;
    if ((insn & 0xFFE0001F) != 0xD4200000) return 0;          // 不是 brk | not a brk
    unsigned id = (insn >> 5) & 0xFFFF;
    // 0xB0xx = 空指针保护桩；0xB1xx = np315 诊断探针（只记录后继续执行）
    return (((id & 0xFF00) == NULL_GUARD_BRK) || ((id & 0xFF00) == 0xB100u)) ? id : 0;
}
static int null_guard_trap(ucontext_t *uc) {
    uintptr_t pc = (uintptr_t)UC_PC(uc);
    // 内核给的 BRK 返回地址约定不同：有的指向 brk 本身，有的已指向下一条 ⇒ 两个位置都试
    // Kernels differ on the preferred return address for BRK: some point at the brk, some already past it.
    unsigned id = our_brk_id(pc);
    if (!id) { id = our_brk_id(pc - 4); if (id) pc -= 4; }
    if (!id) return 0;                                        // 不是我们的 | not ours
    // np315 诊断：0xB1xx = 探针桩 —— 只记录，然后把 PC 推进到桩的下一个字（= 原第一条指令），
    // 既不改语义，也能回答"x0 在入口到底是什么、取指到底跟不跟补丁"。
    if ((id & 0xFF00) == 0xB100) {
        fprintf(stderr, "\n[dm] 【诊断】探针桩 #%u [%s] x0=%#llx x1=%#llx lr=%#llx\n",
                id & 0xFF, dm_run_tag(),
                (unsigned long long)UC_X(uc, 0), (unsigned long long)UC_X(uc, 1),
                (unsigned long long)UC_LR(uc));
        off("lr", (uint64_t)UC_LR(uc));
        UC_PC(uc) = pc + 4;
        return 1;
    }
    static int seen;
    if (seen++ < 16) {
        fprintf(stderr, "\n[dm] 【红】空指针保护命中 #%u [%s]\n", id & 0xFF, dm_run_tag());
        off("pc", (uint64_t)pc); off("lr", (uint64_t)UC_LR(uc));
        // np333（修正）：保护桩内部先 `str x19,[sp,#-0x20]!`，所以
        //   [sp+0x10] = 被调函数保存的 x29，[sp+0x18] = 它的返回地址（=0x45f9dc，已对上）
        //   被调函数的入口 sp = sp+0x20 ⇒ 构造函数自己的帧在 sp+0x20 起
        //   ⇒ **构造函数的返回地址（= 谁调用 0x45f9c8）在 [sp+0x38]**
        {
            uint64_t sp = (uint64_t)UC_SP(uc);
            fprintf(stderr, " np333 调用者链：");
            for (int d = 0; d < 4; d++) {
                uint64_t ret = *(uint64_t *)(uintptr_t)(sp + 0x20 * (uint64_t)(d + 1) + 0x18);
                const char *nm = "?"; uint64_t off = 0;
                for (int k = 0; k < g_nmod; k++) {
                    uint64_t lo = (uint64_t)(uintptr_t)g_mod[k].base, hi = lo + g_mod[k].size;
                    if (ret >= lo && ret < hi) { nm = g_mod[k].name; off = ret - lo; break; }
                }
                fprintf(stderr, " [%d]%s+%#llx", d, nm, (unsigned long long)off);
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, " x0=%#llx x1=%#llx x2=%#llx x3=%#llx sp=%#llx\n",
                (unsigned long long)UC_X(uc, 0), (unsigned long long)UC_X(uc, 1),
                (unsigned long long)UC_X(uc, 2), (unsigned long long)UC_X(uc, 3),
                (unsigned long long)UC_SP(uc));
    }
    UC_PC(uc) = UC_LR(uc);                                    // 等价于原来那条 ret | same as the old ret
    return 1;
}
// np327：**自己扫栈**，把"栈上落在客体代码段里的返回地址"按从近到远打出来。
// 为什么需要：改客体代码在本机不生效（np315–318），没法在入口下桩；而崩溃现场的线程转储被
// _exit 截断，拿不到调用链。栈里的返回地址是**数据**，扫它不需要任何补丁。
// np332：把地址格式化成 "模块+偏移"（两个缓冲轮换，便于一行里打两个地址）
// np339：倒出"各 NVN 入口最后一次是从客体的哪里被调的"（thunk 记录的 x30）。
static const char *fmt_off(uint64_t a);
extern void nvn_unimpl_entry(void);   // 与 hle_nvn.c 一致
static void dump_nvn_last_calls(void) {
    extern uint64_t nvn_last_ra[];
    extern const char *nvn_names[];
    typedef struct { _Atomic uint64_t cnt; void *fn; } nvn_slot_t;
    extern nvn_slot_t nvn_slot[];
    struct { const char *nm; uint64_t off; unsigned long long seq; } e[523];
    int n = 0;
    for (unsigned i = 0; i < 523; i++) {
        uint64_t ra = nvn_last_ra[i];
        if (!ra) continue;
        unsigned long long seq = atomic_load(&nvn_slot[i].cnt);
        if (!seq) continue;
        const char *nm = "?"; uint64_t off = 0;
        for (int k = 0; k < g_nmod; k++) {
            uint64_t lo = (uint64_t)(uintptr_t)g_mod[k].base, hi = lo + g_mod[k].size;
            if (ra >= lo && ra < hi) { nm = g_mod[k].name; off = ra - lo; break; }
        }
        e[n].nm = nm; e[n].off = off; e[n].seq = seq; n++;
    }
    // 按"该入口最后一次调用的全局序号"降序 = 最近调用的排前面
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++)
        if (e[b].seq > e[a].seq) { typeof(e[0]) t = e[a]; e[a] = e[b]; e[b] = t; }
    fprintf(stderr, "[dm] np341 NVN 最近调用（时间序，前 16）：\n");
    for (int a = 0; a < n && a < 16; a++)
        fprintf(stderr, "   %-40s <- %s+%#llx  (seq %llu)\n", e[a].nm, e[a].nm ? "" : "",
                (unsigned long long)e[a].off, e[a].seq);
    if (!n) fprintf(stderr, "   （没有记录）\n");
    // np342：**未实现但被调用过**的入口 = 嫌疑名单（返回 0 会让调用方跳过创建子对象 ⇒ 满盘空指针）
    fprintf(stderr, "[dm] np342 未实现但被调用过的 NVN 入口：\n");
    int bad = 0;
    for (unsigned i = 0; i < 523; i++) {
        if (nvn_last_ra[i] && nvn_slot[i].fn == (void *)nvn_unimpl_entry) {
            fprintf(stderr, "   ★ %s  （调用点 %s）\n", nvn_names[i], fmt_off(nvn_last_ra[i]));
            bad++;
        }
    }
    if (!bad) fprintf(stderr, "   （没有 —— 所有被调用的入口都有真实现）\n");
}
static const char *fmt_off(uint64_t a) {
    static char b[2][64]; static int k;
    char *o = b[k++ & 1];
    for (int i = 0; i < g_nmod; i++) {
        uint64_t lo = (uint64_t)(uintptr_t)g_mod[i].base, hi = lo + g_mod[i].size;
        if (a >= lo && a < hi) { snprintf(o, 64, "%s+%#llx", g_mod[i].name, (unsigned long long)(a - lo)); return o; }
    }
    snprintf(o, 64, "%#llx", (unsigned long long)a);
    return o;
}
static void dump_guest_stack_chain(uint64_t sp) {
    const char *nm; uint64_t off;
    uint64_t *p = (uint64_t *)(sp & ~7ull);
    int shown = 0;
    fprintf(stderr, "[dm] np327 栈返回地址链（sp=%#llx）：\n", (unsigned long long)sp);
    for (int i = 0; i < 4096 && shown < 24; i++) {
        uint64_t v = 0;
        // 直接读；越界会再崩，所以先按页粗判（栈区总是可读的，这里只在 sp..sp+32KB 内走）
        if ((uintptr_t)p >= sp + 0x8000) break;
        v = *p;
        nm = NULL; off = 0;
        for (int k = 0; k < g_nmod; k++) {
            uint64_t lo = (uint64_t)(uintptr_t)g_mod[k].base, hi = lo + g_mod[k].size;
            if (v >= lo && v < hi) { nm = g_mod[k].name; off = v - lo; break; }
        }
        if (nm && off < 0x2000000ull) {
            fprintf(stderr, "   %2d: %s+%#llx   (%#llx)\n", shown, nm, (unsigned long long)off, (unsigned long long)v);
            shown++;
        }
        p++;
    }
    if (!shown) fprintf(stderr, "   （栈上没找到指向客体代码的返回地址）\n");
}
static void on_crash(int sig, siginfo_t *si, void *uc_) {
    ucontext_t *uc = uc_;
    if (sig == SIGTRAP) {
        if (null_guard_trap(uc)) return;
        if (g_trap_hook && g_trap_hook(uc)) return;
    } else if (sig == SIGILL) {
        // np313：极少数平台把 BRK 投递成 SIGILL。若真是我们的桩，照样接住并继续，
        // 别让它走崩溃路径把进程杀掉、白丢一轮真机（判据：读指令字判断，不是我们的就照常崩）。
        // A few platforms deliver BRK as SIGILL; catch ours here too so the run is not wasted. Anything
        // that is not our brk falls through to the normal crash path.
        if (null_guard_trap(uc)) return;
    }
    // np332：**先打一行最小现场**——不读任何内存，保证一定能落地。
    // 教训：np330 在读 [sp+0x18] 时如果 sp 是坏的，handler 自己再崩，连 CRASH 行都出不来（np331 那轮就是这样）。
    // np335：**吞掉空指针写**（门票① 的最后手段）。
    // 依据：0 页映射被 mmap_min_addr 拒绝（np331），而崩溃点是 `str wzr,[x19,#0x60]`（x19=0）。
    // 这里不模拟指令、不重试，只**跳过**这条往低地址写的存储（PC+=4），让构造函数继续跑完。
    // 为什么可接受：往 0 地址写本身就是未定义行为；跳过它等价于"这个对象没被构造出来"，
    // 而游戏对未构造对象的常见处理是继续走（这正是 np296 想达到的效果，只是现在不需要改码）。
    if (sig == SIGSEGV || sig == SIGBUS) {
        uintptr_t pc = (uintptr_t)UC_PC(uc), bad = (uintptr_t)si->si_addr;
        if (bad < 0x1000 && pc > 0x1000) {
            uint32_t w = *(volatile uint32_t *)pc;
            // 只认"存储"类指令（str/stp/stur/strb/strh），别把读错误地吞掉
            // 放宽到整个 load/store 家族：构造函数里后面还有 `stp xzr,xzr,[x19,#0x50]` 这类成对存储。
            int is_store = ((w & 0x0C000000u) == 0x08000000u);
            if (is_store) {
                static uint64_t seen[8]; static int nseen;
                int known = 0;
                for (int i = 0; i < nseen; i++) if (seen[i] == pc) known = 1;
                if (!known && nseen < 8) {
                    seen[nseen++] = pc;
                    // np337：**帧原样 dump**（不再按序言猜偏移）——找出返回地址到底存在哪、
                    // 以及 0xa00000002 这个垃圾值是从哪来的（第一次吞写时帧还完整）。
                    if (nseen == 1) {   // 第一次吞写（PC 是绝对地址，不是偏移，之前判错了）
                        uint64_t sp0 = (uint64_t)UC_SP(uc);
                        fprintf(stderr, "[dm] np337 帧原样（sp=%#llx）：\n", (unsigned long long)sp0);
                        for (int k = -4; k < 20; k++) {
                            uint64_t v = *(uint64_t *)(uintptr_t)(sp0 + (uint64_t)(k * 8));
                            const char *nm = NULL; uint64_t off = 0;
                            for (int m = 0; m < g_nmod; m++) {
                                uint64_t lo = (uint64_t)(uintptr_t)g_mod[m].base, hi = lo + g_mod[m].size;
                                if (v >= lo && v < hi) { nm = g_mod[m].name; off = v - lo; break; }
                            }
                            fprintf(stderr, "   [sp%+3d] %#018llx%s%s\n", k * 8,
                                    (unsigned long long)v, nm ? "  = " : "", nm ? fmt_off(v) : "");
                        }
                    }
                    fprintf(stderr, "[dm] np335 吞掉空指针写：pc=%s addr=%#llx insn=%08x x0=%#llx\n",
                            fmt_off(pc), (unsigned long long)bad, w, (unsigned long long)UC_X(uc, 0));
                }
                UC_PC(uc) = pc + 4;
                return;
            }
        }
    }
    if (sig == SIGSEGV || sig == SIGBUS)
        fprintf(stderr, "\n[dm] FAULT sig=%d pc=%s lr=%s addr=%#llx x0=%#llx sp=%#llx fp=%#llx\n",
                sig, fmt_off((uint64_t)UC_PC(uc)), fmt_off((uint64_t)UC_LR(uc)),
                (unsigned long long)(uintptr_t)si->si_addr,
                (unsigned long long)UC_X(uc, 0), (unsigned long long)UC_SP(uc),
                (unsigned long long)UC_FP(uc));
    if (sig == SIGSEGV || sig == SIGBUS) dump_nvn_last_calls();
    // np330：**sp 基调用链**。0x45f9c8 的序言是 `stp x29,x30,[sp,#0x10]`，
    // 所以调用者的返回地址就在 [sp+0x18]，上一帧的 fp 在 [sp+0x10] —— 不需要 x29（它在本现场是垃圾）。
    if (sig == SIGSEGV || sig == SIGBUS) {
        uint64_t sp0 = (uint64_t)UC_SP(uc);
        if (sp0 < 0x10000 || sp0 > 0x800000000000ull) { fprintf(stderr, "[dm] np330 sp 不合理，跳过链\n"); goto np330_done; }
        uint64_t ret0 = *(uint64_t *)(uintptr_t)(sp0 + 0x18);
        uint64_t fp0  = *(uint64_t *)(uintptr_t)(sp0 + 0x10);
        fprintf(stderr, "[dm] np330 调用链（sp=%#llx）：\n", (unsigned long long)sp0);
        uint64_t fp = fp0;
        for (int lvl = 0; lvl < 16; lvl++) {
            uint64_t ret = ret0;
            if (lvl) { uint64_t *f = (uint64_t *)(uintptr_t)fp; ret = f[1]; fp = f[0]; }
            const char *nm = NULL; uint64_t off = 0;
            for (int k2 = 0; k2 < g_nmod; k2++) {
                uint64_t lo = (uint64_t)(uintptr_t)g_mod[k2].base, hi = lo + g_mod[k2].size;
                if (ret >= lo && ret < hi) { nm = g_mod[k2].name; off = ret - lo; break; }
            }
            fprintf(stderr, "   %2d: %s+%#llx\n", lvl, nm ? nm : "?", (unsigned long long)off);
            if (!fp) break;
        }
        np330_done: ;
    }
    if (sig == SIGSEGV || sig == SIGBUS) {
        int is_write = (int)((UC_ESR(uc) >> 6) & 1);   // ESR 的 WnR 位 | The WnR bit of ESR
        for (int i = 0; i < g_nfault_hook; i++) if (g_fault_hook[i](si->si_addr, is_write, uc)) return;
    }
    fprintf(stderr, "[dm] CRASH sig=%d [%s]", sig, dm_run_tag());
    // np254 临时探针（BotW 专属）：崩点 main+0x9fcfd8 的 blr x8 取自
    //   x8 = *(main+0x227ff90) → *(main+0x22d7c30)
    // 文件里这两处一个是 RELATIVE 槽、一个是全 0，运行时到底是什么值只能现场读。
    // 定位完就删。
    if (getenv("DM_DUMP_GLUE")) {
        for (int i = 0; i < g_nmod; i++) {
            module_t *m = &g_mod[i];
            if (strcmp(m->name, "main") || m->size <= 0x22d7c40) continue;
            uint64_t a = *(volatile uint64_t *)(m->base + 0x227ff90);
            uint64_t b = *(volatile uint64_t *)(m->base + 0x22d7c30);
            // 宿主 .so 基址（取本文件里一个已知函数）—— 用它把 b 换算成 so+偏移，再 nm 查名字
            fprintf(stderr, " [so=%p so+%#llx] [glue main+0x227ff90=%#llx → 0x22d7c30 内容=%#llx]",
                    (void *)&dm_backtrace,
                    (unsigned long long)(b - (uint64_t)(uintptr_t)&dm_backtrace),
                    (unsigned long long)a, (unsigned long long)b);
            fprintf(stderr, " [周围: %#llx %#llx %#llx %#llx]",
                    (unsigned long long)*(volatile uint64_t *)(m->base + 0x22d7c20),
                    (unsigned long long)*(volatile uint64_t *)(m->base + 0x22d7c28),
                    (unsigned long long)*(volatile uint64_t *)(m->base + 0x22d7c38),
                    (unsigned long long)*(volatile uint64_t *)(m->base + 0x22d7c40));
            break;
        }
    }
    // DM_GUARDDBG=1：崩溃现场再读一次 sdk 的 locale 守卫（np242 排障：
    // 装载后是 0、DT_INIT 期间被谁写成非 0，就能定案）
    if (getenv("DM_GUARDDBG")) {
        for (int i = 0; i < g_nmod; i++) {
            module_t *m = &g_mod[i];
            if (m->size > 0x78f1e8)
                fprintf(stderr, " [guard %s+0x78f1e8=%u +0x78f1e0=%#llx]",
                        m->name, (unsigned)*(volatile uint8_t *)(m->base + 0x78f1e8),
                        (unsigned long long)*(volatile uint64_t *)(m->base + 0x78f1e0));
            // 运行时字节：把 locale::__global 附近 64 字节打出来，和 patched.bin 比对
            // （静态反汇编与实测对不上时，这是唯一能定案的手段）
            if (m->size > 0x29e100 && !strcmp(m->name, "sdk")) {
                const uint8_t *p = m->base + 0x29e000;
                fprintf(stderr, "\n[dm] 运行时 %s+0x29e000:", m->name);
                for (int k = 0; k < 0xc0; k++) {
                    if (!(k % 16)) fprintf(stderr, "\n[dm]   %#06x:", 0x29e000 + k);
                    fprintf(stderr, " %02x", p[k]);
                }
                fprintf(stderr, "\n");
            }
        }
    }
    off("addr", (uint64_t)si->si_addr); off("pc", UC_PC(uc)); off("lr", UC_LR(uc));
    fprintf(stderr, " sp=%#llx\n", UC_SP(uc));
    for (int i = 0; i < 29; i++) fprintf(stderr, "x%d=%#llx%s", i, UC_X(uc, i), i % 6 == 5 ? "\n" : " ");
    // np260 临时探针（BotW 专属，定位完删）：崩点是 `ldr x0,[x19,#0x1b0]; ldr x8,[x0]`，
    // x0 = 0。把 x19 这个对象的 vptr 和几个关键字段打出来 ——
    // 有了 vptr 就能反查是哪个类、进而找到"本该构造它"的那个函数。
    if (getenv("DM_DUMP_GLUE")) {
        uint64_t o = UC_X(uc, 19);
        if (o) {
            volatile uint64_t *p = (volatile uint64_t *)(uintptr_t)o;
            fprintf(stderr, " [obj x19=%#llx vptr=%#llx +0x150=%#llx +0x158=%#llx +0x1b0=%#llx"
                            " +0x1d0=%#llx +0x208=%#x +0x198=%#llx +0xf8=%#llx]\n",
                    (unsigned long long)o, (unsigned long long)p[0],
                    (unsigned long long)p[0x150/8], (unsigned long long)p[0x158/8],
                    (unsigned long long)p[0x1b0/8], (unsigned long long)p[0x1d0/8],
                    (unsigned)*(volatile uint32_t *)(p + 0x208/8),
                    (unsigned long long)p[0x198/8], (unsigned long long)p[0xf8/8]);
        // np262：构造函数 0x9fc624 从 *(main+0x227f5e0) 取一个单例，调它的 vt[0](this, idx, &out)，
        // 要求 out(idx=0)==0x34 且 out(idx=1)>=0x20，不满足就**提前 return**（什么都不建）。
        // 把那个单例、它的 vptr、vt[0] 打出来 —— 就能定位是哪个查询在答这个值。
        {
            module_t *mm = NULL;
            for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { mm = &g_mod[i]; break; }
            if (mm && mm->size > 0x2280000) {
                uint64_t obj = *(volatile uint64_t *)(mm->base + 0x227f5e0);
                fprintf(stderr, " [query-obj=%#llx (main+%#llx) vptr=%#llx", (unsigned long long)obj,
                        (unsigned long long)(obj - (uint64_t)(uintptr_t)mm->base),
                        obj ? (unsigned long long)*(volatile uint64_t *)(uintptr_t)obj : 0ULL);
                // np295：帧链在崩点经常断（x29 链到 0xffffffff），改成**只认代码段地址**扫栈。
                // 原来扫 [base, base+size) 会把 .data 里的指针也当成返回地址，真正的调用链被挤掉。
                // 代码段只到 ~0x1200000，用它当上界，512 个 qword 足够覆盖到 nnMain。
                {
                    uint64_t spv = UC_SP(uc);
                    volatile uint64_t *st = (volatile uint64_t *)(uintptr_t)spv;
                    uint64_t lo = (uint64_t)(uintptr_t)mm->base;
                    uint64_t hi = lo + (mm->size < 0x1200000 ? mm->size : 0x1200000);
                    int shown = 0;
                    fprintf(stderr, " [np295 代码段栈扫 sp=%#llx：", (unsigned long long)spv);
                    for (int i = 0; i < 512 && shown < 24; i++) {
                        uint64_t v = st[i];
                        if (v >= lo && v < hi && !(v & 3)) { fprintf(stderr, " main+%#llx", (unsigned long long)(v - lo)); shown++; }
                    }
                    fprintf(stderr, " ]\n");
                    uint64_t fp = UC_FP(uc);
                    fprintf(stderr, " [np295 fp=%#llx 起 12 格：", (unsigned long long)fp);
                    if (fp > 0xffff && !(fp & 7)) {
                        volatile uint64_t *f = (volatile uint64_t *)(uintptr_t)fp;
                        for (int i = 0; i < 12; i++) fprintf(stderr, " %#llx", (unsigned long long)f[i]);
                    }
                    fprintf(stderr, " ]\n");
                }
                { unsigned long f = 0; unsigned long n = dm_guest_malloc_stat(&f);
                  fprintf(stderr, " [np294 客体 malloc→宿主 调用 %lu 次，失败 %lu 次]\n", n, f); }
                // np272：帧链在这里断了（x29 链到 0xffffffff），改**扫栈内存**找返回地址。
                // 把 sp 起 128 个 qword 里落在 main 代码段范围内的值打出来 —— 那就是真实调用链。
                {
                    uint64_t spv = UC_SP(uc);
                    volatile uint64_t *st = (volatile uint64_t *)(uintptr_t)spv;
                    int shown = 0;
                    fprintf(stderr, " [栈扫描 sp=%#llx：", (unsigned long long)spv);
                    for (int i = 0; i < 128 && shown < 12; i++) {
                        uint64_t v = st[i];
                        if (v >= (uint64_t)(uintptr_t)mm->base && v < (uint64_t)(uintptr_t)mm->base + mm->size) {
                            uint64_t off = v - (uint64_t)(uintptr_t)mm->base;
                            if (off < mm->size) { fprintf(stderr, " main+%#llx", (unsigned long long)off); shown++; }
                        }
                    }
                    fprintf(stderr, " ]\n");
                }
                // np271：崩点 main+0x9a4dc8 依赖的那张表。
                //   *(main+0x22655a0) → holder（.bss）
                //   *holder           → 表对象（计数在 +0x28、表体在 +0x30）
                // 离线已确认 main+0x22655a0 全二进制只有读没有写 ⇒ 只能上机看它运行时是什么。
                {
                    uint64_t holder = *(volatile uint64_t *)(mm->base + 0x22655a0);
                    fprintf(stderr, " [表 holder=%#llx", (unsigned long long)holder);
                    if (holder) {
                        uint64_t tobj = *(volatile uint64_t *)(uintptr_t)holder;
                        fprintf(stderr, " 表对象=%#llx", (unsigned long long)tobj);
                        if (tobj) {
                            uint32_t cnt = *(volatile uint32_t *)(uintptr_t)(tobj + 0x28);
                            uint64_t body = *(volatile uint64_t *)(uintptr_t)(tobj + 0x30);
                            fprintf(stderr, " 计数=%u 表体=%#llx", cnt, (unsigned long long)body);
                            if (body) {
                                volatile uint64_t *bp = (volatile uint64_t *)(uintptr_t)body;
                                fprintf(stderr, " 表体[0..3]=%#llx %#llx %#llx %#llx",
                                        (unsigned long long)bp[0], (unsigned long long)bp[1],
                                        (unsigned long long)bp[2], (unsigned long long)bp[3]);
                            }
                        }
                    }
                    fprintf(stderr, "]\n");
                }
                // np267：静态对象槽 *(main+0x2273a18) 及邻近 8 格。
                // 现场：崩点 main+0xba5ac8 的 this = 0x10，而该槽里正是 0x10。
                // 邻近全 0/垃圾 ⇒ 从没写过；只有这格异常 ⇒ 写了一半。两种修法不同。
                {
                    uint64_t *slot = (uint64_t *)(mm->base + 0x2273a18);
                    fprintf(stderr, " [静态对象槽 main+0x2273a18 = %#llx；邻近：", (unsigned long long)*slot);
                    for (int i = -3; i <= 4; i++) fprintf(stderr, " %#llx", (unsigned long long)slot[i]);
                    fprintf(stderr, "]\n");
                }
                // np283：崩点 main+0xba5ac8 的 this 来自惰性单例取用（0x45f9bc 读 *(main+0x2273a18)），
                // 而 main 里**写该槽的站点 0 处** ⇒ 槽由跨模块/全局初始化填，本次运行时为 0。
                // np273 已定出 5 个同类单例的槽址；一次全打出来：
                //   全 0      ⇒ 分配整段没跑（查哪次全局初始化该建它们）
                //   部分有值  ⇒ 只差某一个，直接锁定是哪个类的单例
                {
                    static const unsigned long kSlots[] = { 0x2273a18, 0x2270c18, 0x2271308, 0x2272f48, 0x2272fe0, 0x2275f90 };
                    fprintf(stderr, " [单例槽：");
                    for (unsigned si = 0; si < sizeof kSlots / sizeof kSlots[0]; si++) {
                        uint64_t v = *(volatile uint64_t *)(mm->base + kSlots[si]);
                        fprintf(stderr, " %#lx=%#llx%s", kSlots[si], (unsigned long long)v, v ? "" : "(空)");
                    }
                    fprintf(stderr, " ]\n");
                }
                // np285：客体 AS 的已登记区间 —— 判定崩溃地址（如 0x340000ca）是未映射还是越界
                { void svc_map_dump(void); svc_map_dump(); }
                // np287：主堆读数 —— this=0 若是分配失败，这里能一句话说清（总量/余量/最大可分配块）
                {
                    // np289：先**主动去问**分配器（dm_guest_heap_stat 会调 sdk 的 GetFreeSize /
                    // GetMaxAllocatable 并刷新 g_gheap_last），失败才退回上次读数。
                    // np287 那版只读 g_gheap_last，而它从没被采集过，打出来全是 0，判不了内存。
                    int dm_guest_heap_stat(unsigned long long *, unsigned long long *, unsigned long long *);
                    void dm_guest_heap_last(unsigned long long *);
                    unsigned long long h[3] = {0,0,0};
                    int fresh = dm_guest_heap_stat(&h[0], &h[1], &h[2]);
                    if (!fresh) dm_guest_heap_last(h);
                    fprintf(stderr, " [主堆%s：总量 %llu MB、余量 %llu MB、最大可分配块 %llu KB]\n",
                            fresh ? "（实时）" : "（上次读数）", h[0], h[1], h[2]);
                }
                // np290：崩点链 0x634fa0 → 0x9f33f0 → … → 0xba5aa4 里，0x9f33f0 走的是
                // "用调用方传进来的分配器"（np289 已排除另两条）。分配器疑似就是最后设进
                // TLS slot=4 的那个指针，把它指向的对象前 0x40 字节打出来，
                // 和日志里的 MemoryPool 地址比对：是池 ⇒ 类型/生命周期问题；不是 ⇒ 另一类对象。
                {
                    uint64_t dm_last_tls4(void);
                    uint64_t a = dm_last_tls4();
                    fprintf(stderr, " [TLS slot=4 最后值 = %#llx", (unsigned long long)a);
                    if (a > 0xffff) {
                        volatile uint64_t *q = (volatile uint64_t *)(uintptr_t)a;
                        fprintf(stderr, "，对象前 0x100 字节：");
                        for (int i = 0; i < 32; i++) fprintf(stderr, " %#llx", (unsigned long long)q[i]);
                    }
                    fprintf(stderr, " ]\n");
                }
                // np291：把客体 0xa0e2f8「取线程默认分配器」整条复刻一遍，再看分配器的虚表。
                // 0x9f33f0 的判据是 `cbnz x1`：x1==0（默认参数）才走这里，取到的分配器再被
                // `ldr x8,[x1]; ldr x3,[x8,#0x30]; br x3` 调它的第 7 个虚函数。this=0 只能是
                // 这条返回 0 —— 所以必须看清它取到了什么、那张表的 +0x30 是什么。
                //   0xa0e2f8: x8 = *(main+0x22783d8); x8 = *(x8); slot = *(u32*)(x8+0x88)
                //             tls = GetTlsValue(slot)
                //             alloc = tls ? *(tls+0xb0) : *(*(main+0x226f0a0) + 8)
                {
                    uint64_t dm_tlsval(unsigned slot);
                    uint64_t desc = *(volatile uint64_t *)(mm->base + 0x22783d8);
                    fprintf(stderr, " [np291 分配器链：desc=%#llx", (unsigned long long)desc);
                    unsigned slot = 0xffffffffu;
                    uint64_t tls = 0;
                    if (desc > 0xffff) {
                        uint64_t o = *(volatile uint64_t *)(uintptr_t)desc;
                        if (o > 0xffff) {
                            slot = *(volatile uint32_t *)(uintptr_t)(o + 0x88);
                            tls = dm_tlsval(slot);
                            fprintf(stderr, " tlsobj=%#llx slot=%u tls=%#llx", (unsigned long long)o, slot,
                                    (unsigned long long)tls);
                        }
                    }
                    uint64_t gv = *(volatile uint64_t *)(mm->base + 0x226f0a0);
                    uint64_t gvptr = gv > 0xffff ? *(volatile uint64_t *)(uintptr_t)gv : 0;
                    uint64_t alloc = 0;
                    if (tls > 0xffff) alloc = *(volatile uint64_t *)(uintptr_t)(tls + 0xb0);
                    else if (gvptr > 0xffff) alloc = *(volatile uint64_t *)(uintptr_t)(gvptr + 8);
                    fprintf(stderr, " gv=%#llx gvptr=%#llx => alloc=%#llx",
                            (unsigned long long)gv, (unsigned long long)gvptr, (unsigned long long)alloc);
                    if (alloc > 0xffff) {
                        uint64_t vt = *(volatile uint64_t *)(uintptr_t)alloc;
                        fprintf(stderr, " vt=%#llx(main+%#llx)", (unsigned long long)vt,
                                (unsigned long long)(vt - (uint64_t)(uintptr_t)mm->base));
                        if (vt > 0xffff) {
                            volatile uint64_t *v = (volatile uint64_t *)(uintptr_t)vt;
                            fprintf(stderr, " vt[0..7]=");
                            for (int i = 0; i < 8; i++) fprintf(stderr, " %#llx", (unsigned long long)v[i]);
                        }
                    }
                    fprintf(stderr, " ]\n");
                }
                if (obj) {
                    uint64_t vt = *(volatile uint64_t *)(uintptr_t)obj;
                    if (vt) fprintf(stderr, " vt0=main+%#llx vt1=main+%#llx]\n",
                        (unsigned long long)(*(volatile uint64_t *)(uintptr_t)vt) - (uint64_t)(uintptr_t)mm->base,
                        (unsigned long long)(*(volatile uint64_t *)(uintptr_t)(vt + 8)) - (uint64_t)(uintptr_t)mm->base);
                    else fprintf(stderr, "]\n");
                } else fprintf(stderr, "]\n");
            }
        }
        }
    }
    fprintf(stderr, "\n");
    // np298：**无条件**把空穴桩与被打补丁的入口读回来。
    // 为什么放顶层：原来塞在 `if (mm && mm->size > 0x2280000)` 里，实测那个分支没进
    // （日志里连 np295/np294 都没打），所以先无条件把 size 与内容一起打出来。
    {
        module_t *mm2 = NULL;
        for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { mm2 = &g_mod[i]; break; }
        fprintf(stderr, " [%s] [np298 nmod=%d mm=%p size=%#zx]", dm_run_tag(), g_nmod, (void *)mm2,
                mm2 ? (size_t)mm2->size : (size_t)0);
        if (mm2) {
            uint64_t cave = (uint64_t)(uintptr_t)(mm2->base + 0x2404a70);
            fprintf(stderr, " 空穴=");
            for (int i = 0; i < 12; i++) fprintf(stderr, " %08x", *(volatile uint32_t *)(uintptr_t)(cave + i * 4));
            fprintf(stderr, " | 入口 45f9c8=%08x 45f9cc=%08x 105f9e4=%08x",
                    *(volatile uint32_t *)(uintptr_t)(mm2->base + 0x45f9c8),
                    *(volatile uint32_t *)(uintptr_t)(mm2->base + 0x45f9cc),
                    *(volatile uint32_t *)(uintptr_t)(mm2->base + 0x105f9e4));
        }
        { void dm_dump_alloc_state(const char *); dm_dump_alloc_state("崩溃现场"); }
        // np328：把**调用者**（subsdk0+0x10cb80 附近）的运行期指令 dump 出来。
        // 为什么：subsdk0 的 .bin 是压缩映像，离线反汇编对不上；运行期内存里才是真指令。
        // 调用链来自 np327 的栈扫描：main+0x45f9dc ← subsdk0+0x10cb80 ← sdk+0x6c21d0 …
        for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "subsdk0")) {
            uint32_t *q = (uint32_t *)(g_mod[i].base + 0x10cb30);
            fprintf(stderr, "[dm] np328 subsdk0+0x10cb30:");
            for (int k = 0; k < 28; k++) fprintf(stderr, " %08x", q[k]);
            fprintf(stderr, "\n");
            break;
        }
        fprintf(stderr, " | 原始 pc=%#llx 该处指令=%08x lr=%#llx",
                (unsigned long long)UC_PC(uc),
                *(volatile uint32_t *)(uintptr_t)UC_PC(uc),
                (unsigned long long)UC_LR(uc));
        fprintf(stderr, "\n");
    }
    dm_backtrace(UC_FP(uc));
    if (g_dump_hook) g_dump_hook();
    _exit(128 + sig);
}
// 信号备用栈是每线程的：每个客体线程起跑时都要调（栈溢出时才能在备用栈上打出现场）
// 备用栈随线程退出释放：pthread 键的析构在线程退出时调用（先停用备用栈再 free）。游戏会频繁起短命线程，不释放每个漏 64 KiB。
// The signal alternate stack is per thread: every guest thread must call this when it starts running (only then can the crash state be printed on the alternate stack on a stack overflow)
// The alternate stack is released when the thread exits: the pthread key destructor is called at thread exit (disable the alternate stack first, then free). The game frequently starts short-lived threads, and without releasing each one leaks 64 KiB.
#if !TARGET_OS_TV   // tvOS 的 SDK 没有 sigaltstack：不设备用栈，崩溃处理在原栈上跑（栈溢出时打不出现场） | The tvOS SDK has no sigaltstack: no alternate stack is set, so the crash handler runs on the original stack (no crash state can be printed on stack overflow)
static pthread_key_t g_altstack_key;
static pthread_once_t g_altstack_once = PTHREAD_ONCE_INIT;
static void altstack_free(void *p) { stack_t off = {.ss_flags = SS_DISABLE}; sigaltstack(&off, 0); free(p); }
static void altstack_key_init(void) { if (pthread_key_create(&g_altstack_key, altstack_free)) die("altstack key"); }
void crash_thread_init(void) {
    pthread_once(&g_altstack_once, altstack_key_init);
    if (pthread_getspecific(g_altstack_key)) return;                 // 同一线程重复调用不重复分配 | Repeated calls on the same thread do not allocate again
    stack_t ss = {.ss_sp = malloc(65536), .ss_size = 65536}; sigaltstack(&ss, 0);
    pthread_setspecific(g_altstack_key, ss.ss_sp);
}
#define CRASH_SA_ONSTACK SA_ONSTACK
#else
void crash_thread_init(void) {}
#define CRASH_SA_ONSTACK 0
#endif
void crash_install(void) {
    crash_thread_init();
    // np304：启动就打一行本轮标识 —— 读 hilog 时"哪一块是这一轮"由此判定，不再靠猜。
    fprintf(stderr, "[dm] 本轮启动 %s\n", dm_run_tag());
    struct sigaction sa = {.sa_sigaction = on_crash, .sa_flags = SA_SIGINFO | CRASH_SA_ONSTACK};
    sigaction(SIGALRM, &sa, 0);
    // 两种看门狗都只在测试显式设了 DM_WATCH / DM_WATCH_STALL 时开：产品（玩家从桌面启动，无环境变量）一个不开。
    // 读代码查到：iOS main.m 缺省 setenv DM_WATCH=30 → 玩家启动 30 s 被 SIGALRM 杀；无进展看门狗缺省 15 s → 暂停菜单（拍停）15 s 被杀
    // Both watchdogs are turned on only when a test explicitly sets DM_WATCH / DM_WATCH_STALL: the product (the player launches from the desktop, no environment variables) turns on neither.
    // Found by reading the code: iOS main.m did setenv DM_WATCH=30 by default → the player's launch was killed by SIGALRM at 30 s; the no-progress watchdog defaulted to 15 s → the pause menu (ticks stopped) was killed at 15 s
    const char *watch = getenv("DM_WATCH"), *stall = getenv("DM_WATCH_STALL");
    if (watch) alarm(atoi(watch));
    if (watch || stall) {
        g_watch_stall = stall && atoi(stall) > 0 ? (unsigned)atoi(stall) : 15;
        atomic_store(&g_watch_last_ns, watch_now_ns());
        pthread_t th; if (pthread_create(&th, NULL, watchdog_main, NULL)) die("看门狗线程");
        pthread_detach(th);
    }
    sigaction(SIGSEGV, &sa, 0); sigaction(SIGBUS, &sa, 0); sigaction(SIGILL, &sa, 0); sigaction(SIGTRAP, &sa, 0);
}
