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
#include <sys/uio.h>   // np348b: process_vm_readv 安全读客体内存
#include <unistd.h>
#include <sys/syscall.h>   // np544：SYS_gettid —— 崩溃现场要能说出是哪个线程 | np544: which thread crashed
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

// ---------- np539：崩溃现场**直写**（绕开 stderr 管道） ----------
// 为什么必须直写：鸿蒙侧把宿主 stderr 接进一根管道，再由一个转发线程逐行 OH_LOG_Print（napi_init.c 的
// stderr_to_hilog）。可信号 11 一到，系统会**冻结其它线程**（hilog 里那行 `FREEZE_signo_11`），转发线程不再
// 被调度 ⇒ on_crash 里所有 fprintf(stderr) 全烂在管道里，日志上只剩系统的 DFX 块、看不到我们的 pc/lr。
// np538 真机就是这个现象：崩溃线程 nvn-translate，`FAULT` 一行都没有。
// 做法：平台层启动时把一个沙箱内文件的 fd 交进来，on_crash 第一件事就是 write() 这一行（write 是
// async-signal-safe 的）；应用下次启动时把这个文件回灌进 hilog。
// np539: write the crash state straight to a file descriptor. On HarmonyOS the host's stderr goes through a pipe
// into a forwarding thread; a SIGSEGV freezes the other threads (the `FREEZE_signo_11` line), so that thread never
// runs and every fprintf(stderr) in on_crash is lost -- np538's device log had no FAULT line at all. The platform
// layer hands in an fd for a file inside the app sandbox, and on_crash writes one line to it first; the app echoes
// that file into hilog on the next launch.
static int g_crash_fd = -1;
void crash_set_fd(int fd) { g_crash_fd = fd; }
static void crash_note(const char *s) { if (g_crash_fd >= 0 && s) { ssize_t w = write(g_crash_fd, s, strlen(s)); (void)w; } }

// np544：崩溃现场再补三个读数——线程名、线程号、以及 sp 底下还有没有映射过的页。
// 为什么：np543 修完后新崩点是 `addr = sp + 0x110`、`sp` 比客体主线程栈区起点低 3.1 MB，
// 看着像"栈被用穿"，但当轮既不知道崩在哪个线程，也不知道 sp 底下还剩多少可用栈页。
// np544: three more readings at the crash site -- thread name, tid, and whether any mapped page remains
// just below sp. np543's new crash had addr = sp + 0x110 with sp 3.1 MB below the guest main stack
// region, which looks like stack exhaustion, but the thread was unknown.
static int cpage_mapped(uintptr_t a) {
    void *p = (void *)(a & ~(uintptr_t)0xfff); unsigned char v = 0;
    return mincore(p, 0x1000, &v) == 0;
}
static long cmapped_below(uintptr_t sp) {   // 向下最多探 1 MB：返回最近的可映射页距离，-1 = 1 MB 内全空 | nearest mapped page below sp within 1 MB, -1 = hole
    for (long d = 0; d <= (1 << 20); d += 0x1000) if (cpage_mapped(sp - (uintptr_t)d)) return d;
    return -1;
}

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
// np397：**栈扫描回溯**。为什么不能只靠 x29 链：sdk 的导入跳板（sdk+0x7b1474 那一段）会
// `adrp x29; ldr x29,[x29,#0x18]` **改写帧指针**，一过跳板 FP 链就断，打出来的"帧"是垃圾
// （np384/385/391 三次被误导）。栈扫描不受影响：在 [fp, fp+0x600) 里找落在模块**代码段**里的 8 字节值，
// 按出现顺序当成返回地址。会多几个假阳性，但**不会漏掉真的调用者**。
// np397: stack-scan backtrace. The x29 chain alone is not usable: the sdk's import trampolines
// (the block at sdk+0x7b1474) do `adrp x29; ldr x29,[x29,#0x18]`, i.e. they REWRITE the frame pointer, so the
// chain breaks as soon as it crosses a trampoline and the printed frames are garbage (np384/385/391 were all
// misled). A stack scan is immune: look for 8-byte values inside module CODE ranges in [fp, fp+0x600) and treat
// them, in order, as return addresses. A few false positives, but it never misses the real caller.
static int in_code(uint64_t a) {
    for (int i = 0; i < g_nmod; i++) {
        uint64_t lo = (uint64_t)g_mod[i].base, hi = lo + (g_mod[i].text_end ? g_mod[i].text_end : g_mod[i].size);
        if (a >= lo && a < hi) return 1;
    }
    return 0;
}
static void dump_nvn_last_calls(void);   // np480：定义在下方
// np480：给看门狗路径用的 NVN 最近调用转储（dump_nvn_last_calls 是 static，这里包一层）。
// 用途：EndRecording 之后没有 QueueSubmit——最后调的 16 个 NVN 入口能直接指出游戏在做什么
// （Fence？Sync？还是别的 Submit 变体）。
// np480: NVN last-calls dump for the watchdog path (dump_nvn_last_calls is static; wrap it).
void dm_dump_nvn_last_calls(void) { dump_nvn_last_calls(); }
// np465：**崩溃处理自己不能崩**。原来这里直接解引用 `fp+k`，而 fp 落在栈映射末尾时
// （实测 addr=0x5e0be30000）读越界 ⇒ 处理器自己 SIGSEGV，把真正的崩点盖掉。
// 改用 vm_read_overwrite：失败就跳过，绝不碰未映射页。
// np465: the crash handler must never crash. It used to dereference fp+k directly; when fp sits at the end
// of the stack mapping (addr=0x5e0be30000 observed) that read faults and hides the real crash. Use
// vm_read_overwrite instead: on failure just skip.
static int scan_rd(uint64_t a, void *out, size_t n) {
    vm_size_t g = 0;
    return !vm_read_overwrite(mach_task_self(), a, n, (vm_address_t)out, &g) && g == n;
}
void dm_backtrace_scan(uint64_t fp) {
    if (!fp || (fp & 7)) return;
    fprintf(stderr, "[dm] bt(栈扫描):");
    int shown = 0;
    for (int k = 0; k < 0x600 && shown < 16; k += 8) {   // 变量名不能叫 off：那是个宏（crash_off） | do not name it `off`, that is a macro
        uint64_t v = 0;
        if (!scan_rd(fp + k, &v, sizeof v)) continue;   // np465：读不到就跳过（不再直接解引用）
        if (!in_code(v)) continue;
        // np400：**候选返回地址必须"前一条是 bl/blr"**。栈上还会放函数地址（vtable、函数指针表），
        // 它们落在代码段里但不是返回地址 —— np397 的 `sdk+0x8dd58`（ro::Bind）就是这么来的，
        // 判决实验（np398）证明它一次都没被调用。加这一条能把假象滤掉。
        // np400: a candidate return address must be preceded by a bl/blr. The stack also holds plain function
        // addresses (vtables, pointer tables) that live in code but are not return addresses -- that is exactly how
        // np397 produced sdk+0x8dd58 (ro::Bind), which the np398 experiment proved is never called.
        if (v < 4) continue;
        uint32_t prev = 0;
        if (!scan_rd(v - 4, &prev, sizeof prev)) continue;   // np465：同上
        int is_bl = (prev & 0xFC000000u) == 0x94000000u;          // bl  #imm26
        int is_blr = (prev & 0xFFFFFC1Fu) == 0xD63F0000u;         // blr xN
        if (!is_bl && !is_blr) continue;
        off("", v);
        shown++;
    }
    if (!shown) fprintf(stderr, " （栈上没有任何代码地址：fp 可能已被跳板改写）");
    fputc('\n', stderr);
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
    extern uint64_t nvn_last_ts[];   // np348：真时间序（thunk 记的 cntvct）
    extern const char *nvn_names[];
    typedef struct { _Atomic uint64_t cnt; void *fn; } nvn_slot_t;
    extern nvn_slot_t nvn_slot[];
    struct { const char *entry; const char *nm; uint64_t off; unsigned long long seq; } e[523];
    int n = 0;
    for (unsigned i = 0; i < 523; i++) {
        uint64_t ra = nvn_last_ra[i];
        if (!ra) continue;
        // np348：排序键换成**调用时间戳**（cntvct 单调计数），而不是每槽累计次数
        unsigned long long seq = nvn_last_ts[i];
        if (!seq) seq = atomic_load(&nvn_slot[i].cnt);   // 兜底：时间戳没写（老产物）时退回计数
        const char *nm = "?"; uint64_t off = 0;
        for (int k = 0; k < g_nmod; k++) {
            uint64_t lo = (uint64_t)(uintptr_t)g_mod[k].base, hi = lo + g_mod[k].size;
            if (ra >= lo && ra < hi) { nm = g_mod[k].name; off = ra - lo; break; }
        }
        e[n].entry = nvn_names[i]; e[n].nm = nm; e[n].off = off; e[n].seq = seq; n++;
    }
    // 按"该入口最后一次调用的时间戳"降序 = 最近调用的排前面
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++)
        if (e[b].seq > e[a].seq) { typeof(e[0]) t = e[a]; e[a] = e[b]; e[b] = t; }
    fprintf(stderr, "[dm] np348 NVN 最近调用（真时间序，前 16）：\n");
    unsigned long long tmax = n ? e[0].seq : 0;
    for (int a = 0; a < n && a < 16; a++)
        fprintf(stderr, "   %-44s <- %s+%#llx  (t-%llu)\n",
                e[a].entry ? e[a].entry : "?", e[a].nm ? e[a].nm : "?",
                (unsigned long long)e[a].off, tmax - e[a].seq);
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
// np432：**缺页计数**。目的只有一个：把"主线程停在一条写指令上不动"的两种解释分开——
// ① 反复缺页重试（计数暴涨，且 addr/pc 恒定）；② 根本没缺页，就是停着（计数 0）。
// 之前只能靠"日志里没出现 CRASH"来推断"没缺页"，那是不成立的：钩子接住的缺页本来就不打日志。
// np432: fault counter. One purpose: separate the two readings of "the main thread sits on a store instruction" --
// (1) repeated fault-and-retry (count explodes, addr/pc constant); (2) no fault at all, it is simply parked (count 0).
// Until now "no CRASH line in the log" was the only evidence for "no fault", and that does not hold: a fault taken
// by a hook never prints anything.
static _Atomic uint64_t g_nfault, g_fault_pc, g_fault_addr;
void dm_fault_stats(unsigned long long *n, unsigned long long *pc, unsigned long long *addr) {
    *n = atomic_load(&g_nfault); *pc = atomic_load(&g_fault_pc); *addr = atomic_load(&g_fault_addr);
}
static void on_crash(int sig, siginfo_t *si, void *uc_) {
    ucontext_t *uc = uc_;
    if (sig == SIGSEGV || sig == SIGBUS) {   // 先记一笔再做任何判断（钩子接住的分支也要算进去） | count first, before any decision (hook-handled paths included)
        atomic_fetch_add(&g_nfault, 1);
        atomic_store(&g_fault_pc, (uint64_t)UC_PC(uc));
        atomic_store(&g_fault_addr, (uint64_t)(uintptr_t)(si ? si->si_addr : 0));
        // np539：第一件事就是直写一行（不经过 stdio / 管道 / hilog），保证现场一定落地。
        // np539: the very first action is a direct write of one line (no stdio, no pipe, no hilog).
        {
            char line[512];
            // np540：顺手把 nvn 回放的"最后一条命令"带上（图形引擎导出，没有图形单测时就是 -1）。
            // np540: also carry the last NVN replay command (exported by the graphics engine; -1 in a graphics-only build).
            extern volatile unsigned dm_last_cmd_op __attribute__((weak));
            extern volatile unsigned dm_last_cmd_size __attribute__((weak));
            unsigned lastop = (&dm_last_cmd_op) ? dm_last_cmd_op : 0xffffffffu;
            unsigned lastsz = (&dm_last_cmd_size) ? dm_last_cmd_size : 0;
            // ★ 第一行必须**只做最小工作**：np545 实测，把 pthread_getname_np 之类放进第一行会
            //   让处理器自己在里面崩掉，于是连 pc 都留不下来（那轮 crash.txt 是空的）。
            //   所以先落最小现场，附加读数一律排在后面，各自失败都不影响前面已经写下去的内容。
            // The first line must do the minimum: on device, adding pthread_getname_np here killed the
            // handler itself so even pc was lost (that run left crash.txt empty). Write the minimum first,
            // then append extra readings, each of which can fail without losing what is already on disk.
            int n = snprintf(line, sizeof line, "FAULT sig=%d pc=%s lr=%s addr=%#llx sp=%#llx fp=%#llx lastop=%u/%u\n",
                             sig, fmt_off((uint64_t)UC_PC(uc)), fmt_off((uint64_t)UC_LR(uc)),
                             (unsigned long long)(uintptr_t)(si ? si->si_addr : 0),
                             (unsigned long long)UC_SP(uc), (unsigned long long)UC_FP(uc),
                             lastop, lastsz);
            if (n > 0) crash_note(line);
            // 附加：sp 附近还有没有映射过的页（判断"栈被用穿"）。mincore 是纯系统调用，安全。
            {
                uintptr_t csp = (uintptr_t)UC_SP(uc);
                int n2 = snprintf(line, sizeof line, "FAULT2 spmap=%d below=%ld\n", cpage_mapped(csp), cmapped_below(csp));
                if (n2 > 0) crash_note(line);
            }
            // np539：再补一条 fp 链——libc 内部崩时 lr 可能是残留值（np539 那轮的 lr 落在一条没有 bl 的指令上），
            // 只靠 pc/lr 分不清"谁传进来的坏指针"。每帧都用 process_vm_readv 读，读崩了就停，不会再制造第二次崩溃。
            // np539: also dump the frame-pointer chain. Inside libc the lr can be stale (np539's lr landed on an
            // instruction that has no bl), so pc/lr alone cannot say who passed the bad pointer. Each frame is read
            // with process_vm_readv; a bad read just stops the walk instead of crashing a second time.
            {
                uint64_t fp = (uint64_t)UC_FP(uc);
                for (int i = 0; i < 16 && fp; i++) {
                    uint64_t pair[2] = { 0, 0 };
                    struct iovec li = { pair, 16 }, ri = { (void *)(uintptr_t)fp, 16 };
                    if (process_vm_readv(getpid(), &li, 1, &ri, 1, 0) != 16) break;
                    int n2 = snprintf(line, sizeof line, "FRAME %d ret=%s fp=%#llx\n",
                                      i, fmt_off(pair[1]), (unsigned long long)pair[0]);
                    if (n2 > 0) crash_note(line);
                    if (pair[0] <= fp) break;
                    fp = pair[0];
                }
            }
        }
    }
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
        // np351：**默认不再吞写**。np349 真机证明 np335 正在掩盖真正的病因：
        // 构造函数 `main+0x45f9c8` 被以 `this=NULL` 调用，4 条写（含写 vptr 那句 `str x8,[x19]`）
        // 全被吞掉，对象从头到尾没被构造，几百毫秒后才在无关代码里踩栈垃圾崩。
        // 现在默认让它在第一时间崩：崩点就是"构造 NULL 对象的那一行"，`lr` 就是真正把 NULL 传进来的人。
        // 想恢复旧行为（吞写继续跑）设 `DM_SWALLOW=1`。
        // np351: no longer swallow by default. np349 proved np335 masks the real cause: ctor main+0x45f9c8
        // is entered with this=NULL, all 4 stores (including the vptr store) are skipped, the object is never
        // constructed, and the process dies hundreds of ms later on unrelated garbage. Crashing immediately puts
        // the fault back on the NULL-object construction line, and `lr` names whoever passed NULL.
        // Set `DM_SWALLOW=1` to restore the old swallow-and-continue behaviour.
        if (bad < 0x1000 && pc > 0x1000 && getenv("DM_SWALLOW") != NULL) {
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
    // np435：崩溃现场顺带把"每个文件读了几次/多少字节"打出来。启动期的崩溃常常早于 5 秒的滴答，
    // 不在这里打就永远看不到（np435 就是 0.7 秒崩的）。
    // np435: also dump the per-file read accounting at the crash site. Early-boot crashes usually beat the 5 s
    // tick, so without this it is never printed (np435 crashed 0.7 s in).
    if (sig == SIGSEGV || sig == SIGBUS) { void dm_fs_dump_reads(void); dm_fs_dump_reads(); }
    // np369：**fp 链逐帧打印**。np330 读 [sp+0x18] 只能看最近一帧，而这次要回答的是
    // "谁调用了 0x45f9c8"。帧指针链是 `x29` 指向本帧保存的 x29/x30 对（[fp]=旧 x29、[fp+8]=返回地址），
    // 顺链走 8 帧就能把真实调用者列出来（每帧都用 process_vm_readv，不会二次崩）。
    // np369: walk the frame-pointer chain and print each frame's saved return address; np330 only read the
    // innermost frame, and the question here is who called 0x45f9c8.
    if ((sig == SIGSEGV || sig == SIGBUS) && UC_FP(uc)) {
        uint64_t fp = (uint64_t)UC_FP(uc);
        fprintf(stderr, "[dm] np369 fp 链（每帧 返回地址）：\n");
        for (int i = 0; i < 8 && fp; i++) {
            uint64_t pair[2] = { 0, 0 };
            struct iovec li = { pair, 16 }, ri = { (void *)(uintptr_t)fp, 16 };
            if (process_vm_readv(getpid(), &li, 1, &ri, 1, 0) != 16) { fprintf(stderr, "   帧%d 读不到\n", i); break; }
            fprintf(stderr, "   帧%d ret=%s  fp=%#llx\n", i, pair[1] ? fmt_off(pair[1]) : "0",
                    (unsigned long long)pair[0]);
            if (pair[0] <= fp) break;
            fp = pair[0];
        }
    }
    // np364：**守卫记录槽**。9 个守卫桩把 (x0 = this, x30 = 调用点)
    // 写进数据段 main+0x2405a70 起的槽位（每 16 字节一个）。崩的时候读出来，
    // "每个入口收到的 this 到底是什么、是谁递进来的"就变成直接读数，不用再靠帧链推断。
    // np364: the guard stubs now record instead of blocking, storing (x0 = this, x30 = call site) into
    // cells at main+0x2405a70 (16 bytes each). Reading them at the crash turns "what this did each entry get,
    // and who passed it" into a direct read instead of frame-chain inference.
    if (sig == SIGSEGV || sig == SIGBUS) {
        static const char *gnm[10] = { "就地构造辅助", "类构造器", "包装A", "包装B", "包装B上层", "包装C", "包装D", "包装E", "分配器", "类构造器中段" };
        for (int k7 = 0; k7 < g_nmod; k7++) {
            if (strcmp(g_mod[k7].name, "main")) continue;
            uint64_t cb = (uint64_t)(uintptr_t)g_mod[k7].base;
            fprintf(stderr, "[dm] np364 守卫记录槽（this / 调用点）：\n");
            for (int i = 0; i < 10; i++) {
                uint64_t v[2] = { 0, 0 };
                struct iovec li = { v, 16 }, ri = { (void *)(uintptr_t)(cb + 0x2405a70 + 0x10 * (uint64_t)i), 16 };
                if (process_vm_readv(getpid(), &li, 1, &ri, 1, 0) != 16) {
                    fprintf(stderr, "   %-14s 读不到\n", gnm[i]);
                    continue;
                }
                fprintf(stderr, "   %-14s this=%#llx  caller=%s\n", gnm[i],
                        (unsigned long long)v[0], v[1] ? fmt_off(v[1]) : "0（未被调用）");
                // np379：探针槽（i>=9）存的是 {sp, x30} —— 顺着 sp 打 16 个字，
                // 找"跳进函数体中段"那一方留下的返回地址。
                // np379: probe cells (i >= 9) hold {sp, x30}; dump 16 words at that sp to find the jumper.
                if (i >= 9 && v[0] > 0x1000) {
                    uint64_t w16[16] = { 0 };
                    struct iovec li2 = { w16, sizeof w16 }, ri2 = { (void *)(uintptr_t)v[0], sizeof w16 };
                    if (process_vm_readv(getpid(), &li2, 1, &ri2, 1, 0) == (ssize_t)sizeof w16) {
                        fprintf(stderr, "      栈(sp=%#llx):", (unsigned long long)v[0]);
                        for (int q = 0; q < 16; q++)
                            fprintf(stderr, " %s", w16[q] ? fmt_off(w16[q]) : "0");
                        fprintf(stderr, "\n");
                    }
                }
            }
            break;
        }
    }
    // np362：**看空指针保护桩的真实字节**。np296 把 7 个构造入口改成跳进代码洞（main+0x2404a70 起，
    // 每 0x20 一个桩），桩本应是 `cbz x0,+0xC / 原首指令 / b 回函数体 / ret`。
    // 真机却是 this=0 仍然进到函数体（`mov x19,x0` 之后才崩）⇒ 怀疑 cbz 没生效，
    // 也就是洞里真实的字节与设计不符。这里把洞里的前 4 个字直接打出来。
    // np362: dump the real bytes of the null-this guard stubs. np296 turns 7 constructor entries into jumps
    // into the code cave (main+0x2404a70, one stub every 0x20) whose body should be
    // `cbz x0,+0xC / original first instruction / b back / ret`. On device this=0 still reaches the body,
    // so the cbz seems not to take effect -- print the four words actually sitting in the cave.
    if (sig == SIGSEGV || sig == SIGBUS) {
        for (int k6 = 0; k6 < g_nmod; k6++) {
            if (strcmp(g_mod[k6].name, "main")) continue;
            uint64_t cbase = (uint64_t)(uintptr_t)g_mod[k6].base;
            for (int i = 0; i < 10; i++) {
                uint64_t a = cbase + 0x2404a70 + (uint64_t)0x40 * (uint64_t)i;   // np367：stride 已改 0x40
                uint32_t w[8] = {0};
                int ok = 1;
                for (int q = 0; q < 8; q++) {
                    struct iovec li = { &w[q], 4 }, ri = { (void *)(uintptr_t)(a + 4 * (uint64_t)q), 4 };
                    if (process_vm_readv(getpid(), &li, 1, &ri, 1, 0) != 4) { ok = 0; break; }
                }
                if (i == 0) fprintf(stderr, "[dm] np368 桩实况（9 个，stride 0x40，各 8 字）：\n");
                if (ok) fprintf(stderr, "   桩%d %#llx: %08x %08x %08x %08x %08x %08x %08x %08x%s\n", i,
                                (unsigned long long)a, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
                                w[0] ? "" : "   ← 空槽（未安装）");
                else    fprintf(stderr, "   桩%d %#llx: 读不到\n", i, (unsigned long long)a);
            }
            break;
        }
    }
    // np361：**现场字节 vs 文件字节**。旧现场里出现过 `入口 45f9c8=147e9432 45f9cc=a9017bfd`，
    // 而文件里 0x45f9cc 是 `stp x29,x30,[sp,#0x10]`（a9017bfd 对应的是 `[sp,#-0x10]!`）——
    // 内存与文件不一致 ⇒ 装载期补丁改过这段代码，这正是"静态分析（callers.py/cf_reach）一直和运行时
    // 对不上"的原因。崩点前后各打 8 条原始字，离线与 main.patched.bin 逐字比对即可定谳。
    // np361: live bytes vs file bytes around the fault. An old dump showed 0x45f9cc holding a different
    // instruction than the file, i.e. a load-time patch rewrote this code -- which is why static analysis
    // (callers.py / cf_reach) kept disagreeing with runtime. Dump 8 raw words before and after pc; compare
    // with main.patched.bin offline to settle it.
    if ((sig == SIGSEGV || sig == SIGBUS) && (uint64_t)UC_PC(uc) > 0x10000) {
        uint64_t pcx = (uint64_t)UC_PC(uc) & ~3ull, base = pcx - 0x20;
        fprintf(stderr, "[dm] np361 现场原始字（%#llx 起 16 条，pc=%s）：\n",
                (unsigned long long)base, fmt_off((uint64_t)UC_PC(uc)));
        for (int i = 0; i < 16; i++) {
            uint32_t w = 0;
            struct iovec l = { &w, 4 }, r = { (void *)(uintptr_t)(base + (uint64_t)i * 4), 4 };
            if (process_vm_readv(getpid(), &l, 1, &r, 1, 0) != 4) { fprintf(stderr, "   +%#04x 读不到\n", i * 4); continue; }
            fprintf(stderr, "   +%#04x %08x%s\n", i * 4, w, base + (uint64_t)i * 4 == pcx ? "   ← pc" : "");
        }
    }
    // np354：**直接打印帧里的返回地址**。np330 的 `?+0` 有两种含义（值为 0 / 不在任何模块内），
    // 而构造函数 0x45f9c8 的 5 个调用点全是 `bl`，返回地址不可能是 0 ⇒ 大概率是"落在宿主里"。
    // 这里把 [sp+0x10]（保存的 x29）与 [sp+0x18]（保存的 x30 = 调用者返回地址）原样打出来，不再靠推断。
    // np354: print the frame's saved return address directly. np330's "?+0" is ambiguous (zero vs not-in-any-module),
    // and all 5 call sites of ctor 0x45f9c8 are `bl`, so the return address cannot be zero -- it is likely a host
    // address. Dump [sp+0x10] (saved x29) and [sp+0x18] (saved x30 = caller return address) verbatim.
    if (sig == SIGSEGV || sig == SIGBUS) {
        uint64_t sp = (uint64_t)UC_SP(uc), v10 = 0, v18 = 0;
        struct iovec l1 = { &v10, 8 }, r1 = { (void *)(uintptr_t)(sp + 0x10), 8 };
        struct iovec l2 = { &v18, 8 }, r2 = { (void *)(uintptr_t)(sp + 0x18), 8 };
        int o1 = process_vm_readv(getpid(), &l1, 1, &r1, 1, 0) == 8;
        int o2 = process_vm_readv(getpid(), &l2, 1, &r2, 1, 0) == 8;
        fprintf(stderr, "[dm] np354 帧 [sp+0x10]=%s  [sp+0x18](返回地址)=%s\n",
                o1 ? fmt_off(v10) : "读不到", o2 ? fmt_off(v18) : "读不到");
        if (o2 && v18) {
            int inhost = 1;
            for (int k = 0; k < g_nmod; k++) {
                uint64_t lo = (uint64_t)(uintptr_t)g_mod[k].base, hi = lo + g_mod[k].size;
                if (v18 >= lo && v18 < hi) { inhost = 0; break; }
            }
            if (inhost) {
                fprintf(stderr, "   ⇒ 返回地址 %#llx 不在任何客体模块内（宿主/栈/堆）\n",
                        (unsigned long long)v18);
                // np355：把包含这个地址的 /proc/self/maps 行原样打出来 —— 直接点名是哪个库
                // np355: print the /proc/self/maps line containing that address, naming the library directly
                FILE *mp3 = fopen("/proc/self/maps", "r");
                if (mp3) {
                    char ln[512];
                    while (fgets(ln, sizeof ln, mp3)) {
                        unsigned long long lo3 = 0, hi3 = 0;
                        if (sscanf(ln, "%llx-%llx", &lo3, &hi3) == 2 && v18 >= lo3 && v18 < hi3) {
                            size_t n3 = strlen(ln);
                            while (n3 && (ln[n3 - 1] == '\n' || ln[n3 - 1] == '\r')) ln[--n3] = 0;
                            fprintf(stderr, "   map 行: %s\n", ln);
                            break;
                        }
                    }
                    fclose(mp3);
                }
            }
        }
    }
    // np352：**当前是哪个客体线程**。构造函数被以 this=NULL 调用、且其栈里保存的 x30=0
    // （线程入口特征）⇒ 必须先确认线程身份与它的 arg 是不是本来就是 0，再谈"谁传的空 this"。
    // np352: which guest thread are we on? The ctor is entered with this=NULL and its saved x30 is 0
    // (the thread-entry signature), so identify the thread and its arg before blaming anyone.
    if (sig == SIGSEGV || sig == SIGBUS) {
        extern int dm_thread_self_info(uint64_t *, uint64_t *, uint64_t *, uint32_t *);
        uint64_t te = 0, ta = 0, ts = 0; uint32_t th = 0;
        if (dm_thread_self_info(&te, &ta, &ts, &th))
            fprintf(stderr, "[dm] np352 当前客体线程 h=%#x entry=%s arg=%#llx 栈顶=%#llx\n",
                    th, fmt_off(te), (unsigned long long)ta, (unsigned long long)ts);
        else
            fprintf(stderr, "[dm] np352 当前不是客体线程（宿主线程里崩的）\n");
        // np373：把全部客体线程的 entry/arg 与 ThreadType 结构打出来 —— 找"真正的每线程入口"
        { void dm_thread_dump_all(void); dm_thread_dump_all(); }
    }
    // np349：**pc 落到模块外**（0 或 0xa00000002）⇒ 不是"某条指令崩"，而是"ret/br 去了垃圾地址"。
    // 两个变体 pc=lr=0 与 pc=lr=0xa00000002 都符合 `ret`（x30 被写成垃圾）⇒ 首要嫌疑是**栈被写坏/清零**。
    // 现场佐证：np330 的 16 帧全是 ?+0，即 [sp+0x10]/[sp+0x18] 都是 0 —— 帧链本身就断了。
    // 这里做两件事：① sp 附近原样 dump（看是不是一片 0）；② 从 sp 向上扫 0x4000，把落在客体模块内的
    // qword 全列出来 —— 顶层帧坏了也能重建真实调用链。全部用 process_vm_readv，不会二次崩。
    // np349: pc outside every module (0 or 0xa00000002) means "ret/br went to garbage", not "an instruction faulted".
    // Both variants (pc=lr=0, pc=lr=0xa00000002) fit `ret` with a corrupted x30, so a smashed/zeroed stack is the prime suspect
    // (np330's 16 frames are all ?+0, i.e. [sp+0x10]/[sp+0x18] are 0 -- the frame chain is already broken).
    if (sig == SIGSEGV || sig == SIGBUS) {
        uint64_t pcn = (uint64_t)UC_PC(uc); int pcin = 0;
        for (int k = 0; k < g_nmod; k++) {
            uint64_t lo = (uint64_t)(uintptr_t)g_mod[k].base, hi = lo + g_mod[k].size;
            if (pcn >= lo && pcn < hi) { pcin = 1; break; }
        }
        if (!pcin) {
            uint64_t sp = (uint64_t)UC_SP(uc);
            fprintf(stderr, "[dm] np349 pc=%#llx 不在任何模块内 —— sp 原样（sp=%#llx）\n",
                    (unsigned long long)pcn, (unsigned long long)sp);
            int zeros = 0;
            for (int k = -4; k < 9; k++) {
                uint64_t v = 0;
                struct iovec l = { &v, 8 }, r = { (void *)(uintptr_t)(sp + (uint64_t)(k * 8)), 8 };
                if (process_vm_readv(getpid(), &l, 1, &r, 1, 0) == 8) {
                    if (!v) zeros++;
                    fprintf(stderr, "   [sp%+d] = %#llx\n", k * 8, (unsigned long long)v);
                } else fprintf(stderr, "   [sp%+d] = （读不到）\n", k * 8);
            }
            fprintf(stderr, "[dm] np349 sp 附近 13 个字里 %d 个是 0%s\n", zeros,
                    zeros >= 10 ? "  ⇒ **栈被清零**" : "");
            fprintf(stderr, "[dm] np349 栈内候选返回地址（sp..sp+0x4000，只列落在模块内的，最多 20 条）：\n");
            int shown = 0;
            for (uint64_t a = sp; a < sp + 0x4000 && shown < 20; a += 8) {
                uint64_t v = 0;
                struct iovec l = { &v, 8 }, r = { (void *)(uintptr_t)a, 8 };
                if (process_vm_readv(getpid(), &l, 1, &r, 1, 0) != 8) break;
                if (!v) continue;
                for (int k = 0; k < g_nmod; k++) {
                    uint64_t lo = (uint64_t)(uintptr_t)g_mod[k].base, hi = lo + g_mod[k].size;
                    if (v >= lo && v < hi) {
                        fprintf(stderr, "   +%#llx: %s\n", (unsigned long long)(a - sp), fmt_off(v));
                        shown++; break;
                    }
                }
            }
            if (!shown) fprintf(stderr, "   （一个都没有 —— 栈里没有任何客体代码地址）\n");
        }
    }
    // np350：**宿主 .so 的地址不该出现在客体寄存器/栈里**。
    // 真机现场：x8=0x5dd1f24150、x9=0x5dd0272000、[x19+0xf8]=[x19+0x18]=0x5dd1f56f00 —— 全是 libdms2.so 内的地址。
    // 说明某个 HLE 返回值或写进客体对象的字段把宿主指针递进了客体；客体若拿它当函数指针/对象指针用，就是
    // "ret 到垃圾地址"（pc=lr=0 / 0xa00000002 / 0x100000000 三个变体的共同来源之一）。
    // 这里把"落在**可执行映射**、又**不在任何客体模块内**"的寄存器与栈槽全部列出来，一次看清污染面。
    // np350: host .so addresses must not appear in guest registers/stack. The last device run had x8/x9/[x19+0xf8]
    // all inside libdms2.so. Classify every register and stack slot that lands in an executable mapping which is
    // not a guest module -- that is the "host pointer leaked into the guest" surface.
    if (sig == SIGSEGV || sig == SIGBUS) {
        static char mbuf[65536];
        FILE *mp2 = fopen("/proc/self/maps", "r");
        size_t mn = mp2 ? fread(mbuf, 1, sizeof mbuf - 1, mp2) : 0;
        if (mp2) fclose(mp2);
        if (mn) {
            mbuf[mn] = 0;
            fprintf(stderr, "[dm] np350 宿主代码地址污染检查（可执行映射、且不在客体模块内）\n");
            int shown = 0;
            for (int r = 0; r <= 28; r++) {
                uint64_t v = (uint64_t)UC_X(uc, r);
                if (!v) continue;
                char *line = mbuf;
                while (line && *line) {
                    char *nl = strchr(line, '\n');
                    unsigned long long lo = strtoull(line, NULL, 16);
                    char *dash = strchr(line, '-');
                    if (!dash) break;
                    unsigned long long hi = strtoull(dash + 1, NULL, 16);
                    char *sp2 = strchr(line, ' ');
                    int exec = (sp2 && sp2[1] && sp2[3] == 'x');   // "lo-hi perms ..."，perms[3] 是 x 位
                    if (exec && v >= lo && v < hi) {
                        int guest = 0;
                        for (int k5 = 0; k5 < g_nmod; k5++) {
                            uint64_t gl = (uint64_t)(uintptr_t)g_mod[k5].base, gh = gl + g_mod[k5].size;
                            if (v >= gl && v < gh) { guest = 1; break; }
                        }
                        if (!guest) {
                            fprintf(stderr, "   x%-2d = %#llx  ← 宿主可执行映射 %#llx-%#llx\n",
                                    r, (unsigned long long)v, lo, hi);
                            shown++;
                        }
                        break;
                    }
                    line = nl ? nl + 1 : NULL;
                }
            }
            if (!shown) fprintf(stderr, "   寄存器里没有宿主代码地址\n");
        }
    }
    // np348b：**vtable 探针** —— 崩点紧接 EndRecording 之后，客体在 0x9fd1ac 做
    //   `ldr x8,[x19]; ldr x8,[x8,#0xf8]; blr x8`（虚调用）。x19 就是那个对象；
    // 把 [x19]（vtable 指针）与几个关键槽读出来，就能判断是"虚表槽是垃圾"还是"对象本身被写坏"。
    // 安全读用 process_vm_readv（对不可读地址返回错误，**不会二次崩**）。
    if ((sig == SIGSEGV || sig == SIGBUS) && UC_X(uc, 19) && getenv("DM_NO_VT") == NULL) {
        uint64_t o = (uint64_t)UC_X(uc, 19);
        uint64_t vals[8] = {0}; int ok[8] = {0};
        const long offs[8] = {0, 0xf8, 0x18, 0x98, 0x1b0, 0x1d0, 0x208, 0x10};
        for (int i = 0; i < 8; i++) {
            struct iovec l = { &vals[i], 8 }, r = { (void *)(uintptr_t)(o + offs[i]), 8 };
            ok[i] = process_vm_readv(getpid(), &l, 1, &r, 1, 0) == 8;
        }
        fprintf(stderr, "[dm] np348b x19=%#llx  vtable=[x19]=%#llx%s\n", (unsigned long long)o,
                (unsigned long long)vals[0], ok[0] ? "" : "（读不到）");
        for (int i = 1; i < 8; i++)
            fprintf(stderr, "   [x19+%#lx] = %#llx%s\n", (unsigned long)offs[i],
                    (unsigned long long)vals[i], ok[i] ? "" : "（读不到）");
        // 虚表本身的前 8 个槽（看是不是真虚表：应当落在某模块映像里）
        if (ok[0] && vals[0]) {
            for (int i = 0; i < 8; i++) {
                uint64_t v = 0;
                struct iovec l = { &v, 8 }, r = { (void *)(uintptr_t)(vals[0] + (uint64_t)i * 8), 8 };
                if (process_vm_readv(getpid(), &l, 1, &r, 1, 0) != 8) break;
                const char *nm = NULL; uint64_t off = 0;
                for (int k4 = 0; k4 < g_nmod; k4++) {
                    uint64_t l2 = (uint64_t)(uintptr_t)g_mod[k4].base, h2 = l2 + g_mod[k4].size;
                    if (v >= l2 && v < h2) { nm = g_mod[k4].name; off = v - l2; break; }
                }
                fprintf(stderr, "   vtable[%d] = %#llx  %s\n", i, (unsigned long long)v,
                        nm ? fmt_off(v) : "（不在模块内）");
                (void)off;
            }
        }
    }
    // np348：定点内存扫描 —— 崩在"数据当地址"（pc/addr 是个小数值）时，找出**谁持有这个值**。
    // 现场 0xa00000002 = 两个 int（低 2 高 10）的 8 字节小端 = 0x0000000A00000002 ⇒ 精确 8 字节搜。
    // 安全：先读 /proc/self/maps，只扫**可读**区（PROT_NONE 页读下去会在 handler 里二次崩）。
    if ((sig == SIGSEGV || sig == SIGBUS) && (uint64_t)(uintptr_t)si->si_addr &&
        (uint64_t)(uintptr_t)si->si_addr < 0x100000000ull && getenv("DM_NO_SCAN") == NULL) {
        uint64_t bad = (uint64_t)(uintptr_t)si->si_addr;
        FILE *mp = fopen("/proc/self/maps", "r");
        int hits = 0;
        unsigned long long scanned = 0;
        fprintf(stderr, "[dm] np348 内存扫描：谁持有 %#llx（只扫可读区，最多报 8 处）\n", (unsigned long long)bad);
        if (mp) {
            char line[512];
            while (fgets(line, sizeof line, mp) && hits < 8) {
                unsigned long long lo = 0, hi = 0;
                char perm[8] = {0};
                if (sscanf(line, "%llx-%llx %7s", &lo, &hi, perm) != 3) continue;
                if (perm[0] != 'r') continue;
                if (hi - lo > (256ull << 20)) hi = lo + (256ull << 20);   // 单段最多扫 256MB，防超时
                // np353：**绝不解引用** —— 文件映射被截断时直接读会 SIGBUS 把处理器自己打死
                // （2026-10-26 真机：np348 扫描让 handler 崩在宿主 .so+0xb4920，sig=7，整轮读数作废）。
                // 改用 process_vm_readv 按 64 KB 分块读，读不全就跳过这一块。
                // np353: never dereference -- reading a truncated file mapping raises SIGBUS and kills the handler
                // itself (device run 2026-10-26: the np348 scan crashed the handler at host .so+0xb4920, sig=7).
                // Read in 64 KB chunks through process_vm_readv and skip any chunk that cannot be read.
                static uint8_t vbuf[65536];
                for (unsigned long long a0 = lo & ~7ull; a0 < hi && hits < 8; a0 += sizeof vbuf) {
                    size_t want = (size_t)((hi - a0) < sizeof vbuf ? (hi - a0) : sizeof vbuf);
                    struct iovec li = { vbuf, want }, ri = { (void *)(uintptr_t)a0, want };
                    ssize_t got = process_vm_readv(getpid(), &li, 1, &ri, 1, 0);
                    if (got <= 0) continue;
                    for (size_t o2 = 0; o2 + 8 <= (size_t)got && hits < 8; o2 += 8) {
                        uint64_t v; memcpy(&v, vbuf + o2, 8);
                        if (v != bad) continue;
                        uint64_t a = a0 + o2;
                        const char *nm = NULL; uint64_t off = 0;
                        for (int k3 = 0; k3 < g_nmod; k3++) {
                            uint64_t l2 = (uint64_t)(uintptr_t)g_mod[k3].base, h2 = l2 + g_mod[k3].size;
                            if (a >= l2 && a < h2) { nm = g_mod[k3].name; off = a - l2; break; }
                        }
                        if (nm) fprintf(stderr, "   [%d] %s+%#llx\n", hits, nm, (unsigned long long)off);
                        else    fprintf(stderr, "   [%d] %#llx（宿主/堆区，map 段 %#llx-%#llx）\n",
                                        hits, (unsigned long long)a, lo, hi);
                        hits++;
                    }
                }
                scanned += (hi > lo ? hi - lo : 0);
            }
            fclose(mp);
        }
        fprintf(stderr, "   （扫了 %.0f MB，命中 %d 处%s）\n", scanned / 1048576.0, hits,
                hits ? "" : " —— 值不在任何可读内存里，说明它是算出来的");
    }
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
    // np539：把安装结果打出来。np538 的现象是"崩了却一行现场都没有"，第一个要排除的假设就是
    // "sigaction 静默失败、我们的处理器根本没进链"（系统 sigchain 日志里只出现两个系统处理器）。
    // np538's symptom was a crash with no site line at all; the first hypothesis to rule out is that sigaction
    // silently failed and our handler never entered the chain (the system sigchain log showed only two handlers).
    {
        struct sigaction chk; int rc1 = sigaction(SIGSEGV, 0, &chk);
        fprintf(stderr, "[dm] crash 安装 SIGSEGV rc=%d 当前处理器=%p fd=%d\n",
                rc1, (void *)chk.sa_sigaction, g_crash_fd);
    }
}

// ---- np386：断言钩子 --------------------------------------------------------------
// 零售 SDK 把断言文本裁空了（expr 是空串，见 sdk+0x5c77ef），`svc Break` 也不带载荷 ⇒
// 从日志里完全看不出"哪个断言"。唯一还带着"是谁"的信息在 `nn::diag::detail::AbortImpl` 的参数里：
//   AbortImpl(expr, file, func, line, const Result* r, const UserExceptionInfo*, const char* fmt, ...)
// 所以把这个函数入口改成跳到宿主：先打 file/func/line/Result，再把原指令写回去、按原路径继续自杀。
// 这样每一轮真机都能直接读到断言位置，不用再靠 fp 链猜（np384/385 就被 fp 链坑过两次）。
// ---- np386: assertion hook -------------------------------------------------------
// The retail SDK strips the assertion text (expr is an empty string, see sdk+0x5c77ef) and svc Break
// carries no payload, so the log cannot say WHICH assertion fired. The only remaining "who" is in the
// arguments of nn::diag::detail::AbortImpl(expr, file, func, line, Result*, UserExceptionInfo*, fmt, ...).
// So its entry is redirected to the host: print file/func/line/Result, then put the original instructions
// back and continue on the original path (which aborts by itself). From now on every device round can
// read the assertion site directly instead of guessing from the frame-pointer chain (np384/385 were both
// misled by that chain).
static uint8_t *g_abort_site[4];     // 断言入口可以有多处（sdk 里至少两个：0x3442c4 与 0x443f8） | there can be several assert entries in the sdk
static uint32_t g_abort_save[4][4];
static int g_nabort;

static int guest_readable(const void *p, size_t n) {   // 只认落在已装载模块映像里的指针 | only pointers inside a loaded module image
    uintptr_t a = (uintptr_t)p;
    if (!a) return 0;
    for (int i = 0; i < g_nmod; i++) {
        uintptr_t lo = (uintptr_t)g_mod[i].base, hi = lo + g_mod[i].size;
        if (a >= lo && a + n <= hi) return 1;
    }
    return 0;
}
static void guest_str(const void *p, char *out, size_t cap) {   // 断言串都是 ASCII；不可读就给空 | assertion strings are ASCII; unreadable -> empty
    out[0] = 0; if (!p || cap < 2) return;
    for (size_t i = 0; i + 1 < cap; i++) {
        if (!guest_readable((const char *)p + i, 1)) break;
        char c = ((const char *)p)[i];
        if (!c) break;
        out[i] = (c >= 0x20 && c < 0x7f) ? c : '.';
        out[i + 1] = 0;
    }
}

static void dm_abort_hook(const char *expr, const char *file, const char *func, int line,
                          const void *res, const void *exc, const char *fmt, ...) {
    char fe[200], ff[200], fn[200], fm[200];
    guest_str(expr, fe, sizeof fe); guest_str(file, ff, sizeof ff);
    guest_str(func, fn, sizeof fn); guest_str(fmt, fm, sizeof fm);
    unsigned rv = (res && guest_readable(res, 4)) ? *(const unsigned *)res : 0;
    // 调用者地址才是"哪个断言"的真答案：断言入口是共用尾块，fp 链又不可靠（np384/385 被坑过两次）。
    // The caller address IS the answer to "which assertion": the assert entries are shared tail blocks and the
    // frame-pointer chain is unreliable (np384/385 were misled twice).
    uint64_t lr = (uint64_t)(uintptr_t)__builtin_return_address(0);
    module_t *lm = module_of((void *)lr);
    char where[64] = "?";
    if (lm) snprintf(where, sizeof where, "%s+%#llx", lm->name, (unsigned long long)(lr - (uint64_t)lm->base));
    LOG("【断言】调用者 %s  %s  (%s:%d)  Result=%#x%s%s", where, fn[0] ? fn : "?", ff[0] ? ff : "?", line, rv,
        fe[0] ? "  expr=" : "", fe);
    // np402：**在断言这一刻**扫栈。Break 时链上已经压了 AbortImpl→Break→信号 好几层，FP 早被跳板搅乱；
    // 这里我们是客体直接调用进来的，本帧第一个槽就是客体的 x29，扫它最干净。
    // np402: scan the stack AT the assertion. By the time Break runs, several layers (AbortImpl -> Break -> the
    // signal) are on top and the FP is already scrambled by the trampolines; here the guest called us directly, so
    // slot 0 of our frame is the guest's x29 -- the cleanest point to scan from.
    { void dm_backtrace_scan(uint64_t); dm_backtrace_scan(((uint64_t *)__builtin_frame_address(0))[0]); }
    if (fm[0]) LOG("        格式：%s", fm);
    // 软断言：还原原指令后**直接返回**，让调用方继续走（断言在 SDK 里是"不该发生"的路径，
    // 但移植期我们宁愿继续跑、把后面还有多少关卡一次看清；每一条都记了日志，不会静默）。
    // 不调原函数的原因：0x3442c4 的参数约定我们并不知道（真机实测传进去就 SIGSEGV，addr 是 x0 里的客体地址）。
    // Soft assert: restore the original instructions and RETURN, letting the caller continue. In the SDK an
    // assertion is a "should never happen" path, but during a port we would rather keep running and see how many
    // more gates are left; every one is logged, so nothing is silent. The original is not called because its
    // calling convention is unknown (on device it segfaulted immediately, addr = the guest pointer in x0).
    // 不还原、也不调原函数：入口一直保持改跳，这样**每一条**断言都被接住（还原只会在第一条之后失效，
    // 第二条就按原路径自杀）。要恢复致命行为，删掉 main.c 里的 dm_assert_hook_install() 即可。
    // Neither restore nor call the original: the entry stays redirected so EVERY assertion is caught (restoring
    // would only work for the first one, the second would abort as before). To get the fatal behaviour back,
    // drop the dm_assert_hook_install() call in main.c.
}

void dm_assert_hook_install(void) {
    if (g_nabort) return;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "sdk")) {
        // np388：钩 0x3442c4 而不是 0x442b4 —— 断言块里 `bl` 的是前者（见 sdk+0x237100 的反汇编：
        // 断言分支末尾 `bl #0x3442c4`，参数 x0..x2 = expr/file/func 三个空串、w3 = 0、x4 = &Result）。
        // 0x442b4 那次钩子从未被调用，说明它不是断言主路径。
        // np388: hook 0x3442c4, not 0x442b4 -- the assert blocks `bl` the former (see the disassembly of
        // sdk+0x237100: the assert branch ends with `bl #0x3442c4`, x0..x2 = the three empty strings,
        // w3 = 0, x4 = &Result). The 0x442b4 hook was never called, so that is not the main assert path.
        // 两处都钩：0x3442c4 是断言块直接 bl 的那个（np390 拿到 MakeLanguageCode 就是靠它），
        // 0x443f8 是 nn::diag::detail::OnAssertionFailure（np391 那轮的 bt 里出现、没被接住所以仍 Break）。
        // Both are hooked: 0x3442c4 is what the assert blocks `bl` (it gave us MakeLanguageCode in np390), and
        // 0x443f8 is nn::diag::detail::OnAssertionFailure (seen in the np391 backtrace; unhooked, so it still broke).
        // 只钩 0x3442c4。0x443f8（OnAssertionFailure）np392 试过：它是**重试循环里**的一环，
        // 返回不退出 ⇒ 同一断言每 12 ms 刷一次（一场 2994 行），流程反而更早卡死。
        // Only 0x3442c4 is hooked. np392 tried 0x443f8 (OnAssertionFailure) too: it sits inside a retry loop,
        // so returning instead of aborting made the same assertion fire every 12 ms (2994 lines) and blocked
        // progress earlier.
        static const uint64_t kSite[] = {0x3442c4};
        for (unsigned k = 0; k < sizeof kSite / sizeof *kSite && g_nabort < 4; k++) {
            uint8_t *at = g_mod[i].base + kSite[k];
            memcpy(g_abort_save[g_nabort], at, 16);
            g_abort_site[g_nabort] = at;
            char why[48]; snprintf(why, sizeof why, "断言钩子 sdk+%#llx", (unsigned long long)kSite[k]);
            dm_guest_redirect(at, (void *)dm_abort_hook, g_abort_save[g_nabort][0], why);
            g_nabort++;
        }
        return;
    }
    LOG("【黄】断言钩子：没有 sdk 模块，装不上");
}

// ---- np393：语言码补丁（行为补丁，不是仪器） --------------------------------------
// 真机定谳（np390/392）：`nn::settings::detail::MakeLanguageCode(&code, value)` 内部要去查 set 服务的语言表，
// 我们只有假句柄 ⇒ 它返回错误 Result ⇒ 上层断言（而且那个断言在重试循环里，软断言会变成风暴）。
// 移植期只认一个语言，所以直接接管它：把宿主自己的 LanguageCode（与 nn::oe::GetDesiredLanguage 同一个值）
// 写进 *out，返回 0（成功）。这是**行为补丁**——出画面后应改成按 profile 配语言。
// ---- np393: language-code shim (a behaviour patch, not instrumentation) --------------
// Settled on device (np390/392): nn::settings::detail::MakeLanguageCode(&code, value) looks the language table
// up through the `set` service, which we only have as a fake handle, so it returns an error Result and the
// caller asserts (and that assert sits in a retry loop, so a soft assert turns into a storm). A port needs only
// one language, so it is taken over: write the host's own LanguageCode (the same value nn::oe::GetDesiredLanguage
// returns) into *out and return 0. This is a BEHAVIOUR PATCH -- once a frame shows, make it profile-driven.
static uint32_t dm_make_lang_code(void *out, int value) {   // LanguageCode = 8 字节定长串（"zh-Hans" / "en-US"…） | a fixed 8-byte string
    (void)value;
    if (!out) return 0xDA690400u;              // 与 SDK 对空指针的回法一致 | same as the SDK's null-pointer result
    static int n; if (n++ < 4) LOG("语言码补丁被调用（第 %d 次）→ 写 zh-Hans，返回 0", n);
    const char *e = getenv("DM_LANG");
    char code[8] = {0};
    strncpy(code, e ? e : "zh-Hans", sizeof code - 1);
    memcpy(out, code, sizeof code);
    return 0;
}
void dm_lang_shim_install(void) {
    static int done;
    if (done) return;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "sdk")) {
        uint8_t *at = g_mod[i].base + 0x192074;   // nn::settings::detail::MakeLanguageCode（sdk 导出符号表实测）
        uint32_t first; memcpy(&first, at, 4);
        dm_guest_redirect(at, (void *)dm_make_lang_code, first, "语言码补丁 MakeLanguageCode");
        done = 1;
        return;
    }
}

// ---- np398：判决实验：nn::ro::detail::Bind 是不是那个断言 ----------------------------------
// 栈扫描与 FP 链都指到 sdk+0x8dd58（nn::ro::detail::Bind），但它的反汇编里没有断言块。
// 所以直接做判决：把它改跳成"返回 0"，看 Break 消不消失。消失 ⇒ 断言就在这条链上（再想怎么正确实现）；
// 不消失 ⇒ 两套回溯都被同一个假象骗了，得换手段（PC 环形缓冲）。
// ---- np398: decisive experiment: is nn::ro::detail::Bind the assertion? ----------------------
// Both the stack scan and the FP chain point at sdk+0x8dd58 (nn::ro::detail::Bind), yet its disassembly
// contains no assert block. So make it a decisive test: redirect it to "return 0" and see whether the Break
// goes away. If it does, the assertion is on this path (then implement it properly); if not, both backtracers
// were fooled by the same artefact and a PC ring buffer is needed instead.
static uint64_t dm_ro_bind(void *mod, unsigned idx) {
    static int n;
    if (n++ < 8) LOG("nn::ro::detail::Bind(%p, %u) → 返回 0（判决实验）", mod, idx);
    return 0;
}
void dm_ro_bind_probe_install(void) {
    static int done;
    if (done) return;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "sdk")) {
        uint8_t *at = g_mod[i].base + 0x8dd58;
        uint32_t first; memcpy(&first, at, 4);
        dm_guest_redirect(at, (void *)dm_ro_bind, first, "判决实验 ro::Bind");
        done = 1;
        return;
    }
}
