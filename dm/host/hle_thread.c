// Horizon 内核的线程与同步 SVC，原生实现：每个客体线程 = 一个宿主 pthread，跑在 SDK 给的客体栈上。
// SDK 的 nn::os 原样运行，只把它下面的「内核」换成宿主：
//   线程   0x08 CreateThread 0x09 StartThread 0x0A ExitThread 0x0B SleepThread 0x0C/0x0D 优先级 0x0E/0x0F 核掩码
//          0x10 GetCurrentProcessorNumber 0x25 GetThreadId 0x19 CancelSynchronization
//   锁     0x1A ArbitrateLock 0x1B ArbitrateUnlock（标签字 = 持有者句柄 | 0x40000000 有等待者）
//   条件   0x1C WaitProcessWideKeyAtomic 0x1D SignalProcessWideKey
//   地址   0x34 WaitForAddress 0x35 SignalToAddress
//   事件   0x45 CreateEvent 0x11 SignalEvent 0x12 ClearEvent 0x17 ResetSignal 0x18 WaitSynchronization 0x16 CloseHandle
// 语义按公开 Horizon 内核行为（Atmosphère mesosphère 的 KConditionVariable / KAddressArbiter / KSynchronization）。
// 所有等待挂在一把全局锁 G 上，每线程一个条件变量；先求正确，性能后量。
// Thread and synchronization SVCs of the Horizon kernel, implemented natively: each guest thread = one host pthread, running on the guest stack given by the SDK.
// The SDK's nn::os runs as is; only the "kernel" underneath it is replaced by the host:
//   Threads    0x08 CreateThread 0x09 StartThread 0x0A ExitThread 0x0B SleepThread 0x0C/0x0D priority 0x0E/0x0F core mask
//              0x10 GetCurrentProcessorNumber 0x25 GetThreadId 0x19 CancelSynchronization
//   Locks      0x1A ArbitrateLock 0x1B ArbitrateUnlock (tag word = owner handle | 0x40000000 has waiters)
//   Condition  0x1C WaitProcessWideKeyAtomic 0x1D SignalProcessWideKey
//   Address    0x34 WaitForAddress 0x35 SignalToAddress
//   Events     0x45 CreateEvent 0x11 SignalEvent 0x12 ClearEvent 0x17 ResetSignal 0x18 WaitSynchronization 0x16 CloseHandle
// Semantics follow the public Horizon kernel behavior (KConditionVariable / KAddressArbiter / KSynchronization of Atmosphère mesosphère).
// All waits hang on one global lock G, with one condition variable per thread; correctness first, performance to be measured later.
#include "dm.h"
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <time.h>
#include <mach/mach.h>
#include <pthread/qos.h>
#include <dlfcn.h>
#include <setjmp.h>   // np562（照上游 955f274）：客体线程退回宿主栈再 pthread_exit | np562 (after upstream 955f274): guest threads exit on the host stack

#define R_TIMEOUT     0xEA01u   // 内核模块 1，描述 117 | Kernel module 1, description 117
#define R_CANCELLED   0xEC01u   // 118
#define R_BAD_HANDLE  0xE401u   // 114
#define R_BAD_STATE   0xFA01u   // 125
#define R_OUT_OF_RES  0xCE01u   // 103
#define WAITERS_BIT   0x40000000u
#define H_CUR_THREAD  0xFFFF8000u

enum { O_FREE, O_THREAD, O_EVENT_R, O_EVENT_W, O_TRANSFER };   // np458：O_TRANSFER = CreateTransferMemory 的句柄
typedef struct event { int signaled; int refs; int auto_clear; } event_t;   // auto_clear：被"假音频时钟"触发的事件，等待消费后自动清除（np469）
typedef struct thr {
    uint32_t handle; uint64_t id;
    uint64_t entry, arg, stack_top; int prio, core; uint64_t mask;
    uint64_t born_ns, exit_ns; uint32_t born_tick, exit_tick;   // 建立 / 退出的时刻与当时的游戏拍（现场导出用：作业工人是不是在卡住前退的） | Time of creation / exit and the game tick at that time (for the state dump: did the job worker exit before the stall)
    int started, exited, cancel;
    pthread_t pt; pthread_cond_t cv;
    // 当前等待：kind 0=无 1=锁 2=条件 3=地址 4=同步对象
    // Current wait: kind 0=none 1=lock 2=condition 3=address 4=synchronization object
    int wkind, woke; uint64_t waddr, wkey; uint32_t wtag; int64_t wres;
    // 自旋中：block_inner 放开 G 轮询 woke 的那段（持 G 置位、清零）。wake 见到它就不发信号
    // Spinning: the stretch where block_inner polls woke with G released (set and cleared under G). wake sends no signal then
    int spinning;
    uint64_t wseq;             // 登记等待的先后序号：同优先级先到先醒（与 Horizon 内核 KThread 等待树同序） | Sequence number of wait registration: at equal priority first come, first woken (same order as the Horizon kernel's KThread wait tree)
    struct thr *next;          // 等待链 | Wait chain
    struct thr *all;           // 全部线程链（现场导出用） | All-threads chain (for the state dump)
    int last_svc; uint64_t last_lr;
    int closed;                // 句柄已关：与 exited 同时成立才释放对象（Horizon：线程对象随最后一个引用消失） | Handle closed: the object is freed only when this holds together with exited (Horizon: the thread object goes away with the last reference)
    uint64_t nsvc;             // SVC 次数（现场导出用） | SVC count (for the state dump)
    mach_port_t mport;         // 宿主线程端口（取 CPU 时间） | Host thread port (to get CPU time)
    uint32_t qos;
} thr_t;
static thr_t *g_all;

static pthread_mutex_t G = PTHREAD_MUTEX_INITIALIZER;
#define MAX_H 4096
static struct { int type; void *obj; } g_h[MAX_H];
static pthread_key_t g_keythr;
// np562 ★（照上游 955f274，OnePlus 13 实测）：客体线程的「回宿主栈」跳回点。
//   thread_exit 原来在**客体自己的栈**上跑 pthread_exit——它要跑各 TLS 析构、jemalloc 退还线程缓存
//   会整理空闲块、调用很深；客体栈有的只有十几 KB（我们这儿实测 16 KB），必溢出。
//   做法：thread_main 进客体栈**之前**在自己的（1 MiB 宿主）栈上 sigsetjmp，thread_exit 收尾后
//   siglongjmp 回来，pthread_exit 落在宿主栈上。跳回点不放进 thr_t：t 可能在 thread_exit 里就被释放。
//   门（gate）从不换 x18 ⇒ 跳回时恢复的是普通宿主状态，安全。
// np562 (after upstream 955f274, measured on a OnePlus 13): the guest thread's jump point back to the host
//   stack. thread_exit used to run pthread_exit on the guest's own stack -- TLS destructors, jemalloc's tcache
//   flush with its deep call chain -- and guest stacks can be as small as 16 KB (measured here), so it overflows.
//   thread_main sigsetjmps on its own 1 MiB host stack before entering the guest stack; thread_exit siglongjmps
//   back and pthread_exit runs on the host stack. The jump point is not kept in thr_t because t may be freed
//   inside thread_exit. The gate never swaps x18, so the jump restores plain host state.
static pthread_key_t g_keyhome;
static uint64_t g_next_id = 0x51;
static thr_t *g_wait;          // 全部在等的线程（锁 / 条件 / 地址 / 同步） | All waiting threads (lock / condition / address / synchronization)

static uint32_t h_new(int type, void *obj) {     // 调用方持 G | Caller holds G
    for (int i = 1; i < MAX_H; i++) if (!g_h[i].type) { g_h[i].type = type; g_h[i].obj = obj; return 0x1000 + i; }
    die("句柄表满");
}
static void *h_peek(uint32_t h, int type) {      // 只查不报 | Lookup only, no report
    if (h == H_CUR_THREAD && type == O_THREAD) return pthread_getspecific(g_keythr);
    unsigned i = h - 0x1000;
    if (i == 0 || i >= MAX_H || g_h[i].type != type) return NULL;
    return g_h[i].obj;
}
static void *h_get(uint32_t h, int type) {       // 查不到即报（句柄错是真错误） | Reports if not found (a wrong handle is a real error)
    void *o = h_peek(h, type);
    if (!o) LOG("句柄无效 %#x（要类型 %d）", h, type);
    return o;
}
static thr_t *cur(void) {
    thr_t *t = pthread_getspecific(g_keythr);
    if (!t) die("非客体线程调用了线程 SVC");
    return t;
}
// np352：给崩溃处理器用——报出"当前这个客体线程是谁"（entry / arg / 栈顶）。
// 真机现场：构造函数 `main+0x45f9c8` 被以 `this=NULL` 调用，而它栈里保存的 x30 = 0
// （aarch64 上 LR=0 是线程入口的特征）⇒ 先要确认是哪个线程、它的 arg 本来就是不是 0。
// np352: for the crash handler -- report which guest thread we are on (entry / arg / stack top).
// On device the ctor main+0x45f9c8 is entered with this=NULL and its saved x30 is 0 (the aarch64
// thread-entry signature), so we must first know which thread this is and whether its arg was 0.
int dm_thread_self_info(uint64_t *entry, uint64_t *arg, uint64_t *stack_top, uint32_t *handle) {
    thr_t *t = pthread_getspecific(g_keythr);
    if (!t) return 0;
    if (entry) *entry = t->entry;
    if (arg) *arg = t->arg;
    if (stack_top) *stack_top = t->stack_top;
    if (handle) *handle = t->handle;
    return 1;
}
// np373：**把每个客体线程的 entry/arg 连同 arg 指向的结构 dump 出来**。
// 为什么要这个：dm_thread_enter(entry, arg, sp) 把 **arg 放进 x0** 再跳进 entry，
// 而 CreateThread 的 entry 全是 sdk+0x8b2d8（SDK 线程蹦床）——真正的"每线程入口函数"
// 在 ThreadType 结构里，蹦床从 x0 指向的结构取它。门票① 的现象（构造函数 this=arg=0）
// 只有"蹦床取出来的入口就是那个构造函数"才成立，所以直接把结构打出来找那个指针。
// np373: dump every guest thread's entry/arg plus the structure arg points at. dm_thread_enter puts
// **arg into x0** before jumping to entry, and every CreateThread entry is sdk+0x8b2d8 (the SDK thread
// trampoline) -- the real per-thread function lives inside the ThreadType struct at x0. The observed
// "constructor sees this = arg = 0" only holds if the trampoline takes that constructor as the entry.
// np555 ★ 现场导出节拍：每 5 秒把各客体线程"最后一条 svc + 正在等谁"打一遍。
//   为什么要有：卡死时没有崩溃、没有信号，事后什么也看不到；而"谁在等谁"恰恰是唯一能推进的读数。
//   调用方是 16 ms 的假音频时钟线程（该线程在本平台一定被调度，见 se_tick 里的注释）。
// np555: the state-dump tick -- every 5 s print each guest thread's last svc and what it waits on. A stall leaves no
//   crash and no signal, so nothing is visible afterwards; "who waits for whom" is the only reading that moves us.
//   Called from the 16 ms fake audio clock thread, which is definitely scheduled on this platform.
void dm_thread_state_dump(const char *why) {
    // np555：**不拿 G**。上一版持 G 打印，结果一行都没出来 —— 恰恰说明某个线程攥着 G 不放，
    //   这本身就是"全卡住"的直接解释（客体线程全在等 G，谁也进不去）。诊断函数必须能在
    //   "锁已经被谁攥死"的情况下继续工作，所以这里只做无锁快照（字段是普通整数，撕裂读可以接受）。
    //   kind=1（锁等待）时顺手把锁字里的**持有者句柄**解出来：tag = 持有者 | 0x40000000(有等待者)。
    // np555: do NOT take G. The previous version held G and printed nothing -- which itself says some thread is
    //   sitting on G, the direct explanation for "everything is stuck" (all guest threads wait for G). A diagnostic
    //   must keep working while the lock is held hostage, so this is a lock-free snapshot (plain int fields; a torn
    //   read is acceptable). For kind=1 (lock wait) it also decodes the owner handle from the lock word:
    //   tag = owner | 0x40000000 (has waiters).
    int n = 0;
    for (thr_t *t = g_all; t && n < 24; t = t->all, ++n) {
        int k = t->wkind;
        const char *kn = (const char *[]){ "none", "lock", "cond", "addr", "sync" }[(k >= 0 && k < 5) ? k : 0];
        uint64_t own = 0;
        if (k == 1 && t->waddr) { uint32_t tag = 0; memcpy(&tag, (const void *)(uintptr_t)t->waddr, 4); own = tag & ~0x40000000u; }
        fprintf(stderr, "[dm] THR h=%#x st=%s%s svc=%#x n=%llu wait=%s addr=%#llx owner=%#llx lastlr=%#llx\n",
                t->handle, t->started ? "run" : "idle", t->exited ? "/exit" : "",
                t->last_svc, (unsigned long long)t->nsvc, kn,
                (unsigned long long)t->waddr, (unsigned long long)own, (unsigned long long)t->last_lr);
    }
    if (!n) fprintf(stderr, "[dm] THR （还没有客体线程）%s\n", why);
}
void dm_thread_dump_all(void) {
    static char buf[0x300];
    for (thr_t *t = g_all; t; t = t->all) {
        fprintf(stderr, "[dm] np373 线程 h=%#x entry=%#llx arg=%#llx started=%d exited=%d\n",
                t->handle, (unsigned long long)t->entry, (unsigned long long)t->arg,
                t->started, t->exited);
        if (!t->arg) continue;
        struct iovec li = { buf, sizeof buf }, ri = { (void *)(uintptr_t)t->arg, sizeof buf };
        if (process_vm_readv(getpid(), &li, 1, &ri, 1, 0) != (ssize_t)sizeof buf) {
            fprintf(stderr, "     结构读不到\n");
            continue;
        }
        fprintf(stderr, "     +0x00:");
        for (int q = 0; q < 96; q++) fprintf(stderr, " %016llx", (unsigned long long)((uint64_t *)buf)[q]);
        fprintf(stderr, "\n");
    }
}

// ---------- 等待原语（持 G 调用）----------
// ---------- Wait primitives (called holding G) ----------
static void wait_unlink(thr_t *t) {
    for (thr_t **p = &g_wait; *p; p = &(*p)->next) if (*p == t) { *p = t->next; break; }
    t->next = NULL;
}
// 按内核顺序挑等待者：优先级数值小者优先，同优先级先登记者优先（持 G）。kind 1 按锁地址、2 按条件键、3 按地址匹配
// Pick a waiter in kernel order: numerically lower priority first, at equal priority the earlier registered first (holding G). kind 1 matches by lock address, 2 by condition key, 3 by address
static uint64_t g_wseq;
static thr_t *pick(int kind, uint64_t a) {
    thr_t *b = NULL;
    for (thr_t *t = g_wait; t; t = t->next) {
        if (t->wkind != kind || (kind == 2 ? t->wkey : t->waddr) != a) continue;
        if (!b || t->prio < b->prio || (t->prio == b->prio && t->wseq < b->wseq)) b = t;
    }
    return b;
}
// 目标在自旋时不发信号：它见到 woke 就持 G 返回，不进条件等待；安卓 bionic 的 pthread_cond_signal 不论有没有人在等都做一次
// futex 唤醒系统调用。计数给帧 TSV（@wakes、@wake_skips）
// No signal when the target is spinning: it returns under G once it sees woke and never enters the condition wait. Android
// bionic's pthread_cond_signal makes a futex wake system call whether or not anyone waits. The counts go to the frame TSV
// (@wakes, @wake_skips)
static uint64_t g_wakes, g_wake_skips;
void dm_hle_wake_counts(uint64_t *wakes, uint64_t *skips) { *wakes = __atomic_load_n(&g_wakes, __ATOMIC_RELAXED); *skips = __atomic_load_n(&g_wake_skips, __ATOMIC_RELAXED); }
static void wake(thr_t *t, int64_t res) {
    wait_unlink(t); t->wkind = 0; t->wres = res;
    // woke 最后写：block_inner 可能不持 G 轮询它
    // woke last: block_inner may poll it without G
    __atomic_store_n(&t->woke, 1, __ATOMIC_RELEASE);
    __atomic_store_n(&g_wakes, g_wakes + 1, __ATOMIC_RELAXED);
    if (t->spinning) __atomic_store_n(&g_wake_skips, g_wake_skips + 1, __ATOMIC_RELAXED);
    else pthread_cond_signal(&t->cv);
}
// 挂起直到被唤醒或超时（ns<0 无限、0 立即超时）；返回 wres 或 R_TIMEOUT
// Suspend until woken or timed out (ns<0 infinite, 0 immediate timeout); returns wres or R_TIMEOUT
static int64_t block_inner(thr_t *t, int kind, int64_t ns);
static int64_t block(thr_t *t, int kind, int64_t ns) {   // 启动时间线：按等待种类累计（全线程）+ 主线程单列 | Boot timeline: accumulated by wait kind (all threads) + the main thread listed separately
    if (!g_flag.bootprof) return block_inner(t, kind, ns);
    uint64_t t0 = now_ns(); int64_t r = block_inner(t, kind, ns); uint64_t d = now_ns() - t0;
    static const int id[] = {0, PROF_WAIT_LOCK, PROF_WAIT_CV, PROF_WAIT_ADDR, PROF_WAIT_SYNC};
    dm_prof_add(id[kind & 7], d, 0);
    if (t->handle == 0x1001) dm_prof_add(PROF_MAIN_WAIT, d, 0);
    return r;
}
// 主线程的短等待先自旋再睡：D2R 主线程每帧约 45 次等工作线程持有的客体锁，81% 只等 10–25 µs（Redmi K50 Ultra 营地），其中大半是
// futex 睡下、唤醒与核心退出空闲的开销，而不是持锁时间。先照常登记为等待者（交接仍会选中它），放开 G，轮询自己的 woke 至多
// g_spin_ns，再照旧睡。宿主不要求就不开：DM_HLE_SPIN_US（安卓 App 设置）；只有客体主线程自旋。
// 对照：DM_HLE_SPIN_AB=N 时只在偶数个 N 秒里自旋，与引擎的 gfx-ab-toggle 同相位（单调时钟秒数 / N 为偶数 = 开）。
// Short waits of the main thread spin before they sleep. D2R's main thread waits ~45 times a frame on guest locks held by its
// job workers, 81% of them for 10–25 us (Redmi K50 Ultra camp), and most of that is the futex sleep, the wake-up and the
// core's idle exit rather than the holder. The thread registers as a waiter as before (so a handoff still picks it), releases
// G and polls its woke flag for up to g_spin_ns, then sleeps as before. Off unless the host asks: DM_HLE_SPIN_US (the Android
// app sets it); only the guest main thread spins.
// A/B: with DM_HLE_SPIN_AB=N the thread spins only in even N-second periods, in phase with the engine's gfx-ab-toggle
// (monotonic seconds / N even = on).
static uint64_t g_spin_ns; static int g_spin_init; static int64_t g_spin_ab;
static int spin_phase_on(void) {
    if (g_spin_ab <= 0) return 1;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ((ts.tv_sec / g_spin_ab) & 1) == 0;
}
static int64_t block_inner(thr_t *t, int kind, int64_t ns) {
    t->wkind = kind; t->woke = 0; t->next = g_wait; g_wait = t; t->wseq = ++g_wseq;
    // np555 ★ "等了不回来"这件事必须留下证据：卡死时最后一行 WAIT-IN 就是它在等谁。
    //   只记**无限/超长等待**（ns<0 或 >1 秒），每线程每种对象最多记一次，全进程上限 400 行，
    //   所以正常跑起来几乎不产生噪声，一旦卡住却一定有一行（np554 的现场就是"最后一命令 op=19 之后没了"）。
    // np555: a wait that never returns must leave evidence -- on a stall the last WAIT-IN line names the object.
    //   Only infinite/long waits (ns<0 or >1 s) are logged, once per thread and object kind, 400 lines per process,
    //   so normal runs stay quiet but a stall always has its line (np554 ended right after command op=19).
    if (ns < 0 || ns > 1000000000ll) {
        static int n; static struct { uint32_t h; int k; uint64_t a; } seen[64]; static int nseen;
        int dup = 0;
        for (int i = 0; i < nseen; i++) if (seen[i].h == t->handle && seen[i].k == kind && seen[i].a == t->waddr) { dup = 1; break; }
        if (!dup && nseen < 64) seen[nseen++] = (typeof(seen[0])){t->handle, kind, t->waddr};
        if (!dup && n < 400) { ++n;
            LOG("[wait-in] h=%#x kind=%d addr=%#llx key=%#llx ns=%lld", t->handle, kind,
                (unsigned long long)t->waddr, (unsigned long long)t->wkey, (long long)ns); }
        // np557：主线程一旦开始**无限等锁**，立刻把全线程状态打一遍（此刻正持 G，读数一致）。
        //   为什么要在这个时刻打：np556 里连 16 ms 的音频时钟线程都停了 ⇒ 说明有线程攥着 G 不放，
        //   于是"每 5 秒打一次"的节拍根本没机会跑。改成由**等待者**在登记时打，就一定打得出。
        // np557: the moment the main thread starts an infinite lock wait, dump every thread once (we hold G here, so the
        //   snapshot is consistent). Why here: in np556 even the 16 ms audio clock thread stopped, which means some
        //   thread is sitting on G, so a "print every 5 s" tick never gets to run. Printing from the waiter always works.
        if (t->handle == 0x1001 && kind == 1 && ns < 0) {
            // np558：**每 3 秒一次**，不是只打一次。只打一次会撞在正常运行期（主线程每帧都会进无限等锁），
            //   拿到的是"某一帧的普通状态"；卡死要看的恰恰是**同一幅图重复不变**。
            //   ⇒ 判据：日志末尾连续几次 [stall] 的 owner/wait 完全一样 = 真死锁。
            // np558: every 3 s, not once. A single dump lands during normal running (the main thread enters an
            //   infinite lock wait every frame) and shows an ordinary frame; a stall is recognised by the same picture
            //   repeating. Criterion: identical owner/wait in the last several [stall] blocks = a real deadlock.
            static uint64_t next; extern uint64_t now_ns(void);
            if (now_ns() >= next) {
                next = now_ns() + 3000000000ull;
                for (thr_t *q = g_all; q; q = q->all) {
                    int kq = q->wkind; const char *kn = (const char *[]){ "none", "lock", "cond", "addr", "sync" }[(kq >= 0 && kq < 5) ? kq : 0];
                    uint64_t own = 0;
                    if (kq == 1 && q->waddr) { uint32_t tag = 0; memcpy(&tag, (const void *)(uintptr_t)q->waddr, 4); own = tag & ~0x40000000u; }
                    LOG("[stall] h=%#x st=%s%s svc=%#x n=%llu wait=%s addr=%#llx owner=%#llx lr=%#llx",
                        q->handle, q->started ? "run" : "idle", q->exited ? "/exit" : "",
                        q->last_svc, (unsigned long long)q->nsvc, kn,
                        (unsigned long long)q->waddr, (unsigned long long)own, (unsigned long long)q->last_lr);
                }
                // np567：随 [stall] 打 0x1d 的 key 直方图（谁 signal 过哪个条件变量、几次）。
                //   卡死判据：0x100d 等的 key 计数为 0 ⇒ 那个条件变量从未被 signal。
                // np567: dump the 0x1d per-key histogram with [stall]. Stall criterion: the key 0x100d waits
                //   on has count 0 ⇒ that condition variable was never signalled.
                {
                    extern void dm_cv_hist(uint64_t (*out)[2], int *n, int max);
                    uint64_t rows[32][2]; int nr = 0;
                    dm_cv_hist(rows, &nr, 32);
                    for (int i = 0; i < nr; i++)
                        LOG("[cvh] key=%#llx n=%llu by=%#llx", (unsigned long long)rows[i][0],
                            (unsigned long long)rows[i][1] >> 32, (unsigned long long)(rows[i][1] & 0xFFFFFFFFu));
                }
            }
        }
    }
    if (ns == 0) { wait_unlink(t); t->wkind = 0; return R_TIMEOUT; }
    struct timespec dl;
    int det_wait = ns > 0 && dm_detclock_enabled();
    uint64_t guest_deadline = det_wait ? guest_ns() + (uint64_t)ns : 0;
    if (ns > 0 && !det_wait) { clock_gettime(CLOCK_REALTIME, &dl); dl.tv_sec += ns / 1000000000; dl.tv_nsec += ns % 1000000000;
                  if (dl.tv_nsec >= 1000000000) { dl.tv_sec++; dl.tv_nsec -= 1000000000; } }
    if (!g_spin_init) { const char *e = getenv("DM_HLE_SPIN_US"); g_spin_ns = e ? (uint64_t)atoi(e) * 1000 : 0; e = getenv("DM_HLE_SPIN_AB"); g_spin_ab = e ? atoi(e) : 0; g_spin_init = 1; }   // under G
    if (g_spin_ns && t->handle == 0x1001 && (ns < 0 || (uint64_t)ns > g_spin_ns) && spin_phase_on()) {
        t->spinning = 1;
        pthread_mutex_unlock(&G);
        uint64_t end = now_ns() + g_spin_ns;
        while (!__atomic_load_n(&t->woke, __ATOMIC_ACQUIRE) && now_ns() < end) {
#if defined(__aarch64__)
            __asm__ volatile("yield");
#endif
        }
        pthread_mutex_lock(&G);
        t->spinning = 0;
        if (t->woke) return t->wres;   // woken while spinning: wres was written under G before the waker released it
    }
    while (!t->woke) {
        int rc;
        if (det_wait) {
            uint64_t now = guest_ns();
            if (now >= guest_deadline) { wait_unlink(t); t->wkind = 0; return R_TIMEOUT; }
            (void)(guest_deadline - now);
            rc = pthread_cond_wait(&t->cv, &G);
        } else rc = ns > 0 ? pthread_cond_timedwait(&t->cv, &G, &dl) : pthread_cond_wait(&t->cv, &G);
        if (rc == ETIMEDOUT && !t->woke && (!det_wait || guest_ns() >= guest_deadline)) { wait_unlink(t); t->wkind = 0; return R_TIMEOUT; }
    }
    return t->wres;
}

// ---------- 锁 ----------
// ---------- Locks ----------
static inline uint32_t ld32(uint64_t a) { return __atomic_load_n((uint32_t *)a, __ATOMIC_SEQ_CST); }
static inline uint64_t ld64(uint64_t a) { return __atomic_load_n((uint64_t *)a, __ATOMIC_SEQ_CST); }
static inline void st32(uint64_t a, uint32_t v) { __atomic_store_n((uint32_t *)a, v, __ATOMIC_SEQ_CST); }
static inline int cas32(uint64_t a, uint32_t *exp, uint32_t v) { return __atomic_compare_exchange_n((uint32_t *)a, exp, v, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); }

// 释放 addr 处的锁：交给等在它上面的第一个线程，或清零（持 G）
// Release the lock at addr: hand it to the first thread waiting on it, or clear it to zero (holding G)
// np560：0x1c/0x1d 交接取证计数器（DM_CV_PROBE=1 打开，只打前若干条）。
//   注意不能写成静态初始化里读 getenv：.so 加载时 envs 还没设（那是 napi 起宿主之前设的），
//   必须在第一次调用时惰性读。
// np560: counters for 0x1c/0x1d hand-off evidence (on with DM_CV_PROBE=1, first few only). Read the env lazily:
//   at .so load time the env is not set yet (napi sets it just before starting the host).
static int g_cv_probe = -1, g_cv_sig;
// np566：handoff 交接对象打点的计数（复用 cv_enabled() 的开关，DM_CV_PROBE 也打开它）
static unsigned g_nhandoff;
// np560：取号式限流（0x1c 与 0x1d 各自计数）。
//   上一版只在结果行自增、交接行不自增 ⇒ 交接行无限打，np560 一轮 35,876 行，把日志预算与判读全冲垮。
//   env 也必须惰性读：.so 加载时 envs 还没设（napi 起宿主前才设）。
// np560: ticket-based throttling, one counter per op. The previous version incremented only in the result line, not the
//   hand-off line, so hand-offs printed forever (35,876 lines in one run). The env must be read lazily: it is not set at
//   .so load time.
static int cv_enabled(void) {
    if (g_cv_probe < 0) g_cv_probe = getenv("DM_CV_PROBE") ? 0 : (1 << 20);
    return g_cv_probe == 0;
}
static int cv_take(int *n, int cap) { if (!cv_enabled() || *n >= cap) return 0; ++*n; return 1; }

// np567：0x1d 的 per-key 直方图（条件变量 → signal 次数/最后发起者），随 [stall] 打出。
// np567: per-key histogram of 0x1d (condition → signal count / last signaller), dumped with [stall].
static struct { uint64_t key; unsigned n; uint32_t h; } g_cv_hist[32]; static int g_cvh_n;
void dm_cv_hist(uint64_t (*out)[2], int *n, int max) {
    for (int i = 0; i < g_cvh_n && i < max; i++) {
        out[i][0] = g_cv_hist[i].key;
        out[i][1] = ((uint64_t)g_cv_hist[i].n << 32) | g_cv_hist[i].h;
    }
    *n = g_cvh_n < max ? g_cvh_n : max;
}

static void lock_handoff(uint64_t addr) {
    thr_t *w = pick(1, addr); int more = 0;
    if (!w) { st32(addr, 0); return; }
    for (thr_t *t = g_wait; t; t = t->next) if (t != w && t->wkind == 1 && t->waddr == addr) { more = 1; break; }
    // np566：交接对象打点（DM_HANDOFF_PROBE=1）。要回答的问题：0x100d 持锁又在 0x1c 里循环时，
    //   释放锁的 handoff **把锁交给了谁**？如果答案总是"0x100d 自己"（它还挂在同地址的锁等待链上），
    //   那就是它每次被交回、醒来、又进 0x1c —— 饥饿的直接机制，pick 的去重/过滤就是嫌疑。
    // np566: record hand-off targets (DM_HANDOFF_PROBE=1). Question: while 0x100d holds the lock and spins in 0x1c,
    //   WHO receives the lock at release? If it is always 0x100d itself (still linked on the same lock's wait chain),
    //   it gets the lock back every time, wakes, re-enters 0x1c -- the direct starvation mechanism.
    if (cv_enabled() && g_nhandoff < 240) {
        thr_t *cur_t = cur();
        ++g_nhandoff;
        LOG("[ho] rel=%#x -> w=%#x addr=%#llx word=%#x more=%d", cur_t ? cur_t->handle : 0, w->handle,
            (unsigned long long)addr, ld32(addr), more);
    }
    st32(addr, w->wtag | (more ? WAITERS_BIT : 0));
    wake(w, 0);
}
// 让 t 去拿 addr 处的锁：空闲就直接拿到并唤醒，否则置等待者位并转为锁等待（持 G）
// Make t take the lock at addr: if free, it gets it directly and is woken, otherwise set the waiters bit and turn into a lock wait (holding G)
static void lock_acquire_for(thr_t *t, uint64_t addr, uint32_t tag) {
    uint32_t v = ld32(addr);
    for (;;) {
        if (v == 0) { if (cas32(addr, &v, tag)) { wake(t, 0); return; } continue; }
        if (cas32(addr, &v, v | WAITERS_BIT)) break;
    }
    t->wkind = 1; t->waddr = addr; t->wtag = tag;      // 仍在 g_wait 里，改为锁等待 | Still in g_wait, changed to a lock wait
    t->wseq = ++g_wseq;                                  // 内核把它新插进锁持有者的等待队列：同优先级排在已有锁等待者之后 | The kernel inserts it anew into the lock owner's wait queue: at equal priority it goes after the existing lock waiters
}

// ---------- 线程 ----------
// ---------- Threads ----------
extern void dm_thread_enter(uint64_t entry, uint64_t arg, uint64_t sp);   // entry.S
uint32_t (*dm_thread_tick_fn)(void);   // 取当前游戏拍；diag.c 初始化时登记（单测不登记，拍号记 0） | Gets the current game tick; registered when diag.c initializes (unit tests do not register it, the tick number is recorded as 0)
void crash_thread_init(void);
void guest_thread_init(void);   // thread.c：每线程 TSD 槽（门、x18、TLS、TLR）唯一设置点 | thread.c: the single place that sets the per-thread TSD slots (gate, x18, TLS, TLR)
static void *thread_main(void *p) {
    thr_t *t = p;
    pthread_setspecific(g_keythr, t);
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    qos_class_t qc = QOS_CLASS_UNSPECIFIED; int rp = 0;
    pthread_get_qos_class_np(pthread_self(), &qc, &rp); t->qos = (uint32_t)qc;
    t->mport = pthread_mach_thread_np(pthread_self());
    crash_thread_init();
    guest_thread_init();
    { void guest_flush_icache(int); guest_flush_icache(0); }   // np304：本线程执行前清补丁点 icache（不广播，只能各清各的） | np304: flush patched points before this thread runs guest code (no broadcast, so each thread flushes its own)
    pthread_mutex_lock(&G); while (!t->started) pthread_cond_wait(&t->cv, &G); pthread_mutex_unlock(&G);
    {   // np562：记下回宿主栈的跳回点，再进客体栈（见 g_keyhome 的注释） | record the jump home, then enter the guest stack
        sigjmp_buf home;
        pthread_setspecific(g_keyhome, &home);
        if (sigsetjmp(home, 0) == 0) {
            dm_thread_enter(t->entry, t->arg, t->stack_top);   // 不返回：客体须经 ExitThread 退出 | Does not return: the guest must exit through ExitThread
            die("客体线程入口返回");
        }
        // 已跳回宿主栈：这里打日志是安全的（1 MiB 栈）。前 20 次与之后每 50 次记一行，真机上看得到线程确实从这条路退；t 可能已释放，不碰它
        // Back on the host stack: logging is safe here (1 MiB stack). First 20 exits and every 50th log a line; t may already be freed, so it is not touched
        { static int n; int k = __atomic_add_fetch(&n, 1, __ATOMIC_RELAXED); if (k <= 20 || k % 50 == 0) LOG("客体线程回到宿主栈退出（第 %d 次）", k); }
        pthread_exit(NULL);
    }
}
static void thr_free(thr_t *t) {   // 持 G；从全部线程链摘下并释放 | Holding G; unlink from the all-threads chain and free
    for (thr_t **p = &g_all; *p; p = &(*p)->all) if (*p == t) { *p = t->all; break; }
    pthread_cond_destroy(&t->cv); free(t);
}
static const char *thr_name(thr_t *t, char *buf, size_t n);
static void thread_exit(thr_t *t) {
    t->exit_ns = now_ns(); t->exit_tick = dm_thread_tick_fn ? dm_thread_tick_fn() : 0;
    // 这里不打日志：退出发生在客体线程自己的栈上（有的只有十几 KiB），带浮点的 printf 会把栈撑爆（真机冒烟：LibcurlResolver 退出时 SIGBUS，回溯 fprintf → __dtoa）。时刻只记下来，现场导出时再打
    // No logging here: the exit happens on the guest thread's own stack (some are only a dozen or so KiB), and a printf with floating point would blow the stack (real-device smoke test: SIGBUS when LibcurlResolver exits, backtrace fprintf → __dtoa). Only the time is recorded, and it is printed later in the state dump
    pthread_mutex_lock(&G);
    t->exited = 1;
    for (thr_t *w = g_wait; w; ) { thr_t *n = w->next; if (w->wkind == 4) wake(w, -1); w = n; }   // 同步等待者重查 | Synchronization waiters re-check
    pthread_setspecific(g_keythr, NULL);
    if (t->closed) thr_free(t);
    pthread_mutex_unlock(&G);
    // np562：回宿主栈去退出（g_keyhome）；没有跳回点的（进程主线程）照旧就地退出
    // np562: go back to the host stack to exit (g_keyhome); a thread without a jump point (the process main thread) exits in place
    sigjmp_buf *home = pthread_getspecific(g_keyhome);
    if (home) { pthread_setspecific(g_keyhome, NULL); siglongjmp(*home, 1); }
    pthread_exit(NULL);
}
void dm_thread_returned(void) { die("客体线程入口函数返回（Horizon 要求 ExitThread）"); }

// 主线程登记，返回其句柄（交给 nnosInitialize 作主线程句柄）
// Register the main thread and return its handle (passed to nnosInitialize as the main thread handle)
uint32_t svc_thread_main_init(void) {
    pthread_key_create(&g_keythr, NULL);
    pthread_key_create(&g_keyhome, NULL);   // np562：回宿主栈的跳回点 | jump point back to the host stack
    thr_t *t = calloc(1, sizeof *t);
    pthread_cond_init(&t->cv, NULL);
    t->started = 1; t->prio = 44; t->core = 0; t->mask = 7; t->pt = pthread_self(); t->mport = pthread_mach_thread_np(t->pt);
    pthread_mutex_lock(&G); t->handle = h_new(O_THREAD, t); t->id = g_next_id++; t->all = g_all; g_all = t; pthread_mutex_unlock(&G);
    pthread_setspecific(g_keythr, t);
    qos_class_t qc = QOS_CLASS_UNSPECIFIED; int rp = 0;
    pthread_get_qos_class_np(pthread_self(), &qc, &rp); t->qos = (uint32_t)qc;
    return t->handle;
}

static uint64_t g_main_yield_n, g_main_yield_ns;   // 只有客体主线程写；诊断行在同一线程读 | Written only by the guest main thread; the diagnostic line reads it on the same thread
void dm_main_yield_stats(uint64_t *n, uint64_t *ns) { *n = g_main_yield_n; *ns = g_main_yield_ns; }
uint32_t dm_guest_work_qos(unsigned *count) {
    uint32_t qos = QOS_CLASS_UNSPECIFIED; unsigned n = 0;
    pthread_mutex_lock(&G);
    for (thr_t *t = g_all; t; t = t->all) {
        if (t->handle == 0x1001) continue;
        if (!n) qos = t->qos; else if (qos != t->qos) qos = UINT32_MAX;
        n++;
    }
    pthread_mutex_unlock(&G);
    if (count) *count = n;
    return qos;
}

// ---------- 同步对象 ----------
// ---------- Synchronization objects ----------
// 假服务句柄：`sm::GetServiceHandle` 在宿主里没有服务端，只发一个 0x50xx 的号码（见 hle_svc.c）。
// 它们**不在句柄表里**，但客体确实会拿它去等（例：np384 真机 `nn::os::TryWaitSystemEvent` 等 fs 会话）。
// 语义按"空闲对象"处理：**合法但永不触发** —— 等它是超时，不是坏句柄（坏句柄会让 SDK 断言自杀）。
// Fake service handles: sm::GetServiceHandle has no server side in the host and only hands out a 0x50xx
// number (see hle_svc.c). They are NOT in the handle table, yet the guest really waits on them (np384 on
// device: nn::os::TryWaitSystemEvent waiting on the fs session). Treat them as "idle objects": valid but
// never signaled -- waiting on one is a timeout, not a bad handle (a bad handle makes the SDK assert).
static int fake_service_handle(uint32_t h) { return h >= 0x5000 && h <= 0x50ff; }
static int sig_obj(uint32_t h, int *bad) {    // 1=已触发（持 G） | 1 = signaled (holding G)
    unsigned i = h - 0x1000;
    if (h == H_CUR_THREAD) return 0;
    if (fake_service_handle(h)) return 0;
    if (i == 0 || i >= MAX_H) { *bad = 1; return 0; }
    switch (g_h[i].type) {
    case O_THREAD: return ((thr_t *)g_h[i].obj)->exited;
    case O_EVENT_R: return ((event_t *)g_h[i].obj)->signaled;
    default: *bad = 1; return 0;
    }
}
static void wake_sync_waiters(void) {
    for (thr_t *w = g_wait; w; ) { thr_t *n = w->next; if (w->wkind == 4) wake(w, -1); w = n; }
}

static int64_t now_mono_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1000000000ll + ts.tv_nsec; }
void dm_guest_clock_advanced(void) {
    if (!dm_detclock_enabled()) return;
    pthread_mutex_lock(&G);
    for (thr_t *t = g_wait; t; t = t->next) if (t->wkind) pthread_cond_signal(&t->cv);
    pthread_mutex_unlock(&G);
}

// 处理线程 / 同步类 SVC；返回 1 = 已处理
// Handle thread / synchronization SVCs; returns 1 = handled
static const char *thr_name(thr_t *t, char *nb, size_t n) {   // SDK ThreadType+0x1a0 = 名字指针 | SDK ThreadType+0x1a0 = name pointer
    if (!t->arg) return t->handle == 0x1001 ? "主线程" : "?";
    uint64_t np = 0; vm_size_t got = 0;
    if (vm_read_overwrite(mach_task_self(), t->arg + 0x1a0, 8, (vm_address_t)&np, &got) || !np ||
        vm_read_overwrite(mach_task_self(), np, n - 1, (vm_address_t)nb, &got)) return "?";
    nb[got] = 0; return nb;
}
// 全部活着的客体线程（采样器 DM_SAMPLE=毫秒:* 用）：把 pthread 填进 out，返回个数
// All live guest threads (used by the sampler DM_SAMPLE=ms:*): fill the pthreads into out, return the count
int thr_list(pthread_t *out, int max) {
    int n = 0;
    pthread_mutex_lock(&G);
    for (thr_t *t = g_all; t && n < max; t = t->all) if (!t->exited && t->started) out[n++] = t->pt;
    pthread_mutex_unlock(&G);
    return n;
}
// 浸泡读数（DM_SOAK）：句柄表占用、存活客体线程、线程对象总数（含已退出未释放）
// Soak readings (DM_SOAK): handle table usage, live guest threads, total thread objects (including exited but not yet freed)
void thr_counts(int *handles, int *live, int *objs) {
    int nh = 0, nl = 0, no = 0;
    pthread_mutex_lock(&G);
    for (int i = 1; i < MAX_H; i++) nh += g_h[i].type != O_FREE;
    for (thr_t *t = g_all; t; t = t->all) { no++; nl += t->started && !t->exited; }
    pthread_mutex_unlock(&G);
    *handles = nh; *live = nl; *objs = no;
}
// 按 pthread 取客体线程名（按线程 CPU 采样 DM_THRCPU 用，memstat.m）：不是客体线程返回 0
// Get the guest thread name by pthread (used by per-thread CPU sampling DM_THRCPU, memstat.m): returns 0 if not a guest thread
int thr_name_of(pthread_t pt, char *out, size_t n) {
    int ok = 0; char nb[64];
    pthread_mutex_lock(&G);
    for (thr_t *t = g_all; t; t = t->all) if (t->pt == pt && !t->exited) { snprintf(out, n, "%#x %s", t->handle, thr_name(t, nb, sizeof nb)); ok = 1; break; }
    pthread_mutex_unlock(&G);
    return ok;
}
uint32_t thr_cur_handle(void) { thr_t *t = pthread_getspecific(g_keythr); return t ? t->handle : 0; }   // 只读 TSD，信号处理里可用 | Only reads TSD, usable in a signal handler
// 按线程名找活着的客体线程（采样器用）；找不到返回 0
// Find a live guest thread by thread name (used by the sampler); returns 0 if not found
int thr_find_by_name(const char *name, pthread_t *out) {
    char nb[32]; int ok = 0;
    pthread_mutex_lock(&G);
    for (thr_t *t = g_all; t; t = t->all) if (!t->exited && !strcmp(thr_name(t, nb, sizeof nb), name)) { *out = t->pt; ok = 1; break; }
    pthread_mutex_unlock(&G);
    return ok;
}
// 现场导出（看门狗 / 崩溃时由 crash.c 调；只读、不取锁）：每个客体线程在等什么、最近一次 SVC 从哪发出
// 现场导出时取别的客体线程的回溯：暂停 → 读寄存器 → 沿帧指针链走（客体与门块都保持 x29 链）→ 恢复；只打 main / sdk 帧
// State dump (called from crash.c on watchdog / crash; read-only, takes no lock): what each guest thread is waiting on and where its most recent SVC was issued from
// During a state dump, take the backtrace of other guest threads: suspend → read registers → walk the frame pointer chain (both guest code and gate blocks keep the x29 chain) → resume; only main / sdk frames are printed
static void thr_bt(thr_t *t) {
    if (!t->mport || t->exited || t->mport == mach_thread_self()) return;
    if (thread_suspend(t->mport) != KERN_SUCCESS) return;
    arm_thread_state64_t st; mach_msg_type_number_t n = ARM_THREAD_STATE64_COUNT;
    if (thread_get_state(t->mport, ARM_THREAD_STATE64, (thread_state_t)&st, &n) == KERN_SUCCESS) {
        // 没在等系统调用的线程（在跑客体代码，可能在空转等一个条件）：打被调用者保存寄存器 x19–x28，以及 x19 / x20 指向的内存各 128 字节。
        // 偶发「进局后不出帧」现场：主线程在作业池的「边帮忙边等」（main+0x14f08d8）里空转，x19 = 等待对象、x20 = 作业池——有这些才能看出它在等哪一批、未完成计数是几
        // Threads not waiting in a system call (running guest code, possibly spinning on a condition): print the callee-saved registers x19–x28, plus 128 bytes of the memory pointed to by each of x19 / x20.
        // The state seen in the occasional "no frames after entering a run": the main thread spins in the job pool's "help while waiting" (main+0x14f08d8), x19 = wait object, x20 = job pool -- only with these can one tell which batch it is waiting for and what the unfinished count is
        if (t->wkind == 0) { char rl[400]; int q = snprintf(rl, sizeof rl, "[dm]   %#x 寄存器 pc=", t->handle); uint64_t pc = __darwin_arm_thread_state64_get_pc(st); module_t *pm = module_of((void *)pc);
            q += pm ? snprintf(rl + q, sizeof rl - q, "%s+%#llx", pm->name, pc - (uint64_t)pm->base) : snprintf(rl + q, sizeof rl - q, "%#llx", pc);
            for (int i = 19; i <= 28; i++) q += snprintf(rl + q, sizeof rl - q, " x%d=%#llx", i, st.__x[i]);
            fprintf(stderr, "%s\n", rl);
            for (int i = 19; i <= 20; i++) { uint8_t raw[128]; vm_size_t g = 0; uint64_t a = st.__x[i];
                if (a < 0x100000000ull || vm_read_overwrite(mach_task_self(), a, sizeof raw, (vm_address_t)raw, &g) || g != sizeof raw) continue;
                char hx[420]; int h = 0; for (int k = 0; k < 128; k += 4) h += snprintf(hx + h, sizeof hx - h, "%08x ", *(uint32_t *)(raw + k));
                uint64_t vt = *(uint64_t *)raw; module_t *vm_ = module_of((void *)vt); char vs[64] = ""; if (vm_) snprintf(vs, sizeof vs, "（首 8 字节 = %s+%#llx）", vm_->name, vt - (uint64_t)vm_->base);
                fprintf(stderr, "[dm]   %#x x%d → %#llx 起 128 字节%s: %s\n", t->handle, i, a, vs, hx); } }
        uint64_t fp = __darwin_arm_thread_state64_get_fp(st), lr = __darwin_arm_thread_state64_get_lr(st);
        // 在跑客体代码的线程：栈顶往上 1 KiB 原样打出来（外层各帧保存的寄存器在里面，事后能找回等待对象、资源地址）
        // Threads running guest code: print the 1 KiB above the stack top as is (the registers saved by the outer frames are in there, so the wait object and resource address can be recovered afterwards)
        if (t->wkind == 0 && module_of((void *)__darwin_arm_thread_state64_get_pc(st))) { uint64_t sp = __darwin_arm_thread_state64_get_sp(st), stk[128]; vm_size_t g = 0;   // pc 在客体模块里才打：在睡眠 / 让出里的线程 pc 在系统库，栈没看头 | Printed only if pc is inside a guest module: a thread in sleep / yield has its pc in a system library, and the stack is not worth looking at
            if (sp > 0x100000000ull && !vm_read_overwrite(mach_task_self(), sp, sizeof stk, (vm_address_t)stk, &g) && g == sizeof stk)
                for (int r = 0; r < 128; r += 8) fprintf(stderr, "[dm]   %#x 栈 sp+%#05x: %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx\n", t->handle, r * 8,
                        stk[r], stk[r + 1], stk[r + 2], stk[r + 3], stk[r + 4], stk[r + 5], stk[r + 6], stk[r + 7]); }
        char line[1024]; int p = snprintf(line, sizeof line, "[dm]   %#x 回溯:", t->handle), nf = 0; uint64_t wait_fp = 0;
        for (int k = 0; k < 64 && fp && nf < 20 && p < (int)sizeof line - 40; k++) {
            module_t *m = module_of((void *)lr);
            if (m) { p += snprintf(line + p, sizeof line - p, " %s+%#llx", m->name, lr - (uint64_t)m->base); nf++; }
            uint64_t fr[2]; vm_size_t g2 = 0;
            if (vm_read_overwrite(mach_task_self(), fp, 16, (vm_address_t)fr, &g2) || g2 != 16 || fr[0] <= fp) break;
            if (m && !strcmp(m->name, "main") && lr - (uint64_t)m->base == 0x132db30) wait_fp = fp;   // 返回点在资源等待函数 main+0x132d988 里（它调作业池「边帮忙边等」之后）：此刻的帧指针就是它自己的 | The return point is inside the resource wait function main+0x132d988 (after it calls the job pool's "help while waiting"): the frame pointer at this moment is its own
            fp = fr[0]; lr = fr[1];
        }
        fprintf(stderr, "%s\n", line);
        // 资源等待函数 main+0x132d988（序言 sub sp,#0x50 … add x29,sp,#0x40）在自己栈底摆等待对象 {虚表, 指向事件指针的指针}（0x132db14 stp x8,x23,[sp]），
        // 所以等待对象 = 它的帧指针 −0x40；所指 −0x10 = 资源等待对象 {+0 在途作业, +8 互斥锁指针, +0x10 事件指针}。按返回点 0x132db30 定位，与线程此刻停在等待循环的哪一条指令无关。
        // 旧写法按 0x14f0934 取「上一帧」取错一层，把资源等待对象当成等待对象读，「在途作业 为空」「事件 / 锁 读不到」都是错读
        // The resource wait function main+0x132d988 (prologue sub sp,#0x50 … add x29,sp,#0x40) places the wait object {vtable, pointer to the event pointer} at the bottom of its own stack (0x132db14 stp x8,x23,[sp]),
        // so the wait object = its frame pointer −0x40; what that points to, −0x10 = the resource wait object {+0 in-flight job, +8 mutex pointer, +0x10 event pointer}. It is located by the return point 0x132db30, regardless of which instruction of the wait loop the thread is stopped on right now.
        // The old code took "the previous frame" based on 0x14f0934 and was off by one level, reading the resource wait object as the wait object; 「在途作业 为空」(in-flight job is null) and 「事件 / 锁 读不到」(event / lock unreadable) were both misreads
        if (wait_fp) { uint64_t wo[2] = {0, 0}, res[3] = {0, 0, 0}, wjob = 0; vm_size_t g = 0;
            #define RD(addr, buf) ((addr) > 0x100000000ull && !vm_read_overwrite(mach_task_self(), (addr), sizeof(buf), (vm_address_t)(buf), &g) && g == sizeof(buf))
            #define HEX(tag, addr, nbytes) { uint32_t raw[32]; vm_size_t gg = 0; if ((addr) > 0x100000000ull && !vm_read_overwrite(mach_task_self(), (addr), (nbytes), (vm_address_t)raw, &gg) && gg == (nbytes)) { \
                char hx[300]; int h = 0; for (unsigned q = 0; q < (nbytes) / 4; q++) h += snprintf(hx + h, sizeof hx - h, "%08x ", raw[q]); fprintf(stderr, "[dm]   %#x 作业池等待：%s %#llx: %s\n", t->handle, tag, (unsigned long long)(addr), hx); } \
                else fprintf(stderr, "[dm]   %#x 作业池等待：%s %#llx 读不到\n", t->handle, tag, (unsigned long long)(addr)); }
            if (RD(wait_fp - 0x40, wo)) { module_t *vm_ = module_of((void *)wo[0]);
                fprintf(stderr, "[dm]   %#x 作业池等待：等待对象 %#llx 虚表 %s+%#llx（应为 main+0x20f2408），所指 %#llx\n", t->handle, (unsigned long long)(wait_fp - 0x40), vm_ ? vm_->name : "?", vm_ ? wo[0] - (uint64_t)vm_->base : wo[0], wo[1]);
                if (RD(wo[1] - 0x10, res)) {
                    fprintf(stderr, "[dm]   %#x 作业池等待：资源等待对象 %#llx：在途作业 %#llx（%s）、互斥锁 %#llx、事件 %#llx。判别：在途作业为空 = 信号已被某次等待消费；非空且事件未置位 = 完成信号没来；事件已置位 = 等待循环的问题\n",
                            t->handle, (unsigned long long)(wo[1] - 0x10), res[0], res[0] ? "非空" : "为空", res[1], res[2]);
                    HEX("事件本体 64 字节", res[2], 64);
                    HEX("互斥锁本体 32 字节（+8 重入计数、+0x10 持有线程对象、+0x18 锁字 = 持有者句柄）", res[1], 32);
                    // 作业对象（提交函数 main+0x147f154 填的）：+0x10 标志（取作业时按线程类别掩码过滤，0xeda190 / 0xdf8fac）、+0x20 队列里的下一个、+0x30 起依赖表（低 31 位 = 个数）、+0x60 资源、+0x68 指向事件指针的指针
                    // Job object (filled by the submit function main+0x147f154): +0x10 flags (filtered by a per-thread-class mask when taking a job, 0xeda190 / 0xdf8fac), +0x20 next in the queue, dependency table from +0x30 (low 31 bits = count), +0x60 resource, +0x68 pointer to the event pointer
                    if (res[0]) { HEX("在途作业对象 +0 起 128 字节", res[0], 128); HEX("在途作业对象 +0x80 起 128 字节", res[0] + 0x80, 128); } wjob = res[0]; } }
            // 作业池：线程正停在等待循环 main+0x14f08d8 自己的指令上时 x20 就是它；停在它调出去的函数里时 x20 不可信，不打
            // Job pool: when the thread is stopped on an instruction of the wait loop main+0x14f08d8 itself, x20 is the pool; when it is stopped in a function called from it, x20 cannot be trusted and is not printed
            { uint64_t pc = __darwin_arm_thread_state64_get_pc(st); module_t *pm = module_of((void *)pc); uint64_t off = pm && !strcmp(pm->name, "main") ? pc - (uint64_t)pm->base : 0;
              // 队列（0xeda190）：{+0 尾, +8 头, +0x10 个数, +0x18 锁}，节点 +0x10 标志、+0x20 下一个。顺着头走最多 8 个，看被等的作业在不在这个池里
              // Queue (0xeda190): {+0 tail, +8 head, +0x10 count, +0x18 lock}, node +0x10 flags, +0x20 next. Walk from the head for at most 8 nodes to see whether the awaited job is in this pool
              if (off >= 0x14f0900 && off < 0x14f09b8) { uint64_t qa = st.__x[20] + 0x410, nd = 0; HEX("作业队列头（池 = x20，+0x410 起 32 字节）", qa, 32); int in = 0, cnt = 0;
                  if (RD(qa + 8, &nd)) for (; nd && cnt < 8; cnt++) { uint64_t nb[5] = {0}; if (!RD(nd, nb)) break; fprintf(stderr, "[dm]   %#x 作业池等待：  队列第 %d 个 %#llx 标志 %#x%s\n", t->handle, cnt, (unsigned long long)nd, (uint32_t)nb[2], nd == wjob ? "（就是被等的作业）" : ""); in |= nd == wjob; nd = nb[4]; }
                  fprintf(stderr, "[dm]   %#x 作业池等待：被等的作业%s x20 这个池的队列前 %d 个里\n", t->handle, in ? "在" : "不在", cnt); }
              else fprintf(stderr, "[dm]   %#x 作业池等待：线程不在等待循环自己的指令上，作业池没取\n", t->handle); }
            #undef RD
            #undef HEX
        }
    }
    thread_resume(t->mport);
}
void svc_thread_dump(void) {
    // np480：先倒 NVN 最近调用（EndRecording 后没有 QueueSubmit——看最后调的 16 个入口就知道差哪一步）
    { void dm_dump_nvn_last_calls(void); dm_dump_nvn_last_calls(); }
    static const char *kinds[] = {"运行/未在等", "锁", "条件变量", "地址", "同步对象"};
    int nh = 0; for (int i = 1; i < MAX_H; i++) nh += g_h[i].type != O_FREE;
    fprintf(stderr, "[dm] 句柄表占用 %d / %d，累计建线程 %llu\n", nh, MAX_H, g_next_id - 0x51);
    for (thr_t *t = g_all; t; t = t->all) {
        module_t *m = module_of((void *)t->last_lr);
        // 线程名：SDK ThreadType+0x1a0 是名字指针（ThreadManager::SetInitialThreadNameUnsafe 0x34a2a4）；CreateThread 的 x2 = ThreadType*
        // Thread name: SDK ThreadType+0x1a0 is the name pointer (ThreadManager::SetInitialThreadNameUnsafe 0x34a2a4); x2 of CreateThread = ThreadType*
        char nb[32]; const char *nm = thr_name(t, nb, sizeof nb);
        double cpu = -1;
        if (t->mport && !t->exited) { thread_basic_info_data_t bi; mach_msg_type_number_t n = THREAD_BASIC_INFO_COUNT;
            if (thread_info(t->mport, THREAD_BASIC_INFO, (thread_info_t)&bi, &n) == KERN_SUCCESS)
                cpu = bi.user_time.seconds + bi.user_time.microseconds / 1e6 + bi.system_time.seconds + bi.system_time.microseconds / 1e6; }
        char own[96] = ""; if (t->wkind == 1 && t->waddr) { uint32_t w = 0; vm_size_t g2 = 0;   // 锁等待：锁字 = 持有者句柄 | 0x40000000 | Lock wait: lock word = owner handle | 0x40000000
            if (!vm_read_overwrite(mach_task_self(), t->waddr, 4, (vm_address_t)&w, &g2)) snprintf(own, sizeof own, " 持有者 %#x", w & ~WAITERS_BIT); }
        if (t->wkind == 2 && t->waddr) { uint32_t w = 0, k = 0; vm_size_t g2 = 0;              // 条件等待：关联锁字与条件键的当前值（卡死现场缺这一项） | Condition wait: current values of the associated lock word and the condition key (the deadlock dump lacked this item)
            vm_read_overwrite(mach_task_self(), t->waddr, 4, (vm_address_t)&w, &g2); vm_read_overwrite(mach_task_self(), t->wkey, 4, (vm_address_t)&k, &g2);
            snprintf(own, sizeof own, " 锁字 %#x 条件键 %#llx=%#x", w, t->wkey, k); }
        uint8_t raw[64]; vm_size_t g3 = 0;                                                   // 所等对象（读写锁 / 条件变量所在结构）原始 64 字节：看计数与持有者 | Raw 64 bytes of the object being waited on (the structure holding the read-write lock / condition variable): to see the counts and the owner
        int have_raw = (t->wkind == 1 || t->wkind == 2 || t->wkind == 3) && t->waddr && !vm_read_overwrite(mach_task_self(), t->waddr & ~0xfull, 64, (vm_address_t)raw, &g3) && g3 == 64;
        char life[96]; uint64_t nowns = now_ns();
        if (t->exited) snprintf(life, sizeof life, " 建于拍 %u、退于拍 %u（%.1f 秒前）", t->born_tick, t->exit_tick, (nowns - t->exit_ns) / 1e9); else if (!t->born_ns) snprintf(life, sizeof life, " 进程启动时就在"); else snprintf(life, sizeof life, " 建于拍 %u（%.1f 秒前）", t->born_tick, (nowns - t->born_ns) / 1e9);
        fprintf(stderr, "[dm] 线程 %#x 「%s」优先级 %d 核 %d 掩码 %#llx CPU %.2fs svc %llu %s%s%s 等:%s addr=%#llx%s 最近 svc %#x lr=%s+%#llx\n", t->handle, nm, t->prio, t->core, (unsigned long long)t->mask,
                cpu, t->nsvc, t->exited ? "已退出" : "", t->started ? "" : "未启动", life, kinds[t->wkind & 7], t->waddr, own, t->last_svc,
                m ? m->name : "?", m ? t->last_lr - (uint64_t)m->base : t->last_lr);
        if (have_raw) { char hx[200]; int q = 0; for (int i = 0; i < 64; i += 4) q += snprintf(hx + q, sizeof hx - q, "%08x ", *(uint32_t *)(raw + i));
                        fprintf(stderr, "[dm]   %#x 所等对象 %#llx 起 64 字节: %s\n", t->handle, t->waddr & ~0xfull, hx); }
        thr_bt(t);
    }
    // 宿主自建线程（预取、Granny 后台、图形出图、音频等）：不在客体线程表里，按 mach 线程枚举，dladdr 符号化前 12 帧
    // Threads created by the host itself (prefetch, Granny background, graphics present, audio, etc.): not in the guest thread table, enumerated as mach threads, the first 12 frames symbolized with dladdr
    thread_act_array_t ths; mach_msg_type_number_t nth = 0;
    if (task_threads(mach_task_self(), &ths, &nth) == KERN_SUCCESS) {
        for (mach_msg_type_number_t i = 0; i < nth; i++) {
            int guest = 0; for (thr_t *t = g_all; t; t = t->all) if (t->mport == ths[i]) { guest = 1; break; }
            if (!guest && ths[i] != mach_thread_self() && thread_suspend(ths[i]) == KERN_SUCCESS) {
                arm_thread_state64_t st; mach_msg_type_number_t n = ARM_THREAD_STATE64_COUNT;
                if (thread_get_state(ths[i], ARM_THREAD_STATE64, (thread_state_t)&st, &n) == KERN_SUCCESS) {
                    char line[1200]; int p = snprintf(line, sizeof line, "[dm] 宿主线程 %#x:", ths[i]);
                    uint64_t fp = __darwin_arm_thread_state64_get_fp(st), pc = __darwin_arm_thread_state64_get_pc(st);
                    for (int k = 0; k < 12 && p < (int)sizeof line - 80; k++) {
                        Dl_info di; module_t *m = module_of((void *)pc);
                        if (m) p += snprintf(line + p, sizeof line - p, " %s+%#llx", m->name, pc - (uint64_t)m->base);
                        else if (dladdr((void *)pc, &di) && di.dli_sname) p += snprintf(line + p, sizeof line - p, " %s+%#lx", di.dli_sname, (unsigned long)(pc - (uint64_t)di.dli_saddr));
                        else p += snprintf(line + p, sizeof line - p, " %#llx", pc);
                        uint64_t fr[2]; vm_size_t g2 = 0;
                        if (!fp || vm_read_overwrite(mach_task_self(), fp, 16, (vm_address_t)fr, &g2) || g2 != 16 || fr[0] <= fp) break;
                        fp = fr[0]; pc = fr[1];
                    }
                    fprintf(stderr, "%s\n", line);
                }
                thread_resume(ths[i]);
            }
            mach_port_deallocate(mach_task_self(), ths[i]);
        }
        vm_deallocate(mach_task_self(), (vm_address_t)ths, nth * sizeof *ths);
    }
    // 消息队列现场（hle_msgq.c）：哪个队列有人发、有人收、此刻几个线程阻塞在它上面。
    // "worker 在 ReceiveMessageQueue 上等" 与 "队列里有没有货" 是两件事，这里一次看全。
    // Message-queue state (hle_msgq.c): which queues are sent to, received from, and how many threads are blocked on each right now.
    // "a worker waits in ReceiveMessageQueue" and "is there anything in the queue" are two different things; this shows both at once.
    { void dm_msgq_dump(void); dm_msgq_dump(); }
    // 探针计数（diag.c）：卡住时把"入口被调了几次、返回点被调了几次"一次打全——差值就是阻塞所在的那一段。
    // Probe counts (diag.c): on a stall print how many times each entry and each return point was reached — the gap is the blocked segment.
    { void dm_probe_dump_counts(void); dm_probe_dump_counts(); }
}

// np343 对齐 nxnative：占进程线程资源上限名额的游戏线程数——已创建（无论是否 started）且未退出（hle_svc.c GetInfo FreeThreadCount 用）。
// Horizon 里 pm 给应用的限额是 96 个线程；SDK 11+ 的 nn::os::GetFreeThreadCount 拿它决定建不建线程，
// 回 0 会让主线程在启动时等一个永远不会出现的线程（nxnative 实测：Monster Train 2 卡死在 main+0x3e1ac4）
int thr_limit_used(void) {
    int n = 0;
    pthread_mutex_lock(&G);
    for (thr_t *t = g_all; t; t = t->all) n += !t->exited;
    pthread_mutex_unlock(&G);
    return n;
}

int svc_thread(int num, uint64_t *regs) {
    { thr_t *me = pthread_getspecific(g_keythr); if (me) { me->last_svc = num; me->last_lr = regs[30]; me->nsvc++; } }
    // np243：带上返回地址 —— 一轮"逐页扫内存"的 svc 只有靠 lr 才能定位是客体的哪个函数在扫
    // np243：lr 直接翻成「模块+偏移」，省得再猜基址
    if (g_flag.thr && num != 0x0B) {
        thr_t *me = pthread_getspecific(g_keythr);
        module_t *mm = module_of((void *)regs[30]);
        LOG("[%#x] svc %#x x0=%#llx x1=%#llx x2=%#llx x3=%#llx lr=%s+%#llx", me ? me->handle : 0, num,
            regs[0], regs[1], regs[2], regs[3], mm ? mm->name : "?", mm ? regs[30] - (uint64_t)mm->base : regs[30]);
    }
    switch (num) {
    case 0x08: {  // CreateThread(→w1 句柄; x1 入口, x2 参数, x3 栈顶, w4 优先级, w5 核) | CreateThread(→w1 handle; x1 entry, x2 argument, x3 stack top, w4 priority, w5 core)
        {   // np549：客体栈区整片提交（只改权限位，页按需分配）。不提交的话，栈顶贴着栈区起点的
            //   那个线程往下一长就踩到 PROT_NONE ⇒ si_code=2 的 SIGSEGV。
            // np549: commit the whole guest stack region (permission bits only; pages stay demand-allocated).
            //   Without it, the thread whose stack top sits at the bottom of the region faults with si_code=2.
            void dm_guest_commit_stack(void); dm_guest_commit_stack();
        }
        thr_t *t = calloc(1, sizeof *t);
        pthread_cond_init(&t->cv, NULL);
        t->entry = regs[1]; t->arg = regs[2]; t->stack_top = regs[3] & ~0xFull;
        // np357：**入参对不对**。真机崩点是"构造函数 main+0x45f9c8 被以 this=NULL 调用"，
        // 而 dm_thread_enter 是尾跳转（不改 lr）⇒ 现象与"某个线程的 entry 就是这个构造函数、
        // arg=0"完全一致。这里把前 8 次 CreateThread 的原始入参记下来，一眼看穿是游戏传错还是我们读错寄存器。
        // np357: are the arguments right? The crash is "ctor main+0x45f9c8 entered with this=NULL", and
        // dm_thread_enter tail-branches (does not touch lr), which fits "some thread's entry is that
        // constructor with arg=0" exactly. Record the raw arguments of the first 8 CreateThread calls.
        {
            static int ncreate;
            if (ncreate < 8) {
                ncreate++;
                module_t *me_ = module_of((void *)regs[1]);
                module_t *ml_ = module_of((void *)regs[30]);
                uint64_t real_fn = 0; memcpy(&real_fn, (void *)regs[2], 8);   // np425：arg 首 8 字节 = 真实入口 | np425: arg[0] = the real entry
                module_t *mf_ = module_of((void *)real_fn);
                LOG("[dm] np357 CreateThread#%d x0=%#llx x1(entry)=%#llx(%s+%#llx) x2(arg)=%#llx x3(sp)=%#llx x4=%#llx x5=%#llx 真实入口=%#llx(%s+%#llx) 调用者=%s+%#llx",
                    ncreate, (unsigned long long)regs[0],
                    (unsigned long long)regs[1], me_ ? me_->name : "?", me_ ? (uint64_t)regs[1] - (uint64_t)me_->base : 0,
                    (unsigned long long)regs[2], (unsigned long long)regs[3],
                    (unsigned long long)regs[4], (unsigned long long)regs[5],
                    (unsigned long long)real_fn, mf_ ? mf_->name : "?", mf_ ? real_fn - (uint64_t)mf_->base : real_fn,
                    ml_ ? ml_->name : "?", ml_ ? (uint64_t)regs[30] - (uint64_t)ml_->base : 0);
            }
        }
        // np343 对齐 nxnative：核 -2（IdealCoreUseProcessValue）= 进程缺省理想核（本游戏为 0），亲和掩码只有它——同 Horizon 内核。
        // 原先原样记 -2，GetCurrentProcessorNumber 就会答 0xFFFFFFFE：按当前核号做下标索引每核引擎槽的代码会越界
        // （nxnative 实测：SMBW 字体引擎以核号选槽，收到 -2 直接崩 main+0x107dbc0）。
        t->prio = (int)regs[4]; t->core = (int)regs[5] >= 0 ? (int)regs[5] : 0; t->mask = 1ull << t->core;
        t->born_ns = now_ns(); t->born_tick = dm_thread_tick_fn ? dm_thread_tick_fn() : 0;
        pthread_mutex_lock(&G); t->handle = h_new(O_THREAD, t); t->id = g_next_id++; t->all = g_all; g_all = t; pthread_mutex_unlock(&G);
        pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setstacksize(&a, 1 << 20);
        pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);   // 退出即回收宿主资源（客体经句柄等待，不 join） | Host resources are reclaimed on exit (the guest waits through the handle, no join)
        if (pthread_create(&t->pt, &a, thread_main, t)) die("pthread_create");
        pthread_attr_destroy(&a);
        if (g_flag.thr) LOG("CreateThread h=%#x entry=%#llx arg=%#llx sp=%#llx prio=%d core=%d", t->handle, t->entry, t->arg, t->stack_top, t->prio, t->core);
        regs[0] = 0; regs[1] = t->handle; return 1;
    }
    case 0x09: {  // StartThread(w0)
        pthread_mutex_lock(&G);
        thr_t *t = h_get((uint32_t)regs[0], O_THREAD);
        if (t) { t->started = 1; pthread_cond_signal(&t->cv); }
        pthread_mutex_unlock(&G);
        regs[0] = t ? 0 : R_BAD_HANDLE; return 1;
    }
    case 0x0A: {  // np411：谁在退出（前 8 次）——np410 的"主线程等工作线程退出"要对照这里
        static int nx; if (nx < 8) { thr_t *t0 = cur(); module_t *me_ = module_of((void *)t0->entry);
            LOG("ExitThread #%d entry=%s+%#llx arg=%#llx", ++nx, me_ ? me_->name : "?", me_ ? (uint64_t)t0->entry - (uint64_t)me_->base : 0, (unsigned long long)t0->arg); }
        thread_exit(cur()); return 1; }
    case 0x0B: {  // SleepThread(x0 ns)；0/-1/-2 = 各类让出 | SleepThread(x0 ns); 0/-1/-2 = the various kinds of yield
        int64_t ns = (int64_t)regs[0];
        // np471：主线程第一次 SleepThread 时打**调用者 + 栈扫描**。现在卡点就是"主线程在睡、没人叫醒它"，
        // 而线程表只给到 sdk+0x7b0f8c 这层 SDK 包装 —— 必须再往上一层看是谁调的。
        // dm_backtrace_scan 已加固（vm_read_overwrite），在 SVC 现场调用也不会再崩。
        // np471: on the main thread's first SleepThread, log the caller plus a stack scan. The blocker is now
        // "the main thread sleeps and nobody wakes it", and the thread table only shows the SDK wrapper.
        { thr_t *me0 = pthread_getspecific(g_keythr);
          if (me0 && me0->handle == 0x1001) {
              static unsigned n0;   // np495：改为每 200 次打一条——栈上的 fp 链值可能是**陈旧残留**
              if (++n0 % 200 == 1) {
                  module_t *mc = module_of((void *)regs[30]);
                  LOG("主线程 SleepThread(ns=%lld) 调用者=%s+%#llx", (long long)ns,
                      mc ? mc->name : "?", mc ? (uint64_t)regs[30] - (uint64_t)mc->base : (uint64_t)regs[30]);
                  // np472：用 **fp 链**（不是扫栈）——门保存区里 x29 是客体自己的 fp，链上是干净的调用序列；
                  // np472: use the fp chain (not a stack scan) -- x29 in the gate save area is the guest's own fp.
                  { void dm_backtrace(uint64_t); dm_backtrace(regs[29]); }
              }
          } }
        uint64_t t0 = g_flag.bootprof ? now_ns() : 0;
        if (ns > 0 && dm_detclock_enabled()) {
            uint64_t deadline = guest_ns() + (uint64_t)ns;
            while (guest_ns() < deadline) { struct timespec ts = {0, 1000000}; nanosleep(&ts, 0); }
        } else if (ns > 0) { struct timespec ts = {ns / 1000000000, ns % 1000000000}; nanosleep(&ts, 0); }
        else {   // 让出（0 / -1 / -2）。主线程的让出次数与耗时记账：游戏渲染任务图「等全部做完」是纯轮询 + 让出（main+0x15ee898），量它占一帧多少 | Yield (0 / -1 / -2). Count and time of the main thread's yields are recorded: the game's render task graph "wait for everything to finish" is pure polling + yield (main+0x15ee898); measure how much of a frame it takes
            thr_t *me = pthread_getspecific(g_keythr);
            if (me && me->handle == 0x1001) { uint64_t y0 = now_ns(); sched_yield(); g_main_yield_ns += now_ns() - y0; g_main_yield_n++; }
            else sched_yield();
        }
        if (g_flag.bootprof) { uint64_t d = now_ns() - t0; dm_prof_add(PROF_SLEEP, d, 0); thr_t *me = pthread_getspecific(g_keythr); if (me && me->handle == 0x1001) dm_prof_add(PROF_MAIN_WAIT, d, 0); }
        return 1;
    }
    case 0x0C: { thr_t *t = h_get((uint32_t)regs[1], O_THREAD); regs[0] = t ? 0 : R_BAD_HANDLE; regs[1] = t ? (uint64_t)t->prio : 0; return 1; }
    case 0x0D: { thr_t *t = h_get((uint32_t)regs[0], O_THREAD); if (t) t->prio = (int)regs[1]; regs[0] = t ? 0 : R_BAD_HANDLE; return 1; }
    case 0x0E: { thr_t *t = h_get((uint32_t)regs[2], O_THREAD);
                 regs[0] = t ? 0 : R_BAD_HANDLE; regs[1] = t ? (uint64_t)(uint32_t)t->core : 0; regs[2] = t ? t->mask : 0; return 1; }
    case 0x0F: {  // SetThreadCoreMask(w0, w1 理想核, x2 掩码)：-2 = 进程缺省（理想核 0、掩码只有它）、-1 = 不在乎、-3 = 理想核不变（同 Horizon 内核）；宿主只记账 | SetThreadCoreMask(w0, w1 ideal core, x2 mask): -2 = use the process default (ideal core 0, mask of just that core), -1 = don't care, -3 = leave the ideal core unchanged (as in the Horizon kernel); the host only keeps the bookkeeping
        thr_t *t = h_get((uint32_t)regs[0], O_THREAD);
        if (t) { int c = (int)regs[1]; if (c >= 0) t->core = c; if (c == -2) { t->core = 0; t->mask = 1; } else if (regs[2]) t->mask = regs[2];
            // np343 对齐 nxnative：理想核不在新掩码里（-2/-3 保留理想核但钉到别的核）时，同 Horizon 把理想核挪到掩码最低核。
            // 否则 GetCurrentProcessorNumber 对钉在核 1/2 的线程仍报核 0，按核分队列的作业系统会去取核 0 的活、核 1/2 的队列没人排空
            if (t->mask && (t->core < 0 || !(t->mask >> t->core & 1))) t->core = __builtin_ctzll(t->mask); }
        regs[0] = t ? 0 : R_BAD_HANDLE; return 1;
    }
    case 0x10: { thr_t *t = cur(); int c = t->core;   // np343 对齐 nxnative：当前核号永远答 0..2；没有理想核时报掩码允许的最低核，而不是掩码外的核 0
                 regs[0] = (uint64_t)(uint32_t)(c >= 0 ? c : t->mask ? __builtin_ctzll(t->mask) : 0); return 1; }
    case 0x25: { thr_t *t = h_get((uint32_t)regs[1], O_THREAD); regs[0] = t ? 0 : R_BAD_HANDLE; regs[1] = t ? t->id : 0; return 1; }

    case 0x1A: {  // ArbitrateLock(w0 持有者句柄, x1 地址, w2 本线程标签) | ArbitrateLock(w0 owner handle, x1 address, w2 this thread's tag)
        thr_t *t = cur(); uint64_t addr = regs[1];
        pthread_mutex_lock(&G);
        int64_t r = 0;
        uint32_t word = ld32(addr);
        if (word == ((uint32_t)regs[0] | WAITERS_BIT)) { t->waddr = addr; t->wtag = (uint32_t)regs[2]; r = block(t, 1, -1); }
        // np566：条件不满足的现场（前 120 条）。主线程 4 万次 svc 0x1a 全走这条 else ⇒ 从不登记为
        //   等待者 ⇒ pick/handoff 永远看不到它（[ho]=0 自洽）。为什么条件不满足：锁字 ≠ w0|WAITERS_BIT。
        //   可能：(a) 锁字没 WAITERS_BIT（游戏用户态 CAS 置位前就发 svc）；(b) w0 ≠ 锁字里的 tag
        //   （tag 复用/错误）。读数直接分辨。
        // np566: snapshot when the condition fails (first 120). The main thread's 40k svc-0x1a all take this else,
        //   never registering as a waiter -- that is why pick/handoff never see it ([ho]=0 is consistent). Why the
        //   mismatch: word ≠ w0|WAITERS_BIT. Either (a) the word lacks WAITERS_BIT (svc before the user-mode CAS
        //   set it), or (b) w0 ≠ the tag in the word (tag reuse/mismatch). The readings tell them apart.
        else if (cv_take(&g_nhandoff, 120)) {
            LOG("[lk] miss h=%#x addr=%#llx word=%#x w0=%#llx w2=%#llx wb=%d tag=%d",
                t->handle, (unsigned long long)addr, word, (unsigned long long)regs[0], (unsigned long long)regs[2],
                !!(word & WAITERS_BIT), word & ~WAITERS_BIT);
        }
        pthread_mutex_unlock(&G);
        regs[0] = (uint64_t)r; return 1;
    }
    case 0x1B: pthread_mutex_lock(&G); lock_handoff(regs[0]); pthread_mutex_unlock(&G); regs[0] = 0; return 1;
    case 0x1C: {  // WaitProcessWideKeyAtomic(x0 锁地址, x1 条件键, w2 标签, x3 超时) | WaitProcessWideKeyAtomic(x0 lock address, x1 condition key, w2 tag, x3 timeout)
        thr_t *t = cur(); uint64_t addr = regs[0], key = regs[1];
        pthread_mutex_lock(&G);
        // 顺序要紧：先置条件键 = 1，再放互斥锁。SDK 的 InternalConditionVariable::Signal 在用户态先读条件键、为 0 就不发 svc 0x1D；
        // 用户态抢互斥锁也不经过 G。原先先 lock_handoff 后置键：放锁后、置键前，别的线程可在用户态抢到锁、改完条件、读到键 0 就不唤醒，
        // 本线程随后挂上去永远等不到（并行负载下资源表读写锁 main+0x320e5e0 卡死：读者计数 0、锁空闲、写者条件键 1、写者仍在等）。
        // 挂进等待链在 block 里、仍持 G：svc 0x1D 要取 G，不会落在这之前
        // The order matters: first set the condition key = 1, then release the mutex. The SDK's InternalConditionVariable::Signal first reads the condition key in user mode, and does not issue svc 0x1D if it is 0;
        // grabbing the mutex in user mode does not go through G either. It used to do lock_handoff first and set the key after: after the release and before the key was set, another thread could grab the lock in user mode, change the condition, read key 0 and not wake anyone,
        // and this thread would then link itself in and wait forever (under parallel load the resource table read-write lock main+0x320e5e0 deadlocked: reader count 0, lock free, writer condition key 1, writer still waiting).
        // Linking into the wait chain happens inside block, still holding G: svc 0x1D has to take G, so it cannot land before this
#ifdef DM_TEST_HOOKS
#define CV_TEST_HOOK() { extern void (*dm_test_cv_after_release)(void); if (dm_test_cv_after_release) dm_test_cv_after_release(); }   // 单测：紧跟放锁之后、挂链之前 | Unit test: right after releasing the lock, before linking into the wait chain
#else
#define CV_TEST_HOOK()
#endif
#ifndef DM_TEST_CV_OLD_ORDER
        st32(key, 1);
        t->wkey = key; t->waddr = addr; t->wtag = (uint32_t)regs[2];
        lock_handoff(addr); CV_TEST_HOOK();
        if (cv_take(&g_cv_probe, 120)) LOG("[cv] handoff h=%#x addr=%#llx 交接后 word=%#x waiters=%d", t->handle,
                                  (unsigned long long)addr, ld32(addr),
                                  (ld32(addr) & WAITERS_BIT) ? 1 : 0);
#else   // 仅单测 t_condvar 的负对照：还原修复前「先放锁、后置键」的顺序 | Negative control for the unit test t_condvar only: restores the pre-fix order "release the lock first, set the key after"
        lock_handoff(addr); CV_TEST_HOOK();
        st32(key, 1);
        t->wkey = key; t->waddr = addr; t->wtag = (uint32_t)regs[2];
#endif
        int64_t r = block(t, 2, (int64_t)regs[3]);   // 超时：不重新拿锁（用户态自己拿） | Timeout: the lock is not re-acquired (user mode takes it itself)
        // np560：0x1c 的交接取证（DM_CV_PROBE=1，前 120 条）。
        //   要分清三种病：① 锁被交回给持有者自己（pick 过滤错）② 每次都被立刻唤醒（假唤醒/wake 配错）
        //   ③ 每次都是超时返回（超时语义错）。这三者的读数完全不同：
        //   ① → handoff 后锁字 = 自己的 tag；② → r 正常但耗时≈0；③ → r=R_TIMEOUT。
        // np560: hand-off evidence for 0x1c (DM_CV_PROBE=1, first 120). Three different illnesses:
        //   1) the lock is handed back to its owner (pick filter wrong); 2) every wait is woken at once (spurious wake /
        //   wake mismatch); 3) every wait times out (timeout semantics wrong). Their readings differ: 1) the word after
        //   hand-off equals our own tag; 2) r is normal but elapsed ~0; 3) r == R_TIMEOUT.
        if (cv_take(&g_cv_probe, 120)) {
            LOG("[cv] 0x1c h=%#x addr=%#llx key=%#llx tag=%#x to=%lld r=%d word=%#x", t->handle,
                (unsigned long long)addr, (unsigned long long)key, t->wtag, (long long)(int64_t)regs[3],
                (int)r, ld32(addr));
        }
        pthread_mutex_unlock(&G);
        regs[0] = (uint64_t)r; return 1;
    }
    case 0x1D: {  // SignalProcessWideKey(x0 条件键, w1 个数，<=0 全部) | SignalProcessWideKey(x0 condition key, w1 count, <=0 all)
        uint64_t key = regs[0]; int cnt = (int)regs[1], n = 0;
        pthread_mutex_lock(&G);
        for (thr_t *w; (cnt <= 0 || n < cnt) && (w = pick(2, key)); n++) lock_acquire_for(w, w->waddr, w->wtag);   // 按内核顺序；转为锁等待或被唤醒后不再是 kind 2 | In kernel order; once turned into a lock wait or woken it is no longer kind 2
        int left = 0; for (thr_t *w = g_wait; w; w = w->next) if (w->wkind == 2 && w->wkey == key) { left = 1; break; }
        if (!left) st32(key, 0);
        if (cv_take(&g_cv_sig, 60)) { ++g_cv_sig;
            LOG("[cv] 0x1d h=%#x key=%#llx cnt=%d 唤醒=%d 仍有等待者=%d", cur()->handle, (unsigned long long)key, cnt, n, left); }
        // np567：0x1d 的 key 直方图（去重记账，随 [stall] 打出）——回答"这个条件变量
        //   整个生命周期里被 signal 过几次、被谁"。卡死判据：0x100d 等的 key=0x…a518 计数为 0。
        // np567: per-key histogram of 0x1d (dedup bookkeeping, dumped with [stall]) -- it answers "how many times
        //   was this condition signalled, by whom". Stall criterion: the key 0x100d waits on has a count of 0.
        {
            int i;
            for (i = 0; i < g_cvh_n; i++) if (g_cv_hist[i].key == key) break;
            if (i == g_cvh_n && g_cvh_n < 32) { g_cv_hist[g_cvh_n].key = key; g_cv_hist[g_cvh_n].n = 0; g_cv_hist[g_cvh_n].h = 0; g_cvh_n++; }
            if (i < g_cvh_n) { g_cv_hist[i].n++; g_cv_hist[i].h = cur()->handle; }
        }
        pthread_mutex_unlock(&G);
        regs[0] = 0; return 1;
    }

    case 0x34: {  // WaitForAddress(x0 地址, w1 类型, w2 值, x3 超时)：0 小于则等、1 小于则减一再等、2 相等则等、3 64 位值相等则等 | WaitForAddress(x0 address, w1 type, w2 value, x3 timeout): 0 wait if less than, 1 decrement if less than then wait, 2 wait if equal, 3 wait if the 64-bit value is equal
        // np343 对齐 nxnative：补类型 3（SDK 19 的 nn::os::WaitForAddress(const long*, long) 用它，之前会落进 32 位小于判断）
        thr_t *t = cur(); uint64_t addr = regs[0]; int type = (int)regs[1]; int32_t val = (int32_t)regs[2];
        pthread_mutex_lock(&G);
        int64_t r; uint32_t v = ld32(addr);
        int ok = type == 3 ? ld64(addr) == regs[2] : type == 2 ? (int32_t)v == val : (int32_t)v < val;
        if (ok && type == 1) { while (!cas32(addr, &v, v - 1)) if (!((int32_t)v < val)) { ok = 0; break; } }
        if (!ok) r = R_BAD_STATE;
        else { t->waddr = addr; r = block(t, 3, (int64_t)regs[3]); }
        pthread_mutex_unlock(&G);
        regs[0] = (uint64_t)r; return 1;
    }
    case 0x35: {  // SignalToAddress(x0 地址, w1 类型, w2 值, w3 个数)：0 唤醒、1 相等则加一并唤醒、2 相等则按等待数改值并唤醒 | SignalToAddress(x0 address, w1 type, w2 value, w3 count): 0 wake, 1 increment if equal and wake, 2 if equal modify the value by waiter count and wake
        uint64_t addr = regs[0]; int type = (int)regs[1]; int32_t val = (int32_t)regs[2]; int cnt = (int)regs[3];
        pthread_mutex_lock(&G);
        int nw = 0; for (thr_t *w = g_wait; w; w = w->next) if (w->wkind == 3 && w->waddr == addr) nw++;
        int64_t r = 0;
        if (type != 0) {
            int32_t nv = val + 1;
            if (type == 2) {
                // np343 对齐 nxnative（mesosphere KAddressArbiter::SignalAndModifyByWaitingCountIfEqual；Ryujinx 与 yuzu 一致）：
                // 全醒（count <= 0）且有等待者时减 2，不是减 1。以前减 1：Unity 的作业栅栏靠这个值判断作业是否做完，
                // 等待者被提前放行，粒子几何作业与主线程对不上数、写出界（纹理对象下标、写空指针）
                if (cnt <= 0) nv = nw > 0 ? val - 2 : val + 1;
                else nv = nw == 0 ? val + 1 : nw <= cnt ? val - 1 : val;
            }
            uint32_t e = (uint32_t)val;
            if (!cas32(addr, &e, (uint32_t)nv)) r = R_BAD_STATE;
        }
        if (!r) { int n = 0;
            for (thr_t *w; (cnt <= 0 || n < cnt) && (w = pick(3, addr)); n++) wake(w, 0); }
        pthread_mutex_unlock(&G);
        regs[0] = (uint64_t)r; return 1;
    }

    case 0x45: {  // CreateEvent(→w1 写端, w2 读端) | CreateEvent(→w1 write end, w2 read end)
        event_t *e = calloc(1, sizeof *e); e->refs = 2;
        pthread_mutex_lock(&G); uint32_t hw = h_new(O_EVENT_W, e), hr = h_new(O_EVENT_R, e); pthread_mutex_unlock(&G);
        // np461：把**进来的 x0–x3** 打出来。如果 x0/x1 是客体堆里的指针（例如 0x5e…），
        // 说明 Horizon 的约定是"出参用指针"（*x0=写端, *x1=读端），我们现在按寄存器返回就错了；
        // 如果 x0/x1 是 0 或小整数，才是"返回值约定"。
        // np461: log the incoming x0-x3. Guest-heap pointers mean Horizon's convention is "out params by
        // pointer" (*x0 = write end, *x1 = read end) and our register-return convention is wrong.
        // np461 ★ 出参是**指针**，不是返回寄存器：实测 x0/x1 都是客体堆地址（相隔 8 字节）。
        // 原来只把句柄放在 x1/x2 返回 ⇒ SDK 的结构体里从来没被写过 ⇒ 读端是栈垃圾（实测 0x4），
        // 于是 SDK 的等待在 100% CPU 上空转、看门狗 15 秒报"卡在 nvn:EndRecording"。
        // 现在按 Horizon 的约定写回：*x0 = 写端, *x1 = 读端（两个都写，日志里能核对方向）。
        // np461: the out params are POINTERS, not return registers (x0/x1 are guest-heap addresses 8 bytes
        // apart). Returning them in x1/x2 left the SDK's struct untouched, so the read end was stack garbage
        // (0x4 observed) and the SDK's wait spun at 100% CPU. Write back per Horizon: *x0 = write, *x1 = read.
        if (regs[0]) *(uint32_t *)regs[0] = hw;
        if (regs[1]) *(uint32_t *)regs[1] = hr;
        { static int n45; if (n45 < 5) { n45++;
            LOG("SVC 0x45 CreateEvent #%d: x0=%#llx x1=%#llx x2=%#llx x3=%#llx → 写端 %#x 读端 %#x",
                n45, regs[0], regs[1], regs[2], regs[3], hw, hr); } }
        regs[0] = 0; regs[1] = hw; regs[2] = hr; return 1;
    }
    case 0x11: case 0x12: case 0x17: {  // SignalEvent(写端) / ClearEvent(任一端) / ResetSignal(读端，未触发报错) | SignalEvent(write end) / ClearEvent(either end) / ResetSignal(read end, error if not signaled)
        pthread_mutex_lock(&G);
        // SignalEvent 只认写端；ClearEvent 两端都认；ResetSignal 只认读端
        // SignalEvent accepts only the write end; ClearEvent accepts both ends; ResetSignal accepts only the read end
        uint32_t h = (uint32_t)regs[0];
        event_t *e = num == 0x11 ? h_peek(h, O_EVENT_W) : num == 0x17 ? h_peek(h, O_EVENT_R)
                   : (h_peek(h, O_EVENT_W) ? h_peek(h, O_EVENT_W) : h_peek(h, O_EVENT_R));
        int64_t r = 0;
        if (!e) { r = (num == 0x17 && h_peek(h, O_THREAD)) ? R_BAD_STATE : R_BAD_HANDLE;
                  if (r == R_BAD_HANDLE) LOG("事件 SVC %#x 句柄无效 %#x", num, h); }
        else if (num == 0x11) { e->signaled = 1; wake_sync_waiters(); }
        else if (num == 0x17 && !e->signaled) r = R_BAD_STATE;
        else e->signaled = 0;
        pthread_mutex_unlock(&G);
        regs[0] = (uint64_t)r; return 1;
    }
    case 0x18: {  // WaitSynchronization(→w1 下标; x1 句柄数组, w2 个数, x3 超时) | WaitSynchronization(→w1 index; x1 handle array, w2 count, x3 timeout)
        thr_t *t = cur(); const uint32_t *hs = (const uint32_t *)regs[1]; int n = (int)regs[2]; int64_t ns = (int64_t)regs[3];
        // np411：等到**线程句柄**时把目标线程的身份打出来（入口 / 参数 / 已退出否）——
        // np410 真机：主线程无限等 0x1003/0x1007（两个工作线程），要分辨"等它退出"还是"等它出错"。
        // np411: when waiting on a thread handle, print the target's identity (entry / arg / exited?) --
        // np410 on device: the main thread waits forever on 0x1003/0x1007 (two workers); distinguish
        // "waiting for it to exit" from "waiting for something that already went wrong".
        // np419：把上限提到 64（卡住后的等待也要看），且**所有句柄类型都打**——
        // np413 之后怀疑主线程等的可能不是线程而是事件/传输对象；只打 O_THREAD 会漏。
        // np419: raise the cap to 64 (stall-phase waits matter too) and print EVERY handle's type --
        // the main thread may be waiting on an event / transfer object, not a thread; O_THREAD-only
        // logging would miss that.
        // np457：**默认关闭**。空转（等一个永远不来的对象）时这条每秒能刷上万行，
        // 12 秒就能把 hilog 冲到 29 万行、把真正有用的早期记录（未知 SVC 首现、断言文本）全冲掉。
        // 要排查等待就开 DM_WAITLOG=1。
        // np457: off by default. In a spin (waiting for something that never arrives) this logs tens of
        // thousands of lines per second -- 290k lines in 12 s, wiping the early records (first-seen unknown
        // SVCs, assertion text). Set DM_WAITLOG=1 when investigating waits.
        static int waitlog = -1;
        if (waitlog < 0) { const char *e = getenv("DM_WAITLOG"); waitlog = (e && *e && *e != '0') ? 1 : 0; }
        // np460：**越界句柄单独记一次**（不受 DM_WAITLOG 控制），并且带上调用者。
        // 为什么要单列：空转的根源就是它——我们的等待对非法句柄立刻返回错误，游戏立刻重试，100% CPU，
        // 看门狗 15 秒后报"无进展"。只看句柄值不知道是谁传的，所以必须带 lr。
        // np460: report an out-of-range handle once (independent of DM_WAITLOG) together with the caller.
        // This is the spin's root: we return an error immediately for an invalid handle, the guest retries
        // at once, CPU goes to 100% and the watchdog reports "no progress" after 15 s.
        for (int i = 0; i < n && i < 4; i++)
            if ((unsigned)(hs[i] - 0x1000) >= MAX_H) {
                static int once; if (!once++) {
                    module_t *mc = module_of((void *)regs[30]);
                    LOG("WaitSync 越界句柄 h=%#x 调用者=%s+%#llx 超时=%lld n=%d",
                        hs[i], mc ? mc->name : "?", mc ? (uint64_t)regs[30] - (uint64_t)mc->base : (uint64_t)regs[30],
                        (long long)ns, n);
                    // np462：这里**不能**调 dm_backtrace_scan —— SVC 现场没有可信的 fp，实测它自己就崩（pc=dm_backtrace_scan+0x8c）。
                }
            }
        { static unsigned nt; if (waitlog && nt++ % 50 == 0) {   // np419b：采样式 | sampled
            for (int i = 0; i < n && i < 4; i++) {
                unsigned i_ = hs[i] - 0x1000;
                if (i_ >= MAX_H) { LOG("WaitSync 句柄越界 h=%#x（非 0x10xx 段？）超时=%lld", hs[i], (long long)ns); continue; }   // np419c：越界也打——句柄可能根本不在 0x10xx 客体段 | log even out-of-range handles
                if (g_h[i_].type == O_THREAD) {
                    thr_t *w = (thr_t *)g_h[i_].obj;
                    module_t *me_ = module_of((void *)w->entry);
                    LOG("WaitSync 等线程 h=%#x entry=%s+%#llx arg=%#llx started=%d exited=%d 超时=%lld",
                        hs[i], me_ ? me_->name : "?", me_ ? (uint64_t)w->entry - (uint64_t)me_->base : 0,
                        (unsigned long long)w->arg, w->started, w->exited, (long long)ns);
                } else if (g_h[i_].type == O_EVENT_R || g_h[i_].type == O_EVENT_W) {
                    event_t *e = (event_t *)g_h[i_].obj;
                    LOG("WaitSync 等事件 h=%#x signaled=%d 超时=%lld", hs[i], e->signaled, (long long)ns);
                } else {
                    LOG("WaitSync 等其他 h=%#x 类型=%d 超时=%lld", hs[i], g_h[i_].type, (long long)ns);
                }
            }
        } }
        // np457：默认关闭（空转时它才是真正的刷屏源：12 秒 12 万行，把早期记录全冲掉）。要开用 DM_WAITLOG=1。
        // np457: off by default -- in a spin this is the real flood source (120k lines in 12 s). DM_WAITLOG=1 enables it.
        { static unsigned nc; static int wl = -1;
          if (wl < 0) { const char *e = getenv("DM_WAITLOG"); wl = (e && *e && *e != '0') ? 1 : 0; }
          if (wl && nc++ % 50 == 0) LOG("WaitSynchronization 调用：n=%d 超时=%lld 首句柄=%#x", n, (long long)ns, n > 0 ? hs[0] : 0); }
        // np425：CreateThread 的 arg 结构首 8 字节 = 真实线程函数指针（0x8b2d8 是统一包装）。
        // 打一次每个线程的真实入口，分辨哪个是 SystemTask 的派发线程。
        // np425: the first 8 bytes of CreateThread's arg struct hold the REAL thread function
        // (0x8b2d8 is just the SDK wrapper). Log each thread's real entry once, to tell which one
        // is the SystemTask dispatcher.
        // 只在等假服务句柄时：它们永远不会有服务端来触发，**无限等就是死锁** ⇒ 上限 100 ms，之后回超时。
        // 有真句柄混在里面时不动（那种等待是靠真事件触发的）。
        // Only when every handle is a fake service handle: nothing will ever signal them, so an infinite wait
        // would deadlock -- cap it at 100 ms and then report a timeout. Untouched when a real handle is mixed in.
        { int all_fake = n > 0; for (int i = 0; i < n; i++) if (!fake_service_handle(hs[i])) { all_fake = 0; break; }
          if (all_fake) { static int logged; if (!logged++) LOG("WaitSynchronization：等的是假服务句柄 %#x（无服务端）→ 最长 100 ms 后回超时", hs[0]);
                          if (ns < 0 || ns > 100000000ll) ns = 100000000ll; } }
        int64_t deadline = ns > 0 ? (dm_detclock_enabled() ? (int64_t)guest_ns() : now_mono_ns()) + ns : 0;
        pthread_mutex_lock(&G);
        int64_t r; int idx = -1;
        for (;;) {
            if (t->cancel) { t->cancel = 0; r = R_CANCELLED; break; }
            int bad = 0;
            for (int i = 0; i < n && idx < 0; i++) if (sig_obj(hs[i], &bad)) idx = i;
            if (bad) {   // np385：坏句柄是 SDK 断言自杀的常见根（TryWaitSystemEvent 等）；把号码打出来才查得下去
                         static int nbad; if (nbad++ < 8) { LOG("WaitSynchronization 坏句柄：n=%d 超时=%lld 句柄=", n, (long long)ns);
                             for (int i = 0; i < n && i < 4; i++) LOG("    [%d] %#x", i, hs[i]); }
                         r = R_BAD_HANDLE; break; }
            if (idx >= 0) {
                // np469：消费语义——被假时钟触发过的事件，等待成功即清除（真机 auto-clear 事件的行为）
                // np469: consume semantics -- an event signalled by the fake clock is cleared once a wait succeeds.
                event_t *ce = h_peek(hs[idx], O_EVENT_R);
                if (!ce) ce = h_peek(hs[idx], O_EVENT_W);
                if (ce && ce->auto_clear) ce->signaled = 0;
                r = 0; break;
            }
            int64_t left = ns < 0 ? -1 : ns == 0 ? 0 : deadline - (dm_detclock_enabled() ? (int64_t)guest_ns() : now_mono_ns());
            if (ns > 0 && left <= 0) { r = R_TIMEOUT; break; }
            if (block(t, 4, left) == R_TIMEOUT) { r = R_TIMEOUT; break; }
        }
        pthread_mutex_unlock(&G);
        regs[0] = (uint64_t)r; regs[1] = idx >= 0 ? (uint64_t)idx : 0; return 1;
    }
    case 0x19: {  // CancelSynchronization(w0 线程) | CancelSynchronization(w0 thread)
        pthread_mutex_lock(&G);
        thr_t *t = h_get((uint32_t)regs[0], O_THREAD);
        if (t) { t->cancel = 1; if (t->wkind == 4) wake(t, -1); }
        pthread_mutex_unlock(&G);
        regs[0] = t ? 0 : R_BAD_HANDLE; return 1;
    }
    case 0x15: {   // CreateTransferMemory(→x1 句柄; x1 地址, x2 大小, x3 权限)
        // np458 ★ 1.6.0 启动卡死的直接原因：这个 SVC 原来落到默认分支——"返回成功但**不写句柄**"，
        // 游戏拿到的是栈上的垃圾（实测 0x4），随后一直 WaitSynchronization(0x4) 空转，
        // 看门狗报「第 15 秒起无进展，卡在 拍 0 / 阶段 nvn:EndRecording」。
        // np458: the direct cause of the 1.6.0 boot stall. This SVC used to hit the default branch --
        // "success but no handle written" -- so the guest got stack garbage (0x4 observed) and spun in
        // WaitSynchronization(0x4); the watchdog reported no progress from second 15, stuck in nvn:EndRecording.
        uint32_t *out = (uint32_t *)regs[0];
        pthread_mutex_lock(&G);
        uint32_t h = h_new(O_TRANSFER, (void *)(uintptr_t)regs[1]);
        pthread_mutex_unlock(&G);
        if (out) *out = h;
        LOG("CreateTransferMemory(addr=%#llx size=%#llx perm=%#llx) → 句柄 %#x", regs[1], regs[2], regs[3], h);
        regs[0] = 0; regs[1] = h; return 1;
    }
    case 0x16: {  // CloseHandle(w0)：事件两端都关才释放；线程句柄关闭后槽位回收，对象在线程也退出后释放 | CloseHandle(w0): an event is freed only when both ends are closed; after a thread handle is closed the slot is recycled, and the object is freed once the thread has also exited
        pthread_mutex_lock(&G);
        unsigned i = (uint32_t)regs[0] - 0x1000;
        if (i > 0 && i < MAX_H && (g_h[i].type == O_EVENT_R || g_h[i].type == O_EVENT_W)) {
            event_t *e = g_h[i].obj; if (--e->refs == 0) free(e);
            g_h[i].type = O_FREE; g_h[i].obj = NULL;
        } else if (i > 0 && i < MAX_H && g_h[i].type == O_TRANSFER) {
            g_h[i].type = O_FREE; g_h[i].obj = NULL;      // np458：转移内存句柄只占槽位，关掉即回收
        } else if (i > 0 && i < MAX_H && g_h[i].type == O_THREAD) {
            thr_t *t = g_h[i].obj;
            g_h[i].type = O_FREE; g_h[i].obj = NULL;
            t->closed = 1;
            if (t->exited) thr_free(t);
        }
        pthread_mutex_unlock(&G);
        regs[0] = 0; return 1;
    }
    }
    return 0;
}

// ---------- 服务层的 SystemEvent ----------
// 真机：服务把内核事件的读端交给客户端，客体经 SDK nn::os 等待。宿主：在客体线程上调 SDK 自己的
// CreateSystemEvent(跨进程=true)，SDK 经 SVC 0x45 建出内核事件；记下写端句柄，之后任何线程（含宿主音频线程）都可触发。
// ---------- SystemEvent of the service layer ----------
// Real device: the service hands the read end of a kernel event to the client, and the guest waits on it through SDK nn::os. Host: on a guest thread, call the SDK's own
// CreateSystemEvent(inter-process=true), and the SDK creates the kernel event through SVC 0x45; the write-end handle is recorded, after which any thread (including the host audio thread) can signal it.
static struct { void *ev; uint32_t wh; } g_se[512];
static int g_nse;
static pthread_mutex_t g_se_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t se_find(void *ev) {
    for (int i = 0; i < g_nse; i++) if (g_se[i].ev == ev) return g_se[i].wh;
    return 0;
}
void hle_bind_system_event(void *ev, int autoclear) {      // 必须在客体线程上调（要执行 SDK 代码） | Must be called on a guest thread (it has to execute SDK code)
    pthread_mutex_lock(&g_se_lock); uint32_t wh = se_find(ev); pthread_mutex_unlock(&g_se_lock);
    if (wh) return;
    if (!pthread_getspecific(g_keythr)) die("hle_bind_system_event 须在客体线程上调用");
    static uint32_t (*create)(void *, int, int); static uint32_t (*getw)(const void *);
    if (!create) { create = guest_sym("_ZN2nn2os17CreateSystemEventEPNS0_15SystemEventTypeENS0_14EventClearModeEb");
                   getw = guest_sym("_ZN2nn2os30GetWritableHandleOfSystemEventEPKNS0_15SystemEventTypeE"); }
    uint32_t r = create(ev, autoclear, 1);
    if (r) die("CreateSystemEvent 失败 %#x", r);
    wh = getw(ev);
    // np459：把 SDK 建出来的写端句柄打出来。若是垃圾值（例如 0x4），说明 SDK 的 CreateSystemEvent
    // 内部那条"建内核事件"的路径没真正生效 —— 后面的 WaitSynchronization 就会空转在那个值上。
    // np459: log the write-end handle the SDK produced. A garbage value (e.g. 0x4) means the SDK's
    // CreateSystemEvent never really created the kernel event, and later waits spin on that value.
    LOG("SystemEvent 绑定：ev=%p 写端句柄=%#x", ev, wh);
    pthread_mutex_lock(&g_se_lock);
    if (g_nse >= (int)(sizeof g_se / sizeof g_se[0])) die("SystemEvent 绑定表满");
    g_se[g_nse].ev = ev; g_se[g_nse++].wh = wh;
    pthread_mutex_unlock(&g_se_lock);
    { void se_tick_start(void); se_tick_start(); }   // np467：第一个事件绑定后就把假音频时钟起起来
}
// np467：**假音频时钟**。我们的 AudioRenderer 是空实现，没有任何回调去"放行已播缓冲"，
// 于是游戏卡在 `WaitSync 等事件 h=0x1003/0x1012 signaled=0 超时=-1`（那两个句柄正是
// hle_bind_system_event 建出来的读端）。这里起一个宿主线程，按 ~60 Hz 把**所有已绑定的**
// SystemEvent signal 一遍，模拟"缓冲被消费"的节奏；事件是 auto-clear 的，多 signal 无害。
// 关掉用 DM_SE_TICK=0。
// np467: fake audio clock. Our AudioRenderer is a stub, so nothing ever releases played buffers and the
// guest blocks in WaitSync on the read ends (0x1003 / 0x1012) forever. A host thread signals every bound
// SystemEvent at ~60 Hz to emulate buffer consumption; the events are auto-clear so extra signals are
// harmless. DM_SE_TICK=0 disables it.
static void *se_tick(void *a) {
    (void)a;
    unsigned tick = 0;
    for (;;) {
        usleep(16000);
        // np555 ★ 借这个"一定会被调度"的线程做**现场导出节拍**：每 313 拍（≈5 秒）打一次各客体线程状态。
        //   为什么要借它：本平台 `pthread_create` 出来的宿主线程体常常不被调度（diag.c 的采样器就是这样，
        //   np555 开 DM_SAMPLE/DM_HIST 一条都没输出），而这个线程每 16 ms 醒一次、实测一定跑。
        //   卡死时（np554：最后一条命令 op=19 之后没有下文）这五行就是唯一能回答"谁在等谁"的读数。
        //   只在 DM_STATE_TICK=1 时打，避免常态刷屏。
        // np555: use this thread -- which is definitely scheduled -- as a state-dump tick: every 313 ticks (~5 s) print
        //   the state of every guest thread. Host threads spawned with pthread_create are often never scheduled on this
        //   platform (that is why diag.c's sampler printed nothing in np555 with DM_SAMPLE/DM_HIST), while this one wakes
        //   every 16 ms and does run. During a stall (np554: nothing after command op=19) these lines are the only reading
        //   that answers "who waits for whom". Only under DM_STATE_TICK=1, so normal runs stay quiet.
        if ((++tick % 313) == 0 && getenv("DM_STATE_TICK")) {
            extern void dm_thread_state_dump(const char *why);
            dm_thread_state_dump("tick");
        }
        void *evs[64]; int n = 0;
        pthread_mutex_lock(&g_se_lock);
        for (int i = 0; i < g_nse && n < 64; i++) evs[n++] = g_se[i].ev;
        pthread_mutex_unlock(&g_se_lock);
        for (int i = 0; i < n; i++) {
            // np469：先打"自动清除"标记再 signal —— 否则事件一直是 signaled，
            // 那个音频线程 20 秒发了 1601 万次 svc 0x18（每次立即返回）把 CPU 烧满。
            // np469: mark auto-clear before signalling; otherwise the event stays signalled forever and the
            // audio thread spun 16M svc 0x18 calls in 20 s.
            pthread_mutex_lock(&G);
            event_t *e = h_peek(se_find(evs[i]), O_EVENT_W);
            if (e) e->auto_clear = 1;
            pthread_mutex_unlock(&G);
            hle_signal_system_event(evs[i]);
        }
    }
    return NULL;
}
void se_tick_start(void) {
    static int started;
    if (started) return;
    const char *e = getenv("DM_SE_TICK");
    if (e && *e == '0') { LOG("【黄】假音频时钟已关（DM_SE_TICK=0）"); return; }
    started = 1;
    pthread_t t; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setstacksize(&at, 256 << 10);
    if (!pthread_create(&t, &at, se_tick, NULL)) LOG("假音频时钟已起：每 16ms signal 一遍已绑定的 SystemEvent（DM_SE_TICK=0 可关）");
    else LOG("【红】假音频时钟起线程失败");
}

void hle_signal_system_event(void *ev) {   // 未绑定的按手动清除即时绑定（只能在客体线程上发生） | An unbound one is bound on the spot as manual-clear (can only happen on a guest thread)
    pthread_mutex_lock(&g_se_lock); uint32_t wh = se_find(ev); pthread_mutex_unlock(&g_se_lock);
    if (!wh) { hle_bind_system_event(ev, 0); pthread_mutex_lock(&g_se_lock); wh = se_find(ev); pthread_mutex_unlock(&g_se_lock); }
    pthread_mutex_lock(&G);
    event_t *e = h_get(wh, O_EVENT_W);
    if (!e) die("SystemEvent 写端句柄失效 %#x", wh);
    e->signaled = 1; wake_sync_waiters();
    pthread_mutex_unlock(&G);
}
