// nn::os 消息队列的宿主记账（+ 可选的轮询收）。**不改行为**，只在真实现外面记一笔。
//
// 为什么需要（2026-10-28）
// ------------------------
// 启动卡在 ksys::InitializeApp Step 4，11 个 worker 全阻塞在 `nn::os::ReceiveMessageQueue`。
// 之前三轮对"队列里到底有没有货"的判读互相矛盾，根因是**证据不足**：
//   · np414 用「全进程 0 次 svc 0x35（SignalToAddress）」推断"没人发消息"——**这条推断不成立**：
//     队列的唤醒走 `InternalConditionVariableImplByHorizon::Signal` → **svc 0x1D**
//     （sdk+0x854bc 的 SendMessageQueue 里 `add x0,q,#0x40; bl Signal`），跟 0x35 无关。
//     而同一份日志里 svc 0x1D 有 438 次（当时被误记成 SetThreadCoreMask，那是 0x0F）。
//   · 客体侧探针要么被 hilog 回绕吃掉，要么只能"前 N 次"，高频路径上等于失明。
// 所以把仪器放到**宿主这一层**：每次 Send/Receive 都过这里，计数是原子的、开销可忽略；
// 汇总按固定间隔打一行（短行、低频，不会被回绕吃掉）；卡住时再随现场导出打全表。
//
// 判读（下一轮真机就看这几行）
// ----------------------------
//   DM_MQ=1，每 5 秒一行：
//     `MQ 汇总：… 发 0 …`      ⇒ 真的没有生产者（轮询救不了，得去查谁该发）
//     `MQ 汇总：… 发 >0 …` 而收方仍阻塞 ⇒ **唤醒链断了**（轮询能救）
//   每个队列首次被收发时打一行，带**调用者**（模块+偏移）——那就是"谁发的 / 谁收的"。
//
// 开关
// ----
//   DM_MQ=1        打开周期汇总与首次触达行（缺省关：产品局零噪声、零内存）
//   DM_MQ_POLL=1   阻塞收改轮询（TryReceive + 100ms）——即 np428 定的 P0 修法。
//                  语义不变（有货就返回、stop 消息照样收到），只是不再依赖"有人唤醒"。
//                  写成 `DM_MQ_POLL=30` 则**进程起来 30 秒后才切**：一次真机跑完 A/B
//                  （前 30 秒照旧阻塞 = 看队列里到底有没有货，之后轮询 = 看能不能救）。
//
// 队列对象布局（从 sdk+0x854bc / 0x859e4 反汇编读出，MessageQueueType）
//   +0x00 非满等待者链  +0x10 非空等待者链  +0x20 缓冲  +0x28 容量  +0x2c 元素数
//   +0x30 读下标        +0x38 临界区        +0x3c 非满条件  +0x40 非空条件
// ---- Host-side accounting for nn::os message queues (plus an optional polling receive). It changes no behaviour: it only records a tally around the real implementation.
// Why (2026-10-28): startup stalls in ksys::InitializeApp Step 4 with 11 workers blocked in nn::os::ReceiveMessageQueue, and three rounds of readings disagreed on "is there anything in the queue". The np414 inference from "0 svc 0x35 process-wide" does not hold: the queue wake-up goes through InternalConditionVariableImplByHorizon::Signal -> svc 0x1D (inside SendMessageQueue at sdk+0x854bc: `add x0,q,#0x40; bl Signal`), nothing to do with 0x35 -- and the same log has 438 svc 0x1D (recorded then as SetThreadCoreMask, which is 0x0F).
// So the instrument moves into the host layer: every Send/Receive passes here, counting is atomic and cheap; the summary is one short line at a fixed interval (never eaten by the hilog ring), and a stall dump prints the whole table.
// Reading it: with DM_MQ=1, one line every 5 s. `发 0` = there really is no producer (polling cannot help); `发 > 0` while receivers still block = the wake-up chain is broken (polling does help). The first send/receive per queue logs the caller (module+offset), which is who sends and who receives.
// Switches: DM_MQ=1 enables the periodic summary and first-touch lines (off by default: zero noise, zero memory in product runs); DM_MQ_POLL=1 turns the blocking receive into a poll (TryReceive + 100 ms), i.e. the P0 fix decided in np428 -- semantics unchanged, it just no longer depends on being woken.
// Queue object layout (read off the disassembly of sdk+0x854bc / 0x859e4, MessageQueueType): +0x00 not-full waiter list, +0x10 not-empty waiter list, +0x20 buffer, +0x28 capacity, +0x2c count, +0x30 read index, +0x38 critical section, +0x3c not-full condition, +0x40 not-empty condition.
#include "hle.h"
#include <stdatomic.h>
#include <time.h>
#include <unistd.h>

int dm_ptr_in_as(const void *p);   // hle_svc.c：指针是否落在客体地址空间里 | hle_svc.c: is the pointer inside the guest address space

#define MQ_MAX 64
typedef struct {
    void *q;
    unsigned sends, trysends, jams, recvs, tryrecvs;
    atomic_int blocked;                 // 此刻有几个线程阻塞在这个队列上 | how many threads are blocked on this queue right now
    unsigned long long wait_ns;         // 累计阻塞时长 | accumulated blocked time
    int cap, count;                     // 最近一次看到的容量 / 元素数 | capacity / element count as last seen
    uint64_t first_lr;                  // 首次触达的调用者 | caller at first touch
} mq_t;
static mq_t g_mq[MQ_MAX];
static int g_nmq;
static pthread_mutex_t g_mq_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic unsigned long long g_mq_sends, g_mq_recvs, g_mq_polls;

static int mq_on(void) { static int v = -1; if (v < 0) v = getenv("DM_MQ") != NULL; return v; }
// DM_MQ_POLL 可以带一个秒数：`DM_MQ_POLL=60` = 进程起来 60 秒后才切轮询。
// 这样**一次真机就能做 A/B**：前 60 秒是原样（阻塞，看队列里到底有没有货），之后是轮询版。
// DM_MQ_POLL may carry a number of seconds: `DM_MQ_POLL=60` switches to polling 60 s after the process starts.
// That makes one device round enough for the A/B: the first 60 s run as before (blocking, showing whether the queue
// ever has anything in it), and polling starts after that.
static int mq_poll(void) {
    static int v = -1, delay, said;
    static uint64_t t0;
    if (v < 0) {
        const char *e = getenv("DM_MQ_POLL");
        v = e != NULL;
        // ⚠ `=1` 必须理解成"立刻"：早期版本按 atoi 取秒数，`DM_MQ_POLL=1` 变成"1 秒后才切"，
        // 而 12 个 worker 在 0.5 秒内就已经阻塞在 sdk 原实现里、永远不会重新进 Receive ⇒
        // 轮询一次都没生效（np429 真机两轮都是 `轮询 0`）。只有 >1 才当秒数。
        // ⚠ `=1` must mean "immediately": the first version read it as seconds, so `DM_MQ_POLL=1` became
        // "switch after 1 s" -- but the 12 workers block inside the sdk implementation within 0.5 s and never
        // re-enter Receive, so polling never took effect (both np429 device runs showed `轮询 0`).
        // Only values > 1 are treated as seconds.
        delay = (e && *e >= '0' && *e <= '9') ? atoi(e) : 0;
        if (delay <= 1) delay = 0;
        t0 = now_ns();
    }
    if (!v) return 0;
    if (!delay) { if (!said++) LOG("MQ 轮询收生效（DM_MQ_POLL，无延迟）：worker 不再依赖唤醒"); return 1; }
    if (now_ns() - t0 < (uint64_t)delay * 1000000000ull) return 0;
    if (!said++) LOG("MQ 轮询收生效（DM_MQ_POLL=%d，第 %d 秒起）：此后的收发与之前对比即 A/B", delay, delay);
    return 1;
}

// 队列对象只有在客体地址空间里才去读字段（D2R 与 BotW 的队列都在客体堆上）
// Only read the fields when the queue object lives in the guest address space (both D2R's and BotW's queues sit on the guest heap)
static void mq_probe(mq_t *e) {
    if (!e || !dm_ptr_in_as(e->q)) return;
    e->cap = (int)((const uint32_t *)e->q)[0x28 / 4];
    e->count = (int)((const uint32_t *)e->q)[0x2c / 4];
}

// 取（或建）一条队列的记录。只在 DM_MQ 打开时建表：产品局不占内存、不加锁。
// Get (or create) the record for a queue. The table is only built when DM_MQ is on: no memory and no locking in product runs.
static mq_t *mq_slot(void *q, const char *what, uint64_t lr) {
    if (!mq_on()) return NULL;
    pthread_mutex_lock(&g_mq_lock);
    mq_t *e = NULL;
    for (int i = 0; i < g_nmq; i++) if (g_mq[i].q == q) { e = &g_mq[i]; break; }
    if (!e && g_nmq < MQ_MAX) {
        e = &g_mq[g_nmq++];
        e->q = q; e->first_lr = lr;
        module_t *m = module_of((void *)(uintptr_t)lr);
        mq_probe(e);
        LOG("MQ 新队列 %p（%s）调用者 %s+%#llx 容量 %d 现有 %d", q, what,
            m ? m->name : "?", m ? (unsigned long long)(lr - (uint64_t)m->base) : 0, e->cap, e->count);
    }
    pthread_mutex_unlock(&g_mq_lock);
    return e;
}

// 原实现（sdk 自己的 nn::os）。首次用时解析：符号表装载后才在。
// The original implementation (the sdk's own nn::os). Resolved on first use: the symbol table only exists after loading.
static void *mq_sym(const char *name) {
    void *p = guest_sym(name);
    if (!p) LOG("【红】消息队列：sdk 里找不到 %s", name);
    return p;
}
#define MQ_LR ((uint64_t)(uintptr_t)__builtin_return_address(0))

// ---- 发送 ---------------------------------------------------------------
HLE_FORCE("_ZN2nn2os16SendMessageQueueEPNS0_16MessageQueueTypeEm",
          "记账：谁在发消息（判断 worker 队列有没有生产者）", void, h_mq_send, void *q, uint64_t msg) {
    static void (*orig)(void *, uint64_t);
    if (!orig) { orig = (void (*)(void *, uint64_t))mq_sym("_ZN2nn2os16SendMessageQueueEPNS0_16MessageQueueTypeEm"); if (!orig) return; }
    mq_t *e = mq_slot(q, "发", MQ_LR);
    orig(q, msg);
    atomic_fetch_add(&g_mq_sends, 1);
    if (e) { e->sends++; mq_probe(e); }
}
HLE_FORCE("_ZN2nn2os19TrySendMessageQueueEPNS0_16MessageQueueTypeEm",
          "记账：非阻塞发（游戏的任务派发走这条时能看到）", int, h_mq_trysend, void *q, uint64_t msg) {
    static int (*orig)(void *, uint64_t);
    if (!orig) { orig = (int (*)(void *, uint64_t))mq_sym("_ZN2nn2os19TrySendMessageQueueEPNS0_16MessageQueueTypeEm"); if (!orig) return 0; }
    mq_t *e = mq_slot(q, "试发", MQ_LR);
    int r = orig(q, msg);
    if (r) atomic_fetch_add(&g_mq_sends, 1);
    if (e) { e->trysends++; if (r) mq_probe(e); }
    return r;
}
HLE_FORCE("_ZN2nn2os15JamMessageQueueEPNS0_16MessageQueueTypeEm",
          "记账：插队发（带内联数据的投递走这条）", void, h_mq_jam, void *q, uint64_t msg) {
    static void (*orig)(void *, uint64_t);
    if (!orig) { orig = (void (*)(void *, uint64_t))mq_sym("_ZN2nn2os15JamMessageQueueEPNS0_16MessageQueueTypeEm"); if (!orig) return; }
    mq_t *e = mq_slot(q, "插队发", MQ_LR);
    orig(q, msg);
    atomic_fetch_add(&g_mq_sends, 1);
    if (e) { e->jams++; mq_probe(e); }
}

// ---- 接收 ---------------------------------------------------------------
// DM_MQ_POLL=1：轮询版。TryReceive 是 sdk 自己的非阻塞收（sdk+0x85a90：自己上锁、取一条、
// 解锁、再唤醒"非满"等待者），所以这里不需要知道队列布局，也没有竞态。
// DM_MQ_POLL=1: the polling version. TryReceive is the sdk's own non-blocking receive (sdk+0x85a90; it locks,
// pops one, unlocks and wakes the not-full waiters), so nothing here needs the queue layout and there is no race.
HLE_FORCE("_ZN2nn2os19ReceiveMessageQueueEPmPNS0_16MessageQueueTypeE",
          "记账 + 可选轮询收（P0：worker 主动查队列，不依赖唤醒）", void, h_mq_recv, uint64_t *out, void *q) {
    static void (*orig)(uint64_t *, void *);
    if (!orig) { orig = (void (*)(uint64_t *, void *))mq_sym("_ZN2nn2os19ReceiveMessageQueueEPmPNS0_16MessageQueueTypeE"); if (!orig) return; }
    mq_t *e = mq_slot(q, "收", MQ_LR);
    atomic_fetch_add(&g_mq_recvs, 1);
    if (e) { e->recvs++; mq_probe(e); }
    if (!mq_poll()) {
        if (e) { atomic_fetch_add(&e->blocked, 1); uint64_t t0 = now_ns(); orig(out, q); e->wait_ns += now_ns() - t0; atomic_fetch_sub(&e->blocked, 1); mq_probe(e); }
        else orig(out, q);
        return;
    }
    static int (*try_)(uint64_t *, void *);
    if (!try_) try_ = (int (*)(uint64_t *, void *))mq_sym("_ZN2nn2os22TryReceiveMessageQueueEPmPNS0_16MessageQueueTypeE");
    if (!try_) { orig(out, q); return; }
    atomic_fetch_add(&g_mq_polls, 1);
    for (;;) {
        if (try_(out, q)) { if (e) { e->tryrecvs++; mq_probe(e); } return; }
        struct timespec ts = {0, 100 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
}
HLE_FORCE("_ZN2nn2os22TryReceiveMessageQueueEPmPNS0_16MessageQueueTypeE",
          "记账：非阻塞收（游戏自己的任务系统用它取任务）", int, h_mq_tryrecv, uint64_t *out, void *q) {
    static int (*orig)(uint64_t *, void *);
    if (!orig) { orig = (int (*)(uint64_t *, void *))mq_sym("_ZN2nn2os22TryReceiveMessageQueueEPmPNS0_16MessageQueueTypeE"); if (!orig) return 0; }
    mq_t *e = mq_slot(q, "试收", MQ_LR);
    int r = orig(out, q);
    if (r) atomic_fetch_add(&g_mq_recvs, 1);
    if (e) { e->tryrecvs++; if (r) mq_probe(e); }
    return r;
}

// ---- 建（只为了把队列登记进表，好在汇总里给它一个名字和容量） -------------------
// ---- init (only so the queue is registered and gets a name and capacity in the summary) ----
HLE_FORCE("_ZN2nn2os22InitializeMessageQueueEPNS0_16MessageQueueTypeEPmm",
          "登记队列（容量），便于汇总与现场导出", void, h_mq_init, void *q, uint64_t *buf, size_t n) {
    static void (*orig)(void *, uint64_t *, size_t);
    if (!orig) { orig = (void (*)(void *, uint64_t *, size_t))mq_sym("_ZN2nn2os22InitializeMessageQueueEPNS0_16MessageQueueTypeEPmm"); if (!orig) return; }
    orig(q, buf, n);
    mq_t *e = mq_slot(q, "建", MQ_LR);
    if (e) { e->cap = (int)n; e->count = 0; }
}

// ---- 汇总与现场导出 -----------------------------------------------------
// 一行式汇总：短、低频（5 秒），不会被 hilog 环形回绕吃掉。只在 DM_MQ 打开时起线程。
// One-line summary: short and infrequent (5 s), never eaten by the hilog ring. The thread only starts when DM_MQ is on.
static void *mq_ticker(void *arg) {
    (void)arg;
    for (;;) {
        struct timespec ts = {5, 0};
        nanosleep(&ts, NULL);
        unsigned long long s = atomic_load(&g_mq_sends), r = atomic_load(&g_mq_recvs), p = atomic_load(&g_mq_polls);
        int blocked = 0; for (int i = 0; i < g_nmq; i++) blocked += atomic_load(&g_mq[i].blocked);
        LOG("MQ 汇总：队列 %d 个  发 %llu  收 %llu  轮询 %llu  此刻阻塞 %d", g_nmq, s, r, p, blocked);
        for (int i = 0; i < g_nmq; i++) {
            mq_t *e = &g_mq[i]; mq_probe(e);
            LOG("   %p 容量 %d 现有 %d | 发 %u 试发 %u 插队 %u | 收 %u 试收 %u | 阻塞 %d 累计 %llums",
                e->q, e->cap, e->count, e->sends, e->trysends, e->jams, e->recvs, e->tryrecvs,
                atomic_load(&e->blocked), e->wait_ns / 1000000ull);
        }
    }
    return NULL;
}
static pthread_once_t g_mq_once = PTHREAD_ONCE_INIT;
static void mq_start(void) {
    if (!mq_on()) return;
    pthread_t t;
    if (!pthread_create(&t, NULL, mq_ticker, NULL)) pthread_detach(t);
}
// 现场导出（崩溃 / 无进展看门狗 / 手动 dump 时调）：全表，不用等 5 秒。
// State dump (on crash / no-progress watchdog / manual dump): the whole table, no need to wait 5 s.
void dm_msgq_dump(void) {
    int n = g_nmq;
    if (!n) { LOG("MQ 现场：没有登记过队列（DM_MQ 未开时正常）"); return; }
    LOG("MQ 现场：队列 %d 个，发 %llu 收 %llu 轮询 %llu", n,
        atomic_load(&g_mq_sends), atomic_load(&g_mq_recvs), atomic_load(&g_mq_polls));
    for (int i = 0; i < n; i++) {
        mq_probe(&g_mq[i]);
        module_t *m = module_of((void *)(uintptr_t)g_mq[i].first_lr);
        LOG("   %p 容量 %d 现有 %d | 发 %u 试发 %u 插队 %u | 收 %u 试收 %u | 阻塞 %d 累计 %llums | 首次调用者 %s+%#llx",
            g_mq[i].q, g_mq[i].cap, g_mq[i].count, g_mq[i].sends, g_mq[i].trysends, g_mq[i].jams,
            g_mq[i].recvs, g_mq[i].tryrecvs, atomic_load(&g_mq[i].blocked), g_mq[i].wait_ns / 1000000ull,
            m ? m->name : "?", m ? (unsigned long long)(g_mq[i].first_lr - (uint64_t)m->base) : 0);
    }
}
// 由宿主启动链调一次（main.c）：只做"要不要起汇总线程"这一件事。
// Called once from the host boot chain (main.c): only decides whether to start the summary thread.
void dm_msgq_install(void) {
    pthread_once(&g_mq_once, mq_start);
    const char *p = getenv("DM_MQ_POLL");
    if (p) LOG("MQ 轮询收已配置（DM_MQ_POLL=%s）：%s", p, (*p >= '0' && *p <= '9') ? "到时切换，之前照旧阻塞" : "立即生效");
    if (mq_on()) LOG("MQ 记账已打开（DM_MQ）：每 5 秒一行汇总，队列首次触达时打调用者");
}
