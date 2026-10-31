// Horizon 内核系统调用（装 SDK 后由 SDK 运行时触发）。只实现 P1 启动链要的：内存布局 + 堆 + 查询 + 调试输出 + 计时。
// 寄存器约定：入参 w1.. / 出参 w0=Result、x1=值（GetInfo / SetHeapSize），与公开 Horizon ABI 一致。
// Horizon kernel system calls (triggered by the SDK runtime once the SDK is loaded). Only what the P1 boot chain needs is implemented: memory layout + heap + query + debug output + timing.
// Register convention: inputs w1.. / outputs w0=Result, x1=value (GetInfo / SetHeapSize), consistent with the public Horizon ABI.
#include "dm.h"
#include "hle.h"
#include <stdatomic.h>
#include <sys/mman.h>
#include <errno.h>
#include <unistd.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>

// 预留一整块客体地址空间，按 Horizon 分区（地址值直接用宿主 mmap 得到的真地址）。只占地址、不占内存。
// 大小从 5 GB 起降档重试（部分设备预留 5 GB 返回 ENOMEM(12)），下限 4864 MiB（游戏要 1 GB 别名+栈 + 3728 MB 堆，4864 − 1024 = 3840 > 3728 + 余量）。
// iOS 上不带「扩展虚拟寻址」权限时上限随设备型号不同（实测：有的 6 GB 成、有的 5120 MiB 不成，需要降到 4864 MiB）——
// 留到 8 GB 的话，签不了那项权限的开发者账号（免费账号）出的包在这一步就退出。
// Reserve one contiguous guest address space, partitioned like Horizon (address values are the real addresses returned by the host mmap). It only takes address space, not memory.
// The size starts at 5 GB and steps down on retry (on some devices reserving 5 GB returns ENOMEM(12)), floor 4864 MiB (the game needs 1 GB alias+stack + a 3728 MB heap; 4864 − 1024 = 3840 > 3728 + margin).
// On iOS without the extended virtual addressing entitlement the limit depends on the device model (measured: 6 GB works on some, 5120 MiB fails on others and must drop to 4864 MiB) —
// reserving up to 8 GB would make builds from developer accounts that cannot sign that entitlement (free accounts) exit at this step.
static size_t g_reserve;           // 实际预留大小（逐档降级后成功的值） | Actual reserved size (the value that succeeded after stepping down)
static uint8_t *g_as;              // ASLR 区基址 = 整块 | ASLR region base = the whole block
static uint8_t *g_heap; static size_t g_heap_cur, g_heap_cap; static int g_heap_rejects;
// 地址段整个落在客体堆当前已开放（可读写）的范围内：这种读不用陷内核核对，直接 memcpy（diag.c tk_rd 等每拍读客体内存的地方用）
// The address range lies entirely within the currently opened (readable and writable) part of the guest heap: such reads need no kernel trap to check, memcpy directly (used where guest memory is read every tick, such as diag.c tk_rd)
int dm_guest_heap_readable(uint64_t a, size_t n) { uint64_t b = (uint64_t)(uintptr_t)g_heap; size_t cur = g_heap_cur; return g_heap && a >= b && n <= cur && a - b <= cur - n; }
// np249：客体主线程的**真实**栈范围（由 main.c 用 pthread_attr_setstack 放进 g_stack 之后登记）。
// 为什么要有它：nn::os 把"主线程的栈"记进地址空间账，并用 GetCurrentStackInfo 断言 sp 落在这个范围内。
// 只要 sp 真在这个区间，两条就同时满足，不用对客体撒谎。
static uint8_t *g_main_stack_lo; static size_t g_main_stack_size;
static uint8_t *g_alias; static size_t g_alias_cap;
static uint8_t *g_stack; static size_t g_stack_cap;
// np249：让 main.c 能在起客体主线程**之前**就把地址空间（含 stack 区）建好 —— 否则
// 拿到的是 NULL（g_stack 原本要等 nnosInitialize 里的 mem_init 才建立，真机实测报"栈区不足 0 +0"）。
void dm_guest_address_space_init(void);
uint8_t *dm_guest_stack_region(size_t *cap) { if (cap) *cap = g_stack_cap; return g_stack; }
void dm_guest_set_main_stack(void *lo, size_t size) { g_main_stack_lo = lo; g_main_stack_size = size; }
// np549 ★ 把客体栈区整片提交为可读写。
// 为什么：客体地址空间是一整块 `mmap(PROT_NONE)` 预留，栈区在 `g_as+0x30000000`（256 MB），
// 而客体线程的栈顶由游戏给（CreateThread 的 x3），落到这段预留里。真机实测：
// 线程 #1 的栈顶 `0x5e4f64bfe0` 距栈区起点只有 0x3FE0（16 KB），它往下一长就踩到**没提交**的页，
// 而 `si_code=2`（ACCERR，页在但没权限）不是 MAPERR —— 正是"PROT_NONE 预留没提交"的特征。
// `mprotect` 只改权限位，页仍然按需分配 ⇒ 256 MB 不占实际内存；也因此比"给每个线程算一段"更简单、
// 更接近真机（真机上整块 RAM 都可读写，栈溢出是悄悄踩坏别的东西，而不是当场缺页）。
// np549: commit the whole guest stack region read/write. The guest address space is one PROT_NONE
// reservation and the stack region is the 256 MB at g_as+0x30000000; guest thread stacks land there with
// the top supplied by the game. On device thread #1's top sat 0x3FE0 (16 KB) above the region start, and
// the first growth touched an uncommitted page -- si_code=2 (ACCERR, page present but not permitted), the
// signature of "PROT_NONE reservation, never committed". mprotect only changes permission bits and pages
// stay demand-allocated, so the 256 MB costs nothing until used; it is also closer to the hardware, where
// the whole RAM is read/write and a stack overflow silently corrupts instead of faulting.
static void mem_init(void);   // np549：下面 dm_guest_commit_stack 要先用到它 | used by dm_guest_commit_stack below
void dm_guest_commit_stack(void) {
    mem_init();
    if (!g_stack || !g_stack_cap) return;
    static int done; if (done) return; done = 1;
    int rc = mprotect(g_stack, g_stack_cap, PROT_READ | PROT_WRITE);
    if (rc) LOG("【红】提交客体栈区失败 errno=%d（%s）", errno, strerror(errno));
    else LOG("客体栈区已提交可读写：%p +%zu MiB", (void *)g_stack, g_stack_cap >> 20);
    // np553：栈区**底部再留 1 MB 溢出余量**。
    //   实测客体线程 #1 的栈顶只比栈区起点高 0x3FE0（16 KB），它用了 0x4A60 就掉到栈区外面
    //   （sp 比起点低 0xA80，落在 alias 区）⇒ ACCERR。真机上那里就是普通 RAM，溢出只是踩坏相邻数据、
    //   不会当场缺页；我们把它提交成可写，行为才和硬件一致（顺带把"栈恰好用穿"从崩溃变成可观测）。
    // np553: keep a 1 MB overflow margin below the stack region. Thread #1's stack top is only 0x3FE0
    //   (16 KB) above the region start and it used 0x4A60, so sp fell 0xA80 below the region (into the
    //   alias area) and faulted with ACCERR. On hardware that area is ordinary RAM and an overflow merely
    //   corrupts neighbours; committing it keeps our behaviour close to the hardware.
    if (g_alias && (uint8_t *)(g_stack - (1u << 20)) >= g_alias) {
        if (mprotect(g_stack - (1u << 20), 1u << 20, PROT_READ | PROT_WRITE) == 0)
            LOG("栈区底部溢出余量已提交 1 MiB");
        else LOG("【黄】栈区底部余量提交失败 errno=%d", errno);
    }
}
void *dm_guest_main_stack(size_t *size) { if (size) *size = g_main_stack_size; return g_main_stack_lo; }


static void mem_init(void) {
    if (g_as) return;
    // 从 5120 MiB 起逐档降级（每档减 128 MiB），下限 4864 MiB（堆容量 = reserve − 1 GB ≥ 3728 MB + 余量）
    // Step down from 5120 MiB (128 MiB per step), floor 4864 MiB (heap capacity = reserve − 1 GB ≥ 3728 MB + margin)
    const size_t start = 5ull << 30, step = 128ull << 20, limit = 4864ull << 20;
    for (size_t sz = start; sz >= limit; sz -= step) {
        g_as = mmap(NULL, sz, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (g_as != MAP_FAILED) { g_reserve = sz; break; }
        int err = errno; LOG("【红】reserve %zu MiB: mmap 失败 errno=%d (%s)，降档重试", sz >> 20, err, strerror(err));
        if (sz == limit) die("reserve guest AS: 试过 %zu–%zu MiB，全部失败", limit >> 20, start >> 20);
    }
    // 固定划分（都在预留块内）：alias 512MB、stack 256MB、heap 其余从 2GB 处起
    // Fixed partition (all inside the reserved block): alias 512MB, stack 256MB, heap the rest starting at 2GB
    g_alias = g_as + 0x10000000; g_alias_cap = 0x20000000;
    g_stack = g_as + 0x30000000; g_stack_cap = 0x10000000;
    g_heap  = g_as + 0x40000000; g_heap_cap  = g_reserve - 0x40000000;
    LOG("guest AS %p +%zu MiB（降档后实际预留 %zu MiB）  heap %p alias %p stack %p", g_as, g_reserve >> 20, g_reserve >> 20, g_heap, g_alias, g_stack);
}
void dm_guest_address_space_init(void) { mem_init(); }

// 应用内存池 = Switch（4 GB 机型）给应用的 3285 MB。游戏启动（main+0x1b66160）把 QueryMemoryInfo 报的总量整块设成堆、失败就减 2 MB 重试，
// 整块交给自己的 StandardAllocator（main+0x32137a0）——所以堆大小 = 池 − 映像占用，宿主报多少它就只有多少。
// 之前这里报 3 GB 且 SetHeapSize 从不失败：堆比真机少约 150 MB，进局高峰主堆只剩约 18 MB，本机负载高时时序一偏申请 128 KB 返回空、
// 客体不判空直接写（CRASH main+0x1b36138，合入后复核 inv_tap）。
// Application memory pool = the 3285 MB the Switch (4 GB model) gives applications. At startup (main+0x1b66160) the game sets the whole total reported by QueryMemoryInfo as the heap, retrying with 2 MB less on failure,
// and hands the whole block to its own StandardAllocator (main+0x32137a0) -- so heap size = pool − image footprint, it has only as much as the host reports.
// Previously 3 GB was reported here and SetHeapSize never failed: the heap was about 150 MB smaller than on the real device, at the run entry peak the main heap had only about 18 MB left, and under high local load a timing shift made a 128 KB request return null,
// which the guest writes to without a null check (CRASH main+0x1b36138, re-verified after merge with inv_tap).
#define APP_POOL 0xCD500000ull
// 报给游戏的池 = Switch 的 3285 MB + 512 MB（DM_APP_POOL_ADD_MB=<整数> 可调，可为负——往小调用来造「主堆用尽」的现场）。
// 为什么多给：游戏自己固定划走约 3.08 GB，主堆余量进局只有 100–180 MB、换图时地板三四十 MB、最大可分配块二十几 MB；用尽时原版多处不判空直接写
// （公开仓库，另见下面 dm_guest_heap_stat）。实测多给的量一比一变成主堆余量（138 → 649 MB），游戏划走的不跟着变；
// 没碰过的堆页不计入常驻内存，余量真被用掉时常驻才涨。堆区上限是预留里的 4 GB（mem_init），3728 MB 放得下；再加要先动预留。
// 取 +512 而不是更大：给 6 GB 内存的机型留余地，真机上限有读数后再定。
// Pool reported to the game = the Switch's 3285 MB + 512 MB (adjustable with DM_APP_POOL_ADD_MB=<integer>, may be negative -- lowering it is how to stage a "main heap exhausted" scene).
// Why more is given: the game carves out a fixed 3.08 GB or so for itself, leaving the main heap only 100–180 MB of free space on entering a game, a floor of thirty to forty MB when changing maps and a largest allocatable block of twenty-odd MB; when it runs out the original game writes through null without checking in many places
// (see also dm_guest_heap_stat below). Measured: the extra amount turns one-to-one into main heap free space (138 -> 649 MB), while what the game carves out does not change with it;
// heap pages never touched do not count toward resident memory, which grows only when the free space is actually used. The heap region limit is the 4 GB in the reservation (mem_init), which fits 3728 MB; adding more means changing the reservation first.
// +512 rather than more: to leave room for devices with 6 GB of RAM; to be revisited once there are readings of the limit on real devices.
static uint64_t app_pool(void) { static uint64_t v; if (!v) { const char *e = getenv("DM_APP_POOL_ADD_MB"); long add = e ? atol(e) : 512; v = APP_POOL + ((uint64_t)add << 20); if (e) LOG("DM_APP_POOL_ADD_MB：应用内存池报 %llu MB（Switch 的 3285 MB %+ld）", (unsigned long long)(v >> 20), add); } return v; }
static size_t image_bytes(void) { size_t n = 0; for (int i = 0; i < g_nmod; i++) n += g_mod[i].size; return (n + 0x1fffff) & ~(size_t)0x1fffff; }

// 游戏主堆读数：主堆 = main+0x32137a0 那个 nn::mem::StandardAllocator（启动时把整块堆交给它）。申请失败返回空，原版多处不判空直接写
// （公开仓库 连战 82 分钟后异步读盘回调 main+0x27ecdc 要 765 KB 连续块得空，memset 写 0 地址；进程常驻内存看不出这件事）。
// 余量 / 最大可分配块调的是分配器自己的查询（sdk+0x2f9d78 / +0x2f9dbc，内部走 CentralHeap::Query、自己加锁），只许在客体线程上调。
// 读到的数留一份在 g_gheap_last 里，分配返回空时的报红直接带上次读数（那一刻不再进分配器）。单位：总量 / 余量 MB，最大块 KB
// Game main heap readout: the main heap = the nn::mem::StandardAllocator at main+0x32137a0 (the whole heap block is handed to it at startup). A failed allocation returns null, and the original game writes through it without a null check in many places
// (after 82 minutes of boss rush, the asynchronous disk-read callback main+0x27ecdc asked for a contiguous 765 KB block, got null, and memset wrote to address 0; the process resident memory does not show this).
// Free space / largest allocatable block call the allocator's own queries (sdk+0x2f9d78 / +0x2f9dbc, which go through CentralHeap::Query and take their own lock) and may only be called on a guest thread.
// The values read are kept in g_gheap_last, and the red report on a null allocation carries the last readout directly (the allocator is not entered again at that moment). Units: total / free in MB, largest block in KB
void *dm_guest_malloc(size_t n, size_t align);   // np320
static module_t *mod_named(const char *n) { for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, n)) return &g_mod[i]; return NULL; }
static _Atomic unsigned long long g_gheap_last[3];
int dm_guest_heap_stat(unsigned long long *total_mb, unsigned long long *free_mb, unsigned long long *maxblk_kb) {
    if (!dm_game_is_d2r()) return 0;   // np382：主堆对象在 main+0x32137a0，是 **D2R 的布局**；别的游戏这个地址在镜像之外，不能读 | the main heap object lives at main+0x32137a0, the D2R layout; for another game that address is outside the image and must not be read
    module_t *mm = mod_named("main"), *ms = mod_named("sdk"); if (!mm || !ms || !g_heap_cur) return 0;
    uint8_t *al = mm->base + 0x32137a0; if (!al[0]) return 0;   // 首字节 = 已初始化（sdk+0x2f9898 写） | First byte = initialized (written by sdk+0x2f9898)
    uint64_t fr = ((uint64_t (*)(void *))(ms->base + 0x2f9d78))(al), mx = ((uint64_t (*)(void *))(ms->base + 0x2f9dbc))(al);
    *total_mb = g_heap_cur >> 20; *free_mb = fr >> 20; *maxblk_kb = mx >> 10;
    g_gheap_last[0] = *total_mb; g_gheap_last[1] = *free_mb; g_gheap_last[2] = *maxblk_kb; return 1;
}
// 上次读到的主堆读数（总量 MB、余量 MB、最大可分配块 KB；没读过全 0）。任何线程可调——不进分配器（iOS 壳在主线程上记内存告警那一行用）
// The last main-heap readout (total MB, free MB, largest allocatable block KB; all 0 if never read). Callable from any thread - does not enter the allocator (the iOS shell uses it on the main thread for the memory-warning log line)
void dm_guest_heap_last(unsigned long long v[3]) { for (int i = 0; i < 3; i++) v[i] = atomic_load(&g_gheap_last[i]); }
// np322：两个重载共用一份实现（客体 malloc 走的是 Em，不是 Emm —— 上一轮拦错了重载）。
static void *stdalloc_alloc_common(void *self, size_t size, size_t align) {
    static void *(*orig)(void *, size_t); if (!orig) { module_t *ms = mod_named("sdk"); orig = (void *(*)(void *, size_t))(ms->base + 0x2f9974); }
    void *p = orig(self, size);
    // np321/322：客体分配器失败就由宿主兜底。实测这台机器上这个 StandardAllocator(main+0x22d71e8)
    // 报"已初始化"却仍返回 0 ⇒ 工厂拿到 this=0 ⇒ 崩在 str wzr,[x19,#0x60]。
    // np321/322: fall back to the host heap whenever the guest allocator fails. Measured on this device the
    // StandardAllocator at main+0x22d71e8 reports initialized yet Allocate(0xa0) returns 0, so the factory gets
    // this = 0 and dies on str wzr,[x19,#0x60].
    // np325：**不再要求 size 非零** —— 真机证明这个入口会被以 size=0 调用（或参数位置与假设不同），
    // 于是旧条件 `size && ...` 直接跳过兜底、把 0 原样返回，正好复现 this=0。
    // np325: do not require a non-zero size. On device this entry is reached with size = 0 (or with a
    // different argument layout), so the old `size && ...` guard skipped the fallback and returned 0 -- exactly
    // the this = 0 we chase.
    if (!p || (uintptr_t)p >= (uintptr_t)-4096) {
        void *h = dm_guest_malloc(size, align);
        static _Atomic int n;
        if (atomic_fetch_add(&n, 1) < 10)
            LOG("np325 客体分配器没给出内存（self=%p size=%zu 返回 %p）→ 宿主堆 %p", self, size, p, h);
        if (!h) LOG("【红】np322 宿主堆也没给出 %zu 字节（对齐 %zu）", size, align);
        p = h;
    }
    return p;
}
HLE_FORCE("_ZN2nn3mem17StandardAllocator8AllocateEm", "客体 malloc 用的就是这个重载：失败由宿主堆兜底", void *, hle_stdalloc_em, void *self, size_t size) {
    return stdalloc_alloc_common(self, size, 16);
}
HLE_FORCE("_ZN2nn3mem17StandardAllocator8AllocateEmm", "带对齐的重载，同一份实现", void *, hle_stdalloc_allocate, void *self, size_t size, size_t align) {
    return stdalloc_alloc_common(self, size, align);
}
// np324：把 handler 地址暴露给 loader，用于装载后**直接改写 GOT 槽**（纯数据写）
void *dm_hle_alloc_em_addr(void) { return (void *)(uintptr_t)&hle_stdalloc_em; }
void *dm_hle_alloc_emm_addr(void) { return (void *)(uintptr_t)&hle_stdalloc_allocate; }
#if 0   /* np322：旧实现留档（只拦了 Emm，导致 np321 的兜底从未走到） */
HLE_FORCE("_ZN2nn3mem17StandardAllocator8AllocateEmm", "照原版分配，只在返回空时报红（原版调用方多处不判空）", void *, hle_stdalloc_allocate_old, void *self, size_t size, size_t align) {
    // np320：**未初始化的分配器**（首字节 = 已初始化标志，0 = 没构造）交给宿主堆。
    // 为什么：客体 malloc(0x9b27a0) 用的那个 StandardAllocator(main+0x22d71e8) 在本作里从没被构造，
    // 客体实现在这种状态下直接返回 0；而"改客体代码去绕开它"已被真机证明不生效（np315–318）。
    // 这条链是：工厂 → Allocate(vptr=0 → 0x9f3444) → 客体 malloc → 就是这个导入 ⇒ 这里放行即可。
    // np320: an uninitialized allocator (byte 0 = the initialized flag) is served from the host heap. The
    // StandardAllocator at main+0x22d71e8 is never constructed in this title and the guest implementation
    // returns 0 in that state; patching guest code to bypass it does not work on this device (np315-318).
    static void *(*orig)(void *, size_t, size_t); if (!orig) { module_t *ms = mod_named("sdk"); orig = (void *(*)(void *, size_t, size_t))(ms->base + 0x2f9974); }
    void *p = orig(self, size, align);
    // np321：**客体分配器失败就由宿主兜底**。实测这台机器上：这个 StandardAllocator(main+0x22d71e8)
    // 首字节是 1（已初始化），但 Allocate(0xa0) 仍返回 0 ⇒ 工厂拿到 this=0 ⇒ 崩在 str wzr,[x19,#0x60]。
    // 与其猜它为什么失败，不如在这里兜住：非零申请、返回空（或 −4096 以内的伪地址）就改给宿主堆。
    // np321: fall back to the host heap whenever the guest allocator fails. Measured on this device: the
    // StandardAllocator at main+0x22d71e8 reports initialized (byte 0 = 1) yet Allocate(0xa0) returns 0, so the
    // factory gets this = 0 and dies on str wzr,[x19,#0x60]. Instead of guessing why, catch it here.
    if (size && (!p || (uintptr_t)p >= (uintptr_t)-4096)) {
        void *h = dm_guest_malloc(size, align);
        static _Atomic int n;
        if (atomic_fetch_add(&n, 1) < 10)
            LOG("np321 客体分配器没给出内存（self=%p 申请 %zu 返回 %p）→ 宿主堆 %p", self, size, p, h);
        if (!h) LOG("【红】np321 宿主堆也没给出 %zu 字节（对齐 %zu）", size, align);
        p = h;
    }
    if (!p && size) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) LOG("【红】游戏堆分配返回空：申请 %zu 字节（对齐 %zu），分配器 %p；上次读数 主堆 %llu MB、余量 %llu MB、最大可分配块 %llu KB", size, align, self,
        (unsigned long long)g_gheap_last[0], (unsigned long long)g_gheap_last[1], (unsigned long long)g_gheap_last[2]); }
    return p;
}
#endif

// np321：客体侧 Free —— 宿主堆给的指针不在客体地址空间里，交给客体分配器会把它当野指针处理。
// 这里只把"客体地址空间内"的指针转发给原实现，其余跳过（泄漏换稳定；等分配器问题查清再收回）。
int dm_ptr_in_as(const void *p);   // 定义在本文件后面
HLE_FORCE("_ZN2nn3mem17StandardAllocator4FreeEPv", "客体地址空间内的指针才转发给原实现", void, hle_stdalloc_free, void *self, void *ptr) {
    if (!ptr) return;
    static void (*orig)(void *, void *); if (!orig) { module_t *ms = mod_named("sdk"); orig = (void (*)(void *, void *))(ms->base + 0x81144); }   // sdk 里 StandardAllocator::Free 的真实地址（离线从 sdk 的 dynsym 查到）
    if (!dm_ptr_in_as(ptr)) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) LOG("np321 Free(%p)：不在客体地址空间（宿主堆给的），跳过", ptr); return; }
    orig(self, ptr);
}

// 内存账（memstat.m）用：客体地址空间整块范围（未初始化时 base = NULL）
// For memory accounting (memstat.m): the range of the whole guest address space block (base = NULL when not initialized)
void svc_guest_range(uint8_t **base, size_t *size) { *base = g_as; *size = g_reserve; }

// np287：指针是否落在客体地址空间里（排障用：宿主把自己 .so 内的地址递给了客体时，
// 用它能一句话说清"这个入参不是客体内存"）。
int dm_ptr_in_as(const void *p) {
    uintptr_t a = (uintptr_t)p;
    return g_as && a >= (uintptr_t)g_as && a < (uintptr_t)g_as + g_reserve;
}


enum { I_CoreMask=0, I_PriorityMask=1, I_AliasAddr=2, I_AliasSize=3, I_HeapAddr=4, I_HeapSize=5,
       I_TotalMem=6, I_UsedMem=7, I_DebuggerAttached=8, I_ResourceLimit=9, I_IdleTick=10, I_RandomEntropy=11,
       I_AslrAddr=12, I_AslrSize=13, I_StackAddr=14, I_StackSize=15, I_SysResTotal=16, I_SysResUsed=17,
       I_ProgramId=18, I_InitProcIdRange=19, I_UserExcAddr=20, I_TotalNonSys=21, I_UsedNonSys=22, I_IsApplication=23,
       I_FreeThreadCount=24 };

static uint8_t g_seen[256];

// 已映射区间登记（MapMemory 栈、MapPhysicalMemory alias 页），QueryMemory 据此回报；SDK 靠它找空位
// Registry of mapped ranges (MapMemory stacks, MapPhysicalMemory alias pages), QueryMemory reports from it; the SDK relies on it to find free space
#define MAX_MAP 4096
static struct { uint64_t base, size, src; uint32_t state; } g_map[MAX_MAP];
static int g_nmap;
// np285：崩时把客体 AS 的已登记区间打出来 —— 判定崩溃地址是"该映射却没映射"还是"越界"。
void svc_map_dump(void) {
    fprintf(stderr, " [客体 AS base=%p size=%#zx，已登记 %d 段：", (void *)g_as, (size_t)g_reserve, g_nmap);
    for (int i = 0; i < g_nmap && i < 20; i++)
        fprintf(stderr, " [%#llx,+%#llx)", (unsigned long long)g_map[i].base, (unsigned long long)g_map[i].size);
    fprintf(stderr, " ]\n");
}
static pthread_mutex_t g_map_lock = PTHREAD_MUTEX_INITIALIZER;
static int map_find(uint64_t a, uint64_t n) {       // 返回 1 基下标，0 = 无重叠 | returns a 1-based index, 0 = no overlap
    for (int i = 0; i < g_nmap; i++) if (a < g_map[i].base + g_map[i].size && g_map[i].base < a + n) return i + 1;
    return 0;
}
// np343：0x04/0x05 已改为调用点全程持锁内联登记，这两个带锁包装不再使用（保留定义会触发未用告警，删）

int svc_thread(int num, uint64_t *regs);


uint64_t svc_dispatch(int num, uint64_t *regs, uint64_t orig_addr) {
    { void hlecount_svc(int); hlecount_svc(num); }   // DM_HLECOUNT：svc 按号计数 | DM_HLECOUNT: svc counts by number
    mem_init();
    if (svc_thread(num, regs)) return regs[0];
    { uint64_t svc_host_call(int, uint64_t *); if (num >= 0x80 && svc_host_call(num, regs)) return regs[0]; }   // 0x80 起：剪口约定的宿主调用（hle_tact.c） | from 0x80: host calls by cut convention (hle_tact.c)
    switch (num) {
    case 0x01: {  // SetHeapSize(out_addr*, size)  → x1 = heap base
        size_t size = regs[1];
        if (size > g_heap_cap) { LOG("SetHeapSize OOM: req %#zx > cap %#zx", size, g_heap_cap); regs[0] = 0xD401; return 0; }
        if (size + image_bytes() > app_pool()) { g_heap_rejects++; regs[0] = 0x10801; return 0; }   // 超出应用内存池：ResultLimitReached。游戏启动时按 2 MB 递减重试直到放得下（main+0x1b66184） | exceeds the application memory pool: ResultLimitReached. At startup the game retries in 2 MB decrements until it fits (main+0x1b66184)
        int rc = 0;
        if (size > g_heap_cur) rc = mprotect(g_heap + g_heap_cur, size - g_heap_cur, PROT_READ | PROT_WRITE);
        else if (size < g_heap_cur) rc = mprotect(g_heap + size, g_heap_cur - size, PROT_NONE);
        if (g_flag.svc) LOG("SetHeapSize %#zx -> base %p mprotect=%d errno=%d", size, g_heap, rc, errno);
        { static int said; if (!said && size) { said = 1; LOG("SetHeapSize：首次设堆成功 %zu MB（应用内存池 %llu MB − 映像 %zu MB；此前被拒 %d 次）", size >> 20, (unsigned long long)(app_pool() >> 20), image_bytes() >> 20, g_heap_rejects); } }
        g_heap_cur = size;
        regs[0] = 0; regs[1] = (uint64_t)g_heap; return 0;
    }
    case 0x02: regs[0] = 0; return 0;   // SetMemoryPermission
    case 0x03: regs[0] = 0; return 0;   // SetMemoryAttribute
    case 0x04: {  // MapMemory(dst, src, size)：SDK 建线程时把堆上的栈搬进栈区 | MapMemory(dst, src, size): when creating a thread the SDK moves the stack on the heap into the stack region
        uint64_t dst = regs[0], src = regs[1], size = regs[2];
        // np343 对齐 nxnative：查找 + 校验 + 登记**全程持锁**（原先 find 在锁外、add 在锁内，
        // 两个线程同时建栈时可同时通过 find、把对方的映射顶掉）
        pthread_mutex_lock(&g_map_lock);
        if ((uint8_t *)dst < g_stack || (uint8_t *)dst + size > g_stack + g_stack_cap || map_find(dst, size)) {
            pthread_mutex_unlock(&g_map_lock);
            LOG("MapMemory 目标非法或已占用 %#llx+%#llx", dst, size); regs[0] = 0xDC01; return 0; }
        // 苹果页 16K、Horizon 页 4K：可访问区扩到 16K 边界（多放开的是同区未用页，不影响语义）
        // Apple pages are 16K, Horizon pages 4K: the accessible region is extended to a 16K boundary (the extra pages opened are unused pages of the same region, semantics unaffected)
        uint64_t lo = dst & ~0x3FFFull, hi = (dst + size + 0x3FFF) & ~0x3FFFull;
        if (mprotect((void *)lo, hi - lo, PROT_READ | PROT_WRITE)) die("MapMemory mprotect errno=%d", errno);
        memcpy((void *)dst, (void *)src, size);   // 搬移语义：映射期间源区不被访问（Horizon 把它设为不可访问） | move semantics: the source region is not accessed while mapped (Horizon makes it inaccessible)
        if (g_nmap >= MAX_MAP) { pthread_mutex_unlock(&g_map_lock); die("映射表满"); }
        g_map[g_nmap++] = (typeof(g_map[0])){dst, size, src, 0x0B /*MappedMemory*/};
        pthread_mutex_unlock(&g_map_lock);
        if (g_flag.svc) LOG("MapMemory %#llx <- %#llx +%#llx", dst, src, size);
        regs[0] = 0; return 0;
    }
    case 0x05: {  // UnmapMemory(dst, src, size)：内容搬回源区；16K 页可能与相邻映射共用，不收回权限 | UnmapMemory(dst, src, size): contents are moved back to the source region; a 16K page may be shared with an adjacent mapping, permissions are not revoked
        uint64_t dst = regs[0], src = regs[1], size = regs[2];
        // np343 对齐 nxnative：与 0x04 同理查找 + 校验 + 删除全程持锁
        pthread_mutex_lock(&g_map_lock);
        int k = map_find(dst, size);
        if (!k || g_map[k - 1].base != dst || g_map[k - 1].size != size || g_map[k - 1].src != src) {
            pthread_mutex_unlock(&g_map_lock);
            LOG("UnmapMemory 与映射不符 %#llx+%#llx", dst, size); regs[0] = 0xD401; return 0; }
        memcpy((void *)src, (void *)dst, size);
        g_map[k - 1] = g_map[--g_nmap];
        pthread_mutex_unlock(&g_map_lock);
        regs[0] = 0; return 0;
    }
    case 0x06: {  // QueryMemory(MemoryInfo* out, PageInfo* pout, addr) → x1 = pageinfo
        uint64_t mi = regs[0], addr = regs[2];
        // MemoryInfo: base(0) size(8) state(0x10) attr(0x14) perm(0x18) ...
        uint8_t *a = (uint8_t *)addr;
        uint64_t base, size, state, perm;
        int k;
        if (g_main_stack_lo && a >= g_main_stack_lo && a < g_main_stack_lo + g_main_stack_size) {
            // 主线程真跑在这段客体栈上（main.c 用 pthread_attr_setstack 放的）⇒ 如实回答，
            // 既不破坏地址空间账，也满足 GetCurrentStackInfo 的断言。
            base=(uint64_t)g_main_stack_lo; size=g_main_stack_size; state=0xB/*Stack*/; perm=3; }
        else if (a >= g_heap && a < g_heap + g_heap_cur) { base=(uint64_t)g_heap; size=g_heap_cur; state=5/*Heap*/; perm=3; }
        else if ((k = map_find(addr, 1))) { base=g_map[k-1].base; size=g_map[k-1].size; state=g_map[k-1].state; perm=3; }
        else if (a >= g_as && a < g_as + g_reserve) {   // 空闲：到相邻映射/堆边界为止 | free: up to the adjacent mapping/heap boundary
            uint64_t lo = (uint64_t)g_as, hi = (uint64_t)g_as + g_reserve;
            if (a >= g_heap) lo = (uint64_t)g_heap + g_heap_cur;
            else if (hi > (uint64_t)g_heap) hi = (uint64_t)g_heap;
            for (int i = 0; i < g_nmap; i++) {
                uint64_t e = g_map[i].base + g_map[i].size;
                if (e <= addr && e > lo) lo = e;
                if (g_map[i].base > addr && g_map[i].base < hi) hi = g_map[i].base;
            }
            base=lo; size=hi-lo; state=0; perm=0; }
        else {
            // 地址在客体地址空间之外：只把**确实装着代码**的地址报成 Code（已装载模块的映像范围），
            // 其余一律 Unmapped。原先"未知地址一律报 Code 一页"是错的 —— SDK 建线程时会拿
            // QueryMemory 逐页找内存边界，到处是 Code 就永远找不到边界，一路扫几十 GB 后 panic
            // （真机：svc 0x6 刷屏、x2 每次 -0x1000，最后 ThreadManager::CreateThread 里 Break）。
            //
            // np249：这里**不再**对"宿主栈地址"或"越界地址"做任何替身回答。理由（两条都真机踩过）：
            //   ① 主线程现在真跑在客体 stack 区（main.c 的 run_guest_on_guest_stack），
            //      sp 查询由上面的主栈分支如实回答，不需要替身；
            //   ② 越界地址必须如实报 Unmapped/perm=0 —— 曾经把它答成 stack(perm=3)，
            //      连**地址 0** 也被这样答，nnosInitialize 便以为 0 是合法结构体、去 ldp 它
            //      ⇒ sdk+0x8c6b4 崩。
            int inmod = 0;
            for (int i = 0; i < g_nmod; i++)
                if (addr >= (uint64_t)g_mod[i].base && addr < (uint64_t)g_mod[i].base + g_mod[i].size) { inmod = 1; break; }
            base = addr & ~0xFFFull; size = 0x1000;
            if (inmod) { state = 3/*Code*/; perm = 5; } else { state = 0/*Unmapped*/; perm = 0; }
        }
        // np247：首次 QueryMemory 打一次 x29 回溯 —— 建线程时有一轮"逐页扫内存"，
        // 靠它才能确定是谁在扫（svc 的 lr 来自改写器尾调用蹦床，不可用）。
        if (g_flag.imports) {
            static int once;
            if (!once++) { LOG("QueryMemory 首次调用 addr=%#llx → 回溯：", (unsigned long long)addr); dm_backtrace(regs[29]); }
        }
        if (mi) { uint64_t *m=(uint64_t*)mi; m[0]=base; m[1]=size; ((uint32_t*)mi)[4]=state; ((uint32_t*)mi)[5]=0; ((uint32_t*)mi)[6]=perm; ((uint32_t*)mi)[7]=0; ((uint32_t*)mi)[8]=0; }
        regs[0] = 0; regs[1] = 0; return 0;
    }
    case 0x21: {   // SendSyncRequest(handle)：宿主没有 IPC 服务端 ⇒ 回一个"成功、无出参"的 CMIF 响应
        // 真机：`SVC 0x21 x0=0x5000 x1=0x5efa51b000`（0x5000 = sm::GetServiceHandle 发出去的假句柄）。
        // 客体随后的断言（从 subsdk0 里读出的原话）：
        //   QueryPointerBufferSize(&pointerBufferSize, m_ClientHandle)
        //   at nn/sf/hipc/client/sf_HipcClientMessage.h
        // ⇒ 它在用这个假句柄做 CMIF 握手，而我们既没服务端也没写出站头。
        //
        // 找请求缓冲的两条路（都试，谁先命中用谁）：
        //   ① x1 直接就是缓冲（上一轮真机看到的值），且首 4 字节 = 'SFCI'；
        //   ② Horizon 的 hipc 把请求缓冲放在**线程本地区**里 —— 扫 TLR 前 0x40 字节的指针，
        //      找首 4 字节 = 'SFCI' 的那个（这样不依赖寄存器约定）。
        // 命中后按 CMIF 写：{ u32 magic='SFCO', u32 result=0 } + 出参区清零。
        uint8_t *msg = (uint8_t *)regs[1];
        uint8_t *tlr = (uint8_t *)pthread_getspecific(g_keytlr);
        uint8_t *from = NULL;
        // ★ 真机实测（np259）：CMIF 请求头**内联在 TLR 里**，但偏移会变 ——
        //   第一发在 +0x10（TLR[0]=0x800000005），第二发在 +0x20（TLR[0]=0x8000000900000004）。
        //   所以不写死偏移：在 TLR 前 0x100 字节里按 4 字节对齐扫 'SFCI'，扫到哪儿就写哪儿。
        if (tlr) {
            for (int i = 0; i < 0x100; i += 4)
                if (!memcmp(tlr + i, "SFCI", 4)) { from = tlr + i; break; }
        }
        // np395：**去掉 g_as 范围检查**。np388 的转储显示"请求缓冲全是 0"，其实是范围检查没通过、
        // 根本没读（打印的是初值 0）。请求缓冲在客体刚写过的页上，直接比较 magic 即可。
        // np395: drop the g_as range test. The np388 dump showed "the request buffer is all zero" only because the
        // range test failed and nothing was read (the printed zeros were the initial value). The buffer sits on a
        // page the guest just wrote, so comparing the magic directly is enough.
        if (!from && msg && !memcmp(msg, "SFCI", 4)) { from = msg; }
        if (!from && tlr) {
            for (int i = 0; i < 0x40; i += 8) {
                uint8_t *q = (uint8_t *)(uintptr_t)*(volatile uint64_t *)(tlr + i);
                if (q >= g_as && q < g_as + g_reserve && !memcmp(q, "SFCI", 4)) { from = q; break; }
            }
        }
        if (from) {
            // np396：命中时把**请求头**也打出来（命令号在里面）—— 应答内容要按号回，先把号看清楚。
            // np396: on a hit also dump the REQUEST header (the command id lives there) -- the reply must be
            // built per command id, so first see what the ids are.
            { static int nshow; if (nshow < 6) {
                nshow++;
                LOG("    CMIF 请求头 %p：", (void *)from);
                for (int i = 0; i < 48; i += 8) { uint64_t v = 0; memcpy(&v, from + i, 8); LOG("      +%#x = %#llx", i, (unsigned long long)v); }
            } }
            uint32_t magic = 0x4F434653u;              // 'SFCO'
            uint32_t res = 0;
            memcpy(from, &magic, 4); memcpy(from + 4, &res, 4);
            memset(from + 8, 0, 0x40);                  // 出参（QueryPointerBufferSize 等从这儿读）
        }
        if (g_flag.imports) {
            static int n;
            if (from) { static int ok; if (ok++ < 3) LOG("SendSyncRequest h=%#llx 已回 'SFCO' @ %p", (unsigned long long)regs[0], (void *)from); }
            else if (tlr) {
                // 没命中也把 TLR 头 0x40 字节打出来 —— 第二发 IPC 的请求头不在 +0x10 了，
                // 得看它到底长什么样（也许换成了 WithUserBuffer 或者已被我们上一发覆写成 SFCO）。
                static int bad;
                // np388：把 regs[1] 指的缓冲前 32 字节也打出来 —— 真机这一发 x1=指针、x2=句柄，
                // 与我们假设的"x1=缓冲、首 4 字节 'SFCI'"不一定一致；看到真实布局才知道该按哪种 CMIF 应答。
                if (bad < 3 && regs[1]) {
                    LOG("    请求缓冲 %p 前 48 字节：", (void *)regs[1]);
                    for (int i = 0; i < 48; i += 8) {
                        uint64_t v = 0; memcpy(&v, (void *)(uintptr_t)(regs[1] + i), 8);   // 客体刚写过，直接读 | the guest just wrote it, read directly
                        LOG("      +%#x = %#llx", i, (unsigned long long)v);
                    }
                }
                if (bad++ < 3) {
                    LOG("SendSyncRequest h=%#llx 未命中：x0..x3=%#llx %#llx %#llx %#llx", (unsigned long long)regs[0],
                        (unsigned long long)regs[1], (unsigned long long)regs[2], (unsigned long long)regs[3], (unsigned long long)regs[4]);
                    for (int i = 0; i < 0x40; i += 8)
                        LOG("    TLR[%#x] = %#llx", i, (unsigned long long)*(volatile uint64_t *)(tlr + i));
                }
            }
            if (n++ < 2 && tlr) {
                // 一次把 TLR 里所有非零 8 字节槽打出来 —— hipc 的请求缓冲指针就在这里面（真机 x1 就是 TLR 本身，
                // 说明 svc 只传了 handle，缓冲地址由内核从 TLS 取）。看到首 4 字节 'SFCI' 的那个就是它。
                LOG("SendSyncRequest h=%#llx 缓冲=%p%s；TLR 非零槽：", (unsigned long long)regs[0],
                    (void *)from, from ? "（已回 SFCO）" : "（未命中）");
                for (int i = 0; i < 0x200; i += 8) {
                    uint64_t v = *(volatile uint64_t *)(tlr + i);
                    if (!v) continue;
                    const char *tag = "";
                    uint8_t *q = (uint8_t *)(uintptr_t)v;
                    if (q >= g_as && q < g_as + g_reserve) {
                        if (!memcmp(q, "SFCI", 4)) tag = " ← 'SFCI' 请求缓冲";
                        else if (!memcmp(q, "SFCO", 4)) tag = " ← 'SFCO'";
                        else tag = " ← 客体地址";
                    }
                    LOG("  TLR[%#x] = %#llx%s", i, (unsigned long long)v, tag);
                }
            }
        }
        regs[0] = 0; return 0;
    }
    case 0x2C: case 0x2D: {  // MapPhysicalMemory / UnmapPhysicalMemory(addr, size)：只许在 alias 区内（SDK 的 VammManager 用） | MapPhysicalMemory / UnmapPhysicalMemory(addr, size): only allowed inside the alias region (used by the SDK's VammManager)
        uint8_t *a = (uint8_t *)regs[0]; size_t size = regs[1];
        if (a < g_alias || a + size > g_alias + g_alias_cap) { LOG("SVC %#x 越出 alias 区 %p+%#zx", num, a, size); regs[0] = 0xD401; return 0; }
        int rc = num == 0x2C ? mprotect(a, size, PROT_READ | PROT_WRITE)
                             : (madvise(a, size, MADV_FREE), mprotect(a, size, PROT_NONE));
        if (rc) die("SVC %#x mprotect %p+%#zx errno=%d", num, a, size, errno);
        if (g_flag.svc) LOG("%sPhysicalMemory %p+%#zx", num == 0x2C ? "Map" : "Unmap", a, size);
        regs[0] = 0; return 0;
    }
    case 0x26: {  // Break(reason, ptr, size)：SDK 致命中止 | Break(reason, ptr, size): SDK fatal abort
        uint32_t res = (regs[1] && regs[2] >= 4) ? *(uint32_t *)regs[1] : 0;
        module_t *mm = module_of((void *)regs[30]);
        static int nb;
        { module_t *ml = module_of((void *)regs[30]);
          LOG("Break#%d reason=%#llx Result=%#x svc 位于 %s+%#llx，lr = %s+%#llx", ++nb, regs[0], res, mm ? mm->name : "?", orig_addr,
              ml ? ml->name : "?", ml ? (uint64_t)regs[30] - (uint64_t)ml->base : (uint64_t)regs[30]); }
        // np383：先把两个参数原样打出来——np382 那轮 reason=0 时 ptr/size 都是 0，说明这个重载不带载荷，
        // 断言文本走的是 svc #0x27（见下）。留着这行，下次一眼能看出是哪种。
        // np383: print both arguments as-is first -- in the np382 run reason=0 came with ptr/size = 0, i.e. this
        // overload carries no payload and the assertion text goes through svc #0x27 (below).
        LOG("Break 参数：ptr=%#llx size=%#llx（%s）", (unsigned long long)regs[1], (unsigned long long)regs[2],
            regs[1] && regs[2] ? "带载荷" : "无载荷，断言文本应已从 svc #0x27 打出");
        // np242：panic 的载荷里带消息（nn::os::Break 的第 2/3 参），打出来才知道是哪个断言
        if (regs[1] && regs[2]) {
            char msg[257]; size_t n = regs[2] < 256 ? (size_t)regs[2] : 256;
            memcpy(msg, (void *)regs[1], n); msg[n] = 0;
            for (size_t k = 0; k < n; k++) if ((unsigned char)msg[k] < 0x20 || (unsigned char)msg[k] > 0x7e) msg[k] = '.';
            LOG("Break 消息：%s", msg);
        }
        dm_backtrace(regs[29]);
        // np397：FP 链过 sdk 的导入跳板就断（跳板改写 x29）⇒ 再打一份栈扫描，那才是可信的调用者
        // np397: the FP chain breaks at the sdk's import trampolines (they rewrite x29), so also print a stack
        // scan -- that is the trustworthy caller list.
        { void dm_backtrace_scan(uint64_t); dm_backtrace_scan(regs[29]); }
        if (regs[0] & 0x80000000ull) return 0;   // 仅通知，可继续 | notification only, can continue
        // ⚠ 退出前必须把日志冲出去：LOG 走 stderr → 管道 → 泵线程 → hilog，
        //   而 _exit() 不会等泵线程，最后几行（正好是 panic 原因）会被掐掉。
        //   np242 真机踩过：只看到 exit code 42，看不到 Break 那行。
        fflush(stderr);
        usleep(150 * 1000);
        _exit(42);
    }
    case 0x1E: regs[0] = 0; regs[1] = guest_ns() * 12 / 625; return 0;   // GetSystemTick
    case 0x27: {  // OutputDebugString(str, len)
        // np383：**默认就记**。NintendoSDK 的断言文本（`nn::diag::detail::AbortImpl` → 这里）只走这条路，
        // 藏在 DM_DBG 后面等于"崩了但不说为什么"——np382 那轮 `Break#1` 就是这么被吞掉的。
        // 有上限：前 4000 条全记，之后只计数，免得刷屏把关键行挤掉。
        // np383: log by default. The NintendoSDK assertion text (nn::diag::detail::AbortImpl -> here) only goes
        // through this path; hiding it behind DM_DBG means "it crashed but will not say why" -- exactly what
        // swallowed the np382 Break#1. Capped: the first 4000 messages are logged, later ones are only counted.
        { static unsigned n; if (n < 4000) { if (!n) LOG("客体调试输出（svc #0x27）开始：");
            // np399：**不能截断**。SDK 的 "User Backtrace" 后面跟着整条调用链，原来只留 511 字节 ⇒ 链被吃掉，
            // 于是只能靠 FP 链/栈扫描猜（两者都被同一个假象骗过，np384/385/391/398 四次）。
            // 现在整段留 4 KB，按 900 字节一段打（hilog 单行有限制）。
            // np399: do NOT truncate. The SDK's "User Backtrace" is followed by the whole call chain, and keeping
            // only 511 bytes ate it -- which is why the FP chain / stack scan had to be guessed from, and both were
            // fooled by the same artefact (np384/385/391/398). Keep 4 KB and print it in 900-byte chunks.
            static char m[4096];
            size_t len = regs[1] < sizeof m - 1 ? (size_t)regs[1] : sizeof m - 1;
            memcpy(m, (void *)regs[0], len); m[len] = 0;
            for (size_t k = 0; k < len; k++) if (!m[k]) m[k] = ' ';
            for (size_t o = 0; o < len; o += 900) { char seg[901]; size_t c = len - o < 900 ? len - o : 900; memcpy(seg, m + o, c); seg[c] = 0; LOG("[guest] %s", seg); } }
          else if (n == 4000) LOG("客体调试输出已达 4000 条，后续只计数");
          n++; }
        regs[0] = 0; return 0;
    }
    case 0x29: {  // GetInfo(out*, id, handle, subid) → x1 = value
        uint64_t id = regs[1], v = 0;
        switch (id) {
        case I_CoreMask: v = 7; break;
        case I_PriorityMask: v = 0; break;
        case I_AliasAddr: v = (uint64_t)g_alias; break;   case I_AliasSize: v = g_alias_cap; break;
        case I_HeapAddr:  v = (uint64_t)g_heap;  break;   case I_HeapSize:  v = g_heap_cap;  break;
        case I_AslrAddr:  v = (uint64_t)g_as;    break;   case I_AslrSize:  v = g_reserve;     break;
        case I_StackAddr: v = (uint64_t)g_stack; break;   case I_StackSize: v = g_stack_cap; break;
        case I_TotalMem: case I_TotalNonSys: v = app_pool(); break;
        case I_UsedMem:  case I_UsedNonSys:  v = g_heap_cur + image_bytes(); break;   // 已用 = 堆 + 映像，与 SetHeapSize 的核对同一口径（原先写死 64 MB）。sdk 的 QueryMemoryInfo（sdk+0x34f4ec）只拿它算「程序占用」那一格； | Used = heap + image, the same accounting as the check in SetHeapSize (it used to be hard-coded to 64 MB). The sdk's QueryMemoryInfo (sdk+0x34f4ec) only uses it to compute the "program usage" field;
                                                 // 「可用总量」那一格直接是池总量，所以游戏启动设堆仍要从整池起按 2 MB 往下减到 池 − 映像（约 34 次被拒）——这是原版在真机上的走法，不是我们的口径错
                                                 // The "total available" field is simply the pool total, so at startup the game still has to step down from the whole pool by 2 MB at a time until pool - image (about 34 refusals) when setting its heap -- this is how the original behaves on the real device, not a mistake in our accounting
        case I_SysResTotal: v = 0x2000000; break; case I_SysResUsed: v = 0x400000; break;
        case I_DebuggerAttached: v = 0; break;
        case I_RandomEntropy: { uint64_t dm_entropy64(void); v = dm_entropy64(); break; }   // DM_DETCLOCK 开时为确定流（hle_os.c） | a deterministic stream when DM_DETCLOCK is on (hle_os.c)
        case I_ProgramId: v = 0x0100726014352000ull; break;
        case I_IsApplication: v = 1; break;
        // np343 对齐 nxnative：用户异常上下文区给一块真 0x1000 对齐缓冲（原先回 0——
        // 较新 SDK 的 SetDyingMessageRegion 会往 +0x1c0 写，地址 0 就是空指针写）
        case I_UserExcAddr: { static _Alignas(0x1000) uint8_t ctx[0x1000]; v = (uint64_t)ctx; break; }
        // np343 对齐 nxnative：nn::os::GetFreeThreadCount（SDK 11+）= 限额 96 − 活着的游戏线程数；
        // 回 0 会让主线程启动时等一个永远不会出现的线程（Monster Train 2 实测卡死）
        case I_FreeThreadCount: { int thr_limit_used(void); int u = thr_limit_used(); v = u < 96 ? 96 - u : 0; break; }
        default: v = 0; break;
        }
        // np247：把客体实际问的 InfoType 和我们的回答打出来。
        // 用途：AddressSpaceAllocator::AllocateSpace 返回 NULL（建线程时崩）最常见的原因是
        // 「它要的区段 id 我们没实现，回 0 ⇒ 分配器范围为空」。
        if (g_flag.imports) {
            static unsigned char seen[256];
            if (!seen[id & 0xff]) { seen[id & 0xff] = 1; LOG("GetInfo id=%llu -> %#llx", (unsigned long long)id, (unsigned long long)v); }
        }
        regs[0] = 0; regs[1] = v; return 0;
    }
    default:
        if (!g_seen[num & 0xff]) { g_seen[num & 0xff] = 1; LOG("SVC %#x x0=%#llx x1=%#llx x2=%#llx (占位返回成功)", num, regs[0], regs[1], regs[2]); }
        regs[0] = 0; return 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// np250：宿主接管 nn::os::detail::AddressSpaceAllocator —— 建线程时的"别名栈"分配。
//
// 为什么接管：客体的这个分配器一直返回 NULL，导致 CreateAliasStackUnsafe 直接 AbortImpl
// （真机 sdk+0x88168）。我们把主线程搬进客体栈区、并如实回答 QueryMemory 之后它**仍然**失败
// ⇒ 那本账的范围来源不明，继续猜的成本高于自己实现：
// 它本质只是"地址空间记账"，而分配出来的地址最终要经 MapAliasStack → svcMapMemory 落到
// **客体 stack 区**（我们的 MapMemory 也只接受 stack 区）。所以这里直接从 g_stack 做
// 16 KB 对齐的 bump 分配，绕开主线程栈所在的尾部 8 MB。
static uint8_t *g_asa_bump;
HLE_FORCE("_ZN2nn2os6detail21AddressSpaceAllocator13AllocateSpaceEmm",
          "宿主接管地址空间分配（从客体 stack 区 bump，绕开主线程栈）",
          void *, h_asa_alloc, void *self, size_t size, size_t align) {
    (void)self;
    if (!g_stack || !g_stack_cap) return NULL;
    if (align < 0x1000) align = 0x1000;
    size_t sz = (size + 0x3FFF) & ~(size_t)0x3FFF;             // 按 16 KB 页向上取整
    uint8_t *lo = g_asa_bump ? g_asa_bump : g_stack;           // 首次从栈区起点开始
    uint8_t *p  = (uint8_t *)(((uintptr_t)lo + align - 1) & ~(uintptr_t)(align - 1));
    size_t msz = 0; (void)dm_guest_main_stack(&msz);
    uint8_t *limit = g_stack + g_stack_cap - (msz ? msz : 0x800000);   // 不碰主线程栈
    if (p + sz > limit) { LOG("【红】AllocateSpace(%#zx) 超出栈区：%p+%#zx > %p", size, (void *)p, sz, (void *)limit); return NULL; }
    g_asa_bump = p + sz;
    if (g_flag.imports) LOG("AllocateSpace(size=%#zx align=%#zx) -> %p", size, align, (void *)p);
    return p;
}
// 守卫区检查：空间是我们自己 bump 出来的，恒通过（返回 bit0=1）
HLE_FORCE("_ZN2nn2os6detail21AddressSpaceAllocator15CheckGuardSpaceEmmm",
          "宿主接管：守卫区检查恒通过", int, h_asa_guard, void *self, size_t a, size_t b) {
    (void)self; (void)a; (void)b; return 1;
}
