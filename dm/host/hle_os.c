// 宿主公共杂项：客体对象旁表、单调时钟、modules 都没定义的 C 库导入。
// （原 nn::os/mem/lmem/C++ ABI 的 136 条 HLE 已删：SDK 自带 nn::os，装载时这些条目一条都没被解析用上。）
// Host common miscellany: guest object side table, monotonic clock, C library imports that none of the modules define.
// (The former 136 HLE entries for nn::os/mem/lmem/C++ ABI have been deleted: the SDK brings its own nn::os, and at load time not one of these entries was resolved and used.)
#include <errno.h>
#include <locale.h>
#include <pthread.h>
#include <string.h>
#include "hle.h"
#include <mach/mach_time.h>
#include <stdatomic.h>

// ---------- 旁表 ----------
// ---------- Side table ----------
#define OM_SIZE (1u << 18)
static struct { const void *k; void *v; } g_om[OM_SIZE];
static pthread_mutex_t g_om_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned om_h(const void *k) { uint64_t x = (uint64_t)k; x ^= x >> 29; x *= 0xbf58476d1ce4e5b9ull; return (unsigned)(x >> 40) & (OM_SIZE - 1); }
void *objmap_get(const void *k) {
    pthread_mutex_lock(&g_om_lock);
    void *r = NULL;
    for (unsigned i = om_h(k);; i = (i + 1) & (OM_SIZE - 1)) {
        if (g_om[i].k == k) { r = g_om[i].v; break; }
        if (!g_om[i].k) break;
    }
    pthread_mutex_unlock(&g_om_lock);
    return r;
}
void objmap_put(const void *k, void *v) {
    pthread_mutex_lock(&g_om_lock);
    for (unsigned i = om_h(k);; i = (i + 1) & (OM_SIZE - 1))
        if (g_om[i].k == k || !g_om[i].k || g_om[i].k == (void *)1) { g_om[i].k = k; g_om[i].v = v; break; }
    pthread_mutex_unlock(&g_om_lock);
}
void *objmap_del(const void *k) {
    pthread_mutex_lock(&g_om_lock);
    void *r = NULL;
    for (unsigned i = om_h(k);; i = (i + 1) & (OM_SIZE - 1)) {
        if (g_om[i].k == k) { r = g_om[i].v; g_om[i].k = (void *)1; g_om[i].v = NULL; break; }
        if (!g_om[i].k) break;
    }
    pthread_mutex_unlock(&g_om_lock);
    return r;
}
uint64_t now_ns(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
}
// 客体时钟：客体能读到的系统计数器（mrs cntpct 门、svc GetSystemTick）都经这里。缺省 = now_ns()。
// 冻结（只供测试驱动 DM_TICK_HOLD 用）：冻结期间客体时钟不走 ⇒ 游戏不会到期下一拍；解冻后扣掉冻结时长，客体时钟连续不跳
// DM_DETCLOCK=<步长 ns>（回放 / 测试局用；产品局不设 = 墙钟）：首次出图起客体时钟改为「每次出图推进固定步长」，不跟墙钟——
// 客户端每帧 dt、服务端拍的补拍快照都由出图次数决定，同一回放两遍局内第 N 帧逐字节可比（画面逐字节证明的地基）。
// 首帧之前仍走墙钟（启动期有按时刻轮询的等待，冻住会卡）；切换时从当时的客体时钟值接续，不跳。冻结期间不推进。
// Guest clock: the system counters the guest can read (mrs cntpct gate, svc GetSystemTick) all go through here. Default = now_ns().
// Freeze (only for the test driver DM_TICK_HOLD): while frozen the guest clock does not advance ⇒ the game's next tick never comes due; after unfreezing the frozen duration is subtracted, the guest clock is continuous with no jump
// DM_DETCLOCK=<step ns> (for replay / test runs; product runs do not set it = wall clock): from the first present on the guest clock becomes "advance a fixed step per present", not following the wall clock --
// the client's per-frame dt and the server tick catch-up snapshots are both determined by the present count, so frame N in a run is byte-for-byte comparable between two passes of the same replay (the foundation of byte-for-byte picture proof).
// Before the first frame the wall clock is still used (the boot period has waits that poll by time, freezing would hang them); at the switch it continues from the guest clock value at that time, no jump. It does not advance while frozen.
static _Atomic uint64_t g_gfrozen, g_glag, g_gdet, g_guest_frame; static _Atomic int g_gfz, g_gdet_on; static uint64_t g_gdet_step = 1;
uint64_t dm_guest_frame(void) { return atomic_load(&g_guest_frame); }
static int detclock_cfg(void) {
    static int c = -1; if (c < 0) { const char *e = getenv("DM_DETCLOCK"); g_gdet_step = e ? strtoull(e, 0, 0) : 0; c = g_gdet_step > 0; }
    return c;
}
static uint64_t wall_guest_ns(void) { return atomic_load(&g_gfz) ? atomic_load(&g_gfrozen) : now_ns() - atomic_load(&g_glag); }
uint64_t guest_ns(void) { return atomic_load(&g_gdet_on) ? atomic_load(&g_gdet) : wall_guest_ns(); }
int dm_detclock_enabled(void) { return detclock_cfg() && atomic_load(&g_gdet_on); }
// 进局对齐：前端 / 选角阶段出图帧数随真实时间变（装载在后台线程按墙钟推进），首帧起累计的客体时钟值因此每局不同——
// 依赖绝对时刻的画面（如生命 / 法力球液面）在同拍仍不同（实测拍 E + 2 帧两遍差约 1.8 万像素，集中在两侧球位）。
// 见到 UI 消息 FrontEndNavigation:ToGame（建局、服务端拍时限初始化之前）即把客体时钟设为固定值 2^52 ns（只许前跳），
// 此后纯按出图推进：拍 E 落在第几帧、那一刻的绝对时刻都只由步长决定。
// Run entry alignment: the number of presented frames in the front end / character select phase varies with real time (loading advances on a background thread by wall clock), so the guest clock value accumulated since the first frame differs every run --
// pictures that depend on absolute time (such as the liquid level of the life / mana orbs) still differ at the same tick (measured: at tick E + 2 frames two passes differ by about 18,000 pixels, concentrated at the two orb positions).
// On seeing the UI message FrontEndNavigation:ToGame (before the run is created and the server tick deadline is initialized) the guest clock is set to the fixed value 2^52 ns (forward jumps only),
// from then on it advances purely by presents: which frame tick E lands on and the absolute time at that moment are determined by the step alone.
#define DET_TOGAME_NS (1ull << 52)
static void det_on_uimsg(const char *cat, const char *act, const char *arg) {
    (void)arg; if (strcmp(cat, "FrontEndNavigation") || strcmp(act, "ToGame")) return;
    uint64_t cur = atomic_load(&g_gdet);
    if (cur >= DET_TOGAME_NS) { LOG("【红】DM_DETCLOCK：进局时客体时钟 %llu ≥ 对齐值 2^52，不回拨（对齐未生效）", (unsigned long long)cur); return; }
    atomic_store(&g_gdet, DET_TOGAME_NS);
    LOG("DM_DETCLOCK：进局（ToGame）客体时钟对齐到 2^52 ns（前跳 %.3f s）", (DET_TOGAME_NS - cur) / 1e9);
}
void dm_guest_frame_tick(void) {
    crash_watch_frame();
    atomic_fetch_add(&g_guest_frame, 1);
    if (!detclock_cfg()) return;
    if (!atomic_load(&g_gdet_on)) {
        atomic_store(&g_gdet, wall_guest_ns()); atomic_store(&g_gdet_on, 1);
        void uimsg_subscribe(void (*)(const char *, const char *, const char *)); uimsg_subscribe(det_on_uimsg);
        LOG("DM_DETCLOCK：首次出图起客体时钟按出图推进，步长 %llu ns；进局时对齐到 2^52 ns", (unsigned long long)g_gdet_step); return;
    }
    if (!atomic_load(&g_gfz)) { atomic_fetch_add(&g_gdet, g_gdet_step); dm_guest_clock_advanced(); }
}
static void guest_clock_freeze_apply(int on) {
    if (on && !atomic_load(&g_gfz)) { atomic_store(&g_gfrozen, now_ns() - atomic_load(&g_glag)); atomic_store(&g_gfz, 1); }
    else if (!on && atomic_load(&g_gfz)) { atomic_store(&g_glag, now_ns() - atomic_load(&g_gfrozen)); atomic_store(&g_gfz, 0); }
}
// 冻结有几个互不相干的原因（0 测试用的冻结拍 / 读取画面，1 App 在后台，2 系统键盘弹着）：任一成立就冻结，全部解除才放行——
// 否则「键盘弹着时切后台再回来」会被回前台那次解冻提前放开
// Freezing has several unrelated reasons (0 freeze tick for testing / loading screen, 1 App in background, 2 system keyboard up): freeze if any holds, release only when all are lifted --
// otherwise "switch to background and back while the keyboard is up" would be released early by the unfreeze on returning to foreground
static pthread_mutex_t g_gfz_mu = PTHREAD_MUTEX_INITIALIZER; static unsigned g_gfz_reasons;
void guest_clock_freeze_for(int reason, int on) {
    pthread_mutex_lock(&g_gfz_mu);
    if (on) g_gfz_reasons |= 1u << reason; else g_gfz_reasons &= ~(1u << reason);
    guest_clock_freeze_apply(g_gfz_reasons != 0);
    pthread_mutex_unlock(&g_gfz_mu);
}
void guest_clock_freeze(int on) { guest_clock_freeze_for(0, on); }
// 熵源：缺省真随机（arc4random）；DM_DETCLOCK 开时改为固定种子的 splitmix64 流（回放画面逐字节可比用，营地蓝火粒子两遍不同）。
// 两处出口（libc getentropy、svc GetInfo RandomEntropy）共用一条流；调用次数在日志里报（首次与每 1000 次）
// Entropy source: true random by default (arc4random); with DM_DETCLOCK on it becomes a fixed-seed splitmix64 stream (so that replay pictures are byte-for-byte comparable, the camp blue fire particles differed between two passes).
// The two outlets (libc getentropy, svc GetInfo RandomEntropy) share one stream; the call count is reported in the log (first call and every 1000 calls)
static _Atomic uint64_t g_detrng = 0x5357444554524e47ull; static _Atomic uint64_t g_detrng_n;
uint64_t dm_entropy64(void) {
    if (!detclock_cfg()) return ((uint64_t)arc4random() << 32) | arc4random();
    uint64_t z = atomic_fetch_add(&g_detrng, 0x9E3779B97F4A7C15ull) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
    uint64_t n = atomic_fetch_add(&g_detrng_n, 1) + 1; if (n == 1 || n % 1000 == 0) LOG("DM_DETCLOCK：确定熵第 %llu 次", (unsigned long long)n);
    return z;
}
// setlocale：客体静态构造器会调 setlocale(LC_ALL, "C") 探容量（BotW sdk 的 init_array 里就有）。
// 没实现时它落在"未实现导入桩"上，桩返回 -1，客体把 -1 当 locale 指针存下来，
// 后面 memset 那个"指针"就 SIGSEGV（np240 真机实测：memset(-1, 0, 0x40)）。
// 另外 Horizon 的 LC_ALL 是掩码 0x7fffffff，宿主 libc（musl）的 LC_ALL 是 6——
// 不换算的话宿主 libc 会以"类别非法"为由返回 NULL，照样崩。所以两边都要处理。
HLE("setlocale", char *, h_setlocale, int category, const char *locale) {
    if ((unsigned)category == 0x7fffffff) category = LC_ALL;   // Horizon 的 LC_ALL 掩码 → 宿主 libc 的 LC_ALL
    char *r = setlocale(category, locale);
    if (!r) LOG("【黄】setlocale(%#x, %s) 失败：%s", category, locale ? locale : "(null)", strerror(errno));
    return r;
}
// newlocale / uselocale / freelocale：同一族，一并接上（掩码同样换算）
HLE("newlocale", void *, h_newlocale, int mask, const char *name, void *base) {
    if ((unsigned)mask == 0x7fffffff) mask = LC_ALL_MASK;
    return newlocale(mask, name, base);
}
HLE("uselocale", void *, h_uselocale, void *l) { return uselocale(l); }
HLE("freelocale", void, h_freelocale, void *l) { freelocale(l); }

// ── C++ 静态构造守卫（np242）──
// 客体 sdk 自带一份 __cxa_guard_acquire（sdk+0x2ee860），它内部先调 **导入** nn::os::LockMutex
// 和 WaitConditionVariable。真机实测：守卫字节明明是 0（文件里也是 0），它却返回 0
// ⇒ 调用方把静态对象当成"已构造"，指针字段还是 NULL ⇒ 崩在 ldr x0,[x0]
// （BotW：locale::__global() 返回 0，sdk+0x29e0e4）。
// 宿主这份只看守卫字节，不依赖客体的锁；用 HLE_FORCE 压过模块内定义（force 表先于 st_shndx 查）。
static pthread_mutex_t g_guard_mu = PTHREAD_MUTEX_INITIALIZER;
HLE_FORCE("__cxa_guard_acquire", "宿主实现 C++ 静态构造守卫（客体版本依赖 nn::os 锁，会误报已构造）",
          int, h_guard_acquire, uint64_t *g) {
    if (!g) return 1;
    pthread_mutex_lock(&g_guard_mu);
    unsigned char b = *(volatile unsigned char *)g;
    pthread_mutex_unlock(&g_guard_mu);
    if (g_flag.imports) LOG("guard_acquire %p 字节=%u -> %d", (void *)g, b, b == 0);
    // DM_GUARD_FORCE=1（排障）：恒返回 1，用来区分"守卫被误判已构造"与"根本没走守卫"
    if (getenv("DM_GUARD_FORCE")) return 1;
    return b == 0;          // 1 = 该我构造；0 = 已经构造过
}
HLE_FORCE("__cxa_guard_release", "宿主实现 C++ 静态构造守卫：置已构造",
          void, h_guard_release, uint64_t *g) {
    if (!g) return;
    if (g_flag.imports) LOG("guard_release %p", (void *)g);
    pthread_mutex_lock(&g_guard_mu);
    *(volatile unsigned char *)g = 1;
    pthread_mutex_unlock(&g_guard_mu);
}
HLE_FORCE("__cxa_guard_abort", "宿主实现 C++ 静态构造守卫：放弃",
          void, h_guard_abort, uint64_t *g) { (void)g; }

// ── locale::__global()（np242）──
// libc++ 的 std::locale 全局对象工厂。客体那份（sdk+0x29e018）走惰性构造：
// 守卫字节 → 构造 → 把 __imp_ 指针写进静态区。真机上它返回了 0，调用方紧接着
// `ldr x0,[x0]`（取 __imp_）就 SIGSEGV（sdk+0x29e0e4）。
// 宿主这份直接给一个**始终存在**的静态 locale：只保证指针非空、__imp_ 非空。
// 客体在这条路径上只做"把 __imp_ 拷进新对象"，不 deref __imp_ 的内容，所以够用；
// 真要 use_facet 时再按需补（届时的报错会指出是哪个 facet）。
static uint64_t g_locale_imp[8];                                   // 假 locale::__imp（先全零）
static uint64_t g_locale_obj[2] = { (uint64_t)g_locale_imp, 0 };    // 假 std::locale（唯一成员 __imp_）
// ⚠ np242 真机结论：**不要覆盖 locale::__global()**。
// 之前它返回 0，是因为 malloc 恒失败（分配器 hook 没装）——客体构造 locale 时 operator new
// 失败、指针没写进去，守卫却被释放了。分配器修好后客体自己的实现就能用。
// 用假对象顶上去反而更糟：客体后续对 __imp_ 做 facet 查询会抛 bad_cast →
// std::terminate → nn::os::Break(reason=0) → exit(42)（本轮实测）。
// 这段代码留着当记录，真要启用时改成 HLE_FORCE 并把 __imp_ 做成完整 facet 表。
#if 0
static uint64_t g_locale_imp[8];
static uint64_t g_locale_obj[2] = { (uint64_t)g_locale_imp, 0 };
HLE_FORCE("_ZNSt3__16locale8__globalEv", "宿主实现 std::locale 全局对象",
          void *, h_locale_global) {
    return g_locale_obj;
}
#endif

// ── 分配器（np242）──
// 客体的 malloc（BotW 在 main+0x9b27a0）第一步先看一个**分配器 hook** [main+0x22d7230]：
//     ldr x9,[0x22d7230]; cbz x9 → 直接返回 NULL
// 这个 hook 本该由 rtld / libc 初始化时装好（真实系统里 rtld 跑 inits 之前就做了），
// 宿主只调了 DT_INIT，没做那一步 ⇒ **所有 malloc 恒返回 NULL** ⇒ 客体 sdk 的
// operator new 走失败分支，最终返回 0xffffffff 被当指针用 ⇒
// memset(0xffffffff, 0, 0x80) 崩（sdk+0x32790c，np242 真机实测）。
// 宿主这份直接接宿主 libc：全进程一个分配器，跨模块 free 也安全。
// ⚠ 已知偏差：客体用 SetHeapSize 预留的那 3110 MB 就成了摆设，游戏改用宿主堆。
//   真要吃满客体堆，正路是复刻 rtld 的 libc 初始化（装那个 hook），后续再做。
HLE_FORCE("malloc", "宿主提供分配器（客体的 malloc 依赖未安装的 hook，恒返回 NULL）",
          void *, h_malloc, size_t n) { return malloc(n); }
HLE_FORCE("calloc", "宿主提供分配器", void *, h_calloc, size_t a, size_t b) { return calloc(a, b); }
HLE_FORCE("realloc", "宿主提供分配器", void *, h_realloc, void *p, size_t n) { return realloc(p, n); }
HLE_FORCE("free", "宿主提供分配器", void, h_free, void *p) { free(p); }
HLE_FORCE("aligned_alloc", "宿主提供分配器", void *, h_aligned_alloc, size_t a, size_t n) { return aligned_alloc(a, n); }

HLE("getentropy", int, g_getentropy, void *p, size_t n) {
    if (!detclock_cfg()) { arc4random_buf(p, n); return 0; }
    for (size_t i = 0; i < n; i += 8) { uint64_t v = dm_entropy64(); memcpy((uint8_t *)p + i, &v, n - i < 8 ? n - i : 8); }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// np252：nn::sm（服务管理器）初始化 —— 宿主没有 sm 服务，空转。
//
// 现场：工人线程跑起来后，线程里 nn::sm::Initialize 的 IPC 那一步（sdk+0x342164，
// sf::hipc::InitializeHipcServiceResolution）返回错误 ⇒ 走 error 分支 ⇒ AbortImpl。
// 我们不需要 sm：各服务（fs / hid / audio / account / time …）都是**按 API 名字**在宿主
// 直接接管的（HLE_FORCE），根本不走 sm 的 IPC。所以把 Initialize 变成"成功空转"。
//
// ⚠ 如果后面出现 nn::sm::GetServiceHandle 被调用（说明有服务没被 HLE，要去找 sm 要端点），
//   那时再单独处理 —— 不要在这里顺手把 GetServiceHandle 也答成功，会把问题推远。
HLE_FORCE("_ZN2nn2sm10InitializeEv", "服务管理器初始化：宿主无 sm 服务，成功空转（各服务按 API 名接管）",
          Result, h_sm_init, void) { return 0; }
HLE_FORCE("_ZN2nn2sm20InitializeForManagerEv", "同上（manager 变体）", Result, h_sm_initmgr, void) { return 0; }
HLE_FORCE("_ZN2nn2sm8FinalizeEv", "服务管理器退出：空转", void, h_sm_fin, void) {}
// np252 续：向 sm 要服务端点 —— 发假句柄，并把**服务名**记下来。
// 这一步能同时办两件事：① 解开 sm::GetServiceHandle 的空指针崩（hid debug pad 初始化在要端点）；
// ② 把"到底哪些服务没被宿主按 API 名接管"直接打印出来（比一个个试错快得多）。
HLE_FORCE("_ZN2nn2sm16GetServiceHandleEPNS_3svc6HandleEPKcm",
          "向 sm 要服务端点：宿主无 sm，给假句柄并记录服务名",
          Result, h_sm_geth, uint32_t *out, const char *name, size_t len) {
    static uint32_t next = 0x5000;
    uint32_t h = next++;
    if (out) *out = h;
    LOG("【黄】sm::GetServiceHandle(\"%.*s\") -> 假句柄 %#x（宿主需按需 HLE 这个服务）",
        (int)(len > 63 ? 63 : (int)len), name ? name : "?", h);
    return 0;
}

// ---------- np282：nn::os 共享内存族（宿主原来一个都没实现）----------
// Background: these went to the unimplemented stub, which leaves x0 untouched. For
// MapSharedMemory(&type, perm) that means the caller's own &type comes back as the "mapped
// address", so the guest memsets its own 48-byte descriptor for 8192 bytes and runs off the end
// (real device: pc=sdk+0x32790c inside memset, addr=0x5e1b401fff, reached from subsdk0+0x320c
// right after CreateSharedMemory(type, 0x2000, perm, perm)).
//
// nn::os::SharedMemoryType is an opaque guest structure whose layout we do not know, so we never
// write into it: the guest pointer is used purely as a key into a host-side table (same
// side-table approach as the NVNbuffer family). That keeps us from guessing field offsets.
#define SHM_MAX 64
static struct { const void *key; void *mem; size_t size; uint32_t handle; } g_shm[SHM_MAX];
static int g_nshm;

static int shm_find(const void *key) {
    for (int i = 0; i < g_nshm; i++) if (g_shm[i].key == key) return i;
    return -1;
}
static void *shm_alloc(size_t n) {          // 客体堆：宿主 malloc 就在客体地址空间内 | Guest heap: host malloc already lives in the guest address space
    return malloc(n ? n : 1);
}
HLE_FORCE("_ZN2nn2os18CreateSharedMemoryEPNS0_16SharedMemoryTypeEmNS0_16MemoryPermissionES3_",
          "建一块共享内存：宿主 malloc 一块，按 type 指针记进边表（不写客体结构）",
          Result, h_os_CreateShm, void *type, size_t size, uint32_t perm, uint32_t perm2) {
    (void)perm; (void)perm2;
    if (!type || size > (256u << 20)) return 1;                  // 非 0 即失败，客体按 Result 判 | Non-zero means failure; the guest checks the Result
    int i = shm_find(type);
    if (i < 0 && g_nshm < SHM_MAX) { i = g_nshm++; g_shm[i].key = type; }
    if (i < 0) return 1;
    free(g_shm[i].mem);
    g_shm[i].mem = shm_alloc(size); g_shm[i].size = size; g_shm[i].handle = 0x6000 + (uint32_t)i;
    if (!g_shm[i].mem) { g_shm[i].key = NULL; g_nshm--; return 1; }
    if (g_flag.imports) LOG("os CreateSharedMemory(%p, %#zx) -> %p (handle %#x)", type, size, g_shm[i].mem, g_shm[i].handle);
    return 0;
}
HLE_FORCE("_ZN2nn2os15MapSharedMemoryEPNS0_16SharedMemoryTypeENS0_16MemoryPermissionE",
          "映射共享内存：返回边表里记的地址（**不能原样返回 &type**，见上方说明）",
          void *, h_os_MapShm, const void *type, uint32_t perm) {
    (void)perm;
    int i = shm_find(type);
    void *p = (i >= 0) ? g_shm[i].mem : NULL;
    if (g_flag.imports) LOG("os MapSharedMemory(%p) -> %p%s", type, p, i >= 0 ? "" : "（边表里没有，返回 NULL）");
    return p;
}
HLE_FORCE("_ZN2nn2os22GetSharedMemoryAddressEPKNS0_16SharedMemoryTypeE",
          "取共享内存地址（与 MapSharedMemory 同一份）", void *, h_os_ShmAddr, const void *type) {
    int i = shm_find(type); return (i >= 0) ? g_shm[i].mem : NULL;
}
HLE_FORCE("_ZN2nn2os21GetSharedMemoryHandleEPKNS0_16SharedMemoryTypeE",
          "取共享内存句柄：给一个稳定假句柄（对端 Attach 用）", uint32_t, h_os_ShmHandle, const void *type) {
    int i = shm_find(type); return (i >= 0) ? g_shm[i].handle : 0;
}
HLE_FORCE("_ZN2nn2os17UnmapSharedMemoryEPNS0_16SharedMemoryTypeE",
          "解除映射：内存仍归 Create 的所有者，这里只记账", void, h_os_UnmapShm, const void *type) { (void)type; }
HLE_FORCE("_ZN2nn2os19DestroySharedMemoryEPNS0_16SharedMemoryTypeE",
          "销毁共享内存：释放并移出边表", void, h_os_DestroyShm, const void *type) {
    int i = shm_find(type);
    if (i >= 0) { free(g_shm[i].mem); g_shm[i] = g_shm[--g_nshm]; }
}
HLE_FORCE("_ZN2nn2os18AttachSharedMemoryEPNS0_16SharedMemoryTypeEmjb",
          "对端附加：按 handle 找到已建的那块，登记到本端 type 上",
          Result, h_os_AttachShm, void *type, size_t size, uint32_t handle, uint8_t) {
    int src = -1;
    for (int i = 0; i < g_nshm; i++) if (g_shm[i].handle == handle) { src = i; break; }
    if (src < 0 || !type) return 1;
    int i = shm_find(type);
    if (i < 0 && g_nshm < SHM_MAX) { i = g_nshm++; g_shm[i].key = type; }
    if (i < 0) return 1;
    g_shm[i].mem = g_shm[src].mem; g_shm[i].size = size ? size : g_shm[src].size; g_shm[i].handle = handle;
    return 0;
}

// ---------- np284：nn::os TLS 槽族（宿主原来一个都没实现）----------
// Background: 0xa0e2f8 (the "get the default allocator" helper) reads a TlsSlot out of an object at
// +0x88 and calls nn::os::GetTlsValue(slot) to fetch the calling thread's allocator. With
// GetTlsValue unimplemented the stub returns the slot number itself (it does not touch x0), so the
// helper ends up treating garbage as an allocator -- which is how the "this = 0" crash at
// main+0xba5ac8 is reached (main+0x634fa0 &c: operator-new-ish helper -> forwarder -> base ctor).
//
// Semantics: a TlsSlot is a small guest-side index; values are per-thread. We back each slot with a
// pthread key created on demand -- the host already has this machinery for its own keys, this is a
// separate namespace so nothing collides.
#define TLS_MAX 128
static pthread_key_t g_tls_key[TLS_MAX];
static uint8_t g_tls_ok[TLS_MAX];      // pthread key 已建
static uint8_t g_tls_alloc[TLS_MAX];   // 已通过 AllocateTlsSlot 交给客体
static pthread_mutex_t g_tls_mu = PTHREAD_MUTEX_INITIALIZER;

static pthread_key_t tls_key(unsigned slot) {
    if (slot >= TLS_MAX) return (pthread_key_t)-1;
    pthread_mutex_lock(&g_tls_mu);
    if (!g_tls_ok[slot]) {
        if (pthread_key_create(&g_tls_key[slot], NULL) == 0) g_tls_ok[slot] = 1;
    }
    pthread_key_t k = g_tls_key[slot];
    pthread_mutex_unlock(&g_tls_mu);
    return k;
}
HLE_FORCE("_ZN2nn2os11GetTlsValueENS0_7TlsSlotE",
          "取线程本地值：按槽号取 pthread 特定值（未建过就是 0）",
          uint64_t, h_os_GetTls, unsigned slot) {
    pthread_key_t k = tls_key(slot);
    if (k == (pthread_key_t)-1) return 0;
    uint64_t v = (uint64_t)(uintptr_t)pthread_getspecific(k);
    // np285：这条链是 this=0 的源头（0xa0e2f8 用它取默认分配器），必须看清谁在设/取
    if (g_flag.imports) { static int n; if (n++ < 24) LOG("os GetTlsValue(slot=%u) -> %#llx", slot, (unsigned long long)v); }
    return v;
}
HLE_FORCE("_ZN2nn2os11SetTlsValueENS0_7TlsSlotEm",
          "设线程本地值", void, h_os_SetTls, unsigned slot, uint64_t v) {
    pthread_key_t k = tls_key(slot);
    if (k != (pthread_key_t)-1) pthread_setspecific(k, (void *)(uintptr_t)v);
    if (g_flag.imports) { static int n; if (n++ < 24) LOG("os SetTlsValue(slot=%u, %#llx)", slot, (unsigned long long)v); }
}
HLE_FORCE("_ZN2nn2os15AllocateTlsSlotEPNS0_7TlsSlotEPFvmE",
          "分配一个 TLS 槽：惰性建 pthread key，槽号就是下标（客体当 TlsSlot 枚举用）",
          Result, h_os_AllocTls, unsigned *out, void (*dtor)(unsigned long)) {
    (void)dtor;                                    // 线程退出时的析构：客体很少用，先不接
    if (!out) return 1;
    if (g_flag.imports) LOG("os AllocateTlsSlot()");
    for (unsigned s = 0; s < TLS_MAX; s++) {
        if (tls_key(s) == (pthread_key_t)-1) continue;
        pthread_mutex_lock(&g_tls_mu);
        int free_slot = !g_tls_alloc[s];
        if (free_slot) g_tls_alloc[s] = 1;
        pthread_mutex_unlock(&g_tls_mu);
        if (free_slot) { *out = s; return 0; }
    }
    return 1;
}
HLE_FORCE("_ZN2nn2os18SdkAllocateTlsSlotEPNS0_7TlsSlotEPFvmE",
          "SDK 版分配 TLS 槽（与上面同一套）", Result, h_os_SdkAllocTls, unsigned *out, void (*dtor)(unsigned long)) {
    return h_os_AllocTls(out, dtor);
}
HLE_FORCE("_ZN2nn2os11FreeTlsSlotENS0_7TlsSlotE", "释放 TLS 槽", void, h_os_FreeTls, unsigned slot) { (void)slot; }
HLE_FORCE("_ZN2nn2os19GetUsedTlsSlotCountEv", "已用 TLS 槽数", int, h_os_TlsCount, void) {
    int n = 0; for (unsigned s = 0; s < TLS_MAX; s++) if (g_tls_ok[s]) n++;
    return n;
}
// libc++ 内部入口（std::__1::__nx_get_tls_value）：与 nn::os::GetTlsValue 同义
HLE_FORCE("_ZNSt3__118__nx_get_tls_valueEN2nn2os7TlsSlotE",
          "libc++ 的 TLS 取值入口（与 nn::os::GetTlsValue 同义）",
          uint64_t, h_nx_GetTls, unsigned slot) { return h_os_GetTls(slot); }
