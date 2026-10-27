// 多模块装载（扮演 Horizon 的 rtld）：映射、占线程槽、门登记、跨模块重定位、按名接管、段权限。
// 代码区（原 text 与补丁区）一拷进来就设成只读+执行，此后装载器只写 data：iOS 上这两块要作为签名代码段原样内嵌。
// Multi-module loading (playing the role of Horizon's rtld): mapping, claiming thread slots, gate registration, cross-module relocation, take-over by name, segment permissions.
// The code area (original text and patch area) is set to read-only+execute as soon as it is copied in, and from then on the loader only writes data: on iOS these two blocks must be embedded as is as signed code segments.
#include <stdatomic.h>
#include <sched.h>
#include <errno.h>
#include "dm.h"
#include <TargetConditionals.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/getsect.h>
#include <mach-o/ldsyms.h>
#endif
#include "hle.h"

module_t g_mod[MAX_MOD];
int g_nmod;
pthread_key_t g_key18, g_keytls, g_keytlr, g_keygate;
pthread_key_t g_keyres[2], g_keyaspect, g_keyaspectui;   // g_keyaspect：宽高比上限参数位（KASPECT）          // 分辨率参数位（改写器 PARAMS，KRES0 掌机 / KRES1 主机）；产物没有即 0、不占 | g_keyaspect: aspect ratio upper-limit parameter slot (KASPECT)          // resolution parameter slots (rewriter PARAMS, KRES0 handheld / KRES1 docked); 0 and not allocated if the build output lacks them
enum { KEY_K18, KEY_KTLS, KEY_KTLR, KEY_KGATE, KEY_KRES0, KEY_KRES1, KEY_KASPECT, KEY_KASPECTUI, KEY_N };

uint64_t resolve_import(module_t *m, const char *name, int weak);
static void tsd_probe(void);
void gate_register(uint64_t ret_addr, const char *kind, int arg, uint64_t orig);

static pthread_key_t key_value(unsigned i) {
    switch (i) {
    case KEY_K18: return g_key18; case KEY_KTLS: return g_keytls; case KEY_KTLR: return g_keytlr;
    case KEY_KGATE: return g_keygate; case KEY_KRES0: return g_keyres[0]; case KEY_KRES1: return g_keyres[1];
    case KEY_KASPECT: return g_keyaspect; default: return g_keyaspectui;
    }
}
static void create_keys(void) {
    if (pthread_key_create(&g_key18, NULL) || pthread_key_create(&g_keytls, NULL) ||
        pthread_key_create(&g_keytlr, NULL) || pthread_key_create(&g_keygate, NULL)) die("创建宿主键失败");
    // 参数位键一律建（不按产物有没有用到判断：iOS 包里没有 params.tsv，按文件判断会漏建、键页写成 0 → 游戏读到 TSD 0 号槽）
    // Parameter slot keys are always created (not decided by whether the build output uses them: an iOS app has no params.tsv, so deciding by file would miss creating them and the key page would be written as 0 → the game reads TSD slot 0)
    if (pthread_key_create(&g_keyres[0], NULL) || pthread_key_create(&g_keyres[1], NULL) ||
        pthread_key_create(&g_keyaspect, NULL) || pthread_key_create(&g_keyaspectui, NULL)) die("创建参数位键失败");
    // 键号先转 unsigned 再放宽：bionic 的 pthread_key_t 是 int（有效键带最高位，如 0x80000020），直接转 unsigned long 会符号扩展成 0xffffffff80000020；苹果是 unsigned long 小整数，打印不变
    // Keys go through unsigned before widening: bionic's pthread_key_t is int (valid keys carry the top bit, e.g. 0x80000020), and a direct unsigned long cast sign-extends to 0xffffffff80000020; on Apple it is a small unsigned long, printed unchanged
    LOG("键号 K18=%lu KTLS=%lu KTLR=%lu KGATE=%lu KRES0=%lu KRES1=%lu KASPECT=%lu KASPECTUI=%lu",
        (unsigned long)(unsigned)g_key18, (unsigned long)(unsigned)g_keytls, (unsigned long)(unsigned)g_keytlr, (unsigned long)(unsigned)g_keygate,
        (unsigned long)(unsigned)g_keyres[0], (unsigned long)(unsigned)g_keyres[1], (unsigned long)(unsigned)g_keyaspect, (unsigned long)(unsigned)g_keyaspectui);
    tsd_probe();
}

// 线程槽基址与键号 → 槽偏移（与 patch_nso.py emit_tp 一一对应）。
// 苹果：pthread TSD 布局固定，槽 = tpidrro_el0 + 8*key。
// 安卓：tpidr_el0 → bionic TCB，TCB[-1]（TLS_SLOT_BIONIC_TLS）= bionic_tls，键 k 的值在 bionic_tls + 键表偏移 + 16k + 8
// （pthread_key_data_t = {seq, data}）。键表偏移是 bionic 内部布局，不写死：装载时设一个已知值、在 bionic_tls 里找到它反推，再用第二个键复核。
// Thread slot base and key number → slot offset (one-to-one with patch_nso.py emit_tp).
// Apple: the pthread TSD layout is fixed, slot = tpidrro_el0 + 8*key.
// Android: tpidr_el0 → bionic TCB, TCB[-1] (TLS_SLOT_BIONIC_TLS) = bionic_tls, the value of key k lives at bionic_tls + key table offset + 16k + 8
// (pthread_key_data_t = {seq, data}). The key table offset is a bionic internal layout and is not hard-coded: at load time set a known value, find it in bionic_tls to derive the offset, then re-check with a second key.
#ifdef __ANDROID__
#define DM_PATCH_TARGET 1
static uint64_t g_tsd_off;
static uint8_t *tsd_base(void) { uint64_t tp; __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp)); return ((uint8_t **)tp)[-1]; }
// bionic 的键号 = 键表下标 | KEY_VALID_FLAG（第 31 位，实测 0x80000006 这种），取下标要去掉标记位
// A bionic key number = key table index | KEY_VALID_FLAG (bit 31; measured values look like 0x80000006); strip the flag bit to get the index
static uint64_t key_index(pthread_key_t k) { return (uint32_t)k & 0x7fffffffu; }
static uint64_t slot_off(pthread_key_t k) { return g_tsd_off + key_index(k) * 16 + 8; }
static void tsd_probe(void) {
    void *const magic = (void *)0x6d4d7d5d4d3d2d1dull, *const magic2 = (void *)0x1d2d3d4d5d7d4d6dull;
    pthread_setspecific(g_key18, magic);
    uint8_t *b = tsd_base(); uint64_t at = 0;
    for (uint64_t o = 8; o < 0x4000 && !at; o += 8) if (*(void **)(b + o) == magic) at = o;
    pthread_setspecific(g_key18, 0);
    if (!at || at < key_index(g_key18) * 16 + 8) die("bionic 键表没找到（键 %#x 设的值不在 bionic_tls 前 16 KB 里）", (unsigned)g_key18);
    g_tsd_off = at - key_index(g_key18) * 16 - 8;
    pthread_setspecific(g_keygate, magic2);
    int ok = *(void **)(b + slot_off(g_keygate)) == magic2;
    pthread_setspecific(g_keygate, 0);
    if (!ok) die("bionic 键表布局不符（键表偏移 %#llx 对键 %#x 不成立）", (unsigned long long)g_tsd_off, (unsigned)g_keygate);
    LOG("bionic 键表偏移 %#llx", (unsigned long long)g_tsd_off);
}
#elif defined(__OHOS__)
// 鸿蒙（musl）：tpidr_el0 − 272 处是 struct pthread 的 tsd 指针，键 k 的值 = tsd[k]（Mate 90 / API 26 真机实测，子线程与主线程一致，
// 见 dm/harmony/probe）。偏移由 patch_nso.py DM_TARGET=ohos 写进改写代码；这里装载时复核，系统 musl 布局一变即停，不读错槽。
// HarmonyOS (musl): tpidr_el0 - 272 holds struct pthread's tsd pointer, and key k's value = tsd[k] (measured on a Mate 90 / API 26,
// same for worker and main threads; see dm/harmony/probe). patch_nso.py DM_TARGET=ohos bakes the offset into the patch code; the
// loader re-checks it here and stops if the system musl layout ever changes, rather than reading the wrong slot.
#define DM_PATCH_TARGET 2
#define DM_OHOS_TSD_OFF 272
static uint8_t *tsd_base(void) { uint64_t tp; __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tp)); return *(uint8_t **)(tp - DM_OHOS_TSD_OFF); }
static uint64_t slot_off(pthread_key_t k) { return (uint64_t)k * 8; }
static void tsd_probe(void) {
    void *const magic = (void *)0x6d4d7d5d4d3d2d1dull;
    pthread_setspecific(g_keygate, magic);
    int ok = *(void **)(tsd_base() + slot_off(g_keygate)) == magic;
    pthread_setspecific(g_keygate, 0);
    if (!ok) die("musl 线程键布局不符（tpidr_el0-%d 处不是 tsd 指针）：系统 libc 变了，需重测偏移并重出改写产物", DM_OHOS_TSD_OFF);
}
#else
#define DM_PATCH_TARGET 0
static uint8_t *tsd_base(void) { uint64_t b; __asm__ volatile("mrs %0, tpidrro_el0" : "=r"(b)); return (uint8_t *)(b & ~7ull); }
static uint64_t slot_off(pthread_key_t k) { return (uint64_t)k * 8; }
static void tsd_probe(void) {}
#endif

// np286：把 musl/bionic 的线程特定数据区开放给 nn::os TLS 族用。
// 客体线程就是宿主 pthread，客体代码里的 TLS 访问已被 patch_nso.py emit_tp 改写成读同一片 tsd，
// 所以 nn::os::GetTlsValue/SetTlsValue 也必须落在这里 —— 另起一套 pthread key 的话两边看到
// 的不是同一份存储（np285 真机实测：客体只调 AllocateTlsSlot，从不调我们的 Get/SetTlsValue）。
// 这里只做"按键号读/写 tsd 槽"，与客体 emit_tp 的算法保持一致。
// np286：nn::os 的 TlsSlot 与宿主自己的 pthread key **共用同一片 tsd**，而客体从 0 开始编号
// （真机实测 slot=0/4），正好盖住宿主的 K18/KTLS/… 那几个键 —— 上一轮就这么把宿主内部踩坏，
// 崩在 libc 里（addr=0x351）。musl 的 tsd 有 128 槽、宿主只用前 8 个，
// 所以给 nn::os 开一段独立区间：下标 DM_TLS_BASE 起，永不与宿主 key 相撞。
#define DM_TLS_BASE 32
void *dm_tsd_base(void) { return tsd_base(); }
uint64_t dm_tsd_read(unsigned key) { return *(volatile uint64_t *)(tsd_base() + slot_off((pthread_key_t)(DM_TLS_BASE + key))); }
void dm_tsd_write(unsigned key, uint64_t v) { *(volatile uint64_t *)(tsd_base() + slot_off((pthread_key_t)(DM_TLS_BASE + key))) = v; }
// K 槽校验：经键页取到的槽与 pthread_setspecific 写的是同一格
// K slot check: the slot reached through the key page is the same cell that pthread_setspecific writes
static void tsd_check(void) {
    pthread_setspecific(g_key18, (void *)0x1234);
    uint64_t off = *(uint64_t *)(g_mod[0].base + g_mod[0].key_page_off + g_mod[0].key_idx[KEY_K18] * 8);
    if (*(uint64_t *)(tsd_base() + off) != 0x1234) die("TSD 布局不符");
    pthread_setspecific(g_key18, 0);
}

static void build_symhash(module_t *m);
static void parse_dynamic(module_t *m) {
    uint32_t mod0 = *(uint32_t *)(m->base + 4);
    if (memcmp(m->base + mod0, "MOD0", 4)) die("%s: no MOD0", m->name);
    m->dyn = (const Elf64_Dyn *)(m->base + mod0 + *(int32_t *)(m->base + mod0 + 4));
    for (const Elf64_Dyn *d = m->dyn; d->d_tag; d++) switch (d->d_tag) {
        case DT_SYMTAB: m->symtab = (const Elf64_Sym *)(m->base + d->d_un.d_ptr); break;
        case DT_STRTAB: m->strtab = (const char *)(m->base + d->d_un.d_ptr); break;
        case DT_INIT: m->init = d->d_un.d_ptr; break;
        case DT_INIT_ARRAY: m->init_array = d->d_un.d_ptr; break;
        case DT_INIT_ARRAYSZ: m->init_arraysz = d->d_un.d_val; break;
    }
    m->nsym = (unsigned)(((const uint8_t *)m->strtab - (const uint8_t *)m->symtab) / sizeof(Elf64_Sym));
    build_symhash(m);
}

static void *module_sym_linear(module_t *m, const char *name) {
    for (unsigned i = 1; i < m->nsym; i++) {
        const Elf64_Sym *s = &m->symtab[i];
        if (s->st_shndx && !strcmp(m->strtab + s->st_name, name)) return m->base + s->st_value;
    }
    return NULL;
}
// 符号名哈希表：按下标升序插入、同名已在则不覆盖 ⇒ 查询结果与线性扫描（首个已定义同名符号）逐条相同。
// ① 宿主装载 91–99 ms 里重定位 + 强制接管占 77–86 ms，主因是每条按名解析的重定位都线性扫两个模块的符号表
// Symbol name hash table: inserted in ascending index order, not overwritten if the same name is already there ⇒ lookup results are identical, entry by entry, to a linear scan (the first defined symbol with that name).
// Of the 91–99 ms of ① host loading, relocation + forced take-over took 77–86 ms, mainly because every by-name relocation scanned the symbol tables of two modules linearly
static uint32_t sym_hash(const char *s) { uint32_t h = 2166136261u; while (*s) h = (h ^ (uint8_t)*s++) * 16777619u; return h; }
static void build_symhash(module_t *m) {
    unsigned cap = 16; while (cap < m->nsym * 2) cap <<= 1;
    m->symh = calloc(cap, sizeof *m->symh); m->symh_mask = cap - 1;
    for (unsigned i = 1; i < m->nsym; i++) {
        const Elf64_Sym *s = &m->symtab[i]; if (!s->st_shndx) continue;
        const char *nm = m->strtab + s->st_name;
        for (unsigned j = sym_hash(nm) & m->symh_mask;; j = (j + 1) & m->symh_mask) {
            if (!m->symh[j]) { m->symh[j] = i; break; }
            if (!strcmp(m->strtab + m->symtab[m->symh[j]].st_name, nm)) break;   // 同名已有更小下标：保留它 | The same name already exists with a smaller index: keep that one
        }
    }
}
static unsigned long g_symchk_n, g_symchk_bad;
void *module_sym(module_t *m, const char *name) {
    unsigned long nq; (void)nq;
    if (!m->symh) return module_sym_linear(m, name);
    void *r = NULL;
    for (unsigned j = sym_hash(name) & m->symh_mask; m->symh[j]; j = (j + 1) & m->symh_mask) {
        const Elf64_Sym *s = &m->symtab[m->symh[j]];
        if (!strcmp(m->strtab + s->st_name, name)) { r = m->base + s->st_value; break; }
    }
    static int chk = -1; if (chk < 0) chk = getenv("DM_SYMCHECK") != NULL;
    if (chk) { void *l = module_sym_linear(m, name); g_symchk_n++; nq = g_symchk_n;
               if (l != r) { g_symchk_bad++; LOG("【红】DM_SYMCHECK：%s 在 %s 哈希 %p ≠ 线性 %p", name, m->name, r, l); } }
    return r;
}
void *guest_sym(const char *name) {
    for (int i = 0; i < g_nmod; i++) { void *p = module_sym(&g_mod[i], name); if (p) return p; }
    return NULL;
}
module_t *module_of(const void *p) {
    for (int i = 0; i < g_nmod; i++)
        if ((const uint8_t *)p >= g_mod[i].base && (const uint8_t *)p < g_mod[i].base + g_mod[i].size) return &g_mod[i];
    return NULL;
}

// 跨模块符号解析：先在已装模块里找定义，找不到再交给 HLE / 桩
// Cross-module symbol resolution: first look for a definition in the loaded modules, and only if not found hand over to HLE / stubs
static void *force_lookup(const char *name);   // np323：前置声明
static uint64_t resolve_sym(module_t *m, const char *name, int weak) {
    // np323：**强制接管优先于模块内定义**。
    // 上面那条注释（"命中即覆盖模块内定义"）一直是这个设计意图，但代码原来是先找模块定义，
    // 于是"某个模块自己也实现了同名符号"时宿主替身永远轮不到 —— 门票① 的
    // nn::mem::StandardAllocator::Allocate 正是这种情况（sdk 自己定义了它）。
    // np323: forced handlers now take precedence over in-module definitions, which is what the comment above
    // always intended. Before, a module that defined the same symbol won, so the host replacement was never
    // reached -- exactly the case for nn::mem::StandardAllocator::Allocate (defined inside sdk).
    void *f = force_lookup(name);
    if (f) return (uint64_t)f;
    for (int i = 0; i < g_nmod; i++) { void *p = module_sym(&g_mod[i], name); if (p) return (uint64_t)p; }
    return resolve_import(m, name, weak);
}

// 强制接管表（hle.h HLE_FORCE，__DATA,swforce 段）：命中即覆盖模块内定义
// Forced take-over table (hle.h HLE_FORCE, the __DATA,swforce section): a hit overrides the definition inside the module
static const hle_force_ent *g_force; static unsigned g_nforce; static unsigned *g_force_hits;
static void force_init(void) {
    unsigned long sz; DM_SECT_GET(swforce, const hle_force_ent, g_force, sz);
    g_nforce = g_force ? (unsigned)(sz / sizeof *g_force) : 0;
    g_force_hits = calloc(g_nforce + 1, sizeof *g_force_hits);
    for (unsigned i = 0; i < g_nforce; i++)
        for (unsigned j = i + 1; j < g_nforce; j++)
            if (!strcmp(g_force[i].name, g_force[j].name)) die("强制接管清单重复：%s", g_force[i].name);
}
static void *force_lookup(const char *name) {
    for (unsigned i = 0; i < g_nforce; i++)
        if (!strcmp(g_force[i].name, name)) {
            if (!g_force_hits[i]++ && g_flag.imports) LOG("force %s  （%s）", name, g_force[i].why);
            void *hlecount_wrap(const char *, const char *, void *); return hlecount_wrap("FORCE", name, g_force[i].fn);   // DM_HLECOUNT 时为计数桩 | A counting stub when DM_HLECOUNT is set
        }
    return NULL;
}
static void force_check(void) {
    // np234（换游戏）：swforce 段是**按游戏**生成的（D2R 的清单里有 nn::oe/nn::fs 等它自己的导入）。
    // 换一款游戏时，清单里属于原游戏的条目自然没有引用——那是预期的，不该因此中止。
    // 真正要抓的错误（产物与宿主不配套）由别处校验：swhle 的符号表匹配失败会逐条报红。
    // 所以这里只记警告，不 die。
    int miss = 0;
    for (unsigned i = 0; i < g_nforce; i++)
        if (!g_force_hits[i]) { LOG("【黄】强制接管条目无任何引用（换游戏时预期如此）：%s", g_force[i].name); miss++; }
    LOG("force: %u 条，命中 %u 条，未引用 %u 条（未引用不致命）", g_nforce, g_nforce - miss, miss);
}

// 读改写器表头（段边界、槽号）；表头以外有任何行即旧版产物（装载期改代码），拒绝
// Read the rewriter table header (segment boundaries, slot numbers); any row beyond the header means an old-format build output (code modified at load time), which is rejected
static void read_layout(module_t *m, const char *dir) {
    char path[1024]; snprintf(path, sizeof path, "%s/%s.fixups.tsv", dir, m->name);
    FILE *f = fopen(path, "r");
    if (!f) die("open %s", path);
    static const char *kn[8] = {"K18", "KTLS", "KTLR", "KGATE", "KRES0", "KRES1", "KASPECT", "KASPECTUI"};
    char line[256], key[32]; unsigned long v, target = 0; unsigned seen = 0;   // 没有 TARGET 行的产物 = 苹果（早于安卓的产物） | A build output without a TARGET line = Apple (predates Android)
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "# %31s %lx", key, &v) != 2) die("%s: 产物是旧改写器出的，换成当前代号（dm/PRODUCT）的产物", path);
        if (!strcmp(key, "TARGET")) target = v;
        else if (!strcmp(key, "PATCH_OFF")) m->patch_off = v;
        else if (!strcmp(key, "KEY_PAGE_OFF")) m->key_page_off = v;
        else if (!strcmp(key, "KEY_PAGE_SIZE")) m->key_page_size = v;
        else if (!strcmp(key, "TEXT_END")) m->text_end = v;
        else if (!strcmp(key, "RO_OFF")) m->ro_off = v;
        else if (!strcmp(key, "DATA_OFF")) m->data_off = v;
        else for (int k = 0; k < 8; k++) {
            size_t n = strlen(kn[k]);
            if (!strncmp(key, "KEY_", 4) && !strncmp(key + 4, kn[k], n) && !strcmp(key + 4 + n, "_IDX")) {
                m->key_idx[k] = (unsigned)strtoul(strrchr(line, '\t') + 1, 0, 10); seen |= 1u << k;
            }
        }
    }
    fclose(f);
    if (!m->patch_off || m->key_page_off % 0x4000 || m->key_page_size != 0x4000 || seen != 0xff)
        die("%s: 产物是旧改写器出的，换成当前代号（dm/PRODUCT）的产物", path);
    static const char *const tname[] = {"苹果", "安卓", "鸿蒙"}, *const tflag[] = {"apple", "android", "ohos"};
    if (target != DM_PATCH_TARGET) die("%s: 产物是给%s出的（TARGET %lu），本机要 %s 的产物（patch_nso.py 用 DM_TARGET=%s 重出）", path,
                                       target < 3 ? tname[target] : "未知平台", target, tname[DM_PATCH_TARGET], tflag[DM_PATCH_TARGET]);
}

static void install_key_page(module_t *m) {
    if (m->key_page_off + m->key_page_size > m->size) die("%s: 键页越界", m->name);
    uint64_t *page = (uint64_t *)(m->base + m->key_page_off);
    memset(page, 0, m->key_page_size);
    for (unsigned i = 0; i < KEY_N; i++) page[m->key_idx[i]] = slot_off(key_value(i));
    for (int k = 0; k < 8; k++) if (!page[m->key_idx[k]]) die("%s: 键页第 %d 项为 0（宿主键没建）——游戏会读到 TSD 0 号槽", m->name, k);
    if (mprotect(m->base + m->key_page_off, m->key_page_size, PROT_READ)) die("%s 键页 mprotect", m->name);
}

static void register_gates(module_t *m, const char *dir) {
    char path[1024], line[256]; snprintf(path, sizeof path, "%s/%s.gates.tsv", dir, m->name);
    FILE *f = fopen(path, "r");
    int ng = 0;
    if (f) {
        unsigned long ret, orig; char kind[16]; int arg;
        while (fgets(line, sizeof line, f))
            if (sscanf(line, "%lx %15s %d %lx", &ret, kind, &arg, &orig) == 4) { gate_register((uint64_t)m->base + ret, kind, arg, orig); ng++; }
        fclose(f);
    }
    LOG("%s: gates %d patch_off %#llx keys %lu/%lu/%lu/%lu", m->name, ng, m->patch_off, (unsigned long)(unsigned)g_key18, (unsigned long)(unsigned)g_keytls, (unsigned long)(unsigned)g_keytlr, (unsigned long)(unsigned)g_keygate);   // 同上：bionic 的 pthread_key_t 是 int | as above: bionic pthread_key_t is int
}

static void relocate(module_t *m) {
    uint64_t rela = 0, relasz = 0, jmprel = 0, pltsz = 0;
    for (const Elf64_Dyn *d = m->dyn; d->d_tag; d++) switch (d->d_tag) {
        case DT_RELA: rela = d->d_un.d_ptr; break;
        case DT_RELASZ: relasz = d->d_un.d_val; break;
        case DT_JMPREL: jmprel = d->d_un.d_ptr; break;
        case DT_PLTRELSZ: pltsz = d->d_un.d_val; break;
    }
    int cnt[3] = {0};
    const Elf64_Rela *tabs[2] = {(const Elf64_Rela *)(m->base + rela), (const Elf64_Rela *)(m->base + jmprel)};
    uint64_t sizes[2] = {relasz, pltsz};
    for (int t = 0; t < 2; t++)
        for (uint64_t i = 0; i < sizes[t] / sizeof(Elf64_Rela); i++) {
            const Elf64_Rela *r = &tabs[t][i];
            uint64_t *where = (uint64_t *)(m->base + r->r_offset);
            unsigned type = ELF64_R_TYPE(r->r_info), si = ELF64_R_SYM(r->r_info);
            if (type == R_AARCH64_RELATIVE) { *where = (uint64_t)m->base + r->r_addend; cnt[0]++; continue; }
            const Elf64_Sym *s = &m->symtab[si];
            uint64_t v = (uint64_t)force_lookup(m->strtab + s->st_name);
            if (!v) v = s->st_shndx ? (uint64_t)m->base + s->st_value : resolve_sym(m, m->strtab + s->st_name, (s->st_info >> 4) == 2);
            // np241：按槽位打印符号名（**独立开关 DM_RELOCS**，不跟 DM_IMPORTS 走）。
            // 客体代码里的导入桩是 adrp+ldr[槽]+br，崩溃回溯给出桩地址时，
            // (桩表基址, 槽内偏移) 就能在这里反查是谁。
            // ⚠ 六个模块合计 9 万条重定位，跟着 DM_IMPORTS 全会把 hilog 冲爆——
            //   所以单独一个开关，只在"要核对某个槽解析到哪"时开。
            static int relocs_log = -1;
            if (relocs_log < 0) relocs_log = getenv("DM_RELOCS") != NULL;
            // 两个关键符号始终打（量小，排障必备）：C++ 静态构造守卫与 std::locale 全局对象
            if (!relocs_log) {
                const char *sn = m->strtab + s->st_name;
                if (!strcmp(sn, "__cxa_guard_acquire") || !strcmp(sn, "_ZNSt3__16locale8__globalEv"))
                    LOG("KEYRELOC %s slot=%#lx %s -> %#lx", m->name, r->r_offset, sn, (unsigned long)v);
            }
            if (relocs_log)
                LOG("reloc %s slot=%#lx type=%u %s -> %#lx", m->name, r->r_offset, type,
                    m->strtab + s->st_name, (unsigned long)v);
            if (type == R_AARCH64_JUMP_SLOT) { *where = v; cnt[2]++; }
            else { *where = v + r->r_addend; cnt[1]++; }       // ABS64 / GLOB_DAT
        }
    LOG("%s: reloc relative=%d abs/glob=%d jump=%d", m->name, cnt[0], cnt[1], cnt[2]);
}

// DM_LOADHASH=1（诊断）：装载完逐模块出地址无关 sha256，证「提前起跑 / 不提前」装载结果逐字节相同。
// 模块基址与宿主 ASLR 每局不同，只有重定位槽的值随之变：复制映像，按同一张重定位表把每个槽换成与地址无关的记号——
// RELATIVE → (0, 加数)；符号目标落在某模块 → (1 + 模块号, 模块内偏移)；否则（宿主强制接管 / 宿主符号）→ (名字 FNV, 加数)；其余字节原样。
// DM_LOADHASH=1 (diagnostics): after loading, output an address-independent sha256 per module, to prove that the load results of "early start / no early start" are byte-identical.
// Module base addresses and host ASLR differ every run, and only the values of the relocation slots change with them: copy the image and, using the same relocation table, replace every slot with an address-independent token --
// RELATIVE → (0, addend); symbol target falls in some module → (1 + module number, offset within the module); otherwise (host forced take-over / host symbol) → (name FNV, addend); all other bytes as is.
#include <CommonCrypto/CommonDigest.h>
void loader_hash_log(void) {
    if (!getenv("DM_LOADHASH")) return;
    for (int k = 0; k < g_nmod; k++) {
        module_t *m = &g_mod[k];
        uint8_t *cp = malloc(m->size); memcpy(cp, m->base, m->size);
        uint64_t rela = 0, relasz = 0, jmprel = 0, pltsz = 0;
        for (const Elf64_Dyn *d = m->dyn; d->d_tag; d++) switch (d->d_tag) {
            case DT_RELA: rela = d->d_un.d_ptr; break; case DT_RELASZ: relasz = d->d_un.d_val; break;
            case DT_JMPREL: jmprel = d->d_un.d_ptr; break; case DT_PLTRELSZ: pltsz = d->d_un.d_val; break;
        }
        const Elf64_Rela *tabs[2] = {(const Elf64_Rela *)(m->base + rela), (const Elf64_Rela *)(m->base + jmprel)};
        uint64_t sizes[2] = {relasz, pltsz}; unsigned nslot = 0;
        for (int t = 0; t < 2; t++)
            for (uint64_t i = 0; i < sizes[t] / sizeof(Elf64_Rela); i++) {
                const Elf64_Rela *r = &tabs[t][i]; uint64_t tok[1] = {0};
                if (ELF64_R_TYPE(r->r_info) != R_AARCH64_RELATIVE) {
                    uint64_t v = *(const uint64_t *)(m->base + r->r_offset); int hit = 0;
                    for (int j = 0; j < g_nmod; j++) if (v >= (uint64_t)g_mod[j].base && v < (uint64_t)g_mod[j].base + g_mod[j].size) { tok[0] = 1 + j; hit = 1; break; }
                    if (!hit) { const char *nm = m->strtab + m->symtab[ELF64_R_SYM(r->r_info)].st_name; uint64_t h = 0xcbf29ce484222325ull;
                                for (; *nm; nm++) h = (h ^ (uint8_t)*nm) * 0x100000001b3ull; tok[0] = h | 1ull << 63; }
                }
                memcpy(cp + r->r_offset, &tok[0], 8);                // 槽 8 字节换成记号头；记号尾在下面按槽序追加 | The 8 slot bytes are replaced by the token head; the token tail is appended below in slot order
                nslot++;
            }
        CC_SHA256_CTX c; CC_SHA256_Init(&c); CC_SHA256_Update(&c, cp, (CC_LONG)m->size);
        for (int t = 0; t < 2; t++)                                // 记号尾（加数 / 偏移）按槽序追加，与槽头一起唯一确定槽值 | Token tails (addend / offset) are appended in slot order, and together with the slot head uniquely determine the slot value
            for (uint64_t i = 0; i < sizes[t] / sizeof(Elf64_Rela); i++) {
                const Elf64_Rela *r = &tabs[t][i]; uint64_t tail = (uint64_t)r->r_addend;
                if (ELF64_R_TYPE(r->r_info) != R_AARCH64_RELATIVE) {
                    uint64_t v = *(const uint64_t *)(m->base + r->r_offset);
                    for (int j = 0; j < g_nmod; j++) if (v >= (uint64_t)g_mod[j].base && v < (uint64_t)g_mod[j].base + g_mod[j].size) { tail = v - (uint64_t)g_mod[j].base; break; }
                }
                CC_SHA256_Update(&c, &tail, 8);
            }
        uint8_t d[32]; CC_SHA256_Final(d, &c); free(cp);
        char hx[65]; for (int i = 0; i < 32; i++) sprintf(hx + 2 * i, "%02x", d[i]);
        LOG("DM_LOADHASH %s 大小 %#zx 重定位槽 %u sha256 %s", m->name, m->size, nslot, hx);
    }
}

#if TARGET_OS_IPHONE
// iOS 不许运行时装入的代码执行，且苹果只许可执行文件有一个带执行权限的段（ITMS-90999）：映像由 dm/ios/tools/gen_embed.py 编进本可执行文件——
// 代码两块（原 text、补丁区）是 __TEXT 里的节，W（ro 余部 + data + bss）是单独的 RW 段。三块在文件里不相邻；这里保留一块连续地址，
// 按原相对位置 vm_remap 回去：代码页 copy=FALSE（与 __TEXT 共享同一批已签名页，r-x），W copy=TRUE（写时复制，用到才分页）。
// 运行时布局与原映像逐字节相同。iOS 上把已签名页 remap 到新地址再执行，libffi 的跳板表（FFI_EXEC_TRAMPOLINE_TABLE）就是这么做的
// iOS does not allow code loaded at run time to execute, and Apple allows an executable only one segment with execute permission (ITMS-90999): the image is compiled into this executable by dm/ios/tools/gen_embed.py --
// the two code blocks (original text, patch area) are sections in __TEXT, and W (the rest of ro + data + bss) is a separate RW segment. The three blocks are not adjacent in the file; here one contiguous address range is reserved,
// and they are vm_remap'ed back at their original relative positions: code pages with copy=FALSE (sharing the same signed pages with __TEXT, r-x), W with copy=TRUE (copy-on-write, paged only when used).
// The run-time layout is byte-identical to the original image. Remapping signed pages to a new address and then executing them on iOS is exactly what libffi's trampoline table (FFI_EXEC_TRAMPOLINE_TABLE) does
#include <mach/mach.h>
static void remap_at(module_t *m, vm_address_t to, const void *from, unsigned long size, boolean_t copy, vm_prot_t need) {
    vm_address_t a = to; vm_prot_t cur = 0, max = 0;
    kern_return_t kr = vm_remap(mach_task_self(), &a, size, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, mach_task_self(), (vm_address_t)from, copy, &cur, &max, VM_INHERIT_NONE);
    if (kr != KERN_SUCCESS || a != to) die("%s vm_remap %p → %#lx（%#lx 字节）失败：%d", m->name, from, (unsigned long)to, size, kr);
    if ((cur & need) != need) die("%s vm_remap 后权限 %#x，缺 %#x", m->name, cur, need);
}
#ifdef DM_SHELL
// 壳（ios/tools/shell_build.sh 出的、不含任何游戏内容的 App）：模块映像不在主可执行文件里，在使用者自己用游戏文件做出来的
// Frameworks/Guest.framework/Guest 里——一个只装段的动态库（dm/tools/guest_dylib.py 写出），节名与内嵌做法相同，下面只换一个映像头去取。
// Shell (the App built by ios/tools/shell_build.sh, containing no game content at all): the module images are not in the main executable but in the
// Frameworks/Guest.framework/Guest that users build themselves from their game files -- a dynamic library that holds only segments (written by dm/tools/guest_dylib.py), with the same section names as the embedded approach; below only the image header to fetch from is swapped.
#include <dlfcn.h>
#include <mach-o/dyld.h>
static const struct mach_header_64 *image_mh(void) {
    static const struct mach_header_64 *mh; if (mh) return mh;
    char exe[4096]; uint32_t n = sizeof exe; if (_NSGetExecutablePath(exe, &n)) die("取不到可执行文件路径");
    char *sl = strrchr(exe, '/'); if (!sl || (size_t)(sl - exe) + 40 > sizeof exe) die("可执行文件路径不对：%s", exe); strcpy(sl, "/Frameworks/Guest.framework/Guest");
    if (!dlopen(exe, RTLD_NOW)) die("这个安装包里还没有游戏程序（%s）：先用 dm/tools/d2rm_pack.py 把你自己的游戏文件做进安装包再装。系统的说法：%s", exe, dlerror());
    for (uint32_t i = 0; i < _dyld_image_count(); i++) if (strstr(_dyld_get_image_name(i), "/Guest.framework/Guest")) mh = (const struct mach_header_64 *)_dyld_get_image_header(i);
    if (!mh) die("Guest 动态库已加载但找不到它的映像头");
    fprintf(stderr, "boot: 模块映像取自 Guest 动态库 %s\n", exe); return mh;
}
#define IMAGE_MH image_mh()
#else
#define IMAGE_MH (&_mh_execute_header)
#endif
static void map_image(module_t *m, const char *dir) {
    (void)dir;
    char sx0[17], sx1[17], dm[17]; snprintf(sx0, sizeof sx0, "__dm%sx0", m->name); snprintf(sx1, sizeof sx1, "__dm%sx1", m->name); snprintf(dm, sizeof dm, "__DM%sW", m->name);
    unsigned long z0, zw, z1;
    const uint8_t *x0 = getsectiondata(IMAGE_MH, "__TEXT", sx0, &z0), *w = getsectiondata(IMAGE_MH, dm, "__img", &zw), *x1 = getsectiondata(IMAGE_MH, "__TEXT", sx1, &z1);
    if (!x0 || !w || !x1) die("内嵌节 %s / %s / %s 缺失", sx0, dm, sx1);
    if (z0 + zw != m->patch_off || (z0 & 0x3FFF) || (zw & 0x3FFF) || (z1 & 0x3FFF) || ((uintptr_t)x0 & 0x3FFF) || ((uintptr_t)w & 0x3FFF) || ((uintptr_t)x1 & 0x3FFF))
        die("%s 内嵌节布局不符（x0 %#lx w %#lx x1 %#lx，PATCH_OFF %#llx）", m->name, z0, zw, z1, (unsigned long long)m->patch_off);
    m->size = m->patch_off + z1;
    vm_address_t base = 0;
    if (vm_allocate(mach_task_self(), &base, m->size, VM_FLAGS_ANYWHERE) != KERN_SUCCESS) die("%s 保留 %#zx 字节地址失败", m->name, m->size);
    remap_at(m, base, x0, z0, FALSE, VM_PROT_READ | VM_PROT_EXECUTE);
    remap_at(m, base + z0, w, zw, TRUE, VM_PROT_READ | VM_PROT_WRITE);
    remap_at(m, base + m->patch_off, x1, z1, FALSE, VM_PROT_READ | VM_PROT_EXECUTE);
    m->base = (uint8_t *)base; install_key_page(m);
}
#else
static uint8_t *slurp(const char *path, size_t *n) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("open %s", path);
    struct stat st; fstat(fd, &st);
    uint8_t *p = malloc(st.st_size);
    if (read(fd, p, st.st_size) != st.st_size) die("read %s", path);
    close(fd); *n = st.st_size; return p;
}

// 拷入匿名页后立即封代码区（页 16 KB）：[0, text_end) 与补丁区只读+执行，此后对代码区的任何写入都会当场崩；
// 之间（ro 与 data 同页只能一起）可写
// Seal the code area immediately after copying into anonymous pages (16 KB pages): [0, text_end) and the patch area are read-only+execute, and from then on any write to the code area crashes on the spot;
// what lies between (ro and data on the same page can only go together) is writable
static void map_image(module_t *m, const char *dir) {
    char path[1024]; snprintf(path, sizeof path, "%s/%s.patched.bin", dir, m->name);
    size_t sz; uint8_t *raw = slurp(path, &sz);
    m->size = (sz + 0x3FFF) & ~0x3FFFull;
    m->base = mmap(NULL, m->size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m->base == MAP_FAILED) die("%s mmap", m->name);
    memcpy(m->base, raw, sz); free(raw); install_key_page(m);
    uint64_t rx = (m->text_end + 0x3FFF) & ~0x3FFFull;
    if (mprotect(m->base, rx, PROT_READ | PROT_EXEC)) die("%s mprotect text", m->name);
    uint64_t po = m->patch_off & ~0x3FFFull;
    if (mprotect(m->base + po, m->size - po, PROT_READ | PROT_EXEC)) die("%s mprotect patch", m->name);
}
#endif

// ---------- 扮演 rtld：装 libc 初始化 hook（np292） ----------
// 客体自己的 malloc / calloc / realloc / free（main+0x9b27a0 / 0x9b2810 / 0x9b2888 / 0x9b28e4）
// 第一步都是同一个门：`ldr x9,[main+0x22d7230]` → `cbz x9` 就直接返回 NULL。
// 这 8 字节本该由 rtld / libc 初始化写好（真机上 rtld 在跑各模块 inits **之前**就做了），
// 而宿主只调了 DT_INIT、没复刻那一步 ⇒ 客体自己的 malloc 族恒返回 NULL（np242 首次记下）。
// np242 当时用「宿主顶替**导入**的 malloc」绕过，但客体内部**直接 bl 0x9b27a0** 的路径绕不过去
// —— np291 的 this=0 崩点正是从这里来的：分配返回 0、调用方不判空直接写。
// 四处读者全部只 `cbz` 判空、从不把它的值当函数指针用 ⇒ 写任意非零即可，写 1 最安全。
// 装好之后客体 malloc 走它自己的 nn::mem::StandardAllocator（main+0x22d71e8，3110 MB 客堆），
// 这才是这套客体地址空间设计本来的样子；宿主顶替的导入版留给 rtld 自己的早期分配。
// ---------- Playing rtld: install the libc init hook (np292) ----------
// The guest's own malloc / calloc / realloc / free (main+0x9b27a0 / 0x9b2810 / 0x9b2888 / 0x9b28e4)
// all start with the same gate: `ldr x9,[main+0x22d7230]` -> `cbz x9` returns NULL right away.
// Those 8 bytes are supposed to be written by rtld / libc init (on the real device rtld does it
// **before** running each module's inits), but the host only calls DT_INIT and never replicated that
// step, so the guest's own malloc family always returns NULL (first recorded in np242).
// np242 worked around it by having the host provide the **imported** malloc, but paths that
// **bl 0x9b27a0 directly** inside the guest cannot be bypassed that way -- and that is exactly where
// np291's this=0 crash comes from: the allocation returns 0 and the caller writes through it unchecked.
// All four readers only `cbz`-test it and never use its value as a function pointer, so any non-zero
// works; 1 is the safest. Once installed, the guest's malloc uses its own nn::mem::StandardAllocator
// (main+0x22d71e8, the 3110 MB guest heap), which is what this guest address-space design intends;
// the host-provided imported malloc stays for rtld's own early allocations.
// np293：客体指令补丁（宿主在 map 之后改写客体代码）。
// 代码区在 map_image 里已是只读+执行，这里临时放开该页、写完复原并刷 icache。
// 只在原值与期望值一致时才写：补丁点对不上就不动，避免把别的东西改坏。
// np293: guest instruction patch (the host rewrites guest code after mapping).
// map_image already leaves the code area read-only+executable; temporarily open that page, write,
// restore, and flush the icache. The write happens only when the current word matches the expected
// one, so a mismatched build is left untouched instead of being silently corrupted.
static void guest_patch32(uint32_t *at, uint32_t expect, uint32_t want, const char *why) {
    uintptr_t pg = (uintptr_t)at & ~(uintptr_t)0xfff;
    size_t len = ((uintptr_t)at & 0xfff) + 4;
    if (mprotect((void *)pg, len, PROT_READ | PROT_WRITE | PROT_EXEC)) { LOG("【红】补丁 %s：放开写权限失败", why); return; }
    uint32_t cur = *(volatile uint32_t *)at;
    if (cur != expect) {
        LOG("【红】补丁 %s：原值 %#x ≠ 期望 %#x，不改（构建号不符？）", why, cur, expect);
    } else {
        *(volatile uint32_t *)at = want;
        __builtin___clear_cache((char *)at, (char *)at + 4);
        LOG("补丁 %s：%#x → %#x", why, expect, want);
    }
    if (mprotect((void *)pg, len, PROT_READ | PROT_EXEC)) LOG("【红】补丁 %s：复原只读失败", why);
}

// np294：改跳目标用计数包一层，崩时能一句话说清"这条分配路到底走没走"。
// np294: the redirect target is wrapped with a counter so a crash dump can tell at a glance whether
// this allocation path is actually taken.
static _Atomic unsigned long g_gmalloc_n, g_gmalloc_fail;
static void *guest_malloc(size_t n) {
    void *p = malloc(n);
    atomic_fetch_add(&g_gmalloc_n, 1);
    if (!p && n) atomic_fetch_add(&g_gmalloc_fail, 1);
    return p;
}
// np320：给 HLE 用的宿主堆入口（与 guest_malloc 共用计数）。
// 用途：客体某些 StandardAllocator 实例从未初始化，客体实现在这种状态下只会返回 0；
// 这些分配改由宿主堆服务（与 np293 想用改码达到的效果一致，但不需要改客体代码）。
void *dm_guest_malloc(size_t n, size_t align) {
    if (!align || align > 0x1000) align = 16;
    if (align <= 16) return guest_malloc(n);
    void *raw = guest_malloc(n + align);
    if (!raw) return NULL;
    return (void *)(((uintptr_t)raw + align - 1) & ~(uintptr_t)(align - 1));
}

unsigned long dm_guest_malloc_stat(unsigned long *fail) {
    if (fail) *fail = atomic_load(&g_gmalloc_fail);
    return atomic_load(&g_gmalloc_n);
}

// np294：把客体某个函数入口改成"跳到宿主函数"。
// 宿主 .so 与客体映像相距远超 `b` 的 ±128MB 射程，所以入口前 16 字节改写成
//   ldr x16, [pc,#8] ; br x16 ; <8 字节目标地址>
// 正好覆盖 4 条指令（被覆盖的都是函数序言，尾跳之后本来也不会执行）。
// np294: redirect a guest function entry to a host function.
// The host .so is far beyond `b`'s +-128MB reach from the guest image, so the first 16 bytes become
//   ldr x16, [pc,#8] ; br x16 ; <8-byte target address>
// which covers exactly 4 instructions (all prologue, dead once we tail-jump away).
static void guest_redirect(uint8_t *at, void *fn, uint32_t expect0, const char *why) {
    uintptr_t pg = (uintptr_t)at & ~(uintptr_t)0xfff;
    size_t len = ((uintptr_t)at & 0xfff) + 16;
    if (*(volatile uint32_t *)at != expect0) {
        LOG("【红】改跳 %s：原值 %#x ≠ 期望 %#x，不改（构建号不符？）", why, *(volatile uint32_t *)at, expect0);
        return;
    }
    if (mprotect((void *)pg, len, PROT_READ | PROT_WRITE | PROT_EXEC)) { LOG("【红】改跳 %s：放开写权限失败", why); return; }
    uint32_t w[2] = { 0x58000050u /* ldr x16,[pc,#8] */, 0xD61F0200u /* br x16 */ };
    memcpy(at, w, 8);
    uint64_t t = (uint64_t)(uintptr_t)fn;
    memcpy(at + 8, &t, 8);
    __builtin___clear_cache((char *)at, (char *)at + 16);
    if (mprotect((void *)pg, len, PROT_READ | PROT_EXEC)) LOG("【红】改跳 %s：复原只读失败", why);
    else LOG("改跳 %s → %p", why, fn);
}

// np386：把上面这个改跳能力开放给宿主其它文件（断言钩子要用）。
// np386: expose the redirect above to other host files (the assertion hook needs it).
void dm_guest_redirect(uint8_t *at, void *fn, uint32_t expect0, const char *why) { guest_redirect(at, fn, expect0, why); }

// np296：AArch64 `b` 指令编码（同一个模块内，±128MB 射程）。
// np379：adrp 编码（rd <- page(addr)，以 insn_at 为基准）
static uint32_t arm64_adrp_rt(int rd, uint64_t addr, uint64_t insn_at) {
    long d = (long)(addr >> 12) - (long)(insn_at >> 12);
    return 0x90000000u | ((uint32_t)(d & 3) << 29) | (((uint32_t)((d >> 2) & 0x7FFFF)) << 5) | (uint32_t)rd;
}
static uint32_t arm64_b(uint64_t from, uint64_t to) {
    int64_t d = (int64_t)to - (int64_t)from;
    return 0x14000000u | ((uint32_t)(d >> 2) & 0x03FFFFFFu);
}

// np296：把 main+0x105f9e4（"就地构造"共享辅助，**730 个调用点**）改成空指针时直接返回。
// 为什么在这里下手：崩点链是 0x45f9c8 → 0x105f9e4 → 0xba5aa4，而 0x105f9e4 有 730 个调用点、
// 帧链在第二帧就断了，逐个定位代价太高；它是这条链上唯一的收口点。
// 语义：this 为空就不构造（原版这里必崩，没有"正常"行为可言），让上层继续走 —— 上层若真的
// 用到了这个空子对象，会在**另一个**位置崩，那正好告诉我们是谁传了空指针。
// np296: make main+0x105f9e4 (the shared "construct in place" helper with **730 call sites**)
// return immediately on a null this. The crash chain is 0x45f9c8 -> 0x105f9e4 -> 0xba5aa4, and with
// 730 callers plus a frame chain that breaks at frame 2, pinning the exact caller is too expensive;
// this is the single choke point on the chain. Semantics: with a null this there is no "correct"
// behaviour to preserve (the original crashes), so let the caller continue; if it really uses the
// null sub-object it will fault **somewhere else**, which is exactly the information we want.
void guest_null_tolerant_ctor(void) {   // np374：改为运行期安装（np297 证明运行期改码取指能跟上）
    // np296：整条"分配→构造"链一次全装空指针保护。
    // 崩点链 0x634fa0/… → 0x3326b8|0x37621c|0x40df2c|0x41c728|0x54f204 → 0x45f9c8 → 0x105f9e4 → 0xba5aa4，
    // 每层都只是把 x0 原样往下传（`mov x19,x0; bl 下一层`），所以 **任一层把空指针挡下来，
    // 整条链就不会崩**。真机上这 6 个入口都只被"先分配再构造"的工厂调用，this 不可能为空；
    // 我们这边为空，说明上游某处与真机不同 —— 先让客体活下去，再看它到底在哪用到了这个空对象。
    // np296: install null guards over the whole allocate-then-construct chain in one shot.
    // Every layer just forwards x0 (`mov x19,x0; bl next`), so **guarding any layer stops the crash**.
    // On the real device these 6 entries are only reached from "allocate then construct" factories and
    // this can never be null; it is null here, so something upstream differs -- keep the guest alive
    // first and then see where it actually uses the null object.
    static const struct { uint64_t off; uint32_t first; int mode; const char *name; } kEnt[] = {
        // np406：**资源管理器 getter**（`ldr x8,[x0,#0x10]; ldr x0,[x8,#0x38]; ret`）。
        // np405 真机：游戏连续跑帧后第一个裸崩点就在它（this=NULL，addr=0x38）。
        // mode 0 的桩形正好合适：cbz x0 → NULL 时直接 ret（x0 已是 0 ⇒ 返回 NULL），
        // 命中时 null_guard_trap() 打调用者链。上游拿到 NULL 会怎么表现，正好是我们想要的信息。
        // np406: the resource-manager getter (ldr x8,[x0,#0x10]; ldr x0,[x8,#0x38]; ret). np405 on device: the
        // first plain crash after the guest runs frames is right here (this=NULL, addr=0x38). Mode 0 fits: cbz x0
        // returns immediately on NULL (x0 is already 0, so it returns NULL), and a hit makes null_guard_trap()
        // print the caller chain. How the caller copes with NULL is exactly the information we want.
        // np407：改 mode 3。np406b 真机：FAULT 时 x0=0x5e9e2fd908 **不是 NULL**，空的是
        // 第一条读出来的 [x0+0x10]（x8），FAULT 在第二条 ldr x0,[x8,#0x38]。
        // np407: switch to mode 3. On device (np406b) x0 was NOT null at the fault; the null value is what the
        // first load read into x8, and the fault is the second load.
        { 0x10831f8, 0xF9400808u, 3, "0x10831f8 资源管理器 getter（np405 裸崩点，两级判空）" },
        { 0x105f9e4, 0xF81E0FF3u, 0, "0x105f9e4 就地构造（np374：改回真守卫，不再无条件 ret）" },
        { 0x45f9c8,  0xF81E0FF3u, 0, "0x45f9c8 类构造器" },
        { 0x3326b8,  0xF81E0FF3u, 0, "0x3326b8 构造包装 A" },
        { 0x37621c,  0xF81E0FF3u, 0, "0x37621c 构造包装 B" },
        { 0x40df2c,  0xF81E0FF3u, 0, "0x40df2c 构造包装 C" },
        { 0x41c728,  0xF81E0FF3u, 0, "0x41c728 构造包装 D" },
        { 0x54f204,  0xA9BE4FF4u, 0, "0x54f204 构造包装 E" },
        // np375：**这条链上唯一没守卫的一环**。0xba5aa4 就是"写 vptr 那一句"所在的构造函数，
        // 真机崩溃点一直是它里面的 `str x8,[x20]`（addr=0）。给它装上守卫，命中时 null_guard_trap()
        // 会打出 np333 调用者链 —— 那就是"谁把这个空对象递给它"的直接答案。
        // np375: the only unguarded link in this chain. 0xba5aa4 is the constructor that stores the vptr and
        // is where the faulting `str x8,[x20]` lives (addr=0). Guarding it makes null_guard_trap() print the
        // np333 caller chain, i.e. who handed in the null object.
        { 0xba5aa4,  0xA9BE4FF4u, 0, "0xba5aa4 就地构造（写 vptr）" },
        // np377：**中段入口**。真机三轮都指向"不是从 0x45f9c8 进来的"：入口守卫一次不命中、
        // 而函数体（0x45f9dc 的 bl）在跑、fp 链还缺了 `str x19,[sp,#-0x20]!` 那次分栈。
        // 把 0x45f9cc（`stp x29,x30,[sp,#0x10]`）也当入口装一个守卫——它一命中，
        // null_guard_trap() 会打出 x0 与调用者链，直接给出"谁跳到中段"。
        // np377: the mid-function entry. Three device runs say the body is entered without passing 0x45f9c8
        // (entry guard never fires, the body runs, and the frame lacks the stack allocation). Guard 0x45f9cc
        // as well; a hit prints x0 and the caller chain.
        { 0x45f9cc,  0xA9017BFDu, 0, "0x45f9cc 类构造器中段" },
        // np378：**两个无条件探针**（mode 2）。前面 11 个守卫零命中，而这段代码确实在跑，
        // 最自洽的解释是"有人直接跳进函数体更深处"。把探针装在 `mov x19,x0` 与 `str wzr,[x19,#0x60]`
        // 上：命中就打一行（宿主 0xB1xx 回调记录后继续，原指令照跑，语义不变），
        // 谁命中就说明执行流真的到过那里、以及当时的 x0/lr 是什么。
        // np378: two unconditional probes (mode 2). Eleven guards never fired while this code does run, which
        // fits a direct jump into the function body. Probes on `mov x19,x0` and `str wzr,[x19,#0x60]`: a hit
        // prints a line (the 0xB1xx handler logs and continues at the original instruction, semantics unchanged).
        { 0x45f9d4,  0xAA0003F3u, 2, "探针 0x45f9d4 mov x19,x0" },
        { 0x45f9dc,  0xB900627Fu, 2, "探针 0x45f9dc str wzr,[x19,#0x60]" },
    };
    module_t *mm = NULL;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { mm = &g_mod[i]; break; }
    if (!mm || mm->size <= 0x2404a80) return;
    uint8_t *cave = mm->base + 0x2404a70;          // 产物尾部 96KB 空穴（实测全 0） | the 96KB hole at the tail of the image (all zeros)
    uintptr_t pg = (uintptr_t)cave & ~(uintptr_t)0xfff;
    if (mprotect((void *)pg, 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC)) { LOG("【红】空穴不可写"); return; }
    for (unsigned i = 0; i < sizeof kEnt / sizeof kEnt[0]; i++) {
        uint8_t *stub = cave + 0x20 * i;
        uint8_t *fn = mm->base + kEnt[i].off;
        if (*(volatile uint32_t *)fn != kEnt[i].first) {
            LOG("【红】空指针保护 %s：原值 %#x ≠ 期望 %#x，跳过", kEnt[i].name, *(volatile uint32_t *)fn, kEnt[i].first);
            continue;
        }
        if (kEnt[i].mode == 2) {   // np379：无条件 brk 探针 + **抓栈顶**（7 字）
            // 为什么加抓栈：np378 证明"执行流是从函数体中段进来的"，但 lr 对跳转进入是陈旧值、
            // fp 链也断，唯一还可能带着"跳转者返回地址"的地方就是 **sp 附近的栈**。
            // 桩形：brk #0xB1xx / adrp x16,槽页 / add x16,#lo / add x17,sp,#0 / stp x17,x30,[x16] / 原第一条 / b 入口+4
            // x17 = sp（AArch64 里 sp 只能经 add 读进通用寄存器），和 x30 一起存进槽位；
            // 宿主崩溃时读槽位 → 顺着那个 sp 打印 16 个字，跳转者的返回地址就在里面。
            // np379: unconditional brk probe that also captures the stack pointer. np378 proved execution enters from
            // the function body, but lr is stale for jumps and the fp chain is broken -- the stack near sp is the only
            // place that can still hold the jumper's return address.
            uint64_t cell = (uint64_t)mm->base + 0x2405a70 + 0x10ull * (uint64_t)i;
            uint32_t pv[7];
            pv[0] = 0xD4200000u | ((0xB100u | (i + 1)) << 5);   // brk #0xB1xx（宿主记录后继续）
            pv[1] = arm64_adrp_rt(16, cell, (uint64_t)stub + 4);
            pv[2] = 0x91000000u | ((uint32_t)(cell & 0xFFF) << 10) | (16u << 5) | 16u;   // add x16,x16,#lo12
            pv[3] = 0x91000000u | (31u << 5) | 17u;                                      // add x17, sp, #0
            pv[4] = 0xA9000000u | (30u << 10) | (16u << 5) | 17u;                        // stp x17,x30,[x16]
            pv[5] = kEnt[i].first;                              // 原第一条
            pv[6] = arm64_b((uint64_t)(stub + 24), (uint64_t)(fn + 4));
            if (*(uint32_t *)stub) { LOG("【红】探针 %s：空穴槽已被占用，跳过", kEnt[i].name); continue; }
            memcpy(stub, pv, sizeof pv);
            __builtin___clear_cache((char *)stub, (char *)stub + 28);
            guest_patch32((uint32_t *)fn, kEnt[i].first, arm64_b((uint64_t)fn, (uint64_t)stub), kEnt[i].name);
            continue;
        }
        if (kEnt[i].mode == 1) {   // np297 对照实验：无条件 ret（不占空穴）
            guest_patch32((uint32_t *)fn, kEnt[i].first, 0xD65F03C0u, kEnt[i].name);
            continue;
        }
        if (kEnt[i].mode == 3) {   // np407：两级 getter 桩（6 条）：
            //   cbz x0,ret        （this 为空 → 返回 0）
            //   原第一条           （x8 = [x0+0x10]）
            //   cbz x8,ret        （内层指针为空 → 返回 0）★ 这就是 np406b 漏掉的那次
            //   b fn+8            （x8 非空 → 回原函数第二条）
            //   ret / ret
            if (*(uint32_t *)stub) { LOG("【红】空指针保护 %s：空穴槽已被占用，跳过", kEnt[i].name); continue; }
            uint32_t gw[6];
            gw[0] = 0xB4000060u;                          // cbz x0, +0x0C → 第 5 条 ret
            gw[1] = kEnt[i].first;                        // 原第一条
            gw[2] = 0u;                                   // 占位，下面重编（cbz x8 → 第 5 条 ret）
            gw[3] = arm64_b((uint64_t)(stub + 12), (uint64_t)(fn + 8));
            gw[4] = 0xD65F03C0u;                          // ret
            gw[5] = 0xD65F03C0u;                          // ret
            // cbz x8 的偏移：从 stub+8 到 stub+16（第 5 条 ret）= +8 字节 = imm19 2
            gw[2] = 0xB4000048u | (2u << 5);              // cbz x8, +8
            memcpy(stub, gw, sizeof gw);
            __builtin___clear_cache((char *)stub, (char *)stub + 24);
            guest_patch32((uint32_t *)fn, kEnt[i].first, arm64_b((uint64_t)fn, (uint64_t)stub), kEnt[i].name);
            continue;
        }
        if (*(uint32_t *)stub) { LOG("【红】空指针保护 %s：空穴槽已被占用，跳过", kEnt[i].name); continue; }
        uint32_t w[4];
        w[0] = 0xB4000060u;                          // cbz x0, +0x0C → 第 4 条 ret
        w[1] = kEnt[i].first;                        // 原第一条（建栈帧）
        w[2] = arm64_b((uint64_t)(stub + 8), (uint64_t)(fn + 4));
        w[3] = 0xD65F03C0u;                          // ret
        memcpy(stub, w, sizeof w);
        __builtin___clear_cache((char *)stub, (char *)stub + 16);
        guest_patch32((uint32_t *)fn, kEnt[i].first, arm64_b((uint64_t)fn, (uint64_t)stub), kEnt[i].name);
        // np363：**装完立刻回读**。真机实测 7 个桩的前三个字都对（cbz / 原指令 / 跳回），
        // 第 4 个字却是 `0xD4360020 + 0x20*i` 这种递增垃圾，而不是设计的 `ret`(0xD65F03C0)。
        // 回读能分辨两件事：装的时候就写错，还是**之后被别人覆盖**。前者是写入路径的问题，
        // 后者说明还有第二处代码在写同一块空穴（那种必须靠"谁写"来定位，不能再猜）。
        // np363: read the stub back right after installing it. On device all seven stubs had their first three
        // words right (cbz / original instruction / branch back) but the fourth was `0xD4360020 + 0x20*i`
        // instead of the intended `ret` (0xD65F03C0). Reading back distinguishes "the write itself is wrong"
        // from "something overwrote it later" -- the latter needs finding the second writer, not more guessing.
        {
            const uint32_t *rb = (const uint32_t *)stub;
            if (rb[0] != 0xB4000060u || rb[1] != kEnt[i].first || rb[3] != 0xD65F03C0u)
                LOG("【红】np363 桩 %s 回读不符：%08x %08x %08x %08x（期望 %08x %08x ? %08x）",
                    kEnt[i].name, rb[0], rb[1], rb[2], rb[3], 0xB4000060u, kEnt[i].first, 0xD65F03C0u);
            else
                LOG("[dm] np363 桩 %s 回读一致（cbz/原指令/b 回/ret 四个字都对）", kEnt[i].name);
        }
    }
    if (mprotect((void *)pg, 0x2000, PROT_READ | PROT_EXEC)) LOG("【红】空穴复原只读失败");
}

static void guest_libc_hook(void) {
    module_t *mm = NULL;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { mm = &g_mod[i]; break; }
    if (!mm || mm->size <= 0x22d7238) return;
    volatile uint64_t *hook = (volatile uint64_t *)(mm->base + 0x22d7230);
    if (*hook) { LOG("libc hook 已有值 %#llx，不动", (unsigned long long)*hook); }
    else {
        *hook = 1;
        LOG("装 libc 初始化 hook（main+0x22d7230 = 1）：客体自己的 malloc/calloc/realloc/free 从此可用");
    }

    // np293：让 0x9f33f0（Allocate(size, alloc, align)）恒走客体自己的标准分配器。
    // 它有且只有三条出口，其中两条都会返回 0：
    //   ① 0x9f3410 cbz x0 → 0x9f3444 → 0x9b27a0（标准分配器）   —— 仅当全局分配器 vptr 为 0 才走
    //   ② 0x9f3414 cbnz x1 → 用调用方传的分配器，它的 Allocate 也可能返回 0
    //   ③ 0x9f3420 cbz x1 → 0x9f3454 `mov x0,xzr; ret`         —— 线程默认分配器为空时**直接返回 0**
    // 而 5711 个调用方**一律不判空**（崩点 main+0xba5ac8 就是 this=0 去写 vptr）。
    // 把 ① 改成无条件 `b 0x9f3444`，三条路就都收敛到客体自己的 malloc（np292 已放开它的门）。
    // 释放侧 0x9f36a0 **不接分配器参数**，而是 `bl 0xa0dfcc` 按指针反查归属分配器 ⇒ 分配/释放仍配对。
    // np300：`0x9f3410` 的补丁已改为离线（vk-spike/s3/fixup.py），运行时不再重复打。
    // 运行时只保留一类补丁：需要**宿主函数地址**的改跳（装载期才知道地址）。

    // np294：客体自己的 malloc 接宿主堆。
    // 实测（np293 真机）：这条路上 StandardAllocator(main+0x22d71e8).Allocate(0x100) 返回
    // **0xffffffff**（失败哨兵），于是 0x9f33f0 把 0xffffffff 当指针返回，调用方不判空
    // 直接 `stp x8,xzr,[x19]` 写它 ⇒ addr=0xffffffff 崩（main+0xa08eb0）。
    // 客体这套 libc 分配器在真机上由 rtld 初始化，我们复刻不完整，短时间补不齐；
    // 直接跳宿主 malloc，分配恒成功。释放侧 0x9f36a0 是"按指针反查归属分配器"（0xa0dfcc），
    // 查不到就跳过 —— 所以最坏是泄漏，不会跨分配器乱 free。
    // np294: point the guest's own malloc at the host heap.
    // Measured on device (np293): on this path StandardAllocator(main+0x22d71e8).Allocate(0x100)
    // returns **0xffffffff** (the failure sentinel), so 0x9f33f0 hands 0xffffffff back as a pointer
    // and the caller `stp x8,xzr,[x19]` writes through it without a null check -> addr=0xffffffff
    // crash (main+0xa08eb0). The guest libc allocator is set up by rtld on the real device and our
    // replica is incomplete, so rather than chase it, jump straight to the host malloc and make
    // allocation always succeed. The free side 0x9f36a0 looks the owning allocator up by pointer
    // (0xa0dfcc) and skips when not found, so the worst case is a leak, never a cross-allocator free.
    guest_redirect(mm->base + 0x9b27a0, (void *)(uintptr_t)&guest_malloc, 0xA9BF7BFDu, "客体 malloc");
    guest_redirect(mm->base + 0x9b2810, (void *)(uintptr_t)&calloc, 0xA9BE4FF4u, "客体 calloc");
    guest_redirect(mm->base + 0x9b2888, (void *)(uintptr_t)&realloc, 0xA9BF7BFDu, "客体 realloc");
}

// np301：装载后**主动清 icache**。
// 现象（真机四次对照）：补丁已写进产物文件、运行时读内存也是补丁后的值（`0x45f9c8 = d65f03c0`），
// 但执行流照样按**旧指令**走（`pc` 落在只有顺序流入才能到达的 `0x45f9dc`，而入口已是 `ret`；
// 全模块扫过 b/b.cond/cbz/tbz，没有任何分支跳进中段）。
// 唯一自洽解释：**icache 里还是上一次运行的旧条目** —— 客体代码每次都 mmap 到同一虚拟地址，
// aarch64 允许 munmap 后重新映射同 VA 时 icache 残留，软件必须自己清。
// 这也解释了"有的补丁生效、有的不生效"：取决于那些 cache line 上次有没有被取过。
// 所以：**装载完成后，对所有离线补丁点所在的 cache line 各刷一次**（只刷补丁点，不刷整段，
// 避免 23MB text 全刷带来的启动开销）。
// np304：改成**每线程**也调一次（原来只在装载线程装载后调一次）。
// 为什么：`__builtin___clear_cache` 在 aarch64 上只对本核发 `ic ivau`，**不广播**；
// 装载补丁发生在装载线程，而真正执行补丁点的可能是另一个核上的客体线程。
// EL0 又没有 `ic ialluis`（EL1 才有）⇒ 跨核只能"谁执行谁自己清"。
// 于是：每个客体线程在**开始跑客体代码之前**清一遍这 8 个点（8 条 cache line，代价可忽略）。
// np304: also flush once per guest thread (previously only once after loading, on the loading thread).
// Why: on aarch64 `__builtin___clear_cache` only issues `ic ivau` on the calling core and does NOT broadcast,
// while the thread that actually executes a patched point may sit on another core; EL0 cannot use `ic ialluis`
// (EL1 only), so cross-core can only be handled by "whoever executes, flushes".
// Hence every guest thread flushes these 8 points right before it starts running guest code (8 cache lines, free).
// np316：把当前线程钉在 CPU0，再清补丁点 icache —— 让"清"和"执行"必然在同一个核上。
// 为什么需要：aarch64 的 `ic ivau` 只影响本核。线程清完 icache 之后若被调度到别的核，
// 那个核上还留着**上一轮进程**的旧代码行（物理页被复用、内容已改，icache 行却还是旧的）
// ⇒ 数据读出来是补丁，取指却走旧指令。这正是本崩点全部观测的唯一自洽解释。
// np316: pin the current thread to CPU0, then flush the patched lines, so "flush" and "execute" are on
// the same core. `ic ivau` is per-core; after a flush the thread may migrate to a core that still holds
// the previous process's line for the same physical page -> data reads the patch, the fetch runs the old code.
void guest_flush_icache(int verbose);   // 前置声明（下面用到）
// np317：运行期把同一份补丁**重写一遍**（值不变），只为触发 mprotect 的页权限往返。
// 依据：np297 的运行期改码生效、而离线写进文件的补丁"内存对、取指不对"（np315 探针桩一次都没触发，
// np316 钉核+清 icache 也没用）⇒ 差别就在那次 mprotect；内核在页表更新时会做本地核的 icache 同步。
// np317: rewrite the same patch bytes at run time (same values) purely to get the mprotect round trip,
// which makes the kernel sync the instruction cache for that page. Evidence: np297's runtime patching worked
// while identical bytes written into the file (and confirmed in memory) are still fetched as the old code,
// even after pinning to one core and clearing the cache (np315/np316).
static __attribute__((unused)) static void dm_repatch_runtime(void) {
    static const uint64_t kEnt[] = {
        0x105f9e4, 0x3326b8, 0x365324, 0x37621c, 0x40df2c, 0x41c728, 0x54f204, 0x45f9c8, 0x9f3410,
    };
    module_t *mm = NULL;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { mm = &g_mod[i]; break; }
    if (!mm) return;
    unsigned done = 0, fail = 0;
    const long pg = 0x4000;
    for (unsigned i = 0; i < sizeof kEnt / sizeof kEnt[0]; i++) {
        uint8_t *a = mm->base + kEnt[i];
        uintptr_t base = (uintptr_t)a & ~(uintptr_t)(pg - 1);
        if (mprotect((void *)base, pg, PROT_READ | PROT_WRITE | PROT_EXEC)) { fail++; continue; }
        uint32_t v = *(volatile uint32_t *)a;
        *(volatile uint32_t *)a = v;                      // 写回同值：只为让数据缓存脏、页表走过一遍
        __builtin___clear_cache((char *)a, (char *)a + 4);
        mprotect((void *)base, pg, PROT_READ | PROT_EXEC);
        done++;
    }
    // 空穴桩整片（0x2404a70 ~ 0x2404c00）一并走一遍
    {
        uint8_t *a = mm->base + 0x2404a70;
        uintptr_t base = (uintptr_t)a & ~(uintptr_t)(pg - 1);
        if (!mprotect((void *)base, pg, PROT_READ | PROT_WRITE | PROT_EXEC)) {
            for (unsigned off = 0; off < 0x180; off += 4) {
                uint32_t v = *(volatile uint32_t *)(a + off);
                *(volatile uint32_t *)(a + off) = v;
            }
            __builtin___clear_cache((char *)a, (char *)a + 0x180);
            mprotect((void *)base, pg, PROT_READ | PROT_EXEC);
        } else fail++;
    }
    LOG("np317 运行期重打补丁：%u 处成功，%u 处 mprotect 失败", done, fail);
}

void dm_pin_and_flush(void) {
    // np365：**守卫记录槽所在页改成可写**。np364 的桩要 `stp x0,x30,[槽]` 把 this 与调用点记下来，
    // 但洞所在段被映射成 RX（真机实测：桩里的 stp 直接 SIGSEGV，addr 正是槽地址）。
    // 桩在 0x2404a70 那一页、槽在 0x2405a70 那一页，隔一页 ⇒ 只放开槽页，桩页仍保持只读。
    // np365: make the guard-cell page writable. The np364 stub stores (this, call site) into a cell, but the
    // cave lives in an RX mapping (on device the stub's stp took SIGSEGV at exactly the cell address).
    // Stubs sit on page 0x2404000 and cells on 0x2405000, so only the cell page is opened; stubs stay RX.
    {
        static int once;
        if (!once) {
            once = 1;
            module_t *mm = NULL;
            for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { mm = &g_mod[i]; break; }
            if (mm) {
                uintptr_t pg = (uintptr_t)(mm->base + 0x2405000);
                int rc = mprotect((void *)pg, 0x1000, PROT_READ | PROT_WRITE);
                LOG("np365 守卫记录槽页（main+0x2405000）mprotect → %s", rc ? "失败" : "RW 成功");
            }
        }
    }
    static int logged;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0, &set);
    // np318 收尾：真机四轮证明"钉核 / 运行期重打"都没能让取指跟上补丁（见 20-分析/真机定性-np315-317.md），
    // 这里恢复成普通清 icache；钉核与重打留作记录，等换手段（下一轮不再默认启用）。
    // np318: four device runs showed that neither pinning nor a runtime re-patch makes the fetch follow the
    // patch, so fall back to a plain cache flush; the experiments stay as documented helpers.
    int rc = sched_setaffinity(0, sizeof set, &set);
    (void)rc;
    guest_flush_icache(0);
    if (!logged++) LOG("np316 钉核 + 清 icache：sched_setaffinity CPU0 → %s", rc == 0 ? "成功" : "失败（继续，只是没钉住）");
}

void guest_flush_icache(int verbose) {
    static const uint64_t kOff[] = {
        0x9f3410,   // np293 0x9f33f0 收敛
        0x105f9e4, 0x45f9c8, 0x3326b8, 0x37621c, 0x40df2c, 0x41c728, 0x54f204,   // np296 空指针保护
    };
    module_t *mm = NULL;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { mm = &g_mod[i]; break; }
    if (!mm) return;
    for (unsigned i = 0; i < sizeof kOff / sizeof kOff[0]; i++) {
        uint8_t *a = mm->base + kOff[i];
        uint8_t *lo = (uint8_t *)((uintptr_t)a & ~(uintptr_t)63);   // cache line 对齐（aarch64 通常 64B）
        __builtin___clear_cache((char *)lo, (char *)lo + 64);
    }
    if (verbose) LOG("已清 %u 个补丁点的 icache（np301/np304）", (unsigned)(sizeof kOff / sizeof kOff[0]));
}

void load_modules(const char *dir, const char **names, int n) {
    // 先全部映射并解析符号表，再统一重定位（跨模块符号要先都在位）
    // DM_BOOTPROF：① 宿主装载细分（毫秒），定 ③ 预重定位 / mmap 的上限
    // Map everything and parse the symbol tables first, then relocate all at once (cross-module symbols must all be in place first)
    // DM_BOOTPROF: breakdown of ① host loading (milliseconds), to set the upper bound for ③ pre-relocation / mmap
    uint64_t t0 = now_ns(), tm = 0, tp = 0, tg = 0, tr = 0, t;
    for (int i = 0; i < n; i++) {
        module_t *m = &g_mod[i];
        strlcpy(m->name, names[i], sizeof m->name);
        read_layout(m, dir);
    }
    create_keys();
    for (int i = 0; i < n; i++) {
        module_t *m = &g_mod[i];
        t = now_ns(); map_image(m, dir); tm += now_ns() - t;   // 返回时代码区已是只读+执行，此后装载器只写 data | On return the code area is already read-only+execute; from here on the loader only writes data
        if (!i) tsd_check();
        t = now_ns(); parse_dynamic(m); tp += now_ns() - t;
        t = now_ns(); register_gates(m, dir); tg += now_ns() - t;
        LOG("%s: base %p size %#zx", names[i], m->base, m->size);
        g_nmod++;
    }
    force_init();
    t = now_ns(); for (int i = 0; i < g_nmod; i++) relocate(&g_mod[i]); tr = now_ns() - t;
    force_check();
    // np300：空指针保护已改为**离线**写进产物（vk-spike/s3/fixup.py），运行时不再重复打。
    guest_libc_hook();   // np292：扮演 rtld 装 libc 初始化 hook（必须在跑各模块 inits 之前） | np292: play rtld installing the libc init hook (must be before running module inits)
    { void hlecount_seal(void); hlecount_seal(); }   // DM_HLECOUNT：重定位完成，计数桩页封成只读可执行 | DM_HLECOUNT: relocation done, seal the counting stub pages as read-only executable
    if (getenv("DM_SYMCHECK")) LOG("DM_SYMCHECK：装载期共核 %lu 次符号查询，哈希与线性不一致 %lu", g_symchk_n, g_symchk_bad);
    guest_flush_icache(1);   // np301/np304：离线补丁写进产物后，装载完必须清 icache（否则可能执行旧指令）
    if (g_flag.bootprof) LOG("① 细分：映射（读 + 拷 + 封页）%.1f ms、解析动态段 %.1f ms、门注册 %.1f ms、重定位 + 强制接管 %.1f ms，合计 %.1f ms",
                             tm / 1e6, tp / 1e6, tg / 1e6, tr / 1e6, (now_ns() - t0) / 1e6);
}

// 扮演 rtld 的 InitModules（rtld+0x830）：按装载链表正序调各模块 DT_INIT。
// 时机由 nn::init::Start 决定：nninitStartup 与 _init_libc2 之后、nnMain 之前（Start 的第 4 个参数回调）。
// Plays the role of rtld's InitModules (rtld+0x830): call each module's DT_INIT in forward order of the load list.
// The timing is decided by nn::init::Start: after nninitStartup and _init_libc2, before nnMain (the 4th argument callback of Start).
// np319：只读探测 —— 判断 this=0 是"客体 mem 没初始化"还是"线程分配器为空"。
// 纯读三处数据，不改任何代码，所以不受"改码不生效"的限制。
// np324：**直接把 GOT 槽写成宿主 handler**。
// 依据：客体 malloc(0x9b27a0) 里是 `bl 0x1632580`（PLT 桩）→ 桩里 `ldr x17,[GOT]` → br。
// 只要把那个 GOT 槽改成我们的 handler，调用就必然落到宿主（与"谁解析的"无关）。
// 这是纯数据写 ⇒ 不受"改客体代码不生效"的限制。
void *dm_hle_alloc_em_addr(void);   // np324：hle_svc.c 提供
void dm_pin_alloc_got(void) {
    module_t *mm = NULL;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { mm = &g_mod[i]; break; }
    if (!mm) return;
    uint8_t *b = mm->base;
    uint32_t w0 = *(volatile uint32_t *)(b + 0x1632580), w1 = *(volatile uint32_t *)(b + 0x1632584);
    if ((w0 & 0x9f000000) != 0x90000000 || (w1 & 0xffc00000) != 0xf9400000) {
        LOG("np324 GOT 定位失败：桩指令 %08x %08x", w0, w1); return;
    }
    int64_t imm = ((w0 >> 29) & 3) | (((w0 >> 5) & 0x7ffff) << 2);
    if (imm & (1ll << 20)) imm -= 1ll << 21;
    uint64_t page = ((uint64_t)(uintptr_t)(b + 0x1632580) & ~0xfffull) + ((uint64_t)imm << 12);
    uint64_t slot = page + ((w1 >> 10) & 0xfff) * 8;
    uint64_t off = slot - (uint64_t)(uintptr_t)b;
    if (off >= mm->size) { LOG("np324 槽 %#llx 超出模块范围", (unsigned long long)off); return; }
    void *want = dm_hle_alloc_em_addr();
    void *cur = *(void **)(uintptr_t)slot;
    LOG("np324 GOT 槽 main+%#llx：当前 %p，宿主 handler %p → %s", (unsigned long long)off, cur, want,
        cur == want ? "已经是我们的" : "改写");
    if (cur != want) *(volatile void **)(uintptr_t)slot = want;
}

void dm_dump_alloc_state(const char *when) {
    module_t *mm = NULL;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) { mm = &g_mod[i]; break; }
    if (!mm) return;
    uint8_t *b = mm->base;
    uint64_t gp   = *(volatile uint64_t *)(b + 0x226f0a0);   // 全局分配器对象指针
    uint64_t hook = *(volatile uint64_t *)(b + 0x22d7230);   // 客体 malloc 的 libc hook（装载期写成 1）
    uint64_t h1   = *(volatile uint64_t *)(b + 0x22d71e8);
    uint64_t vp = 0;
    int in_mod = 0;
    for (int i = 0; i < g_nmod; i++) {
        uintptr_t lo = (uintptr_t)g_mod[i].base, hi = lo + (uintptr_t)g_mod[i].size;
        if (gp >= lo && gp + 8 <= hi) in_mod = 1;
    }
    if (in_mod) vp = *(volatile uint64_t *)(uintptr_t)gp;    // 同进程，直接解引用（前面的范围检查保证不会野指针）
    // np326：把四个 GOT 槽一起打出来 —— 一眼看出"分配链上到底谁被调用"。
    // AllocateEm/Emm/Free 与 GetTlsValue 的槽（离线从 main 的 GLOB_DAT 查得）。
    uint64_t g1 = *(volatile uint64_t *)(b + 0x2263628);   // AllocateEm
    uint64_t g2 = *(volatile uint64_t *)(b + 0x2263638);   // Free
    uint64_t g3 = *(volatile uint64_t *)(b + 0x2263648);   // AllocateEmm
    uint64_t g4 = *(volatile uint64_t *)(b + 0x22634a8);   // GetTlsValue
    unsigned long fails = 0, gmn = (unsigned long)dm_guest_malloc_stat(&fails);
    LOG("[np319 %s] obj=%#llx vptr=%#llx hook=%#llx [0x22d71e8]=%#llx | GOT Em=%#llx Free=%#llx Emm=%#llx TLS=%#llx | gm=%lu fail=%lu",
        when, (unsigned long long)gp, (unsigned long long)vp,
        (unsigned long long)hook, (unsigned long long)h1,
        (unsigned long long)g1, (unsigned long long)g2, (unsigned long long)g3, (unsigned long long)g4,
        gmn, fails);
}

void dm_pin_alloc_got(void);
unsigned long dm_guest_malloc_stat(unsigned long *fail);
// np331：把** 0 页**映射成可写（哑页）。目的：`this=0` 时 `str wzr,[x19,#0x60]` 不再 fault。
// 为什么这么干：改客体代码在本机不生效（np315–318），外部调试器也不可用（设备 uid=2000，无 gdb/ptrace）。
// 这是唯一不改码、不要权限、能把"空指针写"变无害的手段。
// 注意：mmap_min_addr 通常会拒绝映射 0 —— 成功与否都会打印，失败就说明这条路不通。
// np331: map the zero page writable so a null-this store faults nowhere. Guest code patching does not work
// on this device (np315-318) and no external debugger is available, so this is the only no-code, no-privilege
// way to make the null store harmless. mmap_min_addr usually refuses address 0; the result is logged either way.
void dm_map_null_page(void) {
    static int done;
    if (done++) return;
    void *p = mmap((void *)0, 0x10000, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) {
        LOG("【黄】np331 0 页映射失败（%s）：空指针写仍会崩", strerror(errno));
    } else {
        memset(p, 0, 0x10000);
        LOG("np331 0 页已映射为可写（%p）：空指针写会被吞掉", p);
    }
}

void run_module_inits(void) {
    // DM_GUARDDBG=1：跑各模块构造器前，读一下 sdk 的 locale 守卫字节（sdk+0x78f1e8）。
    // 用途：确认"守卫字节在进函数时已非 0"是装载写脏还是别的构造器写脏（np242 排障）。
    static int guarddbg = -1;
    if (guarddbg < 0) guarddbg = getenv("DM_GUARDDBG") != NULL;
    for (int i = 0; i < g_nmod; i++) {
        module_t *m = &g_mod[i];
        if (guarddbg && m->size > 0x78f1e8)
            LOG("GUARDDBG %s+0x78f1e8 = %u", m->name, (unsigned)*(volatile uint8_t *)(m->base + 0x78f1e8));
        if (!m->init) continue;
        LOG("%s: DT_INIT +%#llx（init_array %llu 项）", m->name, m->init, m->init_arraysz / 8);
        ((void (*)(void))(m->base + m->init))();
    }
}
