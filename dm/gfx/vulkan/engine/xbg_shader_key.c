// 着色器键：与 dm/gfx/tools/shader_split.py、以及 nxnative 宿主 keys_hash 同一算法。
// 规则（三处必须逐字一致，否则包名对不上）：
//   ① 代码段长度 ncode = u32(control + 0x704)，且要求 code 非空、ncode ∈ (0, 1<<24)；
//   ② 控制段参与哈希的长度 nctl = u32(control + 0x20)，仅当 0x760 <= nctl < 0x880 时用它，
//      否则用 0x880 —— 这一步把尾部那些随 ASLR 变的指针字段排除在键之外；
//   ③ key = sha256(control[0:nctl] || code[0:ncode]) 的前 8 字节（16 个 hex 字符）。
// 为什么必须带代码段：很多着色器**共用同一份控制段、只有代码不同**（上游工具注释记着某游戏
// 16,600 个阶段着色器只有 3,079 个不同控制段，其中 6,367 例"控制段同、代码不同"）。
// 只哈希控制段会取到别人的包 —— 这正是本移植一直"缺着色器"的根因。
//
// Shader key: the same algorithm as dm/gfx/tools/shader_split.py and nxnative's host keys_hash.
// The rule must match character for character in all three places, or the bundle names will not line up:
//   1) code length ncode = u32(control + 0x704), valid only when code != NULL and 0 < ncode < (1 << 24);
//   2) control length nctl = u32(control + 0x20), used only when 0x760 <= nctl < 0x880, else 0x880 --
//      this is what keeps the trailing ASLR-varying pointer fields out of the key;
//   3) key = first 8 bytes of sha256(control[0:nctl] || code[0:ncode]), as 16 hex characters.
// Why the code area must be included: many shaders share one control section and differ only in code
// (the upstream tool notes 16,600 stage shaders over 3,079 distinct control sections, 6,367 pairs sharing
// a control section with different code). Hashing the control alone picks someone else's bundle -- the
// root cause of the "missing shader" wall this port kept hitting.
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static uint32_t ror(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

typedef struct { uint32_t h[8]; uint64_t bytes; uint8_t buf[64]; size_t used; } sha256_t;

static void sha_init(sha256_t *s) {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, iv, sizeof iv); s->bytes = 0; s->used = 0;
}

static void sha_block(sha256_t *s, const uint8_t *p) {
    uint32_t w[64];
    for (unsigned i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (unsigned i = 16; i < 64; i++)
        w[i] = w[i - 16] + (ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
               (ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10));
    uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3], e = s->h[4], f = s->h[5], g = s->h[6], z = s->h[7];
    for (unsigned i = 0; i < 64; i++) {
        uint32_t t = z + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
        uint32_t u = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        z = g; g = f; f = e; e = d + t; d = c; c = b; b = a; a = t + u;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d;
    s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += z;
}

static void sha_update(sha256_t *s, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    s->bytes += n;
    while (n) {
        size_t take = 64 - s->used; if (take > n) take = n;
        memcpy(s->buf + s->used, p, take); s->used += take; p += take; n -= take;
        if (s->used == 64) { sha_block(s, s->buf); s->used = 0; }
    }
}

static void sha_final(sha256_t *s, uint8_t out[32]) {
    uint64_t bits = s->bytes * 8;
    uint8_t pad = 0x80, zero = 0; sha_update(s, &pad, 1);
    while (s->used != 56) sha_update(s, &zero, 1);
    uint8_t len[8]; for (unsigned i = 0; i < 8; i++) len[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha_update(s, len, 8);
    for (unsigned i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)(s->h[i] >> 24); out[4 * i + 1] = (uint8_t)(s->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(s->h[i] >> 8); out[4 * i + 3] = (uint8_t)s->h[i];
    }
}

static void hex16(const uint8_t md[32], char out[17]) {
    static const char hex[] = "0123456789abcdef";
    for (unsigned i = 0; i < 16; i++) out[i] = hex[(i & 1) ? (md[i / 2] & 15) : (md[i / 2] >> 4)];
    out[16] = 0;
}

// 旧接口：只哈希控制段前 0x880 字节（D2R 的包按这个出，保留以兼容）。
// Legacy entry point: hash the first 0x880 bytes of the control section only (D2R's bundles were built
// this way, so it is kept for compatibility).
int xbg_shader_control_key(const void *data, char out[17]) {
    if (!data || !out) return -1;
    sha256_t s; sha_init(&s); sha_update(&s, data, 0x880);
    uint8_t md[32]; sha_final(&s, md); hex16(md, out);
    return 0;
}

// 正解：控制段按它自己声明的长度 + 代码段。shader_split.py / nxnative keys_hash 同一算法。
// The correct rule: control section up to its own declared length, plus the code area.
int xbg_shader_prog_key(const void *control, const void *code, char out[17]) {
    if (!control || !out) return -1;
    const uint8_t *c = (const uint8_t *)control;
    uint32_t ncode = 0, own = 0;
    memcpy(&ncode, c + 0x704, 4);
    memcpy(&own, c + 0x20, 4);
    uint32_t nctl = (own >= 0x760 && own < 0x880) ? own : 0x880;
    int has_code = code && ncode && ncode < (1u << 24);
    // np535：NVNshaderData.data 是**缓冲地址**，nxnative 侧用 mtl_addr() 翻成宿主缓冲后才哈希；
    // 我们这里直接用裸地址，实测读到的是个前置结构（代码魔数出现在 +0x180）。
    // 稳妥做法：在 data 前 0x400 字节里找代码魔数（图形 0x12345678 / 计算 0x12345679），找到就以它为代码起点。
    // np535: NVNshaderData.data is a buffer address; nxnative resolves it with mtl_addr() before hashing.
    // Used raw we read a leading structure (the code magic sits at +0x180). Look for the magic in the first 0x400 bytes.
    if (has_code) {
        const uint8_t *p = (const uint8_t *)code;
        for (size_t k = 0; k + 4 <= 0x400; k += 4) {
            uint32_t w = 0;
            memcpy(&w, p + k, 4);
            if (w == 0x12345678u || w == 0x12345679u) { code = p + k; break; }
        }
    }
    sha256_t s; sha_init(&s);
    sha_update(&s, control, nctl);
    if (has_code) sha_update(&s, code, ncode);
    uint8_t md[32]; sha_final(&s, md); hex16(md, out);
    return 0;
}
