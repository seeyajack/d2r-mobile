// 游戏身份：宿主里所有「只对某一款游戏成立」的东西，第一步都要问这里（np382）。
//
// 为什么需要这层：D2R 的自动化模块（dm_actions / dm_bossrush / dm_cursor / dm_world …）用
// **硬编码 D2R 地址**（`b + 0x45f9d0`、`b + 0x2d20e4c` 这类）直接调客体函数、读写客体静态数据。
// 换一款游戏，这些地址指向完全无关的代码，参数还取自 D2R 的数据字段（常为 0）——
// 表现就是「this=0、从函数中段进入、崩在跟图形毫无关系的地方」，从崩点反推不出来
// （门票① 就是这么耗掉好几轮的，见 20-分析/门票1根因链条（np319-322）.md）。
//
// 所以判据不是「逐个关模块」，而是「先问这是谁」。判定顺序（先精确后兜底，命中哪条打哪条的日志）：
//   1) DM_GAME 环境变量：d2r / botw / 其他名字（调试、A/B 对照用）
//   2) 编译期 profile：DM_GAME_ID（由 dm/games/<titleid>/game.tsv 生成，见 gen/game_profile.h）
//   3) 布局容纳性：D2R 模块里 main 相对硬编码地址最大到 0x3211700（全树扫描，见下）；
//      镜像比它小 ⇒ 那些地址在本镜像里根本不存在 ⇒ **必然不是 D2R**。
//      （BotW 的 main 是 0x241ca70，比它小 14 MB。）
//   4) 都不成立 ⇒ 未知：D2R 专属模块一律不跑（安全侧）。宁可少个功能，不能拿错地址去跳。
//
// Game identity: everything in the host that only makes sense for one particular game must ask
// here first (np382). The D2R automation modules call guest functions and read guest statics through
// HARD-CODED D2R addresses; in any other image those addresses land in unrelated code, with x0 taken
// from a D2R data field (usually 0) -- which shows up as "this=0, entered mid-function, crashed
// somewhere unrelated" and cannot be reasoned back from the crash site. So the rule is not "switch
// modules off one by one" but "ask who this is". Order: DM_GAME env -> compile-time profile
// (DM_GAME_ID) -> layout fit (the largest main-relative D2R address is 0x3211700, so an image smaller
// than that cannot be D2R at all) -> unknown, which denies D2R-only modules (fail safe).
#include "dm.h"

// profile 只有构建侧生成时才在头文件搜索路径里（gen/）；没有就走兜底判据
// The profile only exists when the build side generates it (gen/); otherwise fall back
#if defined(__has_include)
#  if __has_include("game_profile.h")
#    include "game_profile.h"
#  endif
#endif

// D2R 模块用到的 main 相对硬编码地址上限（全树扫描 `b + 0x…` / `main->base + 0x…` 的最大值：
// dm_vault.c / dm_vault_ui.c / dm_runeword.c 的 0x3211700，其次 dm_world.c 的 0x2f0fee0）。
// 它同时是「这份镜像里有没有可能装着 D2R 的布局」的判据：镜像装不下 ⇒ 一定不是 D2R。
// Upper bound of the main-relative hard-coded addresses used by the D2R modules (maximum over the
// whole tree of `b + 0x...` / `main->base + 0x...`: 0x3211700 in dm_vault.c / dm_vault_ui.c /
// dm_runeword.c, next 0x2f0fee0 in dm_world.c). It doubles as "could this image even contain the D2R
// layout": an image too small for it is definitely not D2R.
#define DM_D2R_LAYOUT_END 0x3211700ull

// D2R 的 program id（本体，Switch 版；hle_misc.c / hle_svc.c 里同一个值）
// D2R's program id (base game, Switch build; the same value as in hle_misc.c / hle_svc.c)
#define DM_D2R_PROGRAM_ID 0x0100726014352000ull

static int d2r_main_size(uint64_t *out) {   // 找 main 模块的映像大小：装不下 D2R 布局就否掉 | main module image size: rules D2R out when too small
    for (int i = 0; i < g_nmod; i++)
        if (!strcmp(g_mod[i].name, "main")) { *out = (uint64_t)g_mod[i].size; return 1; }
    return 0;
}

int dm_game_is_d2r(void) {
    static int v = -1;
    if (v >= 0) return v;

    const char *e = getenv("DM_GAME");
    if (e && *e) {   // ① 环境变量：调试与 A/B 对照用，优先级最高 | env override for experiments and A/B runs
        v = !strcmp(e, "d2r") || !strcmp(e, "d2");
        LOG("游戏身份：%s（来自 DM_GAME）→ D2R 专属模块%s", e, v ? "启用" : "全部不跑");
        return v;
    }
#ifdef DM_GAME_ID
    if (!strcmp(DM_GAME_ID, "0100726014352000")) {   // ② 编译期 profile 说是 D2R | the compile-time profile says D2R
        v = 1;
        LOG("游戏身份：%s %s（编译期 profile）→ D2R 专属模块启用", DM_GAME_ID, DM_GAME_NAME);
        return v;
    }
    v = 0;   // ② profile 明确是别的游戏：不必再猜 | the profile names another game: no need to guess
    LOG("游戏身份：%s %s（编译期 profile）→ D2R 专属模块全部不跑", DM_GAME_ID, DM_GAME_NAME);
    return v;
#else
    uint64_t sz = 0;
    if (!d2r_main_size(&sz)) {   // 模块还没装（早于 load_modules）：不缓存，等装完再判 | modules not loaded yet (before load_modules): do not cache, decide once they are
        static int said;
        if (!said++) LOG("游戏身份：还没有 main 模块（判据待定，装载后再问）");
        return 0;
    }
    if (sz > DM_D2R_LAYOUT_END) {   // ③ 镜像装得下 D2R 的地址布局 | the image is big enough to hold the D2R layout
        v = 1;
        LOG("游戏身份：未知（无 profile / 无 DM_GAME）；main 映像 %#llx 装得下 D2R 布局（>%#llx）→ 按 D2R 处理", (unsigned long long)sz, (unsigned long long)DM_D2R_LAYOUT_END);
        return v;
    }
    v = 0;   // ④ 未知且装不下：D2R 专属模块一律不跑 | unknown and too small: D2R-only modules stay off
    LOG("游戏身份：未知（无 profile / 无 DM_GAME）；main 映像 %#llx 装不下 D2R 布局（≤%#llx）→ D2R 专属模块全部不跑",
        (unsigned long long)sz, (unsigned long long)DM_D2R_LAYOUT_END);
    return v;
#endif
}
