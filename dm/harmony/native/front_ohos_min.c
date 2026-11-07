/* front_ohos_min.c -- minimal HarmonyOS front end for the S2 build tree.
 *
 * Reconstructed 2026-11-07 (np806: this file was referenced by CMakeLists since
 * np565 but never committed, and was removed by the mirror-prune).  Evidence:
 *   - surviving .cxx/.../front_ohos_min.c.o (DWARF decl_line / types / locals)
 *   - llvm-objdump of that object (every function body)
 *   - compile_commands.json (exact clang command, used for a .text byte check)
 * The full-featured front end is front_ohos.c in the same directory; this one
 * deliberately has no QoS / OpenGTX / display-change listeners / soft keyboard.
 * Line numbers below come straight from DWARF decl_line; keeping them aligned
 * keeps the rebuilt object's line table close to the original one.  The original
 * source text itself is gone -- DWARF only preserves its MD5 -- so this is a
 * behaviour-equivalent reconstruction, verified by recompiling with the exact
 * compile_commands.json command and comparing .text.* sections + relocations.
 *
 * Contract: g_window is set by napi_init.c's XComponent callbacks and read by
 * front_metal_layer()/front_text_input().  All cross-thread state is atomic.
 * front_run() refreshes the cached display Hz and then calls the guest entry.
 * front_game_res(): docked -> 1920x1080, handheld -> 1280x720.
 * front_text_input(): -1 without a window, 0 (= cancel) with one; see front.h.
 *
 * REQUIRED by dm/harmony/app/entry/src/main/cpp/CMakeLists.txt (:83) and by
 * vk-spike/s2's CMake lists (:83 / :49).  Keep this file in git.
 */

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <window_manager/oh_display_manager.h>

/* front.h contracts (the S2 build does not include that header here). */
uint64_t front_game_res(int docked);
int front_text_input(const char *title, const char *sub, const char *guide,
                     const char *initial, int max_chars, char *out, int out_cap);

static void *_Atomic g_window;
static _Atomic int g_refresh_hz;

void dm_front_ohos_set_window(void *window) { atomic_store(&g_window, window); }

/* Game output resolution: (u64)height << 32 | width (see front.h). */
/* The S2 spike uses fixed 1920x1080 / 1280x720 profiles; the aspect-ratio and */
/* DM_GAME_RES handling lives in the full front end. */

uint64_t front_game_res(int docked) {
    return docked ? ((uint64_t)1080 << 32 | 1920)
                  : ((uint64_t)720 << 32 | 1280);
}

/* Display refresh rate: query once and cache it.  Values outside 1..999 Hz are */
/* rejected (bad driver data); callers fall back to 60 Hz.  refresh_update() is */
/* inlined into all three callers below -- the .o has no out-of-line copy. */
/* (u64) casting keeps the shift out of the 32-bit domain. */


static void refresh_update(void) {
    uint32_t hz = 0;
    if (OH_NativeDisplayManager_GetDefaultDisplayRefreshRate(&hz) == 0 &&
        hz && hz <= 999)
        atomic_store(&g_refresh_hz, (int)hz);
}
void *front_metal_layer(void) { return atomic_load(&g_window); }

/* GTX (graphics tuner) hooks: no-ops in the minimal front end. */

void front_gtx_register(int who) { (void)who; }

/* front_display_refresh(): name/n are an optional human-readable label; the */
/* minimal front end just clears it. */
double front_display_refresh(char *name, size_t n) {
    if (name && n) *name = 0;
    if (!atomic_load(&g_refresh_hz)) refresh_update();
    int hz = atomic_load(&g_refresh_hz);
    return (double)(hz > 0 ? hz : 60);
}

/* Empty hooks kept for ABI compatibility with the full front end. */

void front_gtx_frame(void) {}
void front_pin_fast(const char *who) { (void)who; }
void front_pin_check(void) {}

int front_screen_max_fps(void) {
    if (!atomic_load(&g_refresh_hz)) refresh_update();
    int hz = atomic_load(&g_refresh_hz);
    return hz > 0 ? hz : 60;
}

/* napi_init.c calls this for its status string. */
/* No native text input: -1 without a window, otherwise 0 (cancel). */
int front_text_input(const char *title, const char *sub, const char *guide,
                     const char *initial, int max_chars, char *out, int out_cap) {
    (void)title; (void)sub; (void)guide; (void)initial;
    (void)max_chars; (void)out; (void)out_cap;
    return atomic_load(&g_window) ? 0 : -1;
}
void front_run(void (*guest)(void)) {
    if (guest) {
        refresh_update();
        guest();
    }
}
