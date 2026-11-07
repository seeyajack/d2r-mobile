/* botw_d2r_feature_stubs.c -- D2R-only feature neutralizers for the OHOS build.
 * Reconstructed 2026-11-07 after np806's mirror-prune accident (this file had
 * been referenced by CMakeLists since np565 but was never committed).  Evidence:
 *   - surviving .o with full DWARF (decl_line / types / locals) + disassembly
 *   - compile_commands.json for a byte-for-byte .text comparison
 *   - build logs preserving the source line of the two implicit declarations
 * Behaviour is intentionally "no D2R features": every rift/vault/cloud/textbox
 * entry point is a no-op or a neutral value.  The four POSIX/clock shims are
 * real implementations (Apple APIs that do not exist on OHOS).
 *
 * REQUIRED by dm/harmony/app/entry/src/main/cpp/CMakeLists.txt (:84) and by
 * vk-spike/s2's CMake lists (:84); that CMake excludes dm_rift.c, dm_vault_ui.c,
 * dm_textbox.c, dm_cloud*.c and dm_bright.c (:44), so these stubs satisfy the
 * remaining references.  Keep this file in git.
 */

#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <pthread.h>
#include <string.h>
#include "dm_rift.h"

/* Apple clock_gettime_nsec_np(): 12 (CLOCK_MONOTONIC_RAW on Darwin) maps to */
/* CLOCK_MONOTONIC here; failure returns 0, same as the upstream shim. */
uint64_t clock_gettime_nsec_np(clockid_t clk) {
    struct timespec t;
    if (clock_gettime(clk == 12 ? CLOCK_MONOTONIC : clk, &t) != 0) return 0;
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* Apple pthread_get_stackaddr_np / pthread_get_stacksize_np. */
void *pthread_get_stackaddr_np(pthread_t t) {
    pthread_attr_t a;
    void *addr = NULL;
    size_t sz = 0;
    if (pthread_getattr_np(t, &a) == 0) {
        pthread_attr_getstack(&a, &addr, &sz);
        pthread_attr_destroy(&a);
    }
    return addr;
}
size_t pthread_get_stacksize_np(pthread_t t) {
    pthread_attr_t a;
    void *addr = NULL;
    size_t sz = 0;
    if (pthread_getattr_np(t, &a) == 0) {
        pthread_attr_getstack(&a, &addr, &sz);
        pthread_attr_destroy(&a);
    }
    return sz;
}

/* nvn_entry.S calls this instead of mach_absolute_time (monotonic ns). */
/* Used by the NVN thunk timing path; 0 on clock failure. */
uint64_t nvn_entry_now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) return 0;
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

int dm_rift_cleared_any(void) { return 0; }
void dm_rift_merc_locked_hint(void) {}
int dm_rift_merc_slot_empty(int loc) { (void)loc; return 1; }
int dm_rift_merc_unlocked(void) { return 0; }
void dm_rift_client_pump(uint8_t *main_base) { (void)main_base; }
void dm_rift_gamble_vals(int best, uint32_t *uniq, uint32_t *set) {
    (void)best;
    if (uniq) *uniq = 0;
    if (set) *set = 0;
}
int dm_rift_menu(dm_rift_menu_t *out) { if (out) memset(out, 0, sizeof(*out)); return 0; }
void dm_rift_menu_peek(dm_rift_menu_t *out) { if (out) memset(out, 0, sizeof(*out)); }
void dm_rift_menu_abandon(void) {}
void dm_rift_menu_close(void) {}
void dm_rift_menu_continue(void) {}
void dm_rift_menu_next(void) {}
void dm_rift_menu_start(int tier) { (void)tier; }
int dm_rift_menu_pending(void) { return 0; }
int dm_rift_menu_seq(void) { return 0; }
int dm_rift_swept(uint32_t id) { (void)id; return 0; }
int dm_rift_stats(dm_rift_stats_t *out) { if (out) memset(out, 0, sizeof(*out)); return 0; }
int dm_rift_tier_hp_x10(int tier, int clvl) { (void)tier; (void)clvl; return 0; }
int dm_rift_tier_level(int tier, int clvl) { (void)tier; (void)clvl; return 0; }

void svc_rift_exit(uint64_t *regs) { (void)regs; }
void svc_save_act(uint64_t *regs) { (void)regs; }
void svc_merc_slot(uint64_t *regs) { (void)regs; }
void svc_merc_slot_cl(uint64_t *regs) { (void)regs; }

/* Vault UI: always empty, never blocks input. */
int dm_vault_ui_blocks(int x, int y) { (void)x; (void)y; return 0; }
int dm_vault_ui_drag(int phase, int x, int y) { (void)phase; (void)x; (void)y; return 0; }
uint32_t dm_vault_ui_held(void) { return 0; }
int dm_vault_ui_over(int x, int y) { (void)x; (void)y; return 0; }
void dm_vault_ui_pump(void) {}
int dm_vault_ui_tap(int x, int y) { (void)x; (void)y; return 0; }
int dm_vault_ui_tip_gate(int x, int y) { (void)x; (void)y; return 0; }

/* Text boxes and server ticks: not present in the BotW payload. */
int dm_textbox_ensure_loaded(void) { return 0; }
void dm_textbox_install(uint8_t *main_base) { (void)main_base; }

/* D2R networking / cloud / brightness server ticks. */
void dm_rift_c2s(uint64_t G, const uint8_t *pkt, uint32_t len) { (void)G; (void)pkt; (void)len; }
void dm_rift_tick(uint64_t G, uint64_t player, uint8_t *main_base) { (void)G; (void)player; (void)main_base; }
void dm_bright_server_tick(uint64_t G) { (void)G; }
void dm_vault_on_uimsg(const char *cat, const char *act, const char *arg) { (void)cat; (void)act; (void)arg; }
void dm_cloudmerge_srv_tick(uint8_t *b, uint64_t G, uint64_t player) { (void)b; (void)G; (void)player; }
int dm_cloud_char_conflict_take(char *out, size_t n) { (void)out; (void)n; return 0; }
int dm_cloud_conflict_take(void) { return 0; }
