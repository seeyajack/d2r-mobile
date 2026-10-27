#include "dm_hud.h"
#include "dm.h"
#include <pthread.h>
#include <stdatomic.h>
// 存储与跨线程交接（读客体在 diag.c 的拍入口里做）：前端 / Mac 触点线程请求档位，拍线程取走并发布 HUD 数字
// Storage and cross-thread handoff (reading the guest is done in the tick entry in diag.c): the front end / Mac touch point thread requests a level, and the tick thread takes it and publishes the HUD numbers
// np382：这一层是「D2R 的 HUD 数字」（等级 / 经验 / 腰带 / 回城），**在模块入口自闸**：非 D2R 时
// 请求一律忽略、读数一律全 0（valid=0）。这样前端无论从哪条路径来问，都拿不到 D2R 的语义。
// np382: this layer holds D2R's HUD numbers (level / experience / belt / town portal) and gates ITSELF at the module
// entry: for another game every request is ignored and every readout is all-zero (valid=0), so no matter which path
// the front end asks through, it never gets D2R semantics.
static _Atomic int g_req;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static dm_hud_info g_info;
void dm_hud_request_players(int n) { if (!dm_game_is_d2r()) return; atomic_store(&g_req, n < 1 ? 1 : n > 8 ? 8 : n); }
int dm_hud_requested_players(void) { return atomic_load(&g_req); }
// 手上物品只有一个真值来源：游戏主线程每帧从客户端玩家的光标物品发布（dm_actions.c dm_held_publish：真是物品、按 id 查得回同一个对象才算），
// 拍线程读的是服务端玩家——两端能不一致（卡死：服务端光标还挂着放进仓库的魔盒，客户端手上早空了，收尾链反复按返回关面板）
// The held item has a single source of truth: the game main thread publishes it every frame from the client player's cursor item (dm_actions.c dm_held_publish: counts only if it is really an item and looking it up by id returns the same object),
// while the tick thread reads the server-side player; the two sides can disagree (the hang: the server cursor still held the Horadric Cube put into the stash, the client hand was long empty, and the wrap-up chain kept pressing Back to close panels)
static _Atomic uint32_t g_held;
void dm_hud_set_held(uint32_t id) { atomic_store(&g_held, id); }
uint32_t dm_hud_held(void) { return atomic_load(&g_held); }
void dm_hud_get(dm_hud_info *out) { if (!dm_game_is_d2r()) { memset(out, 0, sizeof *out); return; } pthread_mutex_lock(&g_mu); *out = g_info; pthread_mutex_unlock(&g_mu); out->holding = (int)atomic_load(&g_held); }
void dm_hud_publish(const dm_hud_info *in) { pthread_mutex_lock(&g_mu); g_info = *in; pthread_mutex_unlock(&g_mu); }
