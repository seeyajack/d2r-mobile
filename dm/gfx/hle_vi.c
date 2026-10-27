// nn::vi（显示层）：真机经 vi 服务开显示、建图层、取原生窗口句柄；句柄交给 NVN 的窗口 builder。
// 宿主：返回宿主静态对象；接 Metal 时原生窗口对应前端的 CAMetalLayer（front_metal_layer()）。
// nn::vi (display layer): on a real device, the vi service is used to open the display, create a layer and get the native window handle; the handle is passed to NVN's window builder.
// Host: returns host static objects; when hooking up Metal, the native window corresponds to the front end's CAMetalLayer (front_metal_layer()).
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "../host/hle.h"

static struct { uint32_t magic; } g_display = {0x50534944}, g_layer = {0x5259414c};
static struct { uint32_t magic; void *layer; } g_native = {0x4e49574e, 0};

HLE_FORCE("_ZN2nn2vi10InitializeEv", "显示服务初始化；宿主无 vi 服务", void, h_vi_Init, void) {}
HLE_FORCE("_ZN2nn2vi18OpenDefaultDisplayEPPNS0_7DisplayE", "开默认显示；宿主给静态对象", Result, h_vi_OpenDisp, void **out) {
    *out = &g_display; return RES_OK;
}
HLE_FORCE("_ZN2nn2vi11CreateLayerEPPNS0_5LayerEPNS0_7DisplayE", "建图层；宿主给静态对象", Result, h_vi_CreateLayer, void **out, void *disp) {
    (void)disp; *out = &g_layer; return RES_OK;
}
HLE_FORCE("_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE", "原生窗口句柄，交给 NVN 窗口 builder；宿主给静态对象", Result, h_vi_GetNative, void **out, void *layer) {
    (void)layer; *out = &g_native; return RES_OK;
}

// ── np242：下面两条 BotW 会用到、D2R 不会，原版 hle_vi.c 没有 ──
// 不补的后果（真机实测）：客体自己 subsdk1 里的 vi 实现跑起来，拿着宿主给的"只有 magic 的
// 假 Layer/Display"去调虚函数 ⇒ 空函数指针 ⇒ subsdk1+0xe230 SIGSEGV。
// 所以凡是客体可能调的 vi 入口都要在宿主截住，别让它碰到假对象内部。
HLE_FORCE("_ZN2nn2vi19SetLayerScalingModeEPNS0_5LayerENS0_11ScalingModeE",
          "图层缩放模式；宿主不做缩放（真机全屏）", Result, h_vi_SetScaling, void *layer, int mode) {
    (void)layer; (void)mode; return RES_OK;
}
// 显示 vsync 事件：宿主给一个"永不触发"的静态事件（内容清零）。
// ⚠ 只清 0x20 字节 —— nn::os::SystemEventType 在 64 位下是 4×8 字节，宁可少清也不要越界写调用方的栈。
// ⚠ 若游戏是靠这个事件配帧（而不是靠 nvnWindowAcquireTexture 阻塞），它会一直等 ⇒ 表现是"卡住不崩"；
//   真机上若卡在这里，就改成每帧由前端 Set 一次。
// np403：**必须真的初始化**这个 SystemEvent。原来只 `memset(ev,0,0x20)` 就返回成功 —— 而
// `nn::os::TryWaitSystemEvent` 只看事件 +0x28 的状态字节（1/2 才合法），0 就 `nn::diag` 断言自杀。
// 真机时序（np402 断言时刻栈扫描）：`main+0x9fd500`（游戏的帧节奏循环）→ TryWaitSystemEvent → 断言。
// 现在用 hle_bind_system_event（它会调客体的 nn::os::CreateSystemEvent 正确初始化），并在每次 present 后触发。
// np403: this SystemEvent must really be initialised. It used to be just memset(ev,0,0x20) + success, but
// nn::os::TryWaitSystemEvent only looks at the state byte at +0x28 (1 or 2 are legal) and asserts via nn::diag on
// anything else. Device timeline (stack scan taken at the assertion, np402): main+0x9fd500 (the game's frame pacing
// loop) -> TryWaitSystemEvent -> assertion. Now hle_bind_system_event initialises it properly (it calls the guest's
// nn::os::CreateSystemEvent) and it is signaled after every present.
static void *g_vsync_ev;
// np404：**假 vsync 时钟**。只在 present 时触发会死锁：游戏的帧节奏循环先等 vsync，而 present 在它后面
// —— 真机 np403 就是这个状态（Break 没了、进程活着、日志停在语言码补丁之后不再动）。
// 所以绑定时起一个 60 Hz 的宿主线程，每 16.6 ms 触发一次；present 那条路也保留（早一帧触发无害）。
// np404: a fake vsync clock. Signalling only on present deadlocks: the game's frame pacing loop waits for vsync
// first and present comes after it -- exactly the np403 device state (no Break, process alive, the log stops right
// after the language shim). So start a 60 Hz host thread when the event is bound; the present path stays too.
static void *vsync_ticker(void *arg) {
    (void)arg;
    for (;;) { usleep(16600); if (g_vsync_ev) hle_signal_system_event(g_vsync_ev); }
    return NULL;
}
static void vsync_ticker_start(void) {
    static int started;
    if (started) return;
    started = 1;
    pthread_t th;
    if (!pthread_create(&th, NULL, vsync_ticker, NULL)) { pthread_setname_np(th, "dm-vsync"); pthread_detach(th); LOG("vi: 假 vsync 时钟已启动（60 Hz）"); }
    else { started = 0; LOG("【红】假 vsync 时钟起不来"); }
}
HLE_FORCE("_ZN2nn2vi20GetDisplayVsyncEventEPNS_2os15SystemEventTypeEPNS0_7DisplayE",
          "显示 vsync 事件；绑内核事件，每帧 present 后触发", Result, h_vi_GetVsync, void *ev, void *disp) {
    (void)disp;
    if (ev) {
        hle_bind_system_event(ev, 0);
        g_vsync_ev = ev;
        vsync_ticker_start();
        LOG("vi: vsync 事件已绑定（60 Hz 时钟 + 每帧 present 触发）");
    }
    return RES_OK;
}
// 每次 present 后触发 vsync 事件（任何线程可调）| Signal the vsync event after each present (any thread)
void dm_vi_vsync_signal(void) { if (g_vsync_ev) hle_signal_system_event(g_vsync_ev); }
