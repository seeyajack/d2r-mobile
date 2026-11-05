#include "../../nvn/nvn_copy_pitch.h"
#include "../../xtrace.h"
#include <pthread.h>
#include "xbg_game_internal.h"
#include "xbg_cpu_timing.h"
#include "xbg_completion_wait.h"
#ifdef DM_COMPLETION_WAIT_SELFTEST
extern "C" int xbg_completion_wait_selftest(char *report,size_t bytes);
#endif
#include "xbg_descriptor_timing.h"
#include "xbg_draw_timing.h"
#include "xbg_provenance.h"
#include "xbg_pipeline_key.h"
#include "xbg_deathsite.h"
#include "xbg_vulkan.h"

// np597：`set_levels=1 且 nlevels=0` 的视图在真机上会出现（实测 depth 附件 mip=0/0），而
//   `nvnTextureViewSetLevels(n<=0)` 一次都没被调到 ⇒ 这个 0 不是游戏传进来的，是别的路径写进来的。
//   NVN 里 0 层不合法；按"从 base 到最后一层"解释，并只打一次日志（不要每个 pass 都吵）。
//   文件作用域：`view()` 与帧缓冲规划在两处不同的类里，必须共用同一个解释。
// np597: a view with set_levels=1 and nlevels=0 does occur on device (depth attachment mip=0/0), and
//   nvnTextureViewSetLevels(n<=0) is never called, so the 0 is written by some other path. "0 levels" is
//   invalid in NVN; interpret it as "from base to the last level" and log once. File scope on purpose: view()
//   and the framebuffer plan live in two different classes and must share one interpretation.
static inline uint32_t xbg_view_nlevels(const view_t *v,uint32_t levels){
    uint32_t n=v&&v->set_levels?uint32_t(v->nlevels):levels;
    if(!n){uint32_t b=v&&v->set_levels?uint32_t(v->base_level):0u;n=levels>b?levels-b:1u;
        static int once;if(!once++)xbg_logf("NVN view nlevels=0（base=%u levels=%u）⇒ 按“从 base 到最后一层”=%u 处理",b,levels,n);}
    return n;
}
#include "xbg_depth.h"
#include "xbg_sampler.h"
#include "xbg_framebuffer.h"
#include "xbg_clear.h"
#include "xbg_blit.h"
#include "xbg_adreno_policy.h"
#ifdef XBG_HAVE_SPIRV_OPT
#include "spirv-tools/optimizer.hpp"
#endif
#include "xbg_bc_policy.h"
#include "xbg_writeback_policy.h"
#include "xbg_clear_spv.h"
#include "xbg_bright_spv.h"
#include "xbg_spirv_quad.h"
#include "xbg_spirv_warp.h"
#include "xbg_spirv_ldc.h"
#include "xbg_spirv_temps.h"
#include "xbg_spirv_raster.h"
#include "xbg_cam.h"
#include "../../nvn/nvn_cover.h"
extern "C" {
#include "../../tex/nvn_deswizzle.h"
// The HLE's write-protect dirty-page tracker over the guest memory pools (dm/gfx/nvn/nvn_dirty.c). Weak: hosts and tests
// without it simply do not get the cross-frame vertex/index cache.
__attribute__((weak)) int nvn_dirty_tracked(const void *addr,size_t len);
__attribute__((weak)) void nvn_dirty_protect(const void *addr,size_t len);
__attribute__((weak)) uint64_t nvn_dirty_now(void);
__attribute__((weak)) int nvn_dirty_since(const void *addr,size_t len,uint64_t stamp);
__attribute__((weak)) void nvn_dirty_slot_stats(int *live,int *used_slots,uint64_t *adds_total);
// np701：宿主看门狗的图形侧等待点注册（dm/host/crash.c）；弱：不链宿主的图形单测静默跳过。
// np701: the watchdog wait-point registration (dm/host/crash.c); weak: silently skipped without the host.
extern "C" __attribute__((weak)) void crash_set_wait_dump(void (*fn)(void));
// 3D 画面加亮的曝光 k（宿主 dm/host/dm_bright.c）；弱：不链宿主的图形单测不加亮
// Exposure k of the 3D picture brightening (the host's dm/host/dm_bright.c); weak: graphics unit tests without the host do not brighten
__attribute__((weak)) float dm_bright_k(void);
// 越肩 / 第一人称镜头的宿主一侧（dm/host/dm_cam.c）：开关、跟随要的朝向、给输入换算发布的矩阵；弱：不链宿主的图形单测镜头关着
// The host side of the over-the-shoulder / first-person camera (dm/host/dm_cam.c): the switch, the heading the follow wants, the matrices published
// for input conversion, how far the wall behind the character is; weak: graphics unit tests without the host keep the camera off
__attribute__((weak)) int dm_cam_mode(void);
__attribute__((weak)) int dm_cam_style(void);
__attribute__((weak)) int dm_cam_attack_dir(float *wx,float *wz);
__attribute__((weak)) void dm_cam_publish_height(float py);
__attribute__((weak)) int dm_cam_ahead_pick(float rx,float rz,int host_ok,float hx,float hz,float fx,float fz,float *ox,float *oz);
__attribute__((weak)) int dm_cam_want_dir(float *wx,float *wz);
__attribute__((weak)) int dm_cam_clearance(float bx,float bz,float *dist,float *px,float *pz);
__attribute__((weak)) float dm_cam_take_turn(void);
__attribute__((weak)) double dm_cam_turn_age(void);
__attribute__((weak)) float dm_cam_tilt_get(void);
__attribute__((weak)) void dm_cam_tilt_clamp(float lo,float hi);
__attribute__((weak)) void dm_cam_publish(const float vpo[16],const float ivpo[16],const float vpp[16],const float ivpp[16],float px,float pz);
}
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/mman.h>
#include <sys/resource.h>
#include <signal.h>
#include <dirent.h>
#if defined(__ANDROID__)
#include <ucontext.h>
#endif
#include <cerrno>
#include <unistd.h>
#include <dlfcn.h>
#include <map>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <functional>
#include <deque>
#include <condition_variable>
#include <set>

// Flags of the NVN memory pool that holds a guest range (dm/gfx/nvn/hle_nvn.c), or -1. Weak: hosts without it keep every
// storage writeback.
extern "C" __attribute__((weak)) int nvn_pool_flags(const void *addr,size_t len);
// 真全屏界面下移（同 nvn_mtl.m 的 g_ui_rect_fn / ui_mat_patch；鸿蒙平台层登记，其他平台不登记 = 一切照旧）：屏幕比 16:9 窄时宿主把界面按 16:9
// 摆在画面底部（dm/host/dm_uiscreen.c），前端登记界面矩形的纵向位置与高（画面高的比例）；界面绘制（4 顶点、2 个属性、顶点 20 字节）的投影矩阵加纵向偏移、
// 剪裁跟着下移，盖满整个界面矩形的无贴图块（开面板压暗、转场黑幕）视口纵向放大到整个画面。
// True-fullscreen UI shift (as nvn_mtl.m g_ui_rect_fn / ui_mat_patch; registered by the HarmonyOS platform layer, nothing changes where nobody registers): on screens
// narrower than 16:9 the host lays the UI out at 16:9 at the bottom of the picture (dm/host/dm_uiscreen.c); the front end registers the UI rectangle's vertical position
// and height (fractions of the picture height). UI draws (4 vertices, 2 attributes, 20-byte vertices) get a vertical offset in the projection matrix and a shifted
// scissor; the untextured block covering the whole UI rectangle (panel dimming, transition curtain) gets its viewport scaled to the whole picture.
static void (*g_ui_rect_fn)(double *y,double *h);
static double g_ui_moff;static uint64_t g_ui_moff_addr;
extern "C" void xbg_set_ui_rect_fn(void (*fn)(double *y,double *h)){g_ui_rect_fn=fn;}
// np676：帧中自检的**钩子指针**。end_pass 在文件很早的位置、且在匿名命名空间里，而探针实现要等到
//   能用 readback_buffer/Commands/gpu_sync 之后——直接前置声明会把两个名字落进**不同的匿名命名空间**，
//   链接期就是 undefined symbol（实测踩过）。用全局函数指针绕开作用域问题。
// np676: hook pointer for the mid-frame probe. end_pass sits early in the file inside the unnamed namespace while
//   the probe needs readback_buffer/Commands/gpu_sync; a plain forward declaration puts the two names in
//   *different* unnamed namespaces and the linker reports an undefined symbol. A global function pointer avoids it.
static void (*g_winpass_hook)() = nullptr;
// np693: set by the host when VK_EXT_depth_clip_control was enabled at device creation (dm/android/vulkan_present.cpp)
extern int dm_depth_clip_control_on;
extern int dm_depth_clip_enable_on;    // np694: VK_EXT_depth_clip_enable was enabled at device creation
// np682：窗口纹理 serial（pass 内涂蓝判别用）| np682: window texture serial (for the in-pass fill test)
static uint64_t g_winfill_serial=0;
// np684：**窗口缓冲 serial 集合**。双缓冲下"本轮 pass 的目标"和"上次 present 的 serial"天然错开一格，
//   拿单值比较会永远不命中（实测如此）。这里把历史上 present 过的 serial 都记下来，按集合匹配。
// np684: set of window-buffer serials. With double buffering the pass target and the last presented serial are
//   always one apart, so comparing against a single value never matches (measured). Match against the set.
static uint64_t g_win_serials[8];static int g_win_n=0;
static bool win_serial_match(uint64_t s){for(int i=0;i<g_win_n;i++)if(g_win_serials[i]==s)return true;return false;}
namespace {
xbg_game_config cfg{};
std::recursive_mutex mutex;
uint32_t deferred_live_profile=0;
// ---- Asynchronous translation (Claude, C076): the game thread queues submits/presents/resource updates;
// one graphics thread translates NVN -> Vulkan and owns the VkQueue, in order. Calls that need results
// (fence waits, readback, finish, command-memory allocation) drain the queue first. Like a real GPU,
// guest memory is read at execution; the game only reuses it after the fence (waits drain).
// files/gfx-sync-submit containing '1' restores synchronous translation.
bool gfx_async=false;std::atomic<uint64_t> gfx_enqueued{0},gfx_executed{0};std::atomic<int> gfx_presents_queued{0};
std::mutex gfx_job_mutex;std::condition_variable gfx_job_cv,gfx_idle_cv;std::deque<std::function<void()>> gfx_jobs;bool gfx_busy=false,gfx_stop=false;std::thread gfx_worker;std::thread::id gfx_worker_id;
void gfx_worker_loop(){
    gfx_worker_id=std::this_thread::get_id();
    pthread_setname_np(pthread_self(),"dm-gfx");
    // Graphics thread on performance cores (QoS USER_INTERACTIVE, as for the audio thread).
    {void *h=dlopen("libqos.so",RTLD_NOW);auto set=h?reinterpret_cast<int(*)(int)>(dlsym(h,"OH_QoS_SetThreadQoS")):nullptr;int r=set?set(5):-1;xbg_logf("NVN async graphics thread QoS=USER_INTERACTIVE rc=%d",r);}
    for(;;){std::function<void()> job;
        {std::unique_lock<std::mutex> l(gfx_job_mutex);gfx_job_cv.wait(l,[]{return gfx_stop||!gfx_jobs.empty();});if(gfx_jobs.empty())return;job=std::move(gfx_jobs.front());gfx_jobs.pop_front();gfx_busy=true;}
        job();
        {std::lock_guard<std::mutex> l(gfx_job_mutex);gfx_busy=false;}gfx_idle_cv.notify_all();
    }
}
void gfx_enqueue(std::function<void()> job){
    if(!gfx_async || std::this_thread::get_id()==gfx_worker_id){job();return;}
    {std::lock_guard<std::mutex> l(gfx_job_mutex);gfx_jobs.push_back(std::move(job));}gfx_job_cv.notify_one();
}
std::atomic<uint64_t> gfx_drain_calls{0},gfx_drain_ns{0},gfx_pace_ns{0},gfx_wait_calls{0};
void gfx_drain(){
    if(!gfx_async || std::this_thread::get_id()==gfx_worker_id)return;
    uint64_t t0=uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    {std::unique_lock<std::mutex> l(gfx_job_mutex);gfx_idle_cv.wait(l,[]{return gfx_jobs.empty() && !gfx_busy;});}
    ++gfx_drain_calls;gfx_drain_ns+=uint64_t(std::chrono::steady_clock::now().time_since_epoch().count())-t0;
}
void gfx_async_stop(){if(!gfx_worker.joinable())return;{std::lock_guard<std::mutex> l(gfx_job_mutex);gfx_stop=true;}gfx_job_cv.notify_all();gfx_worker.join();gfx_stop=false;gfx_async=false;}
char error_text[1024]{};
bool initialized=false,failed=false;
VkBool32 sampler_mirror_clamp_enabled=VK_FALSE;
// np602：VK_EXT_custom_border_color 的可用性（懒探测一次）。NVN 的采样器边框色是任意 RGBA，Vulkan 的定点色
// 只有黑/白/透明三种 ⇒ 其余颜色（1.6.0 真机撞到 0.5,0.5,0.5）必须靠这个扩展。
// np602: availability of VK_EXT_custom_border_color (probed once, lazily). NVN border colours are arbitrary
// RGBA while Vulkan's fixed colours are only black/white/transparent, so anything else (the 1.6.0 payload uses
// mid grey 0.5,0.5,0.5) needs this extension.
VkBool32 custom_border_color_enabled=VK_FALSE, custom_border_without_format=VK_FALSE;
// np610：**每次提交的极小摘要**（环形 16 份）。GPU device lost 只会告诉我们"这条队列提交被判死"，
// 不会说是哪一帧、哪一帧里有没有 volume 切片当附件。所以每次提交记下 pass 数 / 含 volume 附件的 pass 数 /
// draw / dispatch 数，一旦 vkQueueSubmit 非成功就把最近 16 份打出来 —— 这能把"是不是 np599 新开的那条路"变成读数。
// np610: a tiny per-submission digest kept in a 16-entry ring. A device loss only says "this queue submit was
// rejected", never which frame, nor whether that frame had a volume slice as a render target. Keeping pass counts,
// volume-attachment pass counts, draws and dispatches per submit turns "is it the np599 path?" into a reading.
struct SubDigest{uint64_t serial=0;uint32_t passes=0,volpasses=0,draws=0,dispatches=0,volatt=0;};
SubDigest digest_ring[16];unsigned digest_at=0;SubDigest cur_digest;
// np627：**跨槽引用登记**。retire_slot 等栅栏后销毁本槽登记的资源，但若同一 VkBuffer 还被
// "更晚提交、仍在飞"的槽引用着，销毁就是 GPU 侧用已释放内存 = 页错 = device lost。
// device lost 是间歇性的（110/310/114 s），与负载无关 —— 这类竞态最像。断言只在证据出现时报，
// 不改行为；两个槽的架构下最坏情况就是"另一个槽还在飞"。
// np627: cross-slot reference registry. retire_slot destroys resources registered by the finished slot after
// waiting its fence, but if the same VkBuffer is still referenced by a *later, in-flight* slot the destroy
// hands freed memory to the GPU = page fault = device lost. The loss is intermittent (110/310/114 s) and
// load-independent -- exactly this kind of race. Assertion only logs when the evidence appears; behavior unchanged.
struct BufRef{std::vector<const void*> slots;};   // const Slot* 以 void* 存，避免 Slot 定义在前置位置
std::map<VkBuffer,BufRef> buf_slot_refs;
// np629：**提交二分**（DM_SPLIT_DRAWS=N，0=关）。大提交按"每 N 个 draw 切一段"提交：
// 中间段用独立 fence 立即提交（不等待），最后一段挂槽 fence —— GPU 侧看到多个小提交，
// CPU 端语义不变。切段点 = render pass 边界（draw() 的 draw 计数达到 N 且无 open pass）。
// 判定"驱动在第几段翻脸"：每段提交带段号，device lost 时提交摘要 ring 里最后成功的段号就是答案。
// np629: submission bisection (DM_SPLIT_DRAWS=N, 0=off). Big submissions are split every N draws:
// intermediate segments are submitted immediately with their own fences (no waiting), the last segment
// carries the slot's fence -- GPU sees many small submissions, CPU semantics unchanged. Cut points are
// render-pass boundaries. Which segment the driver rejects: the digest ring's last successful segment.
static bool retry_submit_enabled(){static int v=-1;if(v<0){const char *e=getenv("DM_RETRY_SUBMIT");v=(e&&*e=='0')?0:1;}return v!=0;}   // np631：默认开
static int split_draws_env(){static int v=-1;if(v<0){const char *e=getenv("DM_SPLIT_DRAWS");v=e?atoi(e):0;if(v<0)v=0;
    xbg_logf("二分 DM_SPLIT_DRAWS=%d（0=off）",v);}   // np629: log once -- verifies the env chain end to end
  return v;}
static unsigned split_draw_count=0;   // 本槽本段已录 draw 数
static unsigned split_seg=0;          // 本槽已切出的段号（0 = 第一段）
static VkFence split_fences[64]={};   // 中间段的 fence（环形复用；无等待方——np703 核对）
static unsigned split_fence_at=0;
// np703：连续丢段计数与上限（丢段重建见 submit_split_segment）。任一段提交成功即清零；
// 连续到上限 = 队列真死 ⇒ 证据落盘 + FATAL。DM_SPLIT_DROP_LIMIT，缺省 8，0 = 不限。
// np703: consecutive-drop counter and limit (drop-and-rebuild, see submit_split_segment).
//   Any successful submit clears it; hitting the limit means the queue is dead.
static unsigned split_drops=0;
static unsigned split_drop_limit(){static unsigned v=8;static bool read=false;if(!read){read=true;const char *e=getenv("DM_SPLIT_DROP_LIMIT");if(e)v=(unsigned)atoi(e);}return v;}
std::map<VkImageView,BufRef> img_slot_refs;   // np628：图像描述符同款登记（键 = VkImageView，一个图像可有多个视图）
// np603：每个 (阶段,槽) 的绑定次数（见 OP_BIND_UBO 处理处）。只用于"空槽"这条 FATAL 的定性。
// np603: bind counters per (stage, slot); only used to classify the "empty slot" fatal.
uint64_t bind_hits[6][16];
// Formatless storage access per format needs VK_KHR_format_feature_flags2 (core in Vulkan 1.3). Without it (Vulkan 1.1 drivers
// such as Adreno 730 512.615) the spec's "Formats Without Shader Storage Format" rule applies: the shaderStorageImage{Read,Write}
// WithoutFormat features cover exactly these formats, where the format also supports VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT.
bool format_feature_flags2=false;
// NVN DEPTH16 (0x32) is stored as X8_D24 when the driver cannot linearly filter D16_UNORM but can filter X8_D24: the game samples
// shadow maps with linear depth comparison, and Adreno 730 512.615 reports no SAMPLED_IMAGE_FILTER_LINEAR for D16_UNORM (or
// D32_SFLOAT). Transfers convert 16 <-> 24-bit depth with exact rounding; constant depth bias is rescaled to the finer depth unit.
bool depth16_as_d24=false;
uint32_t d16_to_d24(uint32_t k){return uint32_t((uint64_t(k)*2*16777215u+65535u)/(2*65535u));}
uint32_t d24_to_d16(uint32_t v){return uint32_t((uint64_t(v&0xffffffu)*2*65535u+16777215u)/(2*16777215u));}
bool storage_without_format_listed(VkFormat f){
    switch(f){
    case VK_FORMAT_R8G8B8A8_UNORM:case VK_FORMAT_R8G8B8A8_SNORM:case VK_FORMAT_R8G8B8A8_UINT:case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R32_UINT:case VK_FORMAT_R32_SINT:case VK_FORMAT_R32_SFLOAT:case VK_FORMAT_R32G32_UINT:case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32_SFLOAT:case VK_FORMAT_R32G32B32A32_UINT:case VK_FORMAT_R32G32B32A32_SINT:case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UINT:case VK_FORMAT_R16G16B16A16_SINT:case VK_FORMAT_R16G16B16A16_SFLOAT:case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:case VK_FORMAT_R16_SFLOAT:case VK_FORMAT_R16G16B16A16_UNORM:case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_R16G16_UNORM:case VK_FORMAT_R8G8_UNORM:case VK_FORMAT_R16_UNORM:case VK_FORMAT_R8_UNORM:case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16_SNORM:case VK_FORMAT_R8G8_SNORM:case VK_FORMAT_R16_SNORM:case VK_FORMAT_R8_SNORM:case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R8G8_SINT:case VK_FORMAT_R16_SINT:case VK_FORMAT_R8_SINT:case VK_FORMAT_A2B10G10R10_UINT_PACK32:case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R8G8_UINT:case VK_FORMAT_R16_UINT:case VK_FORMAT_R8_UINT:return true;
    default:return false;
    }
}
uint64_t last_attachment_trace=UINT64_MAX;
struct Capture {
    std::string directory;FILE *trace=nullptr;uint64_t frame=0,bytes=0;
    unsigned events=0;bool active=false;
    ~Capture(){if(trace)fclose(trace);}
} capture;
constexpr uint64_t capture_limit=64ull*1024*1024;
void capture_log(const char *format,...){
    if(!capture.trace)return;va_list args;va_start(args,format);vfprintf(capture.trace,format,args);va_end(args);fputc('\n',capture.trace);fflush(capture.trace);
}
bool capture_begin(const char *kind,uint64_t present){
    capture.active=capture.trace && capture.frame==present+1 && capture.events<16;
    if(capture.active){++capture.events;capture_log("EVENT frame=%llu event=%u kind=%s",(unsigned long long)capture.frame,capture.events,kind);}
    return capture.active;
}
void capture_data(const char *label,const void *data,size_t bytes){
    if(!capture.active || !bytes)return;
    if(bytes>capture_limit-capture.bytes){capture_log("SKIP label=%s bytes=%zu reason=64MiB-budget",label,bytes);return;}
    char leaf[256];snprintf(leaf,sizeof(leaf),"f%llu-e%u-%s.bin",(unsigned long long)capture.frame,capture.events,label);
    std::string path=capture.directory+"/"+leaf;FILE *file=fopen(path.c_str(),"wb");
    if(!file){capture_log("IO_ERROR file=%s",leaf);return;}
    size_t written=fwrite(data,1,bytes,file);int closed=fclose(file);capture.bytes+=written;
    capture_log("FILE name=%s bytes=%zu requested=%zu closed=%d",leaf,written,bytes,closed);
}
// Diagnostic reads must remain available after runtime's fatal callback performs
// a nonlocal exit with the backend mutex held. Counters do not own GPU state.
struct Stats {
    std::atomic<uint64_t> submissions{0},draws{0},dispatches{0},presents{0},shader_loads{0};
    void reset(){submissions=0;draws=0;dispatches=0;presents=0;shader_loads=0;}
} stats;

// All access is under the existing recursive gfx mutex. No resource ownership.
XbgProvenance provenance;
FILE *provenance_file=nullptr;
void provenance_close(const char *reason){
    if(!provenance_file){provenance.armed=false;return;}
    FILE *f=provenance_file;provenance_file=nullptr;provenance.armed=false;
    char tail[128];int n=snprintf(tail,sizeof(tail),"STOP reason=%s events=%u draws=%u\n",reason,provenance.events,provenance.draws);
    bool bad=n<0||size_t(n)>=sizeof(tail);
    if(!bad)bad=fwrite(tail,1,size_t(n),f)!=size_t(n);
    if(fflush(f))bad=true;if(fclose(f))bad=true;
    xbg_logf("NVN PROVENANCE_CLOSE reason=%s io=%s",reason,bad?"FAIL":"OK");
}
void provenance_log(const char *format,...){
    if(!provenance_file || !provenance.window(stats.presents.load()))return;
    char line[2048];va_list a;va_start(a,format);int n=vsnprintf(line,sizeof(line)-1,format,a);va_end(a);
    if(n<0||size_t(n)>=sizeof(line)-1){provenance_close("FORMAT");return;}
    line[n++]='\n';if(!provenance.accept(unsigned(n))){provenance_close("LIMIT");return;}
    if(fwrite(line,1,size_t(n),provenance_file)!=size_t(n)||fflush(provenance_file))provenance_close("IO_FAIL");
}
void provenance_draw(){
    if(!provenance_file || !provenance.window(stats.presents.load()))return;
    if(!provenance.draw()){provenance_close("DRAW_LIMIT");return;}
    provenance_log("DRAW ordinal=%u frame=%llu submit=%llu",provenance.draws,(unsigned long long)provenance.target,(unsigned long long)stats.submissions.load());
}

uint64_t timing_clock(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
XbgCpuTiming timing(timing_clock);
thread_local XbgCpuTiming *timing_current=nullptr;
uint64_t timing_first=0,timing_last=0,timing_start_ns=0;
xbg_game_stats timing_before{};
bool timing_active=false;
// thread_local reads in a dlopen'ed libdm.so go through the dynamic TLS resolver: every CPU_SCOPE, perf_clock() and detail
// root paid one on every call, about 0.6% of the D2R camp's game thread in the resolver alone. timing_any counts the
// TimingRoots that set a timer on any thread; while it is 0 every timing_current is null, so the hot paths skip the
// thread_local. A/B target tlsgate reads it every time (tls_gate false). img_fast: get_image's inline front path (A/B
// target imgfast). Both are refreshed once a frame.
std::atomic<int> timing_any{0};bool tls_gate=true,img_fast=true;
// Color images are created without MUTABLE_FORMAT, which on Adreno keeps UBWC compression for render targets (Redmi K50 Ultra
// camp: GPU 42 -> 36 ms, 21.2 -> 23.8 fps). A view in a different format upgrades the image in place (upgrade_mutable), as
// nvn_mtl.m tex_upgrade_pfv does for Metal's lossless compression. Marker gfx-mutable-images = '1' creates every color
// image mutable, as before.
// Sample BC textures in their own formats where the device can (marker gfx-native-bc = 1 / 0, else DM_GFX_NATIVE_BC, which
// the Android app sets to 1). Otherwise they are decoded on upload to RGBA8 / R16 / RG16 / RGBA16F (bcdec): 4-8x the bytes
// in memory and in every texture fetch. Adreno 730 samples all of BC1-7. Exception kept: the iOS ASTC pack copies ASTC
// blocks into the game's BC arrays, which needs a decoded destination.
bool native_bc=false;uint64_t native_bc_images=0,decoded_bc_images=0;int64_t image_bytes_live=0,bc_image_bytes_live=0,bc_data_bytes_live=0;   // live image memory (driver sizes) and BC payload
bool no_mutable_images=true;uint64_t mutable_upgrades=0;unsigned mutable_test=0; // mutable_test (marker gfx-mutable-test = N): force the upgrade on every Nth color image when first sampled (test)
struct Image;void upgrade_mutable(Image &im);
bool perf_timing=true; // always-on wall-time breakdown for NVN PERF (exclusive deltas)
// Every scope reads the clock twice (~85k reads per D2R camp frame; ~105 ns each through the vDSO on Adreno 730
// phones, ~19% of the render thread when every frame was timed, still ~4% at one frame in 8). Time one frame in
// perf_sample (marker gfx-perf-sample = N; 1 = every frame, 0 = off; else DM_GFX_PERF_SAMPLE, else 32): 32 leaves about
// two timed frames per 60-frame window. Parts and descriptor lines are averaged over the timed frames. A timed frame still
// took ~4 ms more of the render thread in the D2R camp on the Redmi K50 Ultra and so missed its vsync: one visible hitch
// every 32 frames. The Android app turns it off (DM_GFX_PERF_SAMPLE=0); measurement runs set the marker.
unsigned perf_sample=32;bool perf_frame_timed=true;uint64_t perf_timed_frames=0;
XbgCpuTiming::Counter perf_timing_before[XbgCpuTiming::Count];
bool descriptor_detail=false;
XbgDescriptorTiming descriptor_timing;
thread_local XbgCpuTiming::Scope *descriptor_root=nullptr;
struct DescriptorRoot {
    bool on=descriptor_detail || !tls_gate;XbgCpuTiming::Scope *previous=nullptr;
    DescriptorRoot(XbgCpuTiming::Scope *scope){if(on){previous=descriptor_root;descriptor_root=descriptor_detail?scope:nullptr;}}
    ~DescriptorRoot(){if(on)descriptor_root=previous;}
};
#define DESC_SCOPE(kind) XbgDescriptorTiming::Scope descriptor_scope(descriptor_detail?&descriptor_timing:nullptr,(descriptor_detail || !tls_gate)?descriptor_root:nullptr,XbgDescriptorTiming::kind)
#define DESC_CALL(kind, expr) [&](){DESC_SCOPE(kind);return (expr);}()

bool draw_detail=false;
XbgDrawTiming draw_timing;
thread_local XbgCpuTiming::Scope *draw_root=nullptr;
struct DrawRoot {
    bool on=draw_detail || !tls_gate;XbgCpuTiming::Scope *previous=nullptr;
    DrawRoot(XbgCpuTiming::Scope *scope){if(on){previous=draw_root;draw_root=draw_detail?scope:nullptr;}}
    ~DrawRoot(){if(on)draw_root=previous;}
};
#define DRAW_SCOPE(kind) XbgDrawTiming::Scope draw_scope(draw_detail?&draw_timing:nullptr,(draw_detail || !tls_gate)?draw_root:nullptr,XbgDrawTiming::kind)
bool live_profile_used=false;
struct PipelineCounts {uint64_t hits=0,misses=0,evictions=0,bypass=0;} pipeline_counts,pipeline_before;
void report_pipeline_cache();
struct TimingRoot {
    XbgCpuTiming *previous=timing_current;bool counted=false;
    TimingRoot(){timing_current=(timing_active||(perf_timing&&perf_frame_timed))?&timing:nullptr;if(timing_current){++timing_any;counted=true;}}
    ~TimingRoot(){timing_current=previous;if(counted)--timing_any;}
};
inline XbgCpuTiming *timing_now(){return (!tls_gate || timing_any.load(std::memory_order_relaxed))?timing_current:nullptr;}
#define CPU_SCOPE(kind) XbgCpuTiming::Scope cpu_scope(timing_now(),XbgCpuTiming::kind)
// Hot-path sub-timers: free on untimed frames.
inline uint64_t perf_clock(){return timing_now()?timing_clock():0;}
void timing_start(){
    timing_active=true;timing_start_ns=timing_clock();pipeline_before=pipeline_counts;
    timing_before={stats.submissions.load(),stats.draws.load(),stats.dispatches.load(),stats.presents.load(),stats.shader_loads.load()};
}
void descriptor_report();
void draw_report();
void timing_report(const char *reason){
    if(!timing_active)return;
    uint64_t elapsed=timing_clock()-timing_start_ns;timing_active=false;timing_first=0;
    uint64_t measured=0;for(const auto &c:timing.counters)measured+=c.exclusive_ns;
    xbg_logf("NVN CPU_TIMING v=1 reason=%s first=%llu last=%llu frames=%llu wall_ns=%llu measured_ns=%llu uncovered_ns=%llu draws=%llu dispatches=%llu submissions=%llu shaders=%llu",reason,
        (unsigned long long)(timing_before.presents+1),(unsigned long long)stats.presents.load(),(unsigned long long)(stats.presents-timing_before.presents),
        (unsigned long long)elapsed,(unsigned long long)measured,(unsigned long long)(elapsed>=measured?elapsed-measured:0),
        (unsigned long long)(stats.draws-timing_before.draws),(unsigned long long)(stats.dispatches-timing_before.dispatches),
        (unsigned long long)(stats.submissions-timing_before.submissions),(unsigned long long)(stats.shader_loads-timing_before.shader_loads));
    report_pipeline_cache();
    static const char *names[]={"other","draw","dispatch","descriptor","pass","buffer","pipeline","submit","fence","present","cleanup","hash","image","log","program","cpuwrite","snapshot"};
    for(unsigned i=0;i<XbgCpuTiming::Count;i++){const auto &c=timing.counters[i];xbg_logf("NVN CPU_TIMING part=%s calls=%llu exclusive_ns=%llu inclusive_ns=%llu max_ns=%llu",names[i],(unsigned long long)c.calls,(unsigned long long)c.exclusive_ns,(unsigned long long)c.inclusive_ns,(unsigned long long)c.max_ns);}
    descriptor_report();
    draw_report();
}
void descriptor_report(){
    if(!descriptor_detail)return;
    const uint64_t total=timing.counters[XbgCpuTiming::Descriptor].exclusive_ns;
    const uint64_t tagged=descriptor_timing.sum();
    xbg_logf("NVN DESC_TIMING v=2 first=%llu last=%llu descriptor_ns=%llu tagged_ns=%llu remaining_ns=%llu coverage=%s clock_reads=%llu",(unsigned long long)(timing_before.presents+1),(unsigned long long)stats.presents.load(),(unsigned long long)total,(unsigned long long)tagged,(unsigned long long)(tagged<=total?total-tagged:0),tagged<=total && !descriptor_timing.invalid && !descriptor_timing.top?"OK":"INVALID",(unsigned long long)descriptor_timing.reads);
    const char *names[]={"setup","buffer_bind","image_lookup","image_transition","image_view","sampler_bind","element_rest"};
    for(unsigned i=0;i<XbgDescriptorTiming::Count;++i)xbg_logf("NVN DESC_TIMING part=%s calls=%llu exclusive_ns=%llu",names[i],(unsigned long long)descriptor_timing.counters[i].calls,(unsigned long long)descriptor_timing.counters[i].ns);
    xbg_logf("NVN DESC_TIMING part=remaining calls=0 exclusive_ns=%llu",(unsigned long long)(tagged<=total?total-tagged:0));
}
void draw_report(){
    if(!draw_detail)return;
    const uint64_t total=timing.counters[XbgCpuTiming::Draw].exclusive_ns;
    const uint64_t tagged=draw_timing.finish(total);
    const uint64_t remaining=tagged<=total?total-tagged:0;
    xbg_logf("NVN DRAW_TIMING v=1 first=%llu last=%llu draw_ns=%llu tagged_ns=%llu remaining_ns=%llu coverage=%s clock_reads=%llu flags=%u",(unsigned long long)(timing_before.presents+1),(unsigned long long)stats.presents.load(),(unsigned long long)total,(unsigned long long)tagged,(unsigned long long)remaining,draw_timing.flags?"INVALID":"OK",(unsigned long long)draw_timing.reads,draw_timing.flags);
    const char *names[]={"writeback","vertex_prepare"};
    for(unsigned i=0;i<XbgDrawTiming::Count;++i)xbg_logf("NVN DRAW_TIMING part=%s calls=%llu exclusive_ns=%llu",names[i],(unsigned long long)draw_timing.counters[i].calls,(unsigned long long)draw_timing.counters[i].ns);
    xbg_logf("NVN DRAW_TIMING part=remaining calls=0 exclusive_ns=%llu",(unsigned long long)remaining);
    xbg_logf("NVN DRAW_TIMING counters storage_ranges=%llu images_visited=%llu writeback_bytes=%llu index_expand_calls=%llu index_expand_bytes=%llu",(unsigned long long)draw_timing.storage_ranges,(unsigned long long)draw_timing.images_visited,(unsigned long long)draw_timing.writeback_bytes,(unsigned long long)draw_timing.index_expand_calls,(unsigned long long)draw_timing.index_expand_bytes);
}
void timing_boundary(){
    if(timing_active){if(stats.presents>=timing_last)timing_report("frames");else if(timing_clock()-timing_start_ns>=60000000000ull)timing_report("60s-next-present");}
    else if(timing_first && stats.presents+1==timing_first)timing_start();
}
// 最近一个统计窗的每帧 GPU / 翻译 CPU 毫秒（宿主 hle_oe.c 的 GPU 时间 3D 比例控制器、hle_nvn.c 的自适应出图档读）| per-frame GPU / translation CPU ms of the last
std::atomic<double> perf_gpu_ms{-1},perf_cpu_ms{-1};
extern "C" double xbg_recent_gpu_ms(void){return perf_gpu_ms.load();}
extern "C" double xbg_recent_cpu_ms(void){return perf_cpu_ms.load();}
bool sync_mode=false; // diagnostic: marker files/gfx-sync-mode resolves all GPU results at each submit end
std::atomic<uint64_t> completed{0};uint64_t submitted=0; // guest fence values: submitted xbg_submit calls / GPU-complete ones (see Slot::covers)
// 完成通知（鸿蒙缺省开，DM_GFX_COMPLETION_WAIT=0/1 改）：游戏线程等 fence 时，目标批次一完成就从发布点被唤醒返回，不必排在后续提交后面等引擎锁。
// 关着时发布点与原来一样直接写 completed。
// Completion notification (on by default on HarmonyOS; DM_GFX_COMPLETION_WAIT=0/1 overrides): a game thread waiting on a fence is woken at the publish point as soon as its
// batch completes, instead of queueing for the engine lock behind later submits. Off, the publish point writes completed directly, as before.
// ifdef __OHOS__
#ifdef __OHOS__
constexpr bool completion_wait_default=true;
#else
constexpr bool completion_wait_default=false;
#endif
bool completion_wait_enabled=completion_wait_default;
// 栅栏等待是否持引擎锁（旧路径）：鸿蒙缺省不持，标记 gfx-legacy-waitall = 1 退回
// ifdef __OHOS__
#ifdef __OHOS__
bool legacy_waitall=false;
#else
bool legacy_waitall=true;
#endif
xbg_completion::Completion completion_wait(completed);
void publish_completed(uint64_t value){
    // 只在原写回完成点发布；关着时不取 completion 小锁，开关仅初始化时读取。
    // Publish only at the original writeback-complete points; off, the small lock is never taken; the mode is fixed at initialization.
    if(completion_wait_enabled)completion_wait.publish(value);else completed=value;
}
VkCommandPool command_pool{};
VkPipelineCache pipeline_cache{};
VkPhysicalDeviceProperties properties{};
VkPhysicalDeviceMemoryProperties memories{};
uint32_t timestamp_bits=0;
VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
void need(bool condition,const char *text) { if (!condition) xbg_fatalf("%s",text); }
void check(VkResult r,const char *call) { if(r!=VK_SUCCESS) xbg_fatalf("%s failed: %d",call,int(r)); }
#define VK(call) check((call),#call)
// np651：**device lost 不再当场 FATAL**。真 lost 时 vkWaitForFences/present 返回 -4，原来直接
//   xbg_fatalf；而同一份代码有的跑 236 秒、有的 50 秒就死，是随机穿透。这里统一改成"记一行红、
//   按未完成返回"，让上层重试或看门狗接手（看门狗至少会 dump 现场，比硬死有用）。
// np651: do not FATAL on device loss. A real lost makes vkWaitForFences/present return -4 and the code used
//   to xbg_fatalf; the same build sometimes ran 236 s and sometimes died at 50 s, i.e. a random penetration.
//   Log once and return "not complete" instead, so the caller retries or the watchdog handles it.
// np658：lost 时刻记账（lost_not_fatal 调）与冷却剩余毫秒（提交站点在 vkQueueSubmit 前调，
// 返回 >0 就先睡这么多再交——让驱动 reset 恢复窗口安静过去，喂给 render_service 的
// "Submit commandlist fail" 归零，连坐杀不再被触发）。DM_LOST_COOLDOWN_MS 可调。
// np662：默认 2500 → **8000**：np662 实测该轮 17 次 CQ fail 全部落在 arm 后 2.5s 之外
// （3.3~7.4s，比 np657 时代的 0.9~1.7s 长得多）——驱动恢复时长本身有波动，冷却要盖住 p75。
// np658: lost timestamp bookkeeping (from lost_not_fatal) and remaining-cooldown query (submitters call
//   before vkQueueSubmit; >0 means sleep that long first). DM_LOST_COOLDOWN_MS tunable.
// np662: default 2500 -> **8000** -- np662 measured all 17 CQ fails landing 3.3~7.4 s after arm
//   (much longer than np657's 0.9~1.7 s); driver recovery time itself varies, cover its p75.
static std::mutex g_lost_mu;static uint64_t g_lost_ns=0;
static unsigned lost_cooldown_ms_env(){
    static unsigned cached=0;static bool read=false;
    if(!read){read=true;const char *e=getenv("DM_LOST_COOLDOWN_MS");cached=e?(unsigned)atoi(e):8000u;}
    return cached;
}
static void lost_arm(){std::lock_guard<std::mutex> l(g_lost_mu);g_lost_ns=timing_clock();}
static unsigned lost_cooldown_ms(){
    std::lock_guard<std::mutex> l(g_lost_mu);
    if(!g_lost_ns)return 0;
    uint64_t elapsed=(timing_clock()-g_lost_ns)/1000000u;
    unsigned cd=lost_cooldown_ms_env();
    return elapsed>=cd?0u:cd-(unsigned)elapsed;
}
bool lost_not_fatal(VkResult r,const char *what){
    if(r!=VK_ERROR_DEVICE_LOST)return false;
    static int n=0;if(n++<8)xbg_logf("【红】%s 返回 device lost(-4) —— 不再 FATAL（第 %d 次）",what,n);
    // np658：**冷却退避**。np657 复盘：render_service 按"5 分钟窗口内 Submit commandlist fail(13)
    // 次数"记账，攒满就连坐 signal 9（37 次全发生在 reset 后 0.9~1.7s 的恢复窗口内 ⇒ 都是
    // 我们的提交撞在驱动恢复期）。lost 时在这里记账，提交站点 submit 前用 lost_cooldown_ms() 睡掉。
    // np658: cooldown backoff. np657 postmortem: render_service counts "Submit commandlist fail(13)" within a
    //   5-minute window and group-kills the app when it fills up (all 37 hits landed 0.9~1.7 s after a reset,
    //   i.e. our submits slamming into the driver's recovery window). lost_not_fatal arms it; submitters
    //   sleep through the remaining window via lost_cooldown_ms() before submitting.
    lost_arm();
    return true;
}

uint32_t memory_type(uint32_t bits,VkMemoryPropertyFlags flags) {
    for(uint32_t i=0;i<memories.memoryTypeCount;i++)
        if((bits&(1u<<i)) && (memories.memoryTypes[i].propertyFlags&flags)==flags) return i;
    xbg_fatalf("No Vulkan memory type for flags %#x",flags);
}
// ---------------------------------------------------------------------------
// Deferred submission. All GPU work of a guest submission is recorded into one
// command buffer (per frame slot) instead of submit+wait per operation. Every
// operation still starts with a full memory barrier, so GPU-side ordering and
// visibility match the former one-submit-per-operation model. The CPU waits
// only where it needs GPU results (storage writeback, readback, queries).
// Guest memory read by the GPU (uniform/storage/vertex/index data, uploads) is
// snapshotted into host arenas at record time, exactly as before.
// ---------------------------------------------------------------------------
uint32_t find_memory(uint32_t bits,VkMemoryPropertyFlags required,VkMemoryPropertyFlags preferred){
    for(uint32_t i=0;i<memories.memoryTypeCount;i++)
        if((bits&(1u<<i)) && (memories.memoryTypes[i].propertyFlags&(required|preferred))==(required|preferred)) return i;
    return memory_type(bits,required);
}
std::map<VkFormat,VkFormatProperties> format_property_cache;
// Core formats (< 256) answer from a flat table: format_properties is asked for every vertex attribute of every draw.
VkFormatProperties format_flat[256];bool format_flat_known[256];
const VkFormatProperties &format_properties(VkFormat f){
    if(uint32_t(f)<256){if(!format_flat_known[f]){vkGetPhysicalDeviceFormatProperties(cfg.physical,f,&format_flat[f]);format_flat_known[f]=true;}return format_flat[f];}
    auto it=format_property_cache.find(f);if(it!=format_property_cache.end())return it->second;
    VkFormatProperties p{};vkGetPhysicalDeviceFormatProperties(cfg.physical,f,&p);return format_property_cache.emplace(f,p).first->second;
}
struct ArenaChunk{VkBuffer buffer{};VkDeviceMemory memory{};uint8_t *mapped{};VkDeviceSize size=0,used=0;bool coherent=true;};
constexpr VkBufferUsageFlags arena_usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
constexpr VkDeviceSize arena_chunk_bytes=32ull*1024*1024;
constexpr VkDeviceSize arena_flush_bytes=96ull*1024*1024;
VkDeviceSize arena_alignment=256;
struct Buffer {
    // A view into an arena chunk. Arena memory lives until the owning frame slot
    // is retired after its fence, so no per-buffer destruction is needed.
    VkBuffer buffer{}; VkDeviceMemory memory{}; VkDeviceSize offset=0; size_t size{}; void *mapped{}; bool coherent=true;
    void flush(){if(coherent)return;VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};r.memory=memory;r.size=VK_WHOLE_SIZE;VK(vkFlushMappedMemoryRanges(cfg.device,1,&r));}
    void invalidate(){if(coherent)return;VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};r.memory=memory;r.size=VK_WHOLE_SIZE;VK(vkInvalidateMappedMemoryRanges(cfg.device,1,&r));}
};
struct Arena {
    std::vector<ArenaChunk> chunks;size_t current=0,peak_chunks=3;bool cached=false;VkDeviceSize used_total=0,peak=0;
    ArenaChunk make(VkDeviceSize bytes){
        CPU_SCOPE(Buffer);ArenaChunk c;c.size=bytes;
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=bytes;ci.usage=arena_usage;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VK(vkCreateBuffer(cfg.device,&ci,nullptr,&c.buffer));
        VkMemoryRequirements r;vkGetBufferMemoryRequirements(cfg.device,c.buffer,&r);
        uint32_t type=cached?find_memory(r.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT):find_memory(r.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if(cached && !(memories.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT))type=find_memory(r.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        c.coherent=(memories.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)!=0;
        {static int logged[2];if(!logged[cached]++)xbg_logf("NVN arena chunk cached=%d memoryType=%u flags=%#x heap=%u",int(cached),type,unsigned(memories.memoryTypes[type].propertyFlags),memories.memoryTypes[type].heapIndex);}
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=r.size;ai.memoryTypeIndex=type;
        VK(vkAllocateMemory(cfg.device,&ai,nullptr,&c.memory));VK(vkBindBufferMemory(cfg.device,c.buffer,c.memory,0));
        void *p=nullptr;VK(vkMapMemory(cfg.device,c.memory,0,VK_WHOLE_SIZE,0,&p));c.mapped=static_cast<uint8_t*>(p);return c;
    }
    Buffer alloc(size_t n){
        need(n && n<=256ull*1024*1024,"Invalid or excessive buffer size");
        for(size_t i=current;;++i){
            if(i==chunks.size())chunks.push_back(make(std::max<VkDeviceSize>(arena_chunk_bytes,(n+arena_alignment-1)&~(arena_alignment-1))));
            ArenaChunk &c=chunks[i];VkDeviceSize at=(c.used+arena_alignment-1)&~(arena_alignment-1);
            if(at+n>c.size)continue;
            current=i; // leftovers in skipped chunks are abandoned until reset
            c.used=at+n;used_total+=n;peak=std::max(peak,used_total);
            Buffer b;b.buffer=c.buffer;b.memory=c.memory;b.offset=at;b.size=n;b.mapped=c.mapped+at;b.coherent=c.coherent;return b;
        }
    }
    // reset() keeps all memory mapped (a synchronous reader may still copy out
    // of it); trim() releases oversized/extra chunks before the next recording.
    size_t keep_chunks=3;
    // current is the highest chunk index the just-finished recording placed anything into (alloc only ever advances it),
    // so current+1 is exactly that recording's chunk usage. Capture it here — trim() runs after this and needs it to
    // decide retention; a plain chunks.size() would pin the historical peak and never shrink.
    // current 是刚结束的那次录制实际用到的最高块号（分配只会前进），所以 current+1 就是这次录制的块用量。
    // 在 reset() 里记下来是因为 trim() 在它之后才跑、要据此决定保留多少块；直接用 chunks.size() 会钉住历史峰值、再也不缩。
    void reset(){peak_chunks=current+1;for(auto &c:chunks)c.used=0;current=0;used_total=0;}
    void trim(){
        size_t kept=0;std::vector<ArenaChunk> keep;
        for(auto &c:chunks){if(c.size==arena_chunk_bytes && kept<keep_chunks){keep.push_back(c);++kept;}else destroy(c);}
        chunks.swap(keep);current=0;
    }
    static void destroy(ArenaChunk &c){if(c.mapped)vkUnmapMemory(cfg.device,c.memory);if(c.buffer)vkDestroyBuffer(cfg.device,c.buffer,nullptr);if(c.memory)vkFreeMemory(cfg.device,c.memory,nullptr);c=ArenaChunk{};}
    void release(){for(auto &c:chunks)destroy(c);chunks.clear();current=0;used_total=0;}
};
// Deferred destruction until the GPU finished the slot that referenced objects.
struct Retire {
    std::vector<std::function<void()>> handles;std::vector<std::shared_ptr<void>> objects;
    void run(){for(auto &h:handles)h();handles.clear();objects.clear();}
};
// Deferred GPU->CPU results. Storage writebacks and counter reports are
// resolved at the end of each guest submission (when the guest may observe
// completion) or earlier if anything touches the affected guest memory.
struct EpochQuery {const void *slot;VkQueryPool pool;uint32_t index;uint64_t value;bool known;};
// Occlusion counters accumulate from the last reset; the game may never reset, so queries are kept
// incrementally: known prefix and folded prefix sum advance once, consumed entries are dropped (C103).
struct EpochState{std::deque<EpochQuery> q;uint64_t base=0,known_upto=0,summed_upto=0,sum=0;uint64_t size()const{return base+q.size();}
    void advance_known(){while(known_upto<size() && q[known_upto-base].known)++known_upto;}
    uint64_t prefix(uint64_t count){while(summed_upto<count && summed_upto<size()){sum+=q[summed_upto-base].value;++summed_upto;}return sum;}
    void trim(){while(base<summed_upto && !q.empty()){q.pop_front();++base;}}};
uint64_t counter_epoch=0;std::map<uint64_t,EpochState> epochs;
struct PendingReport {uint64_t addr,type,epoch;size_t count;const void *slot;VkQueryPool ts;uint32_t ts_index;uint64_t ts_value;bool ts_known;};
std::vector<PendingReport> pending_reports;
// Command-buffer SignalEvent (4 per frame in D2R) completes when the GPU reaches it. Completing all work at the signal stalls
// the CPU on the GPU at every signal; marker file gfx-defer-events = '1' instead submits the work so far without waiting and
// publishes the value when that submission's slot retires (as the Metal backend writes it from a completion handler).
// '2' does not submit at the signal: the value waits in the slot being recorded and is published when that slot retires,
// so a frame stays one submission and the 2-slot ring does not wrap inside it.
// The game reads event values only through nvnEventGetValue, and the host's CPU event initialize, finalize and signal land a
// pending value first (xbg_game_event_read); a command-buffer wait on an event with a pending value completes the GPU work
// first (main's 07ea121). Without the marker the host may set the default through DM_GFX_DEFER_EVENTS (the Android app sets
// 2); a marker '0' turns deferral off.
struct PendingEvent {uint64_t addr;uint32_t mode,v;const void *slot;};
std::vector<PendingEvent> pending_events,recording_events;   // recording_events: mode 2, slot not yet submitted
int defer_events=0;uint64_t events_deferred=0,event_reads=0;
struct PendingWriteback {uint64_t addr,bytes;Buffer buffer;const void *slot;};
std::vector<PendingWriteback> pending_writebacks;
// Storage writebacks superseded by a newer GPU result (gfx-wb-skip). In the D2R camp five compute outputs, 1.37 MB a frame
// (1.25 MB of it a buffer that changes by a few KB), were written back at every retire although nothing read them from guest
// memory: the next frame's dispatches and draws take them from the pending GPU copy (pending_view), and by the time a slot
// retires the next frame has recorded a newer writeback of the same ranges. Reading them back took ~1 ms a frame on Adreno
// 730, whose host-visible memory reads at 1.3 GB/s. A writeback is dropped when a newer pending writeback covers its range,
// the range lies in a pool the game's CPU never reads (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) and no image aliases it (image
// uploads read guest memory directly). Engine reads still get the newest result: the newer pending writeback until its slot
// retires, guest memory after. Off unless the host asks (DM_GFX_WB_SKIP, which the Android app sets to 1); marker
// gfx-wb-skip = 1 / 0; A/B target wbskip.
bool wb_skip=false;uint64_t wb_skipped=0,wb_skipped_bytes=0;
void cpu_access(uint64_t address,uint64_t bytes,unsigned why=0);   // resolve pending GPU results overlapping a CPU access
void resolve_pending();
void resolve_slot_writebacks(const void *slot);
void resolve_known_reports();
extern bool resolving_pending,read_preflight;
// perf, by caller (0 other, 1 vertex, 2 index, 3 storage, 4 texel, 5 uniform): snap_site_bytes = bytes copied in full,
// snap_site_req = bytes asked for, before the vertex cache, write tracking and pending-result views (their copies are logged
// on their own lines: vcache fills, wtrack copied_kb).
uint64_t snap_site_bytes[6]={},snap_site_req[6]={};unsigned snap_site=0;
std::unordered_set<uint64_t> storage_pages;uint64_t storage_snap_count=0;
namespace wtrack{extern uint64_t copied_bytes,dirty_pages,mismatches,fallbacks;}
extern std::atomic<uint64_t> memo_epoch;extern uint64_t key_diff[14];extern uint64_t memo_bump[5];extern uint64_t xmemo_hits,xmemo_drops;struct PinRef{VkDescriptorSetLayout layout;uint32_t refs;};extern std::unordered_map<VkDescriptorSet,PinRef> memo_pinned[2];
extern uint64_t memo_front_hits;extern uint64_t desc_memo_hits,desc_memo_misses,desc_memo_why[6],desc_ns[6],noload_passes,img_cache_hits,img_cache_misses,noload_why[8],binds_skipped,set_binds,set_bind_offsets,set_binds_offsets_only,offset_changes[10],offset_changed_at[32];
namespace warm{extern uint64_t hits,creates,create_ns,create_max_ns;extern std::atomic<uint64_t> built,failed,build_ns,async_built,async_skips,compiled;extern bool async_on;void flush();void register_module(VkShaderModule m,const std::string &key);
    extern bool keep_rewritten;bool take_rewritten(const std::string &key,std::vector<uint32_t> &out);}
// 长帧日志的读数（nvn_vk.c 把它们接到 hle_nvn.c 的 mtl_create_* / mtl_pkg_stats / mtl_upload_* / mtl_gpu_ms）：自启动起的累计，图形线程写、
// 游戏线程读。只算帧里同步做的：管线现建（预热和后台建的不算）、着色器载入（读文件、SPIR-V 改写、建模块）、贴图整份上传（含解交错与 BC 解码）。
// Readings for the long-frame log (nvn_vk.c wires them to hle_nvn.c's mtl_create_* / mtl_pkg_stats / mtl_upload_* / mtl_gpu_ms): totals since
// start, written on the graphics thread and read on the game thread. Only work done synchronously in a frame counts: pipelines created on the
// spot (not warm-up or background builds), shader loads (file read, SPIR-V rewrites, module creation) and whole texture uploads (with the
// deswizzle and BC decode).
namespace hud{
std::atomic<uint64_t> pipelines{0},pipeline_ns{0},shaders{0},shader_ns{0},uploads{0},reuploads{0},upload_bytes{0},upload_ns{0},frame{0};
// 每帧的 GPU 时间：每个槽的时间戳跨度记到提交它时的 HLE 帧（xbg_game_set_frame）；读的是 8 帧以前的帧，与正在写的不是同一格。
// GPU time per frame: each slot's timestamp span goes to the HLE frame current when it was submitted (xbg_game_set_frame); readers ask for a
// frame 8 frames back, never the entry being written.
struct Gpu{std::atomic<uint64_t> frame{~0ull},ns{0};};Gpu gpu[64];
void gpu_add(uint64_t f,uint64_t ns){Gpu &g=gpu[f%64];if(g.frame.load(std::memory_order_relaxed)!=f){g.ns.store(0,std::memory_order_relaxed);g.frame.store(f,std::memory_order_release);}g.ns.fetch_add(ns,std::memory_order_release);}
}
uint64_t cleanup_ns[5]={},writeback_bytes=0; // perf: cleanup phase split
// Driver/support constant buffers are rebuilt only when their inputs change (iOS: reuse parameter
// blocks for identical inputs). Generations bump on binding ops; slot_serial on every frame-slot switch.
uint64_t tex_gen[6]={},buf_gen[6]={},slot_serial=1;
// Last in-run re-upload of a software-decoded BC image (GPUs without BC). The burst-sized stream arena and the upload caches
// of get_image_slow are kept only while these keep coming; soft_bc_idle_ns without one returns them to the normal footprint.
// 最近一次软解 BC 图（显卡不支持 BC）在局内重传的时刻。突发用的 stream arena 块数与 get_image_slow 的上传缓存只在重传持续时保留；
// 超过 soft_bc_idle_ns 没有重传就退回平常的占用。
uint64_t soft_bc_reupload_ns=0;constexpr uint64_t soft_bc_idle_ns=10ull*1000000000;
struct DriverCache{uint64_t gen=0,slot=0;Buffer buffer;};DriverCache driver_cache[6][2];
struct SupportCache{float w=-1,h=-1;uint64_t slot=0;Buffer buffer;};SupportCache support_cache;
struct SetKeyHash{size_t operator()(const std::vector<uint64_t> &k)const{uint64_t h=1469598103934665603ull;for(uint64_t v:k){h^=v;h*=1099511628211ull;h^=h>>32;}return size_t(h);}};
uint64_t set_cache_hits=0,set_cache_misses=0,uniform_trimmed=0;bool template_enabled=true;uint64_t template_updates=0,plain_updates=0;bool set_cache_enabled=true; // same-process A/B showed no gain on Maleoon
// In-process A/B (iOS method: alternate in the same process, same thermal state). files/gfx-ab-toggle holding
// N (seconds) flips ab_on every N seconds; ab_on=false disables the optimization under test. Perf lines carry ab=.
int ab_period=0;bool ab_on=true;
bool ab_frame_on=true;
// Which optimization gfx-ab-toggle switches off in its off phase (marker gfx-ab-target: wtrack, pipefast, query or timestamps).
char ab_target[16]={0};
inline bool ab_off(const char *name){return ab_period>0 && !ab_frame_on && !strcmp(ab_target,name);}
bool ab_enabled(){if(ab_period<=0)return true;ab_on=((std::chrono::steady_clock::now().time_since_epoch().count()/1000000000ll/ab_period)&1)==0;return ab_on;}
struct Slot {
    bool reset_queued=false,reset_busy=false;VkResult reset_result=VK_SUCCESS;   // poolreset: a reset job was queued for this submission / still runs (under poolreset::mu)
    // xview: this submission read a pending result of the other slot through pending_view (from its readback arena).
    bool xview=false;
    // 提交时的 HLE 帧：退休时这次提交的 GPU 时间记到它名下（hud::gpu_add）。
    // The HLE frame at submission: on retire this submission's GPU time is added to it (hud::gpu_add).
    uint64_t hud_frame=0;
    struct WaitCount{std::atomic<int> n{0};WaitCount()=default;WaitCount(const WaitCount&){}WaitCount &operator=(const WaitCount&){n.store(0);return *this;}} waiters;   // 不持引擎锁等本槽栅栏的线程数（xbg_legacy_wait）；复位栅栏前须等它们离开 | threads waiting on this slot's fence without the engine lock (xbg_legacy_wait); the fence is reset only after they leave
    VkCommandBuffer cmd{};VkCommandPool pool{};VkFence fence{};VkQueryPool timestamps{};bool recording=false,pending=false;uint64_t ops=0,covers=0; // covers: guest fence value complete once this slot retires
    std::vector<VkQueryPool> occlusion_pools,timestamp_pools;uint32_t occlusion_used=0,timestamp_used=0;
    Retire retire;Arena stream,readback;std::vector<VkDescriptorPool> pools;size_t pool_next=0;
    // gfx-xview-guard (recording_slot, readback_buffer): ranges of this slot's readback buffers the other slot read through
    // pending views (viewed), the ranges guarded during this recording, and the copies and fills that write guarded
    // buffers from the GPU in a prologue command buffer submitted ahead of cmd.
    struct XRange{VkBuffer buffer;VkDeviceSize begin,end;};struct ProOp{VkBuffer src,dst;VkDeviceSize src_off,dst_off,size;};
    std::vector<XRange> viewed,guard;std::vector<ProOp> pro_ops;VkCommandBuffer pro{};
    // gfx-upload-prologue：序言命令缓冲里整份上传的图像（from：这次提交开始时它的布局）。
    // gfx-upload-prologue: whole image uploads the prologue command buffer performs (from: the image's layout when this submission starts).
    struct ProImage{VkImage image;VkImageLayout from;VkImageAspectFlags aspects;uint32_t levels,layers;VkBuffer buffer;std::vector<VkBufferImageCopy> copies;};
    std::vector<ProImage> pro_images;
    std::map<std::vector<uint64_t>,VkFramebuffer> framebuffers;
    std::map<std::array<uint64_t,4>,VkBufferView> texel_views; // per-frame texel views (same snapshot -> one view)
    Buffer query_results;size_t query_copied=0; // occlusion results copied by the GPU at the end of the frame
    std::vector<std::pair<uint64_t,uint64_t>> query_refs; // (epoch, absolute index) per used occlusion query, in result order
    std::unordered_map<std::vector<uint64_t>,VkDescriptorSet,SetKeyHash> set_cache; // identical descriptor contents -> one set per frame
    std::unordered_map<VkDescriptorSetLayout,std::vector<VkDescriptorSet>> spare_sets; // batch-allocated, reset with the pools
    struct PassRec {
        uint32_t q0=0,q1=0,w=0,h=0,att=0,fmt=0,draws0=0,draws1=0,x=0,y=0,z=0;uint64_t verts0=0,verts1=0;
        bool compute=false,complete=false,mixed=false,dz=false;std::string first,last;
    };
    bool prof=false;VkQueryPool prof_pool{};uint32_t prof_used=0,prof_slot=0,prof_skipped=0;uint64_t prof_frame=0;
    std::vector<PassRec> prof_recs;int prof_open=-1;
    std::unordered_map<VkDescriptorSetLayout,std::vector<VkDescriptorSet>> used_sets;
    std::vector<struct Descriptors*> desc_used; // pooled per-draw descriptor records, returned after this slot's fence // handed out this frame; recycled into spare_sets after the slot fence (no per-frame pool reset)
};
Slot slots[2];unsigned slot_index=0;
// Pending views of the other slot's results (xviews), and arena reuse while a slot with such views was in flight (races).
// gfx-xview-guard (marker 1 / 0, else DM_GFX_XVIEW_GUARD; A/B target xguard): at such a reuse, a new readback buffer that
// overlaps a range the reader viewed is not written by the CPU; its contents go to a stream buffer and a prologue command
// buffer copies them in after a barrier, so the reader's work is done first (guarded allocations a frame and their KB). The
// buffers stay where they were: moving them to another arena or another half of the chunk cost 1.72-1.85 ms of CPU a frame,
// because about 145 memoized descriptor sets a frame bind read-only storage at a fixed buffer and offset.
uint64_t xviews=0,xview_races=0,xview_guarded=0,xview_guarded_bytes=0;bool xview_guard=false;
// gfx-upload-prologue（标记 1 / 0，否则 DM_GFX_UPLOAD_PROLOGUE；A/B 目标 uppro）：这次提交里还没用过的图像，整份上传改由序言命令缓冲在提交之前做，
// 不再在帧中间结束打开的通道再加两道全屏障（每结束一次，分块 GPU 就要把通道的附件写出再读回）。计数：进序言的、仍在命令缓冲里做的、其中打断了通道的。
// gfx-upload-prologue (marker 1 / 0, else DM_GFX_UPLOAD_PROLOGUE; A/B target uppro): a whole upload of an image not yet used in this
// submission is done by the prologue command buffer ahead of it, instead of ending the open pass in mid-frame with two full barriers
// (each end makes a tiling GPU write the pass's attachments out and read them back). Counted: uploads in the prologue, uploads still in
// the command buffer, and those of them that ended an open pass.
uint64_t uploads_prologue=0,uploads_inpass=0,uploads_broke_pass=0;bool upload_prologue=false;
// gfx-shader-reuse（标记 1 / 0 / 2，否则 DM_GFX_SHADER_REUSE；A/B 目标 shreuse）：着色器载入直接用预热线程改写好的 SPIR-V（quad、warp、
// tidy 之后），不在提交线程上再改写；Redmi K50 Ultra 上改写占载入时间的约 80%，大的片段着色器一个 6-9 ms。2：照样改写并比对，用改写的结果。
// gfx-shader-reuse (marker 1 / 0 / 2, else DM_GFX_SHADER_REUSE; A/B target shreuse): a shader load takes the SPIR-V the warm-up threads
// already rewrote (after quad, warp and tidy) instead of rewriting it again on the submit thread; on the Redmi K50 Ultra the rewrites
// are about 80% of a load, 6-9 ms for a large fragment shader. 2: rewrite as well and compare, using the rewritten result.
int shader_reuse=0;uint64_t shaders_reused=0,shader_reuse_mismatches=0;
// Command pool resets off the render thread. vkResetCommandPool took ~0.4 ms of every frame on the Redmi K50 Ultra's render
// thread. After a slot is submitted, the reset thread waits for its fence and resets its pool; retiring the slot first waits
// for that job, so vkResetFences never overlaps the thread's wait (concurrent waits are allowed), and reusing the slot skips
// its reset. Marker gfx-async-pool-reset = 1 / 0, else DM_GFX_ASYNC_POOL_RESET (the Android app sets 1); A/B target
// poolreset, decided per submission. NVN PERF poolreset: resets queued, retirements that had to wait for one, and how long.
// np701：VkSite 前置声明——poolreset 的 fence 等待（下方）也要挂守卫，而 deathguard 命名空间在
//   pass_end_cause（第 833 行）之后才展开。类型在此给全（成员函数体在命名空间里补定义）。
// np701: forward VkSite -- poolreset's fence wait below needs the guard, but the deathguard namespace
//   opens after pass_end_cause (line 833). The type is complete here; member bodies come later.
namespace deathguard{struct VkSite{int slot;VkSite(unsigned site,uint64_t ctx=0);~VkSite();};}
namespace poolreset {
bool enabled=false,stop=false;std::thread *worker=nullptr;std::mutex mu;std::condition_variable cv;   // never destroyed: a joinable std::thread left at exit would terminate
struct Job{Slot *slot;VkFence fence;VkCommandPool pool;};std::vector<Job> jobs;
uint64_t queued=0,waits=0,wait_ns=0;   // render thread only
void run(){
    for(;;){Job j;{std::unique_lock<std::mutex> l(mu);cv.wait(l,[]{return stop || !jobs.empty();});if(jobs.empty())return;j=jobs.front();jobs.erase(jobs.begin());}
        const uint64_t timeout=std::max<uint64_t>(cfg.fence_timeout_ns,10000000000ull);
        VkResult r;{deathguard::VkSite site(XBG_SITE_FENCE_POOLRESET);r=vkWaitForFences(cfg.device,1,&j.fence,VK_TRUE,timeout);}
        // 栅栏一直不亮而驱动又不报 DEVICE_LOST 时，这里只剩 VK_TIMEOUT；把它说清楚，别让崩在后面的检查里只留下一个 2。
        // When the fence never signals and the driver does not report DEVICE_LOST, all that is left here is VK_TIMEOUT; say so,
        // so the fatal check later does not leave just a 2 in the crash report.
        if(r==VK_TIMEOUT)xbg_logf("NVN command pool reset: the slot's fence did not signal within %llu ms (GPU hang)",(unsigned long long)(timeout/1000000ull));
        if(r==VK_SUCCESS)r=vkResetCommandPool(cfg.device,j.pool,0);
        {std::lock_guard<std::mutex> l(mu);j.slot->reset_result=r;j.slot->reset_busy=false;}cv.notify_all();}
}
void start(){if(enabled && !worker){stop=false;worker=new std::thread(run);}}
void queue(Slot &s){if(!s.pool || !worker || ab_off("poolreset"))return;{std::lock_guard<std::mutex> l(mu);s.reset_busy=true;s.reset_queued=true;jobs.push_back({&s,s.fence,s.pool});}cv.notify_one();++queued;}
// Wait for this slot's job, if one is queued: before its fence is reset and before its pool is reused.
void finish(Slot &s){if(!s.reset_queued)return;std::unique_lock<std::mutex> l(mu);if(s.reset_busy){uint64_t t0=timing_clock();cv.wait(l,[&]{return !s.reset_busy;});wait_ns+=timing_clock()-t0;++waits;}}
void drain(){if(!worker)return;{std::lock_guard<std::mutex> l(mu);stop=true;}cv.notify_all();worker->join();delete worker;worker=nullptr;}
}
// Where the CPU blocks on a busy slot fence, and why slots are submitted (NVN PERF block_ms).
// The outermost site wins, except that a ring wrap in recording_slot is always counted as ring.
enum SyncSite{SITE_OTHER,SITE_RING,SITE_PRESENT,SITE_EVENT,SITE_ARENA,SITE_GPU_SYNC,SITE_COMPLETE,SITE_POLL,SITE_LEGACY_WAIT,SITE_COUNT};
int sync_site=SITE_OTHER;uint64_t block_ns[SITE_COUNT],block_calls[SITE_COUNT],submit_by[SITE_COUNT],legacy_wait_ns=0;
void fold_completion_wait_stats(){
    // 调用者持 engine；ns/calls 在同一 completion 小锁下成对取清，快返路径不写普通统计。
    // Caller holds engine; take/reset ns and calls as a pair under completion's lock, never writing plain counters on a lock-free return.
    const auto pending=completion_wait.take_gpu_waits();
    block_ns[SITE_LEGACY_WAIT]+=pending.ns;block_calls[SITE_LEGACY_WAIT]+=pending.calls;legacy_wait_ns+=pending.ns;
}
struct SiteScope{int old;explicit SiteScope(int site,bool force=false):old(sync_site){if(force || sync_site==SITE_OTHER)sync_site=site;}~SiteScope(){sync_site=old;}};
uint64_t write_seq=1,barrier_seq=0; // image writes vs. last full barrier (render-pass merging)
// Render-pass merging: consecutive draws into the same framebuffer share one
// render pass. Any non-draw operation, a texture read of something written
// since the last barrier, or a layout change ends the open pass first.
struct Image;
void bind_cache_reset();
// Attachment images of a pass (8 color + depth): fixed capacity, so building a Pass per draw does not allocate.
struct ImageList{Image *v[9];uint32_t n=0;void push_back(Image *i){if(n<9)v[n++]=i;}Image **begin(){return v;}Image **end(){return v+n;}Image *const *begin()const{return v;}Image *const *end()const{return v+n;}
    size_t size()const{return n;}bool empty()const{return !n;}Image *operator[](size_t i)const{return v[i];}};
// depth_attached / depth_used (NVN PERF pass_ends depth_idle, NVN GPASS dz=1): a pass that carries a depth attachment no
// draw tests and no clear touches; a tiler loads and stores that attachment for nothing.
struct OpenPass {VkRenderPass pass{};VkFramebuffer framebuffer{};VkCommandBuffer cmd{};ImageList images;bool depth_attached=false,depth_used=false;uint64_t serial0=0;} open_pass;   // serial0: 附件 0 的纹理 serial（np682）| serial0: serial of attachment 0
// Framebuffers of cached render passes, kept across frames (creating and destroying one per pass per frame was ~5% of the
// render thread on Adreno 730). Each attachment Image lists the keys of the framebuffers that use its views; the first of
// them to be destroyed (all GPU work that used it is complete by then) takes the framebuffer along. gfx-no-persistent-fb
// keeps the former per-slot cache.
std::map<std::vector<uint64_t>,VkFramebuffer> persistent_framebuffers;bool persistent_fb_enabled=true;uint64_t fb_hits=0,fb_creates=0;
void forget_framebuffers(Image &im,std::vector<VkFramebuffer> *deferred=nullptr);
// Per-slot command pools reset whole (vkResetCommandPool) instead of resetting the command buffer: the buffer reset was
// 3.6% of the render thread on Adreno 730. gfx-no-slot-pools keeps the shared pool. Readback arenas in host-cached memory
// (gfx-readback-cached = '1'): GPU results read back by the CPU (storage writebacks) come from uncached memory otherwise.
bool slot_pools_enabled=true,readback_cached=false;
// Experiment (marker gfx-no-occlusion = '1'): no occlusion queries at all; counter reports then read 0 samples. Measures
// what the per-draw precise queries cost the GPU (count_samples stays on after the game's first counter reset). A/B
// target occ does the same in the off half (on the Redmi K50 Ultra bench the GPU then took 1.75 ms a frame more).
// samples_report_seen: the game has recorded a samples-passed report (D2R never does: its ~8 pending reports a frame are
// all timestamps).
bool heavy_timestamps=false;extern uint64_t pass_fast_hits;   // gfx-heavy-timestamps = '1': timestamp reports end the pass with full barriers, as before
bool no_occlusion_queries=false,samples_report_seen=false;uint64_t samples_reports=0,samples_reported=0,counter_ops[3]={};extern uint64_t pipeline_fast_hits;extern bool pipe_memo;extern uint64_t pipe_memo_hits,pipe_memo_misses;extern uint64_t snap_front_hits,snap_map_hits,snap_invalidations,snap_invalidated;extern std::atomic<uint64_t> quad_rewrites,warp_rewrites,ldc_folds,temps_forwarded,nan_minmax_swaps;namespace vcache{extern bool enabled;extern uint64_t hits,hit_bytes,fills,fill_bytes,refills,volatiles,untracked,flushes;size_t live();VkDeviceSize arena_bytes();}   // counter_ops: resets, timestamp reports, samples reports recorded
// Resident uniform texel copies (texel_acquire, before descriptors()).
struct TexelKey{uint64_t base,bytes;uint32_t format;bool operator==(const TexelKey &o)const{return base==o.base&&bytes==o.bytes&&format==o.format;}};
struct TexelKeyHash{size_t operator()(const TexelKey &k)const{uint64_t x=k.base*0x9E3779B97F4A7C15ull^k.bytes*0xC2B2AE3D27D4EB4Full^k.format;x^=x>>29;return size_t(x*0xBF58476D1CE4E5B9ull);}};
struct TexelResident{Buffer buffer;VkBufferView view{};uint64_t hash=0,used=0,gen=0,checked=0,check_epoch=0;};   // checked: verified in this submission
std::unordered_map<TexelKey,TexelResident,TexelKeyHash> texel_resident[2];Arena texel_arena[2];uint64_t texel_hole_bytes[2]={},texel_live_bytes[2]={};
bool texel_memo_enabled=true;uint64_t texel_gen=0,texel_refreshes=0,texel_fallbacks=0,texel_evictions=0,texel_hits=0;
// A copy verified earlier in this submission stays valid until an in-stream CPU write overlaps it (snapshot_invalidate
// bumps texel_write_epoch), so a ~1 MB range bound by 558 draws is hashed once per frame, not per draw.
uint64_t texel_write_epoch=1,texel_lo=UINT64_MAX,texel_hi=0;void texel_touch(uint64_t address,uint64_t bytes);
void texel_evict(unsigned si,bool all);
// Diagnostic (NVN PERF texel): distinct uniform texel ranges per frame and how many the previous frame also used.
std::unordered_set<uint64_t> texel_frame_keys,texel_prev_keys;uint64_t texel_binds=0,texel_distinct=0,texel_reused=0;
// 显式gfx-pass-profile：每600次呈现采完整一帧的所有slot；整帧上限1024查询/32slot。
// Opt-in gfx-pass-profile: sample all slots of one frame every 600 presents,
// bounded to 1024 timestamp queries and 32 slots per selected frame.
// 标记文件里写「起始 间隔 [帧数]」时按它采样，例如 2115 1 60 = 第 2115 起逐帧采 60 帧（进营地的第一帧落在其中）；空文件同上。
// When the marker holds "FROM PERIOD [COUNT]" frames are sampled by it, e.g. 2115 1 60 = every frame for 60 frames from 2115
// (the camp's first frame falls among them); an empty file samples as above.
constexpr uint32_t pass_prof_query_limit=1024,pass_prof_slot_limit=32;
bool pass_prof_enabled=false;uint64_t pass_prof_next=300,pass_prof_draws=0,pass_prof_frame=UINT64_MAX,pass_prof_period=600,pass_prof_until=UINT64_MAX;
// Vertices (or indices) times instances drawn, for the per-pass verts= of NVN GPASS.
uint64_t pass_prof_verts=0;
uint32_t pass_prof_queries=0,pass_prof_slots=0,pass_prof_skipped_slots=0;
void pass_prof_end(VkCommandBuffer cmd);
void pass_prof_depth_idle(VkCommandBuffer cmd);
uint64_t depth_idle_passes=0;
// Diagnostic (NVN PERF pass_ends): render-pass ends and full barriers per frame, by the guest op being executed
// (pseudo-ops: 60 submit, 61 present, 62 storage-writing draw).
unsigned cur_op=0;uint64_t pass_end_by[64],barrier_by[64];
// np639：**开新 pass 的原因**。每帧 221 个 pass 是这个项目现在最大的开销来源（np634–np638），
//   但"为什么开"一直没读数：可能是换了 VkRenderPass（附件格式组合变了）、换了 framebuffer（附件身份变了）、
//   换了命令缓冲，或者根本没有 pass 开着。四个计数器分开记，另记每帧"开过多少个不同的 framebuffer"。
// np639: why a new pass is opened. 221 passes per frame is the biggest cost here (np634-np638) and there was
//   no reading for the reason: a different VkRenderPass (attachment formats), a different framebuffer
//   (attachment identity), a different command buffer, or no pass open at all. Four counters, plus how many
//   distinct framebuffers were opened per frame.
uint64_t pass_open_by[4];std::set<uint64_t> pass_open_fbs;
// np641：**pass 是被谁关的**。np639 已经证明"断 pass 的不是 RT 切换"，但 `pass_open none` 一直涨，
//   说明还有别处在 draw 之间把 pass 关掉。这里给每个 end_open_pass 调用点一个编号，直方图直接点名。
// np641: who closes the pass. np639 showed it is not the render-target switch, yet `pass_open none` keeps
//   growing, so something else closes the pass between draws. Each end_open_pass call site gets an id and the
//   histogram names it: 1 split segment, 2 submit, 3 Commands (non-draw op), 4 image transition,
//   5 draw's pass switch, 6 storage readback (draw / readback_buffer), 7 clear.
int pass_end_cause_now=0;uint64_t pass_end_cause[8];
static int g_probe_tiny_scissor=[](){const char *e=getenv("DM_PROBE_TINY_SCISSOR");return e?atoi(e):0;}();   // np647：判别探针，默认关
uint64_t prep_read_feedback;   // np644：反馈读（采样当前 pass 附件）而不断 pass 的次数 | np644: feedback reads let through without ending the pass
uint64_t prep_read_end;   // np641：prepare_read 里断 pass 的次数（每帧）| np641: pass ends inside prepare_read per frame
// np643：这一帧开 pass 的顺序（行程编码用）。512 条足够覆盖场景帧的 ~440 次。
// np643: this frame's pass-open order, for run-length encoding. 512 entries cover the scene frame's ~440 opens.
struct PassSeqEnt{uint64_t fb;uint32_t w,h,att;};
PassSeqEnt pass_seq[512];unsigned pass_seq_n=0;
// np701：**判死证据链**。np664 定谳的终局是"提交线程卡在不返回的驱动调用里"，但死亡图只给得出
//   "pc 落在 ld-musl 区间"——回答不了卡在哪个引擎调用（submit？fence？建管线？），更没有停帧前
//   那一帧的 pass/shader/管线清单（离线半程的原料）。这里补齐三件，全部**只记账不拖速**：
//   ① **调用点标定**：每个阻塞驱动调用（submit/fence/pipeline/acquire/present/queue-idle）用 VkSite
//      守卫登记 线程+站点+起始时刻+上下文（建管线=着色器键前 8 字节）。⚠ 所有"超时上限"（fence 10s、
//      acquire 50ms）都由同一个会挂死的驱动执行——真挂死时上限只是理论，所以守卫要挂满全部阻塞站点。
//   ② **deathguard 监视线程**：每秒一拍，某线程阻塞在同一驱动调用超过 DM_DEATHGUARD_MS（缺省 30000，
//      远小于看门狗 90s；大于合法窗口——驱动 reset 自愈 ~8s、fence 名义上限 10s）就打一行【判死】并
//      把证据落盘（两次落盘至少隔 120s，误报不吞真报）。**不退出**：看门狗 90s 的全线程转储照旧当
//      执行者，我们只是把"无名的 90 秒等死"变成"30 秒出全套现场"。
//   ③ **停帧清单常备**：逐 pass 形状（serial/尺寸/附件/深度/绘制数）、本帧绑定过的着色器、本帧建过
//      的管线键（miss/bypass 都算——np700 的 bypass 风暴靠它现形）。present 封存上一帧，转储带
//      sealed（上一完整帧）+ current（进行中帧）。定长 POD、无堆分配；转储线程不持引擎锁，读到
//      撕裂数字就是垃圾值，不会崩。
//   判据（np701）：真机自然死亡时 gfx-death.txt 在 ~30s 出现且点名卡点；gfx-lastframe.txt 的 pass
//   条数与 digest_ring 尾条对账；正常跑 ≥6 分钟零误报。
// np701: the death-evidence chain. np664's endgame was "the submit thread stuck in a driver call that never
//   returns", yet the death map only says "pc inside the ld-musl range" -- which engine call (submit? fence?
//   pipeline build?) stayed unknown, and the last frame's pass/shader/pipeline manifest (the offline half's
//   raw material) was never written. Three additions, all bookkeeping at recording speed:
//   1. Call-site labelling: every blocking driver call gets a VkSite guard (thread + site + start + context,
//      the first 8 key bytes for pipeline builds). Every "timeout" (fence 10 s, acquire 50 ms) is enforced by
//      the very driver that hangs -- on a real hang they are fiction, so the guards cover ALL blocking sites.
//   2. The deathguard thread (1 s tick): a thread blocked in the same call for over DM_DEATHGUARD_MS
//      (default 30000 -- well under the watchdog's 90 s, above every legal window: a driver reset recovers
//      in ~8 s, the fences' nominal cap is 10 s) logs one line and writes the evidence files (at most one
//      dump per 120 s, so a false alarm cannot swallow a real one). It does NOT exit: the watchdog keeps its
//      role at 90 s; a nameless 90-second wait becomes a full scene at 30.
//   3. The last-frame manifest, always resident: per-pass shapes, the frame's bound shaders, its built
//      pipeline keys (misses and bypasses alike -- np700's bypass storm would have been visible here).
//      The previous frame is sealed at present; a dump carries sealed + current. Bounded POD, no allocation;
//      the dump thread takes no engine lock -- a torn read yields a garbage number, never a crash.
namespace deathguard{
// 阻塞站点：编号的单一真源在 xbg_deathsite.h（vulkan_present.cpp 共用），名字给人看；
// 新增站点必须同时给名字，否则转储打出 "site=?"。
// Blocking sites: the numbers' single source of truth is xbg_deathsite.h (shared with
// vulkan_present.cpp); every site needs a name or the dump prints "site=?".
enum Site:unsigned{NONE=XBG_SITE_NONE,SUBMIT=XBG_SITE_SUBMIT,SUBMIT_SPLIT=XBG_SITE_SUBMIT_SPLIT,FENCE_RETIRE=XBG_SITE_FENCE_RETIRE,FENCE_POOLRESET=XBG_SITE_FENCE_POOLRESET,PIPE_GRAPHICS=XBG_SITE_PIPE_GRAPHICS,PIPE_COMPUTE=XBG_SITE_PIPE_COMPUTE,PIPE_OTHER=XBG_SITE_PIPE_OTHER,QUEUE_WAIT_IDLE=XBG_SITE_QUEUE_WAIT_IDLE,ACQUIRE=XBG_SITE_ACQUIRE,PRESENT=XBG_SITE_PRESENT,PRESENT_SUBMIT=XBG_SITE_PRESENT_SUBMIT,PRESENT_FENCE=XBG_SITE_PRESENT_FENCE,SITE_COUNT};
inline const char *site_name(unsigned s){static const char *n[]={"none","vkQueueSubmit","vkQueueSubmit(split)","vkWaitForFences(retire)","vkWaitForFences(poolreset)","vkCreateGraphicsPipelines","vkCreateComputePipelines","vkCreate*Pipeline(other)","vkQueueWaitIdle","vkAcquireNextImageKHR","vkQueuePresentKHR","vkQueueSubmit(present)","vkWaitForFences(present)"};return s<sizeof(n)/sizeof(*n)?n[s]:"site=?";}
struct Entry{std::atomic<uint64_t> t0;std::atomic<unsigned> site;std::atomic<uint32_t> tid;std::atomic<uint64_t> ctx;std::atomic<int> used;};
constexpr unsigned MAX_THREADS=64;
static Entry table[MAX_THREADS];
// 每线程自己的槽位：第一次进站点时登记（满 64 则该线程在转储里缺席，可接受）。
// One slot per thread, claimed on first use; a full table just means that thread shows no line.
static thread_local int my_slot=-1;
static int claim(){
    if(my_slot>=0)return my_slot;
    for(unsigned i=0;i<MAX_THREADS;i++){int expect=0;if(table[i].used.compare_exchange_strong(expect,1)){my_slot=int(i);table[i].tid.store(uint32_t(pthread_self()&0xffffffffu));return my_slot;}}
    return -1;
}
// RAII 守卫：先 t0 后 site（监视线程见 site!=none 而 t0==0 就跳过这一拍），出站点清 site。
// 类型定义已前置（poolreset 的守卫要用）；成员函数体在这里补。
// VkSite guard: t0 first, then site; cleared site-first on the way out. The type was forward-declared
// for poolreset's guard; the member bodies are defined here.
VkSite::VkSite(unsigned s,uint64_t ctx){slot=claim();if(slot<0)return;table[slot].t0.store(timing_clock());table[slot].ctx.store(ctx);table[slot].site.store(s);}
VkSite::~VkSite(){if(slot>=0)table[slot].site.store(NONE);}
// 站点上下文：字符串（管线键）前 8 字节折成 u64，转储按 hex 打。
static uint64_t ctx_of(const char *s){uint64_t v=0;if(s)for(int i=0;i<8&&s[i];i++)v|=uint64_t(uint8_t(s[i]))<<(i*8);return v;}
// ---------- 停帧清单（当前帧 + 封存的上一帧；录制线程写、转储线程读，无锁） ----------
// draws 用 pass_prof_draws 差值（第 4373 行处无条件每 draw ++）。pass 环写满后停记（保前面的），
// build 环写满后计数溢出（溢出本身就是 np700 型风暴的读数）。
struct PassRec{uint32_t w,h,att,samples,draws;uint64_t serial0;uint32_t depth,draws0;};
struct ShaderSeen{uint32_t stage;char key[17];uint32_t pad;uint64_t count;};
struct BuildRec{char key[65];uint64_t ms;};
constexpr unsigned MAX_PASSES=4096,MAX_SHADERS=4096,MAX_BUILDS=1024;
struct FrameManifest{
    uint64_t serial=0;unsigned pass_n=0,shader_n=0,build_n=0;uint64_t build_overflow=0;
    PassRec pass[MAX_PASSES];
    ShaderSeen shader[MAX_SHADERS];
    BuildRec build[MAX_BUILDS];
};
static FrameManifest cur,sealed;
static const void *last_prog[6];   // 相邻 draw 常重复同一着色器：先比指针，变了才进环里扫（每帧开头清零）
static void frame_seal(){
    cur.serial=stats.presents.load();   // np702 修正：先记号再封存——否则 sealed.serial 永远是旧值 0
    sealed=cur;
    cur=FrameManifest{};cur.serial=stats.presents.load();
    for(auto &p:last_prog)p=nullptr;
}
static void pass_begin(uint32_t w,uint32_t h,uint32_t att,uint32_t samples,uint64_t serial0,bool depth){
    if(cur.pass_n>=MAX_PASSES)return;
    PassRec &r=cur.pass[cur.pass_n];r.w=w;r.h=h;r.att=att;r.samples=samples;r.draws=0;r.serial0=serial0;r.depth=depth?1u:0u;r.draws0=uint32_t(pass_prof_draws);
    ++cur.pass_n;
}
static void pass_end(){
    if(!cur.pass_n)return;
    PassRec &r=cur.pass[cur.pass_n-1];r.draws=uint32_t(pass_prof_draws)-r.draws0;
}
static void shader_used(unsigned stage,const void *prog,const char *key){
    if(stage>=6||!prog||!key)return;
    if(last_prog[stage]==prog)return;   // 相邻 draw 常重复同一着色器：先比指针，变了才扫环
    last_prog[stage]=prog;
    for(unsigned i=0;i<cur.shader_n;i++){ShaderSeen &s=cur.shader[i];if(s.stage==stage&&!memcmp(s.key,key,16)){s.count++;return;}}
    if(cur.shader_n>=MAX_SHADERS)return;
    ShaderSeen &s=cur.shader[cur.shader_n];s.stage=stage;unsigned j=0;for(;j<16&&key[j];j++)s.key[j]=key[j];for(;j<16;j++)s.key[j]=0;s.key[16]=0;s.count=1;
    ++cur.shader_n;
}
static void pipeline_built(const char *bytes,uint64_t ms){
    if(cur.build_n>=MAX_BUILDS){cur.build_overflow++;return;}
    BuildRec &r=cur.build[cur.build_n];unsigned j=0;for(;j<64&&bytes[j];j++)r.key[j]=bytes[j];r.key[j]=0;r.ms=ms;
    ++cur.build_n;
}
static void write_lastframe(FILE *f,const char *tag,const FrameManifest &m){
    fprintf(f,"[%s] serial=%llu passes=%u shaders=%u builds=%u builds_overflow=%llu\n",
        tag,(unsigned long long)m.serial,m.pass_n,m.shader_n,m.build_n,(unsigned long long)m.build_overflow);
    unsigned np=m.pass_n;if(np>MAX_PASSES)np=MAX_PASSES;
    for(unsigned i=0;i<np;i++){const PassRec &r=m.pass[i];
        fprintf(f,"P %s %u w=%u h=%u att=%u samples=%u serial0=%llu depth=%u draws=%u\n",tag,i,r.w,r.h,r.att,r.samples,(unsigned long long)r.serial0,r.depth,r.draws);}
    unsigned ns=m.shader_n;if(ns>MAX_SHADERS)ns=MAX_SHADERS;
    for(unsigned i=0;i<ns;i++){const ShaderSeen &s=m.shader[i];fprintf(f,"S %s %s stage=%u n=%llu\n",tag,s.key,s.stage,(unsigned long long)s.count);}
    unsigned nb=m.build_n;if(nb>MAX_BUILDS)nb=MAX_BUILDS;
    for(unsigned i=0;i<nb;i++){const BuildRec &r=m.build[i];fprintf(f,"B %s %s ms=%llu\n",tag,r.key,(unsigned long long)r.ms);}
}
static std::atomic<uint64_t> last_dump_ms;
static void dump_files(const char *why){
    uint64_t now=timing_clock();
    uint64_t last=last_dump_ms.load();
    if(last && now-last<120000000000ull)return;   // 两次落盘至少隔 120s：误报不吞真报
    last_dump_ms.store(now);
    std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');std::string base=cut==std::string::npos?dir:dir.substr(0,cut);
    fprintf(stderr,"[dm] 判死证据落盘 %s\n",why);
    // gfx-lastframe.txt：停帧清单（sealed = 上一完整帧，current = 进行中帧）。
    if(FILE *f=fopen((base+"/gfx-lastframe.txt").c_str(),"wb")){
        fprintf(f,"DM-FRAME-MANIFEST v=1 pid=%d build=%s %s why=%s\n",getpid(),__DATE__,__TIME__,why);
        write_lastframe(f,"sealed",sealed);write_lastframe(f,"current",cur);fclose(f);}
    // gfx-death.txt：判死快照——线程站点表 + 提交摘要环 + lost 窗口 + pass 关闭原因直方图 + 管线计数。
    if(FILE *f=fopen((base+"/gfx-death.txt").c_str(),"wb")){fprintf(f,"DM-DEATH v=1 pid=%d build=%s %s why=%s\n",getpid(),__DATE__,__TIME__,why);
        for(unsigned i=0;i<MAX_THREADS;i++){const Entry &e=table[i];if(!e.used.load())continue;unsigned s=e.site.load();long long age=0;
            if(s){uint64_t t0=e.t0.load();if(t0&&t0<=now)age=(long long)((now-t0)/1000000);}
            fprintf(f,"T slot=%u tid=%u site=%u(%s) age_ms=%lld ctx=%016llx\n",i,e.tid.load(),s,site_name(s),age,(unsigned long long)e.ctx.load());}
        fprintf(f,"digest ring (newest first):\n");
        for(unsigned i=0;i<16;i++){const SubDigest &d=digest_ring[(digest_at+15u-i)&15u];if(!d.serial)continue;
            fprintf(f,"D serial=%llu draws=%u passes=%u disp=%u\n",(unsigned long long)d.serial,d.draws,d.passes,d.dispatches);}
        {std::lock_guard<std::mutex> l(g_lost_mu);unsigned cd_left=0;
         // ⚠ np702 首跑的学费：这里原来调 lost_cooldown_ms()——它内部也锁 g_lost_mu（非递归），
         //   监视线程当场自死锁：death 文件缓冲永不刷盘（0 字节）、锁还被攥着 ⇒ 之后任何提交
         //   调 lost_cooldown_ms() 全部挂住。冷路径数学就地展开（lost_cooldown_ms_env() 无锁）。
         //   The first run's lesson: lost_cooldown_ms() re-locks g_lost_mu inside this guard --
         //   instant self-deadlock (0-byte file, monitor stuck holding the lock, every later
         //   submit hangs on its own lost_cooldown_ms()). Inline the math instead.
         if(g_lost_ns){uint64_t elapsed=(timing_clock()-g_lost_ns)/1000000u;unsigned cd=lost_cooldown_ms_env();cd_left=elapsed>=cd?0u:cd-(unsigned)elapsed;}
         fprintf(f,"lost armed=%s ago_ms=%lld cooldown_left_ms=%u\n",g_lost_ns?"yes":"no",
             g_lost_ns?(long long)((timing_clock()-g_lost_ns)/1000000):0LL,cd_left);}
        fprintf(f,"pass_end_cause:");
        for(unsigned i=0;i<8;i++)fprintf(f," %u=%llu",i,(unsigned long long)pass_end_cause[i]);
        fprintf(f,"\npipeline_counts hits=%llu misses=%llu bypass=%llu evictions=%llu\n",
            (unsigned long long)pipeline_counts.hits,(unsigned long long)pipeline_counts.misses,(unsigned long long)pipeline_counts.bypass,(unsigned long long)pipeline_counts.evictions);
        write_lastframe(f,"sealed",sealed);write_lastframe(f,"current",cur);fclose(f);}
}
// np703：队列死出口——切段路径连续丢段到上限时由提交线程调（队列被连续拒绝，不是挂死，
// deathguard 的 30s 站龄检测抓不到这种死法）。落盘与判死同款证据后，调用方 FATAL。
// np703: the queue-dead exit -- called by the submit thread when the split path hits its drop
//   limit (the queue is being *rejected*, not hung, so the 30s site-age detector can't see it).
//   Writes the same evidence as a verdict, then the caller fatals.
void queue_dead(unsigned seg,VkResult r){
    char why[160];snprintf(why,sizeof why,"队列死：段 %u 连续丢段到上限（最后错误 %d）——队列被连续拒绝",seg,int(r));
    xbg_logf("【判死】%s —— 证据落盘",why);
    dump_files(why);
}
// 监视线程：1s 一拍，只读原子与 POD；判死后进程不退（看门狗 90s 兜底照旧）。
static void *monitor_main(void *){
    uint64_t threshold_ms=30000;if(const char *e=getenv("DM_DEATHGUARD_MS")){long v=atol(e);if(v>=15000)threshold_ms=uint64_t(v);}
    xbg_logf("NVN deathguard 已启动（阈值 %llu 秒，DM_DEATHGUARD_MS 可调；判死只落盘不退出）",(unsigned long long)(threshold_ms/1000));
    for(;;){
        usleep(1000000);
        uint64_t now=timing_clock();unsigned worst_site=NONE;int worst_slot=-1;long long worst_age=0;uint64_t worst_ctx=0;
        for(unsigned i=0;i<MAX_THREADS;i++){const Entry &e=table[i];if(!e.used.load())continue;unsigned s=e.site.load();if(s==NONE)continue;
            uint64_t t0=e.t0.load();if(!t0||t0>now)continue;long long age=(long long)((now-t0)/1000000);
            if(age>(long long)threshold_ms&&age>worst_age){worst_age=age;worst_site=s;worst_slot=int(i);worst_ctx=e.ctx.load();}}
        if(worst_slot<0)continue;
        char why[192];snprintf(why,sizeof why,"线程 tid=%u 卡在 %s 已 %lld 秒 ctx=%016llx（阈值 %llu 秒）",
            table[worst_slot].tid.load(),site_name(worst_site),worst_age/1000,(unsigned long long)worst_ctx,(unsigned long long)(threshold_ms/1000));
        xbg_logf("【判死】%s —— 证据落盘（进程不退，看门狗 90 秒兜底照旧）",why);
        dump_files(why);
    }
    return nullptr;
}
void start(){
    static int begun;if(begun)return;begun=1;
    pthread_t t;if(pthread_create(&t,nullptr,monitor_main,nullptr)==0)pthread_detach(t);
    else xbg_logf("【红】deathguard 监视线程启动失败（判死证据链降级为仅 wait_dump）");
}
// 看门狗 stall/判死转储的图形侧等待点状态（crash_set_wait_dump 注册，弱——不链宿主时注册不了）。
// 无锁、只读原子与 POD，每行有界：线程表最多 64 行 + 摘要 16 行 + 清单两段。
void wait_dump(){
    fprintf(stderr,"[dm-wait] 线程站点表：\n");
    for(unsigned i=0;i<MAX_THREADS;i++){const Entry &e=table[i];if(!e.used.load())continue;unsigned s=e.site.load();long long age=0;
        if(s){uint64_t t0=e.t0.load();uint64_t now=timing_clock();if(t0&&t0<=now)age=(long long)((now-t0)/1000000);}
        fprintf(stderr,"[dm-wait] T slot=%u tid=%u site=%s age_ms=%lld ctx=%016llx\n",i,e.tid.load(),site_name(s),age,(unsigned long long)e.ctx.load());}
    fprintf(stderr,"[dm-wait] 提交摘要环（新→旧）：\n");
    for(unsigned i=0;i<16;i++){const SubDigest &d=digest_ring[(digest_at+15u-i)&15u];if(!d.serial)continue;
        fprintf(stderr,"[dm-wait] D serial=%llu draws=%u passes=%u disp=%u\n",(unsigned long long)d.serial,d.draws,d.passes,d.dispatches);}
    fprintf(stderr,"[dm-wait] 停帧清单（内存，sealed=上一完整帧 current=进行中帧）：\n");
    write_lastframe(stderr,"sealed",sealed);write_lastframe(stderr,"current",cur);
}
// 跨编译单元入口（xbg_deathsite.h 契约）：同线程进出配对，支持嵌套（栈深 4 够用——present 回调里最多一层）。
// Cross-TU entry (the xbg_deathsite.h contract): enter/leave pairs per thread, nesting supported (depth 4).
struct XSite{int slot;unsigned prev;uint64_t prev_ctx;};
static thread_local XSite xstack[4];static thread_local int xdepth=0;
void x_enter(unsigned site,uint64_t ctx){
    int slot=claim();if(slot<0||xdepth>=4)return;
    XSite &x=xstack[xdepth++];x.slot=slot;x.prev=table[slot].site.load();x.prev_ctx=table[slot].ctx.load();
    table[slot].t0.store(timing_clock());table[slot].ctx.store(ctx);table[slot].site.store(site);
}
void x_leave(){
    if(!xdepth)return;
    XSite &x=xstack[--xdepth];
    if(x.prev==NONE)table[x.slot].site.store(NONE);
    else{table[x.slot].site.store(x.prev);table[x.slot].t0.store(timing_clock());table[x.slot].ctx.store(x.prev_ctx);}
}
// draw_bound 读 state（第 2486 行才定义），函数体在那一行之后补定义；这里只留声明。
void draw_bound();
}   // namespace deathguard
// ⚠ xbg_death_site_enter/leave 是 vulkan_present.cpp（另一编译单元）要链接的契约符号——
//   匿名命名空间里 extern "C" 会变成内部链接 ⇒ 定义移到本匿名命名空间闭合之后（下方 "~3040 行"处）。
//   crash_set_wait_dump 的 weak 声明同理上移到文件头部全局作用域。
// ⚠ xbg_death_site_enter/leave are contract symbols linked from vulkan_present.cpp (another TU) --
//   extern "C" inside an anonymous namespace gets internal linkage, so the definitions live after
//   this anonymous namespace closes (below); crash_set_wait_dump's weak decl moved to file head.
// Samples-passed counting: one precise occlusion query per run of counted draws inside a pass (Metal counts per pass,
// nvn_mtl.m), not one per draw. A run ends with its pass, at a counter reset, and at a report (which ends the pass), so a
// report still sums exactly the draws since its reset. ~2200 begin/end pairs per D2R camp frame cost the Adreno 730
// ~10 ms of GPU time and ~4 ms of CPU. gfx-query-per-draw = '1' restores one query per draw.
// The run helpers and names are main's (1b5465e); occlusion_batch_enabled is false with gfx-query-per-draw = '1' or a
// gfx-no-query-batch marker.
struct OpenOcclusion {VkCommandBuffer cmd{};VkQueryPool pool{};uint32_t index=0;} open_occlusion;
bool occlusion_batch_enabled=true;uint64_t occlusion_queries_started=0;
void end_occlusion_query(){
    if(open_occlusion.cmd){vkCmdEndQuery(open_occlusion.cmd,open_occlusion.pool,open_occlusion.index);open_occlusion=OpenOcclusion{};}
}
void begin_occlusion_query(VkCommandBuffer cmd,VkQueryPool pool,uint32_t index){
    need(!open_occlusion.cmd,"Nested occlusion query");
    vkCmdBeginQuery(cmd,pool,index,VK_QUERY_CONTROL_PRECISE_BIT);open_occlusion={cmd,pool,index};++occlusion_queries_started;
}
// np713：阴影分辨率开关 DM_GFX_SHADOW_SCALE=<百分比 20..90，缺省 100=关>。识别规则（通用，
//   不写死游戏）：**单颜色附件 + 正方形 + ≥512** 的 pass（BotW 的线性深度阴影图 896²，30ms/pass、
//   0.36 频率/帧 ⇒ 均 11ms/帧）。做法：视口/裁剪压到 sw×sh 渲染进左上角，pass 结束后经 tmp 图
//   NEAREST blit 放大回全图（同图自 blit 的重叠区非法，必须中转；depth-as-color 用 NEAREST——
//   R32F 无 LINEAR blit 特性）。阴影本来就是模糊语义，块状放大可接受。
//   判据：GPASS 该 pass 的 us 塌降到 ~scale² 倍 + 探针画面无回归 + fps 上行。
// np713: shadow-resolution switch. Identification (generic): exactly one color attachment, square, >=512
//   (BotW's linear-depth shadow map 896^2). Render with viewport/scissor squashed to sw x sh; after the
//   pass, upscale the corner over the full image through a tmp image with a NEAREST blit.
void shadow_scale_blit(VkCommandBuffer cmd);   // np713：定义在文件后部（Image 完整定义 + full_barrier 可见处）
static int shadow_scale_pct=-1;
static bool g_shadow_armed=false;static Image *g_shadow_img=nullptr;static bool g_shadow_scale_once=false;
static unsigned g_shadow_sw=0,g_shadow_sh=0,g_shadow_pw=0,g_shadow_ph=0,g_shadow_quad=0;
static VkImage g_shadow_tmp=nullptr;static VkDeviceMemory g_shadow_tmp_mem=nullptr;
static unsigned g_shadow_tmp_w=0,g_shadow_tmp_h=0;static unsigned g_shadow_tmp_fmt=0;
static bool shadow_scale_on(){if(shadow_scale_pct<0){const char *e=getenv("DM_GFX_SHADOW_SCALE");shadow_scale_pct=e?atoi(e):100;if(shadow_scale_pct>0&&shadow_scale_pct<100)xbg_logf("NVN shadow scale=%d%% (DM_GFX_SHADOW_SCALE)",shadow_scale_pct);}return shadow_scale_pct>0&&shadow_scale_pct<100;}
void end_open_pass(){if(open_pass.cmd){
    // np682：**pass 内涂蓝**。窗口那张图每帧都被绑成 index=0 附件、视口裁剪全尺寸、绘制也挂在同一镜像上，
    //   可读回全是 0。这里在**同一个 pass 内部**直接 vkCmdClearAttachments 涂成蓝色：
    //   屏幕变蓝 ⇒ pass→这张图的通路是通的，问题在"客体的 draw 没产生像素"；
    //   屏幕不变 ⇒ 这条 pass 的附件根本不是我们读/上屏的那张图（fb 层身份问题）。
    //   不额外提交、不动 layout，风险远低于帧中读回那版（那版会崩）。
    // np682: in-pass fill. The image is bound every frame with full viewport/scissor and the draws reference the
    //   same image, yet it reads back as zero. Clear it to blue *inside the same pass*: blue on screen => the
    //   pass->image path works and the guest's draws are what produce nothing; unchanged => this pass's
    //   attachments are not the image we read/present (an identity problem at the framebuffer level).
    {static int fill=-1;if(fill<0){const char *e=getenv("DM_WINPASS_FILL");fill=(e&&*e=='1')?1:0;if(fill)xbg_logf("[上屏] 窗口 pass 内涂蓝 DM_WINPASS_FILL=1");}
     static int seen=0,nfired=0;
     if(seen<20){++seen;xbg_logf("[上屏] pass 关 serial0=%llu 上屏serial=%llu 涂蓝=%d",(unsigned long long)open_pass.serial0,(unsigned long long)g_winfill_serial,fill);}
     if(fill && win_serial_match(open_pass.serial0)){if(++nfired<=5)xbg_logf("[上屏] 涂蓝第 %d 次（serial=%llu）",nfired,(unsigned long long)open_pass.serial0);
        VkClearAttachment a{};a.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;a.colorAttachment=0;
        a.clearValue.color.float32[0]=0.f;a.clearValue.color.float32[1]=0.f;a.clearValue.color.float32[2]=1.f;a.clearValue.color.float32[3]=1.f;
        VkClearRect r{};r.rect.extent={8192,8192};r.layerCount=1;vkCmdClearAttachments(open_pass.cmd,1,&a,1,&r);}}
    end_occlusion_query();vkCmdEndRenderPass(open_pass.cmd);
    // np713：缩放后的阴影放大回全图（shadow_scale_blit 定义在文件后部）
    if(g_shadow_armed && g_shadow_img)shadow_scale_blit(open_pass.cmd);
    g_shadow_armed=false;g_shadow_img=nullptr;if(open_pass.depth_attached && !open_pass.depth_used){++depth_idle_passes;pass_prof_depth_idle(open_pass.cmd);}pass_prof_end(open_pass.cmd);deathguard::pass_end();open_pass=OpenPass{};++pass_end_by[cur_op<64?cur_op:63];++pass_end_cause[pass_end_cause_now&7];}}
// np676：**帧中自检**——在"刚画完窗口那张图"的瞬间读回它左上 64x64，区分两种完全不同的病：
//   A 读到有内容 ⇒ 绘制确实落进了这张图，present 之前/之中被清掉或被另一张图顶替；
//   B 读到全 0   ⇒ 这些 draw 根本没在这张图上产生像素（渲染执行层）。
//   与 present 时刻的自检配对（一个查"最终值"，一个查"刚写完的值"）。实现放在 present 探针旁边。
// np676: mid-frame probe -- right after the pass that draws into the presented image is closed, read back its
//   top-left 64x64. A: non-zero => the draws landed and something clears/replaces it before present.
//   B: all zero => those draws never produced pixels in that image. Definition lives next to the present probe.
static Image *g_winpass_last=nullptr;
void end_pass(int cause){pass_end_cause_now=cause;
    g_winpass_last=open_pass.images.n?open_pass.images.v[0]:nullptr;
    end_open_pass();pass_end_cause_now=0;if(g_winpass_hook)g_winpass_hook();}   // np641：带"谁关的"编号的 end_open_pass | np641: end_open_pass tagged with the call site
// np712：屏障站点标注——PERF 行按调用点计数（barrier sites），找出 478/帧 的真实出处。
// np712: barrier site tagging -- the PERF line counts by call site to locate the 478/frame barriers.
static const char *g_barrier_site="other";
// np712b：冗余屏障跳过。游戏每帧发 ~780 个 nvnCommandBufferBarrier（标题相位实测），Maleoon 上
//   每个 ~58µs（45ms/帧）。两个屏障之间若没有任何 GPU 写（draw/dispatch/copy/clear），屏障没有
//   可保证的可见性 ⇒ 跳过。g_gpu_wrote 由各写入口置位；跳过计数进 PERF（@barrier-skip）。
// np712b: redundant-barrier skip. The game issues ~780 nvnCommandBufferBarrier per frame (measured in
//   the title phase), ~58µs each on Maleoon (45ms/frame). Between two barriers with no GPU write
//   (draw/dispatch/copy/clear) a barrier guarantees nothing ⇒ skip it. g_gpu_wrote is set by every
//   write entry; the skip count goes to PERF (@barrier-skip).
static bool g_gpu_wrote=false;static uint64_t g_barrier_skip=0,g_barrier_emit=0;
static const char *barrier_site_name[32];static uint64_t barrier_site_cnt[32];static unsigned barrier_site_n=0;
static unsigned barrier_site_id(const char *n){for(unsigned i=0;i<barrier_site_n;i++)if(barrier_site_name[i]==n)return i;if(barrier_site_n<32){barrier_site_name[barrier_site_n]=n;return barrier_site_n++;}return 31;}
void full_barrier(VkCommandBuffer cmd){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&b,0,nullptr,0,nullptr);barrier_seq=write_seq;++barrier_by[cur_op<64?cur_op:63];++barrier_site_cnt[barrier_site_id(g_barrier_site)];}
// np640：**pass 内合法的内存屏障**。render pass 内部不能出现 pass 之外的阶段（ALL_COMMANDS 会带进来，
//   验证层报 VUID），所以 pass 开着时把阶段掩码收窄到图形阶段；pass 没开就保持 ALL_COMMANDS 的原语义。
//   用途：不该断 pass 的地方（barrier 回放、storage 回读的拷贝）就地发屏障，别再 end_open_pass()。
// np640: a memory barrier that is legal inside a render pass. Inside a pass the stage mask must not name stages
//   outside it (ALL_COMMANDS does, and the validation layer says so), so the mask is narrowed to graphics stages
//   while a pass is open; with no pass open it keeps full_barrier's ALL_COMMANDS semantics. Used where ending the
//   pass is unnecessary (barrier replay, the storage-readback copy).
void full_barrier_in_pass(VkCommandBuffer cmd){
    VkPipelineStageFlags st=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if(open_pass.cmd==cmd)st=VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT|VK_PIPELINE_STAGE_VERTEX_INPUT_BIT|VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,st,st,0,1,&b,0,nullptr,0,nullptr);barrier_seq=write_seq;++barrier_by[cur_op<64?cur_op:63];++barrier_site_cnt[barrier_site_id(g_barrier_site)];
}
struct PerfCounts{size_t pipelines,layouts,samplers,passes,images,host_segments,host_bytes;};
PerfCounts perf_counts();
struct Perf {
    uint64_t frames=0,submits=0,waits=0,sync_writeback=0,sync_readback=0,sync_query=0,sync_other=0,gpu_ns=0,gpu_samples=0,
        submit_cpu_ns=0,present_cpu_ns=0,passes=0,draws=0,dispatches=0,snapshot_hits=0,snapshot_misses=0,pending_views=0,access_by[12]={},folded_clears=0,why_wait=0,why_poll=0,why_access=0,why_cmdwait=0,why_finish=0,why_slot=0,snapshot_bytes=0,arena_peak=0,window_start=0,ops=0;
} perf;
void snapshot_reset();void last_pass_reset();
void release_descriptors(Slot &s);
void snapshot_invalidate(uint64_t address,uint64_t bytes);
// Diagnostic (marker gfx-event-guard): while a SignalEvent's value is deferred (gfx-defer-events), the page holding the event
// word is made inaccessible, so a read of it that bypasses nvnEventGetValue faults and is counted (NVN PERF evguard), then
// the page is opened again. Applying the event opens it first. Pages the dirty tracker or wtrack protect would conflict:
// run it with gfx-vcache=0, gfx-tex-dirty=0 and gfx-wtrack=0.
namespace evguard {
bool enabled=false;constexpr int slots_n=64;
std::atomic<uintptr_t> page[slots_n],word[slots_n];   // guarded page; the pending event word in it (one per slot)
std::atomic<uint64_t> exact{0},other{0},writes{0},guards{0};std::atomic<uintptr_t> last_pc{0},last_addr{0};
// The first faulting access of a guarded page and who made it: the caller (LR; bionic's memcpy is a leaf), x0-x5 at the fault,
// and the pending event word on that page, so a bulk copy that crosses the word can be told from one that stops before it.
std::atomic<uintptr_t> last_lr{0},last_word{0},last_x[6];std::atomic<uintptr_t> callers[8];std::atomic<uint64_t> caller_hits[8];
inline uintptr_t pg(uintptr_t a){return a&~uintptr_t(4095);}
void reclose();
void guard(uint64_t addr){if(!enabled)return;uintptr_t p=pg(uintptr_t(addr));
    for(int i=0;i<slots_n;i++)if(page[i].load()==p && word[i].load()==uintptr_t(addr))return;
    bool shut=false;for(int i=0;i<slots_n;i++)if(page[i].load()==p)shut=true;
    for(int i=0;i<slots_n;i++){uintptr_t z=0;if(word[i].compare_exchange_strong(z,uintptr_t(addr))){page[i]=p;break;}}
    if(!shut){mprotect(reinterpret_cast<void*>(p),4096,PROT_NONE);++guards;}reclose();}
void open(uintptr_t p){mprotect(reinterpret_cast<void*>(p),4096,PROT_READ|PROT_WRITE);for(int i=0;i<slots_n;i++)if(page[i].load()==p){page[i]=0;word[i]=0;}}
// A faulting access opens its page but keeps the pending words; reclose() shuts such pages again (at every guard(), and
// after each engine copy), so later reads of a word that is still pending fault too.
std::atomic<bool> reopened[slots_n];std::atomic<uint64_t> engine_opens{0},recloses{0};
void reclose(){for(int i=0;i<slots_n;i++)if(reopened[i].exchange(false)){uintptr_t p=page[i].load();if(p){mprotect(reinterpret_cast<void*>(p),4096,PROT_NONE);++recloses;}}}
// The engine's own copies of guest memory (stream_buffer) open the guarded pages they cover for the copy and shut them
// after, so the guard stays on for guest reads; they are counted (engine_opens), not reported as faults.
struct EngineRead{uintptr_t pages[8];int n=0;
    EngineRead(const void *src,size_t bytes){if(!enabled || !bytes)return;uintptr_t lo=pg(uintptr_t(src)),hi=pg(uintptr_t(src)+bytes-1);
        for(int i=0;i<slots_n && n<8;i++){uintptr_t q=page[i].load();if(!q || q<lo || q>hi)continue;bool dup=false;for(int k=0;k<n;k++)if(pages[k]==q)dup=true;
            if(!dup){mprotect(reinterpret_cast<void*>(q),4096,PROT_READ|PROT_WRITE);pages[n++]=q;++engine_opens;}}}
    ~EngineRead(){for(int k=0;k<n;k++){bool still=false;for(int i=0;i<slots_n;i++)if(page[i].load()==pages[k])still=true;if(still)mprotect(reinterpret_cast<void*>(pages[k]),4096,PROT_NONE);}}
};
void unguard(uint64_t addr){if(!enabled)return;uintptr_t p=pg(uintptr_t(addr));for(int i=0;i<slots_n;i++)if(page[i].load()==p){open(p);return;}}
int fault(void *a,int w,void *uc){uintptr_t x=uintptr_t(a),p=pg(x);bool mine=false,hit=false;
    uintptr_t near=0;
    for(int i=0;i<slots_n;i++)if(page[i].load()==p){mine=true;uintptr_t e=word[i].load();near=e;if(x>=e && x<e+4)hit=true;}
    if(!mine)return 0;
    ++(hit?exact:other);if(w)++writes;last_addr=x;last_word=near;
#if defined(__aarch64__) && defined(__ANDROID__)
    if(uc){auto &mc=static_cast<ucontext_t*>(uc)->uc_mcontext;last_pc=uintptr_t(mc.pc);uintptr_t lr=uintptr_t(mc.regs[30]);last_lr=lr;
        for(int i=0;i<6;i++)last_x[i]=uintptr_t(mc.regs[i]);
        for(int i=0;i<8;i++){uintptr_t z=0;if(callers[i].load()==lr || callers[i].compare_exchange_strong(z,lr) || callers[i].load()==lr){++caller_hits[i];break;}}}
#endif
    mprotect(reinterpret_cast<void*>(p),4096,PROT_READ|PROT_WRITE);for(int i=0;i<slots_n;i++)if(page[i].load()==p)reopened[i]=true;return 1;}
}
void apply_event(uint64_t addr,uint32_t mode,uint32_t v){
    evguard::unguard(addr);snapshot_invalidate(addr,4);auto ptr=reinterpret_cast<uint32_t*>(addr);
    if(mode==NVN_EVENT_SIGNAL_MODE_ADD)__atomic_fetch_add(ptr,v,__ATOMIC_SEQ_CST);else{need(mode==NVN_EVENT_SIGNAL_MODE_WRITE,"Unknown event mode");__atomic_store_n(ptr,v,__ATOMIC_SEQ_CST);}
}
// Deferred events in recording order; stop at the first whose slot is still in flight (slots retire in submission order).
void apply_ready_events(){
    size_t k=0;for(;k<pending_events.size();++k){const PendingEvent &e=pending_events[k];if(static_cast<const Slot*>(e.slot)->pending)break;apply_event(e.addr,e.mode,e.v);}
    pending_events.erase(pending_events.begin(),pending_events.begin()+k);
}
void retire_slot(Slot &s){
    if(!s.pending)return;
    poolreset::finish(s);
    {CPU_SCOPE(Fence);bool busy=vkGetFenceStatus(cfg.device,s.fence)==VK_NOT_READY;uint64_t b0=busy?timing_clock():0;
     VkResult wr;{deathguard::VkSite site(deathguard::FENCE_RETIRE);wr=vkWaitForFences(cfg.device,1,&s.fence,VK_TRUE,std::max<uint64_t>(cfg.fence_timeout_ns,10000000000ull));}
     // np633：**fence 死等兜底**。真 device lost 时提交被驱动"成功接收"但实际丢弃 ⇒ fence 永不 signal
     // ⇒ 10 秒超时返回 -4，原 VK() 直接 FATAL（np630run2 看门狗现场就是它）。改成：不 FATAL，
     // vkResetFences 后当完成处理（丢一帧画面，进程活着）；每 3 次报一行。DM_FENCE_SKIP=0 恢复严格。
     // np633: fence-wait fallback. On a real device-lost the submission is "accepted" but dropped, so the
     // fence never signals and the 10 s wait returns -4 (the np630run2 watchdog scene). Skip instead of
     // FATAL: reset the fence and treat as done (one dropped frame, process alive).
     if(wr!=VK_SUCCESS){static int fn=0;
        if(retry_submit_enabled()){if(fn++<3||fn%20==0)xbg_logf("【宽限】fence 等待超时（%d）——重置后跳过本帧等待（第 %d 次）",int(wr),fn);
            vkResetFences(cfg.device,1,&s.fence);}
        else VK(wr);
     }
     if(busy){block_ns[sync_site]+=timing_clock()-b0;++block_calls[sync_site];}}
    ++perf.waits;
    if(s.prof){
        s.prof=false;
        // 栅栏不替代查询可用性检查；按有效位数计算环绕差，只报告完整查询对。
        // Check availability as well as the fence; mask timestamp wrap to valid bits.
        struct Stamp{uint64_t value=0,available=0;};std::vector<Stamp> t(s.prof_used);
        VkResult status=s.prof_used?vkGetQueryPoolResults(cfg.device,s.prof_pool,0,s.prof_used,t.size()*sizeof(Stamp),t.data(),sizeof(Stamp),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT):VK_SUCCESS;
        const uint64_t mask=timestamp_bits<64?(uint64_t(1)<<timestamp_bits)-1:~0ull;
        double per=properties.limits.timestampPeriod/1000.0,pass_us=0,compute_us=0,span_us=-1;size_t complete=0;
        uint64_t first=0,last=0;
        if(status==VK_SUCCESS || status==VK_NOT_READY)for(size_t i=0;i<s.prof_recs.size();++i){const auto &r=s.prof_recs[i];
            if(!r.complete || r.q0>=t.size() || r.q1>=t.size() || !t[r.q0].available || !t[r.q1].available)continue;
            double us=double((t[r.q1].value-t[r.q0].value)&mask)*per;
            if(!complete)first=t[r.q0].value;last=t[r.q1].value;++complete;
            if(r.compute){compute_us+=us;xbg_logf("NVN GPASS frame=%llu slot=%u ring=%u i=%zu kind=compute us=%.0f cs=%s groups=%u,%u,%u",(unsigned long long)s.prof_frame,s.prof_slot,unsigned(&s-slots),i,us,r.first.c_str(),r.x,r.y,r.z);}
            else{pass_us+=us;xbg_logf("NVN GPASS frame=%llu slot=%u ring=%u i=%zu kind=pass us=%.0f %ux%u att=%u fmt=%u draws=%u verts=%llu first=%s last=%s mixed=%u dz=%u",(unsigned long long)s.prof_frame,s.prof_slot,unsigned(&s-slots),i,us,r.w,r.h,r.att,r.fmt,r.draws1-r.draws0,(unsigned long long)(r.verts1-r.verts0),r.first.empty()?"builtin-clear/no-game-draw":r.first.c_str(),r.last.empty()?"builtin-clear/no-game-draw":r.last.c_str(),unsigned(r.mixed),unsigned(r.dz));}
        }
        if(complete)span_us=double((last-first)&mask)*per;
        xbg_logf("NVN GPASS frame=%llu slot=%u ring=%u summary recorded=%zu complete=%zu skipped=%u queries=%u result=%d pass_us=%.0f compute_us=%.0f span_us=%.0f",(unsigned long long)s.prof_frame,s.prof_slot,unsigned(&s-slots),s.prof_recs.size(),complete,s.prof_skipped,s.prof_used,int(status),pass_us,compute_us,span_us);
    }
    if(s.timestamps){uint64_t t[2]{};if(vkGetQueryPoolResults(cfg.device,s.timestamps,0,2,sizeof(t),t,8,VK_QUERY_RESULT_64_BIT)==VK_SUCCESS && t[1]>=t[0]){uint64_t mask=timestamp_bits<64?(uint64_t(1)<<timestamp_bits)-1:~0ull;const uint64_t ns=uint64_t(double((t[1]-t[0])&mask)*properties.limits.timestampPeriod);perf.gpu_ns+=ns;++perf.gpu_samples;hud::gpu_add(s.hud_frame,ns);}}
    CPU_SCOPE(Cleanup);uint64_t c0=timing_clock();
    // One readback per occlusion pool (the fence has signalled, so every used query is available).
    {static std::vector<uint64_t> results;size_t used=s.occlusion_used;results.assign(used,0);
     if(s.query_copied==used && used){s.query_results.invalidate();memcpy(results.data(),s.query_results.mapped,used*8);}
     else for(size_t i=0;i*256<used;i++){uint32_t n=uint32_t(std::min<size_t>(256,used-i*256));if(vkGetQueryPoolResults(cfg.device,s.occlusion_pools[i],0,n,n*sizeof(uint64_t),results.data()+i*256,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT)!=VK_SUCCESS)for(uint32_t k=0;k<n;k++)results[i*256+k]=0;}
     s.query_copied=0;
     for(size_t i=0;i<s.query_refs.size();i++){auto e=epochs.find(s.query_refs[i].first);if(e==epochs.end())continue;uint64_t a=s.query_refs[i].second;
         if(a>=e->second.base && a<e->second.size()){auto &q=e->second.q[a-e->second.base];q.value=i<used?results[i]:0;q.known=true;}}
     s.query_refs.clear();for(auto &e:epochs)e.second.advance_known();}
    for(auto &r:pending_reports)if(r.slot==&s && !r.ts_known){uint64_t v=0;if(vkGetQueryPoolResults(cfg.device,r.ts,r.ts_index,1,sizeof(v),&v,sizeof(v),VK_QUERY_RESULT_64_BIT)==VK_SUCCESS)r.ts_value=v;r.ts_known=true;}
    s.occlusion_used=0;s.timestamp_used=0;uint64_t c1=timing_clock();
    if(!resolving_pending){resolve_slot_writebacks(&s);resolve_known_reports();
        bool waiting=false;for(auto &r:pending_reports)if(r.slot==&s)waiting=true;if(!waiting)publish_completed(std::max(completed.load(),s.covers));}
    uint64_t c2=timing_clock();while(s.waiters.n.load())std::this_thread::yield();VK(vkResetFences(cfg.device,1,&s.fence));   // 栅栏已发信号，等待者马上返回 | the fence has signalled, so waiters return at once
    for(auto &f:s.framebuffers)vkDestroyFramebuffer(cfg.device,f.second,nullptr);s.framebuffers.clear();
    for(auto &v:s.texel_views)vkDestroyBufferView(cfg.device,v.second,nullptr);s.texel_views.clear();
    uint64_t c3=timing_clock();s.retire.run();
    {const void *me=&s;
     for(auto it=buf_slot_refs.begin();it!=buf_slot_refs.end();){auto &v=it->second.slots;v.erase(std::remove(v.begin(),v.end(),me),v.end());if(v.empty())it=buf_slot_refs.erase(it);else ++it;}   // np627
     for(auto it=img_slot_refs.begin();it!=img_slot_refs.end();){auto &v=it->second.slots;v.erase(std::remove(v.begin(),v.end(),me),v.end());if(v.empty())it=img_slot_refs.erase(it);else ++it;}   // np628
    }
    release_descriptors(s);uint64_t c4=timing_clock();
    perf.arena_peak=std::max<uint64_t>(perf.arena_peak,s.stream.peak+s.readback.peak);s.stream.peak=s.readback.peak=0;
    s.stream.reset();s.readback.reset();
    // Every set is fully rewritten before use, so sets are recycled instead of resetting pools (vkAllocateDescriptorSets cost ~15 us/set on Maleoon).
    for(auto &u:s.used_sets){auto &sp=s.spare_sets[u.first];sp.insert(sp.end(),u.second.begin(),u.second.end());u.second.clear();}s.set_cache.clear();
    cleanup_ns[0]+=c1-c0;cleanup_ns[1]+=c2-c1;cleanup_ns[2]+=c3-c2;cleanup_ns[3]+=c4-c3;cleanup_ns[4]+=timing_clock()-c4;
    s.pending=false;s.xview=false;
    if(!resolving_pending)apply_ready_events();   // after this slot's writebacks; inside resolve_pending its own copy runs first
}
Slot &recording_slot(){
    Slot &s=slots[slot_index];
    if(!s.recording){
        {SiteScope site(SITE_RING,true);retire_slot(s);}
        // From here on this slot reuses its readback arena while the other slot, still in flight, may read it through pending
        // views (xview): its GPU reads then race the CPU copies of this submission.
        {const Slot &o=slots[slot_index^1];if(o.pending && o.xview)++xview_races;
         s.guard.clear();if(o.pending && xview_guard && !ab_off("xguard"))s.guard.swap(s.viewed);s.viewed.clear();s.pro_ops.clear();s.pro_images.clear();}
        // Pending writebacks still read this slot's readback arena; finish them before trimming.
        for(auto &w:pending_writebacks)if(w.slot==&s){++perf.why_slot;resolve_pending();break;}
        // Keep what this recording actually needed (3..8): during streamed bursts the arena grew to 7+ chunks
        // per frame, and trim() released them — every burst frame paid chunk creation + first-touch page faults inside the
        // timed staging copy (measured up to 7 ms/image). The readback arena below already self-sizes; the stream one matches.
        // Grow-only: shrinking per-frame defeated the point — normal frames use 1 chunk, and every next burst then re-created + first-touched
        // chunks again (measured unchanged 8ms/image staging over four burst frames 375-397ms apart). Bounded by 8 (256MB/slot),
        // and only while soft-decoded BC images keep re-uploading: once they stop, back to 3.
        // 保留这次录制真正用到的块数（3..8）：流式突发时 arena 会一帧长到 7+ 块，而 trim() 把它们全放了，
        // 于是每一突发帧都要在计时的 staging 拷贝里重新建块并吃首次访问的缺页（实测每张图最多 7 ms）。
        // 只增不减：按帧收缩等于没做——普通帧只用 1 块，下一个突发帧又得重新建块（连测四帧 375-397ms，staging 仍是 8ms/张）。
        // 上限 8 块（每槽 256MB），且只在软解 BC 图持续重传期间保留：停下来以后回到 3 块。
        if(soft_bc_reupload_ns && timing_clock()-soft_bc_reupload_ns<soft_bc_idle_ns)
            s.stream.keep_chunks=std::max({(size_t)3,std::min((size_t)8,s.stream.peak_chunks),s.stream.keep_chunks});
        else s.stream.keep_chunks=3;
        s.stream.trim();s.readback.keep_chunks=s.guard.empty()?1:s.readback.chunks.size();s.readback.trim();
        if(texel_memo_enabled && (slot_serial&15)<2)texel_evict(slot_index,false);   // this slot's copies: its previous submission retired
        if(!s.cmd){if(slot_pools_enabled){VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.queueFamilyIndex=cfg.queue_family;VK(vkCreateCommandPool(cfg.device,&cp,nullptr,&s.pool));}
            VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};a.commandPool=s.pool?s.pool:command_pool;a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;a.commandBufferCount=1;VK(vkAllocateCommandBuffers(cfg.device,&a,&s.cmd));}
        s.readback.cached=readback_cached;
        if(!s.fence){VkFenceCreateInfo f{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VK(vkCreateFence(cfg.device,&f,nullptr,&s.fence));}
        if(!s.timestamps && timestamp_bits){VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_TIMESTAMP;q.queryCount=2;VK(vkCreateQueryPool(cfg.device,&q,nullptr,&s.timestamps));}
        bind_cache_reset();if(s.pool){if(s.reset_queued){poolreset::finish(s);s.reset_queued=false;if(s.reset_result==VK_TIMEOUT)xbg_fatalf("The GPU did not finish this slot's work within %llu ms (fence wait timed out), so the device is hung",(unsigned long long)(std::max<uint64_t>(cfg.fence_timeout_ns,10000000000ull)/1000000ull));else VK(s.reset_result);}else VK(vkResetCommandPool(cfg.device,s.pool,0));}else{VK(vkResetCommandBuffer(s.cmd,0));if(s.pro)VK(vkResetCommandBuffer(s.pro,0));}VkCommandBufferBeginInfo b{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};b.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;VK(vkBeginCommandBuffer(s.cmd,&b));
        if(s.timestamps){vkCmdResetQueryPool(s.cmd,s.timestamps,0,2);vkCmdWriteTimestamp(s.cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,s.timestamps,0);}
        s.recording=true;s.ops=0;
        const uint64_t frame=stats.presents.load()+1;
        if(pass_prof_enabled && timestamp_bits){
            if(stats.presents.load()>=pass_prof_next && stats.presents.load()<pass_prof_until && pass_prof_frame!=frame){pass_prof_frame=frame;pass_prof_next=stats.presents.load()+pass_prof_period;pass_prof_queries=pass_prof_slots=pass_prof_skipped_slots=0;}
            if(frame==pass_prof_frame){
                if(pass_prof_slots<pass_prof_slot_limit){
                    if(!s.prof_pool){VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_TIMESTAMP;q.queryCount=pass_prof_query_limit;VK(vkCreateQueryPool(cfg.device,&q,nullptr,&s.prof_pool));}
                    vkCmdResetQueryPool(s.cmd,s.prof_pool,0,pass_prof_query_limit);s.prof=true;s.prof_used=0;s.prof_recs.clear();s.prof_open=-1;s.prof_skipped=0;s.prof_frame=frame;s.prof_slot=pass_prof_slots++;
                }else ++pass_prof_skipped_slots;
            }
        }
        // Order against work of earlier submissions on this queue.
        {VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;vkCmdPipelineBarrier(s.cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&b,0,nullptr,0,nullptr);barrier_seq=write_seq;}
        for(auto q:s.occlusion_pools)vkCmdResetQueryPool(s.cmd,q,0,256);
        for(auto q:s.timestamp_pools)vkCmdResetQueryPool(s.cmd,q,0,64);
    }
    return s;
}
// np629：把"当前槽已录的命令"作为一段立即提交（独立 fence），然后重开命令缓冲继续录。
// 关键语义：段之间 GPU 顺序执行（同一队列），CPU 不等中间段 —— 最终状态与单次提交等价；
// 差别只在"驱动看到多个小提交"。这就是二分的测法。
// np629: submit what the slot has recorded so far as one segment (own fence), then reopen the command
// buffer and keep recording. Segments execute in queue order, the CPU never waits on intermediate ones --
// the final state equals a single submission; only what the *driver* sees changes. That is the bisection.

// np705：丢段路径的确定性自检钩子（np702 的 DM_DEATHGUARD_TEST 同款思路——验证不靠等运气）。
//   DM_SPLIT_DROP_TEST=1：首个切段提交全部失败（含重试，见下）⇒ 丢段重建触发一次后解除
//     ⇒ 几秒内验证「【丢段重建】+ 进程存活 + 帧继续」。
//   DM_SPLIT_DROP_TEST=2：切段提交永远失败 ⇒ 连续丢段到上限 ⇒ 验证「queue_dead 判死落盘
//     （gfx-death.txt 带 why=队列死）+ FATAL」。建议配 DM_SPLIT_DROP_LIMIT=3 缩短爬行。
//   默认 0 = 关。测试失败是模拟的（不敲队列、不真实提交），fence 未 signal 无害——np703
//   已核对 split_fences 无等待方（只创建/存环/关停销毁）。
// np705: deterministic self-test hook for the drop path (same idea as np702's DM_DEATHGUARD_TEST --
//   verification must not depend on luck). =1: split submits fail until the first drop fires, then
//   disarm (verifies drop-and-rebuild + survival in seconds). =2: split submits always fail, driving
//   consecutive drops to the limit (verifies the queue-dead evidence exit). Default 0 = off. Simulated
//   failures never touch the queue; the unsignaled fence is harmless (np703: split_fences has no waiter).
static int split_drop_test(){static int v=0;static bool read=false;if(!read){read=true;const char *e=getenv("DM_SPLIT_DROP_TEST");if(e)v=atoi(e);}return v;}
static unsigned char split_test_fired=0;   // np705：mode 1 在首次丢段后置 1（解除钩子）
static VkResult split_submit_guarded(VkSubmitInfo *si,VkFence fence){
   const int mode=split_drop_test();
   if(mode==2 || (mode==1 && !split_test_fired))return VK_ERROR_DEVICE_LOST;
   deathguard::VkSite site(deathguard::SUBMIT_SPLIT);
   return vkQueueSubmit(cfg.queue,1,si,fence);
}
void submit_split_segment(){
    Slot &s=slots[slot_index];
    if(!s.recording || !s.cmd)return;
    unsigned saved=cur_op;cur_op=60;
    if(open_pass.cmd==s.cmd)end_pass(1);   // np629c：大 pass 内部也切（见切点注释）
    // 段尾：栅栏 + 时间戳（与 submit_slots 同款收尾，但不做 query 拷贝——那要等最后一发）
    g_barrier_site="split-submit";full_barrier(s.cmd);
    VK(vkEndCommandBuffer(s.cmd));
    VkFence fence=VK_NULL_HANDLE;
    {VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VK(vkCreateFence(cfg.device,&fi,nullptr,&fence));
     split_fences[split_fence_at++%64]=fence;}
    {VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&s.cmd;
     {CPU_SCOPE(Submit);
      cur_digest.serial=slot_serial;
      xbg_logf("【二分】段 %u 提交（draws=%u passes=%u）",split_seg,cur_digest.draws,cur_digest.passes);
      digest_ring[digest_at++&15u]=cur_digest;
      // np658：同款冷却（切段路径也要在恢复窗口内保持安静）。
      // np658: same cooldown (the split path must stay quiet during recovery too).
      if(unsigned cd=lost_cooldown_ms()){usleep(cd*1000u);}
      // np701：守卫只包住**单次**提交调用（重试之间的冷却睡眠不算站龄——56 次重试的合法序列远超阈值）。
      // np705：守卫随提交一起移进 split_submit_guarded()（三处提交点共用，语义不变）。
      // np705：提交走自检钩子（守卫移进钩子；测试模式不敲队列直接给失败）。
      // np705: submit via the self-test hook (the guard moved inside; test mode fails without touching the queue).
      VkResult sr;sr=split_submit_guarded(&si,fence);
      // np632b：切段路径同款重试（np632 实测主路径三次穿透全兜住，切段路径也要兜）。
      // np705：测试模式下跳过重试——模拟失败不该被 56 次重试拖住（自然 reset 武装的 8s 冷却会把
      //   每段拖到分钟级）；丢段重建的入口条件「重试全败」由首败直接达成，与 np661 死法同效。
      // np705: skip retries in test mode -- a simulated failure must not be dragged through 56 retries
      //   (natural resets arm the 8s cooldown, minutes per segment); reaching the drop path on first
      //   failure is the same "all retries failed" entry condition as the np661 death.
      if(sr!=VK_SUCCESS && retry_submit_enabled() && !split_drop_test()){
         for(int attempt=0;attempt<6 && sr!=VK_SUCCESS;++attempt){usleep(3000u<<attempt);
            if(unsigned cd=lost_cooldown_ms()){usleep(cd*1000u);}  // np661：同款冷却（切段路径）。
            sr=split_submit_guarded(&si,fence);}
         for(int attempt=0;attempt<50 && sr!=VK_SUCCESS;++attempt){usleep(100000u);
            if(unsigned cd=lost_cooldown_ms()){usleep(cd*1000u);}  // np661：同款冷却（切段路径长宽限）。
            sr=split_submit_guarded(&si,fence);}
         static int ok3=0;if(sr==VK_SUCCESS && ok3++<3)xbg_logf("【宽限】切段重试成功（段 %u）",split_seg);
      }
      if(sr!=VK_SUCCESS){
         // np703：**丢段重建**（待办 3，治 np661 型死法——重试 56 次全败后 FATAL 直接退出）。
         //   重试全败说明队列当下翻脸，但帧循环不该陪葬：本段已录命令整体丢弃（对 NVN 语义 =
         //   该段跳过，画面缺一块），arm 冷却（np658：恢复窗口内别再敲队列）、重置命令缓冲，
         //   落到下方公共"重开继续录"路径，帧照常推进。
         //   split_fences 无等待方（np703 全文件核对：只创建/存环/关停销毁），未 signal 的本段
         //   fence 不需要任何跳帧记账——11-03 待办里"fence 要走 vkResetFences 跳帧路径"的前提
         //   不成立。
         //   连续丢段到上限（DM_SPLIT_DROP_LIMIT，缺省 8；0=不限）= 队列真死 ⇒ 证据落盘
         //   （deathguard 的队列死出口）再 FATAL，把"无限爬行"变成"带现场的干净退出"。
         // np703: drop-and-rebuild. All retries failing means the queue turned hostile *right now*;
         //   the frame loop must not die with it: discard the recorded segment (NVN semantics: the
         //   segment is skipped, the frame loses a chunk), arm the cooldown (np658: stop knocking
         //   during the recovery window), reset the command buffer and fall through to the common
         //   reopen path below. split_fences has no waiter (checked: create/store/destroy only), so
         //   the unsignaled fence needs no skip-frame bookkeeping. Consecutive drops past the limit
         //   (DM_SPLIT_DROP_LIMIT, default 8; 0 = unlimited) mean the queue is truly dead: write the
         //   evidence (the deathguard's queue-dead exit) and FATAL -- a bounded crawl instead of an
         //   endless one.
         xbg_logf("【红】二分段 %u 提交失败 %d —— 驱动在本段翻脸，丢段重建（本段内容丢弃）",split_seg,int(sr));
         lost_arm();
         if(split_drop_test()==1)split_test_fired=1;   // np705：mode 1 首次丢段后解除钩子（只验一次）
         // np703b：reset 失败不 FATAL（np704 学费前移）。device lost 窗口里 vkResetCommandBuffer
         //   合法地可能返回 DEVICE_LOST，而 check() 对非 SUCCESS 一律 xbg_fatalf——那等于在
         //   "不想死"的路径上留了一脚自爆。这里失败只留痕继续：若 cmd 真的不可用，落到下方
         //   公共路径的 vkBeginCommandBuffer 会失败并以明确站点 FATAL（那才是合法判差点）。
         // np703b: don't FATAL on reset failure. vkResetCommandBuffer may legitimately return
         //   DEVICE_LOST in the storm window, and check() fatals on any non-SUCCESS -- a self-destruct
         //   on the exact path meant to survive. Log and continue; if the cmd is truly unusable the
         //   common path's vkBeginCommandBuffer fails with a clear site and that's the legal FATAL.
         {VkResult rr=vkResetCommandBuffer(s.cmd,0);if(rr!=VK_SUCCESS)xbg_logf("【宽限】丢段后 vkResetCommandBuffer 失败 %d——继续走重开路径",int(rr));}
         if(split_drop_limit() && ++split_drops>=split_drop_limit()){
            deathguard::queue_dead(split_seg,sr);
            xbg_fatalf("split segment %u dropped %u times in a row: %d (queue dead)",split_seg,unsigned(split_drops),int(sr));}
         static int dn=0;if(dn++<6 || dn%20==0)xbg_logf("【宽限】已连续丢段 %u 次（上限 %u），帧继续",unsigned(split_drops),split_drop_limit());
      }else split_drops=0;}   // 任一段提交成功即清零连续丢段计数（本行关 if 与 CPU_SCOPE，下一行关 SubmitInfo 块）
    }
    ++split_seg;cur_digest=SubDigest{};cur_digest.serial=slot_serial;
    // 重开命令缓冲继续录。**必须 bind_cache_reset()**：cache 记的是"旧 cmd 已绑 pipeline/描述符/视口"，
    // 新 cmd 里什么都不在——不清的话下一个 draw 直接拿空 pipeline 去 draw（真机读数：
    // `[rfs]pipeline is null` → `Draw internal fail` → 驱动判死，段 1 开始即挂）。
    // np629d: reset the bind cache after reopening -- it records "bound in the old cmd buffer"; the new one
    // starts empty, so the next draw would submit a null pipeline (device read: `[rfs]pipeline is null`).
    bind_cache_reset();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(s.cmd,&bi));
    cur_op=saved;
}
// Submit the recording slot. With wait, block until all submitted work is done.
void submit_slots(bool wait){
    Slot &s=slots[slot_index];
    if(s.recording){
        if(open_pass.cmd==s.cmd){unsigned saved=cur_op;cur_op=60;end_pass(2);cur_op=saved;}barrier_seq=0;
        // Occlusion results: GPU copies them into mapped memory (vkGetQueryPoolResults is slow on this driver).
        s.query_copied=0;if(s.occlusion_used){size_t used=s.occlusion_used;s.query_results=s.readback.alloc(used*8);
            for(size_t i=0;i*256<used;i++){uint32_t n=uint32_t(std::min<size_t>(256,used-i*256));vkCmdCopyQueryPoolResults(s.cmd,s.occlusion_pools[i],0,n,s.query_results.buffer,s.query_results.offset+i*256*8,8,VK_QUERY_RESULT_64_BIT);}
            VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};hb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;hb.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(s.cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&hb,0,nullptr,0,nullptr);
            s.query_copied=used;}
        if(s.timestamps)vkCmdWriteTimestamp(s.cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,s.timestamps,1);
        VK(vkEndCommandBuffer(s.cmd));
        VkCommandBuffer cbs[2]{s.cmd,VK_NULL_HANDLE};uint32_t ncb=1;
        if(!s.pro_ops.empty() || !s.pro_images.empty()){
            if(!s.pro){VkCommandBufferAllocateInfo pa{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};pa.commandPool=s.pool?s.pool:command_pool;pa.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;pa.commandBufferCount=1;VK(vkAllocateCommandBuffers(cfg.device,&pa,&s.pro));}
            VkCommandBufferBeginInfo pb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};pb.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;VK(vkBeginCommandBuffer(s.pro,&pb));
            // Earlier submissions first (the other slot still reads these ranges), then the copies, then this submission.
            VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};before.srcAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;before.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(s.pro,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&before,0,nullptr,0,nullptr);
            for(const auto &op:s.pro_ops){VkBufferCopy r{op.src_off,op.dst_off,op.size};vkCmdCopyBuffer(s.pro,op.src,op.dst,1,&r);}
            // gfx-upload-prologue：所有图像一道屏障转成传输目标、各自拷贝、再一道屏障回到 GENERAL。
            // gfx-upload-prologue: one barrier takes every image to a transfer destination, the copies, then one barrier back to GENERAL.
            if(!s.pro_images.empty()){
                std::vector<VkImageMemoryBarrier> ib(s.pro_images.size(),VkImageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER});
                for(size_t i=0;i<ib.size();i++){const auto &u=s.pro_images[i];auto &b=ib[i];b.oldLayout=u.from;b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;b.image=u.image;b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
                    b.subresourceRange={u.aspects,0,u.levels,0,u.layers};b.srcAccessMask=u.from==VK_IMAGE_LAYOUT_UNDEFINED?0:VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;}
                vkCmdPipelineBarrier(s.pro,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,uint32_t(ib.size()),ib.data());
                for(const auto &u:s.pro_images)vkCmdCopyBufferToImage(s.pro,u.buffer,u.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,uint32_t(u.copies.size()),u.copies.data());
                for(auto &b:ib){b.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;b.newLayout=VK_IMAGE_LAYOUT_GENERAL;b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;}
                vkCmdPipelineBarrier(s.pro,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,uint32_t(ib.size()),ib.data());
                s.pro_images.clear();}
            VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};after.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;after.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
            vkCmdPipelineBarrier(s.pro,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&after,0,nullptr,0,nullptr);
            VK(vkEndCommandBuffer(s.pro));cbs[0]=s.pro;cbs[1]=s.cmd;ncb=2;s.pro_ops.clear();}
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=ncb;si.pCommandBuffers=cbs;bool main_dropped=false;   // np726：主提交全败转丢帧重建时置位
        {CPU_SCOPE(Submit);
         // np611：**大提交的节奏**。要回答的是"那 44k draw 是一帧、还是加载期间攒下来的十几帧"：
         // 每次提交带上当时的 presents 计数 —— 两次大提交若 presents 相同，说明它们同属一帧（= 攒帧），
         // 若每次大提交 presents 都 +1，说明一帧就这么多（= 单帧太重）。只打前 30 次、之后每 50 次一次，有界。
         // np611: the cadence of big submissions. The question is whether those 44k draws are one frame or a dozen
         // frames accumulated during loading: each line carries the present counter at submit time -- two big
         // submissions with the same presents belong to one frame (accumulated), +1 each time means one frame is
         // genuinely that heavy. Bounded output: the first 30, then every 50th.
         if(cur_digest.draws>=3000){
            static uint64_t t0;static unsigned big_n;if(!t0)t0=timing_clock();
            if(big_n<30 || big_n%50==0)
                xbg_logf("[大提交] #%u serial=%llu t=%.2fs presents=%llu passes=%u draws=%u disp=%u",
                    big_n,(unsigned long long)cur_digest.serial,(timing_clock()-t0)/1e9,
                    (unsigned long long)stats.presents.load(),cur_digest.passes,cur_digest.draws,cur_digest.dispatches);
            ++big_n;
         }
         cur_digest.serial=slot_serial;digest_ring[digest_at++&15u]=cur_digest;cur_digest=SubDigest{};
         // np702：**判死链路自检钩子**。每轮验证的成本大头是"等随机死亡"（33 分钟不死就陪跑 33 分钟）；
         //   闭环判死链路不该靠运气：DM_DEATHGUARD_TEST=<秒> 让首次提交在 VkSite(SUBMIT) 守卫内故意停
         //   N 秒（>30 即可）⇒ deathguard 30 秒必打【判死】并落盘两份证据 ⇒ 3 分钟确定性闭环
         //   （装机 + 起客 + 停 35s + 拉文件）。upstream 的 DM_GFX_PIPE_CRASH_TEST 同款思路：
         //   用测试钩子制造现场，不等真实故障。默认关；触发一次后自动失效（armed→-1）。
         // np702: the death-evidence self-test hook. The expensive part of a verification round is waiting
         //   for a random death (a 33-minute survivor costs 33 minutes). Closing the evidence loop should not
         //   depend on luck: DM_DEATHGUARD_TEST=<s> stalls the first submit inside a VkSite(SUBMIT) guard for
         //   N seconds (>30) ⇒ the deathguard logs 【判死】 at 30 s and writes both files ⇒ a deterministic
         //   3-minute round (install + start + stall 35 s + pull files). Same idea as upstream's
         //   DM_GFX_PIPE_CRASH_TEST: make the scene with a test hook instead of waiting for the real fault.
         //   Off by default; disarms itself after firing (armed→-1).
         {static int armed=-2;if(armed==-2){const char *e=getenv("DM_DEATHGUARD_TEST");armed=e?atoi(e):0;if(armed>0)xbg_logf("【自检】DM_DEATHGUARD_TEST=%d：首次提交处停 %d 秒验证判死落盘",armed,armed);}
          if(armed>0){int secs=armed;armed=-1;deathguard::VkSite site(deathguard::SUBMIT);xbg_logf("【自检】开始停 %d 秒——deathguard 应在 30 秒时判死并落盘",secs);usleep(unsigned(secs)*1000000u);xbg_logf("【自检】停毕恢复正常；证据文件应在沙箱 files/gfx-death.txt / gfx-lastframe.txt");}}
         // np658：reset 恢复窗口内不敲队列（防 Submit commandlist fail 攒账被连坐杀）。
         // np658: keep quiet inside the driver recovery window (avoids piling up Submit commandlist fail).
         if(unsigned cd=lost_cooldown_ms()){usleep(cd*1000u);}
         // np701：守卫只包住单次 vkQueueSubmit（重试之间的睡眠不算站龄）。
         // np701: the guard wraps ONE vkQueueSubmit (sleeps between retries are not site age).
         // np726：主路径自检钩子（确定性验证，不等随机风暴——np702/np705 同款思路）。
         //   DM_SPLIT_DROP_TEST=3：主提交**模拟**失败一次后解除（不碰队列）⇒ 验证主路径丢帧重建：
         //   丢帧日志 + np633 fence-skip 接住 + 下帧内联池重置 + 帧继续。测试轮禁用重试（模拟失败
         //   不该被 56 次重试拖住），与切段路径 mode 1/2 同款语义。默认关。
         // np726: main-path self-test hook (deterministic verification, same idea as np702/np705).
         //   DM_SPLIT_DROP_TEST=3: simulate ONE main-submit failure then disarm (queue untouched) to
         //   verify the main-path drop-and-continue: drop log + np633 fence-skip + inline pool reset
         //   next frame + the loop keeps going. Retries disabled in test mode (a simulated failure must
         //   not be dragged through 56 retries), same semantics as split modes 1/2. Off by default.
         bool main_simulate=false;
         {static bool main_test_hit=false;
          if(split_drop_test()==3 && !main_test_hit){main_test_hit=true;main_simulate=true;
             xbg_logf("【自检】DM_SPLIT_DROP_TEST=3：主提交模拟失败一次——验证主路径丢帧重建（np633 应接住 fence，帧继续）");}}
         VkResult sr;{deathguard::VkSite site(deathguard::SUBMIT);
            sr=main_simulate?VK_ERROR_DEVICE_LOST:vkQueueSubmit(cfg.queue,1,&si,s.fence);}
         // np631：**冷路径首提交重试**。vkQueueSubmit -4 时等 3ms 重试一次（默认开，DM_RETRY_SUBMIT=0 关）。
         // 依据：① 驱动自带恢复（系统日志 "Gpu is reset when queue submit. Recover queue error now."）；
         //       ② np630run3 的模式：挂点总在"新类型工作第一次提交"处 ⇒ 冷路径首次分配失败最像，
         //          恢复后重试即过。判据：重试成功且游戏推进越过原挂点 = 证实；重试也失败 = 原路 FATAL。
         if(sr!=VK_SUCCESS && retry_submit_enabled() && split_drop_test()!=3){   // np726：测试轮禁用重试（模拟失败直达丢帧重建）
            // np631b：指数退避多次重试（3→10→30→100ms，最多 6 次）。首跑读数：3ms 一次能过一次
            // （serial=14968）但第二次不够（serial=14974）——驱动恢复时间不固定，退避到 100ms 兜住。
            static int rn=0;if(rn++<6)xbg_logf("【重试】vkQueueSubmit %d，开始指数退避（serial=%llu）",int(sr),(unsigned long long)slot_serial);
            for(int attempt=0;attempt<6 && sr!=VK_SUCCESS;++attempt){
                usleep(3000u<<attempt);   // 3ms,6ms,12ms,24ms,48ms,96ms —— 指数退避，6 次累计约 189ms
                // np661：重试的每次尝试也在冷却窗内 ⇒ 同样先睡掉剩余再敲（np660 复盘：CQ 失败从
                // 37 降到 14 但没归零，剩余全来自重试循环每 100ms 敲一次）。
                // np661: each retry attempt is also inside the cooldown window -- sleep the remainder first
                //   (np660: CQ failures dropped 37->14 but not zero; the rest came from the retry loop).
                if(unsigned cd=lost_cooldown_ms()){usleep(cd*1000u);}
                {deathguard::VkSite site(deathguard::SUBMIT);sr=vkQueueSubmit(cfg.queue,1,&si,s.fence);}
                static int ok=0;if(sr==VK_SUCCESS && ok++<6)xbg_logf("【重试】第 %d 次尝试成功（%ums）",attempt+1,3u<<attempt);
            }
            // np632：**长宽限**——6 次快退避全败（np631b 实测：驱动打 "Recover error end and return
            // device lost" 后仍可能需要更长时间恢复），不再 FATAL，改 100ms × 50 次（累计 5 秒）继续试。
            // 每 10 次报一行进度；全败才 FATAL（保持可观测）。这兜住的是"驱动 reset 流程超过 189ms"的场景。
            for(int attempt=0;attempt<50 && sr!=VK_SUCCESS;++attempt){
                usleep(100000u);
                // np661：长宽限同款（5 秒 × 每 100ms 一发 = 恢复窗口内最多 50 发，全部喂给记账）。
                // np661: same for the long grace loop (5 s x every 100 ms = up to 50 knocks inside the window).
                if(unsigned cd=lost_cooldown_ms()){usleep(cd*1000u);}
                {deathguard::VkSite site(deathguard::SUBMIT);sr=vkQueueSubmit(cfg.queue,1,&si,s.fence);}
                static int ok2=0;if(sr==VK_SUCCESS && ok2++<3)xbg_logf("【宽限】长宽限第 %d 次尝试成功",ok2);
                else if(attempt%10==9)xbg_logf("【宽限】已等待 %d 次（%ums）仍 -4",attempt+1,(attempt+1)*100);
            }
         }
         if(sr!=VK_SUCCESS){
            // np610：设备丢失时把最近 16 次提交的形状打出来（见 SubDigest 的注释）
            // np610: on a lost device dump the shape of the last 16 submissions (see the SubDigest comment)
            xbg_logf("【红】vkQueueSubmit %d：最近 16 次提交摘要（volPass>0 表示那一帧有 volume 切片当渲染目标）",int(sr));
            for(unsigned k=0;k<16;k++){const SubDigest &d=digest_ring[(digest_at+k)&15u];if(!d.serial)continue;
                xbg_logf("  提交 serial=%llu passes=%u volPass=%u volAtt=%u draws=%u dispatch=%u",
                    (unsigned long long)d.serial,d.passes,d.volpasses,d.volatt,d.draws,d.dispatches);}
            // np726：主路径丢帧重建（待办 18；np720a/np722a 两次 FATAL 的站点）。np703/705 只把丢段
            //   重建接在切段路径，主路径（帧末收尾提交）重试全败仍直接 FATAL。语义与切段路径同款：
            //   本帧收尾提交整体丢弃（画面缺最后一截），arm 冷却（np658：恢复窗口内不敲队列），帧继续——
            //   ① fence 不 signal ⇒ retire_slot 的 np633 fence-skip（10s 超时→reset→当完成，且照常做
            //     本槽全部清理：writeback/查询/arena/描述符）接住；
            //   ② 本槽 cmd/pro 均未提交 ⇒ 池内无在飞命令 ⇒ **不排 poolreset**（排了 worker 会对同一
            //     不亮的 fence 再等 10s 超时 → reset_result=VK_TIMEOUT → 下帧录制入口 FATAL"device is
            //     hung"，FATAL 只是被挪一帧）；下帧 recording_slot 走内联 vkResetCommandPool；
            //   ③ 连续丢段/丢帧与切段路径共用 split_drops（切段成功或主提交成功都清零），到
            //     DM_SPLIT_DROP_LIMIT 上限 = 队列真死 ⇒ queue_dead 证据落盘再 FATAL（干净退出）。
            // np726: main-path frame drop-and-continue (todo 18; the site that FATALed in np720a/np722a).
            //   np703/705 wired drop-and-rebuild into the split path only; the end-of-frame submit still
            //   went straight to FATAL after all retries. Same semantics as the split path: discard this
            //   end-of-frame submission (the frame loses its tail), arm the cooldown, keep going --
            //   (1) the unsignaled fence is caught by retire_slot's np633 fence-skip (timeout -> reset ->
            //   treated as done, full slot cleanup still runs); (2) nothing from this slot's pool is in
            //   flight (cmd/pro never submitted), so DON'T queue poolreset -- its worker would wait 10s on
            //   the same dead fence, set reset_result=VK_TIMEOUT and only move the FATAL one frame later
            //   (recording setup's "device is hung"); the next recording_slot resets the pool inline;
            //   (3) consecutive drops share split_drops with the split path (any successful submit clears
            //   it); past DM_SPLIT_DROP_LIMIT = queue truly dead: queue_dead evidence, then a clean FATAL.
            lost_arm();
            if(split_drop_limit() && ++split_drops>=split_drop_limit()){
                deathguard::queue_dead(split_seg,sr);
                xbg_fatalf("vkQueueSubmit(cfg.queue,1,&si,s.fence) failed: %d after %u consecutive drops (queue dead)",int(sr),unsigned(split_drops));}
            main_dropped=true;
            static int mn=0;if(mn++<6 || mn%20==0)xbg_logf("【宽限】主提交丢弃（np726 丢帧重建）：已连续丢 %u 次（上限 %u），帧继续",unsigned(split_drops),split_drop_limit());}
         else split_drops=0;   // np726：主提交成功也清零连续丢段计数（与切段路径"任一段成功即清零"同款）
        }++submit_by[sync_site];if(!main_dropped)poolreset::queue(s);
        s.recording=false;s.pending=true;s.covers=submitted;s.hud_frame=hud::frame.load(std::memory_order_relaxed);slot_index^=1;++perf.submits;snapshot_reset();++slot_serial;
        if(split_draws_env()>0){static int n;if(n++<8 && split_seg)xbg_logf("【二分】本帧共 %u 段",split_seg);split_seg=0;split_draw_count=0;}   // np629
        pending_events.insert(pending_events.end(),recording_events.begin(),recording_events.end());recording_events.clear();
    }
    if(wait){retire_slot(slots[slot_index]);retire_slot(slots[slot_index^1]);}
}
// Pipelines persist across launches (the driver validates the header and
// ignores incompatible data), so later launches skip shader compilation.
size_t pipelines_unsaved=0;uint64_t pipeline_save_frame=0;constexpr uint64_t compile_ns=2000000;
std::string pipeline_cache_path(){std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');return (cut==std::string::npos?dir:dir.substr(0,cut))+"/vk-pipeline-cache.bin";}
std::vector<uint8_t> load_pipeline_cache(){
    std::vector<uint8_t> data;FILE *f=fopen(pipeline_cache_path().c_str(),"rb");if(!f)return data;
    fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);
    if(n>32 && n<(256l<<20)){data.resize(size_t(n));if(fread(data.data(),1,data.size(),f)!=data.size())data.clear();}
    fclose(f);
    if(data.size()>=32){uint32_t header[4];memcpy(header,data.data(),16);if(header[1]!=1 || header[2]!=properties.vendorID || header[3]!=properties.deviceID || memcmp(data.data()+16,properties.pipelineCacheUUID,16))data.clear();}
    xbg_logf("NVN pipeline cache load bytes=%zu",data.size());return data;
}
// The cache is written by its own thread: a save serializes the whole cache (17.7 MB on the Redmi K50 Ultra) and writes
// it, which ran on the graphics thread every 300 frames while pipelines were new. Marker gfx-sync-cache-save = '1' saves on
// the graphics thread as before. Only real compiles count as unsaved: a creation under 2 ms came from the driver's cache,
// and with a warm cache each start counted its ~200 warm-up builds as new and rewrote the cache about six times.
bool sync_cache_save=false;std::thread cache_saver;std::atomic<bool> cache_save_busy{false};
void write_pipeline_cache(size_t unsaved){
    uint64_t t0=timing_clock();size_t n=0;bool ok=false;
    if(vkGetPipelineCacheData(cfg.device,pipeline_cache,&n,nullptr)==VK_SUCCESS && n){std::vector<uint8_t> data(n);
        if(vkGetPipelineCacheData(cfg.device,pipeline_cache,&n,data.data())==VK_SUCCESS){std::string path=pipeline_cache_path(),tmp=path+".tmp";
            if(FILE *f=fopen(tmp.c_str(),"wb")){ok=fwrite(data.data(),1,n,f)==n;ok=fclose(f)==0&&ok;if(ok)ok=rename(tmp.c_str(),path.c_str())==0;}}}
    xbg_logf("NVN pipeline cache save bytes=%zu new=%zu ok=%d ms=%.1f on=%s",n,unsaved,int(ok),double(timing_clock()-t0)/1e6,sync_cache_save?"graphics":"saver");
}
void cache_save_join(){if(cache_saver.joinable())cache_saver.join();}
void save_pipeline_cache(){
    if(!pipeline_cache || !pipelines_unsaved || cache_save_busy.load())return;
    warm::flush();const size_t unsaved=pipelines_unsaved;pipelines_unsaved=0;
    if(sync_cache_save){write_pipeline_cache(unsaved);return;}
    cache_save_join();cache_save_busy=true;
    cache_saver=std::thread([unsaved]{setpriority(PRIO_PROCESS,0,10);write_pipeline_cache(unsaved);cache_save_busy=false;});
}
// np693: NVN/GL 的裁剪空间是 -w..w。客户着色器按这个约定写（全屏 quad 就放在 z=-w），Vulkan 的 0..w 会把它们
//   整片裁掉。`DM_CLIP_NEG_ONE=0` 可关掉做 A/B；不设则跟随设备是否支持。
// np693: NVN/GL clip space is -w..w and the guest shaders use it (the full-screen quad sits at z=-w); Vulkan's
//   0..w clips that away. `DM_CLIP_NEG_ONE=0` turns it off for A/B; unset follows device support.
// np694: GL 的 depth clamp。设备没有 depth_clip_control 时的替代方案：把 z 的裁剪换成钳制，
//   这样 z=-w 的几何不再被整片裁掉。`DM_DEPTH_CLAMP=0` 关掉做 A/B；不设则跟随设备支持。
// np694: GL-style depth clamp. The fallback when the device lacks depth_clip_control: turn z clipping into
//   clamping so geometry at z=-w survives. `DM_DEPTH_CLAMP=0` disables it for A/B; unset follows support.
static bool depth_clamp_on(){
    static int v=-1;
    if(v<0){const char *e=getenv("DM_DEPTH_CLAMP");
        v = e && *e ? (*e=='1') : (dm_depth_clip_enable_on?1:0);
        xbg_logf("NVN caps depthClamp dev=%d use=%d",dm_depth_clip_enable_on,v);}
    return v!=0;
}
static bool clip_neg_one(){
    static int v=-1;
    if(v<0){const char *e=getenv("DM_CLIP_NEG_ONE");
        v = e && *e ? (*e=='1') : (dm_depth_clip_control_on?1:0);
        xbg_logf("NVN caps depthClipControl dev=%d use=%d",dm_depth_clip_control_on,v);}
    return v!=0;
}
extern bool tex_dirty;extern uint64_t tex_dirty_tracked,tex_dirty_checks,tex_dirty_changed;   // texture checks through the dirty tracker (get_image)
// 3D 画面加亮的读数（定义在 dispatch 前的 namespace bright）
// Readings of the 3D picture brightening (defined in namespace bright before dispatch)
namespace bright{extern bool enabled;extern uint64_t applied;extern float last_k;}
void perf_frame(){
    {texel_distinct+=texel_frame_keys.size();for(auto k:texel_frame_keys)if(texel_prev_keys.count(k))++texel_reused;texel_prev_keys.swap(texel_frame_keys);texel_frame_keys.clear();}
    if(completion_wait_enabled)fold_completion_wait_stats();
    if(perf_frame_timed)++perf_timed_frames;perf_frame_timed=perf_sample && stats.presents.load()%perf_sample==0;   // the next frame
    {bool was=ab_frame_on;ab_frame_on=ab_enabled();tls_gate=!ab_off("tlsgate");img_fast=!ab_off("imgfast");if(was!=ab_frame_on && ab_period>0){perf=Perf{};perf.window_start=timing_clock();set_cache_hits=set_cache_misses=0;return;}}
    pipelines_unsaved+=size_t(warm::compiled.exchange(0));
    if(pipelines_unsaved && stats.presents.load()>=pipeline_save_frame+300){pipeline_save_frame=stats.presents.load();save_pipeline_cache();}
    uint64_t now=timing_clock();if(!perf.window_start){perf.window_start=now;perf.frames=0;return;}
    // np639：**大帧立刻报一次**。原来固定每 60 帧一份，而"场景帧"每帧开 221 个 pass、只活 1–2 帧就挂，
    //   永远落不到报告窗口里（np635/np636 两轮白跑）。这一帧开了 >100 个 pass 就当场出报告。
    // np639: report immediately on a big frame. The report used to be fixed at every 60 frames, but the scene
    //   frame opens 221 passes and lives for 1-2 frames before the hang, so it never landed in a window
    //   (np635/np636 were wasted rounds). More than 100 passes opened in one frame reports right away.
    bool big_frame=false;
    {static uint64_t last_passes;uint64_t pnow=perf.passes;big_frame=(pnow>=last_passes && pnow-last_passes>100);last_passes=pnow;
     // np643：**把这一帧开 pass 的顺序打出来**（行程编码）。要回答的是：443 次开 pass 是在 54 个 framebuffer
     //   之间 A→B→A→B 抖动（= 我们规划有问题），还是有序遍历（= 游戏本来就这样）。
     // np643: dump this frame's pass-open order as run-length runs. The question: are the 443 opens thrashing
     //   A->B->A->B between 54 framebuffers (our planner's fault) or an ordered walk (the game's own design)?
     if(big_frame){
         unsigned i=0,runs=0;
         while(i<pass_seq_n){unsigned j=i+1;while(j<pass_seq_n && pass_seq[j].fb==pass_seq[i].fb)++j;
             if(runs<160)xbg_logf("PASSSEQ run=%u fb=%#llx %ux%u att=%u",j-i,(unsigned long long)pass_seq[i].fb,pass_seq[i].w,pass_seq[i].h,pass_seq[i].att);
             ++runs;i=j;}
         xbg_logf("PASSSEQ total_opens=%u runs=%u distinct_fb=%zu",pass_seq_n,runs,pass_open_fbs.size());
     }
     pass_seq_n=0;
     if(++perf.frames<60 && !big_frame)return;}
    double wall=double(now-perf.window_start)/1e6,f=double(perf.frames);
    if(poolreset::enabled){xbg_logf("NVN PERF poolreset queued=%.2f waits=%.2f wait_ms=%.3f",poolreset::queued/f,poolreset::waits/f,poolreset::wait_ns/1e6/f);poolreset::queued=poolreset::waits=poolreset::wait_ns=0;}
    const double tf=perf_timed_frames?double(perf_timed_frames):1.0;   // parts and descriptor timers come from the timed frames only
    xbg_logf("NVN PERF v=1 sync=%d frames=%llu wall_ms=%.1f fps=%.2f submit_cpu_ms=%.2f present_cpu_ms=%.2f gpu_ms=%.2f gpu_samples=%llu submits=%.2f waits=%.2f sync_wb=%llu sync_rb=%llu sync_q=%llu ops=%.1f folded=%.1f passes=%.1f draws=%.1f dispatches=%.1f why=w%llu/p%llu/a%llu(%llu,%llu,%llu,%llu,%llu,%llu|%llu,%llu,%llu,%llu)/c%llu/f%llu/s%llu views=%llu snap_hit=%llu snap_miss=%llu snap_mb=%.1f arena_peak_mb=%.1f pipelines=%zu layouts=%zu samplers=%zu passes=%zu images=%zu host_segments=%zu host_mb=%.1f",
        int(sync_mode),(unsigned long long)perf.frames,wall,f*1000.0/wall,double(perf.submit_cpu_ns)/1e6/f,double(perf.present_cpu_ns)/1e6/f,perf.gpu_samples?double(perf.gpu_ns)/1e6/f:-1.0,(unsigned long long)perf.gpu_samples,
        double(perf.submits)/f,double(perf.waits)/f,(unsigned long long)perf.sync_writeback,(unsigned long long)perf.sync_readback,(unsigned long long)perf.sync_query,double(perf.ops)/f,double(perf.folded_clears)/f,double(perf.passes)/f,double(perf.draws)/f,double(perf.dispatches)/f,
        (unsigned long long)perf.why_wait,(unsigned long long)perf.why_poll,(unsigned long long)perf.why_access,(unsigned long long)perf.access_by[1],(unsigned long long)perf.access_by[2],(unsigned long long)perf.access_by[3],(unsigned long long)perf.access_by[4],(unsigned long long)perf.access_by[5],(unsigned long long)perf.access_by[6],(unsigned long long)perf.access_by[8],(unsigned long long)perf.access_by[9],(unsigned long long)perf.access_by[10],(unsigned long long)perf.access_by[11],(unsigned long long)perf.why_cmdwait,(unsigned long long)perf.why_finish,(unsigned long long)perf.why_slot,(unsigned long long)perf.pending_views,(unsigned long long)perf.snapshot_hits,(unsigned long long)perf.snapshot_misses,double(perf.snapshot_bytes)/1048576.0,double(perf.arena_peak)/1048576.0,perf_counts().pipelines,perf_counts().layouts,perf_counts().samplers,perf_counts().passes,perf_counts().images,perf_counts().host_segments,double(perf_counts().host_bytes)/1048576.0);
    if(!timing_active){
        static const char *names[]={"other","draw","dispatch","descriptor","pass","buffer","pipeline","submit","fence","present","cleanup","hash","image","log","program","cpuwrite","snapshot"};
        char line[1024];int at=snprintf(line,sizeof(line),"NVN PERF parts_ms_per_frame timed=%llu",(unsigned long long)perf_timed_frames);
        for(unsigned i=0;i<XbgCpuTiming::Count && at<int(sizeof(line))-48;i++){const auto &c=timing.counters[i];if(c.calls<perf_timing_before[i].calls)perf_timing_before[i]=XbgCpuTiming::Counter{};uint64_t ns=c.exclusive_ns-perf_timing_before[i].exclusive_ns,calls=c.calls-perf_timing_before[i].calls;at+=snprintf(line+at,sizeof(line)-at," %s=%.2f/%.1f",names[i],double(ns)/1e6/tf,double(calls)/tf);perf_timing_before[i]=c;}
        xbg_logf("%s",line);
    }
    xbg_logf("NVN PERF storage_snapshots_per_frame=%.1f union_mb_per_frame=%.2f",storage_snap_count/f,storage_pages.size()*4096.0/1048576.0/tf);storage_pages.clear();storage_snap_count=0;
    xbg_logf("NVN PERF wtrack copied_kb_per_frame=%.1f dirty_pages_per_frame=%.1f mismatches=%llu fallbacks=%llu",wtrack::copied_bytes/1024.0/f,wtrack::dirty_pages/f,(unsigned long long)wtrack::mismatches,(unsigned long long)wtrack::fallbacks);wtrack::copied_bytes=wtrack::dirty_pages=0;
    xbg_logf("NVN PERF uniform_trimmed_bindings=%llu snap_mb_per_frame other=%.2f vertex=%.2f index=%.2f storage=%.2f texel=%.2f uniform=%.2f requested vertex=%.2f index=%.2f storage=%.2f",(unsigned long long)uniform_trimmed,snap_site_bytes[0]/1048576.0/f,snap_site_bytes[1]/1048576.0/f,snap_site_bytes[2]/1048576.0/f,snap_site_bytes[3]/1048576.0/f,snap_site_bytes[4]/1048576.0/f,snap_site_bytes[5]/1048576.0/f,snap_site_req[1]/1048576.0/f,snap_site_req[2]/1048576.0/f,snap_site_req[3]/1048576.0/f);for(auto &b:snap_site_bytes)b=0;for(auto &b:snap_site_req)b=0;
    xbg_logf("NVN PERF quad_rewrites=%llu warp_rewrites=%llu ldc_folds=%llu temps_forwarded=%llu nan_minmax=%llu",(unsigned long long)quad_rewrites.load(std::memory_order_relaxed),(unsigned long long)warp_rewrites.load(std::memory_order_relaxed),(unsigned long long)ldc_folds.load(std::memory_order_relaxed),(unsigned long long)temps_forwarded.load(std::memory_order_relaxed),(unsigned long long)nan_minmax_swaps.load(std::memory_order_relaxed));
    xbg_logf("NVN PERF snapfront front_hits=%.1f map_hits=%.1f invalidations=%.1f erased=%.1f",snap_front_hits/f,snap_map_hits/f,snap_invalidations/f,snap_invalidated/f);xbg_logf("NVN PERF textures bc_native=%llu bc_decoded=%llu live_images=%.1fMB bc_images=%.1fMB bc_payload=%.1fMB",(unsigned long long)native_bc_images,(unsigned long long)decoded_bc_images,image_bytes_live/1048576.0,bc_image_bytes_live/1048576.0,bc_data_bytes_live/1048576.0);snap_front_hits=snap_map_hits=snap_invalidations=snap_invalidated=0;
    if(bright::enabled){xbg_logf("NVN PERF bright applied=%.2f k=%.2f",bright::applied/f,bright::last_k);bright::applied=0;}
    if(tex_dirty){xbg_logf("NVN PERF texdirty tracked=%.1f checks=%.1f changed=%.1f",tex_dirty_tracked/f,tex_dirty_checks/f,tex_dirty_changed/f);tex_dirty_tracked=tex_dirty_checks=tex_dirty_changed=0;}
    if(vcache::enabled){xbg_logf("NVN PERF vcache hits=%.1f (%.2fMB) fills=%.1f (%.2fMB) refills=%.2f volatile=%llu untracked=%llu flushes=%llu entries=%zu arena=%.1fMB pools=%d",vcache::hits/f,vcache::hit_bytes/1048576.0/f,vcache::fills/f,vcache::fill_bytes/1048576.0/f,vcache::refills/f,(unsigned long long)vcache::volatiles,(unsigned long long)vcache::untracked,(unsigned long long)vcache::flushes,vcache::live(),vcache::arena_bytes()/1048576.0,[]{int live=0,used=0;uint64_t adds=0;if(nvn_dirty_slot_stats)nvn_dirty_slot_stats(&live,&used,&adds);return used;}());vcache::hits=vcache::hit_bytes=vcache::fills=vcache::fill_bytes=vcache::refills=0;}
    {static uint64_t pm=0,pe=0;xbg_logf("NVN PERF pipelines_new=%llu evicted=%llu warm_hits=%llu created=%llu create_ms=%.1f create_max_ms=%.1f warm_built=%llu warm_failed=%llu warm_ms=%.0f async_skips=%llu async_built=%llu",(unsigned long long)(pipeline_counts.misses-pm),(unsigned long long)(pipeline_counts.evictions-pe),(unsigned long long)warm::hits,(unsigned long long)warm::creates,warm::create_ns/1e6,warm::create_max_ns/1e6,(unsigned long long)warm::built.load(),(unsigned long long)warm::failed.load(),warm::build_ns.load()/1e6,(unsigned long long)warm::async_skips.exchange(0),(unsigned long long)warm::async_built.load());warm::hits=warm::creates=warm::create_ns=warm::create_max_ns=0;pm=pipeline_counts.misses;pe=pipeline_counts.evictions;}
    xbg_logf("NVN PERF setcache ab=%d hits=%.1f misses=%.1f template=%.1f plain=%.1f memo_hits=%.1f memo_misses=%.1f memo_front=%.1f why=texel%.1f/simg%.1f/dm%.1f/dynbuf%.1f/store%.1f",int(ab_frame_on),set_cache_hits/f,set_cache_misses/f,template_updates/f,plain_updates/f,desc_memo_hits/f,desc_memo_misses/f,memo_front_hits/f,desc_memo_why[0]/f,desc_memo_why[1]/f,desc_memo_why[2]/f,desc_memo_why[3]/f,desc_memo_why[4]/f);desc_memo_hits=desc_memo_misses=memo_front_hits=0;for(auto &w:desc_memo_why)w=0;xbg_logf("NVN PERF keydiff shader%.0f pool%.0f ubo%.0f uaddr%.0f size%.0f vp%.0f alpha%.0f drv%.0f tex%.0f samp%.0f saddr%.0f same%.0f new%.0f len%.0f",key_diff[0]/tf,key_diff[1]/tf,key_diff[2]/tf,key_diff[3]/tf,key_diff[4]/tf,key_diff[5]/tf,key_diff[6]/tf,key_diff[7]/tf,key_diff[8]/tf,key_diff[9]/tf,key_diff[10]/tf,key_diff[11]/tf,key_diff[12]/tf,key_diff[13]/tf);for(auto &k:key_diff)k=0;   // per timed frame
    xbg_logf("NVN PERF binds_skipped_per_frame=%.1f xmemo_hits=%.1f drops=%llu bumps=%llu/%llu/%llu/%llu/%llu pinned=%zu/%zu",binds_skipped/f,xmemo_hits/f,(unsigned long long)xmemo_drops,(unsigned long long)memo_bump[0],(unsigned long long)memo_bump[1],(unsigned long long)memo_bump[2],(unsigned long long)memo_bump[3],(unsigned long long)memo_bump[4],memo_pinned[0].size(),memo_pinned[1].size());binds_skipped=0;{char top[160]="";int at=0;bool used[32]={};
     for(int r=0;r<4;r++){int best=-1;for(int k=0;k<32;k++)if(!used[k] && offset_changed_at[k] && (best<0 || offset_changed_at[k]>offset_changed_at[best]))best=k;if(best<0)break;used[best]=true;at+=snprintf(top+at,sizeof(top)-size_t(at),"%s%d:%.1f",r?",":"",best,offset_changed_at[best]/f);}
     xbg_logf("NVN PERF binds set_binds=%.1f offsets_per_bind=%.1f offsets_only=%.1f changed n1=%.1f n2=%.1f n3=%.1f n4+=%.1f at=%s",set_binds/f,set_binds?double(set_bind_offsets)/double(set_binds):0.0,set_binds_offsets_only/f,offset_changes[1]/f,offset_changes[2]/f,offset_changes[3]/f,(offset_changes[4]+offset_changes[5]+offset_changes[6]+offset_changes[7]+offset_changes[8]+offset_changes[9])/f,top);}
    set_binds=set_bind_offsets=set_binds_offsets_only=0;for(auto &x:offset_changes)x=0;for(auto &x:offset_changed_at)x=0;xmemo_hits=0;xbg_logf("NVN PERF noload_passes_per_frame=%.2f img_cache_hits=%.1f misses=%.1f why=%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f",noload_passes/f,img_cache_hits/f,img_cache_misses/f,noload_why[0]/f,noload_why[1]/f,noload_why[2]/f,noload_why[3]/f,noload_why[4]/f,noload_why[5]/f,noload_why[6]/f,noload_why[7]/f);for(auto &w:noload_why)w=0;noload_passes=0;img_cache_hits=img_cache_misses=0;xbg_logf("NVN PERF desc_ns_per_frame memo_key=%.2f tmpl_update=%.2f alloc=%.2f setkey=%.2f build=%.2f hit_total=%.2f",desc_ns[0]/1e6/tf,desc_ns[1]/1e6/tf,desc_ns[2]/1e6/tf,desc_ns[3]/1e6/tf,desc_ns[4]/1e6/tf,desc_ns[5]/1e6/tf);for(auto &x:desc_ns)x=0;template_updates=plain_updates=0;set_cache_hits=set_cache_misses=0;
    xbg_logf("NVN PERF async=%d drains=%.2f drain_ms=%.2f pace_ms=%.2f waits=%.2f",int(gfx_async),gfx_drain_calls.exchange(0)/f,gfx_drain_ns.exchange(0)/1e6/f,gfx_pace_ns.exchange(0)/1e6/f,gfx_wait_calls.exchange(0)/f);xbg_logf("NVN PERF events defer=%d deferred=%.2f cpu_reads_pending=%llu epochs=%zu reports=%zu",int(defer_events),events_deferred/f,(unsigned long long)event_reads,epochs.size(),pending_reports.size());if(evguard::enabled){Dl_info di{};uintptr_t pc=evguard::last_pc.load();const char *mod=pc && dladdr(reinterpret_cast<void*>(pc),&di) && di.dli_fname?di.dli_fname:"?";xbg_logf("NVN PERF evguard guards=%llu exact=%llu other=%llu writes=%llu last_addr=%#llx last_pc=%#llx in %s",(unsigned long long)evguard::guards.load(),(unsigned long long)evguard::exact.load(),(unsigned long long)evguard::other.load(),(unsigned long long)evguard::writes.load(),(unsigned long long)evguard::last_addr.load(),(unsigned long long)pc,mod);
     xbg_logf("NVN PERF evguard engine_opens=%llu recloses=%llu",(unsigned long long)evguard::engine_opens.load(),(unsigned long long)evguard::recloses.load());
     xbg_logf("NVN PERF evguard last word=%#llx x0=%#llx x1=%#llx x2=%#llx x3=%#llx x4=%#llx x5=%#llx",(unsigned long long)evguard::last_word.load(),(unsigned long long)evguard::last_x[0].load(),(unsigned long long)evguard::last_x[1].load(),(unsigned long long)evguard::last_x[2].load(),(unsigned long long)evguard::last_x[3].load(),(unsigned long long)evguard::last_x[4].load(),(unsigned long long)evguard::last_x[5].load());
     for(int i=0;i<8;i++){uintptr_t lr=evguard::callers[i].load();if(!lr)break;Dl_info li{};const char *lm="?",*ls="?";uintptr_t base=0,sym=0;
         if(dladdr(reinterpret_cast<void*>(lr),&li)){if(li.dli_fname)lm=li.dli_fname;if(li.dli_sname)ls=li.dli_sname;base=uintptr_t(li.dli_fbase);sym=uintptr_t(li.dli_saddr);}
         xbg_logf("NVN PERF evguard caller lr=%#llx %s+%#llx (%s+%#llx) hits=%llu",(unsigned long long)lr,lm,(unsigned long long)(lr-base),ls,(unsigned long long)(sym?lr-sym:0),(unsigned long long)evguard::caller_hits[i].load());}}events_deferred=0;
    {static const char *site_names[SITE_COUNT]={"other","ring","present","event","arena","gpu_sync","complete","poll","legacy_wait"};
     char line[768];int at=snprintf(line,sizeof(line),"NVN PERF block_ms");
     for(int i=0;i<SITE_COUNT && at<int(sizeof(line))-40;i++)if(block_calls[i])at+=snprintf(line+at,sizeof(line)-at," %s=%.2f/%.2f",site_names[i],block_ns[i]/1e6/f,block_calls[i]/f);
     at+=snprintf(line+at,sizeof(line)-at," legacy_wait_total=%.2f submit_by",legacy_wait_ns/1e6/f);
     for(int i=0;i<SITE_COUNT && at<int(sizeof(line))-40;i++)if(submit_by[i])at+=snprintf(line+at,sizeof(line)-at," %s=%.2f",site_names[i],submit_by[i]/f);
     xbg_logf("%s",line);for(int i=0;i<SITE_COUNT;i++)block_ns[i]=block_calls[i]=submit_by[i]=0;legacy_wait_ns=0;}
    {char line[1536];int at=snprintf(line,sizeof(line),"NVN PERF pass_ends");
     for(int i=0;i<64 && at<int(sizeof(line))-24;i++)if(pass_end_by[i])at+=snprintf(line+at,sizeof(line)-at," op%d=%.1f",i,pass_end_by[i]/f);
     at+=snprintf(line+at,sizeof(line)-at," fb_hits=%.1f fb_creates=%.2f fb_live=%zu texel_binds=%.1f distinct=%.1f reused=%.1f resident=%zu/%zu hits=%.1f refresh=%.1f fallback=%.2f evict=%.2f occlusion=%.1f reports=%.1f samples=%.0f ops=%.1f/%.1f/%.1f pending=%zu pipe_fast=%.1f pass_fast=%.1f barriers",fb_hits/f,fb_creates/f,persistent_framebuffers.size(),texel_binds/f,texel_distinct/f,texel_reused/f,texel_resident[0].size(),texel_resident[1].size(),texel_hits/f,texel_refreshes/f,texel_fallbacks/f,texel_evictions/f,occlusion_queries_started/f,samples_reports/f,samples_reported/f,counter_ops[0]/f,counter_ops[1]/f,counter_ops[2]/f,pending_reports.size(),pipeline_fast_hits/f,pass_fast_hits/f);pipeline_fast_hits=0;pass_fast_hits=0;fb_hits=fb_creates=0;occlusion_queries_started=samples_reports=samples_reported=0;for(auto &c:counter_ops)c=0;texel_binds=texel_distinct=texel_reused=0;texel_hits=texel_refreshes=texel_fallbacks=texel_evictions=0;
     for(int i=0;i<64 && at<int(sizeof(line))-24;i++)if(barrier_by[i])at+=snprintf(line+at,sizeof(line)-at," op%d=%.1f",i,barrier_by[i]/f);
     for(unsigned i=0;i<barrier_site_n && at<int(sizeof(line))-48;i++)if(barrier_site_cnt[i])at+=snprintf(line+at,sizeof(line)-at," @%s=%.1f",barrier_site_name[i],barrier_site_cnt[i]/(double)f);
     at+=snprintf(line+at,sizeof(line)-at," @skip=%.1f @emit=%.1f",g_barrier_skip/(double)f,g_barrier_emit/(double)f);
     if(at<int(sizeof(line))-24)at+=snprintf(line+at,sizeof(line)-at," depth_idle=%.1f",depth_idle_passes/f);
     // np639：开新 pass 的四个原因（none 没有 pass 开着 / cmd 换命令缓冲 / rp 换 VkRenderPass / fb 换 framebuffer），
     //   以及窗口内"每帧开过多少个不同的 framebuffer"（fb_distinct 接近 pass 数 ⇒ 每个 pass 的附件身份都不同）。
     // np639: the four reasons a new pass is opened (none / cmd buffer / VkRenderPass / framebuffer) and how many
     //   distinct framebuffers were opened per frame (fb_distinct near the pass count means every pass has its own).
     at+=snprintf(line+at,sizeof(line)-at," pass_open none=%.1f cmd=%.1f rp=%.1f fb=%.1f fb_distinct=%.1f",
         pass_open_by[0]/f,pass_open_by[1]/f,pass_open_by[2]/f,pass_open_by[3]/f,double(pass_open_fbs.size())/f);
     // np641：pass 是被谁关的（1 切段 / 2 提交 / 3 非 draw 命令 / 4 图像转换 / 5 draw 换 pass / 6 回读 / 7 清屏）
     // np641: who closed the pass (1 split / 2 submit / 3 non-draw op / 4 image transition / 5 draw's pass switch / 6 readback / 7 clear)
     at+=snprintf(line+at,sizeof(line)-at," pass_end_by_cause 1=%.1f 2=%.1f 3=%.1f 4=%.1f 5=%.1f 6=%.1f 7=%.1f prep_read_end=%.1f",
         pass_end_cause[1]/f,pass_end_cause[2]/f,pass_end_cause[3]/f,pass_end_cause[4]/f,pass_end_cause[5]/f,pass_end_cause[6]/f,pass_end_cause[7]/f,prep_read_end/f);
     at+=snprintf(line+at,sizeof(line)-at," prep_read_feedback=%.1f",prep_read_feedback/f);
     xbg_logf("%s",line);for(int i=0;i<64;i++)pass_end_by[i]=barrier_by[i]=0;for(unsigned i=0;i<barrier_site_n;i++)barrier_site_cnt[i]=0;g_barrier_skip=g_barrier_emit=0;depth_idle_passes=0;
     for(int i=0;i<4;i++)pass_open_by[i]=0;pass_open_fbs.clear();for(int i=0;i<8;i++)pass_end_cause[i]=0;prep_read_end=0;prep_read_feedback=0;}
    xbg_logf("NVN PERF cleanup_ms_per_frame queries=%.2f writebacks=%.2f(%.1fMB) views=%.2f retire=%.2f arena_pools=%.2f",cleanup_ns[0]/1e6/f,cleanup_ns[1]/1e6/f,writeback_bytes/1048576.0/f,cleanup_ns[2]/1e6/f,cleanup_ns[3]/1e6/f,cleanup_ns[4]/1e6/f);for(auto &c:cleanup_ns)c=0;writeback_bytes=0;
    xbg_logf("NVN PERF xview views=%.2f races=%.2f guarded=%.2f guarded_kb=%.1f",xviews/f,xview_races/f,xview_guarded/f,xview_guarded_bytes/1024.0/f);xviews=xview_races=xview_guarded=xview_guarded_bytes=0;
    if(uploads_prologue || uploads_inpass){xbg_logf("NVN PERF uploads window prologue=%llu inpass=%llu broke_pass=%llu",(unsigned long long)uploads_prologue,(unsigned long long)uploads_inpass,(unsigned long long)uploads_broke_pass);uploads_prologue=uploads_inpass=uploads_broke_pass=0;}
    if(shaders_reused || shader_reuse_mismatches){xbg_logf("NVN PERF shader_reuse window reused=%llu mismatches=%llu",(unsigned long long)shaders_reused,(unsigned long long)shader_reuse_mismatches);shaders_reused=shader_reuse_mismatches=0;}
    if(wb_skip){xbg_logf("NVN PERF wbskip ab=%d skipped=%.1f (%.2fMB)",int(ab_frame_on),wb_skipped/f,wb_skipped_bytes/1048576.0/f);wb_skipped=wb_skipped_bytes=0;}
    if(pipe_memo){xbg_logf("NVN PERF pipememo ab=%d hits=%.1f misses=%.1f",int(ab_frame_on),pipe_memo_hits/f,pipe_memo_misses/f);pipe_memo_hits=pipe_memo_misses=0;}
    {double gm=perf.gpu_samples?double(perf.gpu_ns)/1e6/f:-1.0,cm=double(perf.submit_cpu_ns)/1e6/f;perf_gpu_ms.store(gm);perf_cpu_ms.store(cm);}
    perf=Perf{};perf.window_start=now;perf_timed_frames=0;
}
void occlusion_query(VkQueryPool &pool,uint32_t &index){
    Slot &s=recording_slot();index=s.occlusion_used%256;
    if(s.occlusion_used/256==s.occlusion_pools.size()){VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_OCCLUSION;q.queryCount=256;VkQueryPool p;VK(vkCreateQueryPool(cfg.device,&q,nullptr,&p));s.occlusion_pools.push_back(p);end_open_pass();vkCmdResetQueryPool(s.cmd,p,0,256);}
    pool=s.occlusion_pools[s.occlusion_used/256];{auto &e=epochs[counter_epoch];e.q.push_back({&s,pool,index,0,false});s.query_refs.push_back({counter_epoch,e.size()-1});}++s.occlusion_used;
}
void timestamp_query(VkQueryPool &pool,uint32_t &index,const void *&owner){
    Slot &s=recording_slot();index=s.timestamp_used%64;owner=&s;
    if(s.timestamp_used/64==s.timestamp_pools.size()){VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_TIMESTAMP;q.queryCount=64;VkQueryPool p;VK(vkCreateQueryPool(cfg.device,&q,nullptr,&p));s.timestamp_pools.push_back(p);end_open_pass();vkCmdResetQueryPool(s.cmd,p,0,64);}
    pool=s.timestamp_pools[s.timestamp_used/64];++s.timestamp_used;
}
// Synchronous point: everything recorded so far has completed on the GPU.
void gpu_sync(uint64_t &reason){++reason;SiteScope site(SITE_GPU_SYNC);submit_slots(true);}
void retire_handle(std::function<void()> f){recording_slot().retire.handles.push_back(std::move(f));}
template<class T>void retire_object(std::shared_ptr<T> p){recording_slot().retire.objects.push_back(std::move(p));}
Buffer stream_buffer(size_t n,const void *data=nullptr,size_t copy=0,bool zero=true){
    Buffer b=recording_slot().stream.alloc(n);
    if(data && copy){need(copy<=n,"Buffer copy overflow");evguard::EngineRead er(data,copy);memcpy(b.mapped,data,copy);}
    if(zero && copy<n)memset(static_cast<uint8_t*>(b.mapped)+copy,0,n-copy);
    b.flush();return b;
}
Buffer readback_buffer(size_t n,const void *data=nullptr,size_t copy=0){
    Slot &s=recording_slot();Buffer b=s.readback.alloc(n);
    for(const auto &g:s.guard)if(g.buffer==b.buffer && b.offset<g.end && g.begin<b.offset+n){
        if(data && copy)need(copy<=n,"Buffer copy overflow");
        Buffer src=stream_buffer(n,data,data?copy:0,true);s.pro_ops.push_back({src.buffer,b.buffer,src.offset,b.offset,VkDeviceSize(n)});
        ++xview_guarded;xview_guarded_bytes+=n;return b;}
    if(data && copy){need(copy<=n,"Buffer copy overflow");memcpy(b.mapped,data,copy);}else copy=0;
    if(copy<n)memset(static_cast<uint8_t*>(b.mapped)+copy,0,n-copy);   // only the bytes the copy leaves
    b.flush();return b;
}
// Per-submission snapshot cache of guest memory ranges. Guest command streams
// cannot change memory between draws except through in-stream CPU operations
// (uniform updates, buffer copies, events, writebacks), which invalidate it.
// Bump arena for the per-submission snapshot maps' nodes. The maps are dropped wholesale every submission: freeing ~3k nodes
// one by one with new/delete (scudo's locking) was ~3% of the render thread, and walking the trees to recycle them still
// ~1% of the D2R camp's on the Redmi K50 Ultra (__tree destroy under snapshot_reset). Nodes are never freed singly now:
// snapshot_reset builds both maps empty in place (keys and Buffer are trivially destructible, and the allocator frees
// nothing) and rewinds the arena. The maps are never destroyed, so no destructor at exit walks the rewound nodes.
// Graphics thread only (engine mutex).
struct SnapArena{
    static constexpr size_t chunk_bytes=256*1024;std::vector<uint8_t*> chunks;size_t chunk=0,used=0;
    void *alloc(size_t n){n=(n+15)&~size_t(15);need(n<=chunk_bytes,"Snapshot map node too large");
        if(chunks.empty() || used+n>chunk_bytes){if(!chunks.empty())++chunk;used=0;if(chunk==chunks.size())chunks.push_back(static_cast<uint8_t*>(::operator new(chunk_bytes)));}
        void *p=chunks[chunk]+used;used+=n;return p;}
    void rewind(){chunk=0;used=0;}
};
SnapArena &snap_arena(){static SnapArena *a=new SnapArena;return *a;}
template<class T> struct SnapAlloc {
    using value_type=T;SnapAlloc()=default;template<class U> SnapAlloc(const SnapAlloc<U>&){}
    T *allocate(size_t n){return static_cast<T*>(snap_arena().alloc(n*sizeof(T)));}
    void deallocate(T*,size_t){}   // snapshot_reset rewinds the arena
    template<class U> bool operator==(const SnapAlloc<U>&)const{return true;}template<class U> bool operator!=(const SnapAlloc<U>&)const{return false;}
};
using SnapMap=std::map<std::pair<uint64_t,uint64_t>,Buffer,std::less<std::pair<uint64_t,uint64_t>>,SnapAlloc<std::pair<const std::pair<uint64_t,uint64_t>,Buffer>>>;
static_assert(std::is_trivially_destructible<Buffer>::value,"snapshot_reset skips the map nodes' destructors");
SnapMap &snapshots=*new SnapMap(),&uniform_snapshots=*new SnapMap();
void snapshot_maps_reset(){new(&snapshots) SnapMap();new(&uniform_snapshots) SnapMap();snap_arena().rewind();}
uint64_t snapshot_maxlen=0;
void desc_memo_reset();
// Direct-mapped front for the two snapshot maps: most lookups repeat within a submission (~12k of 15k per D2R camp frame).
// Entries carry snapshot_epoch, which every reset bumps, so a stale entry never matches. Every front entry mirrors a map
// entry of the same key, so an invalidation clears just the front slots of the map entries it erases; bumping the epoch
// there (A/B target snapinval) emptied the whole front on every in-stream CPU write, overlapping or not.
struct SnapFront{uint64_t address=0,bytes=0,epoch=0;Buffer buffer;};SnapFront snap_front[2][2048];uint64_t snapshot_epoch=1;
uint64_t snap_front_hits=0,snap_map_hits=0,snap_invalidations=0,snap_invalidated=0;   // perf (NVN PERF snapfront)
inline SnapFront &snap_front_for(int which,uint64_t a,uint64_t n){uint64_t x=a*0x9E3779B97F4A7C15ull^n*0xC2B2AE3D27D4EB4Full;return snap_front[which][(x^(x>>31))&2047];}
void snapshot_reset(){snapshot_maps_reset();snapshot_maxlen=0;desc_memo_reset();++texel_write_epoch;++snapshot_epoch;last_pass_reset();}   // the guest may write memory between submissions: texel copies are re-verified
void snapshot_invalidate(uint64_t address,uint64_t bytes){
    ++snap_invalidations;if(ab_off("snapinval"))++snapshot_epoch;
    if(address<texel_hi && address+bytes>texel_lo)texel_touch(address,bytes);
    uint64_t lo=address>snapshot_maxlen?address-snapshot_maxlen:0;int which=0;
    for(auto *m:{&snapshots,&uniform_snapshots}){
        for(auto it=m->lower_bound({lo,0});it!=m->end() && it->first.first<address+bytes;){uint64_t b=it->first.first,n=it->first.second,e=b+n;
            if(address<e && address+bytes>b){SnapFront &f=snap_front_for(which,b,n);if(f.address==b && f.bytes==n)f.epoch=0;++snap_invalidated;it=m->erase(it);}else ++it;}
        ++which;}
}
// GPU-resident view of a guest range whose newest content is a pending
// (not yet written back) GPU result. The producing operation is followed by a
// full barrier, so later reads of the view observe it in queue order.
bool pending_view(uint64_t address,uint64_t bytes,VkDeviceSize align,Buffer &out){
    for(auto w=pending_writebacks.rbegin();w!=pending_writebacks.rend();++w){
        if(!(address<w->addr+w->bytes && address+bytes>w->addr))continue;
        if(address<w->addr || address+bytes>w->addr+w->bytes)return false;
        VkDeviceSize delta=address-w->addr;if(align>1 && (w->buffer.offset+delta)%align)return false;
        out=w->buffer;out.offset+=delta;out.mapped=static_cast<uint8_t*>(out.mapped)+delta;out.size=bytes;++perf.pending_views;
        if(w->slot!=&slots[slot_index]){++xviews;slots[slot_index].xview=true;
            const_cast<Slot*>(static_cast<const Slot*>(w->slot))->viewed.push_back({out.buffer,out.offset,out.offset+bytes});
            static unsigned n=0;if(n++<24)xbg_logf("NVN XVIEW frame=%llu addr=%#llx bytes=%llu writeback=%#llx+%llu",(unsigned long long)stats.presents.load(),(unsigned long long)address,(unsigned long long)bytes,(unsigned long long)w->addr,(unsigned long long)w->bytes);}
        return true;
    }
    return false;
}
// Write-tracked storage snapshots. Large read-only storage bindings (one 19.5 MB buffer per frame in the field,
// of which only ~7 pages change) used to be copied whole every frame. Guest pages are write-protected after a
// copy; a SIGSEGV on a tracked page marks it dirty and re-enables writes, so each frame copies only dirty pages
// into a persistent per-address buffer (two versions; a version is reused only once its frame slot retired).
// Every 240th use is verified against guest memory in full. On by default where the host routes faults here
// (xbg_wtrack_use_host_hooks); files/gfx-no-wtrack or gfx-wtrack = '0' disables it.
namespace wtrack {
constexpr size_t min_bytes=1u<<20;constexpr unsigned max_regions=8,versions=2;bool enabled=true;int marker=-1;   // marker: gfx-wtrack 1 / 0 (or gfx-no-wtrack), -1 = none
struct Region{std::atomic<uintptr_t> begin{0},end{0};std::atomic<uint8_t> *dirty=nullptr;};
Region regions[max_regions];std::atomic<unsigned> nregions{0};struct sigaction previous{};bool installed=false,host_hooks=false;
// Re-enable writes to the tracked pages in [a,a+len) and mark them dirty. Signal-safe: atomics and mprotect only.
bool unprotect(uintptr_t a,size_t len){if(!len || a>UINTPTR_MAX-len)return false;bool any=false;unsigned n=nregions.load(std::memory_order_acquire);
    for(unsigned i=0;i<n;i++){uintptr_t b=regions[i].begin.load(std::memory_order_relaxed),e=regions[i].end.load(std::memory_order_relaxed);
        uintptr_t lo=std::max(a,b),hi=std::min(a+len,e);if(lo>=hi)continue;size_t p0=(lo-b)>>12,p1=(hi-1-b)>>12;
        if(mprotect(reinterpret_cast<void*>(b+(p0<<12)),(p1-p0+1)<<12,PROT_READ|PROT_WRITE))continue;
        for(size_t pg=p0;pg<=p1;pg++)regions[i].dirty[pg].store(1,std::memory_order_release);any=true;}
    return any;}
void handler(int sig,siginfo_t *si,void *uc){
    if(unprotect(uintptr_t(si->si_addr),1))return;
    if(previous.sa_flags&SA_SIGINFO){if(previous.sa_sigaction){previous.sa_sigaction(sig,si,uc);return;}}
    else if(previous.sa_handler!=SIG_DFL && previous.sa_handler!=SIG_IGN && previous.sa_handler){previous.sa_handler(sig);return;}
    signal(sig,SIG_DFL); // return and re-fault into the default crash path
}
struct Version{VkBuffer buffer{};VkDeviceMemory memory{};uint8_t *mapped=nullptr;bool coherent=true;uint64_t last_use=0;std::vector<uint8_t> stale;};
struct Tracked{uint64_t address=0;size_t bytes=0;uintptr_t pbegin=0;size_t npages=0;unsigned region=0;Version v[versions];uint64_t uses=0;size_t verify_next=0;};
std::map<std::pair<uint64_t,size_t>,std::unique_ptr<Tracked>> tracked;bool exhausted=false;
uint64_t copied_bytes=0,dirty_pages=0,mismatches=0,fallbacks=0;
bool activate(Tracked &t){
    if(!installed){   // a host that routes faults to xbg_wtrack_fault marks this installed (xbg_wtrack_use_host_hooks)
        struct sigaction sa{};sa.sa_sigaction=handler;sa.sa_flags=SA_SIGINFO|SA_ONSTACK|SA_NODEFER;sigemptyset(&sa.sa_mask);if(sigaction(SIGSEGV,&sa,&previous)){enabled=false;return false;}installed=true;}
    unsigned n=nregions.load();if(n>=max_regions)return false;
    t.pbegin=uintptr_t(t.address)&~uintptr_t(4095);uintptr_t pend=(uintptr_t(t.address)+t.bytes+4095)&~uintptr_t(4095);
    for(unsigned i=0;i<n;i++){uintptr_t b=regions[i].begin.load(),e=regions[i].end.load();if(t.pbegin<e && pend>b)return false;} // overlapping regions unsupported
    // Pages the HLE's dirty-page tracker has ever protected (the vertex/index cache) are its for good: its fault hook runs
    // first and takes a fault on such a page as already handled, so a protection of ours there would re-fault forever.
    if(nvn_dirty_since && nvn_dirty_tracked)for(uintptr_t pg=t.pbegin;pg<pend;pg+=4096){const void *q=reinterpret_cast<const void*>(pg);if(nvn_dirty_tracked(q,1) && nvn_dirty_since(q,1,0)){xbg_logf("NVN wtrack region refused addr=%#llx: page %#llx belongs to the dirty-page tracker",(unsigned long long)t.address,(unsigned long long)pg);return false;}}
    size_t npages=(pend-t.pbegin)>>12;
    auto *dirty=new std::atomic<uint8_t>[npages];for(size_t i=0;i<npages;i++)dirty[i].store(0);
    for(auto &v:t.v){
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=t.bytes;ci.usage=arena_usage;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VK(vkCreateBuffer(cfg.device,&ci,nullptr,&v.buffer));
        VkMemoryRequirements r;vkGetBufferMemoryRequirements(cfg.device,v.buffer,&r);uint32_t type=find_memory(r.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        v.coherent=(memories.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)!=0;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=r.size;ai.memoryTypeIndex=type;VK(vkAllocateMemory(cfg.device,&ai,nullptr,&v.memory));VK(vkBindBufferMemory(cfg.device,v.buffer,v.memory,0));
        void *p=nullptr;VK(vkMapMemory(cfg.device,v.memory,0,VK_WHOLE_SIZE,0,&p));v.mapped=static_cast<uint8_t*>(p);v.stale.assign(npages,1);
    }
    regions[n].dirty=dirty;regions[n].begin.store(t.pbegin);regions[n].end.store(pend);nregions.store(n+1,std::memory_order_release);t.region=n;
    if(mprotect(reinterpret_cast<void*>(t.pbegin),pend-t.pbegin,PROT_READ)){regions[n].end.store(t.pbegin);return false;}
    t.npages=npages;xbg_logf("NVN wtrack region=%u addr=%#llx bytes=%zu pages=%zu",n,(unsigned long long)t.address,t.bytes,t.npages);return true;
}
void copy_page(Tracked &t,Version &v,size_t pg){
    uint64_t lo=std::max<uint64_t>(t.address,t.pbegin+(pg<<12)),hi=std::min<uint64_t>(t.address+t.bytes,t.pbegin+((pg+1)<<12));
    if(hi>lo){memcpy(v.mapped+(lo-t.address),reinterpret_cast<const void*>(uintptr_t(lo)),size_t(hi-lo));copied_bytes+=hi-lo;}
}
bool snapshot(uint64_t address,size_t bytes,Buffer &out){
    auto &slot=tracked[{address,bytes}];
    if(!slot){if(exhausted)return false;slot=std::make_unique<Tracked>();slot->address=address;slot->bytes=bytes;if(!activate(*slot)){slot->npages=0;exhausted=nregions.load()>=max_regions;}}
    Tracked &t=*slot;if(!t.npages)return false;
    // Clear the flag, re-protect, then copy: a write after the protect faults and re-marks the page.
    Region &r=regions[t.region];
    for(size_t pg=0;pg<t.npages;){
        if(!r.dirty[pg].exchange(0,std::memory_order_acq_rel)){++pg;continue;}
        size_t end=pg+1;while(end<t.npages && r.dirty[end].exchange(0,std::memory_order_acq_rel))++end;
        if(mprotect(reinterpret_cast<void*>(t.pbegin+(pg<<12)),(end-pg)<<12,PROT_READ)){enabled=false;xbg_logf("NVN wtrack mprotect failed errno=%d; disabled",errno);return false;}
        for(size_t i=pg;i<end;i++)for(auto &v:t.v)v.stale[i]=1;dirty_pages+=end-pg;pg=end;
    }
    Version *pick=nullptr;
    for(auto &v:t.v)if(v.last_use==slot_serial){pick=&v;break;}
    if(pick && std::find(pick->stale.begin(),pick->stale.end(),uint8_t(1))!=pick->stale.end())pick=nullptr; // changed within this slot: take another version
    if(!pick)for(auto &v:t.v)if(v.last_use+2<=slot_serial && (!pick || v.last_use>pick->last_use))pick=&v;
    if(!pick){++fallbacks;return false;}
    for(size_t pg=0;pg<t.npages;pg++)if(pick->stale[pg]){copy_page(t,*pick,pg);pick->stale[pg]=0;}
    // Spot check for writes the fault path missed: 16 pages per use, cycling through the range (a full pass of a 19.6 MB
    // buffer every ~300 uses, about the old rate). Reading the copy back is slow (arena memory on Adreno 730 reads at
    // ~1.1 GB/s), so the former check of the whole range every 240 uses read 19.6 MB in one frame, ~18 ms at that speed.
    ++t.uses;{size_t bad=0;for(unsigned k=0;k<16 && k<t.npages;k++){size_t pg=t.verify_next++%t.npages;uint64_t lo=std::max<uint64_t>(t.address,t.pbegin+(pg<<12)),hi=std::min<uint64_t>(t.address+t.bytes,t.pbegin+((pg+1)<<12));
            if(hi>lo && memcmp(pick->mapped+(lo-t.address),reinterpret_cast<const void*>(uintptr_t(lo)),size_t(hi-lo))){++bad;copy_page(t,*pick,pg);}}
        if(bad){mismatches+=bad;xbg_logf("NVN wtrack MISMATCH addr=%#llx pages=%zu (untracked writes; resynced)",(unsigned long long)t.address,bad);}}
    if(!pick->coherent){VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};mr.memory=pick->memory;mr.size=VK_WHOLE_SIZE;VK(vkFlushMappedMemoryRanges(cfg.device,1,&mr));}
    pick->last_use=slot_serial;out=Buffer{};out.buffer=pick->buffer;out.memory=pick->memory;out.offset=0;out.size=bytes;out.mapped=pick->mapped;out.coherent=pick->coherent;return true;
}
// Renderer shutdown, once the queue is idle (from main's 65721a2): the tracked guest pages become writable again and the
// tracked copies are freed, so a restarted renderer tracks afresh; a fault handler the engine installed itself is replaced by
// the previous one, while host fault hooks stay with the host. Tracking stays enabled for the next start.
void release(){
    for(unsigned i=0;i<nregions.load();i++){
        uintptr_t b=regions[i].begin.load(),e=regions[i].end.load();
        if(e>b)need(mprotect(reinterpret_cast<void*>(b),e-b,PROT_READ|PROT_WRITE)==0,"Restore tracked guest pages failed");
        delete[] regions[i].dirty;regions[i].dirty=nullptr;regions[i].begin=0;regions[i].end=0;
    }
    nregions=0;
    for(auto &t:tracked)for(auto &v:t.second->v){
        if(v.mapped)vkUnmapMemory(cfg.device,v.memory);
        if(v.buffer)vkDestroyBuffer(cfg.device,v.buffer,nullptr);
        if(v.memory)vkFreeMemory(cfg.device,v.memory,nullptr);
    }
    tracked.clear();exhausted=false;
    if(installed && !host_hooks){sigaction(SIGSEGV,&previous,nullptr);installed=false;}
}
}
// Host fault hooks for write tracking. A host with one SIGSEGV handler of its own (dm crash.c) calls
// xbg_wtrack_use_host_hooks() once and routes faults here instead of letting the engine install a handler;
// xbg_wtrack_prepare is for host writes into guest memory that do not fault (a syscall gets EFAULT).
// With the host owning the fault handler, write tracking is on unless a marker turns it off (Redmi K50 Ultra camp A/B: CPU
// -1.15 ms per frame, no mismatches); without it, it stays opt-in (gfx-wtrack = '1') with the engine's own handler.
extern "C" void xbg_wtrack_use_host_hooks(void){wtrack::installed=wtrack::host_hooks=true;if(wtrack::marker!=0)wtrack::enabled=true;xbg_logf("NVN wtrack through host hooks enabled=%d",int(wtrack::enabled));}
extern "C" int xbg_wtrack_fault(void *addr){return wtrack::unprotect(uintptr_t(addr),1);}
extern "C" void xbg_wtrack_prepare(void *addr,size_t len){if(len)wtrack::unprotect(uintptr_t(addr),len);}
// Texture content checks through the write-protect tracker (nvn_dirty.c), as nvn_mtl.m tex_mtl does on iOS: once a texture
// has been unchanged for 4 checks its pages are write-protected with the tracker's stamp, and a check then hashes it only if
// the guest wrote its pages since. The hash checks re-hashed every bound texture on its 16th submission (~0.27 ms of the
// render thread per D2R fight frame on the Redmi K50 Ultra) and saw a streamed write up to 16 submissions late. Untracked
// pools keep them. Marker gfx-tex-dirty = 1 / 0, else DM_GFX_TEX_DIRTY (the Android app sets 1); A/B target texdirty;
// NVN PERF texdirty: textures that started tracking, checks that hashed after a write, changes found.
bool tex_dirty=false;uint64_t tex_dirty_tracked=0,tex_dirty_checks=0,tex_dirty_changed=0;
// Cross-frame copies of vertex and index ranges (marker gfx-vcache = 1 / 0, else DM_GFX_VCACHE, which the Android app sets
// to 1; A/B target vcache). In the D2R camp 99% of the 6.4 MB of
// vertex data and all 1.45 MB of index data a frame copies are the same as the frame before (diagnostic gfx-snap-census).
// A range is copied once into a persistent arena and reused while the HLE's dirty-page tracker reports no write to its
// pages since the copy: take the stamp, write-protect the pages, then copy, so any later write faults and is newer than the
// stamp. Copies are never rewritten (a changed range gets a new one), so in-flight submissions keep reading theirs. A range
// that changes again within 8 submissions of being copied is volatile: it is left to the per-submission snapshot for 600
// submissions, and its pages are no longer protected. Ranges overlapping write-tracked storage are skipped and write tracking
// refuses pages this tracker has protected: two trackers must never share a page (each would take the other's faults).
// Entries in a deleted pool are dropped, and past the budget the whole arena is handed to the retiring slot and the cache
// starts over.
namespace vcache {
bool enabled=false;uint64_t hits=0,hit_bytes=0,fills=0,fill_bytes=0,refills=0,volatiles=0,untracked=0,flushes=0;
constexpr size_t min_bytes=256;constexpr VkDeviceSize budget=64ull<<20;constexpr uint64_t volatile_window=8,retry_after=600;
enum State:uint8_t{EMPTY,CACHED,VOLATILE,UNTRACKED};
struct Entry{Buffer buffer;uint64_t stamp=0,filled=0,retry=0;State state=EMPTY;};
struct KeyHash{size_t operator()(const std::pair<uint64_t,uint64_t> &k)const{uint64_t x=k.first*0x9E3779B97F4A7C15ull^k.second*0xC2B2AE3D27D4EB4Full;return size_t(x^(x>>29));}};
std::unordered_map<std::pair<uint64_t,uint64_t>,Entry,KeyHash> map;Arena arena;
size_t live(){return map.size();}
VkDeviceSize arena_bytes(){return arena.used_total;}
bool overlaps_wtrack(uint64_t a,uint64_t n){unsigned k=wtrack::nregions.load(std::memory_order_acquire);
    for(unsigned i=0;i<k;i++){uint64_t b=wtrack::regions[i].begin.load(std::memory_order_relaxed),e=wtrack::regions[i].end.load(std::memory_order_relaxed);if(a<e && a+n>b)return true;}return false;}
void flush(){if(arena.chunks.empty() && map.empty())return;std::vector<ArenaChunk> old;old.swap(arena.chunks);arena.current=0;arena.used_total=0;arena.peak=0;map.clear();++flushes;
    retire_handle([old]()mutable{for(auto &c:old)Arena::destroy(c);});}   // after the recording slot retires, no submission reads them
void drop_pool(uint64_t base,uint64_t bytes){for(auto it=map.begin();it!=map.end();){if(it->first.first<base+bytes && it->first.first+it->first.second>base)it=map.erase(it);else ++it;}}
bool lookup(uint64_t address,size_t bytes,Buffer &out){
    if(bytes<min_bytes || !nvn_dirty_since || !nvn_dirty_tracked || !nvn_dirty_protect || !nvn_dirty_now)return false;
    const void *p=reinterpret_cast<const void*>(uintptr_t(address));Entry &e=map[{address,uint64_t(bytes)}];
    if(e.state==VOLATILE || e.state==UNTRACKED){if(slot_serial<e.retry)return false;e.state=EMPTY;}
    if(e.state==CACHED){
        if(!nvn_dirty_since(p,bytes,e.stamp)){++hits;hit_bytes+=bytes;out=e.buffer;return true;}
        if(slot_serial-e.filled<volatile_window){e.state=VOLATILE;e.retry=slot_serial+retry_after;++volatiles;return false;}
        ++refills;
    }
    if(!nvn_dirty_tracked(p,bytes) || overlaps_wtrack(address,bytes)){e.state=UNTRACKED;e.retry=slot_serial+retry_after;++untracked;return false;}
    if(arena.used_total+bytes>budget)flush();
    Entry &f=map[{address,uint64_t(bytes)}];   // flush() may have cleared the map
    uint64_t stamp=nvn_dirty_now();nvn_dirty_protect(p,bytes);
    Buffer b=arena.alloc(bytes);memcpy(b.mapped,p,bytes);b.flush();
    f.buffer=b;f.stamp=stamp;f.filled=slot_serial;f.state=CACHED;++fills;fill_bytes+=bytes;out=b;return true;
}
}
Buffer snapshot_slow(uint64_t address,size_t bytes,VkDeviceSize align);
Buffer snapshot(uint64_t address,size_t bytes,VkDeviceSize align=4){CPU_SCOPE(Snapshot);
    SnapFront &f=snap_front_for(0,address,bytes);if(f.epoch==snapshot_epoch && f.address==address && f.bytes==bytes){++perf.snapshot_hits;++snap_front_hits;return f.buffer;}
    Buffer b=snapshot_slow(address,bytes,align);f.address=address;f.bytes=bytes;f.epoch=snapshot_epoch;f.buffer=b;return b;
}
Buffer snapshot_slow(uint64_t address,size_t bytes,VkDeviceSize align){
    auto key=std::make_pair(address,uint64_t(bytes));auto it=snapshots.find(key);
    if(it!=snapshots.end()){++perf.snapshot_hits;++snap_map_hits;return it->second;}
    {Buffer view;if(pending_view(address,bytes,align,view)){snapshots.emplace(key,view);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,bytes);return view;}}
    ++perf.snapshot_misses;perf.snapshot_bytes+=bytes;snap_site_req[snap_site<6?snap_site:0]+=bytes;if(snap_site==3){static unsigned logged=0;if(logged<40 && bytes>=65536){++logged;xbg_logf("NVN PERF storage_snapshot addr=%#llx bytes=%zu frame=%llu",(unsigned long long)address,bytes,(unsigned long long)stats.presents.load());}++storage_snap_count;if(timing_now())for(uint64_t pg=address>>12;pg<=(address+bytes-1)>>12;pg++)storage_pages.insert(pg);}   // page union: timed frames only (~4.8k inserts per D2R frame)
    cpu_access(address,bytes,1);   // a pending GPU result that only partly covers the range (or is misaligned for a view) lands first
    if(snap_site==3 && bytes>=wtrack::min_bytes && wtrack::enabled && !ab_off("wtrack")){Buffer t;if(wtrack::snapshot(address,bytes,t)){snapshots.emplace(key,t);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,bytes);return t;}}
    if((snap_site==1 || snap_site==2) && vcache::enabled && !ab_off("vcache")){Buffer v;if(vcache::lookup(address,bytes,v)){snapshots.emplace(key,v);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,bytes);return v;}}
    snap_site_bytes[snap_site<6?snap_site:0]+=bytes;
    Buffer b=stream_buffer(bytes,reinterpret_cast<const void*>(address),bytes,false);snapshots.emplace(key,b);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,bytes);return b;
}
struct Commands {
    VkCommandBuffer command{};bool draw=false;
    // Non-draw operations keep the former submit-per-operation semantics: they
    // end any open pass and are fenced by full barriers before and after.
    explicit Commands(bool is_draw=false):draw(is_draw){Slot &s=recording_slot();command=s.cmd;++perf.ops;++s.ops;
        if(!draw){end_pass(3);g_barrier_site="finish";full_barrier(command);}}
    void finish(){if(!draw){end_pass(3);g_barrier_site="finish";full_barrier(command);}}
};
VkFormat format(int f) {
    static const VkFormat table[]={VK_FORMAT_UNDEFINED,
        VK_FORMAT_R8_UNORM,VK_FORMAT_R8_SNORM,VK_FORMAT_R8_UINT,VK_FORMAT_R8_SINT,
        VK_FORMAT_R16_SFLOAT,VK_FORMAT_R16_UNORM,VK_FORMAT_R16_SNORM,VK_FORMAT_R16_UINT,VK_FORMAT_R16_SINT,
        VK_FORMAT_R32_SFLOAT,VK_FORMAT_R32_UINT,VK_FORMAT_R32_SINT,
        VK_FORMAT_R8G8_UNORM,VK_FORMAT_R8G8_SNORM,VK_FORMAT_R8G8_UINT,VK_FORMAT_R8G8_SINT,
        VK_FORMAT_R16G16_SFLOAT,VK_FORMAT_R16G16_UNORM,VK_FORMAT_R16G16_SNORM,VK_FORMAT_R16G16_UINT,VK_FORMAT_R16G16_SINT,
        VK_FORMAT_R32G32_SFLOAT,VK_FORMAT_R32G32_UINT,VK_FORMAT_R32G32_SINT,
        VK_FORMAT_R8G8B8_UNORM,VK_FORMAT_R8G8B8_SNORM,VK_FORMAT_R8G8B8_UINT,VK_FORMAT_R8G8B8_SINT,
        VK_FORMAT_R16G16B16_SFLOAT,VK_FORMAT_R16G16B16_UNORM,VK_FORMAT_R16G16B16_SNORM,VK_FORMAT_R16G16B16_UINT,VK_FORMAT_R16G16B16_SINT,
        VK_FORMAT_R32G32B32_SFLOAT,VK_FORMAT_R32G32B32_UINT,VK_FORMAT_R32G32B32_SINT,
        VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_R8G8B8A8_SNORM,VK_FORMAT_R8G8B8A8_UINT,VK_FORMAT_R8G8B8A8_SINT,
        VK_FORMAT_R16G16B16A16_SFLOAT,VK_FORMAT_R16G16B16A16_UNORM,VK_FORMAT_R16G16B16A16_SNORM,VK_FORMAT_R16G16B16A16_UINT,VK_FORMAT_R16G16B16A16_SINT,
        VK_FORMAT_R32G32B32A32_SFLOAT,VK_FORMAT_R32G32B32A32_UINT,VK_FORMAT_R32G32B32A32_SINT,
        VK_FORMAT_S8_UINT,VK_FORMAT_D16_UNORM,VK_FORMAT_X8_D24_UNORM_PACK32,VK_FORMAT_D32_SFLOAT,VK_FORMAT_D24_UNORM_S8_UINT,VK_FORMAT_D32_SFLOAT_S8_UINT};
    if(f>0 && f<int(sizeof(table)/sizeof(*table))) return table[f];
    // NVN ASTC 0x79-0x86 (LDR) / 0x87-0x94 (sRGB), order 4x4 5x4 5x5 6x5 6x6 8x5 8x6 8x8 10x5 10x6 10x8 10x10 12x10 12x12 = Vulkan order.
    if(f>=0x79 && f<=0x94){int k=(f-0x79)%14;return VkFormat(int(VK_FORMAT_ASTC_4x4_UNORM_BLOCK)+2*k+(f>=0x87?1:0));}
    // NVN BC 0x41-0x50: DXT1 RGB/RGBA, DXT3, DXT5, the same four sRGB, RGTC1 U/S, RGTC2 U/S, BPTC unorm/sRGB, BPTC sfloat/ufloat
    // (the decoder: 0x41 opaque BC1, 0x4f signed BC6H). Used as such only with native_bc; otherwise BC is decoded on upload.
    if(f>=0x41 && f<=0x50){static const VkFormat bc[16]={VK_FORMAT_BC1_RGB_UNORM_BLOCK,VK_FORMAT_BC1_RGBA_UNORM_BLOCK,VK_FORMAT_BC2_UNORM_BLOCK,VK_FORMAT_BC3_UNORM_BLOCK,
        VK_FORMAT_BC1_RGB_SRGB_BLOCK,VK_FORMAT_BC1_RGBA_SRGB_BLOCK,VK_FORMAT_BC2_SRGB_BLOCK,VK_FORMAT_BC3_SRGB_BLOCK,VK_FORMAT_BC4_UNORM_BLOCK,VK_FORMAT_BC4_SNORM_BLOCK,
        VK_FORMAT_BC5_UNORM_BLOCK,VK_FORMAT_BC5_SNORM_BLOCK,VK_FORMAT_BC7_UNORM_BLOCK,VK_FORMAT_BC7_SRGB_BLOCK,VK_FORMAT_BC6H_SFLOAT_BLOCK,VK_FORMAT_BC6H_UFLOAT_BLOCK};return bc[f-0x41];}
    // np601：0x69 起是 RGB10A2 家族（NVN 头里 NVN_FORMAT_RGB10A2SN=0x69 / RGB10A2I=0x6A / RGB10A2_UI2F=0x6B / RGB10A2_I2F=0x6C）。
    // 1.6.0 真机读到 TeraMeshRigidBody（物理网格）那一串文件时撞的就是 0x69。0x69/0x6A 的**存储**与 Vulkan 的
    // SNORM/UINT PACK32 一一对应，直接映射；0x6B/0x6C 是"整数存储 + 着色器里当浮点用"，需要着色器侧转换语义，
    // 不在这里糊弄（真撞上了再按需处理，宁可报清楚也不要静默画错）。
    // np601: the RGB10A2 family (NVN_FORMAT_RGB10A2SN=0x69, RGB10A2I=0x6A, RGB10A2_UI2F=0x6B, RGB10A2_I2F=0x6C).
    // On the 1.6.0 payload this is what the TeraMeshRigidBody (physics mesh) stream hit. 0x69/0x6A map 1:1 onto
    // Vulkan's SNORM/UINT PACK32 storage; 0x6B/0x6C need shader-side conversion, so they are left to fail loudly
    // instead of being approximated silently.
    switch(f){case 0x37:case 0x38:return VK_FORMAT_R8G8B8A8_SRGB;case 0x3d:return VK_FORMAT_A2B10G10R10_UNORM_PACK32;case 0x3e:return VK_FORMAT_A2B10G10R10_UINT_PACK32;case 0x3f:return VK_FORMAT_B10G11R11_UFLOAT_PACK32;case 0x40:return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case 0x69:return VK_FORMAT_A2B10G10R10_SNORM_PACK32;case 0x6a:return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case 0x6d:return VK_FORMAT_R8G8B8A8_UNORM;case 0x9a:return VK_FORMAT_B8G8R8A8_UNORM;default:xbg_fatalf("Unsupported NVN format %#x",f);}
}
VkImageAspectFlags aspect(int f){if(f==0x31)return VK_IMAGE_ASPECT_STENCIL_BIT;if(f>=0x32 && f<=0x34)return VK_IMAGE_ASPECT_DEPTH_BIT;if(f==0x35 || f==0x36)return VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT;return VK_IMAGE_ASPECT_COLOR_BIT;}
VkFormat decoded_format(uint32_t pixels) {
    switch(pixels){
    case XBG_RGBA8_UNORM:return VK_FORMAT_R8G8B8A8_UNORM;
    case XBG_RGBA8_SRGB:return VK_FORMAT_R8G8B8A8_SRGB;
    case XBG_R16_UNORM:return VK_FORMAT_R16_UNORM;
    case XBG_RG16_UNORM:return VK_FORMAT_R16G16_UNORM;
    case XBG_R16_SNORM:return VK_FORMAT_R16_SNORM;
    case XBG_RG16_SNORM:return VK_FORMAT_R16G16_SNORM;
    case XBG_RGBA16_SFLOAT:return VK_FORMAT_R16G16B16A16_SFLOAT;
    case XBG_R8_UNORM:return VK_FORMAT_R8_UNORM;
    case XBG_RG8_UNORM:return VK_FORMAT_R8G8_UNORM;
    default:xbg_fatalf("Invalid decoded BC pixel format %u",pixels);
    }
}
// Read-only capability report, logged once at initialization: what this device actually supports, so a missing capability can be
// named precisely (a Vulkan 1.1 driver lacks much of what 1.3 drivers report) instead of being read off the first fatal error.
void capability_report(){
    const VkPhysicalDeviceFeatures &f=cfg.enabled_core;const VkPhysicalDeviceLimits &l=properties.limits;
    xbg_logf("NVN caps device=\"%s\" vendor=%#x id=%#x api=%u.%u.%u driver=%#x",properties.deviceName,properties.vendorID,properties.deviceID,
        VK_API_VERSION_MAJOR(properties.apiVersion),VK_API_VERSION_MINOR(properties.apiVersion),VK_API_VERSION_PATCH(properties.apiVersion),properties.driverVersion);
    xbg_logf("NVN caps features drawParameters=%u storageReadWithoutFormat=%u storageWriteWithoutFormat=%u storageExtendedFormats=%u gatherExtended=%u "
        "tessellation=%u geometry=%u occlusionQueryPrecise=%u depthBounds=%u depthClamp=%u fillModeNonSolid=%u bc=%u astcLdr=%u etc2=%u "
        "anisotropy=%u logicOp=%u dualSrcBlend=%u independentBlend=%u clipDistance=%u cullDistance=%u float64=%u int64=%u int16=%u "
        "robustBufferAccess=%u",
        unsigned(cfg.enabled_shader_draw_parameters),f.shaderStorageImageReadWithoutFormat,f.shaderStorageImageWriteWithoutFormat,f.shaderStorageImageExtendedFormats,
        f.shaderImageGatherExtended,f.tessellationShader,f.geometryShader,f.occlusionQueryPrecise,f.depthBounds,f.depthClamp,f.fillModeNonSolid,
        f.textureCompressionBC,f.textureCompressionASTC_LDR,f.textureCompressionETC2,f.samplerAnisotropy,f.logicOp,f.dualSrcBlend,f.independentBlend,
        f.shaderClipDistance,f.shaderCullDistance,f.shaderFloat64,f.shaderInt64,f.shaderInt16,
        // np615：把 robustBufferAccess 的实际启用状态打进 caps（引擎注释里"越界读靠它"是个前提，值得留读数）
        unsigned(f.robustBufferAccess));
    xbg_logf("NVN caps subgroup size=%u stages=%#x ops=%#x quadAllStages=%u",subgroup.subgroupSize,subgroup.supportedStages,subgroup.supportedOperations,subgroup.quadOperationsInAllStages);
    xbg_logf("NVN caps limits ubRange=%u sbRange=%u ubDyn=%u sbDyn=%u ubAlign=%llu sbAlign=%llu texelAlign=%llu stageSamplers=%u stageSampled=%u "
        "stageStorageImages=%u stageUB=%u stageSB=%u boundSets=%u colorAttachments=%u sharedMemory=%u workgroupInvocations=%u viewports=%u maxAnisotropy=%g",
        l.maxUniformBufferRange,l.maxStorageBufferRange,l.maxDescriptorSetUniformBuffersDynamic,l.maxDescriptorSetStorageBuffersDynamic,
        (unsigned long long)l.minUniformBufferOffsetAlignment,(unsigned long long)l.minStorageBufferOffsetAlignment,(unsigned long long)l.minTexelBufferOffsetAlignment,
        l.maxPerStageDescriptorSamplers,l.maxPerStageDescriptorSampledImages,l.maxPerStageDescriptorStorageImages,l.maxPerStageDescriptorUniformBuffers,
        l.maxPerStageDescriptorStorageBuffers,l.maxBoundDescriptorSets,l.maxColorAttachments,l.maxComputeSharedMemorySize,l.maxComputeWorkGroupInvocations,
        l.maxViewports,double(l.maxSamplerAnisotropy));
    // Every core format up to ASTC 12x12 sRGB (BC, ETC2 and ASTC included) as vkformat:optimalTilingFeatures/bufferFeatures
    // (VkFormatFeatureFlagBits); formats with neither are left out.
    std::string line;unsigned n=0;
    for(int v=1;v<=int(VK_FORMAT_ASTC_12x12_SRGB_BLOCK);v++){const VkFormatProperties &p=format_properties(VkFormat(v));if(!p.optimalTilingFeatures && !p.bufferFeatures)continue;
        char item[40];snprintf(item,sizeof(item)," %d:%#x/%#x",v,p.optimalTilingFeatures,p.bufferFeatures);line+=item;
        if(++n%10==0){xbg_logf("NVN caps formats%s",line.c_str());line.clear();}}
    if(!line.empty())xbg_logf("NVN caps formats%s",line.c_str());
}
struct Image {
    tex_t source{}; uint64_t provenance_epoch=0,provenance_writer=0;unsigned provenance_write_site=0; VkImage image{};VkDeviceMemory memory{};VkFormat fmt{};VkImageLayout layout=VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspects{};VkImageUsageFlags usage{};uint32_t width{},height{},depth=1,levels{},layers{},bpp{};VkSampleCountFlagBits samples=VK_SAMPLE_COUNT_1_BIT;bool uploaded=false,gpu_written=false,decoded=false,mutable_format=false,vol_compat=false;VkImageCreateInfo create{}; // create: as created (no pNext), for upgrade_mutable
    uint64_t cpu_hash=0,checked_submission=UINT64_MAX,last_write=0;uint32_t stable_checks=0;int64_t mem_bytes=0,bc_bytes=0,bc_data=0;
    // 最近一次经 get_image 取用时的 slot_serial：等于当前值，说明这次提交里已有命令用过它（gfx-upload-prologue 据此判断能否提前上传）。
    // slot_serial at the last get_image of this image: equal to the current value means a command of this submission already used it
    // (gfx-upload-prologue decides by it whether an upload may go ahead of the submission).
    uint64_t slot_use=0;
    uint64_t dirty_stamp=0,dirty_seen=0;   // tex_dirty: the tracker stamp of the checked content (0: not tracked yet) and the tracker generation last checked
    std::map<std::array<uint32_t,8>,VkImageView> views;std::vector<std::vector<uint64_t>> fb_keys; // fb_keys: persistent framebuffers using these views
    ~Image(){image_bytes_live-=mem_bytes;bc_image_bytes_live-=bc_bytes;bc_data_bytes_live-=bc_data;forget_framebuffers(*this);for(auto &v:views)vkDestroyImageView(cfg.device,v.second,nullptr);if(image)vkDestroyImage(cfg.device,image,nullptr);if(memory)vkFreeMemory(cfg.device,memory,nullptr);}
    void transition(VkCommandBuffer cmd,VkImageLayout next){
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};b.oldLayout=layout;b.newLayout=next;b.image=image;b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange={aspects,0,levels,0,layers};b.srcAccessMask=layout==VK_IMAGE_LAYOUT_UNDEFINED?0:VK_ACCESS_MEMORY_WRITE_BIT|VK_ACCESS_MEMORY_READ_BIT;b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);layout=next;
    }
    VkFormat view_format(const view_t *v) const {
        // np623：视图格式等于**源格式**时走 fmt（原有逻辑）；另外接受**别名列**——nv 场景里
        // 同一张 BC 纹理会以 sRGB/非 sRGB 两种 NVN 枚举开视图（0x41↔0x45、0x42↔0x46、…、0x4d↔0x4e），
        // 引擎的 bc_linear_format 正是做这个归一。别名相等 ⇒ 与相等同义，不该报 FATAL。
        // np623: a view whose format equals the source format already goes through `fmt`; additionally accept
        // **aliases** -- the game opens views of the same BC texture in both sRGB and non-sRGB NVN enums
        // (0x41<->0x45 ... 0x4d<->0x4e); bc_linear_format normalises exactly this. An alias match is equivalent
        // to an exact match and must not FATAL.
        if(!v || !v->set_format || v->format==source.d.format)return fmt;
        if(decoded && source.d.format>=0x41 && source.d.format<=0x50 && v->format>=0x41 && v->format<=0x50
           && source.d.format!=v->format){
            // 先试别名放行（用源判定兼容，别用视图的），只有真的不兼容才落到下面的 need。
            if(xbg_bc_view_compatible(source.d.format,v->format)){ /* fall through to the decoded path below */ }
        }
        if(decoded){
            // np623：**两条合法路径**，对应两种真实场景——
            // ① BC→BC 视图（换 sRGB/线性别名，读压缩数据）：按 bc_linear_format 归一后必须同类；
            // ② 解码数据上的标量/向量视图（读解压结果）：BC5 的解码数据是双通道 16 位（RG），
            //    游戏开 RGBA16UI（0x2c）这类视图做**法线解压**——要求视图每像素字节数 == 解码格式的
            //    bytes_per_pixel（0x2c=RGBA16 是 8 字节？不对：按定义校验要看 bpp 一致才放行，
            //    不一致就报错并打读数。0x49 的解码 bpp=2（XBG_R16/RG16 族），0x2c 每像素 8 字节
            //    ⇒ 8≠2 会撞读数 ⇒ 先打读数再说，别猜。
            // np623: two legitimate paths. (1) BC->BC views (sRGB/linear alias of the compressed data) must stay
            // in the same bc_linear_format class. (2) Scalar/vector views of the **decoded** data: BC5's decoded
            // form is two 16-bit channels (RG), and the game opens e.g. RGBA16UI (0x2c) views for normal
            // decompression. Gate path 2 on matching bytes-per-pixel; if it does not match, FATAL with both
            // formats so the numbers speak instead of a guess.
            if(xbg_bc_view_compatible(source.d.format,v->format)){
                xbg_bc_format info;need(xbg_bc_format_info(v->format,&info)==0,"Invalid BC view format");
                return decoded_format(info.pixels);
            }
            xbg_bc_format src_info;need(xbg_bc_format_info(source.d.format,&src_info)==0,"Invalid BC source format");
            int bw=0,bh=0,vbytes=nvn_fmt_bpb(v->format,&bw,&bh);
            static int n;if(n++<4)xbg_logf("【黄】BC 解码源上的重解释视图 source=%#x view=%#x srcBpp=%u viewBpp=%d（放行，画面待对账）",source.d.format,v->format,unsigned(src_info.bytes_per_pixel),vbytes);
            return format(v->format);
        }
        if(source.d.format>=0x41 && source.d.format<=0x50){need(xbg_bc_view_compatible(source.d.format,v->format),"BC texture view must retain compression and signedness");return format(v->format);}   // native BC
        int bw,bh,bytes=nvn_fmt_bpb(v->format,&bw,&bh);
        need(aspects==VK_IMAGE_ASPECT_COLOR_BIT && aspect(v->format)==aspects && bw==1 && bh==1 && bytes==int(bpp),"Incompatible NVN texture view format class");
        return format(v->format);
    }
    VkImageView view(const view_t *v,bool attachment=false,bool storage=false){
        uint32_t base=v&&v->set_levels?v->base_level:0,n=xbg_view_nlevels(v,levels);
        // 3D 纹理的“层”是**深度切片**（L.layers 恒为 1，切片数在 d.depth）⇒ 视图的层号必须拿 depth 校验，
        // 否则游戏渲染到第 1 片时会被判“越界”。这也是 NVN 允许渲染进 volume 切片的前提。
        // A 3D texture's "layers" are depth slices (L.layers is always 1; the slice count lives in d.depth), so a
        // view's layer number must be validated against depth -- otherwise rendering into slice 1 reads as
        // out-of-range. This is also the premise for NVN allowing a render target inside a volume.
        const bool volume=source.d.target==2;const uint32_t span=volume?std::max(1u,depth):layers;
        // 未显式给层时：volume 仍是“1 层”（=第 0 片，与改动前 L.layers 等价）；数组纹理仍是整条。
        // With no explicit layer: a volume is still "one layer" (slice 0, equal to L.layers before this change);
        // an array texture still means the whole range.
        uint32_t layer=v&&v->set_layers?v->min_layer:0,ls=v&&v->set_layers?v->nlayers:(volume?1u:layers);
        need(base<levels && n && n<=levels-base && layer<span && ls && ls<=span-layer,"Invalid texture view range");
        // 采样/存储用的 3D 视图按规范 baseArrayLayer 必须是 0（只有附件的 2D 兼容视图才允许指到某一层）。
        // 这里保持改动前的口径：非附件视图一律从第 0 层起、单层，免得把“只对附件放宽”变成全局放宽。
        // A sampled/storage 3D view must start at slice 0 (only an attachment's 2D-compatible view may name a
        // slice), so keep the pre-change rule for non-attachment views: slice 0, one layer. The relaxation above
        // is meant for attachments only.
        if(volume && !attachment)need(layer==0 && ls<=1,"3D texture view must start at slice 0");
        if(attachment){
            // np626：**放行层叠渲染**（真机读数：2D_ARRAY 深度图 896×896、layerCount=3 一次渲三层，
            // 是阴影级联/环境贴图一类的用法；Vulkan 原生支持）。附件视图保留完整层区间（不再强制 1 层），
            // framebuffer 的 layers 由 create_pass 末尾按附件的层区间取（见 np626 的 fb.layers）。
            // 3D 体纹理的切片渲染走既有分支（volume 的 ls 恒 1）。
            // np626: allow layered rendering. On-device read: a 2D_ARRAY depth texture 896x896 with layerCount=3
            // rendered in one pass (shadow-cascade / env-map style); Vulkan supports this natively. Attachment
            // views keep their full layer range (no longer forced to 1); the framebuffer's `layers` is taken from
            // the attachments at the end of create_pass. Volume slice rendering keeps its existing branch.
            n=ls;
        }
        VkFormat vf=view_format(v);
        if(vf!=fmt){static std::set<std::pair<int,int>> seen;if(seen.insert({int(fmt),int(vf)}).second)xbg_logf("NVN viewfmt image=%d view=%d nvn=%#x->%#x attachment=%d storage=%d mutable=%d",int(fmt),int(vf),source.d.format,v?v->format:0,int(attachment),int(storage),int(mutable_format));
            if(!mutable_format)upgrade_mutable(*this);}
        else if(mutable_test && !mutable_format && !attachment && !storage && aspects==VK_IMAGE_ASPECT_COLOR_BIT && source.serial%mutable_test==0)upgrade_mutable(*this);
        int target=v&&v->set_target?v->target:source.d.target;
        VkImageViewType type=target==2?VK_IMAGE_VIEW_TYPE_3D:target==0?VK_IMAGE_VIEW_TYPE_1D:target==3?VK_IMAGE_VIEW_TYPE_1D_ARRAY:target==4||target==6?VK_IMAGE_VIEW_TYPE_2D_ARRAY:VK_IMAGE_VIEW_TYPE_2D;
        // 3D 当渲染目标：图像必须带 2D_ARRAY_COMPATIBLE，下面那句 attachment 分支会把它建成 VIEW_TYPE_2D
        // （baseArrayLayer=切片）。没有这个标志就说明设备拒了这种格式/用法组合，宁可 FATAL 报清原因，
        // 也不要静默按 3D 视图画错。
        // A 3D render target requires the image to carry 2D_ARRAY_COMPATIBLE; the attachment branch below then
        // builds a VIEW_TYPE_2D with baseArrayLayer = slice. Without the flag the device refused this
        // format/usage combination, so fail loudly rather than silently rendering through a 3D view.
        if(attachment && volume)need(vol_compat,"3D texture was created without 2D-compatible view support");
        if(!attachment && (target==8 || target==9))type=target==8?VK_IMAGE_VIEW_TYPE_CUBE:VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        if(attachment){need(source.d.target!=0 && source.d.target!=3,"1D render targets require 2D storage adaptation");type=VK_IMAGE_VIEW_TYPE_2D;}
        VkImageAspectFlags view_aspects=!attachment && (aspects&VK_IMAGE_ASPECT_DEPTH_BIT)?VK_IMAGE_ASPECT_DEPTH_BIT:aspects;
        VkImageUsageFlags view_usage=attachment?(aspects==VK_IMAGE_ASPECT_COLOR_BIT?VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT):storage?VK_IMAGE_USAGE_STORAGE_BIT:VK_IMAGE_USAGE_SAMPLED_BIT;
        need((usage&view_usage)==view_usage,"Image was not allocated for requested view usage");
        const VkFormatProperties &supported=format_properties(vf);
        VkFormatFeatureFlags feature=attachment?(aspects==VK_IMAGE_ASPECT_COLOR_BIT?VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT:VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT):storage?VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT:VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        need(supported.optimalTilingFeatures&feature,"Actual view format does not support requested usage");
        std::array<uint32_t,8> key{base,n,layer,ls,uint32_t(vf),uint32_t(type),view_aspects,view_usage};auto it=views.find(key);if(it!=views.end())return it->second;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};vi.image=image;vi.viewType=type;vi.format=vf;vi.subresourceRange={aspects,base,n,layer,ls};
        vi.subresourceRange.aspectMask=view_aspects;
        VkImageViewUsageCreateInfo vu{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};vu.usage=view_usage;vi.pNext=&vu;
        unsigned view_code=v&&v->set_format?v->format:source.d.format;
        if(!attachment && !storage && (view_code==0x37 || view_code==0x6d))vi.components.a=VK_COMPONENT_SWIZZLE_ONE;
        VkImageView result;VK(vkCreateImageView(cfg.device,&vi,nullptr,&result));views[key]=result;return result;
    }
};
std::map<uint64_t,std::shared_ptr<Image>> images;
// Address-ordered index of image storage, rebuilt lazily when images change.
struct ImageRange {uint64_t begin,end;Image *image;};
std::vector<ImageRange> image_index;bool image_index_dirty=true;uint64_t image_index_maxlen=0;
void rebuild_image_index(){
    image_index.clear();image_index_maxlen=0;
    for(auto &e:images){uint64_t b=uintptr_t(e.second->source.mem);if(!b)continue;uint64_t n=e.second->source.L.size;image_index.push_back({b,b+n,e.second.get()});image_index_maxlen=std::max(image_index_maxlen,n);}
    std::sort(image_index.begin(),image_index.end(),[](const ImageRange &a,const ImageRange &b){return a.begin<b.begin;});image_index_dirty=false;
}
template<class F> void for_images_overlapping(uint64_t address,uint64_t bytes,F f){
    if(image_index_dirty)rebuild_image_index();
    uint64_t lo=address>image_index_maxlen?address-image_index_maxlen:0;
    auto it=std::lower_bound(image_index.begin(),image_index.end(),lo,[](const ImageRange &r,uint64_t v){return r.begin<v;});
    for(;it!=image_index.end() && it->begin<address+bytes;++it)if(it->end>address)f(*it->image);
}
// A sampled/storage read inside a draw: barrier only if the image may have been
// written since the last full barrier, is an attachment of the open pass, or
// needs a layout change.
void prepare_read(Image &im,VkCommandBuffer cmd){
    bool in_open=open_pass.cmd && std::find(open_pass.images.begin(),open_pass.images.end(),&im)!=open_pass.images.end();
    bool need_layout=im.layout!=VK_IMAGE_LAYOUT_GENERAL,need_barrier=im.last_write>barrier_seq;
    // np641：**只有真需要才断 pass**。np641 实测：这一处每帧断 445 次 pass，是"每帧 ~440 个 pass"的真凶
    //   （`pass_end_by_cause 4=445`，而 storage 回读那条是 0）。三种情况分开处理：
    //   ① 布局不是 GENERAL ⇒ 必须断 pass（图像布局转换不能在 pass 内做）；
    //   ② 图像是本 pass 的附件（反馈读）⇒ 必须断 pass（Vulkan 不允许同时当附件和采样）；
    //   ③ 只是"上次屏障之后被写过" ⇒ **pass 内发一条内存屏障就够**，不用断。
    // np641: only end the pass when it is really required. Measured: this site ended the pass 445 times per
    //   frame and is the real cause of "~440 passes per frame" (`pass_end_by_cause 4=445`; the storage-readback
    //   site is 0). Three cases are now separated: (1) layout is not GENERAL -> must end the pass, an image
    //   layout transition cannot happen inside one; (2) the image is an attachment of the open pass (feedback
    //   read) -> must end the pass, Vulkan forbids sampling an attachment; (3) it was merely written since the
    //   last barrier -> a memory barrier inside the pass is enough, no need to end it.
    if(!need_layout){
        // np644：**反馈读不再断 pass**。np643 的读数：同一个 8 附件 1600×900 的 framebuffer 被连续开
        //   131/80/185 次（`PASSSEQ run=`），每帧 399 次 × ~92 MB ≈ 37 GB —— 而**断完重开的新 pass 附件
        //   还是同一批**，采样冲突原样存在 ⇒ 这次断 pass 既没换来正确性、又白付带宽。
        //   按 NVN 的 tile 内存语义：采样当前目标就让它读（Vulkan 上是未定义行为，但驱动通常按当前内容读）。
        //   真机判据：pass 数应 512 → ~120、GPU 每帧从 1.1–1.6 s 掉下来、帧数推进。
        // np644: a feedback read no longer ends the pass. np643 read: the same 8-attachment 1600x900 framebuffer
        //   was opened 131/80/185 times in a row (`PASSSEQ run=`), 399 times per frame x ~92 MB = ~37 GB -- while
        //   the reopened pass has the SAME attachments, so the sampling conflict is unchanged: ending the pass
        //   bought no correctness and paid the bandwidth. Follow NVN's tile-memory semantics and let the read
        //   through (undefined in Vulkan, but drivers read the current contents). Device criterion: passes
        //   512 -> ~120, GPU per frame well below 1.1-1.6 s, frames advance.
        if(in_open)++prep_read_feedback;
        // np648：**反馈读不再补屏障**。np644 时这里对每次 in_open 都发一条全内存屏障，实测每帧 3901 次；
        //   tile 型 GPU 上 pass 内的全内存屏障会逼 tile flush，代价可能比省下的 pass 重开还高。
        //   而且这条屏障本来也修不了"附件同时被采样"的冲突（Vulkan 上就是未定义），所以只在
        //   "上次屏障之后被写过"时才发。
        // np648: no barrier for a feedback read. np644 issued a full memory barrier for every in_open
        //   (3901 per frame measured); on a tiled GPU an in-pass full barrier forces a tile flush, which can
        //   cost more than the pass reopen it saved -- and it cannot fix "an attachment is also sampled"
        //   anyway (undefined in Vulkan). Only emit it when the image was written since the last barrier.
        if(need_barrier)full_barrier_in_pass(cmd);
        return;
    }
    ++prep_read_end;
    end_pass(4);im.transition(cmd,VK_IMAGE_LAYOUT_GENERAL);g_barrier_site="transition";full_barrier(cmd);
}
// np713：阴影低分辨率渲染的放大回填（由 end_open_pass 在 pass 关闭后调用）。tmp 中转避免同图
// 自 blit 的重叠区非法；NEAREST（R32F 无 LINEAR blit 特性）；写标记照 GPU 写记。
// np713: upscale for the low-res shadow render (called from end_open_pass after the pass closes).
void shadow_scale_blit(VkCommandBuffer cmd){
    Image &im=*g_shadow_img;g_barrier_site="shadow-preblit";full_barrier(cmd);
    if(!g_shadow_tmp || g_shadow_tmp_w!=g_shadow_sw || g_shadow_tmp_h!=g_shadow_sh || g_shadow_tmp_fmt!=uint32_t(im.fmt)){
        if(g_shadow_tmp){vkDestroyImage(cfg.device,g_shadow_tmp,nullptr);vkFreeMemory(cfg.device,g_shadow_tmp_mem,nullptr);g_shadow_tmp={};}
        VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};ic.imageType=VK_IMAGE_TYPE_2D;ic.format=VkFormat(im.fmt);
        ic.extent={g_shadow_sw,g_shadow_sh,1};ic.mipLevels=1;ic.arrayLayers=1;ic.samples=VK_SAMPLE_COUNT_1_BIT;ic.tiling=VK_IMAGE_TILING_OPTIMAL;ic.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if(vkCreateImage(cfg.device,&ic,nullptr,&g_shadow_tmp)==VK_SUCCESS){
            VkMemoryRequirements mr;vkGetImageMemoryRequirements(cfg.device,g_shadow_tmp,&mr);
            VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ma.allocationSize=mr.size;ma.memoryTypeIndex=memory_type(mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if(vkAllocateMemory(cfg.device,&ma,nullptr,&g_shadow_tmp_mem)==VK_SUCCESS && vkBindImageMemory(cfg.device,g_shadow_tmp,g_shadow_tmp_mem,0)==VK_SUCCESS){
                g_shadow_tmp_w=g_shadow_sw;g_shadow_tmp_h=g_shadow_sh;g_shadow_tmp_fmt=uint32_t(im.fmt);
                VkImageMemoryBarrier t{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};t.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;t.newLayout=VK_IMAGE_LAYOUT_GENERAL;t.image=g_shadow_tmp;
                t.srcQueueFamilyIndex=t.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;t.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
                t.srcAccessMask=0;t.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&t);
            }else{vkDestroyImage(cfg.device,g_shadow_tmp,nullptr);g_shadow_tmp={};}
        }
    }
    if(g_shadow_tmp){
        VkImageCopy cp{};cp.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};cp.dstSubresource=cp.srcSubresource;cp.extent={g_shadow_sw,g_shadow_sh,1};
        vkCmdCopyImage(cmd,im.image,VK_IMAGE_LAYOUT_GENERAL,g_shadow_tmp,VK_IMAGE_LAYOUT_GENERAL,1,&cp);
        g_barrier_site="shadow-postcopy";full_barrier(cmd);
        // np713b：blit → N² 象限 CopyImage（50% = 4 份）。BLIT_DST 在 fmt124 上不保证存在（Maleoon
        //   首轮后 render_service cppcrash 紧跟 blit，嫌疑未排除），CopyImage 只需 TRANSFER 位，任何
        //   格式安全。仅支持整倍数缩放（begin_pass 里已卡 pw%sw==0）。
        // np713b: blit -> N^2 quadrant CopyImage (50% = 4 copies). BLIT_DST is not guaranteed for fmt124
        //   (first round: render_service cppcrash right after the blit, suspicion unresolved); CopyImage only
        //   needs TRANSFER bits and is safe for any format. Integer scales only (checked in begin_pass).
        for(unsigned qy=0;qy<g_shadow_quad;++qy)for(unsigned qx=0;qx<g_shadow_quad;++qx){
            VkImageCopy q{};q.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};q.dstSubresource=q.srcSubresource;
            q.srcOffset={0,0,0};q.dstOffset={int32_t(qx*g_shadow_sw),int32_t(qy*g_shadow_sh),0};q.extent={g_shadow_sw,g_shadow_sh,1};
            vkCmdCopyImage(cmd,g_shadow_tmp,VK_IMAGE_LAYOUT_GENERAL,im.image,VK_IMAGE_LAYOUT_GENERAL,1,&q);}
        g_barrier_site="shadow-postquads";full_barrier(cmd);
        im.last_write=++write_seq;im.gpu_written=true;
        if(!g_shadow_scale_once){g_shadow_scale_once=true;xbg_logf("NVN shadow scale: %ux%u -> %ux%u blit（DM_GFX_SHADOW_SCALE）",g_shadow_pw,g_shadow_ph,g_shadow_sw,g_shadow_sh);}
    }
}

void prepare_attachment(Image &im,VkCommandBuffer cmd){
    bool in_open=open_pass.cmd && std::find(open_pass.images.begin(),open_pass.images.end(),&im)!=open_pass.images.end();
    if(im.layout!=VK_IMAGE_LAYOUT_GENERAL && !in_open){end_pass(4);im.transition(cmd,VK_IMAGE_LAYOUT_GENERAL);}
    im.last_write=++write_seq;
}
void provenance_image(const char *event,const tex_t *t,const Image &im,const view_t *v=nullptr){
    if(!provenance_file || !provenance.window(stats.presents.load()))return;
    provenance_log("IMAGE event=%s requested=%llu source=%llu image=%p mem=%p pool=%p off=%lld bytes=%llu createEpoch=%llu currentEpoch=%llu writer=%llu writeSite=%u writerPhase=declared-or-complete-range-unspecified uploaded=%u gpu=%u fmt=%d target=%d extent=%dx%dx%d levels=%d samples=%d flags=%x stride=%d pkg=%d viewFmt=%d mip=%d/%d layer=%d/%d",
    event,(unsigned long long)t->serial,(unsigned long long)im.source.serial,(void*)im.image,t->mem,(void*)(t->mem && t->off>=0 && uint64_t(t->off)<=uint64_t(t->mem)?uint64_t(t->mem)-uint64_t(t->off):0),(long long)t->off,(unsigned long long)t->L.size,(unsigned long long)im.provenance_epoch,(unsigned long long)provenance.epoch((t->mem && t->off>=0 && uint64_t(t->off)<=uint64_t(t->mem)?uint64_t(t->mem)-uint64_t(t->off):0)),(unsigned long long)im.provenance_writer,im.provenance_write_site,im.uploaded,im.gpu_written,t->d.format,t->d.target,t->d.width,t->d.height,t->d.depth,t->d.levels,t->d.samples,t->d.flags,t->d.stride,t->d.pkg_block_h,v&&v->set_format?v->format:t->d.format,v&&v->set_levels?v->base_level:0,v&&v->set_levels?v->nlevels:int(im.levels),v&&v->set_layers?v->min_layer:0,v&&v->set_layers?v->nlayers:int(im.layers));
}
void provenance_write(Image &im,unsigned site){im.provenance_writer=++provenance.sequence;im.provenance_write_site=site;provenance_image("WRITE",&im.source,im);}

// Invoked only after a completed game draw/clear fence. Read the real Vulkan
// subresources and restore their original layout before guest execution resumes.
void capture_image(Image &im,const view_t *view,const char *role){
    if(!capture.active)return;
    uint32_t base=view&&view->set_levels?view->base_level:0,levels=view&&view->set_levels?view->nlevels:im.levels;
    uint32_t layer=view&&view->set_layers?view->min_layer:0,layers=view&&view->set_layers?view->nlayers:im.layers;
    capture_log("IMAGE role=%s serial=%llu image=%p source=%p sourceBytes=%llu format=%#x vk=%u extent=%ux%ux%u levels=%u baseMip=%u count=%u baseLayer=%u layers=%u layout=%u uploaded=%u gpuWritten=%u",role,(unsigned long long)im.source.serial,(void*)im.image,im.source.mem,(unsigned long long)im.source.L.size,im.source.d.format,unsigned(im.fmt),im.width,im.height,im.depth,im.levels,base,levels,layer,layers,unsigned(im.layout),im.uploaded,im.gpu_written);
    if(im.samples!=VK_SAMPLE_COUNT_1_BIT || im.layout==VK_IMAGE_LAYOUT_UNDEFINED){capture_log("SKIP role=%s reason=uninitialized-or-multisample",role);return;}
    if(base>=im.levels || levels>im.levels-base || layer>=im.layers || layers>im.layers-layer){capture_log("SKIP role=%s reason=invalid-view",role);return;}
    unsigned subresources=0;
    for(uint32_t l=layer;l<layer+layers;l++)for(uint32_t m=base;m<base+levels;m++){
        if(++subresources>16){capture_log("SKIP role=%s reason=16-subresource-limit",role);return;}
        uint32_t w=std::max(1u,im.width>>m),h=std::max(1u,im.height>>m),z=std::max(1u,im.depth>>m);
        VkImageAspectFlags plane=im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT?VK_IMAGE_ASPECT_DEPTH_BIT:im.aspects;
        uint32_t bpp=im.bpp;if(plane==VK_IMAGE_ASPECT_DEPTH_BIT){xbg_depth_format f;if(xbg_depth_format_info(im.source.d.format,&f)){capture_log("SKIP reason=unknown-depth-format");return;}bpp=im.source.d.format==0x32 && depth16_as_d24?4:f.depth_bytes;}
        uint64_t bytes=uint64_t(w)*h*z*bpp;
        if(!bytes || bytes>capture_limit-capture.bytes){capture_log("SKIP role=%s mip=%u layer=%u bytes=%llu reason=64MiB-budget",role,m,l,(unsigned long long)bytes);continue;}
        Buffer buffer=readback_buffer(size_t(bytes));Commands command;VkImageLayout saved=im.layout;
        im.transition(command.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);VkBufferImageCopy region{};region.imageSubresource={plane,m,l,1};region.imageExtent={w,h,z};
        region.bufferOffset=buffer.offset;vkCmdCopyImageToBuffer(command.command,im.image,im.layout,buffer.buffer,1,&region);
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};host.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;host.srcQueueFamilyIndex=host.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;host.buffer=buffer.buffer;host.offset=buffer.offset;host.size=buffer.size;
        vkCmdPipelineBarrier(command.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&host,0,nullptr);
        im.transition(command.command,saved);command.finish();gpu_sync(perf.sync_readback);buffer.invalidate();
        char label[160];snprintf(label,sizeof(label),"%s-s%llu-m%u-l%u",role,(unsigned long long)im.source.serial,m,l);
        capture_log("PIXELS label=%s vk=%u aspect=%u width=%u height=%u depth=%u rowBytes=%u restoredLayout=%u",label,unsigned(im.fmt),plane,w,h,z,w*bpp,unsigned(im.layout));
        capture_data(label,buffer.mapped,size_t(bytes));
    }
}
std::map<uintptr_t,size_t> pools;
struct RegisteredTexture{tex_t texture{};view_t view{};bool has_view=false;uint64_t revision=0;};
// Bumped by every event that can fail a descriptor memo entry's per-image checks: a texture registration that changes, an
// image created or released for a serial. A memo hit repeats those checks only when it moved since the entry's last ones.
uint64_t memo_check_gen=1;
// 只比较可影响图像/视图的语义字段，忽略 padding、snapshot 指针及未启用的 view 字段。
// Compare image/view semantics only, ignoring padding, snapshot pointers and inactive view fields.
static bool registered_texture_equal(const RegisteredTexture &a,const RegisteredTexture &b){
    const tex_t &x=a.texture,&y=b.texture;
    if(x.serial!=y.serial || x.mem!=y.mem || x.pool!=y.pool || x.off!=y.off ||
       x.d.target!=y.d.target || x.d.format!=y.d.format || x.d.flags!=y.d.flags || x.d.samples!=y.d.samples ||
       x.d.width!=y.d.width || x.d.height!=y.d.height || x.d.depth!=y.d.depth || x.d.levels!=y.d.levels || x.d.stride!=y.d.stride || x.d.pkg_block_h!=y.d.pkg_block_h ||
       x.L.size!=y.L.size || x.L.layer_size!=y.L.layer_size ||
       x.L.align!=y.L.align || x.L.layers!=y.L.layers || x.L.bpb!=y.L.bpb || x.L.tileH!=y.L.tileH || x.L.tileD!=y.L.tileD ||
       !!x.has_layout!=!!y.has_layout || (x.has_layout && memcmp(x.layout,y.layout,sizeof(x.layout))))return false;
    const view_t &u=a.view,&v=b.view;
    return a.has_view==b.has_view && (!a.has_view ||
        (!!u.set_levels==!!v.set_levels && (!u.set_levels || (u.base_level==v.base_level && u.nlevels==v.nlevels)) &&
         !!u.set_layers==!!v.set_layers && (!u.set_layers || (u.min_layer==v.min_layer && u.nlayers==v.nlayers)) &&
         !!u.set_format==!!v.set_format && (!u.set_format || u.format==v.format) &&
         !!u.set_target==!!v.set_target && (!u.set_target || u.target==v.target)));
}
static bool register_texture_record(RegisteredTexture &slot,const tex_t &texture,const view_t *view){
    RegisteredTexture next;next.texture=texture;
    if(view && (view->set_levels || view->set_layers || view->set_format || view->set_target)){next.view=*view;next.has_view=true;}
    bool changed=!slot.revision || !registered_texture_equal(slot,next);
    next.revision=slot.revision+uint64_t(changed);slot=next;if(changed)++memo_check_gen;return changed;
}
struct MemoImage{const tex_t *first;Image *second;const RegisteredTexture *record;uint64_t revision;};
using PoolKey=std::pair<uintptr_t,int64_t>;
PoolKey pool_key(const dpool_t *p){need(p && p->pool,"Missing descriptor pool");return {uintptr_t(p->pool->mem),p->off};}
std::map<PoolKey,std::map<int,RegisteredTexture>> textures;
std::map<int,RegisteredTexture> legacy_textures;std::map<int,samp_par> legacy_samplers;
std::map<int,RegisteredTexture> *texture_pool_cache=nullptr;PoolKey texture_pool_cache_key{}; // descriptors(): last texture pool (entries are never erased before shutdown)
std::map<PoolKey,std::map<int,samp_par>> samplers;
std::map<uint32_t,RegisteredTexture> null_textures;
struct HostSegment{cmd_t *owner;std::unique_ptr<uint8_t[]> memory;size_t bytes;};
std::vector<HostSegment> host_segments;
std::map<uintptr_t,size_t> host_ranges; // address-ordered index of host_segments
size_t host_bytes=0;
uint64_t hash_memory(const void *memory,size_t n){CPU_SCOPE(Hash);const uint8_t *p=static_cast<const uint8_t*>(memory);
    // Word-at-a-time 4-lane mix: change detection only, never persisted.
    uint64_t h[4]={1469598103934665603ull,0x9e3779b97f4a7c15ull,0xc2b2ae3d27d4eb4full,0x165667b19e3779f9ull};size_t i=0;
    for(;i+32<=n;i+=32){for(unsigned k=0;k<4;k++){uint64_t w;memcpy(&w,p+i+8*k,8);h[k]=(h[k]^w)*0x100000001b3ull;h[k]^=h[k]>>29;}}
    uint64_t hash=h[0]^(h[1]*3)^(h[2]*5)^(h[3]*7)^n;for(;i<n;i++){hash^=p[i];hash*=1099511628211ull;}return hash;}
// Grow-only upload scratch: keeps capacity between calls and never zero-fills. std::vector<uint8_t>::resize
// value-initializes, and on this path that meant a full-image memset plus malloc/free on every (re)upload —
// measured on a Mali-G925 (in-run BC re-uploads) at ~4-5 ms per 4 MB image, roughly half the per-cycle cost.
// Only used from the single thread that records submissions, like the vectors it replaces.
// 只增不减的上传暂存：跨调用保留容量、且从不填零。std::vector<uint8_t>::resize 会做值初始化，在这条路径上等于
// 每次（重）上传都要整图 memset 外加 malloc/free——Mali-G925 实测 4 MB 图约 4-5 ms，约占单次循环一半。
// 与它替换掉的 vector 一样，只由录制提交的那一个线程使用。
struct UpScratch{uint8_t *p=nullptr;size_t cap=0,n=0;
    void reset(){n=0;}
    void resize(size_t m){if(m<=cap){n=m;return;}size_t c=m+(m>>1)+65536u;uint8_t *q=static_cast<uint8_t*>(::operator new(c));if(n)memcpy(q,p,n);::operator delete(p);p=q;cap=c;n=m;}
    uint8_t *data(){return p;} const uint8_t *data()const{return p;} size_t size()const{return n;}
    void release(){::operator delete(p);p=nullptr;cap=n=0;}
    ~UpScratch(){::operator delete(p);}};
thread_local UpScratch up_linear,up_blocks;   // get_image's upload buffers | get_image 的上传缓冲
// Differential block decode for re-uploaded decoded BC images. Consecutive re-saves of the streamed cubes
// (BC6H 256x256x6) change only 1.7-1.8% of their 16-byte blocks, yet every cycle decoded the whole image — 6.2 ms of the
// 8.3 ms per cycle. BC blocks are independent, so decode only the blocks whose compressed bytes changed and keep the last
// decoded image as the base: identical output, ~1 ms per cycle. Bounded to 16 images, ordered by hotness, and an image must
// have re-uploaded 3 times before it takes a slot, so load-time one-offs cannot evict the images that cycle every frame.
// 重上传的解码 BC 图做块级差分解码：连续两次重存的流式立方图（BC6H 256x256x6）只改了 1.7-1.8% 的 16 字节块，
// 但每次都整图解码（每循环 8.3 ms 里 6.2 ms 在解码）。BC 的块彼此独立，所以只解压缩字节变了的块、以上次解码结果
// 为基底：输出完全一致，每循环约 1 ms。最多 16 张，按热度排序，且一张图要先重传 3 次才占槽，
// 这样载入期的一次性大图不会把每帧都循环的图挤出去。
struct UpDiff{
    std::vector<uint8_t> src,src_new,out;std::vector<uint32_t> off;   // off: byte offset of each sub-resource inside src/src_new | off：各子资源在 src/src_new 内的字节偏移
    uint32_t fmt=0,layers=0,levels=0,width=0,height=0,bpp=0;size_t total=0,total_src=0;unsigned verify=2;bool ready=false;
    uint64_t used=0;   // last-use stamp; the hot re-uploaded images must win the slots over the load-time ones | 最近使用时间戳：真正循环的图要优先于载入期一次性图拿到槽位
};
std::map<uint64_t,UpDiff> up_diffs;uint64_t up_diff_clock=0,up_diff_bytes=0;std::map<uint64_t,unsigned> up_diff_seen;
// Byte budget of the differential state (src + src_new + out of every slot) and the size of the re-upload counter map.
// 差分状态的字节预算（所有槽的 src + src_new + out 之和），以及重传计数表的条目上限。
constexpr uint64_t up_diff_budget=64ull*1024*1024;constexpr size_t up_diff_seen_max=4096;
size_t up_diff_size(const UpDiff &d){return d.src.size()+d.src_new.size()+d.out.size();}
void upload_cache_release(){up_diffs.clear();up_diff_bytes=0;up_diff_seen.clear();up_linear.release();up_blocks.release();}
void invalidate_cpu_range(uint64_t address,uint64_t bytes){CPU_SCOPE(CpuWrite);
    need(address+bytes>=address,"CPU write range overflow");cpu_access(address,bytes,4);snapshot_invalidate(address,bytes);
    for_images_overlapping(address,bytes,[&](Image &im){if(draw_detail)draw_timing.image();uint64_t begin=uintptr_t(im.source.mem),end=begin+im.source.L.size;
        // A full storage overwrite replaces the prior GPU contents. A partial
        // overwrite requires GPU-to-NVN readback/merge before applying CPU data.
        need(!im.gpu_written || (address<=begin && address+bytes>=end),"Partial CPU write aliases GPU-written texture; readback/merge required");
        // The device now holds GPU-written contents, not the bytes cpu_hash describes: forget it so the upload below cannot be
        // skipped because the guest happened to write back what it last uploaded.
        // 设备里现在是 GPU 写出的内容，不是 cpu_hash 所描述的字节：清掉它，免得客户机恰好写回上次上传的字节时下面的上传被跳过。
        if(im.gpu_written)im.cpu_hash=0;
        im.gpu_written=false;provenance_write(im,__LINE__);im.uploaded=false;im.checked_submission=UINT64_MAX;
    });
}

bool resolving_pending=false;
// 仅后续写回完整覆盖、池明确不允许CPU读且无图像alias时跳过；半开范围溢出不成立。
// Skip only when a later writeback fully covers a known CPU_NO_ACCESS range with no image alias; overflowing half-open ranges never qualify.
bool writeback_superseded(size_t i){
    static_assert(NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS==1,"Writeback pool flag contract changed");
    return xbg_wb_superseded(pending_writebacks,i,wb_skip && nvn_pool_flags && !ab_off("wbskip"),
        [](uint64_t address,uint64_t bytes){return nvn_pool_flags(reinterpret_cast<const void*>(uintptr_t(address)),size_t(bytes));},
        [](uint64_t address,uint64_t bytes){bool image=false;for_images_overlapping(address,bytes,[&](Image &){image=true;});return image;});
}
// Writebacks of a slot whose fence has signalled can be applied without waiting.
void resolve_slot_writebacks(const void *slot){
    if(pending_writebacks.empty())return;resolving_pending=true;
    std::vector<PendingWriteback> keep;
    for(size_t i=0;i<pending_writebacks.size();i++){auto &w=pending_writebacks[i];if(w.slot!=slot){keep.push_back(w);continue;}
        if(writeback_superseded(i)){++wb_skipped;wb_skipped_bytes+=w.bytes;continue;}
        writeback_bytes+=w.bytes;w.buffer.invalidate();invalidate_cpu_range(w.addr,w.bytes);memcpy(reinterpret_cast<void*>(w.addr),w.buffer.mapped,w.bytes);}
    pending_writebacks.swap(keep);resolving_pending=false;
}
void write_report(const PendingReport &r){
    uint64_t samples=0;auto e=epochs.find(r.epoch);if(e!=epochs.end()){e->second.advance_known();samples=e->second.prefix(r.count);}
    uint64_t value=r.ts_value;if(timestamp_bits<64)value&=(uint64_t(1)<<timestamp_bits)-1;
    if(r.type==NVN_COUNTER_TYPE_SAMPLES_PASSED){++samples_reports;samples_reported+=samples;}
    NVNcounterData result{};result.counter=r.type==NVN_COUNTER_TYPE_SAMPLES_PASSED?samples:0;result.timestamp=uint64_t(double(value)*properties.limits.timestampPeriod*12.0/625.0);
    snapshot_invalidate(r.addr,sizeof(result));evguard::EngineRead er(reinterpret_cast<void*>(r.addr),sizeof(result));memcpy(reinterpret_cast<void*>(r.addr),&result,sizeof(result));
}
// Reports whose timestamp and samples are all known are written in order.
void resolve_known_reports(){
    size_t done=0;
    for(;done<pending_reports.size();done++){const auto &r=pending_reports[done];if(!r.ts_known)break;auto e=epochs.find(r.epoch);bool known=true;if(e!=epochs.end()){e->second.advance_known();known=e->second.known_upto>=std::min<uint64_t>(r.count,e->second.size());}if(!known)break;write_report(r);}
    if(done)pending_reports.erase(pending_reports.begin(),pending_reports.begin()+done);
    // Fold known queries into the running sum up to the earliest count still needed by a pending report, then drop them.
    for(auto &e:epochs){uint64_t limit=e.second.known_upto;for(auto &r:pending_reports)if(r.epoch==e.first)limit=std::min<uint64_t>(limit,r.count);e.second.prefix(limit);e.second.trim();}
    // A report only ever names the epoch current when it was recorded, so an older epoch that no pending report names is
    // dead (retire skips query refs to erased epochs). With deferred events a report is always pending, so erasing only
    // when none was pending let the map grow by every counter reset (~13 per D2R frame).
    for(auto it=epochs.begin();it!=epochs.end();){bool used=it->first==counter_epoch;for(auto &r:pending_reports)if(r.epoch==it->first){used=true;break;}
        if(used)++it;else it=epochs.erase(it);}
}
void resolve_pending(){
    if(resolving_pending || (pending_writebacks.empty() && pending_reports.empty()))return;
    resolving_pending=true;
    gpu_sync(pending_writebacks.empty()?perf.sync_query:perf.sync_writeback);
    auto writebacks=std::move(pending_writebacks);pending_writebacks.clear();
    for(auto &w:writebacks){w.buffer.invalidate();invalidate_cpu_range(w.addr,w.bytes);memcpy(reinterpret_cast<void*>(w.addr),w.buffer.mapped,w.bytes);}
    for(auto &r:pending_reports)write_report(r);
    pending_reports.clear();
    for(auto it=epochs.begin();it!=epochs.end();)if(it->first!=counter_epoch)it=epochs.erase(it);else ++it;
    apply_ready_events();
    resolving_pending=false;
}
// 先完成待写回/报告，再取得命令缓冲、描述符、图像或槽引用；同步会提交并退役槽。
// Resolve writebacks/reports before acquiring command, descriptor, image or slot references; synchronization submits and retires slots.
// 保守同步可能减少异步重叠；范围化优化须另验整个调用链。
// Conservative synchronization may reduce asynchronous overlap; range-based optimization needs whole-call-chain validation.
// 默认开启，与 main 相同；宿主可设 DM_GFX_READ_PREFLIGHT=0 关闭（标记 gfx-read-preflight 为 1 / 0 时以标记为准）。
// Android 关闭：延后事件下几乎总有报告待写，每个 draw 前同步会让 GPU 与翻译串行；Redmi K50 Ultra 营地与 Den of Evil
// 战斗的记录中，没有一次 CPU 读碰到待写回或待写报告（NVN PERF 的 why=a0）。
// On by default, as on main; a host sets DM_GFX_READ_PREFLIGHT=0 to turn it off (a gfx-read-preflight marker, 1 / 0,
// wins). Android does: with deferred events a report is nearly always pending, so a sync before every draw would serialize
// the GPU with translation, and in the Redmi K50 Ultra camp and Den of Evil fight logs no CPU read ever met a pending
// writeback or report (why=a0 in NVN PERF).
bool read_preflight=true;
void preflight_cpu_reads(){if(read_preflight)resolve_pending();}
void cpu_access(uint64_t address,uint64_t bytes,unsigned why){
    if(!bytes)return;
    need(xbg_wb_range_valid(address,bytes),"CPU access range overflow");
    if(resolving_pending)return;
    for(auto &w:pending_writebacks)if(xbg_wb_overlaps(address,bytes,w.addr,w.bytes)){++perf.why_access;if(why<12)++perf.access_by[why];resolve_pending();return;}
    // GPU reads of counter memory (snapshots) see the last resolved report, i.e.
    // one frame of latency for occlusion-driven effects instead of a full stall.
    if(why==1 || why==2)return;
    for(auto &r:pending_reports)if(xbg_wb_overlaps(address,bytes,r.addr,sizeof(NVNcounterData))){++perf.why_access;++perf.access_by[7];resolve_pending();return;}
}
// Guest fence completion. Work recorded for submitted guest command buffers is
// complete when every frame slot is retired; deferred results are then resolved.
void complete_all(){SiteScope site(SITE_COMPLETE);resolve_pending();submit_slots(true);publish_completed(submitted);}
// Block until guest fence `fence` is complete, retiring only the slots needed.
void wait_fence(uint64_t fence){
    for(int i=0;i<2 && fence>completed;i++){Slot &older=slots[slot_index].pending?slots[slot_index]:slots[slot_index^1];if(!older.pending || older.covers<fence)break;retire_slot(older);}
    if(fence>completed)complete_all();
}
bool gpu_poll_complete(){
    SiteScope site(SITE_POLL);
    Slot &cur=slots[slot_index];if(cur.recording && cur.ops)submit_slots(false);
    for(auto &s:slots)if(s.pending){if(vkGetFenceStatus(cfg.device,s.fence)!=VK_SUCCESS)return false;}
    ++perf.why_poll;complete_all(); // everything signaled: retiring/resolving does not block
    return true;
}
// Vulkan buffer/image transfers name exactly one depth/stencil aspect. Keep
// both packed NVN channels, with independent tightly packed planes and copies.
template<class Buf> void append_depth_planes(Buf &data,std::vector<VkBufferImageCopy> &copies,
    uint32_t format,uint32_t w,uint32_t h,uint32_t slices,const void *packed,size_t packed_bytes,
    size_t row,size_t image,const VkBufferImageCopy &region){
    xbg_depth_format f;need(xbg_depth_format_info(format,&f)==0,"Unknown NVN depth transfer format");
    bool widen=format==0x32 && depth16_as_d24;   // image texels are X8_D24 (4 bytes), guest texels D16 (2 bytes)
    uint64_t pixels=uint64_t(w)*h*slices;need(pixels<=256ull*1024*1024/5,"Depth staging extent exceeds bound");
    size_t begin=(data.size()+15)&~size_t(15),depth_bytes=size_t(pixels)*(widen?4:f.depth_bytes);
    size_t stencil_begin=(begin+depth_bytes+3)&~size_t(3),stencil_bytes=size_t(pixels)*f.stencil_bytes;
    need(stencil_begin+stencil_bytes<=256ull*1024*1024,"Depth staging allocation exceeds bound");data.resize(stencil_begin+stencil_bytes);
    std::vector<uint8_t> d16(packed && widen?size_t(pixels)*2:0);
    if(packed)need(xbg_depth_extract(format,w,h,slices,packed,packed_bytes,row,image,
        f.depth_bytes?(widen?d16.data():data.data()+begin):nullptr,widen?d16.size():depth_bytes,f.stencil_bytes?data.data()+stencil_begin:nullptr,stencil_bytes)==0,"Invalid packed NVN depth upload extent");
    for(size_t i=0;i<d16.size()/2;i++){uint16_t k;memcpy(&k,&d16[i*2],2);uint32_t v=d16_to_d24(k);memcpy(data.data()+begin+i*4,&v,4);}
    if(f.depth_bytes){auto r=region;r.bufferOffset=begin;r.imageSubresource.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT;copies.push_back(r);}
    if(f.stencil_bytes){auto r=region;r.bufferOffset=stencil_begin;r.imageSubresource.aspectMask=VK_IMAGE_ASPECT_STENCIL_BIT;copies.push_back(r);}
}

void forget_framebuffers(Image &im,std::vector<VkFramebuffer> *deferred){
    for(auto &k:im.fb_keys){auto f=persistent_framebuffers.find(k);if(f==persistent_framebuffers.end())continue;   // already gone with another image
        if(deferred)deferred->push_back(f->second);else vkDestroyFramebuffer(cfg.device,f->second,nullptr);persistent_framebuffers.erase(f);}
    im.fb_keys.clear();
}
// Direct-mapped front for images.find (~12k get_image calls per D2R camp frame): erasing from or clearing images bumps
// images_epoch, so a front entry never outlives its Image.
struct ImgFront{uint64_t serial=0,epoch=0;Image *im=nullptr;};ImgFront img_front[1024];uint64_t images_epoch=1;
// The Image currently registered for a serial, or null (the same front as get_image).
Image *find_image(uint64_t serial){ImgFront &f=img_front[(serial*0x9E3779B97F4A7C15ull)>>54];if(f.epoch==images_epoch && f.serial==serial)return f.im;
    auto it=images.find(serial);if(it==images.end())return nullptr;f.serial=serial;f.epoch=images_epoch;f.im=it->second.get();return f.im;}
Image &get_image_slow(const tex_t *t,bool upload);
Image &get_image_read(const tex_t *t,bool upload);
// The common case, about 10k times a D2R camp frame: the image is in the front cache, and either nothing is uploaded or it was
// checked in this submission already, so none of get_image_slow's checks would run. Inline, without the slow path's large
// frame and timing scope. Provenance logging takes the slow path.
inline Image &get_image(const tex_t *t,bool upload){
    if(read_preflight)return get_image_read(t,upload);
    if(img_fast && t && t->serial && t->magic==MAGIC('T','E','X',0) && !provenance_file){
        const ImgFront &front=img_front[(t->serial*0x9E3779B97F4A7C15ull)>>54];
        if(front.epoch==images_epoch && front.serial==t->serial){Image &im=*front.im;
            if(!upload || (im.checked_submission==stats.submissions && (im.uploaded || im.gpu_written))){im.slot_use=slot_serial;return im;}}}
    return get_image_slow(t,upload);
}
Image &get_image_slow(const tex_t *t,bool upload){CPU_SCOPE(Image);
    need(t && t->serial && t->magic==MAGIC('T','E','X',0),"Invalid NVN texture");
    ImgFront &front=img_front[(t->serial*0x9E3779B97F4A7C15ull)>>54];Image *hit=front.epoch==images_epoch && front.serial==t->serial?front.im:nullptr;
    auto it=hit?images.end():images.find(t->serial);
    if(!hit){
    if(it==images.end() && t->mem){
        // Reinitialized or copied NVN objects with the same storage/layout share
        // GPU contents. Serial-only caches would incorrectly re-upload stale CPU
        // bytes when the new object aliases an already-rendered attachment.
        Image *alias=nullptr;for_images_overlapping(uint64_t(uintptr_t(t->mem)),1,[&](Image &c){if(!alias && c.source.mem==t->mem && !memcmp(&c.source.d,&t->d,sizeof(t->d)))alias=&c;});
        if(alias)for(auto &candidate:images)if(candidate.second.get()==alias){it=images.emplace(t->serial,candidate.second).first;image_index_dirty=true;++memo_check_gen;++memo_bump[3];provenance_image("ALIAS",t,*it->second);break;}
    }
    if(it==images.end()){
        need(t->d.samples<=1 || t->d.samples==2 || t->d.samples==4 || t->d.samples==8,"Unsupported texture sample count");
        need(t->d.target==0 || t->d.target==1 || t->d.target==2 || t->d.target==3 || t->d.target==4 || t->d.target==5 || t->d.target==6 || t->d.target==7 || t->d.target==8 || t->d.target==9,"Unsupported texture target (1D/3D/buffer handled separately)");
        auto p=std::make_shared<Image>();p->source=*t;p->provenance_epoch=provenance.epoch((t->mem && t->off>=0 && uint64_t(t->off)<=uint64_t(t->mem)?uint64_t(t->mem)-uint64_t(t->off):0));p->width=t->d.width;p->height=(t->d.target==0 || t->d.target==3)?1:t->d.height;p->depth=t->d.target==2?t->d.depth:1;p->levels=t->d.levels;p->layers=t->L.layers;p->aspects=aspect(t->d.format);p->samples=VkSampleCountFlagBits(std::max(1,t->d.samples));
        need(p->width && p->height && p->levels && p->levels<=16 && p->layers,"Invalid texture dimensions");
        // 设备不能线性过滤 R16G16_UNORM（Mali-G925 实测：optimalTilingFeatures 缺 FILTER_LINEAR）时，BC4/BC5 解码成 8 位通道，
        // 否则带线性过滤的采样器会拿不到可过滤格式。 | When the device cannot linearly filter R16G16_UNORM (Mali-G925: no FILTER_LINEAR),
        // BC4/BC5 decode to 8-bit channels so linear-filtered samplers stay valid.
        {static int bc8=-1;if(bc8<0){VkFormatProperties q;vkGetPhysicalDeviceFormatProperties(cfg.physical,VK_FORMAT_R16G16_UNORM,&q);bc8=!(q.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);if(bc8)xbg_logf("NVN caps bc 8bit: R16G16_UNORM optimalFeatures=%#x lacks SAMPLED_IMAGE_FILTER_LINEAR",q.optimalTilingFeatures);}if(bc8)xbg_bc_8bit_channels=1;}
        xbg_bc_format info;p->decoded=xbg_bc_format_info(t->d.format,&info)==0;
        if(p->decoded && native_bc && p->samples==VK_SAMPLE_COUNT_1_BIT){const VkFormatFeatureFlags need_bc=VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT|VK_FORMAT_FEATURE_TRANSFER_SRC_BIT|VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
            VkFormatProperties np;vkGetPhysicalDeviceFormatProperties(cfg.physical,format(t->d.format),&np);if((np.optimalTilingFeatures&need_bc)==need_bc)p->decoded=false;}
        if(xbg_bc_format_info(t->d.format,&info)==0)++(p->decoded?decoded_bc_images:native_bc_images);
        if(p->decoded){p->bpp=info.bytes_per_pixel;p->fmt=decoded_format(info.pixels);need(p->samples==VK_SAMPLE_COUNT_1_BIT,"BC texture cannot be multisampled");}
        else {p->fmt=format(t->d.format);p->bpp=t->L.bpb;if(t->d.format==0x32 && depth16_as_d24)p->fmt=VK_FORMAT_X8_D24_UNORM_PACK32;}
        VkFormatProperties fp;vkGetPhysicalDeviceFormatProperties(cfg.physical,p->fmt,&fp);
        VkImageUsageFlags usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)usage|=VK_IMAGE_USAGE_SAMPLED_BIT;
        if((t->d.flags&0x4) && (fp.optimalTilingFeatures&VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && (p->samples==VK_SAMPLE_COUNT_1_BIT || cfg.enabled_core.shaderStorageImageMultisample))usage|=VK_IMAGE_USAGE_STORAGE_BIT;
        if(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)usage|=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)usage|=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};ci.imageType=t->d.target==2?VK_IMAGE_TYPE_3D:t->d.target==0||t->d.target==3?VK_IMAGE_TYPE_1D:VK_IMAGE_TYPE_2D;ci.format=p->fmt;ci.extent={p->width,p->height,p->depth};ci.mipLevels=p->levels;ci.arrayLayers=p->layers;ci.samples=p->samples;ci.tiling=VK_IMAGE_TILING_OPTIMAL;ci.usage=usage;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        p->usage=usage;if(p->aspects==VK_IMAGE_ASPECT_COLOR_BIT && !no_mutable_images)ci.flags|=VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;p->mutable_format=(ci.flags&VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)!=0;
        if(t->d.target==8 || t->d.target==9)ci.flags|=VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        // 3D 纹理要能“按切片”当渲染目标，图像必须带 2D_ARRAY_COMPATIBLE —— 这是 Vulkan 里渲进 volume 切片的
        // 唯一正路（真机上由驱动保证切片与 3D 采样看到的是同一份存储）。设备可能对这种格式/用法组合不给
        // OPTIMAL 平铺，所以先用它探一次；不支持就撤回标志，退回“3D 不能当附件”的旧行为并记一笔。
        // A 3D texture used as a render target needs 2D_ARRAY_COMPATIBLE -- the only sanctioned way to render into a
        // volume slice (the driver guarantees the slice and 3D sampling see the same storage). The device may refuse
        // OPTIMAL tiling for this format/usage combination, so probe first; on refusal drop the flag and fall back to
        // the previous "3D cannot be an attachment" behaviour, leaving a note.
        const bool want_vol=t->d.target==2;p->vol_compat=want_vol;
        if(want_vol)ci.flags|=VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
        VkImageFormatProperties supported;VkResult qr=vkGetPhysicalDeviceImageFormatProperties(cfg.physical,ci.format,ci.imageType,ci.tiling,ci.usage,ci.flags,&supported);
        if(qr!=VK_SUCCESS && want_vol){
            ci.flags&=~(VkImageCreateFlags)VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;p->vol_compat=false;
            qr=vkGetPhysicalDeviceImageFormatProperties(cfg.physical,ci.format,ci.imageType,ci.tiling,ci.usage,ci.flags,&supported);}
        VK(qr);need(supported.sampleCounts&p->samples,"Texture sample count unsupported for actual format/usage");need(p->samples==VK_SAMPLE_COUNT_1_BIT || p->levels==1,"Multisample textures cannot have mipmaps");
        if(want_vol){static std::set<std::pair<uint32_t,unsigned>> seen;if(seen.insert({t->d.depth,unsigned(p->fmt)}).second)
            xbg_logf("NVN vol compat serial=%llu depth=%u mip=%u vk=%u supported=%d",(unsigned long long)t->serial,t->d.depth,t->d.levels,unsigned(p->fmt),int(p->vol_compat));}
        VK(vkCreateImage(cfg.device,&ci,nullptr,&p->image));p->create=ci;VkMemoryRequirements r;vkGetImageMemoryRequirements(cfg.device,p->image,&r);
        p->mem_bytes=int64_t(r.size);image_bytes_live+=p->mem_bytes;if(xbg_bc_format_info(t->d.format,&info)==0){p->bc_bytes=p->mem_bytes;p->bc_data=int64_t(t->L.size);bc_image_bytes_live+=p->bc_bytes;bc_data_bytes_live+=p->bc_data;}
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=r.size;ai.memoryTypeIndex=memory_type(r.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);VK(vkAllocateMemory(cfg.device,&ai,nullptr,&p->memory));VK(vkBindImageMemory(cfg.device,p->image,p->memory,0));
        it=images.emplace(t->serial,std::move(p)).first;image_index_dirty=true;++memo_check_gen;++memo_bump[4];provenance_image("CREATE",t,*it->second);
    }
    hit=it->second.get();front.serial=t->serial;front.epoch=images_epoch;front.im=hit;}
    Image &im=*hit;
    // NVN CPU-coherent pools may be written without FlushMappedRange. Detect
    // such writes once per submit; command CPU writes also invalidate below.
    if(upload && im.uploaded && !im.gpu_written && im.checked_submission!=stats.submissions){
        if(im.dirty_stamp && !ab_off("texdirty")){   // tracked: hash only after a guest write to its pages since the stamp
            uint64_t now=nvn_dirty_now();
            if(now!=im.dirty_seen && nvn_dirty_since(t->mem,t->L.size,im.dirty_stamp)){
                nvn_dirty_protect(t->mem,t->L.size);uint64_t stamp=nvn_dirty_now();++tex_dirty_checks;   // protect and stamp before reading
                if(hash_memory(t->mem,t->L.size)!=im.cpu_hash){im.uploaded=false;im.stable_checks=0;++tex_dirty_changed;}else im.dirty_stamp=stamp;
                now=stamp;}
            im.dirty_seen=now;}
        // Textures unchanged for 4 checks are re-verified every 16th submission (staggered). With tex_dirty the 4th unchanged
        // check write-protects a tracked texture instead (protect and stamp before the hash), and the tracker answers from then
        // on: protecting at upload faulted ~40k times per level load on the Redmi, as the loader went on writing those pages.
        else if(im.stable_checks<4 || ((stats.submissions+t->serial)&15)==0){
            const bool track=tex_dirty && im.stable_checks==3 && !ab_off("texdirty") && nvn_dirty_tracked && nvn_dirty_protect && nvn_dirty_now && nvn_dirty_since && nvn_dirty_tracked(t->mem,t->L.size);
            uint64_t stamp=0;if(track){nvn_dirty_protect(t->mem,t->L.size);stamp=nvn_dirty_now();}
            uint64_t hash=hash_memory(t->mem,t->L.size);if(hash!=im.cpu_hash){im.uploaded=false;im.stable_checks=0;}else{++im.stable_checks;if(track){im.dirty_stamp=im.dirty_seen=stamp;++tex_dirty_tracked;}}}
        im.checked_submission=stats.submissions;}
    // Diagnostic: CPU writes into textures whose contents the GPU produced (copies) are invisible to us.
    if(upload && im.gpu_written && im.decoded && t->mem && ((stats.submissions+t->serial)&63)==0 && im.checked_submission!=stats.submissions){im.checked_submission=stats.submissions;uint64_t hash=hash_memory(t->mem,t->L.size);if(im.cpu_hash && hash!=im.cpu_hash){static unsigned n=0;if(n++<40)xbg_logf("NVN CPUGPU texture changed by CPU after GPU write serial=%llu fmt=%#x %ux%u layers=%u levels=%u target=%d",(unsigned long long)t->serial,t->d.format,im.width,im.height,im.layers,im.levels,t->d.target);}im.cpu_hash=hash;}
    // Verification proved the differential output byte-identical over 6147 cycles; it costs a full decode per checked cycle, so
    // it is opt-in now (DM_BC_DIFFVERIFY=1) — leaving it on made every slot change pay a full decode and slowed bursts 4x.
    // 自校验已证明 6147 次差分输出与全量解码逐字节相同；它每校验一次就要多跑一次全量解码，所以改为按需开启
    // （DM_BC_DIFFVERIFY=1）——常开时每次换槽都付一次全量解码，突发帧慢了 4 倍。
    static int up_diff_verify=-1;if(up_diff_verify<0){const char *e=getenv("DM_BC_DIFFVERIFY");up_diff_verify=e&&*e=='1';}
    if(upload && !im.uploaded && !im.gpu_written){
        const uint64_t up_t0=timing_clock();const bool reupload=im.cpu_hash!=0;
        // Free the upload caches once soft-decoded BC re-uploads have stopped. Done here, before this upload touches them, and not
        // in recording_slot: stream_buffer below can start a recording while the scratch and the differential state are in use.
        // 软解 BC 重传停下来以后释放上传缓存。放在这里、在本次上传用到它们之前，而不放在 recording_slot：下面的 stream_buffer
        // 可能在暂存与差分状态还在用的时候开新录制。
        if(soft_bc_reupload_ns && up_t0-soft_bc_reupload_ns>=soft_bc_idle_ns){soft_bc_reupload_ns=0;upload_cache_release();}
        if(reupload && im.decoded)soft_bc_reupload_ns=up_t0;
        // Skip when the guest bytes equal what the device already holds. The streamed HDR cubes (BC6H 256x256x6) cycle
        // GetData -> CPU store -> Load 50+ times per frame on some scenes, and engine-side invalidation marks them dirty even
        // when the game restored identical bytes; each needless cycle costs a full BC decode. The same hash answers the
        // per-submission recheck above; using it here makes that question free.
        // The skip-check hash and the uploaded-content hash describe the same bytes on the same thread — compute once. On the
        // recurring ~110-small-image cycles the doubled hash was ~40% of the per-image cost (~0.25 ms of 0.65 ms).
        // 客户机字节与设备里已有内容相同时直接跳过上传。流式 HDR 立方图（BC6H 256x256x6）在某些场景里一帧要走
        // 50+ 次 GetData -> CPU 写 -> Load，游戏即使把同样的字节写了回去，引擎侧的失效也会把它标脏，
        // 每次多余循环都要付一次完整 BC 解码；上面每次提交的复查用的是同一个哈希，这里顺手复用，等于白拿。
        // 跳过判定用的哈希和上传后记录内容用的哈希是同一份字节、同一个线程算的，所以只算一次——
        // 在反复出现的 ~110 张小图循环里，算两遍约占单张成本 40%（0.65 ms 里的 0.25 ms）。
        uint64_t fresh_hash=0;const bool can_skip=t->mem&&im.cpu_hash;
        if(can_skip&&((fresh_hash=hash_memory(t->mem,t->L.size))==im.cpu_hash)){im.uploaded=true;im.checked_submission=stats.submissions;}
        else{
        im.dirty_stamp=0;   // a new upload starts on the hash checks again
        need(im.samples==VK_SAMPLE_COUNT_1_BIT,"Cannot upload uninitialized multisample image from CPU storage");
        need(t->mem && t->pool && t->off>=0 && uint64_t(t->off)<=t->pool->size && t->L.size<=t->pool->size-uint64_t(t->off),"Texture storage outside memory pool");
        if(im.aspects!=VK_IMAGE_ASPECT_COLOR_BIT)xbg_logf("NVN depth initial upload serial=%llu format=%#x vkFormat=%u target=%d extent=%ux%ux%u levels=%u layers=%u source=%p bytes=%llu",(unsigned long long)t->serial,t->d.format,unsigned(im.fmt),t->d.target,im.width,im.height,im.depth,im.levels,im.layers,t->mem,(unsigned long long)t->L.size);
        UpScratch &linear=up_linear;linear.reset();std::vector<VkBufferImageCopy> copies;
        // Choose the differential state for this image and pre-compute its sub-resource layout. | 为这张图挑出差分状态并预先算好子资源布局。
        UpDiff *dif=nullptr;uint8_t *out_base=nullptr;size_t run=0;
        // Any decoded BC image qualifies (the cube family is not the only one cycling): the slots are hotness-ordered, so a
        // load-time image never keeps a slot away from the images that actually re-upload every frame.
        // 任何解码 BC 图都够格（会循环的不止立方图那一族）：槽位按热度排序，载入期的图不会占着槽不让每帧重传的图用。
        if(up_diff_seen.size()>=up_diff_seen_max && !up_diff_seen.count(t->serial))up_diff_seen.clear();   // stale serials of freed images | 已释放图的旧序号
        unsigned &seen=up_diff_seen[t->serial];if(reupload)++seen;   // only images that really cycle are worth a slot | 只有真正在循环的图才配占槽
        if(im.decoded && im.depth==1 && t->L.size>=16ull*1024 && t->L.size<=up_diff_budget/32 && seen>=3){
            // Evict least recently used slots until a new one fits both the count (16) and the byte budget; the incoming image's
            // own size is not known before its layout, so its payload is capped above at budget/32 (decoded output is at most 8x the
            // BC payload, so one image stays under budget/4 and the total under the budget).
            // 按最久未用淘汰，直到新槽同时满足条数（16）与字节预算；进来的图在算布局前不知道大小，所以上面限单张负载 ≤ 预算/32
            // （解码输出最多是 BC 负载的 8 倍，单张不超过预算的 1/4，总量不超过预算）。
            if(!up_diffs.count(t->serial))while(!up_diffs.empty() && (up_diffs.size()>=16 || up_diff_bytes>up_diff_budget*3/4)){
                auto old=up_diffs.begin();for(auto i=up_diffs.begin();i!=up_diffs.end();++i)if(i->second.used<old->second.used)old=i;
                up_diff_bytes-=up_diff_size(old->second);up_diffs.erase(old);}
            UpDiff &d=up_diffs[t->serial];up_diff_bytes-=up_diff_size(d);d.used=++up_diff_clock;
            bool same=d.ready && d.fmt==uint32_t(t->d.format) && d.layers==im.layers && d.levels==im.levels && d.width==im.width && d.height==im.height && d.bpp==im.bpp;
            size_t ts=0,to=0;std::vector<uint32_t> offs;
            for(uint32_t layer=0;layer<im.layers;layer++)for(uint32_t level=0;level<im.levels;level++){
                uint32_t w=std::max(1u,im.width>>level),h=std::max(1u,im.height>>level);
                offs.push_back(uint32_t(ts));int bw=0,bh=0;int bb=nvn_fmt_bpb(t->d.format,&bw,&bh);
                ts+=size_t((w+bw-1)/bw)*((h+bh-1)/bh)*size_t(bb);
                to=(to+15)&~size_t(15);to+=size_t(w)*h*im.bpp;}
            if(!same || d.total!=to || d.total_src!=ts || d.off.size()!=offs.size())d=UpDiff{};
            d.fmt=t->d.format;d.layers=im.layers;d.levels=im.levels;d.width=im.width;d.height=im.height;d.bpp=im.bpp;
            d.total=to;d.total_src=ts;d.off=offs;
            d.src.resize(ts);d.out.resize(to);d.src_new.resize(ts);   // src_new is the deswizzle target of this cycle, including the first one | src_new 是本轮反 swizzle 的目标，第一轮也是
            up_diff_bytes+=up_diff_size(d);
            dif=&d;out_base=d.out.data();
        }
        for(uint32_t layer=0;layer<im.layers;layer++)for(uint32_t level=0;level<im.levels;level++){
            uint32_t w=std::max(1u,im.width>>level),h=std::max(1u,im.height>>level),depth=std::max(1u,im.depth>>level);
            if(im.aspects!=VK_IMAGE_ASPECT_COLOR_BIT){
                size_t bytes=size_t(w)*h*depth*im.bpp;need(bytes<=256ull*1024*1024,"Packed depth image too large");std::vector<uint8_t> packed(bytes);
                if(t->d.flags&0x10){need(!layer && !level && depth==1,"Linear depth mip/layer/volume layout unsupported");size_t row=t->d.stride?t->d.stride:(size_t(w)*im.bpp+31)&~size_t(31);need(row>=size_t(w)*im.bpp && row*h<=t->L.size,"Linear depth storage extent invalid");for(uint32_t y=0;y<h;y++)memcpy(packed.data()+size_t(y)*w*im.bpp,t->mem+y*row,size_t(w)*im.bpp);}
                else need(nvn_deswizzle(&t->d,level,layer,t->mem,packed.data(),size_t(w)*im.bpp)==0,"Depth image deswizzle failed");
                VkBufferImageCopy region{};region.imageSubresource={0,level,layer,1};region.imageExtent={w,h,depth};
                append_depth_planes(linear,copies,t->d.format,w,h,depth,packed.data(),packed.size(),0,0,region);continue;
            }
            if(!out_base)run=linear.size();   // track the running output size for the differential path | 差分路径下跟踪累计输出长度
            size_t begin=(run+15)&~size_t(15);
            int cbw=1,cbh=1;bool native_blocks=!im.decoded && nvn_fmt_bpb(t->d.format,&cbw,&cbh) && (cbw>1 || cbh>1);
            if(native_blocks){ // ASTC etc.: copy whole blocks, rows of blocks
                uint32_t bx=(w+cbw-1)/cbw,by=(h+cbh-1)/cbh;linear.resize(begin+size_t(bx)*by*depth*im.bpp);
                need(nvn_deswizzle(&t->d,level,layer,t->mem,linear.data()+begin,size_t(bx)*im.bpp)==0,"Compressed texture deswizzle failed");
                VkBufferImageCopy c{};c.bufferOffset=begin;c.imageSubresource={im.aspects,level,layer,1};c.imageExtent={w,h,depth};copies.push_back(c);continue;}
            if(!out_base)linear.resize(begin+size_t(w)*h*depth*im.bpp);else run=begin+size_t(w)*h*depth*im.bpp;
            if(im.decoded){
                int bw,bh,block_bytes=nvn_fmt_bpb(t->d.format,&bw,&bh);uint32_t blocks_x=(w+bw-1)/bw,blocks_y=(h+bh-1)/bh;
                UpScratch &blocks=up_blocks;if(!dif)blocks.resize(size_t(blocks_x)*blocks_y*depth*block_bytes);
                uint8_t *bdst=dif?dif->src_new.data()+dif->off[size_t(layer)*im.levels+level]:blocks.data();
                need(nvn_deswizzle(&t->d,level,layer,t->mem,bdst,size_t(blocks_x)*block_bytes)==0,"BC image deswizzle failed");
                uint8_t *odst=(dif?out_base:linear.data())+begin;
                if(dif&&dif->ready){   // decode only the blocks whose compressed bytes changed | 只解压缩字节发生了变化的块
                    const uint32_t bo=dif->off[size_t(layer)*im.levels+level];uint8_t pixels[128];
                    for(uint32_t by=0;by<blocks_y;by++)for(uint32_t bx=0;bx<blocks_x;bx++){
                        size_t bi=(size_t(by)*blocks_x+bx)*block_bytes;
                        if(!memcmp(dif->src.data()+bo+bi,bdst+bi,block_bytes))continue;
                        need(xbg_decode_block(t->d.format,bdst+bi,block_bytes,pixels,sizeof(pixels))==0,"BC block decode failed");
                        for(unsigned y=0;y<4 && by*4+y<h;y++)for(unsigned x=0;x<4 && bx*4+x<w;x++)
                            memcpy(odst+(((size_t)by*4+y)*w+bx*4+x)*im.bpp,pixels+(y*4+x)*im.bpp,im.bpp);}
                }else{
                    need(xbg_decode_bc_linear(t->d.format,w,h,depth,bdst,size_t(blocks_x)*blocks_y*depth*block_bytes,0,0,odst,size_t(w)*h*depth*im.bpp)==0,"BC image decode failed");}
            }else if(t->d.flags&0x10){need(!layer && !level && depth==1,"Linear mip/layer/volume texture not implemented");size_t pitch=t->d.stride?t->d.stride:(w*im.bpp+31)&~size_t(31);for(uint32_t y=0;y<h;y++)memcpy(linear.data()+begin+y*w*im.bpp,t->mem+y*pitch,w*im.bpp);}
            else need(nvn_deswizzle(&t->d,level,layer,t->mem,linear.data()+begin,size_t(w)*im.bpp)==0,"Texture deswizzle failed");
            VkBufferImageCopy c{};c.bufferOffset=begin;c.imageSubresource={im.aspects,level,layer,1};c.imageExtent={w,h,depth};copies.push_back(c);
        }
        const size_t up_bytes=out_base?run:linear.size();
        Buffer staging=stream_buffer(up_bytes,out_base?out_base:linear.data(),up_bytes,false);for(auto &copy:copies)copy.bufferOffset+=staging.offset;
        // gfx-upload-prologue：这次提交里还没有命令用过这张图，它的新内容可以排在整个提交之前，前面没有读到旧内容的命令。
        // gfx-upload-prologue: no command of this submission used this image yet, so its new contents can go ahead of the whole
        // submission; no earlier command read the old contents.
        if(upload_prologue && im.slot_use!=slot_serial && !ab_off("uppro")){Slot &s=recording_slot();
            s.pro_images.push_back({im.image,im.layout,im.aspects,im.levels,im.layers,staging.buffer,copies});im.layout=VK_IMAGE_LAYOUT_GENERAL;++uploads_prologue;}
        else{++uploads_inpass;if(open_pass.cmd)++uploads_broke_pass;
            Commands c;im.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);vkCmdCopyBufferToImage(c.command,staging.buffer,im.image,im.layout,copies.size(),copies.data());im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();}
        im.uploaded=true;provenance_write(im,__LINE__);im.cpu_hash=can_skip?fresh_hash:hash_memory(t->mem,t->L.size);im.checked_submission=stats.submissions;
        if(dif){   // this cycle's compressed source becomes the next comparison base | 本轮的压缩源成为下一轮的比较基准
            dif->src.swap(dif->src_new);
            if(!dif->ready)dif->ready=true;
            else if(dif->verify && up_diff_verify){--dif->verify;   // opt-in: compare against a full decode of the same bytes | 按需开启：与同样字节的全量解码结果对比
                bool bad=false;size_t at=0;std::vector<uint8_t> ref(dif->total);
                for(uint32_t layer=0;layer<dif->layers&&!bad;layer++)for(uint32_t level=0;level<dif->levels&&!bad;level++){
                    uint32_t w=std::max(1u,dif->width>>level),h=std::max(1u,dif->height>>level);
                    int bw=0,bh=0;int bb=nvn_fmt_bpb(dif->fmt,&bw,&bh);uint32_t bxs=(w+bw-1)/bw,bys=(h+bh-1)/bh;
                    size_t begin=(at+15)&~size_t(15);at=begin+size_t(w)*h*dif->bpp;
                    if(xbg_decode_bc_linear(dif->fmt,w,h,1,dif->src.data()+dif->off[size_t(layer)*dif->levels+level],size_t(bxs)*bys*size_t(bb),0,0,ref.data()+begin,size_t(w)*h*dif->bpp))bad=true;
                    else if(memcmp(ref.data()+begin,dif->out.data()+begin,size_t(w)*h*dif->bpp))bad=true;}
                xbg_logf("NVN bd-verify serial=%llu %s",(unsigned long long)t->serial,bad?"MISMATCH - differential disabled for this image":"ok");
                if(bad){dif->ready=false;dif->verify=0;}}
        }
        if(reupload)++hud::reuploads;else ++hud::uploads;hud::upload_bytes+=up_bytes;hud::upload_ns+=timing_clock()-up_t0;
        }
    }
    provenance_image(upload?"GET_UPLOAD":"GET",t,im);im.slot_use=slot_serial;return im;
}
// read_preflight: main's order (cc91eb1). Copy the descriptor, resolve CPU reads of its storage, then look the image up.
Image &get_image_read(const tex_t *t,bool upload){
    need(t && t->serial && t->magic==MAGIC('T','E','X',0),"Invalid NVN texture");
    need(!t->mem || xbg_wb_range_valid(uint64_t(uintptr_t(t->mem)),t->L.size),"Texture CPU read range overflow or empty");
    bool read=upload || images.find(t->serial)==images.end();
    return xbg_image_after_cpu_access(*t,read,
        [](uint64_t address,uint64_t bytes){cpu_access(address,bytes,9);},
        [upload](const tex_t &stable)->Image &{return get_image_slow(&stable,upload);});
}
const RegisteredTexture &null_texture(const xbg_shader_binding &b){
    // Original host evidence: main+0x1746820 explicitly binds descriptor 0xFFFFF
    // for unused slots; descriptor 0 is the game's null handle. Only those
    // defined null cases receive zero resources. Missing in-pool IDs still fail.
    need(b.type!=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && b.type!=VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,"Null writable image descriptor needs robustness2 nullDescriptor support");
    uint32_t key=b.flags&0xffffff00u;auto it=null_textures.find(key);if(it!=null_textures.end())return it->second;
    unsigned dim=(b.flags&(1u<<31))?(b.flags>>8)&7:1;bool array=b.flags&(1u<<12);unsigned scalar=(b.flags>>16)&3;
    need(!(b.flags&(1u<<13)),"Null multisample texture descriptor unsupported");
    RegisteredTexture r;tex_t &t=r.texture;t.magic=MAGIC('T','E','X',0);t.serial=nvn_new_serial();t.d.width=t.d.height=t.d.depth=t.d.levels=1;
    t.d.format=b.flags&(1u<<14)?0x34:scalar==1?0x2f:scalar==2?0x30:dim==5?0x2e:0x25;
    t.d.target=dim==0?(array?3:0):dim==1?(array?4:1):dim==2?2:dim==3?(array?9:8):dim==5?10:-1;need(t.d.target>=0,"Unknown null texture dimension");
    nvn_tex_layout(&t.d,&t.L);
    if(dim==5){static uint8_t zero[65536]{};t.mem=zero;t.L.size=sizeof(zero);}
    else{Image &im=get_image(&t,false);Commands c;im.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageSubresourceRange range{im.aspects,0,1,0,im.layers};if(im.aspects==VK_IMAGE_ASPECT_COLOR_BIT){VkClearColorValue zero{};vkCmdClearColorImage(c.command,im.image,im.layout,&zero,1,&range);}else{VkClearDepthStencilValue zero{};vkCmdClearDepthStencilImage(c.command,im.image,im.layout,&zero,1,&range);}im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();im.gpu_written=true,provenance_write(im,__LINE__);}
    xbg_logf("NVN defined null sampled resource: dim=%u array=%d scalar=%u",dim,array,scalar);return null_textures.emplace(key,r).first->second;
}

/* SHA-256 over the original NVN 0x880-byte control segment. */
std::string control_key(const void *data){char key[17];need(xbg_shader_control_key(data,key)==0,"Missing shader control segment");return key;}
struct Shader {std::string key;uint32_t stage;VkShaderModule module{};std::vector<xbg_shader_binding> bindings;std::vector<uint32_t> inputs,input_classes;std::vector<uint8_t> c1;
    std::set<uint32_t> storage_writes; // Vulkan bindings of storage buffers the SPIR-V may write
    bool discards=true; // fragment shader may kill/demote (conservative default)
    XbgRasterFacts raster; // position input (vertex), declared outputs and SampleMask (fragment): first_draw_covers
    Buffer c1_buffer{};bool c1_ready=false;
    ~Shader(){if(module)vkDestroyShaderModule(cfg.device,module,nullptr);}};
// Storage buffers written by a shader. Pointers rooted at a storage-buffer
// variable that reach OpStore/OpCopyMemory/atomics, function calls, phis or
// selects mark that binding written; anything unprovable marks all written.
// Bytes of each uniform block a shader can actually read (iOS c1/cbuf packing idea): Ryujinx declares
// constant buffers as struct{vec4 data[4096];}. When every access uses a constant element index, only
// (max index + 1) * 16 bytes need copying per draw; any other use keeps the full block.
std::map<uint32_t,uint32_t> spirv_uniform_extent(const std::vector<uint32_t> &w){
    std::map<uint32_t,uint32_t> binding,constant,stride,member_type,var_type,result;std::set<uint32_t> uniform_vars,full;std::map<uint32_t,uint32_t> max_index;
    std::map<uint32_t,uint32_t> pointer_pointee,struct_member0;
    for(size_t i=5;i<w.size();){uint32_t n=w[i]>>16,op=w[i]&65535;if(!n||i+n>w.size())return {};
        switch(op){
        case 71:if(n>=4 && w[i+2]==33)binding[w[i+1]]=w[i+3];if(n>=4 && w[i+2]==6)stride[w[i+1]]=w[i+3];break; // Binding / ArrayStride
        case 43:if(n>=4)constant[w[i+2]]=w[i+3];break; // OpConstant (32-bit)
        case 30:if(n>=3)struct_member0[w[i+1]]=w[i+2];break; // OpTypeStruct member 0
        case 32:if(n>=4)pointer_pointee[w[i+1]]=w[i+3];break;
        case 59:if(n>=4 && w[i+3]==2){uniform_vars.insert(w[i+2]);var_type[w[i+2]]=w[i+1];}break; // OpVariable Uniform
        default:break;
        }
        i+=n;
    }
    for(size_t i=5;i<w.size();){uint32_t n=w[i]>>16,op=w[i]&65535;
        if(op==59){i+=n;continue;}
        if(op==5||op==6||op==15||op==71||op==72||op==332||op==5632){i+=n;continue;} // names/entry-point interface/decorations reference without reading
        for(uint32_t k=1;k<n;k++){uint32_t id=w[i+k];if(!uniform_vars.count(id))continue;
            bool ok=(op==65||op==66) && k==3 && n>=6;
            if(ok){auto c=constant.find(w[i+5]);auto m=constant.find(w[i+4]);if(c==constant.end()||m==constant.end()||m->second!=0)ok=false;else max_index[id]=std::max(max_index[id],c->second);}
            if(!ok)full.insert(id);}
        i+=n;
    }
    for(uint32_t v:uniform_vars){auto b=binding.find(v);if(b==binding.end()||full.count(v))continue;
        auto pt=pointer_pointee.find(var_type[v]);if(pt==pointer_pointee.end())continue;auto m0=struct_member0.find(pt->second);if(m0==struct_member0.end())continue;
        auto st=stride.find(m0->second);if(st==stride.end()||st->second!=16)continue;
        auto mi=max_index.find(v);result[b->second]=mi==max_index.end()?16:(mi->second+1)*16;}
    return result;
}
std::set<uint32_t> spirv_storage_writes(const std::vector<uint32_t> &w,bool &unknown){
    std::map<uint32_t,uint32_t> binding,storage_class,root;std::set<uint32_t> written_vars;unknown=false;
    std::set<uint32_t> buffer_block_types;std::map<uint32_t,uint32_t> pointer_pointee;
    for(size_t i=5;i<w.size();){uint32_t n=w[i]>>16,op=w[i]&65535;if(!n||i+n>w.size()){unknown=true;break;}
        auto mark=[&](uint32_t ptr){auto r=root.find(ptr);if(r!=root.end())written_vars.insert(r->second);};
        switch(op){
        case 71:if(n>=4 && w[i+2]==33)binding[w[i+1]]=w[i+3];if(n>=3 && w[i+2]==3)buffer_block_types.insert(w[i+1]);break; // OpDecorate Binding/BufferBlock
        case 32:if(n>=4)pointer_pointee[w[i+1]]=w[i+3];break; // OpTypePointer
        case 59:if(n>=4){storage_class[w[i+2]]=w[i+3];root[w[i+2]]=w[i+2];}break; // OpVariable
        // Function parameters: call sites passing storage pointers are marked
        // at OpFunctionCall, so stores through parameters need no extra mark.
        case 55:if(n>=3)root[w[i+2]]=0xffffffffu;break;
        case 65:case 66:case 67:case 83:case 124:if(n>=4){auto r=root.find(w[i+3]);if(r!=root.end())root[w[i+2]]=r->second;}break;
        case 62:case 63:case 228:if(n>=2){if(root.count(w[i+1]))mark(w[i+1]);else unknown=true;}break; // OpStore/OpCopyMemory/OpAtomicStore
        case 57:for(uint32_t k=4;k<n;k++)mark(w[i+k]);break; // pointer arguments to calls
        case 169:case 245:for(uint32_t k=3;k<n;k++)mark(w[i+k]);break; // select/phi over pointers
        default:if(((op>=229 && op<=242) || op==6035 || op==5250) && n>=4){if(root.count(w[i+3]))mark(w[i+3]);else unknown=true;}break;
        }
        i+=n;
    }
    std::set<uint32_t> out;
    for(uint32_t v:written_vars){auto c=storage_class.find(v);if(c==storage_class.end())continue;if(c->second==12 || c->second==2){auto b=binding.find(v);if(b!=binding.end())out.insert(b->second);else unknown=true;}}
    return out;
}
std::map<std::string,std::unique_ptr<Shader>> shaders;
const uint32_t nvn_stage[6]={0,3,4,2,1,5};
struct State {
    p_set_rt rt{};Shader *shader[6]{};NVNbufferRange ub[6][16]{},sb[6][16]{},vb[16]{};
    uint64_t tex[6][128]{},samp[6][128]{},img[6][16]{};dpool_t texture_pool{},sampler_pool{};
    uint32_t attrib[16]{};vs_st stream[16]{};blend_st blend[8]{};color_st color{};chmask_st mask{0xffffffff};ms_st ms{};poly_st poly{0,1,2,0};ds_st ds{0,1,2,0,8,8,1,1,1,1,1,1};
    VkViewport viewport{0,0,1,1,0,1};VkRect2D scissor{{0,0},{UINT32_MAX,UINT32_MAX}};float blend_color[4]{},bias[3]{};uint32_t stencil_compare[2]{~0u,~0u},stencil_write[2]{~0u,~0u},stencil_ref[2]{};int64_t copy_stride=0,copy_image_stride=0;uint32_t patch_size=3;
    xbg_depth_bounds bounds{0,0,1};bool count_samples=false;uint64_t samples_passed=0;
} state;
// np701：每 draw 记本帧绑定过的着色器（判死停帧清单）。便宜：6 次空判 + 6 次指针比较。
// np701: record the frame's bound shaders per draw (death manifest). Cheap: 6 null checks + 6 pointer compares.
void deathguard::draw_bound(){
    for(unsigned st=0;st<6;st++)if(state.shader[st])shader_used(st,state.shader[st],state.shader[st]->key.c_str());
}
// Pipeline memo (gfx-pipe-memo): every input the draw's VkGraphicsPipelineCreateInfo is built from. When a draw's inputs
// equal the previous draw's, it gets the previous draw's pipeline without serializing the create info into a key and
// looking it up (xbg_graphics_pipeline_key with its module-id lookups was ~6% of the game thread on Adreno 730). Raw
// copies of the state fields, so differing padding can only cost a miss. Off unless the host asks (DM_GFX_PIPE_MEMO, which
// the Android app sets to 1); marker gfx-pipe-memo = 1 / 0; A/B target pipememo.
struct PipeMemoKey{const void *shader[5];const void *layout,*pass_def;VkRenderPass pass;VkPipelineLayout pipeline_layout;int32_t prim;uint32_t samples,rt_n,colors,depth,depth_format;
    uint8_t fixed[offsetof(State,viewport)-offsetof(State,attrib)];               // attrib, stream, blend, color, mask, ms, poly, ds
    uint8_t stencil[offsetof(State,copy_stride)-offsetof(State,blend_color)];      // blend_color, bias, stencil compare / write / reference
    uint8_t tail[offsetof(State,count_samples)-offsetof(State,patch_size)];};      // patch_size, bounds
PipeMemoKey pipe_memo_key,pipe_memo_next;VkPipeline pipe_memo_pipeline=VK_NULL_HANDLE;bool pipe_memo=false;uint64_t pipe_memo_hits=0,pipe_memo_misses=0;
void pipe_memo_fill(PipeMemoKey &k,const void *layout,const void *pass_def,VkRenderPass pass,VkPipelineLayout pipeline_layout,int prim,VkSampleCountFlagBits samples,uint32_t colors){
    memset(&k,0,sizeof k);for(unsigned i=0;i<5;i++)k.shader[i]=state.shader[i];k.layout=layout;k.pass_def=pass_def;k.pass=pass;k.pipeline_layout=pipeline_layout;k.prim=prim;k.samples=uint32_t(samples);
    k.rt_n=uint32_t(state.rt.n);k.colors=colors;k.depth=state.rt.depth!=nullptr;k.depth_format=state.rt.depth?uint32_t(state.rt.depth->d.format):0;
    const uint8_t *b=reinterpret_cast<const uint8_t*>(&state);memcpy(k.fixed,b+offsetof(State,attrib),sizeof k.fixed);memcpy(k.stencil,b+offsetof(State,blend_color),sizeof k.stencil);memcpy(k.tail,b+offsetof(State,patch_size),sizeof k.tail);
}
void validate_capabilities(const std::vector<uint32_t> &w,uint32_t stage){
    need(w.size()>=5 && w[0]==0x07230203,"Invalid packaged SPIR-V");bool group_operations=false;VkSubgroupFeatureFlags groups=0;
    for(size_t i=5;i<w.size();){uint32_t count=w[i]>>16,op=w[i]&65535;need(count && i+count<=w.size(),"Invalid SPIR-V instruction bounds");if(op>=333 && op<=366)group_operations=true;
        if(op==17){uint32_t cap=w[i+1];bool ok=true;switch(cap){case 1:case 46:case 47:case 50:break;case 2:ok=cfg.enabled_core.geometryShader;break;case 3:ok=cfg.enabled_core.tessellationShader;break;case 10:ok=cfg.enabled_core.shaderFloat64;break;case 11:ok=cfg.enabled_core.shaderInt64;break;case 22:ok=cfg.enabled_core.shaderInt16;break;case 25:ok=cfg.enabled_core.shaderImageGatherExtended;break;case 32:ok=cfg.enabled_core.shaderClipDistance;break;case 33:ok=cfg.enabled_core.shaderCullDistance;break;case 35:ok=cfg.enabled_core.sampleRateShading;break;case 49:ok=cfg.enabled_core.shaderStorageImageExtendedFormats;break;case 55:ok=cfg.enabled_core.shaderStorageImageReadWithoutFormat;break;case 56:ok=cfg.enabled_core.shaderStorageImageWriteWithoutFormat;break;case 61:groups|=VK_SUBGROUP_FEATURE_BASIC_BIT;break;case 62:groups|=VK_SUBGROUP_FEATURE_VOTE_BIT;break;case 63:groups|=VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;break;case 64:groups|=VK_SUBGROUP_FEATURE_BALLOT_BIT;break;case 65:groups|=VK_SUBGROUP_FEATURE_SHUFFLE_BIT;break;case 66:groups|=VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;break;case 4427:ok=cfg.enabled_shader_draw_parameters;break;default:xbg_fatalf("Shader stage %u has unhandled capability %u",stage,cap);}if(!ok)xbg_fatalf("Shader stage %u requires disabled capability %u",stage,cap);}
        if(op==10)xbg_fatalf("Shader stage %u declares extension requiring explicit backend support",stage);i+=count;
    }
    need((subgroup.supportedOperations&groups)==groups,"Required subgroup operations unavailable");if(group_operations)need(subgroup.supportedStages&(1u<<stage),"Actual subgroup operation unsupported in shader stage");
}
// Fragment-shader quad-neighbour shuffles rewritten to fine derivatives (xbg_spirv_quad.cpp): on Adreno the shuffles read
// garbage in helper invocations. The conservative correction skips bit patterns and unproved control/index flows; the
// translator's SHFL helper calls are rewritten at the call site when they are proven quad swaps of float values in uniform
// control flow (D2R's form: 271 of its 278 calls). Both module
// loaders apply it, so the warm-up builds the same modules the game binds. Off unless the host asks
// (DM_GFX_QUAD_DERIVATIVES, which the Android app sets to 1; marker gfx-quad-derivatives = 1 / 0).
bool quad_derivatives=false;std::atomic<uint64_t> quad_rewrites{0};
void quad_rewrite(std::vector<uint32_t> &words,uint32_t stage,const char *key,const char *loader){
    if(!quad_derivatives || stage!=4)return;
    std::vector<uint32_t> out;xbg_spirv_quad_stats counts;
    size_t k=xbg_spirv_quad_derivatives(words.data(),words.size(),out,&counts);
    if(k){words.swap(out);quad_rewrites.fetch_add(k,std::memory_order_relaxed);}
    if(counts.candidates || counts.calls){
        char line[768];int at=snprintf(line,sizeof(line),"NVN quad shader=%s loader=%s candidates=%zu rewritten=%zu guarded_nonfinite=%zu calls=%zu calls_rewritten=%zu",key,loader,counts.candidates,counts.rewritten,counts.guarded_nonfinite,counts.calls,counts.calls_rewritten);
        for(size_t r=0;r<size_t(xbg_quad_skip::count) && at<int(sizeof(line))-64;++r)if(counts.skipped[r])at+=snprintf(line+at,sizeof(line)-at," skip_%s=%zu",xbg_spirv_quad_skip_name(xbg_quad_skip(r)),counts.skipped[r]);
        for(size_t r=0;r<size_t(xbg_quad_call_skip::count) && at<int(sizeof(line))-64;++r)if(counts.call_skipped[r])at+=snprintf(line+at,sizeof(line)-at," call_skip_%s=%zu",xbg_spirv_quad_call_skip_name(xbg_quad_call_skip(r)),counts.call_skipped[r]);
        xbg_logf("%s",line);
    }
}
// Compute modules translated for 32-lane warps, on wider subgroups (xbg_spirv_warp.cpp): every 32 lanes act as their own
// warp, as the Ryujinx translator emits for wider hosts. On Adreno (64-lane subgroups) the clutter, grass, fluid and PPR
// shaders' warp-aggregated appends otherwise lose lanes 32-63 and overwrite entries. Both module loaders apply it. Off unless
// the host asks (DM_GFX_WARP32, which the Android app sets to 1; marker gfx-warp32 = 1 / 0), and only where subgroups are
// wider than 32 lanes.
bool warp32=false;std::atomic<uint64_t> warp_rewrites{0};
void warp_rewrite(std::vector<uint32_t> &words,uint32_t stage,const char *key,const char *loader){
    if(!warp32 || stage!=5 || subgroup.subgroupSize<=32)return;
    std::vector<uint32_t> out;size_t k=xbg_spirv_warp32(words.data(),words.size(),subgroup.subgroupSize,out);
    if(k){
        // A vote-only module can acquire Ballot capability during rewriting; check the transformed requirements too.
        validate_capabilities(out,stage);words.swap(out);warp_rewrites.fetch_add(k,std::memory_order_relaxed);
        xbg_logf("NVN warp32 shader=%s loader=%s subgroupSize=%u rewrites=%zu",key,loader,subgroup.subgroupSize,k);
    }
}
// Two exact clean-ups of the translator's output, after the rewrites above (they match its form), in both module loaders. Each
// is off unless the host asks; the Android app sets both to 1, and a marker file (1 / 0) still wins.
// - Constant-buffer component fold (xbg_spirv_ldc.cpp; DM_GFX_LDC_FOLD, marker gfx-ldc-fold): an LDC with a computed address
//   is addressed by 32-bit word, so a skinning shader selects each bone-matrix component dynamically, although bone * 48 makes
//   every component a constant.
// - Single-store temporaries forwarded (xbg_spirv_temps.cpp; DM_GFX_FORWARD_TEMPS, marker gfx-forward-temps): every value
//   sits in its own Function variable, stored once and loaded in the same block. Forwarded after the fold, the folded
//   constants reach the access chains directly.
// Redmi K50 Ultra camp at scale 0.75 with both: GPU 15.70 -> 14.34 ms a frame; the shadow map 1,498 -> 711 us, the main pass
// 6,019 -> 5,435 us, the depth pre-pass 1,356 -> 1,059 us. Either one alone changed nothing measurable.
bool ldc_fold=false,forward_temps=false;std::atomic<uint64_t> ldc_folds{0},temps_forwarded{0};
// Maxwell 的浮点 min / max 与饱和（FMNMX、.SAT）遇 NaN 取另一个操作数（饱和得 0）；翻译器出的是 GLSL.std.450 的 FMin / FMax /
// FClamp，有 NaN 操作数时结果 Vulkan 不作规定，驱动可以把 NaN 传下去，写进 UNORM 目标就成了黑块。DM_GFX_NAN_MINMAX（标记
// gfx-nan-minmax）把它们原位换成 NMin / NMax / NClamp：操作数不变，非 NaN 时结果相同，遇 NaN 时与 Maxwell 一致。
// Maxwell's float min / max and saturation (FMNMX, .SAT) take the other operand when one is NaN (saturation gives 0); the
// translator emits GLSL.std.450 FMin / FMax / FClamp, whose result with a NaN operand Vulkan leaves undefined, so a driver may
// pass the NaN on, and written to a UNORM target it comes out black. DM_GFX_NAN_MINMAX (marker gfx-nan-minmax) swaps them in
// place for NMin / NMax / NClamp: the same operands, the same result for values that are not NaN, and Maxwell's result for NaN.
bool nan_minmax=false;std::atomic<uint64_t> nan_minmax_swaps{0};
size_t nan_minmax_swap(std::vector<uint32_t> &w){
    uint32_t glsl=0;size_t k=0;
    for(size_t i=5;i<w.size();){
        uint32_t op=w[i]&0xFFFFu,n=w[i]>>16;if(!n || i+n>w.size())break;
        if(op==11 && n>=5 && !memcmp(&w[i+2],"GLSL.std.450",13))glsl=w[i+1];
        else if(op==12 && n>=5 && glsl && w[i+3]==glsl){uint32_t &e=w[i+4];if(e==37){e=79;++k;}else if(e==40){e=80;++k;}else if(e==43){e=81;++k;}}
        i+=n;
    }
    return k;
}
void tidy_rewrite(std::vector<uint32_t> &words){
    std::vector<uint32_t> out;
    if(nan_minmax){size_t k=nan_minmax_swap(words);if(k)nan_minmax_swaps.fetch_add(k,std::memory_order_relaxed);}
    if(ldc_fold){size_t k=xbg_spirv_ldc_fold(words.data(),words.size(),out);if(k){words.swap(out);ldc_folds.fetch_add(k,std::memory_order_relaxed);}}
    if(forward_temps){size_t k=xbg_spirv_forward_temps(words.data(),words.size(),out);if(k){words.swap(out);temps_forwarded.fetch_add(k,std::memory_order_relaxed);}}
}
// gfx-inline：子组指令写在被调用的辅助函数里的着色器（导入包里 187 个），载入时把辅助函数全部内联进入口（SPIRV-Tools 的
// merge-return、inline-entry-points-exhaustive、eliminate-dead-functions，与 spirv-opt 同名三步，结果逐字节相同）；
// Adreno 750 族缺省打开，见 xbg_adreno_policy.h。SPIRV-Tools 只编进安卓包（XBG_HAVE_SPIRV_OPT），别的平台这一步不存在。
// gfx-inline: a shader whose subgroup instructions sit in a called helper (187 in the import pack) gets every helper inlined
// into the entry point at load (the SPIRV-Tools passes merge-return, inline-entry-points-exhaustive and
// eliminate-dead-functions, the same three as spirv-opt, byte-identical output); on by default on the Adreno 750 family, see
// xbg_adreno_policy.h. SPIRV-Tools is built into the Android package only (XBG_HAVE_SPIRV_OPT); elsewhere this step is absent.
bool inline_helpers=false;
bool helper_has_subgroup(const std::vector<uint32_t> &w){
    std::unordered_set<uint32_t> entry;uint32_t cur=0;bool in=false;
    for(size_t i=5;i<w.size();){const uint32_t op=w[i]&0xffff,wc=w[i]>>16;if(!wc || i+wc>w.size())return false;
        if(op==15 && wc>2)entry.insert(w[i+2]);else if(op==54 && wc>2){cur=w[i+2];in=true;}else if(op==56)in=false;
        else if(op>=333 && op<=366 && in && !entry.count(cur))return true;i+=wc;}
    return false;
}
bool inline_rewrite(std::vector<uint32_t> &words,const std::string &key){
    if(!inline_helpers || !helper_has_subgroup(words))return false;
#ifdef XBG_HAVE_SPIRV_OPT
    spvtools::Optimizer o(SPV_ENV_VULKAN_1_1);o.RegisterPass(spvtools::CreateMergeReturnPass());o.RegisterPass(spvtools::CreateInlineExhaustivePass());o.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());
    std::vector<uint32_t> out;if(o.Run(words.data(),words.size(),&out) && !helper_has_subgroup(out)){words.swap(out);return true;}
    xbg_logf("NVN shader %s: inlining its helpers failed; the bundle is used as it is (gfx-inline)",key.c_str());
#else
    (void)key;
#endif
    return false;
}
// Diagnostic (files/dm/spv-override/<key>.spv): this SPIR-V replaces the bundle's, for offline-optimized test modules.
bool spv_override(const std::string &key,std::vector<uint32_t> &words){
    std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');std::string path=(cut==std::string::npos?dir:dir.substr(0,cut))+"/spv-override/"+key+".spv";
    FILE *f=fopen(path.c_str(),"rb");if(!f)return false;fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);if(n<20 || n%4){fclose(f);return false;}
    std::vector<uint32_t> w(size_t(n)/4);bool ok=fread(w.data(),1,size_t(n),f)==size_t(n);fclose(f);if(!ok || w[0]!=0x07230203u)return false;
    words.swap(w);xbg_logf("NVN spv override %s words=%zu",key.c_str(),words.size());return true;
}
// np525 ★ 缺着色器包时把现场留下来。
//   为什么：原来只在 fatal 里报一句"缺 xxx.xgs"，证据全丢，于是分不清两种完全不同的情况——
//   ① 这款游戏确实多了一个着色器（离线索引该重出）；② 运行时控制块与离线静态块**系统性不同**
//      （那样 17,308 个包会全部对不上，整条离线链要按运行时形态重算）。
//   做法：把控制段原样 0x880 字节写到 <shader_directory>/<键>.ctl（只写 .ctl，不影响 .xgs 计数/校验），
//   离线拿它与 romfs 里最接近的控制段逐字节比对即可定谳。
// np525: keep the evidence when a bundle is missing ...
static void dump_missing_shader(const std::string &key,const void *control,uint64_t code){
    if(!control)return;
    std::string path=std::string(cfg.shader_directory)+"/"+key+".ctl";
    bool ok=false;
    if(FILE *f=fopen(path.c_str(),"wb")){ok=fwrite(control,1,0x880,f)==0x880;fclose(f);}
    xbg_logf("NVN missing shader %s: control@%p dumped=%d -> %s",key.c_str(),control,int(ok),path.c_str());
    // 沙箱里的 .ctl 沙盒外读不到（shell 受 SELinux 限制），所以再把字节按 hex 打进日志：
    // hilog 能读到，离线拼回 2176 字节即可。xbg_logf 的行缓冲是 1024，每段 0x180（=768 hex 字符）留足余量。
    const uint8_t *p=reinterpret_cast<const uint8_t *>(control);
    for(size_t off=0;off<0x880;off+=0x180){
        char line[1024];int n=snprintf(line,sizeof line,"ctlhex %s +%03zx ",key.c_str(),off);
        for(size_t i=0;i<0x180 && off+i<0x880;i++)n+=snprintf(line+n,sizeof line-n,"%02x",p[off+i]);
        xbg_logf("%s",line);
    }
    // np530：下面的"深度取证"（codehint/codehex/ptrprobe/datahex）会去读指针目标，
    //   其中 ptrprobe 读的是"同 4GB 段"的候选地址——那不等于一定有效，实测会把进程读崩。
    //   所以只留 .ctl 与 ctlhex（控制块本身一定有效），深度取证改成 DM_SHADER_DUMP=1 显式开。
    // np530: the deep dumps below dereference pointer targets (ptrprobe only checks the 4 GB segment, which is not
    //   proof of validity) and did crash the process. Keep the always-safe .ctl + ctlhex; make the rest opt-in.
    const char *dv=getenv("DM_SHADER_DUMP");if(!dv || *dv!='1')return;
    // 控制块尾部三个 8 字节字段是指向**代码段**的指针（np525 实测：同一次崩点两轮只在这 6 个字节上不同）。
    // 想把键改成"代码段哈希"就必须先证明能拿到代码，所以把最小那个指针起 4 KB 也打出来，
    // 离线拿它所有前缀的哈希与 17,308 个包记录的代码哈希求交——命中就说明这条路可行。
    uint64_t hint=0;
    for(size_t off=0x848;off<=0x860;off+=8){uint64_t v=0;memcpy(&v,p+off,8);if(v && (!hint || v<hint))hint=v;}
    // np537：这两个字段不一定是指针（实测 65f0 这个着色器里就是 0x8f0 这种小整数），
    //   直接按指针读会把进程读崩 ⇒ 只在与控制块同 4GB 段时才去解引用。
    if(hint && (hint>>32)!=(reinterpret_cast<uint64_t>(control)>>32))hint=0;
    if(hint){
        xbg_logf("codehint %s %p",key.c_str(),reinterpret_cast<void*>(uintptr_t(hint)));
        const uint8_t *c=reinterpret_cast<const uint8_t *>(uintptr_t(hint));
        for(size_t off=0;off<0x1000;off+=0x180){
            char line[1024];int n=snprintf(line,sizeof line,"codehex %s +%03zx ",key.c_str(),off);
            for(size_t i=0;i<0x180 && off+i<0x1000;i++)n+=snprintf(line+n,sizeof line-n,"%02x",c[off+i]);
            xbg_logf("%s",line);
        }
    }
    // np526：控制块里像指针的字段，逐个看它指向哪——目标是代码段的话，头 4 字节应是 0x12345678。
    // 只认与控制块同一段（高 24 位相同）的地址，既过滤掉小整数/魔数，也把越界读的风险压到很低。
    {
        const uint64_t self=reinterpret_cast<uint64_t>(control);
        for(size_t off=0;off<0x880;off+=8){
            uint64_t v=0;memcpy(&v,p+off,8);
            // 只认与控制块**高 32 位相同**的地址（同一个 4GB 段）：既滤掉小整数/魔数，也避免野指针。
            // 注意别用 ~0ull<<40 之类的掩码——这台机器的地址只有 40 位，高 24 位全 0，相与恒为 0，条件会永远不成立。
            if(v<=0x10000)continue;                       // 小整数 / 魔数，不碰
            if((v>>32)!=(self>>32))continue;              // 不在同一个 4GB 段，不碰
            uint64_t head=0;memcpy(&head,reinterpret_cast<const void*>(uintptr_t(v)),8);
            char bytes[17];for(int k=0;k<8;k++)snprintf(bytes+2*k,3,"%02x",reinterpret_cast<const uint8_t*>(&head)[k]);
            xbg_logf("ptrprobe %s +%03zx -> %p head=%s",key.c_str(),off,reinterpret_cast<void*>(uintptr_t(v)),bytes);
        }
    }
    // np527：代码段在 NVNshaderData.data 里（不是控制块里）。抓 0x2000 字节，离线拿它的所有前缀哈希
    // 与 17,308 个包记录里的代码哈希求交：命中就同时得到"代码起点对不对"和"代码长度规则"。
    // np619：**按真实代码长度 dump**。原来固定读 0x2000 字节，但 ncode 取自 control+0x704，
    //   实测最大到 0x3b00（15 KB）⇒ 大着色器永远dump不全，离线被判"代码长度不足"而丢掉。
    //   键是 sha256(control[0:nctl] ‖ code[0:ncode])，代码少一个字节，包名就对不上。
    //   上限取 0x8000（32 KB，够所有实测值），并且只在 ncode 合法时才读。
    // np619: dump the real code length. It used to read a fixed 0x2000 bytes, but ncode (control+0x704) reaches
    //   0x3b00 (15 KB) on device, so large shaders were always truncated and dropped offline. The key is
    //   sha256(control[0:nctl] || code[0:ncode]) -- one byte short and the bundle name will not match.
    //   Cap at 0x8000 (32 KB, above every measured value) and only read when ncode is valid.
    if(code && (code>>32)==(reinterpret_cast<uint64_t>(control)>>32)){
        uint32_t ncode=0;memcpy(&ncode,p+0x704,4);
        size_t want=(ncode>0 && ncode<0x8000)?ncode:0x2000;
        const uint8_t *c=reinterpret_cast<const uint8_t *>(uintptr_t(code));
        xbg_logf("datahex %s %p head=%02x%02x%02x%02x ncode=%u",key.c_str(),reinterpret_cast<void*>(uintptr_t(code)),c[0],c[1],c[2],c[3],ncode);
        for(size_t off=0;off<want;off+=0x180){
            char line[1024];int n=snprintf(line,sizeof line,"datahex %s +%03zx ",key.c_str(),off);
            for(size_t i=0;i<0x180 && off+i<want;i++)n+=snprintf(line+n,sizeof line-n,"%02x",c[off+i]);
            xbg_logf("%s",line);
        }
    }
}
// np528 ★ 占位着色器：**只用于诊断**，让"缺包就死"变成"缺包先跳过去画个错的"，
//   好在一轮里看清"着色器之外还剩几道门"。画面必然是错的，因此默认关闭、必须显式开关、每次都报红。
//   用引擎自带的最小 SPIR-V（清屏 vs/ps、亮度过场 cs）当占位；gs/hs/ds 没有内置件，直接放弃（仍走 fatal）。
// np528: placeholder shader for diagnostics only ...
// np529 ★ 缺着色器时"丢弃本帧剩余命令"的开关（诊断用，DM_SHADER_SKIP=1）：
//   比占位着色器更保守——完全不建管线、不碰 inputs/bindings 形状，只是把这一帧剩下的命令扔掉，
//   让流程能继续跑到 present，好回答"着色器之外还剩几道门"。画面必然是错的。
// np529: drop the rest of this frame's commands when a shader bundle is missing (diagnostics only).
static bool g_shader_skip_abort=false;
// np670：**绘制落点记账**（诊断用）。用 present 自检时回答一个很具体的问题：这一帧的绘制到底画进了哪张图？
//   屏幕上黑、而窗口纹理（present 拿到的那个 serial）每帧都被绑成渲染目标，两种可能必须分开：
//   ① 绘制确实进了窗口纹理 ⇒ 黑在别处（layout/store/present 拷贝）；② 绘制全进了别的 serial ⇒ 窗口纹理
//   只是被"绑了一下"，真正的合成步骤缺失。只读诊断：不加锁、不进日志热路径。
// np670: where do the draws land? Companion to the present probe. Diagnostic only, no locks, no hot-path logging.
static uint64_t g_draws_since_present=0,g_draws_to_present_serial=0,g_last_draw_serial=0,g_present_serial_hint=0;static void *g_last_draw_image=nullptr;
// np598：DM_SHADER_SKIP 是否开着（只读一次）。draw() 要用它决定"没有顶点着色器"是丢这一条还是报错。
// np598: whether DM_SHADER_SKIP is on (read once). draw() uses it to decide whether a missing vertex shader
//   means "drop this draw" or a hard error.
static bool shader_skip_on(){static int v=-1;if(v<0){const char *e=getenv("DM_SHADER_SKIP");v=(e&&*e=='1')?1:0;}return v!=0;}
static Shader *placeholder_shader(const std::string &key,unsigned stage){
    const uint32_t *w=nullptr;size_t n=0;
    switch(stage){
        case 0:w=xbg_clear_vs;n=sizeof xbg_clear_vs/sizeof *xbg_clear_vs;break;
        case 4:w=xbg_clear_fs_float;n=sizeof xbg_clear_fs_float/sizeof *xbg_clear_fs_float;break;
        case 5:w=xbg_bright_lut_cs;n=sizeof xbg_bright_lut_cs/sizeof *xbg_bright_lut_cs;break;
        default:return nullptr;
    }
    Shader *p=new Shader();
    p->key=key;p->stage=stage;p->discards=true;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};ci.codeSize=n*4;ci.pCode=w;
    VK(vkCreateShaderModule(cfg.device,&ci,nullptr,&p->module));
    xbg_logf("【诊断】占位着色器 %s stage=%u：画面会是错的，只用来探路",key.c_str(),stage);
    return p;
}
Shader *load_shader(const void *control,uint64_t code,unsigned slot){
    // np532 ★ 主键 = 控制段(按自身长度) + **代码段**，与 dm/gfx/tools/shader_split.py 和
    //   nxnative 宿主 keys_hash 同一算法。之前只哈希控制段前 0x880 字节，既带进了随 ASLR 变的指针，
    //   又漏掉了代码段（大量着色器共用控制段、只有代码不同），所以包名永远对不上。
    //   两个旧键保留作后备：control-only 兼容 D2R 的包；masked 在 ctl_len 异常时兜底。
    std::string pkey;{
        // 代码段指针只在"与控制块同一个 4GB 段"时才用：键里要读 ncode 字节，野指针会当场崩。
        // nxnative 那边靠 mtl_addr() 做映射校验，我们这里用段比较作等价保险。
        const void *ck=(code && (code>>32)==(reinterpret_cast<uint64_t>(control)>>32))?reinterpret_cast<const void*>(uintptr_t(code)):nullptr;
        char k[17];if(xbg_shader_prog_key(control,ck,k)==0)pkey=k;
    }
    std::string key=control_key(control);
    std::string mkey;
    {uint8_t buf[0x880];memcpy(buf,control,0x880);memset(buf+0x848,0,0x20);char k[17];if(xbg_shader_control_key(buf,k)==0)mkey=k;}
    const std::string *cands[3]={&pkey,&key,&mkey};
    for(const std::string *c:cands){if(c->empty())continue;auto it=shaders.find(*c);if(it!=shaders.end())return it->second.get();}
    const uint64_t load_t0=timing_clock();
    FILE *f=nullptr;std::string path;
    for(const std::string *c:cands){
        if(c->empty())continue;
        std::string p=std::string(cfg.shader_directory)+"/"+*c+".xgs";
        if((f=fopen(p.c_str(),"rb"))){path=p;key=*c;break;}
    }
    if(!f){
        // 取证/落盘用主键（prog key，跨运行稳定）；另外两个键打进日志便于对照。
        xbg_logf("shaderkey prog=%s ctl=%s masked=%s",pkey.c_str(),key.c_str(),mkey.c_str());
        dump_missing_shader(pkey.empty()?key:pkey,control,code);
        const char *sk=getenv("DM_SHADER_SKIP");
        if(sk && *sk=='1'){g_shader_skip_abort=true;xbg_logf("【诊断】缺着色器 %s → 本帧丢弃剩余命令（DM_SHADER_SKIP=1，画面会是错的）",key.c_str());return nullptr;}
        const char *ph=getenv("DM_SHADER_PLACEHOLDER");
        if(ph && *ph=='1'){
            Shader *q=placeholder_shader(key,slot);
            if(q){Shader *r=shaders.emplace(key,std::unique_ptr<Shader>(q)).first->second.get();return r;}
        }
        xbg_fatalf("Missing genuine shader bundle %s",path.c_str());
    }
    fseek(f,0,SEEK_END);long length=ftell(f);rewind(f);need(length>=32 && length<=16*1024*1024,"Invalid shader bundle length");std::vector<uint8_t> bytes(length);need(fread(bytes.data(),1,bytes.size(),f)==bytes.size(),"Shader bundle read failed");fclose(f);
    xbg_shader_header h;memcpy(&h,bytes.data(),sizeof(h));need(h.magic==0x31534758 && h.version==1 && h.stage<6 && h.binding_count<=256 && h.input_count<=16 && h.c1_bytes<=65536 && h.spirv_bytes%4==0,"Invalid shader bundle header");
    size_t offset=sizeof(h),expected=offset+size_t(h.binding_count)*sizeof(xbg_shader_binding)+size_t(h.input_count)*4+h.c1_bytes+h.spirv_bytes;need(expected==bytes.size(),"Shader bundle extent mismatch");
    // 载入各段耗时（读文件、SPIR-V 改写、分析、建模块），写进下面的 loaded 行。
    // Time of each part of the load (file read, SPIR-V rewrites, analysis, module creation), for the loaded line below.
    const uint64_t load_t1=timing_clock();uint64_t load_t2=0,load_t3=0;
    auto p=std::make_unique<Shader>();p->key=key;p->stage=h.stage;p->bindings.resize(h.binding_count);memcpy(p->bindings.data(),bytes.data()+offset,h.binding_count*sizeof(xbg_shader_binding));offset+=h.binding_count*sizeof(xbg_shader_binding);p->inputs.resize(h.input_count);memcpy(p->inputs.data(),bytes.data()+offset,h.input_count*4);offset+=h.input_count*4;p->c1.assign(bytes.begin()+offset,bytes.begin()+offset+h.c1_bytes);offset+=h.c1_bytes;
    std::vector<uint32_t> words(h.spirv_bytes/4);memcpy(words.data(),bytes.data()+offset,h.spirv_bytes);spv_override(key,words);validate_capabilities(words,h.stage);
    // gfx-shader-reuse：预热线程已改写过这个着色器时直接取用；2 时照样改写，与取来的逐字比对。
    // gfx-shader-reuse: take the warm-up threads' rewrite of this shader when there is one; with 2, rewrite as well and compare word by word.
    std::vector<uint32_t> reused;const bool have_reused=shader_reuse && !ab_off("shreuse") && warm::take_rewritten(key,reused);
    // 预热线程取来的已经内联过（load_module 同一步）；只有自己改写时才内联，免得在图形线程上重做一遍（大着色器 40–90 ms）。
    // What the warm-up hands over is already inlined (the same step in load_module); inline only when rewriting here, so the
    // graphics thread does not redo it (40-90 ms on the large shaders).
    bool inlined=have_reused && shader_reuse!=2 && inline_helpers && helper_has_subgroup(words);
    if(!have_reused || shader_reuse==2){inlined=inline_rewrite(words,key);quad_rewrite(words,h.stage,key.c_str(),"runtime");warp_rewrite(words,h.stage,key.c_str(),"runtime");tidy_rewrite(words);
        if(have_reused && reused!=words){++shader_reuse_mismatches;xbg_logf("NVN shader reuse MISMATCH %s stage=%u words=%zu/%zu",key.c_str(),h.stage,reused.size(),words.size());}}
    else words.swap(reused);
    if(have_reused)++shaders_reused;
    load_t2=timing_clock();
    {auto extent=spirv_uniform_extent(words);uint32_t trimmed=0;for(auto &b:p->bindings)if(b.type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && b.source==XBG_SOURCE_UNIFORM){auto e=extent.find(b.binding);uint32_t declared=b.bytes?b.bytes:65536;if(e!=extent.end() && e->second<declared){b.bytes=e->second;++trimmed;}}uniform_trimmed+=trimmed;}
    {bool kill=false;for(size_t i=5;i<words.size();){uint32_t n=words[i]>>16,op=words[i]&0xffff;if(op==252||op==4416||op==5380){kill=true;break;}if(!n)break;i+=n;}p->discards=kill;}
    p->raster=xbg_spirv_raster_facts(words.data(),words.size(),h.stage);
    {bool unknown=false;p->storage_writes=spirv_storage_writes(words,unknown);if(unknown){for(auto &b:p->bindings)if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)p->storage_writes.insert(b.binding);}}
    if(h.stage==0){p->input_classes.resize(p->inputs.size());need(xbg_vertex_input_classes(words.data(),words.size(),p->inputs.data(),p->inputs.size(),p->input_classes.data())==0,"Unsupported vertex input scalar interface");}
    load_t3=timing_clock();
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};ci.codeSize=words.size()*4;ci.pCode=words.data();VK(vkCreateShaderModule(cfg.device,&ci,nullptr,&p->module));warm::register_module(p->module,key);Shader *out=p.get();shaders.emplace(key,std::move(p));++stats.shader_loads;++hud::shaders;const uint64_t load_t4=timing_clock();hud::shader_ns+=load_t4-load_t0;
    xbg_logf("NVN shader %s stage=%u loaded words=%zu ms read=%.2f rewrite=%.2f analyse=%.2f module=%.2f reused=%d inlined=%d",key.c_str(),h.stage,words.size(),(load_t1-load_t0)/1e6,(load_t2-load_t1)/1e6,(load_t3-load_t2)/1e6,(load_t4-load_t3)/1e6,int(have_reused),int(inlined));return out;
}
// nvnProgramSetShaders assigns a new serial, and NVN shader control/code memory
// must stay unchanged while a program uses it: cache resolved shaders by serial
// instead of re-hashing every control segment on each bind.
std::map<uint64_t,std::array<Shader*,6>> program_cache;
const std::array<Shader*,6> &program_shaders(const prog_t *prog){
    if(prog->serial){auto it=program_cache.find(prog->serial);if(it!=program_cache.end())return it->second;}
    // np536：把每个阶段各自的键打出来，先排除"控制段/代码段配错对"这一种错法。
    if(getenv("DM_SHADER_STAGEKEYS")){
        for(unsigned i=0;i<prog->nstage && i<6;i++){
            const uint8_t *c=reinterpret_cast<const uint8_t*>(prog->sd[i].control);
            if(!c)continue;
            uint32_t own=0,n704=0;memcpy(&own,c+0x20,4);memcpy(&n704,c+0x704,4);
            char k[17];xbg_shader_prog_key(c,reinterpret_cast<const void*>(uintptr_t(prog->sd[i].data)),k);
            xbg_logf("stagekey %u ctl=%p data=%p own=%#x n704=%#x key=%s",i,(const void*)c,(void*)(uintptr_t)prog->sd[i].data,own,n704,k);
        }
    }
    std::array<Shader*,6> out{};for(unsigned i=0;i<prog->nstage && i<6;i++)out[i]=load_shader(prog->sd[i].control,(uint64_t)prog->sd[i].data,i);
    if(!prog->serial){static std::array<Shader*,6> scratch;scratch=out;return scratch;}
    return program_cache.emplace(prog->serial,out).first->second;
}
} // namespace

// np701：跨编译单元契约符号（xbg_deathsite.h；vulkan_present.cpp 链接）——必须在全局作用域。
// np701: cross-TU contract symbols (xbg_deathsite.h, linked by vulkan_present.cpp) -- global scope required.
extern "C" void xbg_death_site_enter(unsigned site,unsigned long long ctx){deathguard::x_enter(site,ctx);}
extern "C" void xbg_death_site_leave(void){deathguard::x_leave();}
extern "C" void xbg_logf(const char *fmt,...){CPU_SCOPE(Log);char text[1024];va_list a;va_start(a,fmt);vsnprintf(text,sizeof(text),fmt,a);va_end(a);if(cfg.log)cfg.log(cfg.userdata,text);else fprintf(stderr,"[xbg] %s\n",text);}
extern "C" __attribute__((noreturn)) void xbg_fatalf(const char *fmt,...){va_list a;va_start(a,fmt);vsnprintf(error_text,sizeof(error_text),fmt,a);va_end(a);failed=true;xbg_logf("FATAL: %s",error_text);if(cfg.fatal)cfg.fatal(cfg.userdata,error_text);abort();}
extern "C" int xbg_backend_failed(){return failed;}
extern "C" const char *xbg_game_last_error(){return error_text;}
extern "C" int xbg_device_integer(int p,int *out){
    need(initialized && out,"NVN device query before Vulkan initialization");const auto &l=properties.limits;
    switch(p){
    case NVN_DEVICE_INFO_MAX_UNIFORM_BUFFER_SIZE:*out=std::min(65536u,l.maxUniformBufferRange);break;
    case NVN_DEVICE_INFO_COLOR_BUFFER_BINDINGS:*out=std::min(8u,l.maxColorAttachments);break;
    case NVN_DEVICE_INFO_VERTEX_BUFFER_BINDINGS:*out=std::min(16u,l.maxVertexInputBindings);break;
    case NVN_DEVICE_INFO_VERTEX_ATTRIBUTES:*out=std::min(16u,l.maxVertexInputAttributes);break;
    case NVN_DEVICE_INFO_MAX_TEXTURE_ANISOTROPY:*out=cfg.enabled_core.samplerAnisotropy?int(l.maxSamplerAnisotropy):1;break;
    case NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_X:case NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_Y:case NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_Z:*out=l.maxComputeWorkGroupSize[p-NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_X];break;
    case NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_THREADS:*out=l.maxComputeWorkGroupInvocations;break;
    case NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_X:case NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_Y:case NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_Z:*out=l.maxComputeWorkGroupCount[p-NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_X];break;
    case NVN_DEVICE_INFO_MAX_VIEWPORTS:*out=1;break;
    case NVN_DEVICE_INFO_MAX_TEXTURE_LAYERS:*out=l.maxImageArrayLayers;break;
    case NVN_DEVICE_INFO_MAX_TEXTURE_SIZE:case NVN_DEVICE_INFO_MAX_RECTANGLE_TEXTURE_SIZE:*out=l.maxImageDimension2D;break;
    case NVN_DEVICE_INFO_MAX_3D_TEXTURE_SIZE:*out=l.maxImageDimension3D;break;
    case NVN_DEVICE_INFO_MAX_CUBE_MAP_TEXTURE_SIZE:*out=l.maxImageDimensionCube;break;
    case NVN_DEVICE_INFO_MAX_BUFFER_TEXTURE_SIZE:*out=l.maxTexelBufferElements;break;
    case NVN_DEVICE_INFO_MAX_PATCH_SIZE:*out=cfg.enabled_core.tessellationShader?l.maxTessellationPatchSize:0;break;
    case NVN_DEVICE_INFO_SUBPIXEL_BITS:*out=l.subPixelPrecisionBits;break;
    case NVN_DEVICE_INFO_SUPPORTS_MIN_MAX_FILTERING:case NVN_DEVICE_INFO_SUPPORTS_ASTC_FORMATS:
    case NVN_DEVICE_INFO_SUPPORTS_CONSERVATIVE_RASTER:case NVN_DEVICE_INFO_SUPPORTS_ZERO_FROM_UNMAPPED_VIRTUAL_POOL_PAGES:
    case NVN_DEVICE_INFO_SUPPORTS_PASSTHROUGH_GEOMETRY_SHADERS:case NVN_DEVICE_INFO_SUPPORTS_VIEWPORT_SWIZZLE:
    case NVN_DEVICE_INFO_SUPPORTS_SPARSE_TILED_PACKAGED_TEXTURES:case NVN_DEVICE_INFO_SUPPORTS_ADVANCED_BLEND_MODES:
    case NVN_DEVICE_INFO_SUPPORTS_DRAW_TEXTURE:case NVN_DEVICE_INFO_SUPPORTS_TARGET_INDEPENDENT_RASTERIZATION:
    case NVN_DEVICE_INFO_SUPPORTS_FRAGMENT_COVERAGE_TO_COLOR:case NVN_DEVICE_INFO_SUPPORTS_POST_DEPTH_COVERAGE:
    case NVN_DEVICE_INFO_SUPPORTS_SAMPLE_LOCATIONS:case NVN_DEVICE_INFO_SUPPORTS_FRAGMENT_SHADER_INTERLOCK:*out=0;break;
    case NVN_DEVICE_INFO_SUPPORTS_STENCIL8_FORMAT:{VkFormatProperties f;vkGetPhysicalDeviceFormatProperties(cfg.physical,VK_FORMAT_S8_UINT,&f);*out=(f.optimalTilingFeatures&VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)!=0;break;}
    default:return 0; // NVN ABI sizes/alignments are host-emulated, not Vulkan addresses.
    }return 1;
}

namespace {
struct PipelineEntry {VkPipeline pipeline;uint64_t last_used;};
std::map<std::string,PipelineEntry> pipeline_objects;
// First-level lookup (D2R camp: 2198 per frame, ~7% of the render thread with the launch-stable key): a process-local key
// that names the layout and render-pass definitions by their cached, immutable entries instead of copying their words in.
// Only a miss builds the full key (warm-up list, pipeline_objects). Cleared with the pipeline, layout and render-pass
// caches; an evicted pipeline is dropped from it.
struct FastKeyHash{size_t operator()(const std::vector<uint32_t> &v)const{uint64_t h=1469598103934665603ull;for(uint32_t x:v){h^=x;h*=1099511628211ull;}return size_t(h^(h>>29));}};
std::unordered_map<std::vector<uint32_t>,VkPipeline,FastKeyHash> pipeline_fast;bool pipeline_fast_enabled=true;uint64_t pipeline_fast_hits=0;
bool pipe_memo_valid=false;   // the previous draw's pipeline (pipe_memo_key below); cleared wherever pipeline_fast drops entries
uint64_t pipeline_tick=0;
constexpr size_t pipeline_object_limit=4096; // field scenes exceed 256 live pipelines; eviction+rebuild caused 100ms+ hitches
struct PipelineLease {
    VkPipeline pipeline{};bool transient=false;
    ~PipelineLease(){if(transient && pipeline){VkPipeline p=pipeline;retire_handle([p]{vkDestroyPipeline(cfg.device,p,nullptr);});}}
    PipelineLease()=default;
    PipelineLease(const PipelineLease&)=delete;
    PipelineLease& operator=(const PipelineLease&)=delete;
};
void report_pipeline_cache(){
    xbg_logf("NVN CPU_TIMING pipeline_cache hits=%llu misses=%llu evictions=%llu bypass=%llu entries=%zu limit=%zu",(unsigned long long)(pipeline_counts.hits-pipeline_before.hits),(unsigned long long)(pipeline_counts.misses-pipeline_before.misses),(unsigned long long)(pipeline_counts.evictions-pipeline_before.evictions),(unsigned long long)(pipeline_counts.bypass-pipeline_before.bypass),pipeline_objects.size(),pipeline_object_limit);
}
// Pipeline warm-up (the "precompiled shader cache" of console emulators). The Maleoon driver does not grow the
// VkPipelineCache, so every launch recompiled each graphics pipeline on first use, inside the frame (~100+ ms each).
// Every newly created graphics pipeline is appended to files/vk-pipeline-list.bin (shader bundle keys + the exact
// serialized create-info key, with shader modules identified by a stable hash of their bundle key). At start-up,
// background threads rebuild those pipelines from the list; a draw that needs one takes it ready-made.
// files/gfx-no-warm disables building (recording continues).
namespace pipeguard{std::string compute_key(const std::string &key);std::string graphics_key(const std::vector<std::string> &keys);bool skipped(const std::string &key);void begin(const std::string &key);void end(const std::string &key,bool ok);void record(const std::string &key,const char *why);}
namespace warm {
std::mutex mu;std::unordered_map<VkShaderModule,uint64_t> ids;
uint64_t fnv(const std::string &s){uint64_t h=1469598103934665603ull;for(unsigned char c:s){h^=c;h*=1099511628211ull;}return h|(1ull<<63);}
uint64_t module_id(VkShaderModule m){std::lock_guard<std::mutex> l(mu);auto it=ids.find(m);return it==ids.end()?uint64_t(m):it->second;}
std::unordered_map<uint64_t,std::string> id_keys;
void register_module(VkShaderModule m,const std::string &key){std::lock_guard<std::mutex> l(mu);uint64_t id=fnv(key);ids[m]=id;id_keys[id]=key;}
struct Rec{std::vector<std::string> keys;std::vector<uint32_t> words;};
std::vector<Rec> recs;std::unordered_set<std::string> known;std::vector<uint8_t> pending;
std::unordered_map<std::string,VkPipeline> ready;std::vector<std::thread> threads;std::atomic<bool> stop{false};std::atomic<size_t> next{0};
std::atomic<uint64_t> built{0},failed{0},build_ns{0},compiled{0};uint64_t hits=0,creates=0,create_ns=0,create_max_ns=0;bool enabled=true;
std::unordered_map<std::string,VkShaderModule> modules;std::unordered_map<std::string,VkPipelineLayout> layouts;std::unordered_map<std::string,VkRenderPass> passes;
std::vector<VkDescriptorSetLayout> set_layouts;
std::string list_path(){std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');return (cut==std::string::npos?dir:dir.substr(0,cut))+"/vk-pipeline-list.bin";}
void record(const VkGraphicsPipelineCreateInfo &ci,const XbgPipelineKey &key){
    std::string bytes(reinterpret_cast<const char*>(key.words),size_t(key.count)*4);
    std::vector<std::string> keys;{std::lock_guard<std::mutex> l(mu);if(!known.insert(bytes).second)return;
        for(uint32_t i=0;i<ci.stageCount;i++){auto it=ids.find(ci.pStages[i].module);if(it==ids.end())return;keys.push_back(id_keys[it->second]);}}
    auto put=[](uint32_t v){for(int i=0;i<4;i++)pending.push_back(uint8_t(v>>(8*i)));};
    put(0x57504c58);put(uint32_t(keys.size()));for(auto &k:keys){put(uint32_t(k.size()));pending.insert(pending.end(),k.begin(),k.end());}
    put(key.count);pending.insert(pending.end(),bytes.begin(),bytes.end());
}
void flush(){if(pending.empty())return;FILE *f=fopen(list_path().c_str(),"ab");if(!f)return;bool ok=fwrite(pending.data(),1,pending.size(),f)==pending.size();ok=fclose(f)==0&&ok;if(ok)pending.clear();}
// gfx-shader-reuse：预热改写好的 SPIR-V，按着色器键留到 load_shader 取走（取走即删）；没用到的随 shutdown 释放。
// gfx-shader-reuse: SPIR-V the warm-up rewrote, kept by shader key until load_shader takes it (taking removes it); unused entries
// are freed at shutdown.
bool keep_rewritten=false;std::unordered_map<std::string,std::vector<uint32_t>> rewritten;
bool take_rewritten(const std::string &key,std::vector<uint32_t> &out){std::lock_guard<std::mutex> l(mu);auto it=rewritten.find(key);if(it==rewritten.end())return false;out.swap(it->second);rewritten.erase(it);return true;}
VkShaderModule load_module(const std::string &key){
    {std::lock_guard<std::mutex> l(mu);auto it=modules.find(key);if(it!=modules.end())return it->second;}
    FILE *f=fopen((std::string(cfg.shader_directory)+"/"+key+".xgs").c_str(),"rb");if(!f)return VK_NULL_HANDLE;
    fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);if(n<32 || n>16*1024*1024){fclose(f);return VK_NULL_HANDLE;}std::vector<uint8_t> b(static_cast<size_t>(n));bool ok=fread(b.data(),1,b.size(),f)==b.size();fclose(f);if(!ok)return VK_NULL_HANDLE;
    xbg_shader_header h;memcpy(&h,b.data(),sizeof(h));if(h.magic!=0x31534758 || h.version!=1 || h.spirv_bytes%4)return VK_NULL_HANDLE;
    size_t off=sizeof(h)+size_t(h.binding_count)*sizeof(xbg_shader_binding)+size_t(h.input_count)*4+h.c1_bytes;if(off+h.spirv_bytes!=b.size())return VK_NULL_HANDLE;
    std::vector<uint32_t> w(h.spirv_bytes/4);memcpy(w.data(),b.data()+off,h.spirv_bytes);spv_override(key,w);inline_rewrite(w,key);quad_rewrite(w,h.stage,key.c_str(),"warm");warp_rewrite(w,h.stage,key.c_str(),"warm");tidy_rewrite(w);
    if(keep_rewritten){std::lock_guard<std::mutex> l(mu);rewritten.emplace(key,w);}
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};ci.codeSize=w.size()*4;ci.pCode=w.data();VkShaderModule m;if(vkCreateShaderModule(cfg.device,&ci,nullptr,&m)!=VK_SUCCESS)return VK_NULL_HANDLE;
    std::lock_guard<std::mutex> l(mu);auto ins=modules.emplace(key,m);if(!ins.second){vkDestroyShaderModule(cfg.device,m,nullptr);return ins.first->second;}return m;
}
struct Reader{const uint32_t *w;size_t n,i=0;bool bad=false;uint32_t u(){if(i>=n){bad=true;return 0;}return w[i++];}uint64_t q(){uint64_t lo=u();return lo|(uint64_t(u())<<32);}float f(){uint32_t b=u();float x;memcpy(&x,&b,4);return x;}};
VkPipelineLayout make_layout(const uint32_t *w,uint32_t n){
    std::string k(reinterpret_cast<const char*>(w),size_t(n)*4);{std::lock_guard<std::mutex> l(mu);auto it=layouts.find(k);if(it!=layouts.end())return it->second;}
    Reader r{w,n};VkDescriptorSetLayoutCreateInfo s{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};s.flags=r.u();uint32_t nb=r.u();if(nb>4096)return VK_NULL_HANDLE;std::vector<VkDescriptorSetLayoutBinding> bs(nb);
    for(auto &b:bs){b.binding=r.u();b.descriptorType=VkDescriptorType(r.u());b.descriptorCount=r.u();b.stageFlags=r.u();}s.bindingCount=nb;s.pBindings=bs.data();
    VkPipelineLayoutCreateInfo p{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};p.flags=r.u();uint32_t sets=r.u(),np=r.u();if(sets!=1 || np>16 || r.bad)return VK_NULL_HANDLE;std::vector<VkPushConstantRange> pr(np);for(auto &x:pr){x.stageFlags=r.u();x.offset=r.u();x.size=r.u();}
    if(r.bad || r.i!=n)return VK_NULL_HANDLE;VkDescriptorSetLayout sl;if(vkCreateDescriptorSetLayout(cfg.device,&s,nullptr,&sl)!=VK_SUCCESS)return VK_NULL_HANDLE;
    p.setLayoutCount=1;p.pSetLayouts=&sl;p.pushConstantRangeCount=np;p.pPushConstantRanges=pr.data();VkPipelineLayout pl;if(vkCreatePipelineLayout(cfg.device,&p,nullptr,&pl)!=VK_SUCCESS){vkDestroyDescriptorSetLayout(cfg.device,sl,nullptr);return VK_NULL_HANDLE;}
    std::lock_guard<std::mutex> l(mu);set_layouts.push_back(sl);layouts.emplace(k,pl);return pl;
}
VkRenderPass make_pass(const uint32_t *w,uint32_t n){
    std::string k(reinterpret_cast<const char*>(w),size_t(n)*4);{std::lock_guard<std::mutex> l(mu);auto it=passes.find(k);if(it!=passes.end())return it->second;}
    Reader r{w,n};VkRenderPassCreateInfo p{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};p.flags=r.u();uint32_t na=r.u();if(na>9)return VK_NULL_HANDLE;std::vector<VkAttachmentDescription> a(na);
    for(auto &x:a){x.flags=r.u();x.format=VkFormat(r.u());x.samples=VkSampleCountFlagBits(r.u());x.loadOp=VkAttachmentLoadOp(r.u());x.storeOp=VkAttachmentStoreOp(r.u());x.stencilLoadOp=VkAttachmentLoadOp(r.u());x.stencilStoreOp=VkAttachmentStoreOp(r.u());x.initialLayout=VkImageLayout(r.u());x.finalLayout=VkImageLayout(r.u());}
    VkSubpassDescription s{};s.flags=r.u();s.pipelineBindPoint=VkPipelineBindPoint(r.u());uint32_t nc=r.u();if(nc>8)return VK_NULL_HANDLE;std::vector<VkAttachmentReference> c(nc);for(auto &x:c){x.attachment=r.u();x.layout=VkImageLayout(r.u());}
    VkAttachmentReference d{};bool hd=r.u()!=0;if(hd){d.attachment=r.u();d.layout=VkImageLayout(r.u());}
    uint32_t nd=r.u();if(nd>8)return VK_NULL_HANDLE;std::vector<VkSubpassDependency> dep(nd);for(auto &x:dep){x.srcSubpass=r.u();x.dstSubpass=r.u();x.srcStageMask=r.u();x.dstStageMask=r.u();x.srcAccessMask=r.u();x.dstAccessMask=r.u();x.dependencyFlags=r.u();}
    if(r.bad || r.i!=n)return VK_NULL_HANDLE;
    s.colorAttachmentCount=nc;s.pColorAttachments=c.data();s.pDepthStencilAttachment=hd?&d:nullptr;p.attachmentCount=na;p.pAttachments=a.data();p.subpassCount=1;p.pSubpasses=&s;p.dependencyCount=nd;p.pDependencies=dep.data();
    VkRenderPass rp;if(vkCreateRenderPass(cfg.device,&p,nullptr,&rp)!=VK_SUCCESS)return VK_NULL_HANDLE;
    std::lock_guard<std::mutex> l(mu);auto ins=passes.emplace(k,rp);if(!ins.second){vkDestroyRenderPass(cfg.device,rp,nullptr);return ins.first->second;}return rp;
}
bool build(const Rec &rec){
    Reader r{rec.words.data(),rec.words.size()};if(r.u()!=1)return false;
    uint32_t ln=r.u();if(r.i+ln>r.n)return false;VkPipelineLayout layout=make_layout(r.w+r.i,ln);r.i+=ln;
    uint32_t pn=r.u();if(r.i+pn>r.n)return false;VkRenderPass pass=make_pass(r.w+r.i,pn);r.i+=pn;if(!layout || !pass)return false;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};ci.flags=r.u();ci.subpass=r.u();uint32_t ns=r.u();if(ns>5 || ns!=rec.keys.size())return false;
    VkPipelineShaderStageCreateInfo st[5]{};for(uint32_t i=0;i<ns;i++){st[i].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;st[i].flags=r.u();st[i].stage=VkShaderStageFlagBits(r.u());uint64_t id=r.q();if(id!=fnv(rec.keys[i]))return false;st[i].module=load_module(rec.keys[i]);if(!st[i].module)return false;st[i].pName="main";}
    VkPipelineVertexInputStateCreateInfo v{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};v.flags=r.u();uint32_t nvb=r.u();if(nvb>32)return false;std::vector<VkVertexInputBindingDescription> vb(nvb);for(auto &x:vb){x.binding=r.u();x.stride=r.u();x.inputRate=VkVertexInputRate(r.u());}
    uint32_t nva=r.u();if(nva>32)return false;std::vector<VkVertexInputAttributeDescription> va(nva);for(auto &x:va){x.location=r.u();x.binding=r.u();x.format=VkFormat(r.u());x.offset=r.u();}
    v.vertexBindingDescriptionCount=nvb;v.pVertexBindingDescriptions=vb.data();v.vertexAttributeDescriptionCount=nva;v.pVertexAttributeDescriptions=va.data();
    VkPipelineInputAssemblyStateCreateInfo a{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};a.flags=r.u();a.topology=VkPrimitiveTopology(r.u());a.primitiveRestartEnable=r.u();
    VkPipelineTessellationStateCreateInfo t{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};bool ht=r.u()!=0;if(ht){t.flags=r.u();t.patchControlPoints=r.u();}
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vp.flags=r.u();vp.viewportCount=r.u();vp.scissorCount=r.u();bool hv=r.u()!=0,hs=r.u()!=0;if(vp.viewportCount>16 || vp.scissorCount>16)return false;
    std::vector<VkViewport> vps(hv?vp.viewportCount:0);for(auto &x:vps){x.x=r.f();x.y=r.f();x.width=r.f();x.height=r.f();x.minDepth=r.f();x.maxDepth=r.f();}
    std::vector<VkRect2D> scs(hs?vp.scissorCount:0);for(auto &x:scs){x.offset.x=int32_t(r.u());x.offset.y=int32_t(r.u());x.extent.width=r.u();x.extent.height=r.u();}
    vp.pViewports=hv?vps.data():nullptr;vp.pScissors=hs?scs.data():nullptr;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.flags=r.u();rs.depthClampEnable=r.u();rs.rasterizerDiscardEnable=r.u();rs.polygonMode=VkPolygonMode(r.u());rs.cullMode=r.u();rs.frontFace=VkFrontFace(r.u());rs.depthBiasEnable=r.u();rs.depthBiasConstantFactor=r.f();rs.depthBiasClamp=r.f();rs.depthBiasSlopeFactor=r.f();rs.lineWidth=r.f();
    VkPipelineMultisampleStateCreateInfo m{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};m.flags=r.u();m.rasterizationSamples=VkSampleCountFlagBits(r.u());m.sampleShadingEnable=r.u();m.minSampleShading=r.f();m.alphaToCoverageEnable=r.u();m.alphaToOneEnable=r.u();bool hm=r.u()!=0;
    uint32_t mask[2]{};if(hm){if(m.rasterizationSamples>VK_SAMPLE_COUNT_64_BIT)return false;for(uint32_t i=0;i<(uint32_t(m.rasterizationSamples)+31)/32;i++)mask[i]=r.u();m.pSampleMask=mask;}
    auto stencil=[&](VkStencilOpState &s){s.failOp=VkStencilOp(r.u());s.passOp=VkStencilOp(r.u());s.depthFailOp=VkStencilOp(r.u());s.compareOp=VkCompareOp(r.u());s.compareMask=r.u();s.writeMask=r.u();s.reference=r.u();};
    VkPipelineDepthStencilStateCreateInfo d{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};d.flags=r.u();d.depthTestEnable=r.u();d.depthWriteEnable=r.u();d.depthCompareOp=VkCompareOp(r.u());d.depthBoundsTestEnable=r.u();d.stencilTestEnable=r.u();stencil(d.front);stencil(d.back);d.minDepthBounds=r.f();d.maxDepthBounds=r.f();
    VkPipelineColorBlendStateCreateInfo b{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};b.flags=r.u();b.logicOpEnable=r.u();b.logicOp=VkLogicOp(r.u());uint32_t nb=r.u();if(nb>8)return false;std::vector<VkPipelineColorBlendAttachmentState> ba(nb);
    for(auto &x:ba){x.blendEnable=r.u();x.srcColorBlendFactor=VkBlendFactor(r.u());x.dstColorBlendFactor=VkBlendFactor(r.u());x.colorBlendOp=VkBlendOp(r.u());x.srcAlphaBlendFactor=VkBlendFactor(r.u());x.dstAlphaBlendFactor=VkBlendFactor(r.u());x.alphaBlendOp=VkBlendOp(r.u());x.colorWriteMask=r.u();}
    b.attachmentCount=nb;b.pAttachments=ba.data();for(float &x:b.blendConstants)x=r.f();
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dy.flags=r.u();uint32_t nd=r.u();if(nd>64)return false;std::vector<VkDynamicState> ds(nd);for(auto &x:ds)x=VkDynamicState(r.u());dy.dynamicStateCount=nd;dy.pDynamicStates=ds.data();
    if(r.bad || r.i!=r.n)return false;
    // np695: warm-rebuilt pipelines must carry the same depth-clip pNext as the on-demand path
    //   (np693/np694) -- the key skips these nodes, so a draw served from the warm list would
    //   otherwise get a pipeline without the GL clip space and be clipped away again.
    // np695: warm-rebuilt pipelines must attach the same depth-clip pNext as the on-demand path
    //   (the key skips these nodes), otherwise draws taken from the warm list lose the GL clip space
    //   again and get clipped in full.
    VkPipelineViewportDepthClipControlCreateInfoEXT clipctl{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT};
    VkPipelineRasterizationDepthClipStateCreateInfoEXT clipst{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT};
    if(clip_neg_one()){clipctl.negativeOneToOne=VK_TRUE;clipctl.pNext=vp.pNext;vp.pNext=&clipctl;}
    if(depth_clamp_on()){clipst.depthClipEnable=VK_FALSE;clipst.pNext=rs.pNext;rs.pNext=&clipst;}
    ci.stageCount=ns;ci.pStages=st;ci.pVertexInputState=&v;ci.pInputAssemblyState=&a;ci.pTessellationState=ht?&t:nullptr;ci.pViewportState=&vp;ci.pRasterizationState=&rs;ci.pMultisampleState=&m;ci.pDepthStencilState=&d;ci.pColorBlendState=&b;ci.pDynamicState=&dy;ci.layout=layout;ci.renderPass=pass;
    const std::string guard=pipeguard::graphics_key(rec.keys);if(pipeguard::skipped(guard))return false;
    uint64_t c0=timing_clock();VkPipeline pl;pipeguard::begin(guard);const VkResult made=[&]{deathguard::VkSite site(deathguard::PIPE_GRAPHICS,deathguard::ctx_of(guard.c_str()));return vkCreateGraphicsPipelines(cfg.device,pipeline_cache,1,&ci,nullptr,&pl);}();pipeguard::end(guard,made==VK_SUCCESS);
    deathguard::pipeline_built(guard.c_str(),(timing_clock()-c0)/1000000ull);
    if(made!=VK_SUCCESS){char why[64];snprintf(why,sizeof why,"failed to build (result=%d)",int(made));pipeguard::record(guard,why);return false;}
    if(timing_clock()-c0>=compile_ns)++compiled;
    std::string bytes(reinterpret_cast<const char*>(rec.words.data()),rec.words.size()*4);
    std::lock_guard<std::mutex> l(mu);if(!ready.emplace(bytes,pl).second)vkDestroyPipeline(cfg.device,pl,nullptr);return true;
}
void worker(){
    setpriority(PRIO_PROCESS,0,5); // below the game and graphics threads
    for(;;){if(stop.load())return;size_t i=next.fetch_add(1);if(i>=recs.size())return;uint64_t t0=timing_clock();if(build(recs[i]))++built;else ++failed;build_ns+=timing_clock()-t0;}
}
// Runtime builds for draws that may be skipped (gfx-async-pipelines): a blended draw that writes no depth, stencil or
// storage and counts no samples the game reads is skipped while one background thread builds its pipeline from the
// record the warm-up list keeps; a later frame takes it ready-made (take). The effect then appears a few frames late
// the first time, instead of the frame waiting 22-50 ms for the driver (OnePlus 13, new spell effects). Off unless the
// host asks (DM_GFX_ASYNC_PIPELINES; marker gfx-async-pipelines 1 / 0).
bool async_on=false;std::deque<Rec> async_q;std::unordered_set<std::string> async_pending,async_failed;std::condition_variable async_cv;std::thread async_thread;
std::atomic<uint64_t> async_built{0},async_skips{0};
void async_worker(){
    setpriority(PRIO_PROCESS,0,5);
    for(;;){Rec r;{std::unique_lock<std::mutex> l(mu);async_cv.wait(l,[]{return stop.load() || !async_q.empty();});if(stop.load())return;r=std::move(async_q.front());async_q.pop_front();}
        bool ok=build(r);if(ok)++async_built;else ++failed;
        std::string bytes(reinterpret_cast<const char*>(r.words.data()),r.words.size()*4);std::lock_guard<std::mutex> l(mu);async_pending.erase(bytes);if(!ok)async_failed.insert(bytes);}
}
// Queue the build of this draw's pipeline (once); false when the caller should build it now: a module without a bundle key,
// a record that failed to build in the background before, or the warm-up threads stopped.
bool async_request(const VkGraphicsPipelineCreateInfo &ci,const XbgPipelineKey &key,const std::string &bytes){
    {std::lock_guard<std::mutex> l(mu);if(async_pending.count(bytes))return true;if(stop.load() || async_failed.count(bytes))return false;Rec r;
        for(uint32_t i=0;i<ci.stageCount;i++){auto it=ids.find(ci.pStages[i].module);if(it==ids.end())return false;r.keys.push_back(id_keys[it->second]);}
        r.words.assign(key.words,key.words+key.count);async_pending.insert(bytes);async_q.push_back(std::move(r));}
    if(!async_thread.joinable())async_thread=std::thread(async_worker);
    async_cv.notify_one();record(ci,key);return true;
}
// 首次启动还没有列表。这时随着色器发的种子（shaders/vk-pipeline-list.seed，用 dm/android/tools/merge_warm_lists.py 合并各设备的
// 列表而成）就成为列表，预热便在第一场战斗之前、而不是在战斗中建好那些设备见过的管线；设备建不了的记录在预热时失败（warm_failed）。
// A first launch has no list yet. A seed shipped with the shaders (shaders/vk-pipeline-list.seed, made from devices' lists
// with dm/android/tools/merge_warm_lists.py) then becomes the list, so the warm-up builds the pipelines those devices saw
// before the first fight instead of in it; records a device cannot build fail in the warm-up (warm_failed).
void seed(){
    const std::string list=list_path();if(FILE *f=fopen(list.c_str(),"rb")){fclose(f);return;}
    const std::string from=std::string(cfg.shader_directory)+"/vk-pipeline-list.seed";FILE *in=fopen(from.c_str(),"rb");if(!in)return;
    const std::string tmp=list+".tmp";FILE *out=fopen(tmp.c_str(),"wb");bool ok=out!=nullptr;size_t total=0;
    for(char buf[65536];ok;){size_t n=fread(buf,1,sizeof buf,in);if(!n)break;ok=fwrite(buf,1,n,out)==n;total+=n;}
    fclose(in);if(out)ok=fclose(out)==0 && ok;if(ok)ok=rename(tmp.c_str(),list.c_str())==0;if(!ok)remove(tmp.c_str());
    xbg_logf("NVN warm list seeded from %s bytes=%zu ok=%d",from.c_str(),total,int(ok));
}
void start(){
    seed();
    FILE *f=fopen(list_path().c_str(),"rb");if(f){fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);std::vector<uint8_t> b(n>0 && n<(64l<<20)?size_t(n):0);bool ok=b.empty() || fread(b.data(),1,b.size(),f)==b.size();fclose(f);
        size_t i=0;auto get=[&](uint32_t &v){if(i+4>b.size())return false;memcpy(&v,b.data()+i,4);i+=4;return true;};
        while(ok && i<b.size()){uint32_t magic,ns,count;if(!get(magic)||magic!=0x57504c58||!get(ns)||ns>5)break;Rec r;bool good=true;
            for(uint32_t s=0;s<ns && good;s++){uint32_t len;if(!get(len)||len>256||i+len>b.size()){good=false;break;}r.keys.emplace_back(reinterpret_cast<const char*>(b.data()+i),len);i+=len;}
            if(!good||!get(count)||count>XbgPipelineKey::Capacity||i+size_t(count)*4>b.size())break;r.words.resize(count);memcpy(r.words.data(),b.data()+i,size_t(count)*4);i+=size_t(count)*4;
            if(known.insert(std::string(reinterpret_cast<const char*>(r.words.data()),r.words.size()*4)).second)recs.push_back(std::move(r));}}
    xbg_logf("NVN warm list=%zu enabled=%d",recs.size(),int(enabled));
    if(!enabled || recs.empty())return;stop=false;next=0;unsigned n=std::min<unsigned>(3,std::max(1u,std::thread::hardware_concurrency()/3));
    for(unsigned i=0;i<n;i++)threads.emplace_back(worker);
}
void shutdown(){{std::lock_guard<std::mutex> l(mu);stop=true;}async_cv.notify_all();for(auto &t:threads)if(t.joinable())t.join();threads.clear();if(async_thread.joinable())async_thread.join();std::lock_guard<std::mutex> l(mu);async_q.clear();async_pending.clear();async_failed.clear();
    for(auto &p:ready)vkDestroyPipeline(cfg.device,p.second,nullptr);ready.clear();for(auto &p:layouts)vkDestroyPipelineLayout(cfg.device,p.second,nullptr);layouts.clear();for(auto s:set_layouts)vkDestroyDescriptorSetLayout(cfg.device,s,nullptr);set_layouts.clear();
    for(auto &p:passes)vkDestroyRenderPass(cfg.device,p.second,nullptr);passes.clear();for(auto &m:modules)vkDestroyShaderModule(cfg.device,m.second,nullptr);modules.clear();ids.clear();id_keys.clear();recs.clear();known.clear();rewritten.clear();}
bool take(const std::string &bytes,VkPipeline &out){std::lock_guard<std::mutex> l(mu);auto it=ready.find(bytes);if(it==ready.end())return false;out=it->second;ready.erase(it);return true;}
}
// Diagnostic (marker gfx-pipe-stats, with the warm-up off): the driver's statistics of each pipeline as it is created, and
// the draws per frame of each pipeline over frames 3000-3059.
PFN_vkGetPipelineExecutablePropertiesKHR pstat_props=nullptr;PFN_vkGetPipelineExecutableStatisticsKHR pstat_stats=nullptr;bool pipe_stats=false,pstat_logged=false;
struct PstatUse{uint64_t draws=0;std::string what;};std::unordered_map<VkPipeline,PstatUse> pstat_use;
std::string pstat_what(){std::string s;for(unsigned st=0;st<6;st++)if(state.shader[st]){if(!s.empty())s+=';';s+=std::to_string(st)+":"+state.shader[st]->key;}return s;}
void pstat_log(VkPipeline pl){
    VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};pi.pipeline=pl;uint32_t n=0;if(pstat_props(cfg.device,&pi,&n,nullptr)!=VK_SUCCESS)return;
    std::vector<VkPipelineExecutablePropertiesKHR> ex(n,VkPipelineExecutablePropertiesKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});pstat_props(cfg.device,&pi,&n,ex.data());
    for(uint32_t i=0;i<n;i++){VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};ei.pipeline=pl;ei.executableIndex=i;uint32_t m=0;pstat_stats(cfg.device,&ei,&m,nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> st(m,VkPipelineExecutableStatisticKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});pstat_stats(cfg.device,&ei,&m,st.data());
        std::string line;for(auto &x:st){char v[48];switch(x.format){case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:snprintf(v,sizeof v,"%u",x.value.b32);break;case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:snprintf(v,sizeof v,"%lld",(long long)x.value.i64);break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:snprintf(v,sizeof v,"%llu",(unsigned long long)x.value.u64);break;default:snprintf(v,sizeof v,"%g",x.value.f64);}
            std::string name=x.name;for(auto &c:name)if(c==' ' || c=='=')c='_';line+=" "+name+"="+v;}
        std::string en=ex[i].name;for(auto &c:en)if(c==' ')c='_';
        xbg_logf("NVN PSTAT pipeline=%p what=%s exec=%u stages=%#x name=%s subgroup=%u%s",(void*)pl,pstat_what().c_str(),i,unsigned(ex[i].stages),en.c_str(),ex[i].subgroupSize,line.c_str());}
}
// 有的驱动在自己的着色器编译器里崩掉建管线的调用（Adreno 750：libllvm-qgl.so 段错误，计算与像素着色器都有，#153、#166），
// 有的直接报错（Adreno 650：-13）。每个建管线的线程（游戏的图形线程、预热线程、异步线程）建之前把这条管线的着色器写进自己的
// 记号文件 gfx-pipe-inflight.<槽>，建完删掉；下次启动还在的就是上次死时正在建的。并发建的线程不止一个，进程也可能只是正好
// 在那几毫秒里被系统杀掉，所以要连着两次启动都在场的才记进 gfx-pipe-skip（崩点固定的那条两次都在，旁边碰巧在建的很少两次都在），之后用它的派发 / 绘制一律跳过（少一个
// 效果，不再每次启动都崩）。驱动报错的当场记下跳过。删掉 gfx-pipe-skip 即恢复。条目：计算管线是 c: 加着色器键，图形管线是
// g: 加各阶段键（+ 连接），只跳这一种组合，不连累共用同一顶点着色器的其它绘制。没有前缀的旧条目是 v2.0.1 在没有内联
// （gfx-inline）时记的，读到就作废：内联后多半能编，真还崩就再记一次。
// Some drivers crash inside their own shader compiler while building a pipeline (Adreno 750: a segfault in libllvm-qgl.so,
// on compute and pixel shaders alike, #153, #166); others return an error (Adreno 650: -13). Every thread that builds
// pipelines (the game's graphics thread, the warm-up threads, the async thread) writes the pipeline's shaders to its own marker
// gfx-pipe-inflight.<slot> before the build and removes it afterwards; a marker the next boot still finds was being built when
// the last one died. Several threads build at once, and the process may just have been killed during those milliseconds, so an entry goes into gfx-pipe-skip only when it was in flight on two boots
// in a row (the one the crash is pinned to is there both times; a bystander rarely is), and its dispatches / draws are skipped
// from then on (one effect fewer instead of a crash on every boot). A driver error is recorded on the spot. Deleting
// gfx-pipe-skip restores them. Entries: a compute pipeline is c: and its shader key, a graphics pipeline is g: and its stage
// keys joined by +, so only that combination is skipped and other draws sharing its vertex shader are not. Entries without a
// prefix were recorded by v2.0.1 without inlining (gfx-inline) and are dropped when read: inlined, the shader most likely
// builds, and if it still crashes it is recorded again.
namespace pipeguard{
std::mutex mu,serial;std::unordered_set<std::string> skip,told,proven;bool loaded=false,careful=false;std::atomic<int> slots{0};std::string crash_test;
// np655：上游 fec76d1（#170/#203）的 pipeline guard——上次死时有在建的管线，这次启动就把还没
//   建成过的管线一条一条建（careful：建之前拿同一把锁 serial、建完放）。预热线程每次按清单
//   同样的顺序取，驱动在罪魁上再崩时，旁边线程上的也多半又在建，并发着建会把它们一起冤枉进
//   gfx-pipe-skip；一条一条建，第二次死时在建的只有罪魁。已建成过的不排队，正常启动照常并行。
//   测试钩子 DM_GFX_PIPE_CRASH_TEST=<键>：轮到建含这个着色器键的管线时进程当场 _exit（留下
//   在建记号、不留崩溃转储）。
//   对 Maleoon 直接相关：这颗驱动的 reset/-4 高发（np609 起的整条链），驱动死在自家编译器里时
//   正好用这个把罪魁管线隔离出来，而不是整轮 reset 之后无从追查。
// np655: upstream fec76d1 (#170/#203) pipeline guard. When the last boot died with pipelines in
//   flight, this boot builds the not-yet-built ones one at a time (careful: take the same serial
//   lock before the build, release after). Warm-up threads take the list in the same order every
//   boot, so a second driver crash on the culprit has only the culprit in flight -- concurrent
//   builds would wrongly blame their neighbours into gfx-pipe-skip. Already-built pipelines do not
//   queue; normal boots stay parallel. Test hook DM_GFX_PIPE_CRASH_TEST=<key>: the process _exits
//   when a pipeline holding that shader key is about to be built (in-flight marker stays, no dump).
//   Directly relevant to Maleoon: this driver resets / returns -4 a lot (the whole np609+ chain),
//   and when the driver dies inside its own compiler this isolates the culprit pipeline instead of
//   leaving an unattributable reset.
std::string base(){std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');return cut==std::string::npos?dir:dir.substr(0,cut);}
std::string marker(){thread_local const int slot=slots.fetch_add(1);return base()+"/gfx-pipe-inflight."+std::to_string(slot);}
// 有阶段认不出键（引擎自己的模块没登记）就返回空串：这条管线不进兜底，免得不同的管线撞成同一个条目。
// An empty string when a stage has no known key (the engine's own modules are not registered): that pipeline is left out
// of the guard, so different pipelines cannot collide into one entry.
std::string compute_key(const std::string &key){return "c:"+key;}
std::string graphics_key(const std::vector<std::string> &keys){std::string k="g:";for(size_t i=0;i<keys.size();i++){if(keys[i].empty())return {};if(i)k+='+';k+=keys[i];}return k;}
void record_locked(const std::string &key,const char *why){
    if(!skip.insert(key).second)return;
    if(FILE *f=fopen((base()+"/gfx-pipe-skip").c_str(),"ab")){fprintf(f,"%s\n",key.c_str());fclose(f);}
    xbg_logf("NVN pipeline %s %s; it is skipped from now on (gfx-pipe-skip)",key.c_str(),why);
}
void record(const std::string &key,const char *why){if(key.empty())return;std::lock_guard<std::mutex> l(mu);record_locked(key,why);}
std::vector<std::string> lines(const std::string &path){std::vector<std::string> out;char line[512];
    if(FILE *f=fopen(path.c_str(),"rb")){while(fgets(line,sizeof line,f)){line[strcspn(line,"\r\n")]=0;if(*line)out.push_back(line);}fclose(f);}return out;}
// 引擎初始化时、预热线程开始建管线之前调用：那之后记号文件就是这次启动的了。
// Called while the engine initializes, before the warm-up threads build anything: from then on the markers are this boot's.
void load(){
    std::lock_guard<std::mutex> l(mu);if(loaded)return;loaded=true;const std::string b=base();
    size_t stale=0;for(auto &k:lines(b+"/gfx-pipe-skip")){if(!k.compare(0,2,"c:") || !k.compare(0,2,"g:"))skip.insert(k);else ++stale;}
    if(stale){std::string keep;for(auto &k:skip)keep+=k+"\n";if(FILE *f=fopen((b+"/gfx-pipe-skip").c_str(),"wb")){fputs(keep.c_str(),f);fclose(f);}
        xbg_logf("NVN pipelines skipped by an older version dropped=%zu (gfx-pipe-skip)",stale);}
    std::unordered_set<std::string> now;
    if(DIR *d=opendir(b.c_str())){while(dirent *e=readdir(d))if(!strncmp(e->d_name,"gfx-pipe-inflight",17)){const std::string path=b+"/"+e->d_name;for(auto &k:lines(path))if(!k.compare(0,2,"c:") || !k.compare(0,2,"g:"))now.insert(k);remove(path.c_str());}closedir(d);}
    const std::string suspect=b+"/gfx-pipe-suspect";std::unordered_set<std::string> before;for(auto &k:lines(suspect))before.insert(k);
    std::string keep;
    for(auto &k:now){if(before.count(k))record_locked(k,"was being built when the last two boots ended");else{keep+=k+"\n";xbg_logf("NVN pipeline %s was being built when the last boot ended",k.c_str());}}
    if(keep.empty())remove(suspect.c_str());else if(FILE *f=fopen(suspect.c_str(),"wb")){fputs(keep.c_str(),f);fclose(f);}
    if(!skip.empty())xbg_logf("NVN pipelines skipped=%zu (gfx-pipe-skip)",skip.size());
    careful=!now.empty();if(careful)xbg_logf("NVN pipelines not built yet are built one at a time this boot (%zu were being built when the last boot ended)",now.size());
    if(const char *e=getenv("DM_GFX_PIPE_CRASH_TEST"))if(*e){crash_test=e;xbg_logf("NVN pipeline crash test armed for shader key %s",e);}
}
bool skipped(const std::string &key){if(key.empty())return false;load();std::lock_guard<std::mutex> l(mu);if(!skip.count(key))return false;if(told.insert(key).second)xbg_logf("NVN pipeline %s skipped (gfx-pipe-skip)",key.c_str());return true;}
// 一条管线第一次建成之后就不再写记号：同样的着色器换状态再建不会走到编译器的另一条路，也省掉之后的文件操作。
// Once a pipeline has built once it is no longer marked: rebuilding the same shaders with other state does not take the
// compiler down a different path, and later builds do not pay for the file operations.
thread_local bool marking=false,holding=false;
void begin(const std::string &key){if(key.empty()){marking=false;return;}load();bool one=false;{std::lock_guard<std::mutex> l(mu);marking=!proven.count(key);one=marking && careful;}
    if(one){serial.lock();holding=true;}
    if(marking)if(FILE *f=fopen(marker().c_str(),"wb")){fprintf(f,"%s\n",key.c_str());fclose(f);}
    if(marking && !crash_test.empty() && key.find(crash_test)!=std::string::npos){xbg_logf("NVN pipeline crash test: exiting while building %s",key.c_str());_exit(134);}}
void end(const std::string &key,bool ok){if(!marking)return;marking=false;remove(marker().c_str());if(ok){std::lock_guard<std::mutex> l(mu);proven.insert(key);}if(holding){holding=false;serial.unlock();}}
}
void pipeline_lookup(PipelineLease &out,const XbgPipelineKey &key,bool cacheable,const VkGraphicsPipelineCreateInfo *graphics,const VkComputePipelineCreateInfo *compute,bool may_skip=false){
    static std::string bytes;bytes.clear();   // reused key buffer (graphics thread only)
    if(cacheable){
        bytes.assign(reinterpret_cast<const char*>(key.words),size_t(key.count)*4);
        auto found=pipeline_objects.find(bytes);
        if(found!=pipeline_objects.end()){++pipeline_counts.hits;found->second.last_used=++pipeline_tick;out.pipeline=found->second.pipeline;return;}
        ++pipeline_counts.misses;
    }else ++pipeline_counts.bypass;
    if(graphics && cacheable && warm::take(bytes,out.pipeline)){++warm::hits;}
    // A draw that may be skipped gets no pipeline while its build is queued (warm::async_request); it is not cached.
    else if(graphics && cacheable && may_skip && warm::async_request(*graphics,key,bytes)){out.pipeline=VK_NULL_HANDLE;out.transient=true;++warm::async_skips;return;}
    else if(graphics){std::vector<std::string> keys;{std::lock_guard<std::mutex> l(warm::mu);for(uint32_t i=0;i<graphics->stageCount;i++){auto it=warm::ids.find(graphics->pStages[i].module);keys.push_back(it==warm::ids.end()?std::string():warm::id_keys[it->second]);}}
        const std::string guard=pipeguard::graphics_key(keys);if(pipeguard::skipped(guard)){out.pipeline=VK_NULL_HANDLE;out.transient=true;return;}
        uint64_t t0=timing_clock();VkGraphicsPipelineCreateInfo g=*graphics;if(pipe_stats)g.flags|=VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;pipeguard::begin(guard);const VkResult made=[&]{deathguard::VkSite site(deathguard::PIPE_GRAPHICS,deathguard::ctx_of(guard.c_str()));return vkCreateGraphicsPipelines(cfg.device,pipeline_cache,1,&g,nullptr,&out.pipeline);}();pipeguard::end(guard,made==VK_SUCCESS);
        deathguard::pipeline_built(guard.c_str(),(timing_clock()-t0)/1000000ull);
        if(made!=VK_SUCCESS){char why[64];snprintf(why,sizeof why,"failed to build (result=%d)",int(made));pipeguard::record(guard,why);out.pipeline=VK_NULL_HANDLE;out.transient=true;return;}
        if(pipe_stats)pstat_log(out.pipeline);uint64_t dt=timing_clock()-t0;if(dt>=compile_ns)++pipelines_unsaved;++warm::creates;warm::create_ns+=dt;++hud::pipelines;hud::pipeline_ns+=dt;warm::create_max_ns=std::max(warm::create_max_ns,dt);if(cacheable)warm::record(*graphics,key);}
    else{uint64_t t0=timing_clock();VkComputePipelineCreateInfo cc=*compute;if(pipe_stats)cc.flags|=VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;const std::string shader_key=pipeguard::compute_key(state.shader[5]->key);pipeguard::begin(shader_key);const VkResult made=[&]{deathguard::VkSite site(deathguard::PIPE_COMPUTE,deathguard::ctx_of(shader_key.c_str()));return vkCreateComputePipelines(cfg.device,pipeline_cache,1,&cc,nullptr,&out.pipeline);}();pipeguard::end(shader_key,made==VK_SUCCESS);deathguard::pipeline_built(shader_key.c_str(),(timing_clock()-t0)/1000000ull);if(made!=VK_SUCCESS){char why[64];snprintf(why,sizeof why,"failed to build (result=%d)",int(made));pipeguard::record(shader_key,why);out.pipeline=VK_NULL_HANDLE;out.transient=true;return;}++pipelines_unsaved;++hud::pipelines;hud::pipeline_ns+=timing_clock()-t0;if(pipe_stats)pstat_log(out.pipeline);}
    if(!cacheable){out.transient=true;return;}
    if(pipeline_objects.size()==pipeline_object_limit){
        // All prior draws/dispatches waited their fence. This lookup occurs
        // before binding the current pipeline; no retained entry is in flight.
        auto oldest=std::min_element(pipeline_objects.begin(),pipeline_objects.end(),[](const auto &a,const auto &b){return a.second.last_used<b.second.last_used;});
        {VkPipeline old=oldest->second.pipeline;retire_handle([old]{vkDestroyPipeline(cfg.device,old,nullptr);});for(auto f=pipeline_fast.begin();f!=pipeline_fast.end();)if(f->second==old)f=pipeline_fast.erase(f);else ++f;pipe_memo_valid=false;}pipeline_objects.erase(oldest);++pipeline_counts.evictions;
    }
    pipeline_objects.emplace(std::string(bytes),PipelineEntry{out.pipeline,++pipeline_tick});
}
void clear_pipeline_objects(){for(auto &p:pipeline_objects)vkDestroyPipeline(cfg.device,p.second.pipeline,nullptr);pipeline_objects.clear();pipeline_fast.clear();pipe_memo_valid=false;}
struct PipelineDefinition {
    std::vector<uint32_t> words;bool valid=false;
    void assign(const XbgPipelineKey &k,bool ok){valid=ok && k.valid;if(valid)words.assign(k.words,k.words+k.count);}
    XbgPipelineKeyView view()const{return {words.data(),uint32_t(words.size()),valid};}
};
// Descriptor set layouts, pipeline layouts and samplers are immutable for a
// given binding list / sampler state, so they are created once and cached.
// Descriptor sets come from per-slot pools that are reset after the slot fence.
struct LayoutEntry {VkDescriptorSetLayout layout{};VkPipelineLayout pipeline_layout{};PipelineDefinition definition;VkDescriptorUpdateTemplate tmpl{};std::vector<uint64_t> tmpl_sig;};
// One descriptor payload slot for update templates (image/buffer info or texel view).
union DescEntry{VkDescriptorImageInfo image;VkDescriptorBufferInfo buffer;VkBufferView view;};

std::map<std::vector<uint32_t>,LayoutEntry> layout_cache;
std::map<std::array<const void*,7>,LayoutEntry*> shader_layouts; // shader set -> layout (bindings are immutable per shader)
std::map<std::array<uint32_t,20>,VkSampler> sampler_cache;
Arena static_arena; // never reset: immutable per-shader c1 constants
std::map<std::pair<float,float>,Buffer> support_static; // support block per viewport size (contents depend on it only)
std::map<size_t,Buffer> zero_blocks;   // np612：未绑定声明槽用的零缓冲，与 support_static 同寿命（static_arena），清理要跟它一起做
struct SamplerFastKey{std::array<uint32_t,11> w;bool operator==(const SamplerFastKey &o)const{return w==o.w;}};
struct SamplerFastHash{size_t operator()(const SamplerFastKey &k)const{uint64_t h=1469598103934665603ull;for(uint32_t v:k.w){h^=v;h*=1099511628211ull;}return size_t(h);}};
std::unordered_map<SamplerFastKey,VkSampler,SamplerFastHash> sampler_fast;
void clear_descriptor_caches(){sampler_fast.clear();pipeline_fast.clear();pipe_memo_valid=false;
    for(auto &l:layout_cache){if(l.second.tmpl)vkDestroyDescriptorUpdateTemplate(cfg.device,l.second.tmpl,nullptr);vkDestroyPipelineLayout(cfg.device,l.second.pipeline_layout,nullptr);vkDestroyDescriptorSetLayout(cfg.device,l.second.layout,nullptr);}layout_cache.clear();shader_layouts.clear();
    for(auto &x:sampler_cache)vkDestroySampler(cfg.device,x.second,nullptr);sampler_cache.clear();
    static_arena.release();support_static.clear();zero_blocks.clear();
}
VkDescriptorSet allocate_set(VkDescriptorSetLayout layout){
    Slot &s=recording_slot();
    for(;;){
        if(s.pool_next==s.pools.size()){
            std::vector<VkDescriptorPoolSize> sizes;for(uint32_t t=0;t<=VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;t++)sizes.push_back({VkDescriptorType(t),t==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER||t==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER||t==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC||t==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER?16384u:2048u});
            VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};ci.maxSets=2048;ci.poolSizeCount=sizes.size();ci.pPoolSizes=sizes.data();
            VkDescriptorPool pool;VK(vkCreateDescriptorPool(cfg.device,&ci,nullptr,&pool));s.pools.push_back(pool);
        }
        // Sets are allocated 16 at a time per layout and handed out from a per-slot spare list.
        auto &spare=s.spare_sets[layout];auto &used=s.used_sets[layout];if(!spare.empty()){VkDescriptorSet set=spare.back();spare.pop_back();used.push_back(set);return set;}
        VkDescriptorSetLayout layouts[16];for(auto &l:layouts)l=layout;VkDescriptorSet sets[16]{};
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=s.pools[s.pool_next];ai.descriptorSetCount=16;ai.pSetLayouts=layouts;
        VkResult r=vkAllocateDescriptorSets(cfg.device,&ai,sets);
        if(r!=VK_SUCCESS){ai.descriptorSetCount=1;r=vkAllocateDescriptorSets(cfg.device,&ai,sets);if(r==VK_SUCCESS){used.push_back(sets[0]);return sets[0];}}
        else{spare.insert(spare.end(),sets+1,sets+16);used.push_back(sets[0]);return sets[0];}
        if(r==VK_ERROR_OUT_OF_POOL_MEMORY || r==VK_ERROR_FRAGMENTED_POOL){++s.pool_next;continue;}
        check(r,"vkAllocateDescriptorSets");
    }
}
struct Descriptors {
    const LayoutEntry *entry=nullptr;VkDescriptorSet set{};VkPipelineLayout pipeline_layout{};std::vector<uint32_t> dynamic_offsets;
    std::vector<Buffer> buffers;std::vector<VkBufferView> buffer_views; // buffers: reserved up front, pointers stay valid
    std::vector<std::pair<std::pair<uint64_t,uint64_t>,Buffer*>> storage; // alias lookup for storage bindings
    std::vector<std::pair<std::pair<uint64_t,uint64_t>,Buffer*>> written; // ranges the GPU may write
    struct InputImage {Image *image;view_t view;bool has_view;std::string label;};
    std::vector<InputImage> capture_inputs;
    ~Descriptors(){for(auto x:buffer_views)vkDestroyBufferView(cfg.device,x,nullptr);}
    Buffer *keep(const Buffer &b){need(buffers.size()<buffers.capacity(),"Descriptor buffer reservation exceeded");buffers.push_back(b);return &buffers.back();}
    bool needs_writeback()const{return !written.empty();}
    void defer_writeback(){const void *slot=&slots[slot_index];for(auto &v:written){if(draw_detail)draw_timing.storage(v.first.second);snapshot_invalidate(v.first.first,v.first.second);pending_writebacks.push_back({v.first.first,v.first.second,*v.second,slot});}}
    void writeback(){for(auto &v:written){if(draw_detail)draw_timing.storage(v.first.second);v.second->invalidate();invalidate_cpu_range(v.first.first,v.first.second);memcpy(reinterpret_cast<void*>(v.first.first),v.second->mapped,v.first.second);}}
    void capture_read_images(){for(auto &i:capture_inputs)capture_image(*i.image,i.has_view?&i.view:nullptr,i.label.c_str());}
};
// Per-draw descriptor records were one make_shared + vector reservations each (≈1800/frame, slow musl malloc). They live
// until the slot's fence like before, then return to a free list with their capacity.
std::vector<std::unique_ptr<Descriptors>> desc_all;std::vector<Descriptors*> desc_free;
Descriptors &acquire_descriptors(){
    Descriptors *d;if(desc_free.empty()){desc_all.push_back(std::make_unique<Descriptors>());d=desc_all.back().get();}else{d=desc_free.back();desc_free.pop_back();}
    recording_slot().desc_used.push_back(d);return *d;
}
void release_descriptors(Slot &s){
    for(Descriptors *d:s.desc_used){for(auto x:d->buffer_views)vkDestroyBufferView(cfg.device,x,nullptr);
        d->entry=nullptr;d->set=VK_NULL_HANDLE;d->pipeline_layout=VK_NULL_HANDLE;d->dynamic_offsets.clear();d->buffers.clear();d->buffer_views.clear();d->storage.clear();d->written.clear();d->capture_inputs.clear();
        desc_free.push_back(d);}
    s.desc_used.clear();
}
// Fast path (iOS: samplers fetched directly instead of re-deriving per draw): raw NVN parameters +
// view format -> VkSampler, before the full conversion/validation below.
VkSampler sampler_slow(const samp_par &p,Image &im,VkFormat viewed,bool integer_border);
VkSampler sampler(const samp_par &p,Image &im,VkFormat viewed,bool integer_border){
    auto bits=[](float f){uint32_t u;memcpy(&u,&f,4);return u;};
    SamplerFastKey k{{uint32_t(p.minf)|uint32_t(p.magf)<<8|uint32_t(p.ws)<<16|uint32_t(p.wt)<<24,uint32_t(p.wr)|uint32_t(p.cmp_mode)<<8|uint32_t(p.cmp_func)<<16|uint32_t(integer_border)<<24,
        bits(p.lod_min),bits(p.lod_max),bits(p.lod_bias),bits(p.aniso),bits(p.border[0]),bits(p.border[1]),bits(p.border[2]),bits(p.border[3]),uint32_t(viewed)}};
    auto it=sampler_fast.find(k);if(it!=sampler_fast.end())return it->second;
    VkSampler out=sampler_slow(p,im,viewed,integer_border);sampler_fast.emplace(k,out);return out;
}
VkSampler sampler_slow(const samp_par &p,Image &im,VkFormat viewed,bool integer_border){
    // np602：惰性探测一次 VK_EXT_custom_border_color。扩展必须**在设备创建时**就启用（dm/android/
    // vulkan_present.cpp 只要设备报告该扩展就会启用），这里只确认"扩展在 + 特性支持"，并打一行读数。
    // np602: probe VK_EXT_custom_border_color once. The extension has to be enabled at device creation
    // (dm/android/vulkan_present.cpp enables it whenever the device reports it); this only confirms
    // "extension present + feature supported" and logs one line.
    {static int probed=0;if(!probed){probed=1;
        uint32_t n=0;if(vkEnumerateDeviceExtensionProperties(cfg.physical,nullptr,&n,nullptr)==VK_SUCCESS && n){
            std::vector<VkExtensionProperties> ext(n);
            if(vkEnumerateDeviceExtensionProperties(cfg.physical,nullptr,&n,ext.data())==VK_SUCCESS){
                bool present=false;for(auto &e:ext)if(!strcmp(e.extensionName,VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME))present=true;
                if(present){
                    VkPhysicalDeviceCustomBorderColorFeaturesEXT f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
                    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f2.pNext=&f;
                    vkGetPhysicalDeviceFeatures2(cfg.physical,&f2);
                    custom_border_color_enabled=f.customBorderColors;custom_border_without_format=f.customBorderColorWithoutFormat;
                }
                xbg_logf("NVN caps customBorderColor ext=%d colors=%u withoutFormat=%u",int(present),unsigned(custom_border_color_enabled),unsigned(custom_border_without_format));
            }}}}
    xbg_nvn_sampler_desc input{p.minf,p.magf,p.ws,p.wt,p.wr,p.cmp_mode,p.cmp_func,p.lod_min,p.lod_max,p.lod_bias,p.aniso,{p.border[0],p.border[1],p.border[2],p.border[3]}};
    // np604：`DM_SAMPLER_BORDER=transparent|black|white` —— **设备没有自定义边框色时**的显式近似开关。
    // 不设就保持严格（-7 + FATAL）。设了就每出现一种新边框色打一行"我把它换成了什么"，
    // 让"画面可能不对"这件事在日志里无法被忽略。
    // np604: DM_SAMPLER_BORDER=transparent|black|white -- the explicit approximation switch for devices without
    // custom border colours. Unset keeps it strict (-7 + FATAL). When set, every distinct border colour gets one
    // line saying what it was replaced with, so "the picture may be wrong" cannot be missed in the log.
    static const uint32_t border_approx=[](){const char *v=getenv("DM_SAMPLER_BORDER");if(!v||!*v)return 0u;
        if(!strcmp(v,"transparent"))return 1u;if(!strcmp(v,"black"))return 2u;if(!strcmp(v,"white"))return 3u;
        xbg_logf("【红】DM_SAMPLER_BORDER=%s 不认识（transparent|black|white），按严格处理",v);return 0u;}();
    xbg_sampler_limits limits{cfg.enabled_core.samplerAnisotropy,sampler_mirror_clamp_enabled,VkBool32(integer_border),properties.limits.maxSamplerAnisotropy,properties.limits.maxSamplerLodBias,custom_border_color_enabled,border_approx};
    float border_rgba[4]={0,0,0,0};
    VkSamplerCreateInfo s{};int result=xbg_sampler_from_nvn(&input,&limits,&s,border_rgba);
    if(!result && border_approx && !custom_border_color_enabled && (p.ws==5||p.wt==5||p.wr==5)){
        static std::set<std::array<uint32_t,4>> seen;std::array<uint32_t,4> bk{};for(int i=0;i<4;i++){memcpy(&bk[size_t(i)],&p.border[i],4);}
        if(seen.insert(bk).second)xbg_logf("NVN sampler border 近似：%g,%g,%g,%g → %s（设备无 custom border colour）",
            double(p.border[0]),double(p.border[1]),double(p.border[2]),double(p.border[3]),
            border_approx==1?"透明黑":border_approx==2?"不透明黑":"不透明白");
    }
    if(result)xbg_fatalf("NVN sampler %s result=%d min=%u mag=%u wrap=%u/%u/%u lod=%g/%g bias=%g aniso=%g compare=%u/%u border=%g,%g,%g,%g texture=%llu format=%#x vk=%u mirrorEnabled=%u",xbg_sampler_error(result),result,p.minf,p.magf,p.ws,p.wt,p.wr,p.lod_min,p.lod_max,p.lod_bias,p.aniso,p.cmp_mode,p.cmp_func,p.border[0],p.border[1],p.border[2],p.border[3],(unsigned long long)im.source.serial,im.source.d.format,unsigned(viewed),sampler_mirror_clamp_enabled);
    const VkFormatProperties &fp=format_properties(viewed);
    if((s.minFilter==VK_FILTER_LINEAR || s.magFilter==VK_FILTER_LINEAR) && !(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))xbg_fatalf("NVN sampler linear filtering unsupported min=%u mag=%u compare=%u/%u format=%#x vk=%u optimalFeatures=%#x texture=%llu decoded=%d imfmt=%u",p.minf,p.magf,p.cmp_mode,p.cmp_func,im.source.d.format,unsigned(viewed),fp.optimalTilingFeatures,(unsigned long long)im.source.serial,int(im.decoded),unsigned(im.fmt));
    // np602：自定义边框色要在 pNext 上挂 VkSamplerCustomBorderColorCreateInfoEXT。format 只在
    // customBorderColorWithoutFormat 可用时才能写 UNDEFINED，否则必须给**实际采样格式**（viewed）。
    // np602: a custom border colour needs VkSamplerCustomBorderColorCreateInfoEXT on pNext. `format` may be
    // UNDEFINED only with customBorderColorWithoutFormat; otherwise it must be the real sampled format (viewed).
    VkSamplerCustomBorderColorCreateInfoEXT cbc{VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    if(s.borderColor==VK_BORDER_COLOR_FLOAT_CUSTOM_EXT || s.borderColor==VK_BORDER_COLOR_INT_CUSTOM_EXT){
        for(int k=0;k<4;k++)cbc.customBorderColor.float32[k]=border_rgba[k];
        cbc.format=custom_border_without_format?VK_FORMAT_UNDEFINED:viewed;
        s.pNext=&cbc;
    }
    need(s.pNext==nullptr || s.pNext==&cbc,"Unexpected sampler extension chain");
    auto bits=[](float f){uint32_t u;memcpy(&u,&f,4);return u;};
    // np602：缓存键补上自定义边框色的四个分量——否则 borderColor 都是 *_CUSTOM_EXT 的两个不同颜色会撞同一个键。
    // np602: the cache key also carries the four custom border components; otherwise two different custom colours
    // share borderColor = *_CUSTOM_EXT and collide on one key.
    std::array<uint32_t,20> key{s.flags,uint32_t(s.magFilter),uint32_t(s.minFilter),uint32_t(s.mipmapMode),uint32_t(s.addressModeU),uint32_t(s.addressModeV),uint32_t(s.addressModeW),bits(s.mipLodBias),s.anisotropyEnable,bits(s.maxAnisotropy),s.compareEnable,uint32_t(s.compareOp),bits(s.minLod),bits(s.maxLod),uint32_t(s.borderColor),s.unnormalizedCoordinates,
        bits(border_rgba[0]),bits(border_rgba[1]),bits(border_rgba[2]),bits(border_rgba[3])};
    auto it=sampler_cache.find(key);if(it!=sampler_cache.end())return it->second;
    VkSampler out;VK(vkCreateSampler(cfg.device,&s,nullptr,&out));sampler_cache.emplace(key,out);return out;
}
void fill_driver(uint8_t *data,unsigned ns,unsigned bank){
    if(bank==0){unsigned base=ns==0?0x110:ns==5?0x310:ns==1?0x510:0;need(base,"Unverified storage-table base for tessellation/geometry stage");for(unsigned r=0;r<16;r++){memcpy(data+base+16*r,&state.sb[ns][r].address,8);uint32_t n=uint32_t(state.sb[ns][r].size);memcpy(data+base+16*r+8,&n,4);}if(ns==5)for(unsigned r=0;r<16;r++){memcpy(data+0x210+16*r,&state.ub[ns][r].address,8);uint32_t n=uint32_t(state.ub[ns][r].size);memcpy(data+0x218+16*r,&n,4);}}
    else{for(unsigned r=0;r<128;r++){memcpy(data+0x168+8*r,&state.tex[ns][r],8);memcpy(data+0x568+8*r,&state.samp[ns][r],8);}for(unsigned r=0;r<9;r++)memcpy(data+0x120+8*r,&state.img[ns][r],8);}
}
// Uniform data is bound with the guest-bound size (rounded to 16 bytes); reads
// past it rely on robustBufferAccess like the reference Ryujinx Vulkan backend.
size_t uniform_range(size_t copy){return std::max<size_t>(16,(copy+15)&~size_t(15));}
Buffer uniform_snapshot_slow(uint64_t address,size_t copy);
Buffer uniform_snapshot_cached(uint64_t address,size_t copy);
// ---------- 越肩 / 第一人称镜头（实验，缺省关；DM_CAM=1 越肩、2 第一人称。宿主 dm/host/dm_cam.c，算法 xbg_cam.h，设计见 docs/over-the-shoulder-camera.zh-CN.md）----------
// 同 nvn_mtl.m 的 cam_patch：主相机常量块（608 字节）与片元槽 10 的场景常量块（1440 字节）按内容认出，绑一份改过的拷贝。拷贝不进快照缓存，
// 所以不改的那条按正交视锥剔除的计算着色器（cull_cs）读到的仍是原版；阴影图那几份是光源视角，签名不匹配。
// Vulkan 只绑着色器读得到的那段（spirv_uniform_extent），只读前几行的着色器拿到的拷贝不到 608 字节：认出主相机或场景块就把绑定长度放到整块（bound），
// 描述符备忘的键、绑定范围与命中时的重拍都用这个长度。Vulkan 没有运行时外推，不用关。
// ---------- Over-the-shoulder / first-person camera (experimental, off by default; DM_CAM=1 over-the-shoulder, 2 first person. Host dm/host/dm_cam.c,
// math xbg_cam.h, design in docs/over-the-shoulder-camera.md) ----------
// As nvn_mtl.m's cam_patch: the main camera constant block (608 bytes) and the scene constant block of fragment slot 10 (1440 bytes) are recognised by
// their content and bound as a modified copy. Copies stay out of the snapshot caches, so the compute shader that culls against the orthographic frustum
// (cull_cs), which is left alone, still reads the original; the shadow-map copies use the light's view and do not match the signature.
// Vulkan binds only the part a shader can read (spirv_uniform_extent), so a shader reading only the first rows gets a copy shorter than 608 bytes: when the
// main camera or scene block is recognised the bound length becomes the whole block (bound), and the descriptor memo key, the bound range and the
// re-snapshot on a memo hit all use that length. Vulkan has no runtime extrapolation to switch off.
namespace cam {
const char *const cull_cs="1188eaedc8db3285";
int mode_=-1;xbg_cam_params params;xbg_cam_state st;uint64_t patches=0,scenes=0,widened=0;
float clear=0,host_px=0,host_pz=0,turned=0,drag_k=0.5f,pull_pitch=40,pull_min=5,pull_near=10,pull_blend=4;bool clear_ok=false;uint64_t drift_frame=0;
struct Copy{uint64_t address=0,bytes=0,epoch=0;Buffer buffer;};std::array<Copy,8> copies;unsigned next=0;
float envf(const char *k,float d){const char *e=getenv(k);return e?float(atof(e)):d;}
int mode(){
    // 档可在游戏中切换（自定义页）：变了就按新档重取参数、重置跟随状态
    // The mode can switch in game (custom page): on a change, reload parameters and reset the follow state
    const int m=dm_cam_mode?dm_cam_mode():0;
    if(m!=mode_){mode_=m;
        if(mode_>0){params=xbg_cam_default_params(mode_);
            // 原神式（宿主 dm_cam_style = 1）的越肩缺省机位：没有上下拖调俯仰，视野靠机位给——前看 7、水平视角 70°、上沿不设远处限制
            // （画面上沿只保留水平线下 3°）。远处游戏还没画的地方会露黑块，用户在 OnePlus 13 上看过：视野更要紧，可以接受。下面的环境变量照样能改
            // Over-the-shoulder defaults of the Genshin style (host dm_cam_style = 1): with no dragged pitch, the view has to come from the rig -- look-ahead 7,
            // horizontal field of view 70 degrees, no far limit for the top edge (only the 3 degrees below the horizon stays). Far ground the game has not drawn
            // yet shows black patches; the user saw them on a OnePlus 13 and judged the view more important. The environment variables below still override
            if(mode_==1 && dm_cam_style && dm_cam_style()==1){params.look=7;params.fov=70;params.reach=0;}
            params.dist=envf("DM_CAM_DIST",params.dist);params.h=envf("DM_CAM_H",params.h);params.look=envf("DM_CAM_LOOK",params.look);params.lookh=envf("DM_CAM_LOOKH",params.lookh);
            params.eye=envf("DM_CAM_EYE",params.eye);params.fwd=envf("DM_CAM_FWD",params.fwd);params.pitch=envf("DM_CAM_PITCH",params.pitch);params.fov=envf("DM_CAM_FOV",params.fov);
            params.tau=envf("DM_CAM_TAU",params.tau);params.near_=envf("DM_CAM_NEAR",params.near_);params.reach=envf("DM_CAM_REACH",params.reach);drag_k=envf("DM_CAM_DRAG",0.5f);pull_pitch=envf("DM_CAM_PULL_PITCH",40);pull_min=envf("DM_CAM_PULL_MIN",5);pull_near=envf("DM_CAM_PULL_NEAR",10);pull_blend=envf("DM_CAM_PULL_BLEND",4);xbg_cam_init(&st);
            xbg_logf("NVN cam mode=%d dist=%g h=%g look=%g lookh=%g eye=%g fwd=%g pitch=%g fov=%g tau=%g near=%g reach=%g drag=%g pull=%g/%g near=%g+%g",mode_,params.dist,params.h,params.look,params.lookh,params.eye,params.fwd,params.pitch,params.fov,params.tau,params.near_,params.reach,drag_k,pull_pitch,pull_min,pull_near,pull_blend);}}
    return mode_;
}
// 绑定长度：认出主相机或场景块（先读客体内存里签名那几行，便宜）就返回整块长度，否则原样返回 copy
// Bound length: the whole block's length when the main camera or scene block is recognised (the signature rows are read from guest memory first, which is
// cheap), otherwise copy unchanged
size_t bound(uint64_t address,size_t copy,size_t full,bool ok){
    if(!ok || mode()<=0)return copy;
    const uint8_t *g=reinterpret_cast<const uint8_t*>(address);
    if(copy<XBG_CAM_BLOCK && full>=XBG_CAM_BLOCK && full<=4096 && xbg_cam_sig(g,272,full)){++widened;return XBG_CAM_BLOCK;}
    if(copy<XBG_CAM_SCENE && full==XBG_CAM_SCENE && xbg_cam_sig10(g,44,full)){++widened;return XBG_CAM_SCENE;}
    return copy;
}
Buffer patch(uint64_t address,size_t copy,const Buffer &src){
    if(copy<XBG_CAM_BLOCK || copy>4096)return src;
    for(auto &c:copies)if(c.epoch==snapshot_epoch && c.address==address && c.bytes==copy)return c.buffer;
    const uint8_t *p=static_cast<const uint8_t*>(src.mapped);if(!p)return src;
    const bool main=xbg_cam_sig(p,copy,copy),scene=!main && xbg_cam_sig10(p,copy,copy);if(!main && !scene)return src;
    Buffer b=stream_buffer(uniform_range(copy),p,copy);uint8_t *d=static_cast<uint8_t*>(b.mapped);
    if(main){
        // 每帧一次跟随：宿主给的朝向（瞄准拖拽 > 移动摇杆）优先，其次人物位移方向；越肩档把两套矩阵发布给输入换算，第一人称不换算输入
        // Follow once per frame: the host's heading (aim drag > move stick) first, then the character's displacement; the over-the-shoulder mode
        // publishes both matrix sets for input conversion, first person does not convert input
        float px,py,pz,wx=0,wz=0;xbg_cam_ground(p,&px,&py,&pz);const int want=dm_cam_want_dir?dm_cam_want_dir(&wx,&wz):0;
        // 原版画面往前挪（宿主 dm_cam.c DM_CAM_AHEAD）：认出这一帧原版镜头带的挪动量，从人物点里扣掉；xbg_cam_build 里同样扣（st.ahead）
        // The original picture moved ahead (host dm_cam.c DM_CAM_AHEAD): recognise the shift this frame's original camera carries and remove it from the
        // character point; xbg_cam_build removes it the same way (st.ahead)
        st.ahead[0]=st.ahead[1]=0;
        if(dm_cam_ahead_pick){float c=0,hx=0,hz=0;const int hok=dm_cam_clearance && dm_cam_clearance(-st.fx,-st.fz,&c,&hx,&hz);dm_cam_ahead_pick(px,pz,hok,hx,hz,st.fx,st.fz,&st.ahead[0],&st.ahead[1]);px-=st.ahead[0];pz-=st.ahead[1];}
        // 体感转向（宿主 dm_cam.c）：每帧取走一次累计的转动；最近一秒内转过就让跟随不动朝向，转动在跟随之后加上
        // Sensor steering (host dm_cam.c): take the accumulated turn once per frame; with a turn within the last second the follow leaves the heading alone,
        // and the turn is added after the follow
        const uint64_t frame=stats.presents.load();float turn=0;
        if(frame!=st.frame){turn=dm_cam_take_turn?dm_cam_take_turn():0;st.hold=dm_cam_turn_age && dm_cam_turn_age()<1.0;}
        // 原神式（宿主 dm_cam_style = 1，越肩档）：镜头被人物拖着走（xbg_cam_drag），不朝摇杆方向追；其余照旧
        // Genshin style (host dm_cam_style = 1, over-the-shoulder): the camera is dragged along by the character (xbg_cam_drag) and does not chase the stick
        // direction; everything else as before
        const int style=mode_==1 && dm_cam_style?dm_cam_style():0;
        // 第一人称不自动跟随，见 nvn_mtl.m cam_build | first person does not follow, see nvn_mtl.m cam_build
        const double t0=st.t;const bool fresh=style==1?xbg_cam_drag(&st,px,pz,frame,timing_clock()*1e-9,params.dist,drag_k):xbg_cam_follow(&st,px,pz,frame,timing_clock()*1e-9,want,wx,wz,mode_==2?1e9f:params.tau);
        if(fresh && turn!=0){xbg_cam_turn(&st,turn);turned+=turn;}
        // 原神式：放技能时人转过去对着目标，镜头跟着转（宿主 dm_cam_attack_dir：0.8 秒内、摇杆都没推时才有；玩家自己刚转过镜头就不动）
        // Genshin style: casting a skill turns the character to face the target and the camera turns with it (host dm_cam_attack_dir: only within 0.8 s and
        // with neither stick pushed; not while the player has just turned the camera)
        if(fresh && style==1 && !st.hold && dm_cam_attack_dir){float ax=0,az=0;if(dm_cam_attack_dir(&ax,&az))xbg_cam_ease(&st,ax,az,t0?st.t-t0:0,params.tau);}
        // 避墙（越肩档）：宿主按碰撞格量的、镜头朝后那个方向上离墙多远，每帧拉一次
        // Wall avoidance (over-the-shoulder mode): how far the wall is in the camera's backward direction, as the host measured on the collision grid, pulled once per frame
        // 原神式只理会离人近的墙（DM_CAM_PULL_NEAR，缺省 10）：镜头在人后 20、高 13，视线在人后 d 处离地约 3 + d / 2，再远的墙一般在视线下面、挡不住人。
        // near 到 near + blend 之间从 near 渐变到满距离，越过门槛时不一下跳 10 个单位（用户 2026-10-29：避墙太敏感）
        // The Genshin style only minds walls close to the character (DM_CAM_PULL_NEAR, default 10): with the camera 20 behind and 13 up, the sight line d
        // behind the character is about 3 + d / 2 above the ground, so farther walls are usually below it and do not hide the character. Between near and
        // near + blend the distance goes from near to the full distance, so crossing the threshold does not jump 10 units (the user, 2026-10-29: the wall
        // avoidance reacts too easily)
        if(fresh && mode_==1){float c=0;clear_ok=dm_cam_clearance && dm_cam_clearance(-st.fx,-st.fz,&c,&host_px,&host_pz);clear=clear_ok?c:0;
            float want=clear;if(style==1 && want>0){if(want>=pull_near+pull_blend)want=0;else if(want>pull_near)want=pull_near+(params.dist-pull_near)*(want-pull_near)/pull_blend;if(want>0 && want<pull_min)want=pull_min;}
            xbg_cam_pull(&st,params.dist,want,t0?st.t-t0:0);}
        // 诊断：人物估计（xbg_cam_ground）和宿主读到的人物位置差 3 个单位以上就记一笔（最多每秒一次）——缩放算式不对、开面板时原版镜头横移，都会在这里露出来
        // Diagnostic: log when the character estimate (xbg_cam_ground) is more than 3 units from the position the host reads (at most once a second) -- a
        // wrong zoom formula or the original camera moving sideways for an open panel shows up here
        if(fresh && clear_ok && hypotf(px-host_px,pz-host_pz)>3 && frame-drift_frame>60){drift_frame=frame;xbg_logf("NVN cam drift player=%.1f,%.1f host=%.1f,%.1f off=%.1f",px,pz,host_px,host_pz,hypotf(px-host_px,pz-host_pz));}
        // 玩家上下拖的俯仰（宿主 dm_cam.c）；算完把宿主的累计值夹回本帧机位能取的范围
        // The player's dragged pitch (host dm_cam.c); afterwards clamp the host's sum to this frame's range
        // 原神式不调俯仰（触屏层也不再送）：以前拖过的俯仰也不用，固定在缺省机位
        // The Genshin style has no pitch (the touch layer no longer sends it): an earlier dragged pitch is not used either, the default rig stays
        st.tilt=style==1?0:dm_cam_tilt_get?dm_cam_tilt_get():0;
        // 原神式：拉近时不超过 pull_pitch 度下俯、离人至少 pull_min（宁可镜头稍微进墙，也不贴在背上或变成头顶视角）
        // Genshin style: when pulled in, no steeper than pull_pitch degrees and at least pull_min from the character (better slightly inside the wall than on
        // the character's back or overhead)
        xbg_cam_params rig=params;if(style==1)rig.pull_pitch=pull_pitch;
        float vp[16],ivp[16];xbg_cam_build(&st,p,d,mode_,&rig,vp,ivp);if(fresh && dm_cam_tilt_clamp)dm_cam_tilt_clamp(st.tilt_lo,st.tilt_hi);
        if(dm_cam_publish)dm_cam_publish(reinterpret_cast<const float*>(p+336),reinterpret_cast<const float*>(p+400),vp,ivp,px,pz);
        if(dm_cam_publish_height)dm_cam_publish_height(py);
        if(patches++%600==0)xbg_logf("NVN cam patch=%llu scene=%llu widened=%llu mode=%d style=%d bytes=%zu player=%.1f,%.1f,%.1f heading=%.2f,%.2f eye=%.1f,%.1f,%.1f clear=%.1f(%d) dist=%.1f host=%.1f,%.1f turned=%.2f hold=%d ahead=%.0f,%.0f",(unsigned long long)patches,(unsigned long long)scenes,(unsigned long long)widened,mode_,style,copy,px,py,pz,st.fx,st.fz,st.eye[0],st.eye[1],st.eye[2],clear,int(clear_ok),st.dcur,host_px,host_pz,turned,st.hold,st.ahead[0],st.ahead[1]);
        if(patches==1){
            // 原版正交画面的大小（世界单位）：宽 2/pr[0]、高 2/pr[5]；原版下俯 30°，所以画面的高在地面上沿视线方向盖住两倍长。
            // 越肩镜头画得出来的东西不出这个范围：按正交视锥剔除的计算着色器和游戏自己挑单位都按它
            // Size of the original orthographic picture (world units): width 2/pr[0], height 2/pr[5]; the original looks down at 30 degrees, so on the ground the
            // picture's height covers twice that length along the sight line. What the over-the-shoulder camera can draw stays inside this area: the compute
            // shader that culls against the orthographic frustum and the game's own choice of units both use it
            float pr[16];memcpy(pr,p+208,sizeof pr);
            if(pr[0]!=0 && pr[5]!=0)xbg_logf("NVN cam original view=%.1fx%.1f ground=%.1fx%.1f (world units) axis offset=%.4f,%.4f",fabsf(2/pr[0]),fabsf(2/pr[5]),fabsf(2/pr[0]),fabsf(4/pr[5]),pr[12],pr[13]);}
    }else{xbg_cam_build10(&st,d);++scenes;}
    b.flush();Copy &c=copies[next++%copies.size()];c.address=address;c.bytes=copy;c.epoch=snapshot_epoch;c.buffer=b;return b;
}
}
// cam_ok = false：按正交视锥剔除的计算着色器，读原版
// cam_ok = false: the compute shader that culls against the orthographic frustum, which reads the original
Buffer uniform_snapshot(uint64_t address,size_t copy,bool cam_ok=true){
    if(g_ui_moff!=0 && address==g_ui_moff_addr && copy>=64){   // 界面下移：投影矩阵另拷一份加纵向偏移，不进快照缓存 | UI shift: a separate copy of the projection matrix with a vertical offset, kept out of the snapshot caches
        Buffer src=uniform_snapshot_cached(address,copy);Buffer b=stream_buffer(uniform_range(copy),src.mapped,copy);
        float m[16],o=float(g_ui_moff);memcpy(m,b.mapped,64);m[1]+=o*m[3];m[5]+=o*m[7];m[9]+=o*m[11];m[13]+=o*m[15];memcpy(b.mapped,m,64);b.flush();return b;}
    Buffer b=uniform_snapshot_cached(address,copy);
    return cam_ok && cam::mode()>0?cam::patch(address,copy,b):b;
}
Buffer uniform_snapshot_cached(uint64_t address,size_t copy){CPU_SCOPE(Snapshot);
    SnapFront &f=snap_front_for(1,address,copy);if(f.epoch==snapshot_epoch && f.address==address && f.bytes==copy){++perf.snapshot_hits;++snap_front_hits;return f.buffer;}
    Buffer b=uniform_snapshot_slow(address,copy);f.address=address;f.bytes=copy;f.epoch=snapshot_epoch;f.buffer=b;return b;
}
Buffer uniform_snapshot_slow(uint64_t address,size_t copy){
    size_t n=uniform_range(copy);auto key=std::make_pair(address,uint64_t(copy));
    auto it=uniform_snapshots.find(key);if(it!=uniform_snapshots.end()){++perf.snapshot_hits;++snap_map_hits;return it->second;}
    {Buffer view;if(pending_view(address,n,properties.limits.minUniformBufferOffsetAlignment,view)){uniform_snapshots.emplace(key,view);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,n);return view;}}
    ++perf.snapshot_misses;perf.snapshot_bytes+=copy;snap_site_bytes[5]+=copy;snap_site_req[5]+=copy;cpu_access(address,copy,2);
    Buffer b=stream_buffer(n,reinterpret_cast<const void*>(address),copy,true);uniform_snapshots.emplace(key,b);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,copy);return b;
}
// Descriptor memo (iOS "same inputs -> reuse the parameter block"): within one guest submit, a draw whose shaders and
// bound inputs (buffer ranges, texture/sampler handles, pools, driver generations, viewport) equal an earlier draw's
// reuses that draw's set and dynamic offsets. Snapshots are per-submit too, so the bound data is the same. On a hit
// every input image still gets its pending CPU upload and read barrier. Not used for compute, written storage,
// texel buffers or storage images. files/gfx-no-descmemo disables it.
// Per-submit resolved sampled-image descriptors: (texture handle, sampler handle, pools, binding kind) -> image, view,
// sampler. A hit still performs the pending CPU upload and the read barrier for the image.
struct ImgKey{uint64_t h,sh,tp,to,sp,so,kind;bool operator==(const ImgKey &o)const{return h==o.h&&sh==o.sh&&tp==o.tp&&to==o.to&&sp==o.sp&&so==o.so&&kind==o.kind;}};
struct ImgKeyHash{size_t operator()(const ImgKey &k)const{uint64_t x=k.h*0x9E3779B97F4A7C15ull^k.sh*0xC2B2AE3D27D4EB4Full^k.tp^(k.to<<17)^(k.sp<<7)^k.so^(k.kind<<40);x^=x>>29;return size_t(x*0xBF58476D1CE4E5B9ull);}};
struct ImgEntry{const tex_t *tex;Image *im;VkImageView view;VkSampler sampler;const RegisteredTexture *record;uint64_t revision;};
std::unordered_map<ImgKey,ImgEntry,ImgKeyHash> img_cache;bool img_cache_enabled=true;uint64_t img_cache_hits=0,img_cache_misses=0;
// Recreate an image created without MUTABLE_FORMAT with it, copy every level and layer, and retire the old image and its
// views (queue order keeps earlier recorded uses on the old one). Cached descriptor sets and the image cache hold the old
// views: memo_epoch and img_cache drop them. Field counts on Metal: 0 of 17473 textures needed this (nvn_mtl.m).
void upgrade_mutable(Image &im){
    VkImageCreateInfo ci=im.create;ci.flags|=VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    VkImage image{};VK(vkCreateImage(cfg.device,&ci,nullptr,&image));VkMemoryRequirements r;vkGetImageMemoryRequirements(cfg.device,image,&r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=r.size;ai.memoryTypeIndex=memory_type(r.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory memory{};VK(vkAllocateMemory(cfg.device,&ai,nullptr,&memory));VK(vkBindImageMemory(cfg.device,image,memory,0));
    Commands c;   // non-draw: ends an open pass, full barriers before and after
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;b.newLayout=VK_IMAGE_LAYOUT_GENERAL;b.image=image;b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange={im.aspects,0,im.levels,0,im.layers};b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);
    bool copied=im.layout!=VK_IMAGE_LAYOUT_UNDEFINED;   // nothing defined to copy yet
    if(copied){im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);std::vector<VkImageCopy> regions;
        for(uint32_t l=0;l<im.levels;l++){VkImageCopy rc{};rc.srcSubresource={im.aspects,l,0,im.layers};rc.dstSubresource=rc.srcSubresource;rc.extent={std::max(1u,im.width>>l),std::max(1u,im.height>>l),std::max(1u,im.depth>>l)};regions.push_back(rc);}
        vkCmdCopyImage(c.command,im.image,VK_IMAGE_LAYOUT_GENERAL,image,VK_IMAGE_LAYOUT_GENERAL,uint32_t(regions.size()),regions.data());}
    c.finish();
    VkImage old_image=im.image;VkDeviceMemory old_memory=im.memory;std::vector<VkImageView> old_views;for(auto &v:im.views)old_views.push_back(v.second);im.views.clear();
    std::vector<VkFramebuffer> old_fbs;forget_framebuffers(im,&old_fbs);
    retire_handle([old_image,old_memory,old_views,old_fbs]{for(auto f:old_fbs)vkDestroyFramebuffer(cfg.device,f,nullptr);for(auto v:old_views)vkDestroyImageView(cfg.device,v,nullptr);vkDestroyImage(cfg.device,old_image,nullptr);vkFreeMemory(cfg.device,old_memory,nullptr);});
    im.image=image;im.memory=memory;im.create=ci;im.mutable_format=true;im.layout=VK_IMAGE_LAYOUT_GENERAL;
    ++memo_epoch;img_cache.clear();++mutable_upgrades;last_pass_reset();
    xbg_logf("NVN mutable upgrade serial=%llu fmt=%d %ux%ux%u levels=%u layers=%u copied=%d total=%llu",(unsigned long long)im.source.serial,int(im.fmt),im.width,im.height,im.depth,im.levels,im.layers,int(copied),(unsigned long long)mutable_upgrades);
}
struct DescMemo{const LayoutEntry *entry;VkDescriptorSet set;VkPipelineLayout pipeline_layout;std::vector<uint32_t> dynamic_offsets;std::vector<MemoImage> images;struct Dyn{uint64_t order;VkBuffer buffer;uint32_t offset;bool uniform;uint8_t src=0,ns=0,bank=0;};std::vector<Dyn> dyn;bool persistent=false;std::vector<std::pair<uint64_t,const void*>> tex_ids;struct Stor{uint64_t base,bytes;VkBuffer buffer;VkDeviceSize offset;};std::vector<Stor> stor;
    struct Texel{const RegisteredTexture *record;uint64_t revision,base,bytes,gen;uint32_t format;};std::vector<Texel> texel;uint64_t checked_gen=0;size_t reserve=0;};   // checked_gen: memo_check_gen at its last passed image checks   // texel: resident uniform texel buffers (texel_acquire)
struct DynUbo{uint64_t order,address;size_t copy;};
// Cross-frame memo (Claude C147): the memo lives per frame slot and survives submits. An entry is persistent when its set
// holds no per-frame snapshot except dynamic uniform buffers (re-snapshotted on every hit, only the offset moves) and the
// support / driver parameter blocks (recomputed on every hit, buffer must be the same arena buffer). Entries with storage
// snapshots stay frame-local (cleared at submit, as before). Persistent sets are pinned out of the slot's set recycling;
// Texture registrations use per-record revisions (checked on every hit); sampler changes bump memo_epoch, which drops
// every slot's memo lazily and returns pinned sets to that slot (recycled only after its next fence).
// files/gfx-no-xmemo keeps the old per-submit behaviour.
std::unordered_map<std::vector<uint64_t>,DescMemo,SetKeyHash> desc_memo_slot[2];std::unordered_map<VkDescriptorSet,PinRef> memo_pinned[2];
size_t desc_memo_local[2]={};   // entries that are not persistent, which desc_memo_reset drops
std::atomic<uint64_t> memo_epoch{1};bool dyn_partial_enabled=true;uint64_t key_diff[14]={};uint64_t memo_bump[5]={};uint64_t memo_slot_epoch[2]={1,1};bool xmemo_enabled=true,memo_dyn_ok=false;uint64_t xmemo_hits=0,xmemo_drops=0;
#define desc_memo desc_memo_slot[slot_index]
// Front of the memo lookup: a draw whose key equals the key last found reuses that entry without hashing the key and
// walking its bucket. Any change that can erase an entry or rehash the map (clear, sweep, erase, insert) moves
// memo_mod_gen, which drops the front. A/B target memofront looks every key up.
std::vector<uint64_t> memo_front_key;std::unordered_map<std::vector<uint64_t>,DescMemo,SetKeyHash>::iterator memo_front_it;
uint64_t memo_mod_gen=1,memo_front_gen=0,memo_front_hits=0;unsigned memo_front_slot=~0u;
uint64_t desc_ns[6]={};bool desc_memo_enabled=true;uint64_t desc_memo_hits=0,desc_memo_misses=0,desc_memo_why[6]={};
std::vector<MemoImage> desc_memo_images;std::vector<DescMemo::Dyn> desc_memo_dyn;bool desc_memo_recording=false;
std::vector<DescMemo::Texel> desc_memo_texel;bool desc_memo_texel_fail=false;
void descriptors_build(Descriptors &out,Commands &commands,bool compute);
// Dynamic uniform buffers (Claude C151): Maleoon allows 8 per pipeline layout. Mode 1 = every uniform buffer dynamic (fits);
// mode 2 = only the per-draw ones (guest uniform buffers, driver blocks) dynamic, the support block (depends on the viewport
// only, kept in the static arena) and c1 constants (static) bound at fixed offsets; mode 0 = none dynamic.
#define UBO_DYN(mode,b) ((b).type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && ((mode)==1 || ((mode)==2 && ((b).source==XBG_SOURCE_UNIFORM||(b).source==XBG_SOURCE_DRIVER0||(b).source==XBG_SOURCE_DRIVER2))))
unsigned dyn_mode_for(bool compute,bool *no_ubo=nullptr){unsigned all=0,vary=0;
    for(unsigned st=0;st<6;st++)if(state.shader[st] && (st==5)==compute)for(auto &b:state.shader[st]->bindings)if(b.type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER){all+=b.count;if(b.source==XBG_SOURCE_UNIFORM||b.source==XBG_SOURCE_DRIVER0||b.source==XBG_SOURCE_DRIVER2)vary+=b.count;}
    if(no_ubo)*no_ubo=all==0;unsigned lim=properties.limits.maxDescriptorSetUniformBuffersDynamic;
    return all && all<=lim?1:(vary && vary<=lim && dyn_partial_enabled?2:0);}
void memo_unpin_all(unsigned si){Slot &s=slots[si];for(auto &pn:memo_pinned[si])s.used_sets[pn.second.layout].push_back(pn.first);memo_pinned[si].clear();desc_memo_slot[si].clear();desc_memo_local[si]=0;++memo_mod_gen;}
void memo_check_epoch(){unsigned si=slot_index;if(memo_slot_epoch[si]!=memo_epoch || memo_pinned[si].size()>8192){if(!desc_memo_slot[si].empty())++xmemo_drops;memo_unpin_all(si);memo_slot_epoch[si]=memo_epoch;}}
// End of a submit for the recording slot: frame-local entries go (their snapshots are per submit); persistent ones stay.
void desc_memo_reset(){img_cache.clear();
    if(!xmemo_enabled){memo_unpin_all(slot_index);return;}
    // Sweep only when the slot holds frame-local entries: walking its ~600-800 persistent entries every submission to find
    // none was ~1% of the D2R camp's render thread on the Redmi K50 Ultra.
    if(!desc_memo_local[slot_index])return;
    auto &m=desc_memo_slot[slot_index];for(auto it=m.begin();it!=m.end();){if(it->second.persistent)++it;else it=m.erase(it);}desc_memo_local[slot_index]=0;++memo_mod_gen;}
Buffer &support_block(){size_t n=1392;auto k=std::make_pair(state.viewport.width,state.viewport.height);
    auto it=support_static.find(k);if(it!=support_static.end())return it->second;
    Buffer s=static_arena.alloc(uniform_range(n));xbg_support_defaults(s.mapped,state.viewport.width,state.viewport.height);s.flush();return support_static.emplace(k,s).first->second;}
// np606：**声明了但客体从没绑过的 UBO 槽**用这块零缓冲顶上（按大小缓存，内容全 0，可共享）。
// 为什么可以这样做：着色器包里"有这条 binding"只表示程序里声明了这个常量块，NVN 并不要求游戏把每个
// 声明过的槽都绑上；真机上读一个没绑的槽是未定义值。参考实现（Ryujinx）同样塞零缓冲。
// 判据（1.6.0 真机）：`NVN bindhits stage=0 … 2:0 …` —— 同一阶段 slot 0 绑了 91,386 次、slot 10 有 158,696 次，
// 而 slot 2 一次都没有 ⇒ 不是"我们槽号算错"，是"游戏根本不用它"。原来这里直接 FATAL，等于用一个
// 声明细节把整局拦下来。
// np606: a zero-filled stand-in for UBO slots the guest declared but never bound (cached by size, all zeros, so
// shareable). A binding in the shader bundle only means the program declares that constant block; NVN never
// requires the game to bind every declared slot, and reading an unbound slot on hardware is undefined. The
// reference implementation (Ryujinx) fills a zero buffer as well. Evidence (1.6.0 device): `NVN bindhits stage=0
// ... 2:0 ...` -- slot 0 was bound 91,386 times and slot 10 158,696 times, slot 2 never, so this is "the game
// simply does not use it", not a wrong slot number. FATALing here blocked the whole run over a declaration detail.
Buffer &zero_block(size_t bytes){auto &cache=zero_blocks;size_t n=uniform_range(bytes);
    auto it=cache.find(n);if(it!=cache.end())return it->second;
    Buffer s=static_arena.alloc(n);memset(s.mapped,0,n);s.flush();return cache.emplace(n,s).first->second;}
Buffer &driver_block(unsigned ns,unsigned bank){uint64_t gen=bank?tex_gen[ns]:buf_gen[ns];
    DriverCache &dc=driver_cache[ns][bank];if(dc.slot!=slot_serial || dc.gen!=gen || !dc.slot){Buffer s=stream_buffer(0x1000);fill_driver(static_cast<uint8_t*>(s.mapped),ns,bank?2:0);s.flush();dc.gen=gen;dc.slot=slot_serial;dc.buffer=s;}
    return dc.buffer;}
// Uniform texel buffers held in resident per-slot copies, so descriptor sets that bind them can be memoized across frames
// (D2R camp: 558 of 2188 draws per frame bind one and were rebuilt every time, ~21 us each against ~2.5 us for a memo hit).
// A copy belongs to one slot (refreshed only while that slot records, after its previous submission retired) and is
// refreshed from guest memory when its content hash changes, before its first read in a frame; content that changes again
// within the frame falls back to the per-frame snapshot. A memo entry checks the copy's generation, so an evicted and
// recreated copy is never mistaken for the one its set holds. gfx-no-texel-memo keeps the former per-frame path.
TexelResident *texel_acquire(uint64_t base,uint64_t bytes,VkFormat fmt){
    if(!bytes || base>UINT64_MAX-bytes || bytes>64ull*1024*1024){++texel_fallbacks;return nullptr;}
    // 待写回的数据走原快照路径，不允许驻留命中绕过 GPU 写入或跨槽排空。
    // Pending results use the original snapshot path; resident hits must not bypass GPU writes or drain across slots.
    // 任何重叠都算：没有 read_preflight 时，这里可能还有只覆盖一部分范围的待写回。
    // Any overlap counts: without read_preflight a pending writeback can cover only part of the range here.
    for(auto &w:pending_writebacks)if(xbg_wb_overlaps(base,bytes,w.addr,w.bytes)){++texel_fallbacks;return nullptr;}
    for(auto &r:pending_reports)if(base<r.addr+16 && base+bytes>r.addr){++texel_fallbacks;return nullptr;}
    for(auto *list:{&pending_events,&recording_events})for(auto &e:*list)if(base<e.addr+4 && base+bytes>e.addr){++texel_fallbacks;return nullptr;}
    unsigned si=slot_index;need(slots[si].recording && !slots[si].pending,"Texel resident requires a retired recording slot");auto &map=texel_resident[si];TexelKey key{base,bytes,uint32_t(fmt)};
    auto it=map.find(key);
    if(it!=map.end() && it->second.checked==slot_serial && it->second.check_epoch==texel_write_epoch){it->second.used=slot_serial;return &it->second;}
    uint64_t h=hash_memory(reinterpret_cast<const void*>(uintptr_t(base)),size_t(bytes));
    if(it==map.end()){
        if(map.size()>=8192 || texel_live_bytes[si]+texel_hole_bytes[si]+bytes>64ull*1024*1024){++texel_fallbacks;return nullptr;}
        TexelResident r;r.buffer=texel_arena[si].alloc(size_t(bytes));
        VkBufferViewCreateInfo v{VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};v.buffer=r.buffer.buffer;v.offset=r.buffer.offset;v.format=fmt;v.range=bytes;VK(vkCreateBufferView(cfg.device,&v,nullptr,&r.view));
        memcpy(r.buffer.mapped,reinterpret_cast<const void*>(uintptr_t(base)),size_t(bytes));r.buffer.flush();r.hash=h;r.used=slot_serial;r.gen=++texel_gen;++texel_refreshes;
        r.checked=slot_serial;r.check_epoch=texel_write_epoch;texel_lo=std::min(texel_lo,base);texel_hi=std::max(texel_hi,base+bytes);
        texel_live_bytes[si]+=bytes;return &map.emplace(key,r).first->second;}
    TexelResident &r=it->second;
    if(r.hash!=h){if(r.used==slot_serial){++texel_fallbacks;return nullptr;}   // 本帧已读取旧内容 | This frame already reads earlier contents
        memcpy(r.buffer.mapped,reinterpret_cast<const void*>(uintptr_t(base)),size_t(bytes));r.buffer.flush();r.hash=h;++texel_refreshes;}
    r.used=slot_serial;r.checked=slot_serial;r.check_epoch=texel_write_epoch;return &r;
}
void texel_touch(uint64_t address,uint64_t bytes){
    auto &map=texel_resident[slot_index];if(map.size()>64){++texel_write_epoch;return;}   // many copies: invalidate coarsely
    for(auto &e:map)if(address<e.first.base+e.first.bytes && address+bytes>e.first.base){++texel_write_epoch;return;}
}
// While a slot starts recording (its previous submission retired): drop copies unused for ~128 submissions; when the holes they
// leave in the slot's arena pass 8 MB, start the slot's store over (memo entries then fail their generation check).
void texel_evict(unsigned si,bool all){
    need(!slots[si].pending && (!slots[si].recording || !slots[si].ops),"Evict texel residents only after slot retirement");
    auto &map=texel_resident[si];
    for(auto it=map.begin();it!=map.end();){if(all || it->second.used+128<slot_serial){vkDestroyBufferView(cfg.device,it->second.view,nullptr);texel_hole_bytes[si]+=it->first.bytes;texel_live_bytes[si]-=it->first.bytes;it=map.erase(it);++texel_evictions;}else ++it;}
    if(all || texel_hole_bytes[si]>8ull*1024*1024){for(auto &e:map)vkDestroyBufferView(cfg.device,e.second.view,nullptr);map.clear();texel_arena[si].release();texel_hole_bytes[si]=texel_live_bytes[si]=0;}
}
void descriptors(Descriptors &out,Commands &commands,bool compute){
    need(!read_preflight || (pending_writebacks.empty() && pending_reports.empty()),"Unresolved CPU reads at descriptor entry");
    uint64_t tm0=perf_clock();struct MemoClock{uint64_t t0;~MemoClock(){}} mc{tm0};
    static std::vector<uint64_t> key;key.clear();static std::vector<uint8_t> ktag;ktag.clear();
    // ktag (which part of the key each word is) only feeds the keydiff census, so it is filled on timed frames only (one in
    // perf_sample); a miss without tags skips the census. A/B target ktag tags every frame again.
    const bool tags=timing_now()!=nullptr || ab_off("ktag");
    bool ok=desc_memo_enabled && !compute && !capture.active && !provenance_file && g_ui_moff==0;
    if(ok){CPU_SCOPE(Descriptor);
        static std::vector<DynUbo> dyn_ubos;dyn_ubos.clear();
        bool no_ubo=false;const unsigned dmode=dyn_mode_for(false,&no_ubo);memo_dyn_ok=dmode!=0 || no_ubo;
        for(unsigned st=0;st<5;st++){key.push_back(uint64_t(uintptr_t(state.shader[st])));if(tags)ktag.push_back(0);}
        key.push_back(state.texture_pool.pool?uint64_t(uintptr_t(state.texture_pool.pool->mem)):0);key.push_back(uint64_t(state.texture_pool.off));key.push_back(uint64_t(state.texture_pool.n));if(tags)ktag.insert(ktag.end(),3,1);
        key.push_back(state.sampler_pool.pool?uint64_t(uintptr_t(state.sampler_pool.pool->mem)):0);key.push_back(uint64_t(state.sampler_pool.off));if(tags)ktag.insert(ktag.end(),2,1);
        for(unsigned st=0;st<5 && ok;st++)if(state.shader[st]){Shader &shader=*state.shader[st];unsigned ns=nvn_stage[st];
            for(auto &b:shader.bindings){
                // Texel view 属于 slot，退休时销毁；memo 不追踪其快照或生命周期，不能保留旧 set。
                // Slot retirement destroys texel views; memo tracks neither their snapshots nor lifetime, so exclude both kinds.
                if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER || (b.type==VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER && !texel_memo_enabled)){ok=false;++desc_memo_why[0];break;}
                if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE){ok=false;++desc_memo_why[1];break;}
                if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && shader.storage_writes.count(b.binding)){ok=false;++desc_memo_why[2];break;}
                for(unsigned element=0;element<b.count;element++){unsigned slot=b.slot+element;
                    switch(b.source){
                    case XBG_SOURCE_UNIFORM:case XBG_SOURCE_STORAGE:{if(slot>=16){ok=false;break;}auto r=b.source==XBG_SOURCE_UNIFORM?state.ub[ns][slot]:state.sb[ns][slot];
                        if(UBO_DYN(dmode,b) && b.source==XBG_SOURCE_UNIFORM && r.address && r.size){size_t copy=cam::bound(r.address,std::min<size_t>(r.size,b.bytes?b.bytes:65536),r.size,true);key.push_back(copy);if(tags)ktag.push_back(2);dyn_ubos.push_back({(uint64_t(b.binding)<<32)|element,r.address,copy});} // address only moves the dynamic offset
                        else{key.push_back(r.address);key.push_back(r.size);if(tags){ktag.push_back(b.source==XBG_SOURCE_UNIFORM?3:10);ktag.push_back(4);}}break;}
                    case XBG_SOURCE_SUPPORT:key.push_back((uint64_t(uint32_t(state.viewport.width))<<32)|uint32_t(state.viewport.height));key.push_back(state.color.alpha_func);if(tags){ktag.push_back(5);ktag.push_back(6);}break;
                    case XBG_SOURCE_C1:break;
                    case XBG_SOURCE_DRIVER0:case XBG_SOURCE_DRIVER2:key.push_back(UBO_DYN(dmode,b)?uint64_t(0xD0D0+b.source):(b.source==XBG_SOURCE_DRIVER0?tex_gen[ns]:buf_gen[ns]));if(tags)ktag.push_back(7);break; // dynamic: block rebuilt from the current state on every hit (driver_block); otherwise its offset is baked into the set
                    case XBG_SOURCE_TEXTURE:case XBG_SOURCE_IMAGE:{if(slot>=(b.source==XBG_SOURCE_IMAGE?16u:128u)){ok=false;break;}key.push_back(b.source==XBG_SOURCE_IMAGE?state.img[ns][slot]:state.tex[ns][slot]);if(tags)ktag.push_back(8);
                        if(b.sampler_slot!=UINT32_MAX){if(b.sampler_slot>=128){ok=false;break;}key.push_back(state.samp[ns][b.sampler_slot]);if(tags)ktag.push_back(9);}break;}
                    default:ok=false;}
                    if(!ok)break;
                }
                if(!ok)break;
            }}
        if(ok){memo_check_epoch();auto hit=desc_memo.end();
            if(memo_front_gen==memo_mod_gen && memo_front_slot==slot_index && !ab_off("memofront") && memo_front_key==key){hit=memo_front_it;++memo_front_hits;}
            else{hit=desc_memo.find(key);if(hit!=desc_memo.end()){memo_front_key=key;memo_front_it=hit;memo_front_gen=memo_mod_gen;memo_front_slot=slot_index;}}
            if(hit==desc_memo.end() && ktag.size()==key.size()){   // diagnostic: which key part differs from the last key of the same shaders
                static std::unordered_map<std::vector<uint64_t>,std::vector<uint64_t>,SetKeyHash> last;std::vector<uint64_t> pre(key.begin(),key.begin()+5);
                auto &prev=last[pre];if(prev.size()==key.size()){bool any=false;for(size_t i=0;i<key.size();i++)if(prev[i]!=key[i]){++key_diff[ktag[i]<12?ktag[i]:11];any=true;}if(!any)++key_diff[11];}else ++key_diff[prev.empty()?12:13];
                prev=key;}
            if(hit!=desc_memo.end()){DescMemo &m=hit->second;
                static std::vector<std::pair<uint64_t,Buffer>> fresh;fresh.clear();
                for(auto &u:dyn_ubos)fresh.push_back({u.order,uniform_snapshot(u.address,u.copy)});
                std::sort(fresh.begin(),fresh.end(),[](const std::pair<uint64_t,Buffer> &a,const std::pair<uint64_t,Buffer> &b){return a.first<b.first;});
                static std::vector<uint32_t> offsets;offsets.clear();bool same=true;size_t fi=0;
                // Room for keep() below (support / driver blocks) and draw()'s vertex/index buffers. The count of the shaders' bindings
                // is stored in the entry when it is made: the shaders are part of the key. A/B target descreserve counts them again.
                if(ab_off("descreserve")){size_t elements=0;for(unsigned st=0;st<5;st++)if(state.shader[st])for(auto &b:state.shader[st]->bindings)elements+=b.count;out.buffers.reserve(elements*2+40);}
                else out.buffers.reserve(m.reserve);
                for(auto &d:m.dyn){if(!d.uniform){
                        if(d.src==XBG_SOURCE_SUPPORT || d.src==XBG_SOURCE_DRIVER0 || d.src==XBG_SOURCE_DRIVER2){
                            Buffer &blk=d.src==XBG_SOURCE_SUPPORT?support_block():driver_block(d.ns,d.bank);if(blk.buffer!=d.buffer){same=false;break;}out.keep(blk);offsets.push_back(uint32_t(blk.offset));}
                        else offsets.push_back(d.offset);continue;}
                    if(fi>=fresh.size() || fresh[fi].first!=d.order || fresh[fi].second.buffer!=d.buffer){same=false;break;}offsets.push_back(uint32_t(fresh[fi].second.offset));++fi;}
                same=same && fi==fresh.size();
                // 每个缓存项均核注册版本（包括 view）和存活图像；仅重建失效项。
                // Every cached entry checks registration revision (including view) and live image identity.
                // Repeated only when memo_check_gen moved since this entry's last pass: re-reading every record was ~12% of
                // descriptors() in the D2R camp (a scattered load per image per draw), and in play no registration changes.
                // A/B target hitgen repeats them on every hit, as before.
                if(same && (m.checked_gen!=memo_check_gen || ab_off("hitgen"))){
                    for(size_t k=0;k<m.images.size();k++){const auto &cached=m.images[k];const tex_t *t=cached.first;
                        if(cached.record->revision!=cached.revision){same=false;break;}
                        if(t->serial!=m.tex_ids[k].first || t->mem!=m.tex_ids[k].second){same=false;break;}
                        if(find_image(t->serial)!=m.images[k].second){same=false;break;}}
                    if(same)m.checked_gen=memo_check_gen;}
                // Resident texel copies: same registration and range, the same copy the set holds (generation), content
                // refreshed if the guest changed it since this slot last recorded.
                if(same)for(auto &tr:m.texel){const tex_t &tt=tr.record->texture;
                    if(tr.record->revision!=tr.revision || uint64_t(uintptr_t(tt.mem))!=tr.base || uint64_t(tt.L.size)!=tr.bytes){same=false;break;}
                    TexelResident *r=texel_acquire(tr.base,tr.bytes,VkFormat(tr.format));if(!r || r->gen!=tr.gen){same=false;break;}++texel_hits;}
                // Read-only storage ranges: refresh the snapshot (write-tracked resident copies only copy dirty pages); the set
                // stays valid only if the range lands in the same buffer at the same offset as when the set was written.
                if(same && m.persistent)for(auto &st:m.stor){snap_site=3;Buffer b=snapshot(st.base,st.bytes,properties.limits.minStorageBufferOffsetAlignment);snap_site=0;
                    if(b.buffer!=st.buffer || b.offset!=st.offset){same=false;break;}}
                if(same){
                // 命中也走既有按 submit 节流的 CPU 内容复核，不强制重传。
                // Hits retain get_image's bounded per-submit CPU checks, without forcing uploads.
                for(auto &ti:m.images){Image &im=get_image(ti.first,true);prepare_read(im,commands.command);}
                out.entry=m.entry;out.set=m.set;out.pipeline_layout=m.pipeline_layout;out.dynamic_offsets=offsets;++desc_memo_hits;if(m.persistent)++xmemo_hits;desc_ns[5]+=perf_clock()-tm0;return;}
                ++desc_memo_why[3];
                if(m.persistent){auto pin=memo_pinned[slot_index].find(m.set);if(pin!=memo_pinned[slot_index].end() && --pin->second.refs==0){slots[slot_index].used_sets[pin->second.layout].push_back(m.set);memo_pinned[slot_index].erase(pin);}}
                if(!m.persistent && desc_memo_local[slot_index])--desc_memo_local[slot_index];desc_memo.erase(hit);++memo_mod_gen;}}
    }
    desc_ns[0]+=perf_clock()-tm0;
    desc_memo_images.clear();desc_memo_dyn.clear();desc_memo_texel.clear();desc_memo_texel_fail=false;desc_memo_recording=ok;
    {uint64_t tb=perf_clock();descriptors_build(out,commands,compute);desc_ns[4]+=perf_clock()-tb;}
    desc_memo_recording=false;
    if(ok && desc_memo_texel_fail){ok=false;++desc_memo_why[0];}
    if(ok && !(out.written.empty() && out.buffer_views.empty() && out.set && out.dynamic_offsets.size()==desc_memo_dyn.size()))++desc_memo_why[4];
    if(ok && out.written.empty() && out.buffer_views.empty() && out.set && out.dynamic_offsets.size()==desc_memo_dyn.size()){++desc_memo_misses;
        std::sort(desc_memo_dyn.begin(),desc_memo_dyn.end(),[](const DescMemo::Dyn &a,const DescMemo::Dyn &b){return a.order<b.order;});
        DescMemo dm{out.entry,out.set,out.pipeline_layout,out.dynamic_offsets,desc_memo_images,desc_memo_dyn};dm.texel=desc_memo_texel;
        {size_t elements=0;for(unsigned st=0;st<5;st++)if(state.shader[st])for(auto &b:state.shader[st]->bindings)elements+=b.count;dm.reserve=elements*2+40;}
        dm.persistent=xmemo_enabled && memo_dyn_ok && out.written.empty();
        for(auto &st:out.storage)dm.stor.push_back({st.first.first,st.first.second,st.second->buffer,st.second->offset});
        for(auto &ti:dm.images)dm.tex_ids.push_back({ti.first->serial,ti.first->mem});
        for(auto &d:dm.dyn)if(!d.uniform && d.src!=XBG_SOURCE_SUPPORT && d.src!=XBG_SOURCE_DRIVER0 && d.src!=XBG_SOURCE_DRIVER2 && d.src!=XBG_SOURCE_C1)dm.persistent=false;
        if(dm.persistent){unsigned si=slot_index;VkDescriptorSetLayout lay=out.entry->layout;
            auto pin=memo_pinned[si].find(out.set);if(pin!=memo_pinned[si].end())++pin->second.refs;
            else{auto &u=slots[si].used_sets[lay];for(size_t i=0;i<u.size();i++)if(u[i]==out.set){u[i]=u.back();u.pop_back();break;}memo_pinned[si].emplace(out.set,PinRef{lay,1});}}
        bool local=!dm.persistent;if(desc_memo.emplace(key,std::move(dm)).second && local)++desc_memo_local[slot_index];++memo_mod_gen;}
}
void descriptors_build(Descriptors &out,Commands &commands,bool compute){CPU_SCOPE(Descriptor);DescriptorRoot descriptor_measurement(&cpu_scope);
    // Uniform buffers use dynamic offsets (when the set fits the device limit) so per-draw constants do not change set contents.
    const unsigned dmode=dyn_mode_for(compute);
    // Aliased NVN storage ranges must refer to the same VkBuffer allocation.
    // Snapshotting each binding separately loses writes observed through aliases.
    {DESC_SCOPE(Setup);
    struct Range{uint64_t begin,end;bool written;};static std::vector<Range> ranges,merged;static std::vector<uint32_t> key;ranges.clear();merged.clear();key.clear();
    {size_t elements=0;for(unsigned st=0;st<6;st++)if(state.shader[st] && (st==5)==compute)for(auto &b:state.shader[st]->bindings)elements+=b.count;out.buffers.reserve(elements*2+40);}
    std::array<const void*,7> shader_key{};for(unsigned st=0;st<6;st++)if((st==5)==compute)shader_key[st]=state.shader[st];shader_key[6]=reinterpret_cast<const void*>(uintptr_t(compute));
    for(unsigned st=0;st<6;st++)if(state.shader[st] && (st==5)==compute)for(auto &b:state.shader[st]->bindings){
        need(b.count && b.count<=128 && b.type<=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,"Unsupported descriptor array/type");
        key.push_back(b.binding);key.push_back(UBO_DYN(dmode,b)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:b.type);key.push_back(b.count);key.push_back(1u<<st);
        if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && (b.source==XBG_SOURCE_STORAGE || b.source==XBG_SOURCE_UNIFORM)){bool written=state.shader[st]->storage_writes.count(b.binding)!=0;for(unsigned i=0;i<b.count;i++){
            need(b.slot+i<16,"Storage descriptor slot out of bounds");auto r=b.source==XBG_SOURCE_STORAGE?state.sb[nvn_stage[st]][b.slot+i]:state.ub[nvn_stage[st]][b.slot+i];
            need(r.address && r.size && r.size<=properties.limits.maxStorageBufferRange && r.address+r.size>=r.address,"Invalid storage descriptor address/range");ranges.push_back({r.address,r.address+r.size,written});
        }}
    }
    std::sort(ranges.begin(),ranges.end(),[](const Range &a,const Range &b){return a.begin<b.begin;});
    for(auto &r:ranges){if(!merged.empty() && r.begin<merged.back().end){merged.back().end=std::max(merged.back().end,r.end);merged.back().written=merged.back().written||r.written;}else merged.push_back(r);}
    for(auto &r:merged){uint64_t base=r.begin,bytes=r.end-base;
        if(r.written){Buffer view,*b;
            // np640：**这一处已回退**。去掉 end_open_pass() 让拷贝留在 pass 内，真机读数反而更差
            //   （GPU 峰值 1.6 s → 3.1 s、pass 数 421 → 495）：tile 型 GPU 上 pass 中途做传输要 flush/restore
            //   tile，比"断 pass 再开"更贵。真正把 pass 关掉的是 readback_buffer() 里那句 end_open_pass()
            //   （`xbg_game.cpp:4566`），删掉这里那句不解决问题。
            // np640: reverted. Dropping end_open_pass() to keep the copy inside the pass made the device reading
            //   worse (GPU peak 1.6 s -> 3.1 s, passes 421 -> 495): on a tiled GPU a mid-pass transfer flushes and
            //   restores the tile, which costs more than ending and reopening the pass. The pass is actually closed
            //   by the end_open_pass() inside readback_buffer() (`xbg_game.cpp:4566`), so removing this one changed
            //   little.
            if(pending_view(base,bytes,1,view)){b=out.keep(readback_buffer(bytes));VkCommandBuffer cmd=recording_slot().cmd;end_pass(6);VkBufferCopy region{view.offset,b->offset,bytes};vkCmdCopyBuffer(cmd,view.buffer,b->buffer,1,&region);g_barrier_site="storage-view-readback";full_barrier(cmd);}
            else{cpu_access(base,bytes,3);b=out.keep(readback_buffer(bytes,reinterpret_cast<void*>(base),bytes));}out.storage.push_back({{base,bytes},b});out.written.push_back({{base,bytes},b});}
        else{snap_site=3;out.storage.push_back({{base,bytes},out.keep(snapshot(base,bytes,properties.limits.minStorageBufferOffsetAlignment))});snap_site=0;}
    }
    LayoutEntry *entry=nullptr;{auto quick=shader_layouts.find(shader_key);if(quick!=shader_layouts.end())entry=quick->second;}
    if(!entry){auto found=layout_cache.find(key);
    if(found==layout_cache.end()){
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for(size_t i=0;i<key.size();i+=4){VkDescriptorSetLayoutBinding l{};l.binding=key[i];l.descriptorType=VkDescriptorType(key[i+1]);l.descriptorCount=key[i+2];l.stageFlags=key[i+3];bindings.push_back(l);}
        LayoutEntry e;VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=bindings.size();li.pBindings=bindings.data();VK(vkCreateDescriptorSetLayout(cfg.device,&li,nullptr,&e.layout));
        VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pi.setLayoutCount=1;pi.pSetLayouts=&e.layout;VK(vkCreatePipelineLayout(cfg.device,&pi,nullptr,&e.pipeline_layout));
        {XbgPipelineKey k;e.definition.assign(k,xbg_pipeline_layout_key(k,li,pi));}
        found=layout_cache.emplace(key,e).first;
    }
    entry=&found->second;shader_layouts.emplace(shader_key,entry);}
    out.entry=entry;out.pipeline_layout=entry->pipeline_layout;
    }
    static std::vector<VkDescriptorBufferInfo> buffer_infos;static std::vector<VkDescriptorImageInfo> image_infos;static std::vector<VkBufferView> texel_views;static std::vector<VkWriteDescriptorSet> writes;static std::vector<std::pair<uint64_t,uint32_t>> dynamic_list;static std::vector<uint64_t> set_key;
    {size_t elements=0;for(unsigned st=0;st<6;st++)if(state.shader[st] && (st==5)==compute)for(auto &b:state.shader[st]->bindings)elements+=b.count;
     buffer_infos.clear();image_infos.clear();texel_views.clear();writes.clear();dynamic_list.clear();buffer_infos.reserve(elements);image_infos.reserve(elements);texel_views.reserve(elements);writes.reserve(elements);}
    for(unsigned st=0;st<6;st++)if(state.shader[st] && (st==5)==compute){Shader &shader=*state.shader[st];unsigned ns=nvn_stage[st];
        for(auto &b:shader.bindings)for(unsigned element=0;element<b.count;element++){DESC_SCOPE(ElementRest);
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};write.dstSet=VK_NULL_HANDLE;write.dstBinding=b.binding;write.dstArrayElement=element;write.descriptorCount=1;write.descriptorType=UBO_DYN(dmode,b)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:VkDescriptorType(b.type);
            if(b.type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER){DESC_SCOPE(BufferBind);
                Buffer *buffer=nullptr;bool is_storage=b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;uint32_t slot=b.slot+element;size_t n=0,buffer_offset=0;const void *data=nullptr;size_t copy=0;
                if(b.source==XBG_SOURCE_UNIFORM || b.source==XBG_SOURCE_STORAGE){need(slot<16,"Buffer slot out of bounds");auto r=b.source==XBG_SOURCE_UNIFORM?state.ub[ns][slot]:state.sb[ns][slot];
                    // np603：**"游戏没绑"与"我们槽位映射错了"的分水岭**。原来这条 FATAL 只说"某个槽是空的"，
                    // 分不清两种完全不同的病：① 客体这一帧真的没绑（上游被我们跳过/丢帧）；
                    // ② 客体绑了，但我们从着色器里推出的 slot 号不对（那就要去查 binding.slot 的来源）。
                    // 所以这里把同一阶段的 16 个槽（UB 与 SB 两个银行都打）连同 shader key 一起打出来：
                    //   全空 ⇒ ①；有别的槽有值 ⇒ ②（把那个号跟 binding 对一下就知道差在哪）。
                    // np603: the dividing line between "the game never bound it" and "our slot mapping is wrong".
                    // The old message only said "a slot is empty", which cannot tell ① the guest really bound nothing
                    // this frame (something upstream was skipped) from ② the guest did bind, but the slot number we
                    // derived from the shader is wrong (then binding.slot's origin is the suspect). So dump all 16
                    // slots of both banks for this stage plus the shader key: all empty => ①; any other slot with
                    // data => ② (compare that index against binding.slot to see the offset).
                    if(!(r.address && r.size)){
                        // np606：不是错误，是"这条 binding 只是声明"。用零缓冲顶上，并按 (shader,binding) 报一次，
                        // 让"这个槽整轮没绑过"在日志里留痕（既不是静默，也不是把整局拦下来）。
                        // np606: not an error, just a declaration. Fill a zero buffer and log once per (shader, binding)
                        // so "this slot was never bound at all" stays visible -- neither silent nor fatal.
                        static std::set<std::pair<std::string,unsigned>> noted;
                        if(noted.insert({shader.key,b.binding}).second)
                            xbg_logf("NVN unbound UBO 用零缓冲：stage=%u shader=%s binding=%u slot=%u bindhits=%llu declared=%u",
                                st,shader.key.c_str(),b.binding,slot,(unsigned long long)bind_hits[ns][slot],b.bytes);
                        size_t want=uniform_range(std::min<size_t>(b.bytes?b.bytes:256,properties.limits.maxUniformBufferRange));
                        Buffer &zb=zero_block(want);data=zb.mapped;copy=want;n=zb.size;buffer=out.keep(zb);
                    }else{
                        data=reinterpret_cast<void*>(r.address);copy=r.size;
                    if(is_storage){n=copy;for(auto &v:out.storage)if(r.address>=v.first.first && r.address+r.size<=v.first.first+v.first.second){buffer=v.second;buffer_offset=r.address-v.first.first;break;}need(buffer,"Storage alias allocation not found");need(buffer_offset%properties.limits.minStorageBufferOffsetAlignment==0,"NVN storage offset violates Vulkan alignment");}
                    else{
                        // 镜头实验：除了按正交视锥剔除的那条计算着色器，任何阶段、任何槽位的主相机都换成透视拷贝（cam::bound / cam::patch）
                        // Camera experiment: the main camera in any stage and any slot becomes the perspective copy (cam::bound / cam::patch), except for the
                        // compute shader that culls against the orthographic frustum
                        const bool cam_ok=!(compute && shader.key.compare(0,16,cam::cull_cs)==0);
                        copy=cam::bound(r.address,std::min<size_t>(copy,b.bytes?b.bytes:65536),r.size,cam_ok);n=uniform_range(copy);buffer=out.keep(uniform_snapshot(r.address,copy,cam_ok));}
                    }
                }else if(b.source==XBG_SOURCE_SUPPORT){n=1392;need(state.color.alpha_func==8,"Nontrivial NVN alpha test not translated");
                    Buffer &sb=support_block();data=sb.mapped;copy=n;buffer=out.keep(sb);}
                else if(b.source==XBG_SOURCE_C1){
                    if(!shader.c1_ready){size_t bytes=uniform_range(shader.c1.size());shader.c1_buffer=static_arena.alloc(bytes);memset(shader.c1_buffer.mapped,0,bytes);if(!shader.c1.empty())memcpy(shader.c1_buffer.mapped,shader.c1.data(),shader.c1.size());shader.c1_buffer.flush();shader.c1_ready=true;}
                    data=shader.c1.data();copy=shader.c1.size();n=shader.c1_buffer.size;buffer=&shader.c1_buffer;}
                else if(b.source==XBG_SOURCE_DRIVER0 || b.source==XBG_SOURCE_DRIVER2){n=0x1000;unsigned bank=b.source==XBG_SOURCE_DRIVER0?0:1;uint64_t gen=bank?tex_gen[ns]:buf_gen[ns]+(ns==5?0:0);
                    DriverCache &dc=driver_cache[ns][bank];if(dc.slot!=slot_serial || dc.gen!=gen || !dc.slot){Buffer s=stream_buffer(n);fill_driver(static_cast<uint8_t*>(s.mapped),ns,bank?2:0);s.flush();dc.gen=gen;dc.slot=slot_serial;dc.buffer=s;}
                    data=dc.buffer.mapped;copy=n;buffer=out.keep(dc.buffer);}
                else xbg_fatalf("Unsupported shader buffer source %u",b.source);
                need(n && n<=size_t(is_storage?properties.limits.maxStorageBufferRange:properties.limits.maxUniformBufferRange),"Descriptor range exceeds device limit");
                {BufRef &r=buf_slot_refs[buffer->buffer];const void *cur=&recording_slot();
                 for(const void *old_slot:r.slots)if(old_slot!=cur){
                     bool alive=false;for(const Slot &sl:slots)if(&sl==old_slot){alive=sl.pending||sl.recording;break;}
                     if(!alive){static int n;if(n++<8)xbg_logf("【红】跨槽引用悬空 VkBuffer=%p 登记槽=%p 当前槽=%p serial=%llu（提前释放的形态）",(void*)buffer->buffer,old_slot,cur,(unsigned long long)slot_serial);}
                 }
                 if(r.slots.empty()||r.slots.back()!=cur){r.slots.push_back(cur);if(r.slots.size()>4)r.slots.erase(r.slots.begin());}}
                if(write.descriptorType==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC){buffer_infos.push_back({buffer->buffer,0,n});if(desc_memo_recording)desc_memo_dyn.push_back({(uint64_t(b.binding)<<32)|element,buffer->buffer,uint32_t(buffer->offset+buffer_offset),b.source==XBG_SOURCE_UNIFORM,uint8_t(b.source),uint8_t(ns),uint8_t(b.source==XBG_SOURCE_DRIVER0?0:1)});dynamic_list.push_back({(uint64_t(b.binding)<<32)|element,uint32_t(buffer->offset+buffer_offset)});}
                else buffer_infos.push_back({buffer->buffer,buffer->offset+buffer_offset,n});write.pBufferInfo=&buffer_infos.back();
                if(capture.active){char label[100];snprintf(label,sizeof(label),"st%u-bind%u-el%u-buffer",st,b.binding,element);capture_log("BUFFER label=%s shader=%s nvnStage=%u sourceKind=%u slot=%u guest=%p copied=%zu descriptorBytes=%zu bufferOffset=%zu",label,shader.key.c_str(),ns,b.source,slot,data,copy,n,buffer_offset);capture_data(label,static_cast<const uint8_t*>(buffer->mapped)+buffer_offset,n);}
            }else{
                DESC_SCOPE(ImageLookup);
                bool img_fast=false,img_cacheable=false;ImgKey ikey{};
                if(img_cache_enabled && !capture.active && !provenance_file && b.source==XBG_SOURCE_TEXTURE && (b.type==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || b.type==VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) && b.slot+element<128 && state.texture_pool.pool
                   && (b.sampler_slot==UINT32_MAX || b.sampler_slot<128)){
                    uint64_t handle=state.tex[ns][b.slot+element];int id=int(handle&0xfffff);
                    if(id!=0 && !(state.texture_pool.n>0 && id>=state.texture_pool.n)){
                        uint64_t sh=b.type==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER?(b.sampler_slot!=UINT32_MAX?state.samp[ns][b.sampler_slot]:handle):0;
                        ikey={handle,sh,uint64_t(uintptr_t(state.texture_pool.pool->mem)),uint64_t(state.texture_pool.off),state.sampler_pool.pool?uint64_t(uintptr_t(state.sampler_pool.pool->mem)):0,uint64_t(state.sampler_pool.off),uint64_t(b.type)|(uint64_t((b.flags>>16)&3)<<8)};img_cacheable=true;
                        auto it=img_cache.find(ikey);
                        auto live=it==img_cache.end()?images.end():images.find(it->second.tex->serial);
                        if(it!=img_cache.end() && it->second.record->revision==it->second.revision &&
                            live!=images.end() && live->second.get()==it->second.im){Image *im=&get_image(it->second.tex,true);
                            prepare_read(*im,commands.command);if(desc_memo_recording)desc_memo_images.push_back({it->second.tex,im,it->second.record,it->second.revision});
                            VkDescriptorImageInfo ii{};ii.imageView=it->second.view;ii.sampler=b.type==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER?it->second.sampler:VK_NULL_HANDLE;ii.imageLayout=VK_IMAGE_LAYOUT_GENERAL;
                            image_infos.push_back(ii);write.pImageInfo=&image_infos.back();img_fast=true;++img_cache_hits;
                            {BufRef &r=img_slot_refs[it->second.view];const void *cur=&recording_slot();
                             for(const void *old_slot:r.slots)if(old_slot!=cur){bool alive=false;for(const Slot &sl:slots)if(&sl==old_slot){alive=sl.pending||sl.recording;break;}
                                 if(!alive){static int n;if(n++<8)xbg_logf("【红】跨槽引用悬空 VkImageView=%p（快路径）登记槽=%p 当前槽=%p serial=%llu",(void*)it->second.view,old_slot,cur,(unsigned long long)slot_serial);}}
                             if(r.slots.empty()||r.slots.back()!=cur){r.slots.push_back(cur);if(r.slots.size()>4)r.slots.erase(r.slots.begin());}}
                        }
                        else if(it!=img_cache.end())img_cache.erase(it);
                    }
                }
                if(!img_fast){
                need(b.source==XBG_SOURCE_TEXTURE || b.source==XBG_SOURCE_IMAGE,"Unsupported image source");unsigned slot=b.slot+element;need(slot<(b.source==XBG_SOURCE_IMAGE?16u:128u),"Image binding slot out of range");uint64_t handle=b.source==XBG_SOURCE_IMAGE?state.img[ns][slot]:state.tex[ns][slot];int id=int(handle&0xfffff);
                const RegisteredTexture *record=nullptr;
                if(id==0 || (state.texture_pool.n>0 && id>=state.texture_pool.n))record=&null_texture(b);
                else{PoolKey pk=pool_key(&state.texture_pool);if(!texture_pool_cache || !(pk==texture_pool_cache_key)){auto pool=textures.find(pk);texture_pool_cache=pool==textures.end()?&legacy_textures:&pool->second;texture_pool_cache_key=pk;}auto tex=texture_pool_cache->find(id);if(tex==texture_pool_cache->end())xbg_fatalf("Unregistered texture descriptor id=%d stage=%u slot=%u",id,ns,slot);record=&tex->second;}
                const auto &registered=*record;const tex_t &texture=registered.texture;
                if(b.type==VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || b.type==VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER){need(texture.d.target==10 && texture.mem && texture.L.size,"Invalid buffer texture");
                    TexelResident *res=nullptr;
                    if(b.type==VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER && texel_memo_enabled && !compute){Buffer pv;
                        if(!pending_view(uint64_t(texture.mem),texture.L.size,1,pv))res=texel_acquire(uint64_t(texture.mem),texture.L.size,format(texture.d.format));
                        if(res){if(desc_memo_recording)desc_memo_texel.push_back({record,record->revision,uint64_t(texture.mem),uint64_t(texture.L.size),res->gen,uint32_t(format(texture.d.format))});
                            texel_views.push_back(res->view);write.pTexelBufferView=&texel_views.back();}
                        else desc_memo_texel_fail=true;}   // the set then holds a per-frame view: not memoized
                    if(!res){
                    Buffer *buf=nullptr;uint64_t base=uint64_t(texture.mem),bytes=texture.L.size;
                    if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER){Buffer view;if(pending_view(base,bytes,1,view)){buf=out.keep(readback_buffer(bytes));VkCommandBuffer cmd=recording_slot().cmd;end_pass(6);VkBufferCopy region{view.offset,buf->offset,bytes};vkCmdCopyBuffer(cmd,view.buffer,buf->buffer,1,&region);g_barrier_site="texel-readback";full_barrier(cmd);}else{cpu_access(base,bytes,8);buf=out.keep(readback_buffer(bytes,texture.mem,bytes));}out.storage.push_back({{base,bytes},buf});out.written.push_back({{base,bytes},buf});}
                    else{snap_site=4;buf=out.keep(snapshot(base,bytes,properties.limits.minTexelBufferOffsetAlignment));snap_site=0;texel_frame_keys.insert((base<<12)^bytes);++texel_binds;}
                    VkBufferViewCreateInfo v{VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};v.buffer=buf->buffer;v.offset=buf->offset;v.format=format(texture.d.format);v.range=bytes;VkBufferView bv;{auto &cache=recording_slot().texel_views;std::array<uint64_t,4> key{uint64_t(v.buffer),v.offset,v.range,uint64_t(v.format)};auto f=cache.find(key);if(f!=cache.end())bv=f->second;else{VK(vkCreateBufferView(cfg.device,&v,nullptr,&bv));cache.emplace(key,bv);}}texel_views.push_back(bv);write.pTexelBufferView=&texel_views.back();}}
                else{if(aspect(texture.d.format)!=VK_IMAGE_ASPECT_COLOR_BIT){auto prior=images.find(texture.serial);if(prior==images.end() || (!prior->second->uploaded && !prior->second->gpu_written))xbg_logf("NVN initial depth descriptor shader=%s stage=%u slot=%u id=%d serial=%llu format=%#x",shader.key.c_str(),ns,slot,id,(unsigned long long)texture.serial,texture.d.format);}
                    VkDescriptorImageInfo ii{};
                    Image &im=get_image(&texture,true);provenance_log("BIND shader=%s stage=%u slot=%u id=%d handle=%llu descriptorMem=%p descriptorOff=%lld type=%u",shader.key.c_str(),ns,slot,id,(unsigned long long)handle,state.texture_pool.pool?state.texture_pool.pool->mem:nullptr,(long long)state.texture_pool.off,unsigned(b.type));provenance_image("BIND",&texture,im,registered.has_view?&registered.view:nullptr);DESC_CALL(ImageTransition, prepare_read(im,commands.command));if(desc_memo_recording)desc_memo_images.push_back({&texture,&im,record,record->revision});ii.imageView=DESC_CALL(ImageView, im.view(registered.has_view?&registered.view:nullptr,false,b.type==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));ii.imageLayout=VK_IMAGE_LAYOUT_GENERAL;
                    if(capture.active){char label[100];snprintf(label,sizeof(label),"st%u-bind%u-el%u-input",st,b.binding,element);out.capture_inputs.push_back({&im,registered.view,registered.has_view,label});capture_log("TEXTURE label=%s shader=%s nvnStage=%u slot=%u handle=%#llx id=%d serial=%llu cachedSerial=%llu image=%p view=%p viewFormat=%u sourceKind=%u",label,shader.key.c_str(),ns,slot,(unsigned long long)handle,id,(unsigned long long)texture.serial,(unsigned long long)im.source.serial,(void*)im.image,(void*)ii.imageView,unsigned(im.view_format(registered.has_view?&registered.view:nullptr)),b.source);}
                    if(b.type==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || b.type==VK_DESCRIPTOR_TYPE_SAMPLER){DESC_SCOPE(SamplerBind);uint64_t sh=handle;if(b.sampler_slot!=UINT32_MAX){need(b.sampler_slot<128,"Sampler slot out of range");sh=state.samp[ns][b.sampler_slot];}int sid=int(sh>>20);samp_par parameters{};if(sid==0){parameters.ws=parameters.wt=parameters.wr=7;parameters.aniso=1;parameters.cmp_func=1;}
                        else{auto sp=samplers.find(pool_key(&state.sampler_pool));auto &table=sp==samplers.end()?legacy_samplers:sp->second;auto sa=table.find(sid);if(sa==table.end())xbg_fatalf("Unregistered sampler id=%d",sid);parameters=sa->second;}ii.sampler=sampler(parameters,im,im.view_format(registered.has_view?&registered.view:nullptr),((b.flags>>16)&3)!=0);if(capture.active)capture_log("SAMPLER st=%u binding=%u sid=%d samplerSlot=%u min=%u mag=%u wrap=%u/%u/%u lod=%g/%g bias=%g",st,b.binding,sid,b.sampler_slot,parameters.minf,parameters.magf,parameters.ws,parameters.wt,parameters.wr,parameters.lod_min,parameters.lod_max,parameters.lod_bias);}
                    if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE){VkFormat viewed=im.view_format(registered.has_view?&registered.view:nullptr);if(!(im.usage&VK_IMAGE_USAGE_STORAGE_BIT))xbg_fatalf("NVN storage image without IMAGE flag serial=%llu format=%#x flags=%#x",(unsigned long long)texture.serial,texture.d.format,texture.d.flags);const VkFormatProperties &fp=format_properties(viewed);need(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,"Format not usable as storage image");
                        if((b.flags&3) && format_feature_flags2){VkFormatProperties3 extra{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};VkFormatProperties2 all{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};all.pNext=&extra;vkGetPhysicalDeviceFormatProperties2(cfg.physical,viewed,&all);if(b.flags&1)need(extra.optimalTilingFeatures&VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT,"Image format lacks formatless-write support");if(b.flags&2)need(extra.optimalTilingFeatures&VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT,"Image format lacks formatless-read support");}
                        else if(b.flags&3){VkFormatProperties fp{};vkGetPhysicalDeviceFormatProperties(cfg.physical,viewed,&fp);bool listed=storage_without_format_listed(viewed) && (fp.optimalTilingFeatures&VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
                            if((b.flags&1) && !(listed && cfg.enabled_core.shaderStorageImageWriteWithoutFormat))xbg_fatalf("Image format %d lacks formatless-write support without VK_KHR_format_feature_flags2 (shader=%s listedWithStorageImage=%d shaderStorageImageWriteWithoutFormat=%u)",int(viewed),shader.key.c_str(),int(listed),unsigned(cfg.enabled_core.shaderStorageImageWriteWithoutFormat));
                            if((b.flags&2) && !(listed && cfg.enabled_core.shaderStorageImageReadWithoutFormat))xbg_fatalf("Image format %d lacks formatless-read support without VK_KHR_format_feature_flags2 (shader=%s listedWithStorageImage=%d shaderStorageImageReadWithoutFormat=%u)",int(viewed),shader.key.c_str(),int(listed),unsigned(cfg.enabled_core.shaderStorageImageReadWithoutFormat));}im.gpu_written=true,provenance_write(im,__LINE__);im.last_write=++write_seq;}
                    image_infos.push_back(ii);write.pImageInfo=&image_infos.back();
                    if(img_cacheable){img_cache.emplace(ikey,ImgEntry{&texture,&im,ii.imageView,ii.sampler,record,record->revision});++img_cache_misses;}
                    {BufRef &r=img_slot_refs[ii.imageView];const void *cur=&recording_slot();
                     for(const void *old_slot:r.slots)if(old_slot!=cur){bool alive=false;for(const Slot &sl:slots)if(&sl==old_slot){alive=sl.pending||sl.recording;break;}
                         if(!alive){static int n;if(n++<8)xbg_logf("【红】跨槽引用悬空 VkImageView=%p（慢路径）登记槽=%p 当前槽=%p serial=%llu",(void*)ii.imageView,old_slot,cur,(unsigned long long)slot_serial);}
                     }
                     if(r.slots.empty()||r.slots.back()!=cur){r.slots.push_back(cur);if(r.slots.size()>4)r.slots.erase(r.slots.begin());}}
                }
                }
            }
            writes.push_back(write);
        }
    }
    // Dynamic offsets are consumed in binding/array-element order.
    std::sort(dynamic_list.begin(),dynamic_list.end());out.dynamic_offsets.clear();for(auto &d:dynamic_list)out.dynamic_offsets.push_back(d.second);
    if(!set_cache_enabled){out.set=allocate_set(out.entry->layout);if(writes.empty())return;
        // Update templates: one driver call with packed payloads instead of parsing VkWriteDescriptorSet per element.
        LayoutEntry &e=*const_cast<LayoutEntry*>(out.entry);static std::vector<uint64_t> sig;sig.clear();
        for(auto &w:writes)sig.push_back((uint64_t(w.dstBinding)<<40)|(uint64_t(w.dstArrayElement)<<8)|uint64_t(w.descriptorType));
        if(template_enabled && !e.tmpl){std::vector<VkDescriptorUpdateTemplateEntry> entries;for(size_t i=0;i<writes.size();i++)entries.push_back({writes[i].dstBinding,writes[i].dstArrayElement,1,writes[i].descriptorType,i*sizeof(DescEntry),sizeof(DescEntry)});
            VkDescriptorUpdateTemplateCreateInfo ti{VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};ti.descriptorUpdateEntryCount=uint32_t(entries.size());ti.pDescriptorUpdateEntries=entries.data();ti.templateType=VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;ti.descriptorSetLayout=e.layout;
            if(vkCreateDescriptorUpdateTemplate(cfg.device,&ti,nullptr,&e.tmpl)==VK_SUCCESS)e.tmpl_sig=sig;else e.tmpl=VK_NULL_HANDLE;}
        if(template_enabled && e.tmpl && sig==e.tmpl_sig){static std::vector<DescEntry> data;data.resize(writes.size());
            for(size_t i=0;i<writes.size();i++){auto &w=writes[i];if(w.pBufferInfo)data[i].buffer=*w.pBufferInfo;else if(w.pImageInfo)data[i].image=*w.pImageInfo;else data[i].view=*w.pTexelBufferView;}
            vkUpdateDescriptorSetWithTemplate(cfg.device,out.set,e.tmpl,data.data());++template_updates;return;}
        for(auto &w:writes)w.dstSet=out.set;vkUpdateDescriptorSets(cfg.device,uint32_t(writes.size()),writes.data(),0,nullptr);++plain_updates;return;}
    uint64_t tk0=perf_clock();
    set_key.clear();set_key.push_back(uint64_t(out.entry->layout));
    for(auto &w:writes){set_key.push_back((uint64_t(w.dstBinding)<<32)|w.dstArrayElement);set_key.push_back(w.descriptorType);
        if(w.pBufferInfo){set_key.push_back(uint64_t(w.pBufferInfo->buffer));set_key.push_back(w.pBufferInfo->offset);set_key.push_back(w.pBufferInfo->range);}
        else if(w.pImageInfo){set_key.push_back(uint64_t(w.pImageInfo->sampler));set_key.push_back(uint64_t(w.pImageInfo->imageView));set_key.push_back(w.pImageInfo->imageLayout);}
        else if(w.pTexelBufferView)set_key.push_back(uint64_t(*w.pTexelBufferView));}
    Slot &slot=recording_slot();bool use_cache=true;auto hit=use_cache?slot.set_cache.find(set_key):slot.set_cache.end();desc_ns[3]+=perf_clock()-tk0;
    if(hit!=slot.set_cache.end()){out.set=hit->second;++set_cache_hits;return;}
    {uint64_t t0=perf_clock();out.set=allocate_set(out.entry->layout);desc_ns[2]+=perf_clock()-t0;}++set_cache_misses;
    if(!writes.empty()){LayoutEntry &e=*const_cast<LayoutEntry*>(out.entry);static std::vector<uint64_t> sig2;sig2.clear();
        for(auto &w:writes)sig2.push_back((uint64_t(w.dstBinding)<<40)|(uint64_t(w.dstArrayElement)<<8)|uint64_t(w.descriptorType));
        if(!e.tmpl){std::vector<VkDescriptorUpdateTemplateEntry> entries;for(size_t i=0;i<writes.size();i++)entries.push_back({writes[i].dstBinding,writes[i].dstArrayElement,1,writes[i].descriptorType,i*sizeof(DescEntry),sizeof(DescEntry)});
            VkDescriptorUpdateTemplateCreateInfo ti{VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};ti.descriptorUpdateEntryCount=uint32_t(entries.size());ti.pDescriptorUpdateEntries=entries.data();ti.templateType=VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;ti.descriptorSetLayout=e.layout;
            if(vkCreateDescriptorUpdateTemplate(cfg.device,&ti,nullptr,&e.tmpl)==VK_SUCCESS)e.tmpl_sig=sig2;else e.tmpl=VK_NULL_HANDLE;}
        if(e.tmpl && sig2==e.tmpl_sig){static std::vector<DescEntry> data;data.resize(writes.size());
            for(size_t i=0;i<writes.size();i++){auto &w=writes[i];if(w.pBufferInfo)data[i].buffer=*w.pBufferInfo;else if(w.pImageInfo)data[i].image=*w.pImageInfo;else data[i].view=*w.pTexelBufferView;}
            {uint64_t t0=perf_clock();vkUpdateDescriptorSetWithTemplate(cfg.device,out.set,e.tmpl,data.data());desc_ns[1]+=perf_clock()-t0;}++template_updates;}
        else{for(auto &w:writes)w.dstSet=out.set;vkUpdateDescriptorSets(cfg.device,uint32_t(writes.size()),writes.data(),0,nullptr);++plain_updates;}}
    if(use_cache)slot.set_cache.emplace(set_key,out.set);
}
// Render passes are cached by their full definition; framebuffers are cached
// per frame slot (they reference image views of possibly released images).
struct Pass {
    const PipelineDefinition *definition=nullptr;
    VkRenderPass pass{},pass_noload{};VkFramebuffer framebuffer{};uint32_t width{},height{},color_slots{};VkSampleCountFlagBits samples=VK_SAMPLE_COUNT_1_BIT;ImageList images;
};
// Draws inside the open pass with unchanged render-target state get the same Pass: non-draw ops end the pass, so nothing
// else touched its attachments since. Only the attachments' write marks are refreshed (as prepare_attachment does).
// Reset with every submission and image upgrade. create_pass ran in full for each of ~2200 draws per D2R camp frame.
struct LastPass{decltype(state.rt) rt;Pass pass;bool valid=false;} last_pass;uint64_t pass_fast_hits=0;
void last_pass_reset(){last_pass.valid=false;}
// 保留零值计数用于与旧版本日志比较；任意 guest draw 不再丢弃附件内容。
// Keep zero counters for baseline logs; arbitrary guest draws retain attachment contents.
uint64_t noload_passes=0;bool noload_enabled=false;
// noload: the same pass with every color loadOp DONT_CARE (noload_enabled; first_draw_covers). Only load ops differ, so it is
// compatible with pass and its pipelines and framebuffers.
struct RenderPassEntry {VkRenderPass pass{},noload{};PipelineDefinition definition;};
std::map<std::vector<uint32_t>,RenderPassEntry> render_pass_cache;
void clear_render_pass_cache(){for(auto &r:render_pass_cache){vkDestroyRenderPass(cfg.device,r.second.pass,nullptr);if(r.second.noload)vkDestroyRenderPass(cfg.device,r.second.noload,nullptr);}render_pass_cache.clear();pipeline_fast.clear();pipe_memo_valid=false;}
bool create_pass(Pass &p,Commands &c,int clear_color=-1,bool clear_depth=false,bool soft=false){CPU_SCOPE(Pass);
    const bool plain_draw=c.draw && clear_color<0 && !clear_depth && !soft && !provenance_file && !capture.active && !ab_off("passfast");
    if(plain_draw && last_pass.valid && open_pass.cmd==c.command && open_pass.pass==last_pass.pass.pass && open_pass.framebuffer==last_pass.pass.framebuffer && !memcmp(&state.rt,&last_pass.rt,sizeof state.rt)){
        p=last_pass.pass;for(auto *im:p.images){im->last_write=++write_seq;im->gpu_written=true;}++pass_fast_hits;return true;}
    // 仅显式设置DM_VK_TRACE_ATTACHMENTS=1时输出成功诊断；首次调用读取，错误始终记录。
    // Read once: opt in with DM_VK_TRACE_ATTACHMENTS=1; errors are always logged.
    static const bool trace_attachments=[](){const char *v=std::getenv("DM_VK_TRACE_ATTACHMENTS");return v && std::strcmp(v,"1")==0;}();
    need(state.rt.n>=0 && state.rt.n<=8,"Invalid color target count");
    bool clear_only=clear_color>=0 || clear_depth;
    std::array<const tex_t*,9> targets{};std::array<const view_t*,9> selected{};std::array<xbg_framebuffer_attachment,9> extents{};
    for(int i=0;i<state.rt.n;i++)if(state.rt.color[i] && (!clear_only || clear_color==i)){targets[i]=state.rt.color[i];selected[i]=state.rt.has_cview[i]?&state.rt.cview[i]:nullptr;}
    if(state.rt.depth && (!clear_only || clear_depth)){targets[8]=state.rt.depth;selected[8]=state.rt.has_dview?&state.rt.dview:nullptr;}
    uint64_t trace=clear_only?1:0;
    ++cur_digest.passes;bool vol_attach=false;
    for(unsigned i=0;i<9;i++)if(targets[i]){
        const tex_t &t=*targets[i];const view_t *v=selected[i];
        if(t.d.target==2){++cur_digest.volatt;vol_attach=true;}   // np610：volume 切片当附件（np599 的能力）| np610: volume slice as an attachment
        need(t.magic==MAGIC('T','E','X',0) && t.serial,"Invalid NVN render attachment");
        // np597：实测会拿到 levelCount==0（全量内容后第一次撞上：depth 附件 index=8，mip=0/0，64x64）。
        //   NVN 里"0 层"不合法，而游戏在真机上不会传它 ⇒ 要么我们的 view 副本过期（SetLevels 发生在
        //   SetRenderTargets 之后），要么 0 表示"从 base 到最后一层"。按后者兜底并打一次日志，
        //   不要让整个 pass 直接 FATAL —— 否则一个附件就把后面的渲染全挡住。
        // np597: levelCount==0 does happen (first hit with the full romfs: depth attachment index=8,
        //   mip=0/0, 64x64). "0 levels" is invalid in NVN and the game does not pass it on real hardware,
        //   so either our view copy is stale (SetLevels after SetRenderTargets) or 0 means "from base to the
        //   last level". Fall back to the latter and log once instead of FATALing the whole pass.
        uint32_t bl=v&&v->set_levels?uint32_t(v->base_level):0u;
        uint32_t nl=xbg_view_nlevels(v,uint32_t(t.d.levels));
        // 3D 附件的“层”同样是深度切片：可用槽位数 = depth（不是 L.layers 的 1），未指定视图层时默认第 0 片。
        // A 3D attachment's "layer" is a depth slice too: the slot count is the depth (not L.layers, which is 1),
        // and an unspecified view layer means slice 0.
        const bool vol=t.d.target==2;const uint32_t span=vol?std::max(1u,t.d.depth):t.L.layers;
        extents[i]={1,t.d.width,t.d.height,t.d.levels,span,bl,nl,
            v&&v->set_layers?uint32_t(v->min_layer):0,v&&v->set_layers?uint32_t(v->nlayers):(vol?1u:t.L.layers),uint32_t(std::max(1,t.d.samples))};
        trace=(trace^t.serial^uint64_t(i)) * 1099511628211ull;trace^=hash_memory(&extents[i],sizeof(extents[i]));
        trace^=uint64_t(v&&v->set_format?v->format:t.d.format)<<32;
    }
    if(vol_attach)++cur_digest.volpasses;   // np610
    xbg_framebuffer_extent plan{};uint32_t bad=UINT32_MAX;int result=xbg_framebuffer_plan(extents.data(),extents.size(),properties.limits.maxFramebufferWidth,properties.limits.maxFramebufferHeight,&plan,&bad);
    if(result || (trace_attachments && trace!=last_attachment_trace)){
        xbg_logf("NVN framebuffer mode=%s result=%d viewport=%g,%g:%gx%g depthRange=%g/%g scissor=%d,%d:%ux%u selected=%ux%u samples=%u",clear_color>=0?"clear-color":clear_depth?"clear-depth-stencil":"draw",result,state.viewport.x,state.viewport.y,state.viewport.width,state.viewport.height,state.viewport.minDepth,state.viewport.maxDepth,state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,plan.width,plan.height,plan.samples);
        for(unsigned i=0;i<9;i++)if(targets[i]){const auto &e=extents[i];uint32_t w=e.base_level<32?std::max(1u,e.width>>e.base_level):0,h=e.base_level<32?std::max(1u,e.height>>e.base_level):0;
            xbg_logf("NVN attachment index=%d serial=%llu format=%#x viewFormat=%#x target=%d mip=%u/%u baseLayer=%u layerCount=%u totalLayers=%u base=%ux%u selected=%ux%u samples=%u",i==8?-1:int(i),(unsigned long long)targets[i]->serial,targets[i]->d.format,selected[i]&&selected[i]->set_format?selected[i]->format:targets[i]->d.format,targets[i]->d.target,e.base_level,e.level_count,e.base_layer,e.layer_count,e.layers,e.width,e.height,w,h,e.samples);
        }
        last_attachment_trace=trace;
    }
    if(result && soft)return false;
    if(result)xbg_fatalf("NVN framebuffer %s result=%d attachment=%d",xbg_framebuffer_error(result),result,bad==UINT32_MAX?-1:int(bad));
    p.width=plan.width;p.height=plan.height;p.samples=VkSampleCountFlagBits(plan.samples);
    static std::vector<VkAttachmentDescription> descriptions;descriptions.clear();static std::vector<VkImageView> views;views.clear();   // not re-entrant: graphics thread
    static std::vector<VkAttachmentReference> colors;colors.assign(size_t(state.rt.n),{VK_ATTACHMENT_UNUSED,VK_IMAGE_LAYOUT_GENERAL});VkAttachmentReference depth{VK_ATTACHMENT_UNUSED,VK_IMAGE_LAYOUT_GENERAL};
    for(unsigned slot=0;slot<9;slot++)if(targets[slot]){
        Image &im=get_image(targets[slot],false);const view_t *view=selected[slot];
        need((slot==8)==(im.aspects!=VK_IMAGE_ASPECT_COLOR_BIT),"Color/depth attachment role mismatches texture aspects");
        if(c.draw)prepare_attachment(im,c.command);else im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);VkAttachmentDescription a{};a.format=im.view_format(view);a.samples=im.samples;a.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;a.storeOp=VK_ATTACHMENT_STORE_OP_STORE;a.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_LOAD;a.stencilStoreOp=VK_ATTACHMENT_STORE_OP_STORE;a.initialLayout=a.finalLayout=VK_IMAGE_LAYOUT_GENERAL;
        uint32_t index=descriptions.size();descriptions.push_back(a);views.push_back(im.view(view,true));p.images.push_back(&im);im.gpu_written=true,provenance_write(im,__LINE__);
        if(slot==8)depth.attachment=index;else colors[slot].attachment=index;
    }
    p.color_slots=colors.size();VkSubpassDescription sub{};sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sub.colorAttachmentCount=colors.size();sub.pColorAttachments=colors.data();sub.pDepthStencilAttachment=depth.attachment!=VK_ATTACHMENT_UNUSED?&depth:nullptr;
    VkSubpassDependency dep[2]{};dep[0]={VK_SUBPASS_EXTERNAL,0,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,0};dep[1]={0,VK_SUBPASS_EXTERNAL,VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT,0};
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};rp.attachmentCount=descriptions.size();rp.pAttachments=descriptions.data();rp.subpassCount=1;rp.pSubpasses=&sub;rp.dependencyCount=2;rp.pDependencies=dep;
    {XbgPipelineKey key;bool valid=xbg_renderpass_key(key,rp);
        if(valid && key.valid){static std::vector<uint32_t> words;words.assign(key.words,key.words+key.count);auto it=render_pass_cache.find(words);
            if(it==render_pass_cache.end()){RenderPassEntry e;VK(vkCreateRenderPass(cfg.device,&rp,nullptr,&e.pass));e.definition.assign(key,true);
                if(noload_enabled && p.samples==VK_SAMPLE_COUNT_1_BIT){std::vector<VkAttachmentDescription> nl(descriptions.begin(),descriptions.end());bool any=false;
                    for(size_t k=0;k<nl.size() && k<p.images.size();k++)if(p.images[k]->aspects==VK_IMAGE_ASPECT_COLOR_BIT){nl[k].loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;any=true;}
                    if(any){VkRenderPassCreateInfo r2=rp;r2.pAttachments=nl.data();VK(vkCreateRenderPass(cfg.device,&r2,nullptr,&e.noload));}}
                it=render_pass_cache.emplace(words,std::move(e)).first;}
            p.pass=it->second.pass;p.pass_noload=it->second.noload;p.definition=&it->second.definition;
        }
        else{VK(vkCreateRenderPass(cfg.device,&rp,nullptr,&p.pass));VkRenderPass rpass=p.pass;retire_handle([rpass]{vkDestroyRenderPass(cfg.device,rpass,nullptr);});static PipelineDefinition invalid;p.definition=&invalid;}
    }
    Slot &slot=recording_slot();static std::vector<uint64_t> fb_key;fb_key.assign({uint64_t(p.pass),p.width,p.height});for(auto v:views)fb_key.push_back(uint64_t(v));
    // Cached render passes live until shutdown, so their framebuffers can outlive the frame; they go with the first of their
    // images (forget_framebuffers). A pass outside the cache is retired with this slot, and so are its framebuffers.
    bool persistent=persistent_fb_enabled && p.definition && p.definition->valid;
    auto &cache=persistent?persistent_framebuffers:slot.framebuffers;
    auto cached=cache.find(fb_key);
    if(cached!=cache.end()){p.framebuffer=cached->second;++fb_hits;}
    else{VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};fb.renderPass=p.pass;fb.attachmentCount=views.size();fb.pAttachments=views.data();fb.width=p.width;fb.height=p.height;
        // np626：层叠渲染时 framebuffer 的 layers 取所有附件层区间的最大值（单层附件=1，整条数组=层数）。
        // np626: for layered rendering the framebuffer's layers is the max of the attachments' layer ranges
        // (a single-layer attachment is 1, a whole-array attachment is its layer count).
        {uint32_t layers=1;for(unsigned i=0;i<9;i++)if(targets[i]){const auto &e=extents[i];layers=std::max(layers,e.layer_count);}fb.layers=layers;}
        VK(vkCreateFramebuffer(cfg.device,&fb,nullptr,&p.framebuffer));cache.emplace(fb_key,p.framebuffer);++fb_creates;
        if(persistent)for(auto *im:p.images)im->fb_keys.push_back(fb_key);}
    if(plain_draw && persistent){last_pass.rt=state.rt;last_pass.pass=p;last_pass.valid=true;}else last_pass.valid=false;
    return true;
}
VkRect2D scissor(const Pass &p){VkRect2D r{};need(xbg_clear_rectangle(p.width,p.height,state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,&r)==0,"Invalid scissor extent");return r;}
int pass_prof_reserve(VkCommandBuffer cmd){Slot &s=slots[slot_index];if(!s.prof || cmd!=s.cmd)return -1;
    // 开始时预留首尾查询，预算耗尽时整段跳过，不能留下只有开头的假计时。
    // Reserve both queries up front; skip the whole interval when the frame budget is exhausted.
    if(s.prof_used+2>pass_prof_query_limit || pass_prof_queries+2>pass_prof_query_limit){++s.prof_skipped;return -1;}
    Slot::PassRec r;r.q0=s.prof_used++;r.q1=s.prof_used++;r.draws0=r.draws1=uint32_t(pass_prof_draws);r.verts0=r.verts1=pass_prof_verts;pass_prof_queries+=2;
    s.prof_recs.push_back(std::move(r));int index=int(s.prof_recs.size())-1;
    vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,s.prof_pool,s.prof_recs[index].q0);return index;
}
void pass_prof_complete(VkCommandBuffer cmd,int index){Slot &s=slots[slot_index];if(!s.prof || cmd!=s.cmd || index<0 || size_t(index)>=s.prof_recs.size())return;
    auto &r=s.prof_recs[index];r.draws1=uint32_t(pass_prof_draws);r.verts1=pass_prof_verts;vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,s.prof_pool,r.q1);r.complete=true;
}
void pass_prof_end(VkCommandBuffer cmd){Slot &s=slots[slot_index];if(!s.prof || cmd!=s.cmd)return;pass_prof_complete(cmd,s.prof_open);s.prof_open=-1;}
void pass_prof_depth_idle(VkCommandBuffer cmd){Slot &s=slots[slot_index];if(!s.prof || cmd!=s.cmd || s.prof_open<0 || size_t(s.prof_open)>=s.prof_recs.size())return;s.prof_recs[size_t(s.prof_open)].dz=true;}
void pass_prof_begin(const Pass &p,VkCommandBuffer cmd){Slot &s=slots[slot_index];s.prof_open=pass_prof_reserve(cmd);if(s.prof_open<0)return;
    auto &r=s.prof_recs[s.prof_open];r.w=p.width;r.h=p.height;r.att=uint32_t(p.images.size());r.fmt=p.images.empty()?0u:uint32_t(p.images[0]->fmt);
}
void pass_prof_draw(VkCommandBuffer cmd){Slot &s=slots[slot_index];if(!s.prof || cmd!=s.cmd || s.prof_open<0)return;
    std::string keys;for(unsigned st=0;st<5;++st)if(state.shader[st]){if(!keys.empty())keys+=';';keys+=std::to_string(st)+":"+state.shader[st]->key;}
    auto &r=s.prof_recs[s.prof_open];if(r.first.empty())r.first=keys;else if(r.last!=keys)r.mixed=true;r.last=std::move(keys);
}
int pass_prof_compute_begin(VkCommandBuffer cmd,const p_dispatch &p){int index=pass_prof_reserve(cmd);if(index<0)return index;
    auto &r=slots[slot_index].prof_recs[index];r.compute=true;r.first=state.shader[5]->key;r.last=r.first;r.x=uint32_t(p.x);r.y=uint32_t(p.y);r.z=uint32_t(p.z);return index;
}
void begin_pass(Pass &p,Commands &c){deathguard::pass_begin(p.width,p.height,p.images.n,uint32_t(p.samples),p.images.n&&p.images.v[0]?p.images.v[0]->source.serial:0ull,state.rt.depth!=nullptr);pass_prof_begin(p,c.command);
    // np713：阴影 pass 识别与备战（规则见 shadow_scale_on 注释）
    g_shadow_armed=false;g_shadow_img=nullptr;
    unsigned sn=p.width>=512?0:0;
    if(shadow_scale_on() && p.images.size()==1 && p.width==p.height && p.width>=768 && p.width%unsigned(shadow_scale_pct)==0 && (sn=p.width/unsigned(shadow_scale_pct))>=2){
        (void)sn;
        g_shadow_img=p.images.v[0];g_shadow_pw=p.width;g_shadow_ph=p.height;
        g_shadow_sw=p.width*unsigned(shadow_scale_pct)/100;g_shadow_sh=p.height*unsigned(shadow_scale_pct)/100;
        g_shadow_armed=true;g_shadow_quad=unsigned(shadow_scale_pct)>0&&(p.width*100)%unsigned(shadow_scale_pct)==0?p.width/g_shadow_sw:0;
        if(!g_shadow_quad)g_shadow_armed=false;}VkRenderPassBeginInfo b{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};b.renderPass=p.pass;b.framebuffer=p.framebuffer;b.renderArea.extent={p.width,p.height};vkCmdBeginRenderPass(c.command,&b,VK_SUBPASS_CONTENTS_INLINE);}
void capture_pass(Pass &pass){
    if(!capture.active)return;
    capture_log("PASS width=%u height=%u samples=%u viewport=%g,%g,%g,%g scissor=%d,%d,%u,%u mask=%#x blendEnable=%#x cull=%u front=%u depthTest=%u depthWrite=%u depthFunc=%u stencil=%u",pass.width,pass.height,unsigned(pass.samples),state.viewport.x,state.viewport.y,state.viewport.width,state.viewport.height,state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,state.mask.mask,state.color.blend_en,state.poly.cull,state.poly.front,unsigned(state.ds.dt),unsigned(state.ds.dw),unsigned(state.ds.df),unsigned(state.ds.st));
    for(int index=0;index<9;index++){
        const tex_t *t=index==8?state.rt.depth:index<state.rt.n?state.rt.color[index]:nullptr;if(!t)continue;
        auto found=images.find(t->serial);if(found==images.end())continue;Image &im=*found->second;
        if(std::find(pass.images.begin(),pass.images.end(),&im)==pass.images.end())continue;
        bool has_view=index==8?state.rt.has_dview:state.rt.has_cview[index];view_t v=has_view?(index==8?state.rt.dview:state.rt.cview[index]):view_t{};
        if(!v.set_levels)v.base_level=0;if(!v.set_layers)v.min_layer=0;v.set_levels=v.set_layers=1;v.nlevels=v.nlayers=1;
        char label[40];snprintf(label,sizeof(label),"rt%d-output",index==8?-1:index);capture_image(im,&v,label);
    }
}
VkPrimitiveTopology topology(int prim){switch(prim){case 0:return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;case 1:return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;case 3:return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;case 4:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;case 5:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;case 6:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;case 10:return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;case 11:return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;case 12:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;case 13:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;case 14:return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;default:xbg_fatalf("Unsupported draw primitive %d",prim);}}
VkStencilOpState stencil(bool back){VkStencilOpState s{};s.failOp=VkStencilOp((back?state.ds.fail_b:state.ds.fail_f)-1);s.passOp=VkStencilOp((back?state.ds.zpass_b:state.ds.zpass_f)-1);s.depthFailOp=VkStencilOp((back?state.ds.zfail_b:state.ds.zfail_f)-1);s.compareOp=VkCompareOp((back?state.ds.sf_b:state.ds.sf_f)-1);s.compareMask=state.stencil_compare[back];s.writeMask=state.stencil_write[back];s.reference=state.stencil_ref[back];return s;}
// Redundant-bind filter for the draw path (~1800 draws/frame re-bound identical pipeline / set / viewport / scissor /
// vertex buffers). Valid only within one command buffer recording; any other bind site (clear pipeline) and every
// command-buffer begin reset it. files/gfx-no-bindcache disables.
struct BindCache{VkCommandBuffer cmd=VK_NULL_HANDLE;VkPipeline pipe=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;VkDescriptorSet set=VK_NULL_HANDLE;
    uint32_t noff=UINT32_MAX,off[32]{};bool has_vp=false,has_sc=false;VkViewport vp{};VkRect2D sc{};VkBuffer vb[16]{};VkDeviceSize vbo[16]{};
    VkBuffer ib=VK_NULL_HANDLE;VkDeviceSize ibo=0;VkIndexType it=VK_INDEX_TYPE_MAX_ENUM;};
BindCache bind_cache;bool bind_cache_enabled=true;uint64_t binds_skipped=0;
// Descriptor-set binds per frame (NVN PERF binds): how many, their dynamic offsets, and how many rebind the same set with new
// offsets only.
uint64_t set_binds=0,set_bind_offsets=0,set_binds_offsets_only=0;
// Offset-only rebinds (NVN PERF binds ... changed): how many dynamic offsets differ from the bound ones, and how often each
// offset position does. A push-constant path for a uniform block that changes on every draw would need them few and fixed.
uint64_t offset_changes[10]={},offset_changed_at[32]={};
void bind_cache_reset(){bind_cache=BindCache{};}
// 顶点数量与全 viewport/scissor 不能证明几何覆盖所有像素（局部三角形也满足）。所以只在几何上证明了的时候才不读旧内容：
// 与 Metal 侧（nvn_mtl.m 的 first_draw_covers）同一套条件与同一个纯函数 nvn_cover.h（单测 t_cover.c），位置从这次绘制
// 实际交给 GPU 的顶点 / 索引快照里读。不成立就照旧 LOAD。
// Vertex count and full viewport/scissor do not prove full pixel coverage: a partial triangle qualifies. So the old contents
// are skipped only when the geometry proves it: the conditions of the Metal side (nvn_mtl.m's first_draw_covers) and the same
// pure function nvn_cover.h (unit test t_cover.c), with the positions read from the vertex and index snapshots this draw
// hands to the GPU. Otherwise LOAD as before. Explicit clears keep their separate clear semantics.
// The first draw of a pass covers every pixel of each color attachment, writing it without blending: the vertex position
// is one input as it is (XbgRasterFacts), the fragment shader declares every attachment's output, never discards and does
// not write SampleMask, no depth (except ALWAYS), stencil or depth-bounds test, fill mode, single sample, 2D targets, and
// nvn_cover holds for the viewport, scissor, culling (the NVN cull and front face, as the Metal side passes them) and
// vertices. Should its winding reading ever differ from the GPU's, the visible triangle reads as culled and the pass simply
// loads. noload_why counts the first failing check:
// 0 shaders, 1 attachment state, 2 depth/stencil, 3 raster state (also: covered but culled), 4 primitive/count,
// 5 position format/stream, 6 vertex/index read, 7 geometry.
uint64_t noload_why[8]={};
bool first_draw_covers(const Pass &p,int prim,int first,int count,const p_draw_idx *indexed,const std::vector<std::pair<uint32_t,Buffer*>> &vbufs,const Buffer *ib,VkIndexType itype){
    const Shader *vs=state.shader[0],*fs=state.shader[4];
    if(!vs || !fs || state.shader[1] || state.shader[2] || state.shader[3] || vs->raster.pos_input<0 || vs->raster.pos_input>=16 || fs->discards || fs->raster.sample_mask){++noload_why[0];return false;}
    for(int i=0;i<state.rt.n;i++)if(state.rt.color[i]){
        if(!(fs->raster.outputs>>i&1) || (state.color.blend_en>>i&1) || ((state.mask.mask>>(4*i))&15)!=15 || state.rt.color[i]->d.target==2){++noload_why[1];return false;}}
    if(state.rt.depth && ((state.ds.dt && state.ds.df!=8) || state.ds.st)){++noload_why[2];return false;}
    if(state.bounds.enable){++noload_why[2];return false;}
    if(state.ms.a2c || state.poly.mode!=2 || p.samples!=VK_SAMPLE_COUNT_1_BIT){++noload_why[3];return false;}
    if((prim!=NVN_COVER_TRIANGLES && prim!=NVN_COVER_TRIANGLE_STRIP) || count<3 || count>64){++noload_why[4];return false;}
    va_st a;memcpy(&a,&state.attrib[vs->raster.pos_input],4);
    const VkFormat f=a.fmt?format(a.fmt):VK_FORMAT_UNDEFINED;const int nc=f==VK_FORMAT_R32G32_SFLOAT?2:f==VK_FORMAT_R32G32B32_SFLOAT?3:f==VK_FORMAT_R32G32B32A32_SFLOAT?4:0;
    if(!nc || a.stream>=16 || !state.stream[a.stream].stride || state.stream[a.stream].divisor){++noload_why[5];return false;}
    const Buffer *vb=nullptr;for(auto &b:vbufs)if(b.first==a.stream)vb=b.second;
    if(!vb || !vb->mapped || (indexed && (!ib || !ib->mapped))){++noload_why[6];return false;}
    const uint64_t stride=state.stream[a.stream].stride;const size_t isz=itype==VK_INDEX_TYPE_UINT16?2:4;
    float P[64][4];
    for(int k=0;k<count;k++){
        int64_t vid;
        if(indexed){if(uint64_t(k+1)*isz>ib->size){++noload_why[6];return false;}const uint8_t *q=static_cast<const uint8_t*>(ib->mapped)+size_t(k)*isz;
            uint32_t x=0;if(isz==2){uint16_t h;memcpy(&h,q,2);x=h;}else memcpy(&x,q,4);vid=int64_t(x)+indexed->base_vertex;}
        else vid=int64_t(first)+k;
        if(vid<0){++noload_why[6];return false;}
        const uint64_t off=uint64_t(vid)*stride+a.offset;
        if(off+4u*unsigned(nc)>vb->size){++noload_why[6];return false;}
        float v[4]={0,0,0,1};memcpy(v,static_cast<const uint8_t*>(vb->mapped)+off,4*size_t(nc));
        if(vs->raster.pos_w1)v[3]=1;
        memcpy(P[k],v,sizeof v);
    }
    const VkRect2D sr=scissor(p);const int sc[4]={sr.offset.x,sr.offset.y,int(sr.extent.width),int(sr.extent.height)};
    const float vp[4]={state.viewport.x,state.viewport.y,state.viewport.width,state.viewport.height};
    // A failure that culling alone causes counts under raster state (3), the rest under geometry (7).
    if(!nvn_cover(P,count,prim,int(state.poly.cull),state.poly.front==1,1,vp,1,sc,long(p.width),long(p.height))){
        ++noload_why[state.poly.cull && nvn_cover(P,count,prim,NVN_COVER_CULL_NONE,1,1,vp,1,sc,long(p.width),long(p.height))?3:7];return false;}
    return true;
}
// gfx-async-pipelines: may this draw be skipped while its pipeline builds? Only when losing it for a few frames loses a
// translucent effect: every written color target blends with the destination (a destination factor other than ZERO, so
// a ONE / ZERO "blend" that overwrites is not one), no depth or stencil write, no storage buffer or image writes in any
// stage, no capture running, and no samples counted once the game has recorded a samples-passed report. D2R counts samples
// for every draw after its first counter reset but never reports them, so counting alone excluded every draw (fc46144).
bool async_skippable(const std::vector<VkPipelineColorBlendAttachmentState> &att,const VkPipelineDepthStencilStateCreateInfo &ds){
    if(!warm::async_on || (state.count_samples && samples_report_seen) || capture.active || ds.depthWriteEnable)return false;
    if(ds.stencilTestEnable && (ds.front.writeMask || ds.back.writeMask))return false;
    bool writes=false;for(auto &a:att)if(a.colorWriteMask){if(!a.blendEnable || a.dstColorBlendFactor==VK_BLEND_FACTOR_ZERO)return false;writes=true;}
    if(!writes)return false;
    for(unsigned i=0;i<5;i++)if(state.shader[i]){if(!state.shader[i]->storage_writes.empty())return false;for(auto &b:state.shader[i]->bindings)if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)return false;}
    return true;
}
// 盖满整个界面矩形的无贴图块（同 nvn_mtl.m ui_cover_all）：顶点 4 个 × 20 字节，投影后正好铺满界面矩形
// An untextured quad covering the whole UI rectangle (as nvn_mtl.m ui_cover_all): 4 vertices × 20 bytes that, after projection, exactly fill the UI rectangle
bool ui_cover_all(double hf){
    auto v=state.vb[0];auto u=state.ub[0][0];if(!v.address || v.size!=80 || !u.address || u.size<64)return false;
    float m[16],x0=9,y0=9,x1=-9,y1=-9;memcpy(m,reinterpret_cast<const void*>(u.address),64);
    for(int i=0;i<4;i++){float f[2];memcpy(f,reinterpret_cast<const uint8_t*>(v.address)+20*i,8);float nx=(m[0]*f[0]+m[4]*f[1]+m[12]+1)/2,ny=(m[1]*f[0]+m[5]*f[1]+m[13]+1)/2;
        x0=std::min(x0,nx);x1=std::max(x1,nx);y0=std::min(y0,ny);y1=std::max(y1,ny);}
    return x0<0.002f && x1>0.998f && y0<0.002f && std::fabs(y1-float(hf))<0.002f;
}
// np539：建管线的阶段路标。np538 的现象是"两条 `NVN shader … loaded` 之后 1 毫秒就 SIGSEGV"，
//   而现场拿不到（信号 11 冻结了 stderr 转发线程）⇒ 至少先知道死在哪一段：取描述符 / 建 render pass /
//   组装顶点输入 / 算管线键 / 真的调 vkCreateGraphicsPipelines。
//   默认只跟**前 2 次** draw（DM_DRAW_TRACE=N 可改，0 = 关），常态运行不产生噪声。
// np539: stage breadcrumbs for pipeline creation. np538 died 1 ms after the two "NVN shader ... loaded" lines with
//   no site to look at (SIGSEGV freezes the stderr forwarding thread), so at minimum learn which stage it dies in:
//   descriptor setup / render pass / vertex input / pipeline key / the actual vkCreateGraphicsPipelines call.
//   Only the first 2 draws are traced (DM_DRAW_TRACE=N overrides, 0 = off), so normal runs stay quiet.
static int g_trace_draws = 0;
static int trace_limit() { static int n = -1; if (n < 0) { const char *e = getenv("DM_DRAW_TRACE"); n = e ? atoi(e) : 2; } return n; }
void dtrace(const char *what) { if (g_trace_draws < trace_limit()) xbg_logf("DRAWTRACE %s", what); }
// np539：命令级路标。np539 真机证明崩点**不在 draw()**（DRAWTRACE 一条都没有），而是在 BIND_PROGRAM
//   之后的下一条命令里。所以把每条命令的 op 打出来（默认只打前 240 条，DM_OP_TRACE=N 可改，0 = 关），
//   崩在哪就看最后一条 OPTRACE —— op 编号见 gfx/nvn/nvn_obj.h 的 enum。
// np539: command-level breadcrumbs. The np539 device run proved the crash is not inside draw() (no DRAWTRACE at
//   all) but in the command right after BIND_PROGRAM, so log every command's op (first 240; DM_OP_TRACE=N
//   overrides, 0 = off); the last OPTRACE line names the handler it died in. Op numbers: enum in gfx/nvn/nvn_obj.h.
static int g_op_trace = 0;
static int op_trace_limit() { static int n = -1; if (n < 0) { const char *e = getenv("DM_OP_TRACE"); n = e ? atoi(e) : 2000; } return n; }
void dispatch_op_trace(unsigned op, unsigned size, unsigned depth) {
    if (g_op_trace >= op_trace_limit()) return;
    ++g_op_trace; xbg_logf("OPTRACE %u size=%u depth=%u", op, size, depth);
}
// np540：把"最后一条命令"记在全局里，崩溃现场直接读——路标可能因预算/日志回绕丢尾巴，这个不会。
// np540: keep the last dispatched command in a global so the crash site can read it; the OPTRACE lines can be lost
// to the budget or a log wrap, this cannot. extern "C" 是为了让 dm/host/crash.c 能直接按名字取到。
extern "C" { volatile unsigned dm_last_cmd_op = 0xffffffffu; volatile unsigned dm_last_cmd_size = 0; }
void dispatch_set_last(unsigned op, unsigned size) { dm_last_cmd_op = op; dm_last_cmd_size = size; }
void draw(int prim,int first,int count,const p_draw_idx *indexed,uint32_t instances=1,uint32_t first_instance=0){g_gpu_wrote=true;preflight_cpu_reads();CPU_SCOPE(Draw);++cur_digest.draws;DrawRoot draw_timing_root(&cpu_scope);provenance_draw();
    need(state.rt.n>=0 && state.rt.n<=8,"Invalid draw color target count");
    dtrace("1 enter");
    VkShaderModule modules[5]{};for(unsigned i=0;i<5;i++)if(state.shader[i])modules[i]=state.shader[i]->module;
    uint32_t active_colors=0;for(int i=0;i<state.rt.n;i++)if(state.rt.color[i])active_colors|=1u<<i;
    static std::vector<VkPipelineShaderStageCreateInfo> stages;stages.assign(5,VkPipelineShaderStageCreateInfo{});uint32_t stage_count=0;   // draw(): graphics thread only, not re-entrant
    // np598：缺包跳过之后，下一帧标志已重置、但游戏没重新绑程序 ⇒ 这里会拿到空 VS（实测第 480 帧）。
    //   skip 模式的画面本来就是错的，这一条 draw 丢掉即可，不该把进程打死。
    // np598: after a skipped bundle the per-frame flag resets, yet the game does not re-bind the program, so a
    //   null vertex shader shows up here (seen at frame 480). In skip mode the picture is already wrong; drop
    //   this draw instead of killing the process.
    if(!modules[0] && shader_skip_on()){static int n;if(n++<4)xbg_logf("【诊断】draw 没有顶点着色器（缺包跳过所致）⇒ 丢掉这一条（第 %d 次）",n);return;}
    int stage_result=xbg_graphics_stages(modules,state.rt.depth!=nullptr,active_colors,state.mask.mask,state.ms.a2c,stages.data(),&stage_count);
    dtrace("2 stages");
    // np614：**np598 np608 的同一类漏洞**——降级模式只改了"缺 VS"这一支，缺 PS/GS 等任何必需阶段
    // 仍然走 FATAL。skip 模式下画面本来就是错的，缺哪个阶段都只该丢掉这一条 draw。
    // （判据：`xbg_graphics_stages` 的 result 非 0 就说明阶段不全；这里按 skip 模式统一兜。）
    // np614: the same class as np598/np608 -- the degradation only covered a missing vertex shader, while a
    // missing fragment/geometry stage still went to FATAL. In skip mode the picture is already wrong, so any
    // missing required stage should just drop this draw.
    if(stage_result){
        if(shader_skip_on()){static int n;if(n++<4)xbg_logf("【诊断】draw 的图形阶段不完整（result=%d，缺包跳过所致）⇒ 丢掉这一条（第 %d 次）",stage_result,n);return;}
        xbg_fatalf("Invalid graphics stage/depth-only state: %d",stage_result);
    }stages.resize(stage_count);need(count>=0 && first>=0,"Negative draw range");if(!count || (indexed && indexed->instances==0))return;provenance_draw();
    const int orig_count=count;std::vector<uint32_t> quad_indices;p_draw_idx quad_indexed{};
    if(prim==7 && !indexed){
        need(instances<=INT32_MAX && first_instance<=INT32_MAX,"Quad instance range exceeds backend limits");
        if(!instances)return;
        uint32_t expanded=0;need(xbg_quad_index_count(count,&expanded)==0,"Quad index expansion exceeds 64MiB bound");
        if(!expanded)return;
        quad_indices.resize(expanded);xbg_draw_call source{7,-1,first,count,int32_t(instances),int32_t(first_instance),0,0},triangles{};
        need(xbg_quad_plan(&source,&triangles,quad_indices.data(),quad_indices.size())==0,"Quad expansion failed");
        quad_indexed={triangles.primitive,triangles.index_type,triangles.count,triangles.base_vertex,triangles.base_instance,triangles.instances,triangles.index_address};
        prim=4;first=0;count=triangles.count;indexed=&quad_indexed;
    }
    if(capture_begin("draw",stats.presents))capture_log("DRAW number=%llu vs=%s ps=%s prim=%d first=%d count=%d indexed=%u instances=%u baseInstance=%u baseVertex=%d",(unsigned long long)stats.draws.load(),state.shader[0]->key.c_str(),state.shader[4]?state.shader[4]->key.c_str():"none",prim,first,count,indexed!=nullptr,indexed?indexed->instances:instances,indexed?indexed->base_instance:first_instance,indexed?indexed->base_vertex:0);
    double ui_uy=0,ui_vs=1;g_ui_moff=0;g_ui_moff_addr=0;
    if(g_ui_rect_fn && orig_count==4 && state.shader[0]->inputs.size()==2 && state.stream[0].stride==20){double uh=1;g_ui_rect_fn(&ui_uy,&uh);
        if(uh>0 && uh<1){if(ui_cover_all(uh))ui_vs=1/uh;else if(ui_uy>0){auto u=state.ub[0][0];if(u.address && u.size>=64 && u.size<=4096){g_ui_moff=2*ui_uy;g_ui_moff_addr=u.address;}}}
        if(ui_vs==1 && g_ui_moff==0)ui_uy=0;}
    Commands c(true);Descriptors &d=acquire_descriptors();descriptors(d,c,false);Pass p;create_pass(p,c);
    dtrace("3 pass");
    static std::vector<VkVertexInputAttributeDescription> attributes;attributes.clear();VkVertexInputBindingDescription binding_slots[16];uint32_t binding_used=0;
    int zero_binding=-1;
    dtrace("4 attr");
    for(size_t input=0;input<state.shader[0]->inputs.size();input++){
        uint32_t loc=state.shader[0]->inputs[input];need(loc<16,"Vertex input location out of bounds");va_st a;memcpy(&a,&state.attrib[loc],4);
        if(!a.fmt){
            if(zero_binding<0)zero_binding=xbg_vertex_zero_binding(state.attrib,state.shader[0]->inputs.data(),state.shader[0]->inputs.size());
            need(zero_binding>=0,"No free binding for constant vertex input");
            uint32_t cls=state.shader[0]->input_classes[input];need(cls<=2,"Unknown vertex scalar type");
            VkFormat f=cls==0?VK_FORMAT_R32G32B32A32_SFLOAT:cls==1?VK_FORMAT_R32G32B32A32_SINT:VK_FORMAT_R32G32B32A32_UINT;
            const VkFormatProperties &fp=format_properties(f);need(fp.bufferFeatures&VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT,"Constant vertex format unsupported");
            attributes.push_back({loc,uint32_t(zero_binding),f,0});need(zero_binding<16,"Zero binding out of range");binding_slots[zero_binding]={uint32_t(zero_binding),0,VK_VERTEX_INPUT_RATE_VERTEX};binding_used|=1u<<zero_binding;
            if(capture.active)capture_log("ATTRIBUTE location=%u constant=zero class=%u binding=%d",loc,cls,zero_binding);
            continue;
        }
        need(a.stream<16,"Vertex stream out of bounds");VkFormat fmt=format(a.fmt);const VkFormatProperties &fp=format_properties(fmt);need(fp.bufferFeatures&VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT,"Vertex format unsupported");need(state.stream[a.stream].divisor<=1,"Instance divisor >1 unsupported");attributes.push_back({loc,a.stream,fmt,a.offset});binding_slots[a.stream]={a.stream,state.stream[a.stream].stride,state.stream[a.stream].divisor?VK_VERTEX_INPUT_RATE_INSTANCE:VK_VERTEX_INPUT_RATE_VERTEX};binding_used|=1u<<a.stream;if(capture.active)capture_log("ATTRIBUTE location=%u stream=%u format=%u vk=%u offset=%u stride=%u divisor=%u",loc,a.stream,a.fmt,unsigned(fmt),a.offset,state.stream[a.stream].stride,state.stream[a.stream].divisor);
    }
    static std::vector<VkVertexInputBindingDescription> bindings;bindings.clear();for(uint32_t i=0;i<16;i++)if(binding_used&(1u<<i))bindings.push_back(binding_slots[i]);   // ascending binding order, as the former std::map
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};vi.vertexBindingDescriptionCount=bindings.size();vi.pVertexBindingDescriptions=bindings.data();vi.vertexAttributeDescriptionCount=attributes.size();vi.pVertexAttributeDescriptions=attributes.data();
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};ia.topology=topology(prim);
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vp.viewportCount=vp.scissorCount=1;
    // np693: guest pipelines run with GL clip-space depth (-w..w) so z=-w survives instead of being clipped.
    VkPipelineViewportDepthClipControlCreateInfoEXT clipctl{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT};
    clipctl.negativeOneToOne=VK_TRUE;clipctl.pNext=vp.pNext;if(clip_neg_one())vp.pNext=&clipctl;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.polygonMode=state.poly.mode==2?VK_POLYGON_MODE_FILL:state.poly.mode==1?VK_POLYGON_MODE_LINE:VK_POLYGON_MODE_POINT;need(rs.polygonMode==VK_POLYGON_MODE_FILL || cfg.enabled_core.fillModeNonSolid,"Non-solid polygon mode disabled");rs.cullMode=state.poly.cull;rs.frontFace=state.poly.front?VK_FRONT_FACE_COUNTER_CLOCKWISE:VK_FRONT_FACE_CLOCKWISE;rs.lineWidth=1;rs.depthBiasEnable=state.poly.offset_en!=0;rs.depthBiasConstantFactor=state.bias[1]*(state.rt.depth && state.rt.depth->d.format==0x32 && depth16_as_d24?256.0f:1.0f);
    rs.depthBiasSlopeFactor=state.bias[0];rs.depthBiasClamp=state.bias[2];need(!rs.depthBiasClamp || cfg.enabled_core.depthBiasClamp,"Depth bias clamp disabled");
    // np694: guest pipelines disable depth clipping (GL depth clamp) so z=-w is clamped, not discarded.
    VkPipelineRasterizationDepthClipStateCreateInfoEXT clipst{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT};
    clipst.depthClipEnable=VK_FALSE;clipst.pNext=rs.pNext;if(depth_clamp_on())rs.pNext=&clipst;   // constant bias units: 2^-16 for D16, 2^-24 for the X8_D24 stand-in; slope and clamp do not depend on the format
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};ms.rasterizationSamples=p.samples;ms.alphaToCoverageEnable=state.ms.a2c;need(!state.ms.sl_en,"Custom sample locations unsupported");need(!state.ms.samples || state.ms.samples==uint32_t(p.samples),"NVN sample count differs from attachments");need(p.samples==VK_SAMPLE_COUNT_1_BIT || state.ms.enable,"Disabled multisampling on multisample target needs coverage emulation");
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};ds.depthTestEnable=state.ds.dt;ds.depthWriteEnable=state.ds.dw;ds.depthCompareOp=VkCompareOp(state.ds.df-1);ds.stencilTestEnable=state.ds.st;ds.front=stencil(false);ds.back=stencil(true);ds.depthBoundsTestEnable=state.bounds.enable;ds.minDepthBounds=state.bounds.minimum;ds.maxDepthBounds=state.bounds.maximum;need(!ds.depthBoundsTestEnable || cfg.enabled_core.depthBounds,"Depth bounds feature disabled");
    need(!(ds.depthTestEnable || ds.stencilTestEnable) || state.rt.depth,"Depth/stencil state requires attachment");
    static std::vector<VkPipelineColorBlendAttachmentState> attachments;attachments.assign(size_t(state.rt.n),VkPipelineColorBlendAttachmentState{});for(int i=0;i<state.rt.n;i++){const blend_st &b=state.blend[i];xbg_draw2d convert{};need(xbg_2d_blend_from_nvn(&convert,(state.color.blend_en>>i)&1,b.src_rgb,b.dst_rgb,b.src_a,b.dst_a,b.eq_rgb,b.eq_a,(state.mask.mask>>(4*i))&15)==0,"Unsupported blend state");attachments[i]=convert.blend;}
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};blend.attachmentCount=attachments.size();blend.pAttachments=attachments.data();memcpy(blend.blendConstants,state.blend_color,16);need(state.color.logic_op==3,"Non-copy logic operation unsupported");
    VkDynamicState dynamic[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dy.dynamicStateCount=2;dy.pDynamicStates=dynamic;
    VkPipelineTessellationStateCreateInfo tess{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};tess.patchControlPoints=state.patch_size;need((state.shader[1]!=nullptr)==(state.shader[2]!=nullptr),"Incomplete tessellation shader pair");
    dtrace("6 ci");
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};ci.stageCount=stages.size();ci.pStages=stages.data();ci.pVertexInputState=&vi;ci.pInputAssemblyState=&ia;ci.pViewportState=&vp;ci.pRasterizationState=&rs;ci.pMultisampleState=&ms;ci.pDepthStencilState=&ds;ci.pColorBlendState=&blend;ci.pDynamicState=&dy;ci.pTessellationState=state.shader[1]?&tess:nullptr;ci.layout=d.pipeline_layout;ci.renderPass=p.pass;
    PipelineLease pipeline;{CPU_SCOPE(Pipeline);
        dtrace("7 key");
        static std::vector<uint32_t> fast;fast.clear();bool use_fast=pipeline_fast_enabled && !ab_off("pipefast") && d.entry->definition.valid && p.definition->valid;
        bool memo=use_fast && pipe_memo && !ab_off("pipememo");
        if(memo){pipe_memo_fill(pipe_memo_next,&d.entry->definition,p.definition,p.pass,d.pipeline_layout,prim,p.samples,active_colors);
            if(pipe_memo_valid && !memcmp(&pipe_memo_next,&pipe_memo_key,sizeof pipe_memo_next)){pipeline.pipeline=pipe_memo_pipeline;++pipe_memo_hits;}else ++pipe_memo_misses;}
        if(use_fast && !pipeline.pipeline){uint64_t a=uint64_t(uintptr_t(&d.entry->definition)),b=uint64_t(uintptr_t(p.definition));uint32_t ids[4]={uint32_t(a),uint32_t(a>>32),uint32_t(b),uint32_t(b>>32)};
            static XbgPipelineKey fkey;fkey.count=0;fkey.valid=true;   // 16 KB: one static, graphics thread only
            if(xbg_graphics_pipeline_key(fkey,ci,{ids,2,true},{ids+2,2,true})){fast.assign(fkey.words,fkey.words+fkey.count);auto f=pipeline_fast.find(fast);if(f!=pipeline_fast.end()){pipeline.pipeline=f->second;++pipeline_fast_hits;}}else use_fast=false;}
        if(!pipeline.pipeline){dtrace("8 create");XbgPipelineKey key;bool cacheable=xbg_graphics_pipeline_key(key,ci,d.entry->definition.view(),p.definition->view());pipeline_lookup(pipeline,key,cacheable,&ci,nullptr,async_skippable(attachments,ds));
            if(use_fast && cacheable && !pipeline.transient)pipeline_fast.emplace(fast,pipeline.pipeline);}
        if(memo && use_fast && pipeline.pipeline && !pipeline.transient){pipe_memo_key=pipe_memo_next;pipe_memo_pipeline=pipeline.pipeline;pipe_memo_valid=true;}}
    if(g_trace_draws<trace_limit()){++g_trace_draws;xbg_logf("DRAWTRACE 9 pipeline=%p prim=%d count=%d",(void*)(uintptr_t)pipeline.pipeline,prim,orig_count);}
    // gfx-async-pipelines: this draw is skipped while its pipeline builds in the background
    if(!pipeline.pipeline)return;
    if(pipe_stats){uint64_t fr=stats.presents.load();if(fr>=3000 && fr<3060){auto &u=pstat_use[pipeline.pipeline];++u.draws;if(u.what.empty())u.what=pstat_what()+" "+std::to_string(p.width)+"x"+std::to_string(p.height);}
        else if(fr>=3060 && !pstat_logged){pstat_logged=true;std::vector<std::pair<VkPipeline,PstatUse>> v(pstat_use.begin(),pstat_use.end());std::sort(v.begin(),v.end(),[](const auto &a,const auto &b){return a.second.draws>b.second.draws;});
            for(size_t i=0;i<v.size() && i<80;i++)xbg_logf("NVN PSTAT use pipeline=%p draws_per_frame=%.1f what=%s",(void*)v[i].first,v[i].second.draws/60.0,v[i].second.what.c_str());}}
    // Buffer copies precede the draw; original guest layout remains unchanged.
    static std::vector<std::pair<uint32_t,Buffer*>> vertex_buffers;vertex_buffers.clear();
    Buffer *ib=nullptr;VkIndexType index_type=VK_INDEX_TYPE_UINT16;
    {DRAW_SCOPE(VertexPrepare);for(auto &b:bindings){if(int(b.binding)==zero_binding){const uint32_t zero[4]={0};vertex_buffers.push_back({b.binding,d.keep(stream_buffer(sizeof(zero),zero,sizeof(zero)))});continue;}auto r=state.vb[b.binding];need(r.address && r.size,"Unbound vertex buffer");snap_site=1;vertex_buffers.push_back({b.binding,d.keep(snapshot(r.address,r.size))});snap_site=0;}
    if(indexed){need(indexed->itype>=0 && indexed->itype<=2 && indexed->instances>0 && indexed->base_instance>=0,"Invalid indexed draw");size_t bytes=size_t(count)*(1u<<indexed->itype);need(indexed->ibuf,"Unbound index buffer");if(indexed->itype==0){if(draw_detail)draw_timing.index(uint64_t(count));std::vector<uint16_t> expanded(count);const uint8_t *src=reinterpret_cast<const uint8_t*>(indexed->ibuf);for(int i=0;i<count;i++)expanded[i]=src[i];ib=d.keep(stream_buffer(expanded.size()*2,expanded.data(),expanded.size()*2));}else{index_type=indexed->itype==1?VK_INDEX_TYPE_UINT16:VK_INDEX_TYPE_UINT32;snap_site=2;ib=indexed==&quad_indexed?d.keep(stream_buffer(bytes,reinterpret_cast<void*>(indexed->ibuf),bytes)):d.keep(snapshot(uint64_t(indexed->ibuf),bytes));snap_site=0;}}
    } // VertexPrepare excludes capture, recording and storage writeback.
    if(capture.active){for(auto &v:vertex_buffers){char label[40];snprintf(label,sizeof(label),"vertex-stream%u",v.first);capture_log("VERTEX label=%s guest=%#llx bytes=%zu",label,(unsigned long long)state.vb[v.first].address,v.second->size);capture_data(label,v.second->mapped,v.second->size);}if(ib){capture_log("INDEX guest=%#llx vkType=%u bytes=%zu",(unsigned long long)indexed->ibuf,unsigned(index_type),ib->size);capture_data("index",ib->mapped,ib->size);}for(auto &input:d.capture_inputs)for(auto *output:p.images)if(input.image==output)capture_log("FEEDBACK input=%s serial=%llu",input.label.c_str(),(unsigned long long)output->source.serial);}
    VkQueryPool occlusion{};uint32_t occlusion_index=0;const bool counting=state.count_samples && !no_occlusion_queries && !ab_off("occ");
    const bool new_run=counting && (!occlusion_batch_enabled || ab_off("query") || !open_occlusion.cmd || open_pass.cmd!=c.command || open_pass.pass!=p.pass || open_pass.framebuffer!=p.framebuffer);
    if(new_run){need(cfg.enabled_core.occlusionQueryPrecise,"Precise samples-passed query disabled");occlusion_query(occlusion,occlusion_index);}   // a new pool ends the pass first
    if(open_pass.cmd!=c.command || open_pass.pass!=p.pass || open_pass.framebuffer!=p.framebuffer){
        // np639：记"为什么开新 pass"（0 没有 pass 开着 / 1 换了命令缓冲 / 2 换了 VkRenderPass / 3 换了 framebuffer）
        // np639: record why a new pass is opened (0 nothing open / 1 other cmd buffer / 2 other VkRenderPass / 3 other framebuffer)
        ++pass_open_by[!open_pass.cmd?0:(open_pass.cmd!=c.command?1:(open_pass.pass!=p.pass?2:3))];
        if(p.framebuffer)pass_open_fbs.insert(uint64_t(uintptr_t(p.framebuffer)));
        if(pass_seq_n<512)pass_seq[pass_seq_n++]={uint64_t(uintptr_t(p.framebuffer)),uint32_t(p.width),uint32_t(p.height),uint32_t(p.images.size())};
        end_pass(5);
        // Proven full overwrite: begin with the color-DONT_CARE twin; open_pass keeps p.pass, so later draws continue this pass.
        if(p.pass_noload && quad_indices.empty() && !ab_off("noload") && ui_uy==0 && ui_vs==1 && first_draw_covers(p,prim,first,count,indexed,vertex_buffers,ib,index_type)){VkRenderPass keep=p.pass;p.pass=p.pass_noload;begin_pass(p,c);p.pass=keep;++noload_passes;}
        else begin_pass(p,c);
        open_pass.cmd=c.command;open_pass.pass=p.pass;open_pass.framebuffer=p.framebuffer;open_pass.images=p.images;open_pass.serial0=p.images.n&&p.images.v[0]?p.images.v[0]->source.serial:0;open_pass.depth_attached=state.rt.depth!=nullptr;++perf.passes;}
    if(state.ds.dt || state.ds.st || state.bounds.enable)open_pass.depth_used=true;
    if(new_run){end_occlusion_query();begin_occlusion_query(c.command,occlusion,occlusion_index);}
    BindCache &bc=bind_cache;if(!bind_cache_enabled || bc.cmd!=c.command || pipeline.transient){bc=BindCache{};bc.cmd=c.command;}
    if(bc.pipe!=pipeline.pipeline){vkCmdBindPipeline(c.command,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline.pipeline);bc.pipe=pipeline.pipeline;}else ++binds_skipped;
    {const uint32_t no=uint32_t(d.dynamic_offsets.size());
     if(bc.layout!=d.pipeline_layout || bc.set!=d.set || bc.noff!=no || no>32 || (no && memcmp(bc.off,d.dynamic_offsets.data(),no*4))){
        ++set_binds;set_bind_offsets+=no;
        if(bc.layout==d.pipeline_layout && bc.set==d.set){++set_binds_offsets_only;
            if(bc.noff==no && no<=32){unsigned changed=0;for(uint32_t k=0;k<no;k++)if(bc.off[k]!=d.dynamic_offsets[k]){++changed;++offset_changed_at[k];}++offset_changes[changed<9?changed:9];}}
        vkCmdBindDescriptorSets(c.command,VK_PIPELINE_BIND_POINT_GRAPHICS,d.pipeline_layout,0,1,&d.set,no,d.dynamic_offsets.data());
        if(no<=32){bc.layout=d.pipeline_layout;bc.set=d.set;bc.noff=no;if(no)memcpy(bc.off,d.dynamic_offsets.data(),no*4);}else bc.noff=UINT32_MAX;}
     else ++binds_skipped;}
    VkViewport viewport{};need(xbg_nvn_viewport(&viewport,state.viewport.x,state.viewport.y,state.viewport.width,state.viewport.height*float(ui_vs),state.viewport.minDepth,state.viewport.maxDepth)==0,"Invalid NVN viewport");
    need(viewport.width<=properties.limits.maxViewportDimensions[0] && -viewport.height<=properties.limits.maxViewportDimensions[1] && viewport.x>=properties.limits.viewportBoundsRange[0] && viewport.x+viewport.width<=properties.limits.viewportBoundsRange[1] && viewport.y>=properties.limits.viewportBoundsRange[0] && viewport.y<=properties.limits.viewportBoundsRange[1] && viewport.y+viewport.height>=properties.limits.viewportBoundsRange[0],"NVN viewport exceeds device bounds");
    if(!bc.has_vp || memcmp(&bc.vp,&viewport,sizeof viewport)){vkCmdSetViewport(c.command,0,1,&viewport);bc.vp=viewport;bc.has_vp=true;}else ++binds_skipped;
    if(g_shadow_armed && open_pass.cmd==c.command && g_shadow_pw){float sx=float(g_shadow_sw)/float(g_shadow_pw),sy=float(g_shadow_sh)/float(g_shadow_ph);
        viewport.x*=sx;viewport.width*=sx;viewport.y*=sy;viewport.height*=sy;}
    VkRect2D sc=scissor(p);
    if(g_shadow_armed && open_pass.cmd==c.command && g_shadow_pw){float sx=float(g_shadow_sw)/float(g_shadow_pw),sy=float(g_shadow_sh)/float(g_shadow_ph);
        sc.offset.x=int32_t(sc.offset.x*sx);sc.offset.y=int32_t(sc.offset.y*sy);sc.extent.width=uint32_t(sc.extent.width*sx);sc.extent.height=uint32_t(sc.extent.height*sy);}
    // np647：**"填充 vs 顶点"判别探针**。`DM_PROBE_TINY_SCISSOR=<n>` 只对**没有颜色附件**的 pass
    //   （阴影/深度预 pass）把 scissor 压到左上 n×n，其它状态一律不动。判据：那一段 pass 的 GPU 时间
    //   塌下来 ⇒ 填充/overdraw 主导；几乎不变 ⇒ 顶点/驱动侧。默认 0 = 关，只用于 A/B。
    // np647: a fill-vs-vertex probe. DM_PROBE_TINY_SCISSOR=<n> clamps the scissor to the top-left n x n for
    //   passes with no color attachments (shadow / depth prepass) and changes nothing else. If that pass's GPU
    //   time collapses it is fill/overdraw bound; if it barely moves it is vertex/driver bound. 0 = off.
    if(g_probe_tiny_scissor>0 && p.color_slots==0){sc.offset={0,0};sc.extent={uint32_t(g_probe_tiny_scissor),uint32_t(g_probe_tiny_scissor)};}
    if(ui_uy>0 || ui_vs!=1){long so=g_ui_moff!=0?lround(ui_uy*p.height):0;int64_t y0=sc.offset.y<=0?0:sc.offset.y+so,y1=ui_vs!=1?int64_t(p.height):int64_t(sc.offset.y)+sc.extent.height+so;   // 剪裁跟着下移；从顶边起的不动（宿主上提的贴顶控件） | the scissor moves down with it; ones starting at the top edge stay (controls the host lifted to the top)
        if(y1>int64_t(p.height))y1=p.height;if(y1<y0)y1=y0;sc.offset.y=int32_t(y0);sc.extent.height=uint32_t(y1-y0);}
    if(!bc.has_sc || memcmp(&bc.sc,&sc,sizeof sc)){vkCmdSetScissor(c.command,0,1,&sc);bc.sc=sc;bc.has_sc=true;}else ++binds_skipped;
    for(auto &b:vertex_buffers){VkDeviceSize off=b.second->offset;if(b.first<16 && bc.vb[b.first]==b.second->buffer && bc.vbo[b.first]==off){++binds_skipped;continue;}
        vkCmdBindVertexBuffers(c.command,b.first,1,&b.second->buffer,&off);if(b.first<16){bc.vb[b.first]=b.second->buffer;bc.vbo[b.first]=off;}}
    pass_prof_draw(c.command);
    // np686：**绘制状态一次性体检**（DM_DRAW_STATE=1）。屏幕全黑而 clear 能上屏 ⇒ 先看 draw 提交时这些
    //   决定"会不会产生像素"的状态：通道掩码 / blend 开关 / 深度测试与写 / 裁剪 / 顶点数。
    //   掩码为 0、顶点数为 0、深度函数非法，都会让整屏一个像素都没有，而 clear 照样生效（clear 不吃掩码）。
    // np686: one-shot draw-state check. Mask 0 / zero vertices / bad depth func all give "no pixels at all"
    //   while clears still land (clears ignore the channel mask).
    {static int want=-1;if(want<0){const char *e=getenv("DM_DRAW_STATE");want=(e&&*e=='1')?0:-2;}
     if(want==0){static int n=0;if(n++<20)xbg_logf("[绘制] #%d prim=%d count=%d indexed=%d instances=%d mask=%#x blendEn=%#x dt=%d dw=%d df=%d st=%d cull=%d vp=%gx%g sc=%ux%u",
        n,prim,count,indexed?1:0,indexed?indexed->instances:instances,state.mask.mask,state.color.blend_en,int(state.ds.dt),int(state.ds.dw),int(state.ds.df),int(state.ds.st),state.poly.cull,state.viewport.width,state.viewport.height,unsigned(scissor(p).extent.width),unsigned(scissor(p).extent.height));}}
    // np687：**顶点/索引流 dump**（DM_VERTEX_DUMP=1）。合成 draw 是 6 索引全屏 quad、状态全对却一个像素都没有，
    //   最可能是顶点数据退化（位置全 0 / w=0 被裁掉）或索引错位。这里只打前 4 个"目标是窗口缓冲"的 draw。
    // np687: vertex/index stream dump. The composite draw is a 6-index fullscreen quad with all state correct yet
    //   writes nothing; the prime suspect is degenerate vertex data (positions 0, or w=0 -> clipped) or a bad
    //   index stream. Only the first 4 draws whose target is a window buffer.
    {static int vd=-1;if(vd<0){const char *e=getenv("DM_VERTEX_DUMP");vd=(e&&*e=='1')?0:-2;}
     if(vd==0 && p.images.n && win_serial_match(p.images.v[0]?p.images.v[0]->source.serial:0)){
        static int n=0;if(n++<4){
            xbg_logf("[顶点] #%d 目标serial=%llu %ux%u 流数=%zu count=%d indexed=%d prim=%d",n,
                (unsigned long long)p.images.v[0]->source.serial,p.width,p.height,vertex_buffers.size(),count,indexed?1:0,prim);
            for(size_t i=0;i<vertex_buffers.size() && i<3;i++){const Buffer *vb=vertex_buffers[i].second;if(!vb)continue;
                char hex[3*64+1]={0};size_t nb=vb->size<64?vb->size:64;const uint8_t *q=static_cast<const uint8_t*>(vb->mapped);
                for(size_t k=0;k<nb;k++)snprintf(hex+3*k,4,"%02x ",q[k]);
                xbg_logf("[顶点]   流%zu binding=%u size=%zu 前%zu字节: %s",i,vertex_buffers[i].first,vb->size,nb,hex);}
            if(state.shader[0])xbg_logf("[顶点]   顶点着色器 key=%s 声明输入=%zu 类=%zu 属性数=%zu",state.shader[0]->key.c_str(),state.shader[0]->inputs.size(),state.shader[0]->input_classes.size(),attributes.size());
            else xbg_logf("[顶点]   顶点着色器为空（属性数=%zu）",attributes.size());
            // np691: the vertex shader's uniform content. The vertex stream is a well-formed +-0.5 quad and every
            //   state bit is right, yet nothing is produced; cb_extent says this shader reads 80 bytes of slot 0.
            //   An all-zero CB (zero matrix/scale) collapses every vertex to the origin.
            for(size_t i=0;i<state.shader[0]->bindings.size() && i<6;i++){const xbg_shader_binding &b=state.shader[0]->bindings[i];
                if(b.source!=XBG_SOURCE_UNIFORM){xbg_logf("[顶点]   绑定#%zu binding=%u type=%u source=%u slot=%u bytes=%u",i,b.binding,b.type,b.source,b.slot,b.bytes);continue;}
                unsigned slot=b.slot;if(slot>=16){xbg_logf("[顶点]   绑定#%zu slot=%u out of range",i,slot);continue;}
                auto r=state.ub[nvn_stage[0]][slot];
                if(!r.address){xbg_logf("[顶点]   uniform slot=%u not set (address=0)",slot);continue;}
                // np691c: cb_extent says this shader reads bytes 0..79 (vec4 0..4) of vp_c3, and the disassembly
                //   confirms it only touches vec4 slots 1..4 (x1..x4 times the input x, plus .y/.z/.w of each).
                //   Print those five vec4s as floats: an all-zero block (or a degenerate rect) collapses the
                //   quad to a single point, which is exactly "state all correct, no pixels".
                const size_t need_bytes=5*16;
                if(r.size<need_bytes){xbg_logf("[顶点]   uniform slot=%u address=%p size=%zu (< %zu needed)",slot,(void*)r.address,size_t(r.size),need_bytes);continue;}
                const float *fb=reinterpret_cast<const float*>(r.address);
                for(unsigned e=0;e<5;e++)xbg_logf("[顶点]   vp_c3[%u] = %.6g %.6g %.6g %.6g",e,fb[e*4],fb[e*4+1],fb[e*4+2],fb[e*4+3]);}
            // np692: the shader's own name is vp_c3 (constant buffer 3) but its descriptor says slot 0, and the block
            //   we read from slot 0 gives an all-degenerate transform (every vertex ends up with y=0). Dump every
            //   vertex uniform slot with a valid range so we can see which one actually holds the blit transform.
            {static int shown=0;if(shown<2){++shown;
                xbg_logf("[顶点]   ---- 顶点 uniform 各槽（前 4 个 vec4）----");
                for(unsigned sl=0;sl<8;sl++){auto q=state.ub[nvn_stage[0]][sl];
                    if(!q.address){xbg_logf("[顶点]   槽%u 未设置",sl);continue;}
                    if(q.size<64){xbg_logf("[顶点]   槽%u addr=%p size=%zu (<64)",sl,(void*)q.address,size_t(q.size));continue;}
                    const float *fq=reinterpret_cast<const float*>(q.address);
                    xbg_logf("[顶点]   槽%u addr=%p size=%zu: [%.6g %.6g %.6g %.6g] [%.6g %.6g %.6g %.6g] [%.6g %.6g %.6g %.6g] [%.6g %.6g %.6g %.6g]",
                        sl,(void*)q.address,size_t(q.size),
                        fq[0],fq[1],fq[2],fq[3],fq[4],fq[5],fq[6],fq[7],
                        fq[8],fq[9],fq[10],fq[11],fq[12],fq[13],fq[14],fq[15]);}}}
            for(size_t i=0;i<bindings.size() && i<4;i++)xbg_logf("[顶点]   绑定%zu binding=%u stride=%u rate=%d",i,bindings[i].binding,bindings[i].stride,int(bindings[i].inputRate));
            for(size_t i=0;i<attributes.size() && i<8;i++)xbg_logf("[顶点]   属性%zu loc=%u binding=%u fmt=%d off=%u",i,attributes[i].location,attributes[i].binding,int(attributes[i].format),attributes[i].offset);
            if(ib){char hex[3*12+1]={0};size_t nb=ib->size<12?ib->size:12;const uint8_t *q=static_cast<const uint8_t*>(ib->mapped);
                for(size_t k=0;k<nb;k++)snprintf(hex+3*k,4,"%02x ",q[k]);
                xbg_logf("[顶点]   索引 size=%zu 前%zu字节: %s",ib->size,nb,hex);}
        }}}
    if(indexed){if(bc.ib!=ib->buffer || bc.ibo!=ib->offset || bc.it!=index_type){vkCmdBindIndexBuffer(c.command,ib->buffer,ib->offset,index_type);bc.ib=ib->buffer;bc.ibo=ib->offset;bc.it=index_type;}else ++binds_skipped;vkCmdDrawIndexed(c.command,count,indexed->instances,0,indexed->base_vertex,indexed->base_instance);}else vkCmdDraw(c.command,count,instances,first,first_instance);
    if(!occlusion_batch_enabled)end_occlusion_query();c.finish();++perf.draws;++pass_prof_draws;pass_prof_verts+=uint64_t(uint32_t(count))*instances;deathguard::draw_bound();
    // np629：提交二分的切点。只在 pass 边界切（本 draw 刚 finish、若无 open pass 就是边界）；
    // N=0 关闭。切点动作在 submit_split_segment()（见 submit_slots 上方）。
    // np629: the cut point. Only at pass boundaries (this draw just finished; no open pass = boundary).
    {int N=split_draws_env();
     if(N>0 && cur_digest.draws && cur_digest.passes>=50){   // np629b：门槛 100→50。np611 读数 43k draw≈546 pass，
        // 到第 8000 draw 时 passes≈97——差 3 个不到 100，门槛把所有切点全挡住了（首跑 0 段的根因）。
        ++split_draw_count;
        // np629c：真机读数（二分 到达 8000 draw：open_pass=0x1 连续多次）——大提交里有一个**超长 render pass**，
        // pass 边界几乎不存在，按"只在边界切"永远切不开。改为：随时可切；若 pass 开着，先 end_open_pass()
        // （状态由引擎的 bind cache 在下一个 draw 重放，语义等价于"驱动多看到一次 pass 间隔"）。
        // np629c: device reads show one huge render pass inside the big submission (open_pass stays set across
        // thousands of draws), so pass-boundary-only cuts never fire. Cut anywhere: end the open pass first
        // (the bind cache replays state on the next draw; the driver just sees one extra pass gap).
        if(split_draw_count>=unsigned(N)){submit_split_segment();split_draw_count=0;}
     }}
    if(d.needs_writeback() || capture.active){unsigned saved=cur_op;cur_op=62;end_pass(6);g_barrier_site="writeback";full_barrier(c.command);cur_op=saved;VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);}
    {DRAW_SCOPE(Writeback);if(capture.active){if(d.needs_writeback())gpu_sync(perf.sync_writeback);d.writeback();}else d.defer_writeback();}stats.draws.fetch_add(1,std::memory_order_relaxed);   // diagnostics only: a seq_cst add (LDADDAL) waits for every earlier store, ~2.7% of the render thread
    // np670：绘制落点记账（见 g_draws_since_present 的注释）| np670: draw landing bookkeeping
    {Image *col=open_pass.images.n?open_pass.images.v[0]:nullptr;++g_draws_since_present;
     if(col){g_last_draw_serial=col->source.serial;g_last_draw_image=(void*)col->image;if(g_present_serial_hint && col->source.serial==g_present_serial_hint)++g_draws_to_present_serial;}}
    if(capture.active){capture_pass(p);d.capture_read_images();capture.active=false;}
}
// 3D 画面加亮（iOS dm/gfx/nvn/nvn_mtl.m bright_lut 的 Vulkan 版；#112 vickwv fe7b655 在 Android 上做过同样的移植）：游戏每帧用
// 计算着色器 552f45c348a61b86 算色调映射的 32^3 调色表；那次派发之后把表拷一份，用 xbg_bright_lut.comp 按宿主 dm_bright_k() 给的
// 曝光 k 原地重写。k ≈ 1（局外、正午）不动。只碰 3D 世界：UI、文字、血球在色调映射之后才合成。
// 开关 gfx-bright / DM_GFX_BRIGHT（缺省开），A/B 目标 bright；DM_BRIGHT_K=<k>（宿主）固定曝光。
// 3D picture brightening (the Vulkan version of bright_lut in iOS dm/gfx/nvn/nvn_mtl.m; #112, vickwv fe7b655, ported the same idea
// to Android): every frame the game computes tone mapping's 32^3 color table with compute shader 552f45c348a61b86; after that
// dispatch the table is copied and rewritten in place by xbg_bright_lut.comp with the exposure k from the host's dm_bright_k().
// Nothing changes at k ~= 1 (out of game, noon). Only the 3D world changes: UI, text and the orbs are composited after tone mapping.
// Switch gfx-bright / DM_GFX_BRIGHT (on by default), A/B target bright; DM_BRIGHT_K=<k> (host) fixes the exposure.
namespace bright {
bool enabled=true,failed=false;uint64_t applied=0;float last_k=1;
VkDescriptorSetLayout set_layout{};VkPipelineLayout layout{};VkPipeline pipeline{};VkSampler sampler{};VkDescriptorPool pool{};
// 每个帧槽 8 套描述符：一帧里可能派发不止一次，录进命令缓冲的套在提交完成前不能再改
// 8 descriptor sets per frame slot: a frame may dispatch more than once, and a set recorded into a command buffer must not change before that submission completes
constexpr unsigned SETS=8;VkDescriptorSet sets[2][SETS]{};unsigned used[2]{};uint64_t used_serial[2]{};
VkImage tmp{};VkDeviceMemory tmp_memory{};VkImageView tmp_view{};VkFormat tmp_format{};uint32_t tmp_w=0,tmp_h=0,tmp_d=0;bool tmp_general=false;
void shutdown(){
    if(tmp_view)vkDestroyImageView(cfg.device,tmp_view,nullptr);if(tmp)vkDestroyImage(cfg.device,tmp,nullptr);if(tmp_memory)vkFreeMemory(cfg.device,tmp_memory,nullptr);
    if(pipeline)vkDestroyPipeline(cfg.device,pipeline,nullptr);if(layout)vkDestroyPipelineLayout(cfg.device,layout,nullptr);if(pool)vkDestroyDescriptorPool(cfg.device,pool,nullptr);
    if(set_layout)vkDestroyDescriptorSetLayout(cfg.device,set_layout,nullptr);if(sampler)vkDestroySampler(cfg.device,sampler,nullptr);
    tmp_view={};tmp={};tmp_memory={};tmp_w=tmp_h=tmp_d=0;tmp_general=false;pipeline={};layout={};pool={};set_layout={};sampler={};for(auto &s:sets)for(auto &x:s)x={};used[0]=used[1]=0;used_serial[0]=used_serial[1]=0;failed=false;
}
bool prepare(){
    if(pipeline)return true;
    if(!cfg.enabled_core.shaderStorageImageWriteWithoutFormat){xbg_logf("NVN bright: off (shaderStorageImageWriteWithoutFormat not enabled)");return false;}
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};si.magFilter=si.minFilter=VK_FILTER_LINEAR;si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK(vkCreateSampler(cfg.device,&si,nullptr,&sampler));
    VkDescriptorSetLayoutBinding b[2]{};b[0]={0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};b[1]={1,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=2;dl.pBindings=b;VK(vkCreateDescriptorSetLayout(cfg.device,&dl,nullptr,&set_layout));
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,4};VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.setLayoutCount=1;li.pSetLayouts=&set_layout;li.pushConstantRangeCount=1;li.pPushConstantRanges=&push;
    VK(vkCreatePipelineLayout(cfg.device,&li,nullptr,&layout));
    VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,2*SETS},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,2*SETS}};VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pi.maxSets=2*SETS;pi.poolSizeCount=2;pi.pPoolSizes=sizes;
    VK(vkCreateDescriptorPool(cfg.device,&pi,nullptr,&pool));
    VkDescriptorSetLayout layouts[2*SETS];for(auto &l:layouts)l=set_layout;VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=pool;ai.descriptorSetCount=2*SETS;ai.pSetLayouts=layouts;
    VK(vkAllocateDescriptorSets(cfg.device,&ai,&sets[0][0]));
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=sizeof(xbg_bright_lut_cs);mi.pCode=xbg_bright_lut_cs;VkShaderModule module{};VK(vkCreateShaderModule(cfg.device,&mi,nullptr,&module));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";ci.layout=layout;
    VkResult r;{deathguard::VkSite site(deathguard::PIPE_OTHER);r=vkCreateComputePipelines(cfg.device,pipeline_cache,1,&ci,nullptr,&pipeline);}vkDestroyShaderModule(cfg.device,module,nullptr);
    if(r!=VK_SUCCESS){pipeline={};xbg_logf("NVN bright: off (compute pipeline result=%d)",int(r));return false;}
    xbg_logf("NVN bright: ready (color table rewrite, exposure from dm_bright_k)");return true;
}
bool prepare_tmp(const Image &lut){
    if(tmp && tmp_format==lut.fmt && tmp_w==lut.width && tmp_h==lut.height && tmp_d==lut.depth)return true;
    if(tmp){VkImage i=tmp;VkImageView v=tmp_view;VkDeviceMemory m=tmp_memory;retire_handle([i,v,m]{vkDestroyImageView(cfg.device,v,nullptr);vkDestroyImage(cfg.device,i,nullptr);vkFreeMemory(cfg.device,m,nullptr);});tmp={};tmp_view={};tmp_memory={};}
    VkFormatProperties fp{};vkGetPhysicalDeviceFormatProperties(cfg.physical,lut.fmt,&fp);
    if(!(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)){xbg_logf("NVN bright: off (color table format %d cannot be sampled linearly)",int(lut.fmt));return false;}
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};ic.imageType=VK_IMAGE_TYPE_3D;ic.format=lut.fmt;ic.extent={lut.width,lut.height,lut.depth};ic.mipLevels=1;ic.arrayLayers=1;ic.samples=VK_SAMPLE_COUNT_1_BIT;
    ic.tiling=VK_IMAGE_TILING_OPTIMAL;ic.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;ic.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
    if(vkCreateImage(cfg.device,&ic,nullptr,&tmp)!=VK_SUCCESS){tmp={};return false;}
    VkMemoryRequirements mr;vkGetImageMemoryRequirements(cfg.device,tmp,&mr);VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ma.allocationSize=mr.size;ma.memoryTypeIndex=memory_type(mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};vi.image=tmp;vi.viewType=VK_IMAGE_VIEW_TYPE_3D;vi.format=lut.fmt;vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    if(vkAllocateMemory(cfg.device,&ma,nullptr,&tmp_memory)!=VK_SUCCESS || vkBindImageMemory(cfg.device,tmp,tmp_memory,0)!=VK_SUCCESS || vkCreateImageView(cfg.device,&vi,nullptr,&tmp_view)!=VK_SUCCESS){
        if(tmp_view)vkDestroyImageView(cfg.device,tmp_view,nullptr);vkDestroyImage(cfg.device,tmp,nullptr);if(tmp_memory)vkFreeMemory(cfg.device,tmp_memory,nullptr);tmp={};tmp_view={};tmp_memory={};return false;}
    tmp_format=lut.fmt;tmp_w=lut.width;tmp_h=lut.height;tmp_d=lut.depth;tmp_general=false;
    xbg_logf("NVN bright: color table %ux%ux%u format=%d",tmp_w,tmp_h,tmp_d,int(tmp_format));return true;
}
// 计算着色器在储存图像槽上绑的那张图（调色表）；认不出返回 nullptr
// The image bound at the compute shader's storage image slot (the color table); nullptr when it cannot be identified
Image *bound_storage_image(){
    for(const auto &b:state.shader[5]->bindings)if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && b.source==XBG_SOURCE_IMAGE && b.slot<16){
        int id=int(state.img[5][b.slot]&0xfffff);if(!id || (state.texture_pool.n>0 && id>=state.texture_pool.n))return nullptr;
        auto pool=textures.find(pool_key(&state.texture_pool));auto &table=pool==textures.end()?legacy_textures:pool->second;auto tex=table.find(id);if(tex==table.end())return nullptr;
        return &get_image(&tex->second.texture,true);}
    return nullptr;
}
// 在调色表那次派发之后、同一个非绘制命令段里调用（前后都有全屏障）
// Called after the color table's dispatch, inside the same non-draw command span (full barriers before and after)
void after_dispatch(Commands &c){
    if(!enabled || failed || !dm_bright_k || state.shader[5]->key.compare(0,16,"552f45c348a61b86") || ab_off("bright"))return;
    float k=dm_bright_k();last_k=k;if(!(k>1.005f))return;if(k>4.0f)k=4.0f;
    Image *lut=bound_storage_image();
    if(!lut || lut->depth<2 || lut->levels!=1 || lut->aspects!=VK_IMAGE_ASPECT_COLOR_BIT || !(lut->usage&VK_IMAGE_USAGE_STORAGE_BIT) || !(lut->usage&VK_IMAGE_USAGE_TRANSFER_SRC_BIT)){
        static bool told=false;if(!told){told=true;xbg_logf("NVN bright: color table not recognised (image=%p), not brightening",(void*)lut);}return;}
    if(!prepare() || !prepare_tmp(*lut)){failed=true;return;}
    unsigned si=unsigned(&recording_slot()-slots)&1;if(used_serial[si]!=slot_serial){used_serial[si]=slot_serial;used[si]=0;}
    if(used[si]>=SETS)return;VkDescriptorSet set=sets[si][used[si]++];
    VkCommandBuffer cmd=c.command;
    if(lut->layout!=VK_IMAGE_LAYOUT_GENERAL)lut->transition(cmd,VK_IMAGE_LAYOUT_GENERAL);
    if(!tmp_general){VkImageMemoryBarrier t{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};t.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;t.newLayout=VK_IMAGE_LAYOUT_GENERAL;t.image=tmp;t.srcQueueFamilyIndex=t.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        t.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};t.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&t);tmp_general=true;}
    g_barrier_site="bright-copy";full_barrier(cmd);
    VkImageCopy region{};region.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};region.dstSubresource=region.srcSubresource;region.extent={lut->width,lut->height,lut->depth};
    vkCmdCopyImage(cmd,lut->image,VK_IMAGE_LAYOUT_GENERAL,tmp,VK_IMAGE_LAYOUT_GENERAL,1,&region);
    g_barrier_site="bright-postcopy";full_barrier(cmd);
    VkDescriptorImageInfo ii[2]{};ii[0]={sampler,tmp_view,VK_IMAGE_LAYOUT_GENERAL};ii[1]={VK_NULL_HANDLE,lut->view(nullptr,false,true),VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2]{};for(int i=0;i<2;i++){w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[i].dstSet=set;w[i].dstBinding=uint32_t(i);w[i].descriptorCount=1;w[i].descriptorType=i?VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w[i].pImageInfo=&ii[i];}
    vkUpdateDescriptorSets(cfg.device,2,w,0,nullptr);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&k);
    vkCmdDispatch(cmd,(lut->width+3)/4,(lut->height+3)/4,(lut->depth+3)/4);
    g_barrier_site="bright-postdispatch";full_barrier(cmd);
    ++applied;
}
}
void dispatch(const p_dispatch &p){g_gpu_wrote=true;preflight_cpu_reads();CPU_SCOPE(Dispatch);++cur_digest.dispatches;provenance_log("DISPATCH shader=%s",state.shader[5]?state.shader[5]->key.c_str():"null");
    // np608：**和 draw 的 np598 是同一条**。缺包跳过之后 `g_shader_skip_abort` 每帧重置，而绑定的程序
    // 还留着、下一帧游戏没重绑 ⇒ 这里 state.shader[5] 仍为空，于是 dispatch 以另一种方式 FATAL。
    // 铁律 9：开一个降级模式就要把消费侧**所有**分支改完——draw 改过了（丢无 VS 的 draw），dispatch 这里漏了。
    // np608: exactly the same shape as np598 for draws. After a missing-package skip, g_shader_skip_abort is
    // reset every frame while the bound program stays, so on the next frame state.shader[5] is still null and the
    // dispatch FATALs instead. Rule 9: turning on a degradation mode means finishing every consumer -- draws were
    // done (drop the draw without a VS), compute was missed.
    if(!state.shader[5] && shader_skip_on()){static int n;if(n++<4)xbg_logf("【诊断】dispatch 没有计算着色器（缺包跳过所致）⇒ 丢掉这一条（第 %d 次）",n);return;}
    need(state.shader[5],"Dispatch has no genuine compute shader");need(p.x>=0 && p.y>=0 && p.z>=0,"Negative compute dispatch");if(!p.x||!p.y||!p.z)return;
    // np630b：对照实验 ②——按 shader 键禁用 compute（DM_SKIP_COMPUTE=<16hex>，逗号分隔可多个）。
    // 用途：np629b 的读数（被拒提交 dispatch 恒在）指向 compute→图形依赖或该 compute 本身；
    // 跳过它跑一轮，device lost 消失 = 嫌疑定谳。画面会缺色调映射（变灰/变暗），只用于对照。
    // np630b: experiment 2 -- skip specific compute shaders by key (DM_SKIP_COMPUTE=<16hex>[,…]).
    // If the device loss disappears with the compute skipped, the suspicion is confirmed; the picture loses
    // tone mapping (dimmer/grayer) -- diagnostics only.
    {static std::vector<std::string> skip_keys;static bool inited=false;
     if(!inited){inited=true;if(const char *e=getenv("DM_SKIP_COMPUTE")){const char *p2=e;while(*p2){const char *q=strchr(p2,',');size_t len=q?q-p2:strlen(p2);if(len==16)skip_keys.push_back(std::string(p2,len));p2=q?q+1:p2+len;}}
         if(!skip_keys.empty()){std::string k;for(auto &x:skip_keys)k+=" "+x;xbg_logf("【对照】compute 禁用清单:%s",k.c_str());}}
     if(!skip_keys.empty())for(auto &k:skip_keys)if(k==state.shader[5]->key){static int n;if(n++<4)xbg_logf("【对照】跳过 compute %s（DM_SKIP_COMPUTE）",k.c_str());return;}}
    if(pipeguard::skipped(pipeguard::compute_key(state.shader[5]->key)))return;need(uint32_t(p.x)<=properties.limits.maxComputeWorkGroupCount[0] && uint32_t(p.y)<=properties.limits.maxComputeWorkGroupCount[1] && uint32_t(p.z)<=properties.limits.maxComputeWorkGroupCount[2],"Compute dispatch exceeds device limit");Commands c;Descriptors &d=acquire_descriptors();descriptors(d,c,true);VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=state.shader[5]->module;ci.stage.pName="main";ci.layout=d.pipeline_layout;PipelineLease pipeline;{CPU_SCOPE(Pipeline);XbgPipelineKey key;bool cacheable=xbg_compute_pipeline_key(key,ci,d.entry->definition.view());pipeline_lookup(pipeline,key,cacheable,nullptr,&ci);}if(!pipeline.pipeline){c.finish();return;}vkCmdBindPipeline(c.command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline.pipeline);vkCmdBindDescriptorSets(c.command,VK_PIPELINE_BIND_POINT_COMPUTE,d.pipeline_layout,0,1,&d.set,uint32_t(d.dynamic_offsets.size()),d.dynamic_offsets.data());int prof_dispatch=pass_prof_compute_begin(c.command,p);vkCmdDispatch(c.command,p.x,p.y,p.z);pass_prof_complete(c.command,prof_dispatch);bright::after_dispatch(c);
    // np630：dispatch 后屏障对照实验。默认（窄）：COMPUTE_SHADER_WRITE → HOST_READ，只够 host 读回；
    // DM_BARRIER_FULL=1（对照）：ALL_COMMANDS × 全内存读写 —— 把"compute 写 → 后续图形读"这条依赖
    // 也盖住。NVN 层游戏假定 dispatch 结果立即可用（它的同步在 NVN 语义里），我们原来只给了窄屏障；
    // np629b 的读数（被拒提交 dispatch 恒在）指向这里。判据：开着跑，device lost 消失 = 定谳。
    {static int full_barrier_env=-1;if(full_barrier_env<0){const char *e=getenv("DM_BARRIER_FULL");full_barrier_env=(e&&*e=='1')?1:0;}
     VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
     VkPipelineStageFlags src=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,dst=VK_PIPELINE_STAGE_HOST_BIT;
     if(full_barrier_env){b.srcAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;src=dst=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
         static int n;if(n++<3)xbg_logf("【对照】dispatch 后全内存屏障已启用（DM_BARRIER_FULL=1）");}
     else{b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_HOST_READ_BIT;}
     vkCmdPipelineBarrier(c.command,src,dst,0,1,&b,0,nullptr,0,nullptr);}
    c.finish();++perf.dispatches;d.defer_writeback();++stats.dispatches;}

struct MaskedClear {
    VkShaderModule vs{},fs{};VkPipelineLayout layout{};VkPipeline pipeline{};
    ~MaskedClear(){if(pipeline)vkDestroyPipeline(cfg.device,pipeline,nullptr);if(layout)vkDestroyPipelineLayout(cfg.device,layout,nullptr);if(fs)vkDestroyShaderModule(cfg.device,fs,nullptr);if(vs)vkDestroyShaderModule(cfg.device,vs,nullptr);}
    void prepare(const Pass &p,int attachment,int scalar,uint32_t cm,uint32_t sm,uint32_t reference){
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};si.codeSize=sizeof(xbg_clear_vs);si.pCode=xbg_clear_vs;VK(vkCreateShaderModule(cfg.device,&si,nullptr,&vs));
        if(cm){const uint32_t *words=scalar==1?xbg_clear_fs_sint:scalar==2?xbg_clear_fs_uint:xbg_clear_fs_float;size_t bytes=scalar==1?sizeof(xbg_clear_fs_sint):scalar==2?sizeof(xbg_clear_fs_uint):sizeof(xbg_clear_fs_float);std::vector<uint32_t> patched(words,words+bytes/4);patched[xbg_clear_fs_location_word]=uint32_t(attachment);si.codeSize=bytes;si.pCode=patched.data();VK(vkCreateShaderModule(cfg.device,&si,nullptr,&fs));}
        VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT,0,16};VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.pushConstantRangeCount=cm?1:0;li.pPushConstantRanges=cm?&push:nullptr;VK(vkCreatePipelineLayout(cfg.device,&li,nullptr,&layout));
        VkPipelineShaderStageCreateInfo stages[2]{};for(int i=0;i<(cm?2:1);i++){stages[i].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;stages[i].stage=i?VK_SHADER_STAGE_FRAGMENT_BIT:VK_SHADER_STAGE_VERTEX_BIT;stages[i].module=i?fs:vs;stages[i].pName="main";}
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vp.viewportCount=vp.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.polygonMode=VK_POLYGON_MODE_FILL;rs.cullMode=VK_CULL_MODE_NONE;rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;rs.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};ms.rasterizationSamples=p.samples;
        VkPipelineColorBlendAttachmentState color{};VkPipelineDepthStencilStateCreateInfo ds{};need(xbg_masked_clear_state(cm,sm,reference,&color,&ds)==0,"Invalid masked clear state");
        // Every other subpass color slot is UNUSED. Identical states across all
        // slots avoid any dependency on the optional independentBlend feature.
        std::vector<VkPipelineColorBlendAttachmentState> colors(p.color_slots,color);VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};blend.attachmentCount=colors.size();blend.pAttachments=colors.data();
        VkDynamicState dyn[2]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dynamic.dynamicStateCount=2;dynamic.pDynamicStates=dyn;
        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};ci.stageCount=cm?2:1;ci.pStages=stages;ci.pVertexInputState=&vi;ci.pInputAssemblyState=&ia;ci.pViewportState=&vp;ci.pRasterizationState=&rs;ci.pMultisampleState=&ms;ci.pDepthStencilState=&ds;ci.pColorBlendState=&blend;ci.pDynamicState=&dynamic;ci.layout=layout;ci.renderPass=p.pass;
        {deathguard::VkSite site(deathguard::PIPE_OTHER);VK(vkCreateGraphicsPipelines(cfg.device,pipeline_cache,1,&ci,nullptr,&pipeline));}
    }
    void encode(Commands &c,const Pass &p,const VkRect2D &rect,const void *color){
        bind_cache_reset();VkViewport viewport{0,0,float(p.width),float(p.height),0,1};vkCmdBindPipeline(c.command,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);vkCmdSetViewport(c.command,0,1,&viewport);vkCmdSetScissor(c.command,0,1,&rect);
        if(color)vkCmdPushConstants(c.command,layout,VK_SHADER_STAGE_FRAGMENT_BIT,0,16,color);vkCmdDraw(c.command,3,1,0,0);
    }
};
void clear_attachment(const p_clear_color *color,const p_clear_ds *depth,int scalar=0){preflight_cpu_reads();
    VkClearAttachment a{};uint32_t cm=0,sm=0;bool partial=false;p_clear_color converted{};
    if(color){
        need(color->index>=0 && color->index<state.rt.n && state.rt.color[color->index],"Clear color attachment missing");need((uint32_t(color->mask)&~15u)==0,"Invalid color clear mask");cm=uint32_t(color->mask);
        Image &target=get_image(state.rt.color[color->index],false);const view_t *view=state.rt.has_cview[color->index]?&state.rt.cview[color->index]:nullptr;
        VkFormat viewed=target.view_format(view);static unsigned notes=0;if(notes++<8){uint32_t bits[4];memcpy(bits,color->rgba,16);xbg_logf("NVN clear-value attachment input=%d nvn=%#x view=%#x vk=%u mask=%#x bits=%08x/%08x/%08x/%08x",scalar,target.source.d.format,view&&view->set_format?view->format:target.source.d.format,unsigned(viewed),cm,bits[0],bits[1],bits[2],bits[3]);}VkClearColorValue value{};int converted_scalar=xbg_clear_value(viewed,scalar,color->rgba,&value);
        if(converted_scalar<0)xbg_fatalf("Color clear conversion failed nvn=%#x vk=%u input=%d rc=%d",target.source.d.format,unsigned(viewed),scalar,converted_scalar);
        converted=*color;memcpy(converted.rgba,&value,16);color=&converted;scalar=converted_scalar;
        if(cm)a.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;a.colorAttachment=color->index;memcpy(&a.clearValue.color,color->rgba,16);partial=cm && cm!=15;
    }else{
        need(state.rt.depth,"Clear depth attachment missing");sm=uint32_t(depth->stencil_mask)&255;
        VkImageAspectFlags requested=(depth->depth_mask?VK_IMAGE_ASPECT_DEPTH_BIT:0)|(sm?VK_IMAGE_ASPECT_STENCIL_BIT:0);need((aspect(state.rt.depth->d.format)&requested)==requested,"Clear requests absent depth/stencil aspect");
        if(depth->depth_mask){need(depth->depth>=0 && depth->depth<=1,"Depth clear value outside normalized range");a.aspectMask|=VK_IMAGE_ASPECT_DEPTH_BIT;}
        partial=sm && sm!=255;if(sm && !partial)a.aspectMask|=VK_IMAGE_ASPECT_STENCIL_BIT;a.clearValue.depthStencil={depth->depth,uint32_t(depth->stencil)&255};
    }
    if(!a.aspectMask && !partial)return;
    // Full (unmasked) clears are recorded inside the draw render pass of the
    // current targets, so a clear and the draws after it share one pass.
    if(!partial && a.aspectMask && !capture.trace && !provenance_file){
        const tex_t *t0=color?state.rt.color[color->index]:state.rt.depth;Image &ti=get_image(t0,false);
        const view_t *tv=color?(state.rt.has_cview[color->index]?&state.rt.cview[color->index]:nullptr):(state.rt.has_dview?&state.rt.dview:nullptr);
        uint32_t lvl=tv && tv->set_levels?tv->base_level:0;uint32_t cw=std::max(1u,ti.width>>lvl),ch=std::max(1u,ti.height>>lvl);VkRect2D keep{};
        bool needs_preserve=lvl<ti.levels && xbg_clear_rectangle(cw,ch,state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,&keep)==0 && xbg_clear_requires_preserve(ti.levels,ti.layers,cw,ch,&keep,false,ti.aspects,a.aspectMask)!=0;
        if(!needs_preserve || ti.gpu_written || ti.uploaded){
            Commands c(true);Pass p;
            if(create_pass(p,c,-1,false,true)){
                if(open_pass.cmd!=c.command || open_pass.pass!=p.pass || open_pass.framebuffer!=p.framebuffer){++pass_open_by[!open_pass.cmd?0:(open_pass.cmd!=c.command?1:(open_pass.pass!=p.pass?2:3))];if(p.framebuffer)pass_open_fbs.insert(uint64_t(uintptr_t(p.framebuffer)));end_pass(7);begin_pass(p,c);open_pass.cmd=c.command;open_pass.pass=p.pass;open_pass.framebuffer=p.framebuffer;open_pass.images=p.images;open_pass.serial0=p.images.n&&p.images.v[0]?p.images.v[0]->source.serial:0;open_pass.depth_attached=state.rt.depth!=nullptr;++perf.passes;}
                if(depth)open_pass.depth_used=true;
                VkClearRect r{scissor(p),0,1};if(r.rect.extent.width && r.rect.extent.height)vkCmdClearAttachments(c.command,1,&a,1,&r);
                ++perf.folded_clears;return;
            }
        }
    }
    if(capture_begin(color?"clear-color":"clear-depth-stencil",stats.presents))capture_log("CLEAR target=%d scalar=%d colorMask=%u stencilMask=%u partial=%u",color?color->index:-1,scalar,cm,sm,unsigned(partial));
    const tex_t *target=color?state.rt.color[color->index]:state.rt.depth;Image &target_image=get_image(target,false);
    const view_t *target_view=color?(state.rt.has_cview[color->index]?&state.rt.cview[color->index]:nullptr):(state.rt.has_dview?&state.rt.dview:nullptr);
    uint32_t clear_level=target_view && target_view->set_levels?target_view->base_level:0;need(clear_level<target_image.levels,"Clear mip outside image");
    uint32_t clear_width=std::max(1u,target_image.width>>clear_level),clear_height=std::max(1u,target_image.height>>clear_level);VkRect2D preserve_rect{};
    need(xbg_clear_rectangle(clear_width,clear_height,state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,&preserve_rect)==0,"Invalid clear scissor");
    bool preserve=xbg_clear_requires_preserve(target_image.levels,target_image.layers,clear_width,clear_height,&preserve_rect,partial,target_image.aspects,a.aspectMask)!=0;
    // LOAD preserves actual previous contents only after importing CPU-backed
    // storage on first use; never mark gpu_written before this preservation.
    if(preserve){need(target_image.samples==VK_SAMPLE_COUNT_1_BIT || target_image.gpu_written,"Partial clear of uninitialized CPU multisample storage unsupported");get_image(target,true);}
    // 后端合成的 masked-clear draw 不属于游戏遮挡计数；非 draw 的 Commands 随后也结束 pass。
    // Backend-generated masked-clear draws do not contribute to the game's occlusion count; the non-draw Commands then ends the pass.
    end_occlusion_query();
    Commands c;Pass p;create_pass(p,c,color?color->index:-1,depth!=nullptr);VkClearRect r{scissor(p),0,1};auto masked_object=std::make_shared<MaskedClear>();MaskedClear &masked=*masked_object;retire_object(masked_object);
    bool area=r.rect.extent.width && r.rect.extent.height;
    if(partial && area)masked.prepare(p,color?color->index:-1,scalar,cm,color?0:sm,color?0:uint32_t(depth->stencil));
    begin_pass(p,c);
    if(area){
        if(!color && a.aspectMask)vkCmdClearAttachments(c.command,1,&a,1,&r);
        if(partial)masked.encode(c,p,r.rect,color?color->rgba:nullptr);
        else if(color)vkCmdClearAttachments(c.command,1,&a,1,&r);
    }
    vkCmdEndRenderPass(c.command);pass_prof_end(c.command);c.finish(); // transient pipeline/modules stay alive through this fence
    if(capture.active){capture_pass(p);capture.active=false;}
}
void region_check(const Image &im,uint32_t level,uint32_t layer,const NVNcopyRegion &r){
    need(level<im.levels && layer<im.layers && r.xoffset>=0 && r.yoffset>=0 && r.zoffset>=0 && r.width>0 && r.height>0 && r.depth>0,"Invalid texture copy region");
    bool volume=im.source.d.target==2;uint64_t zlimit=volume?std::max(1u,im.depth>>level):im.layers;
    need(uint64_t(r.xoffset)+r.width<=std::max(1u,im.width>>level) && uint64_t(r.yoffset)+r.height<=std::max(1u,im.height>>level) && uint64_t(volume?0:layer)+r.zoffset+r.depth<=zlimit,"Texture copy region exceeds image");
}
VkImageSubresourceLayers copy_layers(const Image &im,uint32_t level,uint32_t layer,const NVNcopyRegion &r){return {im.aspects,level,im.source.d.target==2?0:layer+uint32_t(r.zoffset),im.source.d.target==2?1:uint32_t(r.depth)};}
VkOffset3D copy_offset(const Image &im,const NVNcopyRegion &r){return {r.xoffset,r.yoffset,im.source.d.target==2?r.zoffset:0};}
VkExtent3D copy_extent(const Image &im,const NVNcopyRegion &r){return {uint32_t(r.width),uint32_t(r.height),im.source.d.target==2?uint32_t(r.depth):1};}
// Buffer->texture copies write the texture's pool storage on Switch. Mirror that
// into guest memory (block-linear) so the texture stays CPU-authoritative: later
// CPU streaming writes into the same storage keep being detected and uploaded.
bool guest_texture_write(const tex_t *t,unsigned level,unsigned layer,const NVNcopyRegion &r,const uint8_t *src,size_t row,size_t image_stride){
    uint32_t row_bytes,rows,slices,bpb;uint64_t base;unsigned th,td;
    if(!t->mem || r.width<=0 || r.height<=0 || r.depth<=0 || r.xoffset<0 || r.yoffset<0 || r.zoffset<0)return false;
    if(nvn_deswizzle_geom(&t->d,level,layer,&row_bytes,&rows,&slices,&base,&th,&td,&bpb))return false;
    int bw=1,bh=1;if(!nvn_fmt_bpb(t->d.format,&bw,&bh) || r.xoffset%bw || r.yoffset%bh)return false;
    uint32_t bx0=r.xoffset/bw,by0=r.yoffset/bh,nbx=(r.width+bw-1)/bw,nby=(r.height+bh-1)/bh;
    if(uint64_t(bx0+nbx)*bpb>row_bytes || by0+nby>rows || row<size_t(nbx)*bpb)return false;
    bool volume=t->d.target==2;if(volume?uint32_t(r.zoffset+r.depth)>slices:layer+uint32_t(r.zoffset+r.depth)>t->L.layers)return false;
    for(int k=0;k<r.depth;k++){
        uint64_t b=base;uint32_t z=0;
        if(volume)z=uint32_t(r.zoffset+k);
        else{uint32_t a,b2,c,d2;uint64_t lb;unsigned h1,h2;if(nvn_deswizzle_geom(&t->d,level,layer+uint32_t(r.zoffset+k),&a,&b2,&c,&lb,&h1,&h2,&d2))return false;b=lb;}
        for(uint32_t y=0;y<nby;y++)for(uint32_t x=0;x<nbx;x++){
            uint64_t at=b+nvn_bl_offset((bx0+x)*bpb,by0+y,z,row_bytes,rows,th,td);if(at+bpb>t->L.size)return false;
            memcpy(t->mem+at,src+size_t(k)*image_stride+size_t(y)*row+size_t(x)*bpb,bpb);
        }
    }
    return true;
}
void copy_texture_buffer(const p_copy_b2t &request,bool to_buffer){g_gpu_wrote=true;preflight_cpu_reads();
    need(request.tex,"Null buffer image texture");
    tex_t stable=*request.tex;p_copy_b2t p=request;p.tex=&stable;
    // 同步必须先于Image映射/引用；深度拆平面、BC解码和颜色复制随后才可读源字节。
    // Synchronize before acquiring any Image mapping/reference; depth-plane splitting, BC decoding and color copying may read source bytes only afterwards.
    if(p.tex->mem && p.tex->L.size)cpu_access(uint64_t(uintptr_t(p.tex->mem)),p.tex->L.size,9);
    if(!to_buffer){
        int bw=1,bh=1;int bpb=nvn_fmt_bpb(p.tex->d.format,&bw,&bh);xbg_depth_format depth;
        if(xbg_depth_format_info(p.tex->d.format,&depth)==0){bw=bh=1;bpb=int(depth.packed_bytes);}
        size_t row,image,bytes;
        need(bpb>0 && nvn_copy_pitch(p.r.width,p.r.height,p.r.depth,bw,bh,bpb,state.copy_stride,state.copy_image_stride,&row,&image,&bytes) && bytes<=256ull*1024*1024,"Invalid buffer image CPU read range");
        cpu_access(p.src,bytes,9);
    }
    Image &im=get_image(p.tex,to_buffer);
    if(p.has_view && p.view.set_format)(void)im.view_format(&p.view);
    if(im.samples!=VK_SAMPLE_COUNT_1_BIT || (im.decoded&&to_buffer))   // 原生BC在下方独立读回 | Native BC reads back in its own branch below
        xbg_fatalf("NVN %s format=%#x samples=%u region=%d,%d,%d:%dx%dx%d needs %s",to_buffer?"CopyTextureToBuffer":"CopyBufferToTexture",p.tex->d.format,uint32_t(im.samples),p.r.xoffset,p.r.yoffset,p.r.zoffset,p.r.width,p.r.height,p.r.depth,im.decoded?"compressed readback encoding":"depth/stencil or multisample packing");
    need((p.flags&~NVN_COPY_FLAGS_ENGINE_2D_BIT)==0,"Buffer/texture copy flags not implemented");
    uint32_t level=p.has_view&&p.view.set_levels?p.view.base_level:0,layer=p.has_view&&p.view.set_layers?p.view.min_layer:0;region_check(im,level,layer,p.r);need(p.src,"Null buffer image copy address");
    bool full_destination=xbg_copy_is_full({im.width,im.height,im.source.d.target==2?im.depth:im.layers,im.levels,level,layer,p.r.xoffset,p.r.yoffset,p.r.zoffset,p.r.width,p.r.height,p.r.depth})!=0;
    if(!to_buffer && !full_destination)get_image(p.tex,true);

    if(im.aspects!=VK_IMAGE_ASPECT_COLOR_BIT){
        xbg_depth_format f;need(xbg_depth_format_info(p.tex->d.format,&f)==0,"Unknown NVN depth copy format");
        size_t row,image,bytes;
        need(nvn_copy_pitch(p.r.width,p.r.height,p.r.depth,1,1,f.packed_bytes,state.copy_stride,state.copy_image_stride,&row,&image,&bytes) && row%f.packed_bytes==0 && bytes<=256ull*1024*1024,"Invalid packed depth copy stride/extent");
        need(p.src<=UINT64_MAX-bytes,"Depth copy address overflow");
        std::vector<uint8_t> planes;std::vector<VkBufferImageCopy> copies;VkBufferImageCopy region{};
        region.imageSubresource=copy_layers(im,level,layer,p.r);region.imageOffset=copy_offset(im,p.r);region.imageExtent=copy_extent(im,p.r);
        append_depth_planes(planes,copies,p.tex->d.format,p.r.width,p.r.height,p.r.depth,
            to_buffer?nullptr:reinterpret_cast<const void*>(p.src),bytes,row,image,region);
        Buffer staging=to_buffer?readback_buffer(planes.size()):stream_buffer(planes.size(),planes.data(),planes.size(),false);for(auto &copy:copies)copy.bufferOffset+=staging.offset;
        Commands command;im.transition(command.command,to_buffer?VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        if(to_buffer)vkCmdCopyImageToBuffer(command.command,im.image,im.layout,staging.buffer,copies.size(),copies.data());
        else vkCmdCopyBufferToImage(command.command,staging.buffer,im.image,im.layout,copies.size(),copies.data());
        im.transition(command.command,VK_IMAGE_LAYOUT_GENERAL);command.finish();
        if(to_buffer){
            gpu_sync(perf.sync_readback);staging.invalidate();invalidate_cpu_range(p.src,bytes);const uint8_t *dp=nullptr,*sp=nullptr;
            for(auto &r:copies){if(r.imageSubresource.aspectMask==VK_IMAGE_ASPECT_DEPTH_BIT)dp=static_cast<const uint8_t*>(staging.mapped)+(r.bufferOffset-staging.offset);else sp=static_cast<const uint8_t*>(staging.mapped)+(r.bufferOffset-staging.offset);}
            size_t pixels=size_t(p.r.width)*p.r.height*p.r.depth;
            std::vector<uint8_t> d16;if(dp && p.tex->d.format==0x32 && depth16_as_d24){d16.resize(pixels*2);for(size_t i=0;i<pixels;i++){uint32_t v;memcpy(&v,dp+i*4,4);uint16_t k=uint16_t(d24_to_d16(v));memcpy(&d16[i*2],&k,2);}dp=d16.data();}
            need(xbg_depth_merge(p.tex->d.format,p.r.width,p.r.height,p.r.depth,reinterpret_cast<void*>(p.src),bytes,row,image,dp,pixels*f.depth_bytes,sp,pixels*f.stencil_bytes)==0,"Packed NVN depth readback merge failed");
        }else im.gpu_written=true,provenance_write(im,__LINE__);
        xbg_logf("NVN depth %s serial=%llu format=%#x level=%u layer=%u region=%dx%dx%d planes=%zu",to_buffer?"readback":"upload",(unsigned long long)p.tex->serial,p.tex->d.format,level,layer,p.r.width,p.r.height,p.r.depth,copies.size());return;
    }
    if(!im.decoded && p.tex->d.format>=0x41 && p.tex->d.format<=0x50){ // 原生BC保持块编码 | Native BC retains block encoding
        int bw,bh,bpb=nvn_fmt_bpb(p.tex->d.format,&bw,&bh);need(bw==4 && bh==4 && !(p.r.xoffset%4) && !(p.r.yoffset%4),"BC copy origin must be block aligned");
        need(p.r.width%4==0 || uint32_t(p.r.xoffset+p.r.width)==std::max(1u,im.width>>level),"BC copy width must reach a block or mip edge");
        need(p.r.height%4==0 || uint32_t(p.r.yoffset+p.r.height)==std::max(1u,im.height>>level),"BC copy height must reach a block or mip edge");
        XbgBcCopyRows plan{};need(xbg_bc_copy_rows(p.r.width,p.r.height,p.r.depth,bpb,state.copy_stride,state.copy_image_stride,&plan)==0,"Invalid BC copy row/image stride/extent");
        need(p.src<=UINT64_MAX-plan.guest_bytes,"BC copy address overflow");
        if(!to_buffer && !im.gpu_written && !(p.tex->d.flags&0x10)){cpu_access(p.src,plan.guest_bytes,9);invalidate_cpu_range(uint64_t(uintptr_t(p.tex->mem)),p.tex->L.size);
            if(guest_texture_write(p.tex,level,layer,p.r,reinterpret_cast<const uint8_t*>(p.src),plan.row,plan.image)){im.uploaded=false;im.checked_submission=UINT64_MAX;return;}}
        std::vector<uint8_t> packed;
        if(!to_buffer){
            cpu_access(p.src,plan.guest_bytes,9);
            // 部分覆盖前上传初始全部mip/层；已有GPU内容沿用，不重建或转码。
            // Upload initial mips/layers before partial writes; retain existing GPU contents without recreation or transcoding.
            if(!xbg_copy_is_full({im.width,im.height,im.layers,im.levels,level,layer,p.r.xoffset,p.r.yoffset,p.r.zoffset,p.r.width,p.r.height,p.r.depth}))get_image(p.tex,true);
            packed.resize(plan.packed_bytes);need(xbg_bc_copy_transfer(packed.data(),packed.size(),reinterpret_cast<const void*>(p.src),plan.guest_bytes,plan,false)==0,"BC upload packing failed");
        }
        Buffer b=to_buffer?readback_buffer(plan.packed_bytes):stream_buffer(plan.packed_bytes,packed.data(),plan.packed_bytes,false);
        Commands c;im.transition(c.command,to_buffer?VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy r{};r.bufferOffset=b.offset;r.imageSubresource=copy_layers(im,level,layer,p.r);r.imageOffset=copy_offset(im,p.r);r.imageExtent=copy_extent(im,p.r);
        if(to_buffer)vkCmdCopyImageToBuffer(c.command,im.image,im.layout,b.buffer,1,&r);else vkCmdCopyBufferToImage(c.command,b.buffer,im.image,im.layout,1,&r);
        im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();
        if(to_buffer){gpu_sync(perf.sync_readback);b.invalidate();invalidate_cpu_range(p.src,plan.guest_bytes);
            need(xbg_bc_copy_transfer(reinterpret_cast<void*>(p.src),plan.guest_bytes,b.mapped,plan.packed_bytes,plan,true)==0,"BC readback unpacking failed");}
        else im.gpu_written=true,provenance_write(im,__LINE__);
        {static unsigned n=0;if(n++<20)xbg_logf("NVN BC native %s format=%#x level=%u layer=%u region=%dx%dx%d bytes=%zu",to_buffer?"readback":"upload",p.tex->d.format,level,layer,p.r.width,p.r.height,p.r.depth,plan.packed_bytes);}
        return;
    }
    if(im.decoded){
        int bw,bh,bpb=nvn_fmt_bpb(p.tex->d.format,&bw,&bh);need(bw==4 && bh==4 && !(p.r.xoffset%4) && !(p.r.yoffset%4),"BC upload origin must be block aligned");
        need(p.r.width%4==0 || uint32_t(p.r.xoffset+p.r.width)==std::max(1u,im.width>>level),"BC upload width must reach a block or mip edge");
        need(p.r.height%4==0 || uint32_t(p.r.yoffset+p.r.height)==std::max(1u,im.height>>level),"BC upload height must reach a block or mip edge");
        size_t row,image,source_bytes;
        need(nvn_copy_pitch(p.r.width,p.r.height,p.r.depth,bw,bh,bpb,state.copy_stride,state.copy_image_stride,&row,&image,&source_bytes) && row%bpb==0 && source_bytes<=256ull*1024*1024,"Invalid BC copy row stride/extent");
        need(p.src<=UINT64_MAX-source_bytes,"BC copy address overflow");
        size_t bytes=size_t(p.r.width)*p.r.height*p.r.depth*im.bpp;
        if(!im.gpu_written && !(p.tex->d.flags&0x10)){invalidate_cpu_range(uint64_t(uintptr_t(p.tex->mem)),p.tex->L.size);
            if(guest_texture_write(p.tex,level,layer,p.r,reinterpret_cast<const uint8_t*>(p.src),row,image)){im.uploaded=false;im.checked_submission=UINT64_MAX;return;}}
        need(bytes<=256ull*1024*1024,"Decoded BC staging exceeds memory bound");std::vector<uint8_t> decoded(bytes);
        int result=xbg_decode_bc_linear(p.tex->d.format,p.r.width,p.r.height,p.r.depth,reinterpret_cast<const void*>(p.src),source_bytes,row,image,decoded.data(),decoded.size());
        if(result)xbg_fatalf("NVN BC staging decode failed format=%#x level=%u result=%d",p.tex->d.format,level,result);
        Buffer buffer=stream_buffer(bytes,decoded.data(),bytes,false);Commands command;im.transition(command.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};region.bufferOffset=buffer.offset;region.imageSubresource=copy_layers(im,level,layer,p.r);region.imageOffset=copy_offset(im,p.r);region.imageExtent=copy_extent(im,p.r);
        vkCmdCopyBufferToImage(command.command,buffer.buffer,im.image,im.layout,1,&region);im.transition(command.command,VK_IMAGE_LAYOUT_GENERAL);command.finish();im.gpu_written=true,provenance_write(im,__LINE__);
        xbg_logf("NVN BC upload format=%#x level=%u layer=%u region=%dx%dx%d src=%zu decoded=%zu",p.tex->d.format,level,layer,p.r.width,p.r.height,p.r.depth,source_bytes,bytes);return;
    }
    size_t row,image,n;
    need(nvn_copy_pitch(p.r.width,p.r.height,p.r.depth,1,1,im.bpp,state.copy_stride,state.copy_image_stride,&row,&image,&n) && n<=256ull*1024*1024,"Invalid or oversized color copy stride/extent");
    need(p.src<=UINT64_MAX-n,"Color copy address overflow");
    if(!to_buffer && !im.gpu_written && !(p.tex->d.flags&0x10)){invalidate_cpu_range(uint64_t(uintptr_t(p.tex->mem)),p.tex->L.size);
        if(guest_texture_write(p.tex,level,layer,p.r,reinterpret_cast<const uint8_t*>(p.src),row,image)){im.uploaded=false;im.checked_submission=UINT64_MAX;return;}}
    size_t slice_alignment=4;while(slice_alignment%im.bpp)slice_alignment+=4;
    size_t staging_row=size_t(p.r.width)*im.bpp;
    size_t staging_image=((staging_row*size_t(p.r.height)+slice_alignment-1)/slice_alignment)*slice_alignment,staging_bytes=staging_image*size_t(p.r.depth);
    need(staging_bytes<=256ull*1024*1024,"Color staging exceeds memory bound");
    std::vector<uint8_t> upload;
    if(!to_buffer){upload.resize(staging_bytes);
        for(int k=0;k<p.r.depth;k++)for(int y=0;y<p.r.height;y++)
            memcpy(upload.data()+size_t(k)*staging_image+size_t(y)*staging_row,reinterpret_cast<const uint8_t*>(p.src)+size_t(k)*image+size_t(y)*row,size_t(p.r.width)*im.bpp);
    }
    Buffer b=to_buffer?readback_buffer(staging_bytes):stream_buffer(staging_bytes,upload.data(),staging_bytes,false);
    Commands c;im.transition(c.command,to_buffer?VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    // 每层单独记录，允许字节图距不整除行距。
    // Record each slice separately to allow byte image strides indivisible by row strides.
    std::vector<VkBufferImageCopy> copies;
    for(int k=0;k<p.r.depth;k++){
        VkBufferImageCopy r{};r.bufferOffset=b.offset+size_t(k)*staging_image;r.bufferRowLength=0;
        r.imageSubresource=copy_layers(im,level,layer,p.r);r.imageSubresource.layerCount=1;
        r.imageOffset=copy_offset(im,p.r);r.imageExtent={uint32_t(p.r.width),uint32_t(p.r.height),1};
        if(im.source.d.target==2)r.imageOffset.z+=k;else r.imageSubresource.baseArrayLayer+=k;
        copies.push_back(r);
    }
    if(to_buffer)vkCmdCopyImageToBuffer(c.command,im.image,im.layout,b.buffer,uint32_t(copies.size()),copies.data());else vkCmdCopyBufferToImage(c.command,b.buffer,im.image,im.layout,uint32_t(copies.size()),copies.data());im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();
    if(to_buffer){gpu_sync(perf.sync_readback);b.invalidate();invalidate_cpu_range(p.src,n);
        for(int k=0;k<p.r.depth;k++)for(int y=0;y<p.r.height;y++){
            size_t off=size_t(k)*image+size_t(y)*row;
            memcpy(reinterpret_cast<uint8_t*>(p.src)+off,static_cast<const uint8_t*>(b.mapped)+size_t(k)*staging_image+size_t(y)*staging_row,size_t(p.r.width)*im.bpp);
        }
    }else im.gpu_written=true,provenance_write(im,__LINE__);
}
void copy_texture(const p_copy_t2t &p){g_gpu_wrote=true;preflight_cpu_reads();Image &s=get_image(p.src,true),&d=get_image(p.dst,false);{static unsigned n=0;if(n++<60)xbg_logf("NVN COPYT2T src=%llu fmt=%#x %ux%ux%u L%u lv%u -> dst=%llu fmt=%#x %ux%u L%u lv%u decoded=%d/%d region=%d,%d,%d %dx%dx%d -> %d,%d,%d sv=%d/%u/%u dv=%d/%u/%u",(unsigned long long)p.src->serial,p.src->d.format,s.width,s.height,s.depth,s.layers,s.levels,(unsigned long long)p.dst->serial,p.dst->d.format,d.width,d.height,d.layers,d.levels,int(s.decoded),int(d.decoded),p.sr.xoffset,p.sr.yoffset,p.sr.zoffset,p.sr.width,p.sr.height,p.sr.depth,p.dr.xoffset,p.dr.yoffset,p.dr.zoffset,int(p.has_sv),p.has_sv?p.sv.base_level:0,p.has_sv?p.sv.min_layer:0,int(p.has_dv),p.has_dv?p.dv.base_level:0,p.has_dv?p.dv.min_layer:0);}
    if(p.has_sv && p.sv.set_format)(void)s.view_format(&p.sv);
    if(p.has_dv && p.dv.set_format)(void)d.view_format(&p.dv);
    // 已知ASTC-pack源是ASTC4x4，目的BC3及既有RGBA BC目标从建图起解码，不迁移已有内容。
    // Known ASTC-pack sources use ASTC4x4; BC3 and existing RGBA BC destinations are decoded at creation without migrating contents.
    if(!s.decoded && d.decoded && s.fmt>=VK_FORMAT_ASTC_4x4_UNORM_BLOCK && s.fmt<=VK_FORMAT_ASTC_4x4_SRGB_BLOCK && (d.fmt==VK_FORMAT_R8G8B8A8_UNORM || d.fmt==VK_FORMAT_R8G8B8A8_SRGB)
       && p.sr.width==p.dr.width && p.sr.height==p.dr.height && p.sr.depth==p.dr.depth && s.source.d.target!=2 && d.source.d.target!=2){
        VkFormatProperties sf{},df{};vkGetPhysicalDeviceFormatProperties(cfg.physical,s.fmt,&sf);vkGetPhysicalDeviceFormatProperties(cfg.physical,d.fmt,&df);
        uint32_t sl=p.has_sv&&p.sv.set_levels?p.sv.base_level:0,dl=p.has_dv&&p.dv.set_levels?p.dv.base_level:0,sa=p.has_sv&&p.sv.set_layers?p.sv.min_layer:0,da=p.has_dv&&p.dv.set_layers?p.dv.min_layer:0;
        region_check(s,sl,sa,p.sr);region_check(d,dl,da,p.dr);need(&s!=&d,"Same-image copy needs overlap validation");
        bool full_destination=xbg_copy_is_full({d.width,d.height,d.layers,d.levels,dl,da,p.dr.xoffset,p.dr.yoffset,p.dr.zoffset,p.dr.width,p.dr.height,p.dr.depth})!=0;
        if(!full_destination)get_image(p.dst,true);
        XbgBlitRegion source{s.width,s.height,s.layers,s.levels,sl,sa,p.sr.xoffset,p.sr.yoffset,p.sr.zoffset,p.sr.width,p.sr.height,p.sr.depth};
        XbgBlitRegion destination{d.width,d.height,d.layers,d.levels,dl,da,p.dr.xoffset,p.dr.yoffset,p.dr.zoffset,p.dr.width,p.dr.height,p.dr.depth};
        VkImageBlit blit{};VkFilter filter{};
        need(xbg_astc_bc_plan(s.fmt,d.fmt,source,destination,uint32_t(p.flags),sf.optimalTilingFeatures,df.optimalTilingFeatures,&blit,&filter)==0,"ASTC to decoded-BC copy requires matching sRGB, supported regions/flags and blit features");
        Commands c;s.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);d.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);vkCmdBlitImage(c.command,s.image,s.layout,d.image,d.layout,1,&blit,filter);
        s.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);d.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();d.gpu_written=true,provenance_write(d,__LINE__);
        {static unsigned n=0;if(n++<8)xbg_logf("NVN ASTC2BC blit src=%llu fmt=%#x -> dst=%llu fmt=%#x %dx%d lv%u/%u layer%u/%u",(unsigned long long)p.src->serial,p.src->d.format,(unsigned long long)p.dst->serial,p.dst->d.format,p.sr.width,p.sr.height,sl,dl,sa,da);}
        return;
    }
    need(!(s.fmt>=VK_FORMAT_ASTC_4x4_UNORM_BLOCK && s.fmt<=VK_FORMAT_ASTC_12x12_SRGB_BLOCK && d.source.d.format>=0x41 && d.source.d.format<=0x50 && !d.decoded),"ASTC blocks into a native BC texture: the ASTC pack needs decoded BC (marker gfx-native-bc = 0)");
    if(s.decoded || d.decoded)need(s.decoded && d.decoded && xbg_bc_view_compatible(s.source.d.format,d.source.d.format),"BC texture copy must retain compression and signedness");
    need((uint32_t(p.flags)&~3u)==0 && s.fmt==d.fmt && s.aspects==d.aspects && s.samples==d.samples,"Texture copy requires matching formats/aspects and supported flags");
    uint32_t sl=p.has_sv&&p.sv.set_levels?p.sv.base_level:0,dl=p.has_dv&&p.dv.set_levels?p.dv.base_level:0,sa=p.has_sv&&p.sv.set_layers?p.sv.min_layer:0,da=p.has_dv&&p.dv.set_layers?p.dv.min_layer:0;
    region_check(s,sl,sa,p.sr);region_check(d,dl,da,p.dr);need(&s!=&d,"Same-image copy needs overlap validation");need((s.source.d.target==2)==(d.source.d.target==2),"Mixed volume/array copy requires slice mapping");
    bool scaled=p.sr.width!=p.dr.width || p.sr.height!=p.dr.height || p.sr.depth!=p.dr.depth;
    VkImageBlit blit{};VkFilter filter{};
    if(scaled){
        need(!s.decoded && !d.decoded && s.source.d.target!=0 && s.source.d.target!=2 && s.source.d.target!=3 && d.source.d.target!=0 && d.source.d.target!=2 && d.source.d.target!=3 && s.aspects==VK_IMAGE_ASPECT_COLOR_BIT && s.samples==VK_SAMPLE_COUNT_1_BIT,"Scaled blit requires native single-sample color 2D/array images");
        need(s.view_format(p.has_sv?&p.sv:nullptr)==s.fmt && d.view_format(p.has_dv?&p.dv:nullptr)==d.fmt,"Scaled blit cannot reinterpret image views");
        VkFormatProperties sf{},df{};vkGetPhysicalDeviceFormatProperties(cfg.physical,s.fmt,&sf);vkGetPhysicalDeviceFormatProperties(cfg.physical,d.fmt,&df);
        XbgBlitRegion src{s.width,s.height,s.layers,s.levels,sl,sa,p.sr.xoffset,p.sr.yoffset,p.sr.zoffset,p.sr.width,p.sr.height,p.sr.depth};
        XbgBlitRegion dst{d.width,d.height,d.layers,d.levels,dl,da,p.dr.xoffset,p.dr.yoffset,p.dr.zoffset,p.dr.width,p.dr.height,p.dr.depth};
        need(xbg_blit_plan(src,dst,uint32_t(p.flags),xbg_clear_color_class(s.source.d.format),sf.optimalTilingFeatures,df.optimalTilingFeatures,&blit,&filter)==0,"Scaled blit region/type/filter/device capability unsupported");
    }
    // Import initial CPU contents before a partial destination overwrite; this
    // also preserves untouched mip levels and array layers for ordinary copies.
    bool full_destination=xbg_copy_is_full({d.width,d.height,d.source.d.target==2?d.depth:d.layers,d.levels,dl,da,p.dr.xoffset,p.dr.yoffset,p.dr.zoffset,p.dr.width,p.dr.height,p.dr.depth})!=0;
    if(!full_destination){need(d.samples==VK_SAMPLE_COUNT_1_BIT || d.gpu_written,"Partial copy into initial CPU multisample storage unsupported");get_image(p.dst,true);}
    if(scaled){Commands c;s.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);d.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);vkCmdBlitImage(c.command,s.image,s.layout,d.image,d.layout,1,&blit,filter);s.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);d.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();d.gpu_written=true,provenance_write(d,__LINE__);return;}
    Commands c;s.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);d.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageCopy r{};r.srcSubresource=copy_layers(s,sl,sa,p.sr);r.dstSubresource=copy_layers(d,dl,da,p.dr);r.srcOffset=copy_offset(s,p.sr);r.dstOffset=copy_offset(d,p.dr);r.extent=copy_extent(s,p.sr);std::vector<VkImageCopy> regions;
    for(VkImageAspectFlags flag: {VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT),VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT),VkImageAspectFlags(VK_IMAGE_ASPECT_STENCIL_BIT)})if(s.aspects&flag){auto one=r;one.srcSubresource.aspectMask=one.dstSubresource.aspectMask=flag;regions.push_back(one);}
    vkCmdCopyImage(c.command,s.image,s.layout,d.image,d.layout,regions.size(),regions.data());s.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);d.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();d.gpu_written=true,provenance_write(d,__LINE__);}
void partial_clear_texture(Image &im,const p_clear_tex &request,uint32_t level,uint32_t base_layer,int scalar){
    need(im.source.d.target!=2,"Partial 3D texture clear requires 2D slice-view support");
    bool color=im.aspects==VK_IMAGE_ASPECT_COLOR_BIT;need(color || request.mask==15,"Depth/stencil texture channel-mask semantics not implemented");
    for(uint32_t slice=0;slice<uint32_t(request.r.depth);slice++){
        Commands c;Pass p;p.width=std::max(1u,im.width>>level);p.height=std::max(1u,im.height>>level);p.samples=im.samples;p.color_slots=color?1:0;
        uint32_t selected_layer=0;VkRect2D rect{};
        need(xbg_texture_clear_slice(p.width,p.height,im.layers,base_layer,request.r.xoffset,request.r.yoffset,request.r.zoffset,request.r.width,request.r.height,request.r.depth,slice,&rect,&selected_layer)==0,"Invalid texture clear slice");
        view_t view=request.view;view.set_levels=view.set_layers=1;view.base_level=level;view.nlevels=1;view.min_layer=selected_layer;view.nlayers=1;
        VkImageView image_view=im.view(&view,true);im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);
        VkAttachmentDescription attachment{};attachment.format=im.view_format(&view);attachment.samples=p.samples;attachment.loadOp=attachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_LOAD;attachment.storeOp=attachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_STORE;attachment.initialLayout=attachment.finalLayout=VK_IMAGE_LAYOUT_GENERAL;
        VkAttachmentReference ref{0,VK_IMAGE_LAYOUT_GENERAL};VkSubpassDescription sub{};sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sub.colorAttachmentCount=color?1:0;sub.pColorAttachments=color?&ref:nullptr;sub.pDepthStencilAttachment=color?nullptr:&ref;
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};rp.attachmentCount=1;rp.pAttachments=&attachment;rp.subpassCount=1;rp.pSubpasses=&sub;VK(vkCreateRenderPass(cfg.device,&rp,nullptr,&p.pass));
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};fb.renderPass=p.pass;fb.attachmentCount=1;fb.pAttachments=&image_view;fb.width=p.width;fb.height=p.height;fb.layers=1;VK(vkCreateFramebuffer(cfg.device,&fb,nullptr,&p.framebuffer));
        {VkFramebuffer f=p.framebuffer;VkRenderPass rp_=p.pass;retire_handle([f,rp_]{vkDestroyFramebuffer(cfg.device,f,nullptr);vkDestroyRenderPass(cfg.device,rp_,nullptr);});}
        auto masked_object=std::make_shared<MaskedClear>();MaskedClear &masked=*masked_object;retire_object(masked_object);bool partial=color && request.mask!=15;if(partial)masked.prepare(p,0,scalar,uint32_t(request.mask),0,0);
        begin_pass(p,c);
        if(partial)masked.encode(c,p,rect,&request.c);
        else{VkClearAttachment clear{};clear.aspectMask=im.aspects;if(color)memcpy(&clear.clearValue.color,&request.c,16);else clear.clearValue.depthStencil={im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT?request.c.f[0]:0.0f,request.c.u[im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT?1:0]};VkClearRect region{rect,0,1};vkCmdClearAttachments(c.command,1,&clear,1,&region);}
        vkCmdEndRenderPass(c.command);pass_prof_end(c.command);c.finish();
    }
    im.gpu_written=true,provenance_write(im,__LINE__);
}
void clear_texture(const p_clear_tex &input){g_gpu_wrote=true;preflight_cpu_reads();
    p_clear_tex p=input;
    Image &im=get_image(p.tex,false);need((uint32_t(p.mask)&~15u)==0,"Invalid texture clear mask");if(!p.mask)return;
    uint32_t level=p.has_view&&p.view.set_levels?p.view.base_level:0,layer=p.has_view&&p.view.set_layers?p.view.min_layer:0;
    region_check(im,level,layer,p.r);uint32_t w=std::max(1u,im.width>>level),h=std::max(1u,im.height>>level);
    need(p.is_uint>=0 && p.is_uint<=2,"Invalid texture clear scalar type");int scalar=p.is_uint==2?1:p.is_uint==1?2:0;
    if(im.aspects==VK_IMAGE_ASPECT_COLOR_BIT){VkFormat viewed=im.view_format(p.has_view?&p.view:nullptr);static unsigned notes=0;if(notes++<8)xbg_logf("NVN clear-value texture input=%d nvn=%#x view=%#x vk=%u mip=%u mask=%#x bits=%08x/%08x/%08x/%08x",scalar,im.source.d.format,p.has_view&&p.view.set_format?p.view.format:im.source.d.format,unsigned(viewed),level,unsigned(p.mask),p.c.u[0],p.c.u[1],p.c.u[2],p.c.u[3]);VkClearColorValue value{};int converted=xbg_clear_value(viewed,scalar,&p.c,&value);
        if(converted<0)xbg_fatalf("Texture clear conversion failed nvn=%#x view=%#x vk=%u input=%d rc=%d",im.source.d.format,p.has_view&&p.view.set_format?p.view.format:im.source.d.format,unsigned(viewed),scalar,converted);
        memcpy(&p.c,&value,16);scalar=converted;}

    bool partial=p.mask!=15 || p.r.xoffset || p.r.yoffset || uint32_t(p.r.width)!=w || uint32_t(p.r.height)!=h;
    if(partial)need(im.source.d.target!=2,"Partial 3D texture clear requires 2D slice-view support");
    if(im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT)need(p.c.f[0]>=0 && p.c.f[0]<=1,"Texture depth clear outside normalized range");
    if(partial || im.levels>1 || im.layers>1){need(im.samples==VK_SAMPLE_COUNT_1_BIT || im.gpu_written,"Partial clear of uninitialized CPU multisample storage unsupported");get_image(p.tex,true);}
    if(partial){partial_clear_texture(im,p,level,layer,scalar);return;}
    bool volume=im.source.d.target==2;need(!volume || (p.r.zoffset==0 && uint32_t(p.r.depth)==std::max(1u,im.depth>>level)),"Partial volume clear requires shader clear");
    VkFormat viewed=im.view_format(p.has_view?&p.view:nullptr);Commands c;VkClearValue value{};memcpy(&value.color,&p.c,16);
    if(viewed!=im.fmt){
        // VkClearColorImage uses the image's creation format. The original game
        // clears through integer views, so use an actual view/render-pass clear
        // to preserve raw bit patterns instead of numerically converting them.
        need(!volume && im.aspects==VK_IMAGE_ASPECT_COLOR_BIT,"Reinterpreted volume/depth clear unsupported");
        im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);
        std::vector<VkFramebuffer> framebuffers;VkRenderPass pass{};
        VkAttachmentDescription attachment{};attachment.format=viewed;attachment.samples=im.samples;attachment.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;attachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;attachment.initialLayout=attachment.finalLayout=VK_IMAGE_LAYOUT_GENERAL;
        VkAttachmentReference ref{0,VK_IMAGE_LAYOUT_GENERAL};VkSubpassDescription sub{};sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sub.colorAttachmentCount=1;sub.pColorAttachments=&ref;
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};rp.attachmentCount=1;rp.pAttachments=&attachment;rp.subpassCount=1;rp.pSubpasses=&sub;VK(vkCreateRenderPass(cfg.device,&rp,nullptr,&pass));
        for(uint32_t z=0;z<uint32_t(p.r.depth);z++){
            view_t view=p.view;view.set_levels=view.set_layers=1;view.base_level=level;view.nlevels=1;view.min_layer=layer+p.r.zoffset+z;view.nlayers=1;
            VkImageView image_view=im.view(&view,true);VkFramebuffer fb;VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};fi.renderPass=pass;fi.attachmentCount=1;fi.pAttachments=&image_view;fi.width=w;fi.height=h;fi.layers=1;VK(vkCreateFramebuffer(cfg.device,&fi,nullptr,&fb));framebuffers.push_back(fb);
            VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};begin.renderPass=pass;begin.framebuffer=fb;begin.renderArea.extent={w,h};begin.clearValueCount=1;begin.pClearValues=&value;
            if(slots[slot_index].prof){Pass profile;profile.width=w;profile.height=h;profile.images.push_back(&im);pass_prof_begin(profile,c.command);}
            vkCmdBeginRenderPass(c.command,&begin,VK_SUBPASS_CONTENTS_INLINE);vkCmdEndRenderPass(c.command);pass_prof_end(c.command);
        }
        im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();retire_handle([framebuffers,pass]{for(auto f:framebuffers)vkDestroyFramebuffer(cfg.device,f,nullptr);vkDestroyRenderPass(cfg.device,pass,nullptr);});
    }else{
        im.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageSubresourceRange range{im.aspects,level,1,volume?0:layer+uint32_t(p.r.zoffset),volume?1:uint32_t(p.r.depth)};
        if(im.aspects==VK_IMAGE_ASPECT_COLOR_BIT)vkCmdClearColorImage(c.command,im.image,im.layout,&value.color,1,&range);
        else{VkClearDepthStencilValue ds{im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT?p.c.f[0]:0.0f,p.c.u[im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT?1:0]};vkCmdClearDepthStencilImage(c.command,im.image,im.layout,&ds,1,&range);}
        im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();
    }
    im.gpu_written=true,provenance_write(im,__LINE__);
}
// Buffer copy whose source is a pending GPU result: copy on the GPU and keep
// the destination pending too, instead of stalling for the CPU memmove.
// 保守门暂时使 pending-view 链接回退CPU；恢复前须证明源readback跨退役仍存活。
// The conservative gate temporarily falls back to CPU for pending-view chains; restore only after proving source readback lifetime across retirement.
bool gpu_buffer_copy(uint64_t src,uint64_t dst,uint64_t n){preflight_cpu_reads();
    if(!n || (src<dst+n && dst<src+n))return false;
    for(auto &w:pending_writebacks)if(dst<w.addr+w.bytes && dst+n>w.addr)return false;
    Buffer view;if(!pending_view(src,n,1,view))return false;
    invalidate_cpu_range(dst,n);
    Slot &s=recording_slot();end_pass(6);Buffer b=readback_buffer(n);
    VkBufferCopy region{view.offset,b.offset,n};vkCmdCopyBuffer(s.cmd,view.buffer,b.buffer,1,&region);g_barrier_site="copy-buffer";full_barrier(s.cmd);
    pending_writebacks.push_back({dst,n,b,&s});return true;
}
void resolve_texture(const p_copy_t2t &p){preflight_cpu_reads();
    Image &s=get_image(p.src,true),&d=get_image(p.dst,false);
    need(s.samples!=VK_SAMPLE_COUNT_1_BIT && d.samples==VK_SAMPLE_COUNT_1_BIT && s.fmt==d.fmt,"Resolve requires matching MS source / single-sample destination");
    need(s.aspects==VK_IMAGE_ASPECT_COLOR_BIT && d.aspects==VK_IMAGE_ASPECT_COLOR_BIT,"Depth/stencil resolve mode is not yet verified against original NVN");
    uint32_t sl=p.has_sv&&p.sv.set_levels?p.sv.base_level:0,dl=p.has_dv&&p.dv.set_levels?p.dv.base_level:0,sa=p.has_sv&&p.sv.set_layers?p.sv.min_layer:0,da=p.has_dv&&p.dv.set_layers?p.dv.min_layer:0;
    uint32_t sn=p.has_sv&&p.sv.set_layers?p.sv.nlayers:s.layers,dn=p.has_dv&&p.dv.set_layers?p.dv.nlayers:d.layers;
    need(sl<s.levels && dl<d.levels && sa+sn<=s.layers && da+dn<=d.layers && sn==dn,"Resolve view out of range");
    uint32_t w=std::max(1u,s.width>>sl),h=std::max(1u,s.height>>sl);need(w==std::max(1u,d.width>>dl)&&h==std::max(1u,d.height>>dl),"Resolve extents differ");
    Commands c;s.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);d.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageResolve r{};r.srcSubresource={s.aspects,sl,sa,sn};r.dstSubresource={d.aspects,dl,da,dn};r.extent={w,h,1};vkCmdResolveImage(c.command,s.image,s.layout,d.image,d.layout,1,&r);s.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);d.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();d.gpu_written=true,provenance_write(d,__LINE__);
}

// CPU-visible writes occur only after all preceding synchronous GPU work finished.
// Every byte count and opcode is checked; unknown commands terminate with context.
// Consecutive commands nearly always sit in one range: remember the last one (pools change only under the mutex,
// and clear it; host segments are freed only at shutdown).
uintptr_t cmd_range_lo=0;size_t cmd_range_bytes=0;
size_t command_available(const uint8_t *p){uintptr_t address=uintptr_t(p);if(address-cmd_range_lo<cmd_range_bytes)return cmd_range_bytes-(address-cmd_range_lo);auto it=pools.upper_bound(address);if(it!=pools.begin()){--it;if(address>=it->first && address-it->first<it->second){cmd_range_lo=it->first;cmd_range_bytes=it->second;return it->second-(address-it->first);}}{auto h=host_ranges.upper_bound(address);if(h!=host_ranges.begin()){--h;if(address>=h->first && address-h->first<h->second){cmd_range_lo=h->first;cmd_range_bytes=h->second;return h->second-(address-h->first);}}}xbg_fatalf("NVN command address %#llx is outside registered command pools",(unsigned long long)address);}
void execute_draw_call(const xbg_draw_call &p){g_gpu_wrote=true;
    need(p.index_type>=-1 && p.index_type<=2 && p.first>=0 && p.count>=0 && p.instances>=0 && p.base_instance>=0,"Invalid recorded draw plan");
    if(!p.count || !p.instances)return;
    if(p.index_type<0)draw(p.primitive,p.first,p.count,nullptr,uint32_t(p.instances),uint32_t(p.base_instance));
    else{
        uint64_t bytes=uint64_t(p.count)<<p.index_type;
        need(bytes<=command_available(reinterpret_cast<const uint8_t*>(p.index_address)),"Indirect index read exceeds registered storage");
        p_draw_idx indexed{p.primitive,p.index_type,p.count,p.base_vertex,p.base_instance,p.instances,p.index_address};
        draw(p.primitive,0,p.count,&indexed);
    }
}
void execute_indirect(const xbg_indirect &p,bool single){g_gpu_wrote=true;preflight_cpu_reads();
    need(p.arguments && p.max_count>=0 && p.max_count<=1000000,"Invalid indirect draw request");
    uint32_t count=1;
    if(!single){
        need(p.count_address && !(p.count_address&3),"Invalid indirect count address");
        need(command_available(reinterpret_cast<const uint8_t*>(p.count_address))>=4,"Truncated indirect count storage");
        cpu_access(p.count_address,4,5);memcpy(&count,reinterpret_cast<const void*>(p.count_address),4);count=std::min(count,uint32_t(p.max_count));
    }
    for(uint32_t i=0;i<count;i++){
        uint64_t address=0;need(xbg_draw_indirect_address(p.arguments,i,p.stride,p.index_type,&address)==0,"Invalid indirect argument address/stride");
        size_t bytes=p.index_type<0?16:20;
        need(bytes<=command_available(reinterpret_cast<const uint8_t*>(address)),"Indirect arguments exceed registered storage");cpu_access(address,bytes,5);
        // Read at execution, after preceding synchronous GPU storage writeback,
        // never at command recording. Both single and multi use this path.
        xbg_draw_call plan{};int result=xbg_draw_indirect_plan(&plan,p.primitive,p.index_type,p.index_address,reinterpret_cast<const void*>(address),bytes);
        if(result)xbg_fatalf("Invalid indirect draw parameters: %d",result);
        execute_draw_call(plan);
    }
}
template<class T>T payload(const uint8_t *p,size_t n){need(n>=sizeof(T),"Truncated NVN command payload");T out;memcpy(&out,p,sizeof(out));return out;}
template<class T>T variable_payload(const uint8_t *p,size_t n){T out{};memcpy(&out,p,std::min(n,sizeof(T)));return out;}
// ══ np715 阶段 2：帧间复用（跳过「与上一提交结构相同」的 pass 的绘制类 op）。默认关。 ══
// 判据来源（今日 §1j/§1k 实测）：整帧/逐字节摘要不可用（2–4 帧缓冲地址轮换、UBO 有无关变化），
//   但**结构**（op 类型+长度序列）+ 渲染目标身份在相邻帧是稳定的；上屏像素 71.8% 的帧逐字节相同。
// 所以判据 = 「RT 身份键 + 结构摘要（只 hash op/size 8 字节头）」都相同 ⇒ 该 pass 的 S 类 op 全跳。
// 安全设计：① 只按**整 pass** 决定（跳就跳全，绝不半跳）；② 执行期逐 pass 复核身份键，
//   对不上立刻停跳本提交（防止 walk 与 execute 的遍历顺序分叉导致跳错 pass）；
//   ③ 默认关；④ 结果侧用 `[上屏] 源图 hash` 验收（跳过轮必须仍是同一 hash）。
namespace frameskip{
static int mode=-1;
static std::vector<uint64_t> keys,hashes;                  // 本提交：每个 pass 的身份键 / 结构摘要
static std::vector<uint8_t> flags;                         // 1 = 本 pass 的 S 类 op 可跳
static std::vector<uint64_t> prev_keys,prev_hashes,prev_ubo0;
static size_t idx=0;static bool armed=false;                // 执行期游标
static uint64_t skipped_ops=0,skipped_passes=0,submits=0,desync=0,planned=0;
static uint64_t plan_ns=0;   // plan() 的耗时累加（探针自身开销，必须与 execute 分开算）
static bool enabled(){if(mode<0){const char*e=getenv("DM_FRAME_SKIP");mode=e?atoi(e):0;if(mode)xbg_logf("【跳过】DM_FRAME_SKIP=%d 开启（阶段 2，跳过匹配 pass 的绘制类 op）",mode);}return mode>0;}
static bool s_op(uint32_t op){
    return op==OP_DRAW||op==OP_DRAW_IDX||op==OP_DISPATCH||op==OP_CLEAR_COLOR||op==OP_CLEAR_DS
        ||op==(OP_COUNT_+2)||op==(OP_COUNT_+3)||op==(OP_COUNT_+7)||op==(OP_COUNT_+8)||op==(OP_COUNT_+9);
}
// 身份键 = OP_SET_RT 载荷里各渲染目标的 tex_t.serial（指针会随录制副本变，不能用）
static uint64_t rt_key(const uint8_t *p,size_t n,uint64_t *first_serial=nullptr){
    uint64_t h=1469598103934665603ull;int32_t cnt=0;
    if(first_serial)*first_serial=0;
    if(n<4)return h;
    memcpy(&cnt,p,4);
    if(cnt<0)cnt=0;if(cnt>8)cnt=8;
    auto mixv=[](uint64_t hh,const void *q,size_t nb){const uint8_t*b=(const uint8_t*)q;for(size_t i=0;i<nb;i++){hh^=b[i];hh*=1099511628211ull;}return hh;};
    h=mixv(h,&cnt,4);
    for(int i=0;i<cnt;i++){
        size_t off=offsetof(p_set_rt,color)+size_t(i)*sizeof(const tex_t*);
        if(off+sizeof(const tex_t*)>n)break;
        const tex_t *t=nullptr;memcpy(&t,p+off,sizeof(const tex_t*));
        uint64_t s=t?t->serial:0;if(i==0&&first_serial)*first_serial=s;h=mixv(h,&s,8);
    }
    if(offsetof(p_set_rt,depth)+sizeof(const tex_t*)<=n){
        const tex_t *d=nullptr;memcpy(&d,p+offsetof(p_set_rt,depth),sizeof(const tex_t*));
        uint64_t s=d?d->serial:0;h=mixv(h,&s,8);
    }
    return h;
}
struct seg{uint64_t key,hash;uint64_t ser0,ubo0;uint8_t ubo0_seen,uploads;size_t ops;uint32_t draws;};
// 窗口贴图表：写窗口的那趟是**末尾合成**，跳过它就是"这一帧没上屏"（np716f 22 帧不一致里的黑帧正是这个）。
// 照 Metal 世界冻结的做法（nvn_mtl.m:5382-5396）：「界面趟与末尾合成趟照画」，只跳世界段。
static uint64_t win_ser[8];static int win_n=0;
static void note_window_serial(uint64_t s){if(win_n<8){for(int i=0;i<win_n;i++)if(win_ser[i]==s)return;win_ser[win_n++]=s;}}
static bool is_window_serial(uint64_t s){for(int i=0;i<win_n;i++)if(win_ser[i]==s)return true;return false;}
static std::vector<seg> segs;
static void seg_open(uint64_t key,uint64_t ser0){seg s{};s.key=key;s.ser0=ser0;s.hash=1469598103934665603ull;segs.push_back(s);}
static void seg_close(){
    if(segs.empty())return;
    seg &s=segs.back();if(!s.ops)segs.pop_back();      // 空段（只有 SET_RT）不参与判定
}
// 结构摘要：只吃 (op,size) 8 字节头，丢掉全部操作数（地址轮换/时间值都不影响）
static void plan_walk(const uint8_t *pc,unsigned depth,uint64_t &steps){
    if(depth>64)return;
    for(;;){
        if(++steps>10000000ull)return;
        size_t available=command_available(pc);
        if(available<sizeof(op_hdr))return;
        op_hdr hd;memcpy(&hd,pc,sizeof(hd));
        if(hd.size<8||(hd.size&7)||hd.size>available)return;
        if(hd.op==OP_END)return;
        if(hd.op==OP_CALL||hd.op==OP_JUMP){
            if(hd.size<16)return;
            uint64_t next;memcpy(&next,pc+8,8);
            if(hd.op==OP_JUMP){pc=reinterpret_cast<const uint8_t*>(next);continue;}
            plan_walk(reinterpret_cast<const uint8_t*>(next),depth+1,steps);
        }else if(hd.op==OP_SET_RT){
            seg_close();uint64_t s0=0;uint64_t k=rt_key(pc+8,hd.size-8,&s0);seg_open(k,s0);
        }else if(hd.op!=OP_BEGIN){
            // ⚠ 首个 OP_SET_RT 之前的 op（内置 clear / 状态设置）**不建段**：
            //   execute() 是按 OP_SET_RT 数 pass 序号的，这里多开一段就会整体错位一位
            //   ⇒ 执行期第一帧就「身份对不上」（np716a 学费：planned 有值、skipped_ops 恒 0）。
            if(segs.empty()){pc+=hd.size;continue;}
            seg &s=segs.back();
            // ⚠ 只对 **S 类（绘制/清除/派发）** 建摘要：M' 状态 op 是游戏"状态变了才发"的，
            //   同一画面在两帧里的状态 op 序列可以不同（缓存命中就不发）⇒ 把它们算进摘要，
            //   大 pass 永远对不上（np716b 实测：planned 只有 85/149，大 pass 一次没跳）。
            //   这里刻意放宽到"绘制类 op 的类型+长度序列"：它由遮挡剔除结果与合批决定，
            //   与画面同源；内容层面的风险由 `[上屏] 源图 hash` 验收兜住。
            if(s_op(hd.op)){
                s.hash^=uint64_t(hd.op)^(uint64_t(hd.size)<<32);s.hash*=1099511628211ull;
                ++s.draws;
            }
            // 内容判据（照 Metal 的相机 UBO 检查，nvn_mtl.m:5527）：本 pass **第一个** BIND_UBO 的头部
            // 256 B 做指纹。相机/视图矩阵就在这里，矩阵变了 = 画面会变 ⇒ 该 pass 不许跳。
            // 只取第一个、只取 256 B：全量 UBO 内容里有时间/帧号一类无关值，会把判据做成"永远不匹配"（np715g 学费）。
            // 资源上传（贴图/缓冲被改写）⇒ 这个 pass 的画面依赖新数据，**一律不跳**。
            // np718a 学费：相机 UBO 没变、结构也没变，但画面对照帧给出 16c095d0（跳过帧停在旧的
            // 83a6875a）⇒ 变化来自上传过的资源，不是矩阵。上传本身是 M 类、照常执行，所以必须显式挡住。
            if(hd.op==OP_COPY_B2T||hd.op==OP_COPY_T2T||hd.op==OP_COPY_T2B||hd.op==OP_CLEAR_TEX
               ||hd.op==OP_UPDATE_UB||hd.op==OP_COPY_B2B||hd.op==OP_COPY_STRIDE||hd.op==OP_COPY_IMAGE_STRIDE)s.uploads=1;
            if(hd.op==OP_BIND_UBO && !s.ubo0_seen){
                p_bind_buf b{};size_t cp=hd.size-8;if(cp>sizeof(b))cp=sizeof(b);
                memcpy(&b,pc+8,cp);
                if(b.addr>=4096&&b.size){
                    size_t take=size_t(b.size>256?256:b.size);
                    uint64_t hh=1469598103934665603ull;
                    const uint8_t *q=reinterpret_cast<const uint8_t*>(uintptr_t(b.addr));
                    for(size_t i=0;i<take;i++){hh^=q[i];hh*=1099511628211ull;}
                    s.ubo0=hh;s.ubo0_seen=1;
                }
            }
            ++s.ops;
        }
        pc+=hd.size;
    }
}
static void plan(int n,const uint64_t *handles){
    keys.clear();hashes.clear();flags.clear();segs.clear();
    // ⚠ 窗口 serial 还没登记（第一帧之前）⇒ 根本不知道哪趟是上屏合成趟，保护等于没开。
    //   np723a 就是在这里 SIGBUS：早期帧把合成趟跳了。⇒ 未登记一律不跳。
    if(win_n==0){planned=0;segs.clear();return;}
    uint64_t steps=0;
    for(int i=0;i<n;i++)plan_walk(reinterpret_cast<const uint8_t*>(handles[i]),0,steps);
    seg_close();
    planned=0;
    for(size_t i=0;i<segs.size();i++){
        keys.push_back(segs[i].key);hashes.push_back(segs[i].hash);
        uint8_t skip=0;
        if(is_window_serial(segs[i].ser0)){flags.push_back(0);continue;}   // 末尾合成/窗口趟：永不跳
        if(segs[i].uploads){flags.push_back(0);continue;}                   // 含资源上传的 pass：永不跳
        if(!prev_keys.empty()&&segs[i].draws){          // 没有绘制类 op 的 pass 跳不跳都一样，别计入
            for(size_t j=0;j<prev_keys.size();j++)
                if(prev_keys[j]==segs[i].key&&prev_hashes[j]==segs[i].hash
                   &&(j>=prev_ubo0.size()||prev_ubo0[j]==segs[i].ubo0)){skip=1;break;}
        }
        flags.push_back(skip);planned+=skip?1u:0u;
    }
    // 同轮 A/B：DM_FRAME_SKIP_ALT=1 时偶数提交不跳、奇数提交照跳（画面静止时两者像素必须相同）。
    static int alt=-2;
    if(alt==-2){const char*e=getenv("DM_FRAME_SKIP_ALT");alt=(e&&*e=='1')?1:0;if(alt)xbg_logf("【跳过】ALT 交替模式：偶数提交不跳作为对照");}
    if(alt&&(submits%2)==0){for(size_t i=0;i<flags.size();i++)flags[i]=0;planned=0;}
    // 周期性强制全画兜底（缺省每 8 次提交一次）：把"结构对得上但内容变了"的漏网之鱼限制在 N 帧内。
    // 照 Metal 世界冻结的"第一帧必是完整帧"（nvn_mtl.m:5389）再加一层保险。
    // 时间相位 A/B（同包、同一轮里自动交替）：DM_FRAME_SKIP_PHASE_MS=20000 ⇒ 每 20 s 在
    // 「跳过 / 对照」之间切换。帧率对比因此是**同轮同类**，不用每次改 env 重建（省下 90 s 构建）。
    static int phase_ms=-2;
    if(phase_ms==-2){const char*e=getenv("DM_FRAME_SKIP_PHASE_MS");phase_ms=e?atoi(e):0;
        if(phase_ms>0)xbg_logf("【跳过】时间相位 A/B：每 %d ms 在跳过/对照之间切换",phase_ms);}
    if(phase_ms>0){
        static int last=-1;
        int st=int((timing_clock()/1000000ull)/uint64_t(phase_ms))%2;
        if(st!=last){last=st;xbg_logf("【跳过】相位切换 → %s（%llu ms）",st?"对照(不跳)":"跳过",(unsigned long long)(timing_clock()/1000000ull));}
        if(st){for(size_t i=0;i<flags.size();i++)flags[i]=0;planned=0;}
    }
    static int every=-2;
    if(every==-2){const char*e=getenv("DM_FRAME_SKIP_EVERY");every=e?atoi(e):8;if(every>0)xbg_logf("【跳过】每 %d 次提交强制全画一次（DM_FRAME_SKIP_EVERY）",every);}
    if(every>0&&(submits%uint64_t(every))==0){for(size_t i=0;i<flags.size();i++)flags[i]=0;planned=0;}
}
static void exec_begin(){idx=(size_t)-1;armed=!flags.empty()&&planned>0;skipped_ops=0;skipped_passes=0;}
static void exec_pass(const uint8_t *p,size_t n){
    if(!armed)return;
    ++idx;
    if(idx>=keys.size()||rt_key(p,n)!=keys[idx]){
        armed=false;++desync;
        xbg_logf("【跳过】pass 身份对不上 ⇒ 本提交放弃跳过（idx=%zu planned=%llu key_plan=%016llx key_exec=%016llx）",
            idx,(unsigned long long)planned,(unsigned long long)(idx<keys.size()?keys[idx]:0ull),(unsigned long long)rt_key(p,n));
    }
}
static bool exec_skip(){return armed&&idx<flags.size()&&flags[idx];}
static void count_skip(){++skipped_ops;}
static void exec_report(uint64_t frame){
    if(!enabled())return;
    ++submits;
    static int altlog=-2;
    if(altlog==-2){const char*e=getenv("DM_FRAME_SKIP_ALT");altlog=(e&&*e=='1')?1:0;}
    if(skipped_ops||planned||altlog)   // ALT 模式下对照帧（planned=0）也必须记，否则无法配对
        xbg_logf("FRAMESKIP frame=%llu presents=%llu passes=%zu planned=%llu skipped_ops=%llu armed=%d desync=%llu ctrl=%d 累计=%llu",
            (unsigned long long)frame,(unsigned long long)stats.presents.load(),segs.size(),
            (unsigned long long)planned,(unsigned long long)skipped_ops,int(armed),(unsigned long long)desync,
            int(altlog&&planned==0),(unsigned long long)submits);
    prev_ubo0.clear();for(size_t i=0;i<segs.size();i++)prev_ubo0.push_back(segs[i].ubo0);
    prev_keys.swap(keys);prev_hashes.swap(hashes);
}
}
// Graphics thread: retire submitted slots whose fence already signalled (non-blocking) so a game
// thread blocked on an older fence is released while the next frame is still being translated.
void poll_retire(){
    bool any=false;for(auto &s:slots)if(s.pending && !s.recording && vkGetFenceStatus(cfg.device,s.fence)==VK_SUCCESS){retire_slot(s);any=true;}
    if(any){{std::lock_guard<std::mutex> l(gfx_job_mutex);}gfx_idle_cv.notify_all();}
}
void execute(const uint8_t *pc,unsigned depth,uint64_t &budget){
    need(depth<64,"NVN command CALL nesting limit exceeded");
    if(depth==0)g_shader_skip_abort=false;                 // 每帧重新开始判 | restart the decision each frame
    for(;;){need(budget++<10000000,"NVN command count/cycle limit exceeded");
        if(g_shader_skip_abort)return;                     // np529：本帧缺过着色器 ⇒ 剩余命令全丢，保住这一帧能走到 present
        if(gfx_async && (budget&63)==0)poll_retire();
        // Bound host arena use per frame slot; only at command boundaries.
        {Slot &cur=slots[slot_index];if(cur.recording && cur.stream.used_total+cur.readback.used_total>arena_flush_bytes){SiteScope site(SITE_ARENA);submit_slots(false);}}
        size_t available=command_available(pc);need(available>=8,"Truncated command header");op_hdr h;memcpy(&h,pc,8);need(h.size>=8 && !(h.size&7) && h.size<=available,"NVN command size outside pool");const uint8_t *p=pc+8;size_t n=h.size-8;cur_op=h.op;
        dispatch_set_last((unsigned)h.op,(unsigned)h.size);
        dispatch_op_trace((unsigned)h.op,(unsigned)h.size,depth);
        // np715 阶段 2：匹配 pass 的绘制类 op 直接跳过（默认关）。只跳 S 类，M/M' 照常。
        if(frameskip::enabled()){
            if(h.op==OP_SET_RT)frameskip::exec_pass(p,n);
            else if(frameskip::s_op(h.op)&&frameskip::exec_skip()){frameskip::count_skip();pc+=h.size;continue;}
        }
        switch(h.op){
        case OP_END:return;
        case OP_BEGIN:break;
        case OP_JUMP:pc=reinterpret_cast<const uint8_t*>(payload<uint64_t>(p,n));continue;
        case OP_CALL:execute(reinterpret_cast<const uint8_t*>(payload<uint64_t>(p,n)),depth+1,budget);break;
        case OP_SIGNAL_EVENT:{need(n>=16,"Truncated event");uint64_t addr;uint32_t mode,v;memcpy(&addr,p,8);memcpy(&mode,p+8,4);memcpy(&v,p+12,4);need(addr,"Null event address");SiteScope site(SITE_EVENT);
            if(defer_events){   // hand the work so far to the GPU without waiting; publish when that submission retires
                Slot &cur=slots[slot_index];
                if(defer_events==2 && cur.recording && cur.ops){recording_events.push_back({addr,mode,v,&cur});++events_deferred;evguard::guard(addr);break;}
                if(cur.recording && cur.ops)submit_slots(false);
                Slot &last=slots[slot_index^1];   // most recent submission (just made, or the previous one)
                if(last.pending){pending_events.push_back({addr,mode,v,&last});++events_deferred;evguard::guard(addr);}else apply_event(addr,mode,v);
                break;}
            complete_all();cpu_access(addr,4,10);snapshot_invalidate(addr,4);auto ptr=reinterpret_cast<uint32_t*>(addr);if(mode==NVN_EVENT_SIGNAL_MODE_ADD)__atomic_fetch_add(ptr,v,__ATOMIC_SEQ_CST);else{need(mode==NVN_EVENT_SIGNAL_MODE_WRITE,"Unknown event mode");__atomic_store_n(ptr,v,__ATOMIC_SEQ_CST);}break;}
        case OP_COPY_B2B:{need(n>=24,"Truncated buffer copy");uint64_t a[3];memcpy(a,p,24);need(a[0]&&a[1]&&a[2]<=256ull*1024*1024,"Invalid buffer copy");if(gpu_buffer_copy(a[0],a[1],a[2]))break;cpu_access(a[0],a[2],11);invalidate_cpu_range(a[1],a[2]);memmove(reinterpret_cast<void*>(a[1]),reinterpret_cast<void*>(a[0]),a[2]);break;}
        case OP_UPDATE_UB:{need(n>=16,"Truncated uniform update");uint64_t a[2];memcpy(a,p,16);need(a[0] && a[1]<=n-16,"Invalid uniform update");
            // np541：崩点取证。np541 真机：第一条 UPDATE_UB 就往 0xbd40466d98 写（没有映射过）⇒ 直接 SEGV。
            //   这里把目标地址、长度、该页是否已映射打出来（前 8 次），再和 BIND_UBO 的地址对照，
            //   就能分清"游戏给的地址本来就在未映射区"和"录制/回放把地址读错了"。
            // np541: crash-site evidence. On device the very first UPDATE_UB wrote to 0xbd40466d98 (never mapped) and
            //   SEGV'd. Log the destination, length and whether that page is mapped (first 8 times) and compare with the
            //   BIND_UBO addresses: that separates "the game's address is outside anything we mapped" from "recording /
            //   replay read the address wrong".
            {static int nub;if(nub<8){++nub;void *pg=(void*)(uintptr_t)(a[0]&~0xfffull);unsigned char mv=0;
                xbg_logf("UBUPDATE #%d dst=%#llx n=%llu page_mapped=%d",nub,(unsigned long long)a[0],(unsigned long long)a[1],mincore(pg,0x1000,&mv)==0);}}
            invalidate_cpu_range(a[0],a[1]);memcpy(reinterpret_cast<void*>(a[0]),p+16,a[1]);break;}
        case OP_SET_RT:state.rt=payload<p_set_rt>(p,n);break;
        case OP_CLEAR_COLOR:{auto a=payload<p_clear_color>(p,n);clear_attachment(&a,nullptr);break;}
        case OP_COUNT_+9:{auto a=payload<xbg_typed_clear>(p,n);clear_attachment(&a.color,nullptr,a.scalar);break;}
        case OP_CLEAR_DS:{auto a=payload<p_clear_ds>(p,n);clear_attachment(nullptr,&a);break;}
        case OP_VIEWPORTS:{auto a=payload<p_viewports>(p,n);need(a.first==0 && a.n==1,"Multiple viewports unsupported");state.viewport.x=a.v[0][0];state.viewport.y=a.v[0][1];state.viewport.width=a.v[0][2];state.viewport.height=a.v[0][3];break;}
        case OP_SCISSOR:{auto a=payload<p_scissor>(p,n);need(a.w>=0 && a.h>=0,"Invalid scissor dimensions");state.scissor={{a.x,a.y},{uint32_t(a.w),uint32_t(a.h)}};break;}
        case OP_DEPTH_RANGE:{auto a=payload<p_depth_range>(p,n);state.viewport.minDepth=a.n;state.viewport.maxDepth=a.f;break;}
        case OP_BIND_PROGRAM:{CPU_SCOPE(Program);auto a=payload<p_bind_program>(p,n);need(!a.prog || a.prog->nstage<=6,"Invalid bound program");for(unsigned st=0;st<6;st++)if(a.stages&(1u<<nvn_stage[st]))state.shader[st]=nullptr;if(!a.prog)break;const std::array<Shader*,6> &loaded=program_shaders(a.prog);for(unsigned i=0;i<a.prog->nstage;i++){Shader *s=loaded[i];if(!s)continue;if(a.stages&(1u<<nvn_stage[s->stage]))state.shader[s->stage]=s;}break;}
        case OP_BIND_STATE:{auto a=payload<p_bind_state>(p,n);switch(a.kind){case 0:{blend_st b;memcpy(&b,a.bytes,sizeof(b));state.blend[b.target]=b;break;}case 1:memcpy(&state.color,a.bytes,sizeof(state.color));break;case 2:memcpy(&state.mask,a.bytes,sizeof(state.mask));break;case 3:memcpy(&state.ms,a.bytes,sizeof(state.ms));break;case 4:memcpy(&state.poly,a.bytes,sizeof(state.poly));break;case 5:memcpy(&state.ds,a.bytes,sizeof(state.ds));break;default:xbg_fatalf("Unknown NVN state object %d",a.kind);}break;}
        case OP_BIND_VATTR:{auto a=variable_payload<p_bind_vattr>(p,n);need(a.first==0 && a.n>=0 && a.n<=16 && n>=offsetof(p_bind_vattr,st)+size_t(a.n)*4,"Invalid vertex attribute record");need(xbg_vertex_attributes_replace(state.attrib,reinterpret_cast<const uint32_t*>(a.st),a.n)==0,"Invalid vertex attribute replacement");break;}
        case OP_BIND_VSTREAM:{auto a=variable_payload<p_bind_vstream>(p,n);need(a.first>=0 && a.n>=0 && a.first+a.n<=16 && n>=offsetof(p_bind_vstream,st)+size_t(a.n)*sizeof(vs_st),"Invalid vertex stream record");memcpy(state.stream+a.first,a.st,a.n*sizeof(vs_st));break;}
        case OP_BIND_VBUF:{auto a=variable_payload<p_bind_vbuf>(p,n);need(a.first>=0 && a.n>=0 && a.first+a.n<=16 && n>=offsetof(p_bind_vbuf,r)+size_t(a.n)*sizeof(NVNbufferRange),"Invalid vertex buffer record");memcpy(state.vb+a.first,a.r,a.n*sizeof(NVNbufferRange));break;}
        case OP_BIND_UBO:case OP_BIND_SSBO:{auto a=payload<p_bind_buf>(p,n);need(a.stage>=0 && a.stage<6 && a.index>=0 && a.index<16,"Invalid shader buffer slot");
            // np603：**每个 (阶段,槽) 被绑过多少次**。着色器清单里写着"要 slot 2"，而客体侧 slot 2 是空的，
            // 这时必须分清两件事：① 游戏从头到尾就没绑过这个槽（那多半是导出器给的槽号不对，或这条 shader
            // 的那个 UBO 根本没人用）；② 平时会绑、只是这一帧漏了（那要往"谁清掉了"查）。
            // 计数是零成本的（一次自增），只在 fatal 那条路径上打出来。
            // np603: how often each (stage, slot) has ever been bound. The shader bundle asks for slot 2 while the
            // guest side has it empty, and two very different stories fit: ① the game never bound that slot at all
            // (the exporter's slot number is wrong, or that UBO is simply unused) ② it is bound normally and only
            // this frame skipped it (then look for who cleared it). One increment per bind, dumped only on the
            // fatal path.
            if(a.index<16)++bind_hits[a.stage][a.index];
            // np541：与 UBUPDATE 成对取证——看游戏绑定给着色器的缓冲区地址是不是也在未映射区。
            // np541: pairs with UBUPDATE -- check whether the buffer addresses bound to the shaders are also unmapped.
            {static int nb;if(nb<16){++nb;void *pg=(void*)(uintptr_t)(a.addr&~0xfffull);unsigned char mv=0;
                xbg_logf("BINDBUF #%d op=%u st=%d idx=%d addr=%#llx size=%llu page_mapped=%d",nb,(unsigned)h.op,a.stage,a.index,(unsigned long long)a.addr,(unsigned long long)a.size,mincore(pg,0x1000,&mv)==0);}}
            (h.op==OP_BIND_UBO?state.ub:state.sb)[a.stage][a.index]={a.addr,a.size};++buf_gen[a.stage];break;}
        case OP_BIND_TEX:case OP_BIND_SAMP:case OP_BIND_IMG:{auto a=variable_payload<p_bind_handles>(p,n);need(a.stage>=0 && a.stage<6 && a.first>=0 && a.n>=0 && a.n<=32 && a.first+a.n<=(h.op==OP_BIND_IMG?16:128) && n>=offsetof(p_bind_handles,h)+size_t(a.n)*8,"Invalid descriptor binding record");uint64_t *dst=h.op==OP_BIND_IMG?state.img[a.stage]:h.op==OP_BIND_SAMP?state.samp[a.stage]:state.tex[a.stage];memcpy(dst+a.first,a.h,a.n*8);++tex_gen[a.stage];break;}
        case OP_STENCIL:{auto a=payload<p_stencil>(p,n);need(a.face>=0 && a.face<=3 && a.which>=0 && a.which<=2,"Invalid stencil state");for(unsigned i=0;i<2;i++)if(a.face&(1<<i)){if(a.which==0)state.stencil_compare[i]=a.value_mask;else if(a.which==1)state.stencil_write[i]=a.mask;else state.stencil_ref[i]=a.ref;}break;}
        case OP_POLY_OFFSET:{auto a=payload<p_poly_offset>(p,n);state.bias[0]=a.factor;state.bias[1]=a.units;state.bias[2]=a.clamp;break;}
        case OP_DRAW:{auto a=payload<p_draw>(p,n);draw(a.prim,a.first,a.count,nullptr);break;}
        case OP_DRAW_IDX:{auto a=payload<p_draw_idx>(p,n);
            // np662：ibuf==0 = DrawArraysInstanced 的"非索引"约定（hle_nvn.c 同日新增）——
            // 走非索引实例化路径，不碰索引缓冲。first 借 base_vertex 携带（p_draw_idx 无 first 字段）。
            // np662: ibuf==0 marks the non-indexed DrawArraysInstanced convention (added same day in
            //   hle_nvn.c) -- route to the non-indexed instanced path; `first` rides in base_vertex
            //   (p_draw_idx has no first field).
            if(!a.ibuf){draw(a.prim,a.base_vertex,a.count,nullptr,uint32_t(a.instances>0?a.instances:1));break;}
            draw(a.prim,0,a.count,&a);break;}
        case OP_DISPATCH:dispatch(payload<p_dispatch>(p,n));break;
        // np637：**内存屏障不再断 pass**。这一句原来是 `Commands c`——它的非 draw 构造与 `finish()` 各调一次
        //   `end_open_pass()`，于是游戏每个 `nvnCommandBufferBarrier` 都把 8 附件 1600x900 的 pass 关掉重开。
        //   场景帧因此变成 ~500 个 1-draw pass、附件带宽 ~10 GB/帧、GPU 1.1–1.6 s/帧（np634 逐 pass 剖析：
        //   498 个 pass 里 460 个只画 1 个 draw，`pass_ends` 里 op28(BARRIER)=17.5/帧 ≥ op26(DRAW_IDX)=14.0/帧）。
        //   屏障只是内存屏障（不带 layout 转换），在 render pass 内部发是合法的；pass 开着时把阶段掩码收窄到
        //   图形阶段（ALL_COMMANDS 含 pass 外的阶段，验证层会报）。
        // np637: a memory barrier must not end the pass. This used to be `Commands c`, whose non-draw ctor and
        //   finish() each call end_open_pass(), so every nvnCommandBufferBarrier closed and reopened an 8-attachment
        //   1600x900 pass -- the scene frame became ~500 single-draw passes, ~10 GB/frame of attachment traffic and
        //   1.1-1.6 s of GPU time (np634 per-pass profile: 460 of 498 passes had exactly one draw, and pass_ends
        //   showed op28(BARRIER)=17.5/frame vs op26(DRAW_IDX)=14.0/frame). A memory barrier carries no layout
        //   transition, so issuing it inside a render pass is legal; while a pass is open the stage mask is narrowed
        //   to graphics stages (ALL_COMMANDS includes stages outside the pass and the validation layer complains).
        case OP_BARRIER:{Slot &s=recording_slot();++perf.ops;++s.ops;
           // np712b：无写则跳——见 g_gpu_wrote 注释。emit 时清标志（可见性已保证）。
           // np712b: skip when nothing was written (see g_gpu_wrote). Clear the flag on emit.
           if(!g_gpu_wrote){++g_barrier_skip;break;}
           g_barrier_site="op28-inpass";full_barrier_in_pass(s.cmd);g_gpu_wrote=false;++g_barrier_emit;break;}
        case OP_SET_TEX_POOL:{auto a=payload<p_set_pool>(p,n);need(a.pool,"Null texture pool");state.texture_pool=*a.pool;break;}
        case OP_SET_SAMP_POOL:{auto a=payload<p_set_pool>(p,n);need(a.pool,"Null sampler pool");state.sampler_pool=*a.pool;break;}
        case OP_COPY_B2T:copy_texture_buffer(payload<p_copy_b2t>(p,n),false);break;
        case OP_COPY_T2B:copy_texture_buffer(payload<p_copy_b2t>(p,n),true);break;
        case OP_COPY_T2T:copy_texture(payload<p_copy_t2t>(p,n));break;
        case OP_CLEAR_TEX:clear_texture(payload<p_clear_tex>(p,n));break;
        case OP_COPY_STRIDE:state.copy_stride=payload<p_copy_stride>(p,n).stride;need(state.copy_stride>=0,"Negative copy stride");break;
        case OP_COPY_IMAGE_STRIDE:state.copy_image_stride=payload<p_copy_stride>(p,n).stride;need(state.copy_image_stride>=0,"Negative copy image stride");break;
        case OP_BLEND_COLOR:{auto a=payload<p_blend_color>(p,n);memcpy(state.blend_color,a.rgba,16);break;}
        case OP_COUNT_:{uint32_t size=payload<uint32_t>(p,n);need(size && size<=properties.limits.maxTessellationPatchSize,"Invalid tessellation patch size");state.patch_size=size;break;}
        case OP_COUNT_+1:{auto s=payload<sync_t>(p,n);need(s.fenced && s.fence<=submitted+1,"Command wait on unresolved external sync");
            // GPU-side wait on an earlier fence of this same in-order queue: already
            // satisfied by submission order (every operation starts with a barrier).
            break;}
        case OP_COUNT_+2:{uint64_t addr=payload<uint64_t>(p,n);need(addr,"Null indirect dispatch address");cpu_access(addr,12,5);uint32_t xyz[3];memcpy(xyz,reinterpret_cast<void*>(addr),12);p_dispatch a{int32_t(xyz[0]),int32_t(xyz[1]),int32_t(xyz[2]),0};dispatch(a);break;}
        case OP_COUNT_+3:execute_indirect(payload<xbg_indirect>(p,n),false);break;
        case OP_COUNT_+7:execute_draw_call(payload<xbg_draw_call>(p,n));break;
        case OP_COUNT_+8:execute_indirect(payload<xbg_indirect>(p,n),true);break;
        case OP_COUNT_+4:{auto a=payload<xbg_wait_event>(p,n);need(a.address && a.mode<=1,"Invalid event wait");
            for(auto *list:{&pending_events,&recording_events})if(std::any_of(list->begin(),list->end(),[&](const PendingEvent &e){return e.addr==a.address;})){complete_all();break;}
            auto deadline=std::chrono::steady_clock::now()+std::chrono::nanoseconds(cfg.fence_timeout_ns);for(;;){uint32_t value=__atomic_load_n(reinterpret_cast<uint32_t*>(a.address),__ATOMIC_ACQUIRE);if(a.mode==0?value==a.value:int32_t(value-a.value)>=0)break;need(std::chrono::steady_clock::now()<deadline,"NVN event wait timeout");std::this_thread::yield();}break;}
        case OP_COUNT_+5:state.bounds=payload<xbg_depth_bounds>(p,n);break;
        case OP_COUNT_+6:resolve_texture(payload<p_copy_t2t>(p,n));break;
        case OP_RESET_COUNTER:{uint64_t type=payload<uint64_t>(p,n);need(type==NVN_COUNTER_TYPE_SAMPLES_PASSED,"Unsupported pipeline counter reset");state.samples_passed=0;end_occlusion_query();++counter_epoch;state.count_samples=true;++counter_ops[0];break;}   // the open run belongs to the old epoch
        case OP_REPORT_COUNTER:{need(n>=16,"Truncated counter report");uint64_t type,addr;memcpy(&type,p,8);memcpy(&addr,p+8,8);need(addr,"Null counter output");need(type<=NVN_COUNTER_TYPE_TIMESTAMP_TOP,"NVN counter type out of range");if(type!=NVN_COUNTER_TYPE_TIMESTAMP&&type!=NVN_COUNTER_TYPE_SAMPLES_PASSED){static bool once=false;if(!once){once=true;xbg_logf("NVN pipeline-statistics counter type %llu -> counter=0 (this backend produces no pipeline statistics; stats only, safe to zero)",(unsigned long long)type);}}need(timestamp_bits,"Queue family does not support GPU timestamps");++counter_ops[type==NVN_COUNTER_TYPE_TIMESTAMP?1:2];if(type==NVN_COUNTER_TYPE_SAMPLES_PASSED)samples_report_seen=true;
            PendingReport r{addr,type,counter_epoch,epochs[counter_epoch].size(),nullptr,VK_NULL_HANDLE,0,0,false};
            if(type==NVN_COUNTER_TYPE_TIMESTAMP && !heavy_timestamps && !ab_off("timestamps")){
                // A bottom-of-pipe timestamp already waits for all earlier work and may be written inside a render pass: no pass
                // end and no full barriers, so a timestamp report no longer splits a pass on a tiling GPU (4 per D2R frame).
                // Samples reports keep the pass end below: it closes the occlusion run they sum.
                timestamp_query(r.ts,r.ts_index,r.slot);   // a new pool is reset outside the pass first
                Slot &ts=recording_slot();++ts.ops;++perf.ops;vkCmdWriteTimestamp(ts.cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,r.ts,r.ts_index);}
            else{Commands c;timestamp_query(r.ts,r.ts_index,r.slot);vkCmdWriteTimestamp(c.command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,r.ts,r.ts_index);c.finish();}
            pending_reports.push_back(r);break;}
        default:xbg_fatalf("NVN opcode %u at %#llx not implemented",h.op,(unsigned long long)uintptr_t(pc));
        }
        pc+=h.size;
    }
}
PerfCounts perf_counts(){return {pipeline_objects.size(),layout_cache.size(),sampler_cache.size(),render_pass_cache.size(),images.size(),host_segments.size(),host_bytes};}
void destroy_slots(){
    poolreset::drain();for(auto &s:slots)s.reset_queued=false;
    texel_evict(0,true);texel_evict(1,true);
    for(auto &s:slots){s.retire.run();release_descriptors(s);for(auto &v:s.texel_views)vkDestroyBufferView(cfg.device,v.second,nullptr);s.texel_views.clear();for(auto &f:s.framebuffers)vkDestroyFramebuffer(cfg.device,f.second,nullptr);s.framebuffers.clear();for(auto p:s.pools)vkDestroyDescriptorPool(cfg.device,p,nullptr);s.pools.clear();s.pool_next=0;s.stream.release();s.readback.release();s.viewed.clear();s.guard.clear();s.pro_ops.clear();s.pro=VK_NULL_HANDLE;if(s.fence)vkDestroyFence(cfg.device,s.fence,nullptr);if(s.timestamps)vkDestroyQueryPool(cfg.device,s.timestamps,nullptr);if(s.prof_pool)vkDestroyQueryPool(cfg.device,s.prof_pool,nullptr);for(auto q:s.occlusion_pools)vkDestroyQueryPool(cfg.device,q,nullptr);for(auto q:s.timestamp_pools)vkDestroyQueryPool(cfg.device,q,nullptr);if(s.pool)vkDestroyCommandPool(cfg.device,s.pool,nullptr);s=Slot{};}
    for(VkFence f:split_fences)if(f)vkDestroyFence(cfg.device,f,nullptr);   // np629
    slot_index=0;snapshot_reset();
}
} // namespace

// Zero-copy probe (C097): can exportable host-visible Vulkan memory be mmap-ed at an address we choose?
// If yes, guest NVN memory pools can be re-backed by GPU-visible memory in place (iOS: Metal buffers on pool memory).
void zero_copy_probe(){
    for(uint32_t i=0;i<memories.memoryTypeCount;i++)xbg_logf("NVN caps memtype %u flags=%#x heap=%u heapsize=%lluMB",i,memories.memoryTypes[i].propertyFlags,memories.memoryTypes[i].heapIndex,(unsigned long long)(memories.memoryHeaps[memories.memoryTypes[i].heapIndex].size>>20));
    auto getfd=reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(cfg.device,"vkGetMemoryFdKHR"));
    if(!getfd){xbg_logf("NVN caps zerocopy getfd=missing");return;}
    const VkDeviceSize n=4<<20;
    for(uint32_t want:{uint32_t(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT),uint32_t(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}){
        uint32_t type=UINT32_MAX;for(uint32_t i=0;i<memories.memoryTypeCount;i++)if((memories.memoryTypes[i].propertyFlags&want)==want){type=i;break;}
        if(type==UINT32_MAX){xbg_logf("NVN caps zerocopy flags=%#x type=none",want);continue;}
        for(VkExternalMemoryHandleTypeFlagBits ht:{VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT}){
            VkExportMemoryAllocateInfo ex{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};ex.handleTypes=ht;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.pNext=&ex;ai.allocationSize=n;ai.memoryTypeIndex=type;VkDeviceMemory mem{};
            VkResult r=vkAllocateMemory(cfg.device,&ai,nullptr,&mem);if(r!=VK_SUCCESS){xbg_logf("NVN caps zerocopy flags=%#x handle=%#x alloc=%d",want,unsigned(ht),r);continue;}
            VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};gi.memory=mem;gi.handleType=ht;int fd=-1;r=getfd(cfg.device,&gi,&fd);
            int ok=0;void *vk=nullptr;
            if(r==VK_SUCCESS && fd>=0){
                void *at=mmap(nullptr,n,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
                void *m=at==MAP_FAILED?MAP_FAILED:mmap(at,n,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,fd,0);
                if(m!=MAP_FAILED && vkMapMemory(cfg.device,mem,0,n,0,&vk)==VK_SUCCESS){
                    memset(m,0x5a,n);static_cast<uint8_t*>(m)[12345]=0x77;
                    if(!(want&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){VkMappedMemoryRange rr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};rr.memory=mem;rr.size=VK_WHOLE_SIZE;vkInvalidateMappedMemoryRanges(cfg.device,1,&rr);}
                    ok=static_cast<uint8_t*>(vk)[12345]==0x77 && static_cast<uint8_t*>(vk)[0]==0x5a ? 1 : 2;
                    vkUnmapMemory(cfg.device,mem);
                }
                if(at!=MAP_FAILED)munmap(at,n);close(fd);
            }
            xbg_logf("NVN caps zerocopy flags=%#x type=%u handle=%#x getfd=%d mmap_fixed_shared=%s",want,type,unsigned(ht),r,ok==1?"OK":ok==2?"MISMATCH":"FAIL");
            vkFreeMemory(cfg.device,mem,nullptr);
        }
    }
}
extern "C" int xbg_game_initialize(const xbg_game_config *c,char *error,size_t bytes){
    if(!c || c->size!=sizeof(*c) || !c->physical || !c->device || !c->queue || !c->shader_directory || !c->fence_timeout_ns){if(error&&bytes)snprintf(error,bytes,"Invalid xbg_game_config");return -1;}
    std::lock_guard<std::recursive_mutex> lock(mutex);if(initialized){if(error&&bytes)snprintf(error,bytes,"Game graphics already initialized");return -1;}
#ifdef DM_COMPLETION_WAIT_SELFTEST
    if(const char *test=getenv("DM_GFX_COMPLETION_WAIT_TEST"))if(!strcmp(test,"1")){
        // 隔离测试在资源/线程初始化前只跑一次；失败不准进入游戏，不污染真实 completed。
        // Run isolated tests once before resource/thread initialization; failure blocks the game and never changes the real completed counter.
        static bool tested=false;static int result;static char report[256];
        if(!tested){result=xbg_completion_wait_selftest(report,sizeof report);tested=true;}
        if(c->log)c->log(c->userdata,report);else fprintf(stderr,"[xbg] %s\n",report);
        if(result){if(error&&bytes)snprintf(error,bytes,"%s",report);return -1;}
    }
#endif
    // 沿用宿主保证初始化/关闭时没有活跃等待者的前提；不引入重置通知或新会话世代。
    // Retain the host's quiescent initialize/shutdown requirement; do not add reset notifications or session generations.
    fold_completion_wait_stats();completion_wait_enabled=completion_wait_default;
    if(const char *mode=getenv("DM_GFX_COMPLETION_WAIT")){if(!strcmp(mode,"0"))completion_wait_enabled=false;else if(!strcmp(mode,"1"))completion_wait_enabled=true;}
    {std::string dir=c->shader_directory?c->shader_directory:"";size_t cut=dir.find_last_of('/');FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-legacy-waitall").c_str(),"rb");if(m){legacy_waitall=fgetc(m)=='1';fclose(m);}}
    cfg=*c;xbg_logf("NVN completion_wait enabled=%d legacy_waitall=%d",int(completion_wait_enabled),int(legacy_waitall));vkGetPhysicalDeviceProperties(cfg.physical,&properties);vkGetPhysicalDeviceMemoryProperties(cfg.physical,&memories);{char line[512];int at=snprintf(line,sizeof line,"NVN caps memory types");for(uint32_t i=0;i<memories.memoryTypeCount && at<int(sizeof line)-24;i++)at+=snprintf(line+at,sizeof line-at," %u:%#x/h%u",i,unsigned(memories.memoryTypes[i].propertyFlags),memories.memoryTypes[i].heapIndex);xbg_logf("%s",line);}
    VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};pp.pNext=&subgroup;vkGetPhysicalDeviceProperties2(cfg.physical,&pp);
    VkPhysicalDeviceFeatures available;vkGetPhysicalDeviceFeatures(cfg.physical,&available);const VkBool32 *want=reinterpret_cast<const VkBool32*>(&cfg.enabled_core),*have=reinterpret_cast<const VkBool32*>(&available);for(size_t i=0;i<sizeof(available)/sizeof(VkBool32);i++)if(want[i]&&!have[i]){if(error&&bytes)snprintf(error,bytes,"Requested enabled feature %zu is unsupported",i);return -1;}
    uint32_t count;vkGetPhysicalDeviceQueueFamilyProperties(cfg.physical,&count,nullptr);std::vector<VkQueueFamilyProperties> queues(count);vkGetPhysicalDeviceQueueFamilyProperties(cfg.physical,&count,queues.data());if(cfg.queue_family>=count || (queues[cfg.queue_family].queueFlags&(VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT))!=(VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT)){if(error&&bytes)snprintf(error,bytes,"Graphics+compute queue family required");return -1;}
    timestamp_bits=queues[cfg.queue_family].timestampValidBits;
    {const auto &l=properties.limits;VkDeviceSize a=256;for(VkDeviceSize x:{l.minUniformBufferOffsetAlignment,l.minStorageBufferOffsetAlignment,l.minTexelBufferOffsetAlignment,l.optimalBufferCopyOffsetAlignment,l.nonCoherentAtomSize,VkDeviceSize(16)})while(a<x)a<<=1;arena_alignment=a;}
    {std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');std::string marker=(cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-sync-mode";FILE *m=fopen(marker.c_str(),"rb");sync_mode=false;if(m){sync_mode=fgetc(m)=='1';fclose(m);}}
     {std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-defer-events").c_str(),"rb");defer_events=0;if(m){int c=fgetc(m);defer_events=c=='1'?1:c=='2'?2:0;fclose(m);}else if(const char *e=getenv("DM_GFX_DEFER_EVENTS")){int v=atoi(e);defer_events=v==1?1:v==2?2:0;}xbg_logf("NVN defer events=%d (marker gfx-defer-events)",int(defer_events));{FILE *eg=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-event-guard").c_str(),"rb");evguard::enabled=eg!=nullptr;if(eg)fclose(eg);if(evguard::enabled)xbg_logf("NVN event guard on (marker gfx-event-guard)");}{FILE *v=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-vcache").c_str(),"rb");vcache::enabled=false;if(v){vcache::enabled=fgetc(v)!='0';fclose(v);}else if(const char *e=getenv("DM_GFX_VCACHE"))vcache::enabled=atoi(e)!=0;xbg_logf("NVN vertex/index cross-frame cache=%d tracker=%d (marker gfx-vcache)",int(vcache::enabled),int(nvn_dirty_since!=nullptr));}
     {FILE *v=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-tex-dirty").c_str(),"rb");tex_dirty=false;if(v){tex_dirty=fgetc(v)!='0';fclose(v);}else if(const char *e=getenv("DM_GFX_TEX_DIRTY"))tex_dirty=atoi(e)!=0;
      xbg_logf("NVN texture checks through the dirty tracker=%d (marker gfx-tex-dirty)",int(tex_dirty));}{FILE *b=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-native-bc").c_str(),"rb");native_bc=false;if(b){native_bc=fgetc(b)!='0';fclose(b);}else if(const char *e=getenv("DM_GFX_NATIVE_BC"))native_bc=atoi(e)!=0;xbg_logf("NVN native BC textures=%d (marker gfx-native-bc)",int(native_bc));}
     {FILE *b=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-noload").c_str(),"rb");noload_enabled=false;if(b){noload_enabled=fgetc(b)!='0';fclose(b);}else if(const char *e=getenv("DM_GFX_NOLOAD"))noload_enabled=atoi(e)!=0;xbg_logf("NVN proven full-overwrite first draws skip the color load=%d (marker gfx-noload; A/B target noload)",int(noload_enabled));}{FILE *q=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-bright").c_str(),"rb");bright::enabled=true;if(q){bright::enabled=fgetc(q)!='0';fclose(q);}else if(const char *e=getenv("DM_GFX_BRIGHT"))bright::enabled=atoi(e)!=0;xbg_logf("NVN 3D brightening=%d (marker gfx-bright; A/B target bright)",int(bright::enabled));}{FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-perf-sample").c_str(),"rb");perf_sample=32;if(m){if(fscanf(m,"%u",&perf_sample)!=1)perf_sample=32;fclose(m);}else if(const char *e=getenv("DM_GFX_PERF_SAMPLE"))perf_sample=unsigned(atoi(e));perf_frame_timed=perf_sample!=0;perf_timed_frames=0;xbg_logf("NVN perf timing every %u frames (marker gfx-perf-sample; 0 = off)",perf_sample);}}
    for(auto &sl:slots)sl=Slot{};slot_index=0;pass_prof_enabled=false;pass_prof_next=300;pass_prof_period=600;pass_prof_until=UINT64_MAX;pass_prof_frame=UINT64_MAX;pass_prof_draws=0;pass_prof_queries=pass_prof_slots=pass_prof_skipped_slots=0;perf=Perf{};for(auto &c:perf_timing_before)c=XbgCpuTiming::Counter{};static_arena.cached=false;support_static.clear();zero_blocks.clear();format_property_cache.clear();for(auto &k:format_flat_known)k=false;
    VkPhysicalDeviceShaderDrawParametersFeatures draw_parameters{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES};VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};features.pNext=&draw_parameters;vkGetPhysicalDeviceFeatures2(cfg.physical,&features);if(cfg.enabled_shader_draw_parameters && !draw_parameters.shaderDrawParameters){if(error&&bytes)snprintf(error,bytes,"shaderDrawParameters unsupported");return -1;}
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.queueFamilyIndex=cfg.queue_family;cp.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;VkResult r=vkCreateCommandPool(cfg.device,&cp,nullptr,&command_pool);if(r!=VK_SUCCESS){if(error&&bytes)snprintf(error,bytes,"vkCreateCommandPool=%d",r);return r;}
    VkPipelineCacheCreateInfo cache{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};std::vector<uint8_t> cache_data=load_pipeline_cache();cache.initialDataSize=cache_data.size();cache.pInitialData=cache_data.empty()?nullptr:cache_data.data();r=vkCreatePipelineCache(cfg.device,&cache,nullptr,&pipeline_cache);if(r!=VK_SUCCESS){vkDestroyCommandPool(cfg.device,command_pool,nullptr);command_pool=VK_NULL_HANDLE;if(error&&bytes)snprintf(error,bytes,"vkCreatePipelineCache=%d",r);return r;}
    {std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');std::string base=cut==std::string::npos?dir:dir.substr(0,cut);
     const bool rewrites_fault=xbg_adreno_rewrites_fault(properties.vendorID,properties.deviceName);
     if(rewrites_fault)xbg_logf("NVN Adreno 740/750 (device \"%s\"): the quad-derivative and warp32 rewrites default off; a marker file still turns either on",properties.deviceName);
     auto sw=[&](const char *marker,const char *env){FILE *q=fopen((base+marker).c_str(),"rb");const bool present=q!=nullptr;int c=-1;if(q){c=fgetc(q);fclose(q);}return xbg_adreno_rewrite_option(present,c,getenv(env),rewrites_fault);};
     quad_derivatives=sw("/gfx-quad-derivatives","DM_GFX_QUAD_DERIVATIVES");
     warp32=sw("/gfx-warp32","DM_GFX_WARP32");
     xbg_logf("NVN quad shuffles to fine derivatives=%d (marker gfx-quad-derivatives)",int(quad_derivatives));
     xbg_logf("NVN 32-lane warps in wider subgroups=%d subgroup=%u (marker gfx-warp32)",int(warp32),subgroup.subgroupSize);
     {FILE *q=fopen((base+"/gfx-inline").c_str(),"rb");if(q){inline_helpers=fgetc(q)!='0';fclose(q);}else inline_helpers=xbg_adreno_compiler_needs_inline(properties.vendorID,properties.deviceName);
#ifndef XBG_HAVE_SPIRV_OPT
      if(inline_helpers)xbg_logf("NVN helper inlining is not built into this package; shaders load as they are");inline_helpers=false;
#endif
      xbg_logf("NVN shaders with subgroup helpers are inlined at load=%d (marker gfx-inline)",int(inline_helpers));}}
    {std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');std::string base=cut==std::string::npos?dir:dir.substr(0,cut);
     auto dm=[&](const char *marker,const char *env){FILE *q=fopen((base+marker).c_str(),"rb");bool on=false;if(q){on=fgetc(q)!='0';fclose(q);}else if(const char *e=getenv(env))on=atoi(e)!=0;return on;};
     ldc_fold=dm("/gfx-ldc-fold","DM_GFX_LDC_FOLD");forward_temps=dm("/gfx-forward-temps","DM_GFX_FORWARD_TEMPS");nan_minmax=dm("/gfx-nan-minmax","DM_GFX_NAN_MINMAX");
     xbg_logf("NVN NaN-ignoring min / max / clamp like Maxwell=%d (marker gfx-nan-minmax)",int(nan_minmax));
     xbg_logf("NVN translator output clean-up: component fold=%d (marker gfx-ldc-fold) forwarded temporaries=%d (marker gfx-forward-temps)",int(ldc_fold),int(forward_temps));
     {FILE *q=fopen((base+"/gfx-shader-reuse").c_str(),"rb");shader_reuse=0;if(q){int c=fgetc(q);shader_reuse=c=='0'?0:c=='2'?2:1;fclose(q);}else if(const char *e=getenv("DM_GFX_SHADER_REUSE"))shader_reuse=atoi(e);
      warm::keep_rewritten=shader_reuse!=0;xbg_logf("NVN shader loads take the warm-up's rewritten SPIR-V=%d (marker gfx-shader-reuse; 2 also rewrites and compares)",shader_reuse);}}
    pipeguard::load();xbg_module_id_hook=warm::module_id;{std::string lp=warm::list_path();FILE *nw=fopen((lp.substr(0,lp.find_last_of('/'))+"/gfx-no-warm").c_str(),"rb");warm::enabled=!nw;if(nw)fclose(nw);}warm::start();
    // np701：判死证据链上线——监视线程 + 看门狗转储的图形侧等待点（弱符号：不链宿主时静默跳过）。
    // np701: bring up the death-evidence chain -- the monitor thread and the wait-point dump for the
    //   watchdog (weak symbol: silently skipped when the host isn't linked in).
    deathguard::start();if(crash_set_wait_dump)crash_set_wait_dump(&deathguard::wait_dump);
    wtrack::enabled=false;state=State{};state.color.logic_op=3;state.color.alpha_func=8;for(unsigned i=0;i<8;i++)state.blend[i]={i,2,1,2,1,1,1};stats.reset();timing=XbgCpuTiming(timing_clock);pipeline_counts={};pipeline_before={};pipeline_tick=0;timing_active=false;timing_first=timing_last=0;live_profile_used=false;descriptor_detail=false;descriptor_timing=XbgDescriptorTiming{};descriptor_root=nullptr;draw_detail=false;draw_timing=XbgDrawTiming{};draw_root=nullptr;completed=0;last_attachment_trace=UINT64_MAX;sampler_mirror_clamp_enabled=VK_FALSE;failed=false;error_text[0]=0;initialized=true;
    {std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-sync-submit").c_str(),"rb");bool sync_submit=false;if(m){sync_submit=fgetc(m)=='1';fclose(m);}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-async-pool-reset").c_str(),"rb");poolreset::enabled=false;if(w){poolreset::enabled=fgetc(w)!='0';fclose(w);}else if(const char *e=getenv("DM_GFX_ASYNC_POOL_RESET"))poolreset::enabled=atoi(e)!=0;poolreset::start();xbg_logf("NVN async command pool reset=%d (marker gfx-async-pool-reset)",int(poolreset::enabled));}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-bindcache").c_str(),"rb");if(w){bind_cache_enabled=false;fclose(w);}}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-imgcache").c_str(),"rb");if(w){img_cache_enabled=false;fclose(w);}}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-pass-profile").c_str(),"rb");if(w){pass_prof_enabled=true;unsigned long long from=0,period=0,count=0;if(fscanf(w,"%llu %llu %llu",&from,&period,&count)>=2){pass_prof_next=from;pass_prof_period=period?period:1;if(count)pass_prof_until=from+count;}fclose(w);}}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-descmemo").c_str(),"rb");if(w){desc_memo_enabled=false;fclose(w);}xbg_logf("NVN descmemo enabled=%d",int(desc_memo_enabled));}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-dynpartial").c_str(),"rb");if(w){dyn_partial_enabled=false;fclose(w);}}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-xmemo").c_str(),"rb");if(w){xmemo_enabled=false;fclose(w);}xbg_logf("NVN cross-frame memo enabled=%d",int(xmemo_enabled));}{auto marker=[&](const char *name){FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+name).c_str(),"rb");int c=-1;if(w){c=fgetc(w);fclose(w);}return c;};persistent_fb_enabled=marker("/gfx-no-persistent-fb")<0;slot_pools_enabled=marker("/gfx-no-slot-pools")<0;readback_cached=marker("/gfx-readback-cached")=='1' || (marker("/gfx-readback-cached")<0 && getenv("DM_GFX_READBACK_CACHED") && *getenv("DM_GFX_READBACK_CACHED")=='1');texel_memo_enabled=marker("/gfx-no-texel-memo")<0;no_occlusion_queries=marker("/gfx-no-occlusion")=='1';samples_report_seen=false;open_occlusion=OpenOcclusion{};occlusion_queries_started=0;occlusion_batch_enabled=marker("/gfx-query-per-draw")!='1' && marker("/gfx-no-query-batch")<0;heavy_timestamps=marker("/gfx-heavy-timestamps")=='1';xbg_logf("NVN persistent framebuffers=%d slot command pools=%d cached readback=%d texel memo=%d (markers gfx-no-persistent-fb, gfx-no-slot-pools, gfx-readback-cached, gfx-no-texel-memo)",int(persistent_fb_enabled),int(slot_pools_enabled),int(readback_cached),int(texel_memo_enabled));}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-wtrack").c_str(),"rb");wtrack::marker=-1;if(w){wtrack::marker=fgetc(w)=='1'?1:0;fclose(w);}wtrack::enabled=wtrack::marker==1;}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-wtrack").c_str(),"rb");if(w){wtrack::enabled=false;wtrack::marker=0;fclose(w);}xbg_logf("NVN wtrack enabled=%d (markers gfx-no-wtrack, gfx-wtrack=0)",int(wtrack::enabled));}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-mutable-images").c_str(),"rb");no_mutable_images=true;if(w){no_mutable_images=fgetc(w)!='1';fclose(w);}xbg_logf("NVN mutable-format images=%d (marker gfx-mutable-images)",int(!no_mutable_images));}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-mutable-test").c_str(),"rb");mutable_test=0;if(w){if(fscanf(w,"%u",&mutable_test)!=1)mutable_test=0;fclose(w);}if(mutable_test)xbg_logf("NVN mutable upgrade test every %u images (marker gfx-mutable-test)",mutable_test);}
     /* Android: no guest memory remapping probe. */
     {FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-pipe-stats").c_str(),"rb");if(m){fclose(m);pstat_props=reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(vkGetDeviceProcAddr(cfg.device,"vkGetPipelineExecutablePropertiesKHR"));
       pstat_stats=reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(vkGetDeviceProcAddr(cfg.device,"vkGetPipelineExecutableStatisticsKHR"));pipe_stats=pstat_props && pstat_stats;}xbg_logf("NVN pipeline statistics=%d (marker gfx-pipe-stats)",int(pipe_stats));}
     {FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-wb-skip").c_str(),"rb");wb_skip=false;if(m){wb_skip=fgetc(m)!='0';fclose(m);}else if(const char *e=getenv("DM_GFX_WB_SKIP"))wb_skip=atoi(e)!=0;
      xbg_logf("NVN superseded storage writebacks skipped=%d pool_flags=%d (marker gfx-wb-skip)",int(wb_skip),int(nvn_pool_flags!=nullptr));}
     {FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-pipe-memo").c_str(),"rb");pipe_memo=false;pipe_memo_valid=false;if(m){pipe_memo=fgetc(m)!='0';fclose(m);}else if(const char *e=getenv("DM_GFX_PIPE_MEMO"))pipe_memo=atoi(e)!=0;
      xbg_logf("NVN pipeline memo for repeated draw state=%d (marker gfx-pipe-memo)",int(pipe_memo));}
     {FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-read-preflight").c_str(),"rb");read_preflight=true;if(m){read_preflight=fgetc(m)!='0';fclose(m);}else if(const char *e=getenv("DM_GFX_READ_PREFLIGHT"))read_preflight=atoi(e)!=0;
      xbg_logf("NVN CPU reads resolved before each operation=%d (marker gfx-read-preflight)",int(read_preflight));}
     {FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-sync-cache-save").c_str(),"rb");sync_cache_save=false;if(m){sync_cache_save=fgetc(m)=='1';fclose(m);}}
     {FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-xview-guard").c_str(),"rb");xview_guard=false;if(m){xview_guard=fgetc(m)!='0';fclose(m);}else if(const char *e=getenv("DM_GFX_XVIEW_GUARD"))xview_guard=atoi(e)!=0;
      xbg_logf("NVN readback buffers the other slot reads are written by the GPU=%d (marker gfx-xview-guard)",int(xview_guard));}
     {FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-upload-prologue").c_str(),"rb");upload_prologue=false;if(m){upload_prologue=fgetc(m)!='0';fclose(m);}else if(const char *e=getenv("DM_GFX_UPLOAD_PROLOGUE"))upload_prologue=atoi(e)!=0;
      xbg_logf("NVN whole uploads of images not yet used in a submission go ahead of it=%d (marker gfx-upload-prologue)",int(upload_prologue));}
     {FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-async-pipelines").c_str(),"rb");warm::async_on=false;if(m){warm::async_on=fgetc(m)!='0';fclose(m);}else if(const char *e=getenv("DM_GFX_ASYNC_PIPELINES"))warm::async_on=atoi(e)!=0;
      xbg_logf("NVN background builds for skippable draws=%d (marker gfx-async-pipelines)",int(warm::async_on));}
     {uint32_t n=0;vkEnumerateDeviceExtensionProperties(cfg.physical,nullptr,&n,nullptr);std::vector<VkExtensionProperties> ext(n);vkEnumerateDeviceExtensionProperties(cfg.physical,nullptr,&n,ext.data());
      std::string list;for(auto &e:ext){list+=e.extensionName;list+=' ';}
      format_feature_flags2=properties.apiVersion>=VK_API_VERSION_1_3;for(auto &e:ext)if(!strcmp(e.extensionName,VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME))format_feature_flags2=true;
      xbg_logf("NVN format-feature-flags2=%d (formatless storage checks %s)",int(format_feature_flags2),format_feature_flags2?"per format":"by the pre-1.3 format list");
      capability_report();
      {const VkFormatProperties &d16=format_properties(VK_FORMAT_D16_UNORM),&d24=format_properties(VK_FORMAT_X8_D24_UNORM_PACK32);
       const VkFormatFeatureFlags d24_needs=VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT|VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_FORMAT_FEATURE_TRANSFER_SRC_BIT|VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
       bool d16_linear=(d16.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)!=0;depth16_as_d24=!d16_linear && (d24.optimalTilingFeatures&d24_needs)==d24_needs;
       xbg_logf("NVN depth16 stored as %s (D16 linear filtering=%u, X8_D24 features=%#x)",depth16_as_d24?"X8_D24":"D16",unsigned(d16_linear),d24.optimalTilingFeatures);
       xbg_logf("NVN compatibility format_feature_flags2=%d depth16_as_d24=%d",int(format_feature_flags2),int(depth16_as_d24));}
      for(size_t at=0;at<list.size();at+=900)xbg_logf("NVN caps extensions[%zu] %s",at/900,list.substr(at,900).c_str());
      xbg_logf("NVN caps api=%u.%u.%u driver=%u maxUBdyn=%u maxPerStageUB=%u minUBAlign=%llu maxBoundSets=%u",VK_API_VERSION_MAJOR(properties.apiVersion),VK_API_VERSION_MINOR(properties.apiVersion),VK_API_VERSION_PATCH(properties.apiVersion),properties.driverVersion,properties.limits.maxDescriptorSetUniformBuffersDynamic,properties.limits.maxPerStageDescriptorUniformBuffers,(unsigned long long)properties.limits.minUniformBufferOffsetAlignment,properties.limits.maxBoundDescriptorSets);}
     {FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-ab-toggle").c_str(),"rb");ab_period=0;if(m){if(fscanf(m,"%d",&ab_period)!=1)ab_period=0;fclose(m);}{FILE *t=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-ab-target").c_str(),"rb");memset(ab_target,0,sizeof ab_target);if(t){if(fscanf(t,"%15s",ab_target)!=1)ab_target[0]=0;fclose(t);}}xbg_logf("NVN PERF ab_period=%d",ab_period);}
     ++slot_serial;gfx_enqueued=0;gfx_executed=0;gfx_presents_queued=0;gfx_async=false; /* Android game command memory remains guest-owned. */if(gfx_async)gfx_worker=std::thread(gfx_worker_loop);xbg_logf("NVN async translation=%d",int(gfx_async));}
    return 0;
}
// nvnEventGetValue on a deferred event: finish the GPU work the signal covers, then the value is published (apply_ready_events).
extern "C" int xbg_evguard_fault(void *addr,int is_write,void *uc){return evguard::enabled?evguard::fault(addr,is_write,uc):0;}
extern "C" void xbg_game_event_read(const void *addr){
    gfx_drain();
    std::lock_guard<std::recursive_mutex> lock(mutex);if(!initialized || failed)return;
    for(auto *list:{&pending_events,&recording_events})for(auto &e:*list)if(e.addr==uint64_t(reinterpret_cast<uintptr_t>(addr))){++event_reads;complete_all();return;}
}
extern "C" int xbg_game_set_sampler_mirror_clamp_to_edge(VkBool32 enabled){
    if(enabled!=VK_FALSE && enabled!=VK_TRUE)return -1;
    std::lock_guard<std::recursive_mutex> lock(mutex);if(!initialized || failed || stats.submissions || stats.draws || stats.dispatches)return -1;
    if(enabled){
        uint32_t count=0;VkResult result=vkEnumerateDeviceExtensionProperties(cfg.physical,nullptr,&count,nullptr);if(result!=VK_SUCCESS)return result;
        std::vector<VkExtensionProperties> extensions(count);result=vkEnumerateDeviceExtensionProperties(cfg.physical,nullptr,&count,extensions.data());if(result!=VK_SUCCESS)return result;
        bool supported=false;for(const auto &e:extensions)if(!strcmp(e.extensionName,VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME))supported=true;
        if(!supported && properties.apiVersion>=VK_API_VERSION_1_2){VkPhysicalDeviceVulkan12Features core{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f.pNext=&core;vkGetPhysicalDeviceFeatures2(cfg.physical,&f);supported=core.samplerMirrorClampToEdge;}
        if(!supported)return -2;
    }
    sampler_mirror_clamp_enabled=enabled;xbg_logf("NVN sampler mirror-clamp-to-edge actualEnabled=%u",enabled);return 0;
}

extern "C" int xbg_game_provenance(const char *path){
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!path){provenance_close("CLOSE");return 0;}
    if(!initialized || provenance_file || capture.trace || timing_active || timing_first || failed || stats.presents.load()>UINT64_MAX-2)return -1;
    FILE *f=fopen(path,"wx");if(!f)return -2;
    provenance.arm(stats.presents.load());provenance_file=f;
    char begin[128];int n=snprintf(begin,sizeof(begin),"BEGIN target=%llu maxDraws=64 maxEvents=2048 maxText=262144\n",(unsigned long long)provenance.target);
    if(n<0 || size_t(n)>=sizeof(begin) || fwrite(begin,1,size_t(n),f)!=size_t(n) || fflush(f)){provenance_close("OPEN_IO_FAIL");return -2;}
    provenance.bytes=uint64_t(n);return 0;
}

extern "C" int xbg_game_capture_frame(const char *directory,uint64_t frame){
    if(!directory || !directory[0] || !frame || strlen(directory)>800)return -1;
    std::lock_guard<std::recursive_mutex> lock(mutex);if(!initialized || failed || stats.submissions || stats.draws || stats.dispatches || capture.trace || timing_first)return -1;
    std::string path=std::string(directory)+"/capture.txt";FILE *trace=fopen(path.c_str(),"wb");if(!trace)return -2;
    capture.directory=directory;capture.trace=trace;capture.frame=frame;capture.bytes=0;capture.events=0;capture.active=false;
    capture_log("CAPTURE version=1 frame=%llu maxEvents=16 maxBytes=%llu semantics=post-fence-raw-no-state-replacement",(unsigned long long)frame,(unsigned long long)capture_limit);
    xbg_logf("NVN one-frame capture frame=%llu directory=%s maxEvents=16 maxBytes=%llu",(unsigned long long)frame,directory,(unsigned long long)capture_limit);return 0;
}
extern "C" int xbg_game_profile_cpu_live(uint32_t frame_count){gfx_drain();
    if(!frame_count || frame_count>60)return -1;
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const uint64_t current=stats.presents.load();
    if(!initialized || failed || live_profile_used || timing_active || timing_first || capture.trace || provenance_file || current>UINT64_MAX-uint64_t(frame_count)-1)return -1;
    // Requested from inside the present callback (a timing scope is open): arm at the next submit instead.
    if(timing.top){deferred_live_profile=frame_count;live_profile_used=true;return 0;}
    timing=XbgCpuTiming(timing_clock);timing_first=current+2;timing_last=timing_first+frame_count-1;live_profile_used=true;
    xbg_logf("NVN CPU_TIMING armed-live first=%llu count=%u maxSeconds=60 clock=steady elapsed-not-thread-cpu once=1",(unsigned long long)timing_first,frame_count);
    return 0;
}
extern "C" int xbg_game_profile_descriptor_live(uint32_t frame_count){gfx_drain();
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // Reuse exact once/pending/active/nonempty-scope/provenance/capture checks.
    int result=xbg_game_profile_cpu_live(frame_count);if(result)return result;
    descriptor_timing=XbgDescriptorTiming{};descriptor_detail=true;
    xbg_logf("NVN DESC_TIMING armed v=2 first=%llu count=%u once=1 maxSeconds=60",(unsigned long long)timing_first,frame_count);
    return 0;
}
extern "C" int xbg_game_profile_draw_live(uint32_t frame_count){gfx_drain();
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // Reuse the original live once/pending/active/provenance/capture gates.
    int result=xbg_game_profile_cpu_live(frame_count);if(result)return result;
    draw_timing=XbgDrawTiming{};draw_detail=true;
    xbg_logf("NVN DRAW_TIMING armed v=1 first=%llu count=%u once=1 maxSeconds=60",(unsigned long long)timing_first,frame_count);
    return 0;
}
extern "C" int xbg_game_profile_cpu(uint64_t first_frame,uint32_t frame_count){
    if(!first_frame || !frame_count || frame_count>120 || first_frame>UINT64_MAX-frame_count)return -1;
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!initialized || failed || stats.submissions || stats.draws || stats.dispatches || stats.presents || timing_first || capture.trace)return -1;
    timing=XbgCpuTiming(timing_clock);timing_first=first_frame;timing_last=first_frame+frame_count-1;
    xbg_logf("NVN CPU_TIMING armed first=%llu count=%u maxSeconds=60 clock=steady elapsed-not-thread-cpu",(unsigned long long)first_frame,frame_count);
    if(first_frame==1)timing_start();return 0;
}
extern "C" int xbg_game_shutdown(){gfx_drain();gfx_async_stop();std::lock_guard<std::recursive_mutex> lock(mutex);if(!initialized)return 0;timing_report("shutdown-partial");provenance_close("SHUTDOWN");resolve_pending();submit_slots(true);save_pipeline_cache();pending_writebacks.clear();pending_reports.clear();pending_events.clear();recording_events.clear();epochs.clear();VkResult r;{deathguard::VkSite site(deathguard::QUEUE_WAIT_IDLE);r=vkQueueWaitIdle(cfg.queue);}if(r!=VK_SUCCESS)return r;wtrack::release();destroy_slots();clear_pipeline_objects();cache_save_join();warm::shutdown();bright::shutdown();clear_descriptor_caches();clear_render_pass_cache();images.clear();++images_epoch;for(auto &f:persistent_framebuffers)vkDestroyFramebuffer(cfg.device,f.second,nullptr);persistent_framebuffers.clear();image_index.clear();image_index_dirty=true;program_cache.clear();shaders.clear();textures.clear();texture_pool_cache=nullptr;samplers.clear();null_textures.clear();texture_pool_cache=nullptr;pools.clear();host_segments.clear();host_ranges.clear();host_bytes=0;cmd_range_lo=0;cmd_range_bytes=0;vkDestroyPipelineCache(cfg.device,pipeline_cache,nullptr);pipeline_cache=VK_NULL_HANDLE;vkDestroyCommandPool(cfg.device,command_pool,nullptr);command_pool=VK_NULL_HANDLE;initialized=false;return 0;}
extern "C" void xbg_game_get_stats(xbg_game_stats *out){if(out)*out={stats.submissions.load(),stats.draws.load(),stats.dispatches.load(),stats.presents.load(),stats.shader_loads.load()};}
extern "C" void xbg_game_frame_counters(xbg_frame_counters *out){if(out)*out={hud::pipelines.load(),hud::pipeline_ns.load(),hud::shaders.load(),hud::shader_ns.load(),hud::uploads.load(),hud::reuploads.load(),hud::upload_bytes.load(),hud::upload_ns.load()};}
extern "C" void xbg_game_set_frame(uint64_t frame){hud::frame.store(frame,std::memory_order_relaxed);}
extern "C" double xbg_game_gpu_ms(uint64_t frame){const hud::Gpu &g=hud::gpu[frame%64];if(g.frame.load(std::memory_order_acquire)!=frame)return -1;return double(g.ns.load(std::memory_order_acquire))/1e6;}
static void xbg_legacy_pool_add_now(void *p,size_t n){std::lock_guard<std::recursive_mutex> lock(mutex);need(p&&n,"Invalid NVN pool");pools[uintptr_t(p)]=n;cmd_range_lo=0;cmd_range_bytes=0;auto epoch=provenance.pool(uint64_t(p),true);provenance_log("POOL_ADD mem=%p bytes=%zu epoch=%llu",p,n,epoch);}
static void xbg_legacy_pool_del_now(void *p){std::lock_guard<std::recursive_mutex> lock(mutex);provenance_log("POOL_DEL mem=%p epoch=%llu",p,provenance.epoch(uint64_t(p)));provenance.pool(uint64_t(p),false);{auto pit=pools.find(uintptr_t(p));if(pit!=pools.end())vcache::drop_pool(uintptr_t(p),pit->second);}pools.erase(uintptr_t(p));cmd_range_lo=0;cmd_range_bytes=0;}
static void xbg_register_texture_now(const dpool_t *p,int id,const tex_t *t,const view_t *v){std::lock_guard<std::recursive_mutex> lock(mutex);need(p&&id>=0&&id<p->n&&t,"Invalid texture registration");auto &slot=textures[pool_key(p)][id];if(register_texture_record(slot,*t,v))++memo_bump[0];}
static void xbg_register_sampler_now(const dpool_t *p,int id,const samp_par *s){std::lock_guard<std::recursive_mutex> lock(mutex);need(p&&id>=0&&id<p->n&&s,"Invalid sampler registration");auto &slot=samplers[pool_key(p)][id];if(memcmp(&slot,s,sizeof(*s))){++memo_epoch;++memo_bump[1];}slot=*s;}
static void xbg_legacy_tex_release_now(tex_t *t){std::lock_guard<std::recursive_mutex> lock(mutex);if(t){auto i=images.find(t->serial);if(i!=images.end()){provenance_image("RELEASE",t,*i->second);retire_object(i->second);++memo_bump[2];}images.erase(t->serial);image_index_dirty=true;++images_epoch;++memo_check_gen;}}
static void xbg_legacy_prog_release_now(prog_t *p){/* Immutable shader modules are keyed by original control digest, shared safely. */std::lock_guard<std::recursive_mutex> lock(mutex);if(p && p->serial)program_cache.erase(p->serial);}
static void xbg_texture_invalidate_now(const tex_t *t){std::lock_guard<std::recursive_mutex> lock(mutex);if(t){auto i=images.find(t->serial);if(i!=images.end()){need(!i->second->gpu_written,"CPU invalidation of GPU-written texture needs explicit range merge");i->second->uploaded=false;}}}
extern "C" void *xbg_host_command_memory(cmd_t *owner,size_t n){gfx_drain();std::lock_guard<std::recursive_mutex> lock(mutex);need(n<=64*1024*1024 && host_bytes+n<=256ull*1024*1024,"Host command memory cap reached; finalize handles before reusing");auto p=std::make_unique<uint8_t[]>(n);void *out=p.get();host_ranges[uintptr_t(out)]=n;host_segments.push_back({owner,std::move(p),n});host_bytes+=n;return out;}
extern "C" void xbg_release_command_handle(uint64_t handle){
    // A handle can CALL another handle and game-supplied segments can point to
    // host segments. Retain until shutdown until reference ownership is proven.
    (void)handle;
}
extern "C" void xbg_copy_commands(cmd_t *command,int count,const uint64_t *handles){gfx_drain();
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto available=[](void *,const void *p)->size_t{return command_available(static_cast<const uint8_t*>(p));};
    size_t bytes=0;int result=xbg_command_clone(count,handles,nullptr,256ull*1024*1024,available,nullptr,&bytes);
    if(result)xbg_fatalf("CopyCommands invalid source chain: %d",result);
    std::vector<uint8_t> snapshot(bytes);size_t actual=0;
    result=xbg_command_clone(count,handles,snapshot.data(),snapshot.size(),available,nullptr,&actual);
    need(result==0 && actual==bytes,"CopyCommands source changed while recording");
    for(size_t offset=0;offset<bytes;){op_hdr h;memcpy(&h,snapshot.data()+offset,8);xbg_record_command(command,snapshot.data()+offset,h.size);offset+=h.size;}
}
extern "C" void xbg_invoke_command_callback(cmd_t *c,size_t n){uint64_t args[]={uint64_t(c),NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY,n,uint64_t(c->cb_data)};if(cfg.invoke_guest)cfg.invoke_guest(cfg.userdata,reinterpret_cast<void*>(c->cb),args,4);else c->cb(reinterpret_cast<NVNcommandBuffer*>(c),NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY,n,c->cb_data);}
// 阶段 1 / 1b：帧摘要只读探针。默认关、不进 envs。
//   DM_FRAME_DIGEST=1 整帧摘要 / =2 再加 opcode 序列首个差异 / =3 **pass 粒度摘要**（阶段 1b）
// 依据 30-设计/帧级重复判定-阶段0-摘要口径与零变化清单-2026-11-05.md 第 3/4 节：
// 只做 FNV-1a 增量 hash + 序列对照，**不跳过任何 op**、不改 state、不物化整帧命令流。
// ⚠ op 流不含资源数据（xbg_command_clone.c:4-6）⇒ 同时输出 wtrack 脏页；
// wtrack 未注册 region 时 dirty_clean 恒 1（假阳性，np715 实测 0 region）⇒ 别拿它当判据。
// ⚠ np715 轮 A 定谳（今日 §1i）：整帧粒度不可用（same=1 0/1524），
//   但 114/117 个 pass 逐字稳定、占 GPU 时间 98.7% ⇒ 静态性在 **pass 粒度**上成立。
//   ⇒ =3 按「OP_SET_RT 的目标指针」给 pass 身份键（帧间 pass 数 100–416 波动，序号不可用），
//     同键多 pass 用出现序号区分；输出静态 pass 占比与「可省 S 类 op / draws」份额。
namespace framedigest{
static int mode=-1;
static uint64_t seq=0,prev_hash=0,prev_dirty=0;
static std::vector<uint32_t> prev_ops,cur_ops;
static uint64_t mix(uint64_t h,const uint8_t *p,size_t n){
    for(size_t i=0;i<n;i++){h^=p[i];h*=1099511628211ull;}
    return h;
}
// ---- 阶段 1b：pass 粒度 ----
struct pass_digest{uint64_t key,hash;size_t bytes;uint32_t ops,draws,sops;};
static const int HISTN=4;                        // 帧距历史：1..4 帧（缓冲环轮换时命中会出现在 d>1）
static std::vector<pass_digest> hist[HISTN];
static int hist_fill=0;
static std::vector<pass_digest> cur_passes;
static pass_digest seg;
static bool s_op(uint32_t op){   // 设计文档 §2 的 S 类（可跳过：纯绘制/清除）
    return op==OP_DRAW||op==OP_DRAW_IDX||op==OP_DISPATCH||op==OP_CLEAR_COLOR||op==OP_CLEAR_DS
        ||op==(OP_COUNT_+2)||op==(OP_COUNT_+3)||op==(OP_COUNT_+7)||op==(OP_COUNT_+8)||op==(OP_COUNT_+9);
}
static bool draw_op(uint32_t op){
    return op==OP_DRAW||op==OP_DRAW_IDX||op==(OP_COUNT_+3)||op==(OP_COUNT_+7)||op==(OP_COUNT_+8);
}
// 身份键 = OP_SET_RT 载荷里各渲染目标的 **tex_t.serial**（不是指针、不是原始载荷字节）。
// 为什么不用原始载荷：`view_t cview[8]` / `has_cview[8]` 里未使用的槽是游戏没写的垃圾，
// 整段 hash 进去会让「同一个 pass」每帧换键（np715b 实测 new 占 90%+，static 只剩 2/162）。
// serial 是贴图对象的初始化序号（nvn_obj.h:35），跨帧稳定；载荷里存的是本序号的不可变副本。
// 不取序号：帧间 pass 数在 100–416 波动（np715 实测）⇒ 序号会整体错位。
static uint64_t seg_base_key(const uint8_t *p,size_t n){
    uint64_t h=1469598103934665603ull;int32_t cnt=0;
    if(n<4)return h;
    memcpy(&cnt,p,4);
    if(cnt<0)cnt=0;if(cnt>8)cnt=8;
    h=mix(h,reinterpret_cast<const uint8_t*>(&cnt),4);
    for(int i=0;i<cnt;i++){                     // color[i] 是本序号的不可变副本 ⇒ 可安全读 serial
        const tex_t *t=nullptr;
        size_t off=offsetof(p_set_rt,color)+size_t(i)*sizeof(const tex_t*);
        if(off+sizeof(const tex_t*)>n)break;
        memcpy(&t,p+off,sizeof(const tex_t*));
        uint64_t ser=t?t->serial:0;
        h=mix(h,reinterpret_cast<const uint8_t*>(&ser),8);
    }
    if(offsetof(p_set_rt,depth)+sizeof(const tex_t*)<=n){
        const tex_t *d=nullptr;
        memcpy(&d,p+offsetof(p_set_rt,depth),sizeof(const tex_t*));
        uint64_t ser=d?d->serial:0;
        h=mix(h,reinterpret_cast<const uint8_t*>(&ser),8);
    }
    return h;
}
static void seg_open(const uint8_t *p,size_t n){
    uint64_t k=seg_base_key(p,n);unsigned occ=0;
    for(size_t i=0;i<cur_passes.size();i++)if(cur_passes[i].key==k)++occ;   // 同键第几次出现
    seg=pass_digest{k^(uint64_t(occ)*0x9E3779B97F4A7C15ull),1469598103934665603ull,0,0,0,0};
    seg.hash=mix(seg.hash,reinterpret_cast<const uint8_t*>(&k),8);          // 同内容不同 RT 也要分开
}
static void seg_close(){if(seg.ops)cur_passes.push_back(seg);seg.ops=0;}
static int match_dist(const pass_digest &c){        // 返回 1..HISTN 表示在第 d 帧前的历史里命中，0 = 未命中
    for(int d=0;d<hist_fill&&d<HISTN;d++)
        for(size_t j=0;j<hist[d].size();j++)
            if(hist[d][j].key==c.key&&hist[d][j].hash==c.hash)return d+1;
    return 0;
}
static void pass_report(uint64_t frame){
    size_t st=0,chg=0,nw=0;uint64_t dt=0,ds=0,st_sops=0,sops=0,bs=0,bt=0;
    long long first=-1;uint64_t first_draws=0;
    const std::vector<pass_digest> &prev=hist_fill?hist[0]:cur_passes;   // 上一帧（首帧退化，无判据）
    for(size_t i=0;i<cur_passes.size();i++){
        const pass_digest &c=cur_passes[i];
        dt+=c.draws;sops+=c.sops;bt+=c.bytes;
        const pass_digest *pv=nullptr;
        for(size_t j=0;j<prev.size();j++)if(prev[j].key==c.key){pv=&prev[j];break;}
        if(pv&&pv->hash==c.hash){++st;ds+=c.draws;st_sops+=c.sops;bs+=c.bytes;}
        else{if(pv)++chg;else ++nw;if(first<0){first=(long long)i;first_draws=c.draws;}}
    }
    // 帧距 1..4：重放层若按 N 帧轮换缓冲地址，命中会出现在 d=N 而不是 d=1
    size_t dhit[HISTN+1]={0};uint64_t ddraw[HISTN+1]={0};
    size_t bigi=0;
    for(size_t i=0;i<cur_passes.size();i++){
        if(cur_passes[i].draws>cur_passes[bigi].draws)bigi=i;
        int d=match_dist(cur_passes[i]);
        if(d>0&&d<=HISTN){++dhit[d];ddraw[d]+=cur_passes[i].draws;}
    }
    int bigd=cur_passes.empty()?0:match_dist(cur_passes[bigi]);
    // 备用口径：不靠身份键，按摘要值对上一帧做多重集交集
    std::vector<uint64_t> pool;pool.reserve(prev.size());
    for(size_t i=0;i<prev.size();i++)pool.push_back(prev[i].hash);
    size_t sth=0;uint64_t dsh=0,ssh=0,bsh=0;
    for(size_t i=0;i<cur_passes.size();i++)
        for(size_t j=0;j<pool.size();j++)if(pool[j]==cur_passes[i].hash){
            pool[j]=pool.back();pool.pop_back();
            ++sth;dsh+=cur_passes[i].draws;ssh+=cur_passes[i].sops;bsh+=cur_passes[i].bytes;break;}
    // 一行一提交：静态份额 + 可省 S 类 op + 帧距命中 + 最大 pass 的命中距离
    xbg_logf("FRAMEDIGEST2 frame=%llu passes=%zu static=%zu chg=%zu new=%zu draws=%llu draws_static=%llu sops=%llu sops_static=%llu bytes=%llu bytes_static=%llu first=%lld first_draws=%llu | d1=%zu d2=%zu d3=%zu d4=%zu ddraws=%llu/%llu/%llu/%llu big=%u bigd=%d hist=%d | hstatic=%zu hdraws=%llu hsops=%llu",
        (unsigned long long)frame,(size_t)cur_passes.size(),st,chg,nw,
        (unsigned long long)dt,(unsigned long long)ds,(unsigned long long)sops,(unsigned long long)st_sops,
        (unsigned long long)bt,(unsigned long long)bs,first,(unsigned long long)first_draws,
        dhit[1],dhit[2],dhit[3],dhit[4],
        (unsigned long long)ddraw[1],(unsigned long long)ddraw[2],(unsigned long long)ddraw[3],(unsigned long long)ddraw[4],
        cur_passes.empty()?0u:cur_passes[bigi].draws,bigd,hist_fill,
        sth,(unsigned long long)dsh,(unsigned long long)ssh);
    for(int d=HISTN-1;d>0;--d)hist[d].swap(hist[d-1]);
    hist[0]=cur_passes;cur_passes.clear();
    if(hist_fill<HISTN)++hist_fill;
}
// =4：**地址无关**摘要（阶段 1b 第二步）。np715f 定谳：op 逐字节摘要被缓冲地址轮换污染
//   （最大 pass 28,674 draws 只在 d=2/d=4 命中 ⇒ 2–4 帧轮换，不是内容变化）。
//   ⇒ 只 hash **结构**（op + size，丢掉操作数里的地址）+ 本 pass 里绑定过的 **UBO 内容**。
//   ⚠ 仍看不见顶点内容变化（2.9M 顶点/帧，逐字节划算不过来）⇒ 阶段 2 必须叠加
//     GPASS 的 draws/verts 签名，别只靠这一条。
static uint64_t ubo_digest(uint64_t addr,uint64_t size){
    if(addr<4096||!size)return 0;
    size_t take=size_t(size>65536?65536:size);         // 单次上限 64 KB（相机/材质 UBO 都远小于此）
    return mix(1469598103934665603ull,reinterpret_cast<const uint8_t*>(uintptr_t(addr)),take);
}
static void walk(const uint8_t *pc,unsigned depth,uint64_t &h,uint64_t &bytes,uint64_t &ops,uint64_t &steps){
    if(depth>64)return;
    for(;;){
        if(++steps>10000000ull)return;                 // 坏 op 流的死循环防护
        size_t available=command_available(pc);
        if(available<sizeof(op_hdr))return;
        op_hdr hd;memcpy(&hd,pc,sizeof(hd));
        if(hd.size<8||(hd.size&7)||hd.size>available)return;
        if(hd.op==OP_END)return;
        if(hd.op==OP_CALL||hd.op==OP_JUMP){
            if(hd.size<16)return;
            uint64_t next;memcpy(&next,pc+8,8);
            if(hd.op==OP_JUMP){pc=reinterpret_cast<const uint8_t*>(next);continue;}
            walk(reinterpret_cast<const uint8_t*>(next),depth+1,h,bytes,ops,steps);
        }else if(hd.op!=OP_BEGIN){
            if(mode>=3){                               // pass 分段：SET_RT 开新段，本 op 属于新段
                if(hd.op==OP_SET_RT){
                    // ⚠ 不把 SET_RT 记录本身 hash 进去：载荷里的 cview/has_cview 未用槽是垃圾，
                    //   一 hash 进去同一条个 pass 每帧换摘要（np715b 学费）。身份已由 key（serial）承担。
                    seg_close();seg_open(pc+8,hd.size-8);++seg.ops;
                }else if(mode>=4){
                    // 地址无关：只进 (op,size) 8 字节头；UBO/SSBO 额外把 **内容** 折进去
                    seg.hash=mix(seg.hash,pc,8);seg.bytes+=hd.size;++seg.ops;
                    if(s_op(hd.op))++seg.sops;
                    if(draw_op(hd.op))++seg.draws;
                    if(hd.op==OP_BIND_UBO||hd.op==OP_BIND_SSBO){
                        p_bind_buf b{};size_t cp=hd.size-8;
                        if(cp>sizeof(b))cp=sizeof(b);
                        memcpy(&b,pc+8,cp);
                        uint64_t u=ubo_digest(b.addr,b.size);
                        seg.hash=mix(seg.hash,reinterpret_cast<const uint8_t*>(&u),8);
                    }
                }else{
                    seg.hash=mix(seg.hash,pc,hd.size);seg.bytes+=hd.size;++seg.ops;
                    if(s_op(hd.op))++seg.sops;
                    if(draw_op(hd.op))++seg.draws;
                }
            }
            h=mix(h,pc,hd.size);bytes+=hd.size;++ops;
            if(mode==2)cur_ops.push_back(hd.op);       // 只有 =2 需要逐 op 序列（1.4M op 别白存）
        }
        pc+=hd.size;
    }
}
static void probe(int n,const uint64_t *handles){
    if(mode<0){const char *e=getenv("DM_FRAME_DIGEST");mode=e?atoi(e):0;}
    if(!mode)return;                                   // 默认关：零开销、行为零变化
    uint64_t h=1469598103934665603ull,bytes=0,ops=0,steps=0;
    if(mode==2)cur_ops.clear();
    if(mode>=3){cur_passes.clear();seg=pass_digest{0,0,0,0,0,0};}
    for(int i=0;i<n;i++)walk(reinterpret_cast<const uint8_t*>(handles[i]),0,h,bytes,ops,steps);
    if(mode>=3)seg_close();
    uint64_t dp=wtrack::dirty_pages;
    uint64_t delta=dp>=prev_dirty?dp-prev_dirty:0;     // PERF 会周期性清零，负值按 0 处理并看 enabled
    int same=seq&&h==prev_hash;
    int dirty_clean=int(wtrack::enabled&&delta==0);
    long long diff=-1;uint32_t opa=0,opb=0;
    if(seq&&mode==2){
        size_t m=std::min(prev_ops.size(),cur_ops.size());
        for(size_t i=0;i<m;i++)if(prev_ops[i]!=cur_ops[i]){diff=(long long)i;opa=prev_ops[i];opb=cur_ops[i];break;}
        if(diff<0&&prev_ops.size()!=cur_ops.size()){
            diff=(long long)m;opa=m<prev_ops.size()?prev_ops[m]:0xffffffffu;opb=m<cur_ops.size()?cur_ops[m]:0xffffffffu;
        }
    }
    xbg_logf("FRAMEDIGEST seq=%llu frame=%llu presents=%llu subs=%llu n=%d ops=%llu bytes=%llu hash=%016llx same=%d wtrack=%d dirty_clean=%d dirty_total=%llu dirty_delta=%llu diff=%lld op_prev=%u op_cur=%u",
        (unsigned long long)seq,(unsigned long long)hud::frame.load(std::memory_order_relaxed),
        (unsigned long long)stats.presents.load(),(unsigned long long)stats.submissions.load(),n,
        (unsigned long long)ops,(unsigned long long)bytes,(unsigned long long)h,same,
        int(wtrack::enabled),dirty_clean,(unsigned long long)dp,(unsigned long long)delta,diff,opa,opb);
    if(mode==2)prev_ops.swap(cur_ops);
    if(mode>=3)pass_report(hud::frame.load(std::memory_order_relaxed));
    prev_hash=h;prev_dirty=dp;++seq;
}
}
static void xbg_submit_now(int n,const uint64_t *handles){std::lock_guard<std::recursive_mutex> lock(mutex);
    if(deferred_live_profile && !timing.top){uint64_t current=stats.presents.load();timing=XbgCpuTiming(timing_clock);timing_first=current+2;timing_last=timing_first+deferred_live_profile-1;deferred_live_profile=0;}
   TimingRoot timing_root;CPU_SCOPE(Other);need(initialized && !failed,"Game Vulkan backend not initialized/healthy");uint64_t begin=timing_clock();snapshot_reset();framedigest::probe(n,handles);uint64_t t_plan=timing_clock();if(frameskip::enabled())frameskip::plan(n,handles);frameskip::plan_ns+=timing_clock()-t_plan;frameskip::exec_begin();uint64_t budget=0;for(int i=0;i<n;i++)execute(reinterpret_cast<const uint8_t*>(handles[i]),0,budget);frameskip::exec_report(hud::frame.load(std::memory_order_relaxed));++stats.submissions;++submitted;if(sync_mode)complete_all();perf.submit_cpu_ns+=timing_clock()-begin;}
extern "C" uint64_t xbg_legacy_last_fence(){return gfx_async?gfx_enqueued.load():[]{std::lock_guard<std::recursive_mutex> lock(mutex);return submitted;}();}
extern "C" int xbg_legacy_wait(uint64_t fence,uint64_t timeout){
    // 不取锁的提前返回只用于新写法；旧写法（非鸿蒙缺省）先判 failed，与原来逐条相同
    // the lock-free early return is new-path only; the old path (default off HarmonyOS) checks failed first, exactly as before
    if(!legacy_waitall && fence<=completed.load())return 1;
    const bool completion_path=completion_wait_enabled && timeout && !legacy_waitall;
    // 新路径的唯一总预算含初锁、重锁与四轮后的回退；旧路径保留原时限语义。
    // One new-path budget includes initial lock, relock and the post-four-round fallback; preserve the old path's timeout semantics.
    const xbg_completion::Deadline deadline(completion_path?timeout:UINT64_MAX);
    // 只等覆盖 fence 的那一批（同 wait_fence / d2r-harmonyos），且等 GPU 时不持引擎锁：游戏每帧都等刚交的那一帧，旧写法先强交、再等全部在途命令，
    // 全程攥着引擎锁，翻译线程的出图每帧被堵约 16 ms。轮询（timeout 0）只 try_lock，翻译线程正持锁就当没完成，不排队。按提交顺序退槽（写回不越序）。
    // 标记 gfx-legacy-waitall=1 / 0 选旧 / 新写法；非鸿蒙缺省旧写法（与原来逐条相同）。
    // Wait only for the batch covering `fence` (as wait_fence / d2r-harmonyos), without holding the engine lock while the GPU works: the game waits every frame for the frame it just
    // submitted; the old code force-submitted, waited for all in-flight work and held the engine lock throughout, blocking translation's present ~16 ms per frame. A poll (timeout 0)
    // only try_locks: if translation holds the lock it reports not done instead of queueing. Slots retire in submission order (writebacks stay ordered). Marker gfx-legacy-waitall=1 / 0 picks the old / new code; off HarmonyOS the old code (unchanged) is the default.
    std::unique_lock<std::recursive_mutex> lock(mutex,std::defer_lock);
    // 仅对可能阻塞的调用记 trace；保持单次 try_lock 轮询与原退休顺序。
    // Trace potentially blocking calls only; retain single-try-lock polls and the original retirement order.
    if(timeout || legacy_waitall){XTRACE_COUNT("dm-legacy-target", fence);XTRACE_COUNT("dm-legacy-completed-before-lock", completed.load());}
    if(!timeout && !legacy_waitall){if(!lock.try_lock())return 0;}else{
        XTRACE_BEGIN("dm-legacy-lock");
        const auto acquired=completion_path?completion_wait.acquire(lock,fence,deadline):(lock.lock(),xbg_completion::Acquire::EngineLocked);
        XTRACE_END();XTRACE_COUNT("dm-legacy-lock-result", int(acquired));
        if(acquired==xbg_completion::Acquire::Completed){XTRACE_COUNT("dm-legacy-completed", completed.load());return 1;}
        if(acquired==xbg_completion::Acquire::TimedOut)return 0;
    }
    if(timeout || legacy_waitall){XTRACE_COUNT("dm-legacy-submitted", submitted);XTRACE_COUNT("dm-legacy-completed", completed.load());}
    if(failed || fence>submitted)return 0;if(fence<=completed)return 1;
    if(completion_path && deadline.expired())return 0;
    if(!legacy_waitall){
        auto begin=std::chrono::steady_clock::now();
        auto retire_done=[&]{for(int i=0;i<2;i++){if(completion_path && deadline.expired())break;Slot &o=slots[slot_index].pending?slots[slot_index]:slots[slot_index^1];if(!o.pending || vkGetFenceStatus(cfg.device,o.fence)!=VK_SUCCESS)break;SiteScope site(SITE_LEGACY_WAIT);retire_slot(o);}};
        for(int round=0;round<4;round++){
            if(completion_path && deadline.expired())return fence<=completed;
            retire_done();if(failed)return 0;if(fence<=completed)return 1;
            if(completion_path && deadline.expired())return 0;
            Slot &a=slots[slot_index],&b=slots[slot_index^1];Slot &older=a.pending?a:b,&newer=a.pending?b:a;Slot &cur=slots[slot_index];
            // 没有在途槽、录制槽也是空的：已翻译的提交全都做完了（游戏常等一次不含 GPU 命令的提交）| Nothing in flight and an empty recording slot: every translated submit is done (the game often waits on a submit with no GPU commands)
            if(!older.pending && !newer.pending && !(cur.recording && cur.ops)){resolve_known_reports();if(pending_writebacks.empty()&&pending_reports.empty()){publish_completed(std::max(completed.load(),submitted));}if(fence<=completed)return 1;break;}
            if(!timeout)return 0;
            Slot *t=older.pending && older.covers>=fence?&older:newer.pending && newer.covers>=fence?&newer:nullptr;
            if(!t){if(cur.recording && cur.ops){SiteScope site(SITE_LEGACY_WAIT);submit_slots(false);continue;}   // 目标还在录制中的槽：交掉再等 | target still in the recording slot: submit it, then wait
                t=newer.pending?&newer:&older;}   // 目标提交没有 GPU 命令：等最新的在途批次 | the target submit has no GPU commands: wait for the newest in-flight batch
            uint64_t remaining=completion_path?deadline.remaining_ns():timeout;if(!completion_path && timeout!=UINT64_MAX){auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();remaining=uint64_t(elapsed)>=timeout?0:timeout-uint64_t(elapsed);}
            XTRACE_COUNT("dm-legacy-slot", t-slots);XTRACE_COUNT("dm-legacy-covers", t->covers);
            XTRACE_COUNT("dm-legacy-completed", completed.load());
            VkFence f=t->fence;t->waiters.n.fetch_add(1);lock.unlock();
            XTRACE_BEGIN("dm-legacy-wait");uint64_t b0=timing_clock();VkResult r=vkWaitForFences(cfg.device,1,&f,VK_TRUE,remaining);uint64_t b1=timing_clock();XTRACE_END();
            // 先释放 waiter 再等待完成/重锁；此后槽可被另一线程复用，不再读取 t。
            // Release the waiter before completion/relock waits; another thread may now reuse the slot, so never read t again.
            t->waiters.n.fetch_sub(1);
            if(completion_path){
                completion_wait.record_gpu_wait(b1-b0);
                if(r==VK_TIMEOUT)return 0;
                // 设备错误不能被 completed 快路吞掉；原 fatal 状态仍在 engine 下处理。
                // Completion must not hide device errors; handle the original fatal state while holding engine.
                if(r!=VK_SUCCESS){if(lost_not_fatal(r,"guest fence wait"))return 0;lock.lock();fold_completion_wait_stats();VK(r);}
                XTRACE_BEGIN("dm-legacy-relock");const auto acquired=completion_wait.acquire(lock,fence,deadline);XTRACE_END();
                XTRACE_COUNT("dm-legacy-relock-result", int(acquired));XTRACE_COUNT("dm-legacy-completed", completed.load());
                if(acquired==xbg_completion::Acquire::Completed)return 1;
                if(acquired==xbg_completion::Acquire::TimedOut)return 0;
                fold_completion_wait_stats();
            }else{
                XTRACE_BEGIN("dm-legacy-relock");lock.lock();XTRACE_END();XTRACE_COUNT("dm-legacy-relock-result", 0);
                XTRACE_COUNT("dm-legacy-completed", completed.load());
                block_ns[SITE_LEGACY_WAIT]+=b1-b0;++block_calls[SITE_LEGACY_WAIT];legacy_wait_ns+=b1-b0;
                if(r==VK_TIMEOUT)return 0;if(lost_not_fatal(r,"guest fence wait"))return 0;VK(r);
            }}
        retire_done();if(fence<=completed)return 1;}
    if(completion_path && deadline.expired())return fence<=completed;
    SiteScope site(SITE_LEGACY_WAIT);uint64_t w0=timing_clock();struct Spent{uint64_t w0;~Spent(){legacy_wait_ns+=timing_clock()-w0;}}spent{w0};
    submit_slots(false);auto begin=std::chrono::steady_clock::now();
    for(auto &slot:slots)if(slot.pending){uint64_t remaining=completion_path?deadline.remaining_ns():timeout;
        if(completion_path && !remaining)return fence<=completed;
        if(!completion_path && timeout!=UINT64_MAX){auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();remaining=uint64_t(elapsed)>=timeout?0:timeout-uint64_t(elapsed);}
        bool busy=vkGetFenceStatus(cfg.device,slot.fence)==VK_NOT_READY;uint64_t b0=busy?timing_clock():0;
        VkResult r=vkWaitForFences(cfg.device,1,&slot.fence,VK_TRUE,remaining);if(busy){block_ns[SITE_LEGACY_WAIT]+=timing_clock()-b0;++block_calls[SITE_LEGACY_WAIT];}
        if(r==VK_TIMEOUT)return 0;if(lost_not_fatal(r,"slot fence wait"))return 0;VK(r);retire_slot(slot);
    }
    resolve_known_reports();if(pending_writebacks.empty()&&pending_reports.empty()){publish_completed(submitted);}
    return fence<=completed;
}
extern "C" void xbg_legacy_finish(){gfx_drain();std::lock_guard<std::recursive_mutex> lock(mutex);need(initialized,"Queue finish before Vulkan initialization");++perf.why_finish;complete_all();{deathguard::VkSite site(deathguard::QUEUE_WAIT_IDLE);VK(vkQueueWaitIdle(cfg.queue));}}
// np665：**上屏源图自检**。要回答的是一个二选一：屏幕全黑是"源图本来就黑"（渲染侧）还是"上屏没生效"
//   （呈现侧）。做法：在第 N 次 present 时（DM_PRESENT_PROBE=N）把**正要上屏的那张纹理**读回 CPU，
//   打一行统计 + 一张 32x18 的 ASCII 亮度图。走日志而不落盘——本平台大 I/O 会楔死客体线程（铁律：1.1MB
//   dump 就有前科），而这张图只需要"有没有内容"，不需要像素级保真。
// np665: presented-source self-check. The question is binary: is the black screen "the source texture is
//   black" (render side) or "present does not reach the screen" (present side)? On the N-th present
//   (DM_PRESENT_PROBE=N) read the texture that is about to be presented back to the CPU and log one stats
//   line plus a 32x18 ASCII luminance map. Log rather than a file: big I/O wedges the guest thread on this
//   platform, and the map only has to answer "is there anything in it".
static int present_probe_target(){static int v=-2;if(v==-2){const char *e=getenv("DM_PRESENT_PROBE");v=e?atoi(e):0;if(v>0)xbg_logf("上屏自检 DM_PRESENT_PROBE=%d（第 %d 次 present 读回源图）",v,v);}return v;}
// np666：按**时间**采样（DM_PRESENT_PROBE_EVERY_MS=T，最多 8 次）。为什么不用 present 计数：存活天花板
//   在 1600~2550 presents 之间随机落地，用计数触发经常"还没采样就卡住了"。按时间采样还顺带给出
//   "上线图像的亮度随时间怎么走"这条曲线（加载期 → 进游戏）。
// np666: sample by wall clock (DM_PRESENT_PROBE_EVERY_MS=T, at most 8 shots). Present-count triggering
//   often never fires because the run dies somewhere between 1600 and 2550 presents; a time series also
//   yields "how the presented brightness evolves" (loading -> in game).
static void winpass_probe_check(){
    static int target=-2;
    if(target==-2){const char *e=getenv("DM_WINPASS_PROBE");target=e?atoi(e):0;if(target>0)xbg_logf("[上屏] 帧中自检 DM_WINPASS_PROBE=%d（窗口 pass 关掉后立刻读回）",target);}
    if(target<=0)return;
    static int seen=0,dbg=0;
    if(dbg<6){++dbg;
        if(g_winpass_last)xbg_logf("[上屏] 帧中候选 last serial=%llu %ux%u hint=%llu seen=%d",
            (unsigned long long)g_winpass_last->source.serial,g_winpass_last->width,g_winpass_last->height,
            (unsigned long long)g_present_serial_hint,seen);
        else xbg_logf("[上屏] 帧中候选 last=null hint=%llu seen=%d",(unsigned long long)g_present_serial_hint,seen);}
    if(!g_winpass_last)return;
    Image &im=*g_winpass_last;
    if(!g_present_serial_hint || im.source.serial!=g_present_serial_hint)return;
    if(++seen<target)return;
    uint32_t w=std::min<uint32_t>(64,im.width),h=std::min<uint32_t>(64,im.height);
    if(!w||!h||im.bpp<4)return;
    Buffer b=readback_buffer(size_t(w)*h*4);
    {Commands q;VkBufferImageCopy r{};r.bufferOffset=b.offset;r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};r.imageExtent={w,h,1};
     vkCmdCopyImageToBuffer(q.command,im.image,im.layout,b.buffer,1,&r);q.finish();}
    submit_slots(false);gpu_sync(perf.sync_readback);b.invalidate();
    const uint8_t *pp=static_cast<const uint8_t*>(b.mapped)+b.offset;uint64_t npx=uint64_t(w)*h,nz=0;uint64_t sum[3]={0,0,0};
    for(uint64_t i=0;i<npx;i++){const uint8_t *q=pp+i*4;for(int k=0;k<3;k++)sum[k]+=q[k];if(q[0]|q[1]|q[2])++nz;}
    xbg_logf("[上屏] 帧中：serial=%llu %ux%u 左上%ux%u 均值 %.1f/%.1f/%.1f 非黑 %llu/%llu = %.1f%% 布局=%d",
        (unsigned long long)im.source.serial,im.width,im.height,w,h,double(sum[0])/double(npx),double(sum[1])/double(npx),double(sum[2])/double(npx),
        (unsigned long long)nz,(unsigned long long)npx,100.0*double(nz)/double(npx),int(im.layout));
}
static bool g_winpass_hook_armed=(g_winpass_hook=&winpass_probe_check,true);
static bool present_probe_due(){    static int every=-2;if(every==-2){const char *e=getenv("DM_PRESENT_PROBE_EVERY_MS");every=e?atoi(e):0;if(every>0)xbg_logf("上屏自检 每 %d ms 采样一次（最多 16 次）",every);}
    if(every<=0)return false;
    static uint64_t t0=0,next=0;static int shots=0;
    uint64_t now=timing_clock()/1000000ull;
    if(!t0){t0=now;next=now+uint64_t(every);}
    if(shots>=16||now<next)return false;   // np715：8→16，为了在同一相位内拿到足够多的像素 hash 样本
    next=now+uint64_t(every);++shots;return true;
}
static void present_probe_report(const Buffer &buf,uint32_t w,uint32_t h,uint32_t bpp){
    if(!buf.mapped || !w || !h || bpp<3)return;
    const uint8_t *p=static_cast<const uint8_t*>(buf.mapped)+buf.offset;
    uint64_t sum[3]={0,0,0},nz=0,n=uint64_t(w)*h;unsigned lo[3]={255,255,255},hi[3]={0,0,0};
    // np691: alpha separates "nothing was written" from "black was written". RGB alone cannot: a draw that never
    //   lands and a draw that outputs (0,0,0,1) both read back as RGB 0.
    uint64_t sum_a=0,na=0;
    for(uint64_t i=0;i<n;i++){const uint8_t *q=p+i*bpp;for(int k=0;k<3;k++){unsigned v=q[k];sum[k]+=v;if(v<lo[k])lo[k]=v;if(v>hi[k])hi[k]=v;}if(q[0]|q[1]|q[2])++nz;
        if(bpp>=4){unsigned a=q[3];sum_a+=a;if(a)++na;}}
    // np715：**像素 hash**。判「画面静止还是在动」不能靠截图（含宿主 UI 与 JPEG 压缩噪声），
    //   也不能只看均值/非黑占比（不同图可能同均值）⇒ 对整张读回做 FNV-1a，日志里逐次可比。
    //   与下面的 ASCII 缩略同源同一次读回：hash 相同即逐字节相同。
    uint64_t ph=1469598103934665603ull;
    {size_t nb=size_t(n)*bpp;for(size_t i=0;i<nb;i++){ph^=p[i];ph*=1099511628211ull;}}
    xbg_logf("[上屏] 源图 %ux%u bpp=%u 均值 %.1f/%.1f/%.1f 非黑像素 %llu/%llu = %.1f%% 范围 R%u-%u G%u-%u B%u-%u 均值A=%.1f A非0=%llu(%.1f%%) hash=%016llx",
        w,h,bpp,double(sum[0])/double(n),double(sum[1])/double(n),double(sum[2])/double(n),
        (unsigned long long)nz,(unsigned long long)n,100.0*double(nz)/double(n),
        lo[0],hi[0],lo[1],hi[1],lo[2],hi[2],
        bpp>=4?double(sum_a)/double(n):-1.0,(unsigned long long)na,bpp>=4?100.0*double(na)/double(n):-1.0,
        (unsigned long long)ph);
    static const char ramp[]=" .:-=+*#%@";const int cols=32,rows=18;
    for(int ry=0;ry<rows;ry++){char line[cols+1];
        for(int rx=0;rx<cols;rx++){
            uint32_t x0=uint32_t(uint64_t(rx)*w/cols),x1=uint32_t(uint64_t(rx+1)*w/cols);if(x1<=x0)x1=x0+1;if(x1>w)x1=w;
            uint32_t y0=uint32_t(uint64_t(ry)*h/rows),y1=uint32_t(uint64_t(ry+1)*h/rows);if(y1<=y0)y1=y0+1;if(y1>h)y1=h;
            uint64_t acc=0,cnt=0;
            for(uint32_t y=y0;y<y1;y++)for(uint32_t x=x0;x<x1;x++){const uint8_t *q=p+(uint64_t(y)*w+x)*bpp;acc+=(uint64_t(q[0])*3+q[1]*6+q[2]);++cnt;}
            unsigned v=cnt?unsigned(acc/cnt/10):0u;line[rx]=ramp[v>9?9:v];}
        line[cols]=0;xbg_logf("[上屏] |%s|",line);}
}
// np673：**别的渲染目标里到底有没有内容**。判"整条渲染没产出"还是"只缺最后合成进窗口"：
//   DM_PRESENT_PROBE_SERIALS=222,6224 时，把这几张纹理在同一时刻读回，只打统计（不打 ASCII 图）。
// np673: is anything in the *other* render targets? Distinguishes "the whole render produces nothing" from
//   "only the final composite into the window is missing". DM_PRESENT_PROBE_SERIALS=222,6224 reads those
//   images back at the same instant and logs stats only.
static void present_probe_other_serials(){
    static int inited=0;
    if(!inited){inited=1;const char *e0=getenv("DM_PRESENT_PROBE_SERIALS");if(!e0||!*e0){inited=2;return;}xbg_logf("[上屏] 旁证：同刻读回 serial 列表 = %s",e0);}
    if(inited!=1)return;
    const char *e=getenv("DM_PRESENT_PROBE_SERIALS");if(!e)return;
    std::string list(e);size_t pos=0;int n=0;
    while(pos<=list.size() && n<6){
        size_t comma=list.find(',',pos);if(comma==std::string::npos)comma=list.size();
        std::string tok=list.substr(pos,comma-pos);pos=comma+1;if(tok.empty())continue;++n;
        uint64_t serial=strtoull(tok.c_str(),nullptr,10);
        auto it=images.find(serial);
        if(it==images.end() || !it->second){xbg_logf("[上屏] 旁证 serial=%llu 不存在",(unsigned long long)serial);continue;}
        Image &oi=*it->second;
        if(oi.samples!=VK_SAMPLE_COUNT_1_BIT || oi.aspects!=VK_IMAGE_ASPECT_COLOR_BIT || oi.bpp<4 || !oi.width || !oi.height){
            xbg_logf("[上屏] 旁证 serial=%llu 跳过（samples=%d aspects=%d bpp=%u %ux%u）",(unsigned long long)serial,int(oi.samples),int(oi.aspects),oi.bpp,oi.width,oi.height);continue;}
        Buffer b=readback_buffer(size_t(oi.width)*oi.height*4);
        {Commands q; VkImageLayout saved=oi.layout;oi.transition(q.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
         VkBufferImageCopy r{};r.bufferOffset=b.offset;r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};r.imageExtent={oi.width,oi.height,1};
         vkCmdCopyImageToBuffer(q.command,oi.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,b.buffer,1,&r);
         VkBufferMemoryBarrier h{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};h.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;h.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
         h.srcQueueFamilyIndex=h.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;h.buffer=b.buffer;h.offset=b.offset;h.size=b.size;
         vkCmdPipelineBarrier(q.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&h,0,nullptr);
         oi.transition(q.command,saved);q.finish();}
        gpu_sync(perf.sync_readback);b.invalidate();
        const uint8_t *pp=static_cast<const uint8_t*>(b.mapped)+b.offset;uint64_t npx=uint64_t(oi.width)*oi.height,nz=0;uint64_t sum[3]={0,0,0};
        for(uint64_t i=0;i<npx;i++){const uint8_t *q=pp+i*4;for(int k=0;k<3;k++)sum[k]+=q[k];if(q[0]|q[1]|q[2])++nz;}
        xbg_logf("[上屏] 旁证 serial=%llu %ux%u 均值 %.1f/%.1f/%.1f 非黑 %llu/%llu = %.1f%% 布局=%d",
            (unsigned long long)serial,oi.width,oi.height,double(sum[0])/double(npx),double(sum[1])/double(npx),double(sum[2])/double(npx),
            (unsigned long long)nz,(unsigned long long)npx,100.0*double(nz)/double(npx),int(oi.layout));
    }
}
// np715：**逐帧像素 hash**（阶段 1b 的上限读数）。目的只有一个：量出「连续多少帧上屏像素逐字节相同」
//   = 阶段 2 可跳帧的**上限**。读回带一次 GPU 同步 ⇒ 这一轮只判画面、不看 fps（会明显更慢）。
//   预算控制：`DM_PRESENT_HASH_FROM`（从第几个 present 开始）+ `DM_PRESENT_HASH_FRAMES`（共多少帧），
//   默认 0 = 关；每帧一行、不打 ASCII 缩略（那是 present_probe_report 的活）。
static int present_hash_frames(){static int v=-2;if(v==-2){const char*e=getenv("DM_PRESENT_HASH_FRAMES");v=e?atoi(e):0;
    xbg_logf("[上屏] 逐帧像素 hash 自报：DM_PRESENT_HASH_FRAMES=%s 解析=%d%s",e?e:"(未设置)",v,v>0?"（含读回同步，本轮不看 fps）":"（关）");}return v;}
// ⚠ 预算必须是函数内 static：曾经把「剩余帧数」放在调用点的局部变量里传引用，
//   结果每次调用都被重新初始化成 0 ⇒ 一次也不触发（np715i 白跑一轮的学费）。
static bool present_hash_now(uint64_t draws){
    static int from=-2;
    static int frames=0,done=0;
    if(from==-2){const char*f=getenv("DM_PRESENT_HASH_FROM");from=f?atoi(f):0;frames=present_hash_frames();}
    if(frames<=0)return false;
    static int all=-1;if(all<0){const char*e=getenv("DM_PRESENT_HASH_ALL");all=(e&&*e=='1')?1:0;}
    if(draws<20000 && !all)return false;                // 只取「重场景帧」：PRE 期 40 fps 会把预算瞬间烧完；
                                                        // 跳过轮 draws 会变 0 ⇒ 用 DM_PRESENT_HASH_ALL=1 放行
    if(stats.presents.load()<uint64_t(from))return false;
    if(done>=frames)return false;
    if(!done)xbg_logf("[上屏] 逐帧像素 hash 首个样本 present=%llu draws=%llu",(unsigned long long)stats.presents.load(),(unsigned long long)draws);
    ++done;return true;
}
// 8 字节步进的 FNV 变体：8 MB 一帧要能扛住 1.2 fps 的节奏（逐字节版会慢 4–8 倍）
static uint64_t pixel_hash8(const uint8_t *p,size_t nb){
    uint64_t h=1469598103934665603ull;size_t i=0;
    for(;i+8<=nb;i+=8){uint64_t w;memcpy(&w,p+i,8);h^=w;h*=1099511628211ull;h^=(h>>29);}
    for(;i<nb;i++){h^=p[i];h*=1099511628211ull;}
    return h;
}
static void present_frame_hash_report(const Buffer &buf,uint32_t w,uint32_t h,uint32_t bpp,uint64_t serial){
    if(!buf.mapped||!w||!h||bpp<3)return;
    const uint8_t *p=static_cast<const uint8_t*>(buf.mapped)+buf.offset;
    size_t nb=size_t(w)*h*bpp;
    uint64_t ph=pixel_hash8(p,nb);
    uint64_t nz=0,sum=0,n=uint64_t(w)*h;
    for(uint64_t i=0;i<n;i++){const uint8_t *q=p+i*bpp;if(q[0]|q[1]|q[2])++nz;sum+=q[0];sum+=q[1];sum+=q[2];}
    xbg_logf("[上屏] 帧hash present=%llu serial=%llu %ux%u hash=%016llx 非黑=%llu(%.1f%%) 均值RGB=%.1f",
        (unsigned long long)stats.presents.load(),(unsigned long long)serial,w,h,
        (unsigned long long)ph,(unsigned long long)nz,100.0*double(nz)/double(n),
        double(sum)/double(n*3));
}
// np721：**一帧的时间去哪了**。只读打点（DM_FRAME_TIME=1），一帧一行：
//   interval = 本次 present 距上次 present 的墙钟间隔（= 帧时）
//   submit   = 本帧在 xbg_submit_now 里录制/遍历 op 的 CPU 时间（perf.submit_cpu_ns 增量）
//   present  = 本次 present 调用自身的耗时
//   gpu      = 本帧 GPU 时间增量（perf.gpu_ns，来自 pass 时间戳）
//   rest     = interval − submit − present ⇒ 客体自己的 CPU + 驱动队列/同步等待（这一项就是 np720 那 300 ms 的候选池）
namespace frametime{
// /proc/self/stat 的 utime+stime（tick=10 ms）。判「rest 是客体 CPU 还是在等」：
// 帧时里 CPU 占用接近 100% ⇒ 客体翻译执行主导；明显低 ⇒ 还在等驱动/队列。
static bool proc_cpu_ms(uint64_t &out){
    FILE *f=fopen("/proc/self/stat","r");if(!f)return false;
    char buf[1024];size_t n=fread(buf,1,sizeof(buf)-1,f);fclose(f);buf[n]=0;
    char *p=strrchr(buf,')');if(!p)return false;++p;
    unsigned long long v[24]={0};int i=0;
    for(char *t=strtok(p," \n");t&&i<24;t=strtok(nullptr," \n"))v[i++]=strtoull(t,nullptr,10);
    if(i<13)return false;
    out=(v[11]+v[12])*10ull;   // v[0]=state(第3字段) ⇒ utime=第14字段=v[11]、stime=v[12]
    return true;
}
static int on=-1;
static bool enabled(){if(on<0){const char*e=getenv("DM_FRAME_TIME");on=(e&&*e=='1')?1:0;if(on)xbg_logf("[帧时] DM_FRAME_TIME=1 开启（一帧一行，只读）");}return on!=0;}
static void report(uint64_t now,uint64_t present_begin){
    if(!enabled())return;
    static uint64_t last=0,last_submit=0,last_gpu=0;static uint64_t n=0;
    double interval=last?(double(now-last)/1e6):0.0;
    // 铁律 37③：perf_* 在 PERF 窗尾被 perf=Perf{} 清零，跨重置做差会 uint64 包回绕
    //   （np722 实测 submit_ms=1.8e13 / gpu_ms=9.2e12）⇒ 先在无符号域钳负再转 double。
    uint64_t d_sub=perf.submit_cpu_ns>=last_submit?perf.submit_cpu_ns-last_submit:0;
    double submit=double(d_sub)/1e6;
    double present=double(now-present_begin)/1e6;
    uint64_t d_gpu=perf.gpu_ns>=last_gpu?perf.gpu_ns-last_gpu:0;
    double gpu=double(d_gpu)/1e6;
    double rest=interval-submit-present;
    static uint64_t last_plan=0,last_cpu=0;
    uint64_t d_plan=frameskip::plan_ns>=last_plan?frameskip::plan_ns-last_plan:0;
    double plan=double(d_plan)/1e6;
    uint64_t cpu=0;double cpu_ms=0.0,cpu_pct=-1.0;
    if(proc_cpu_ms(cpu)){
        if(last_cpu&&interval>0.0){cpu_ms=double(cpu-last_cpu);cpu_pct=100.0*cpu_ms/interval;}
        last_cpu=cpu;
    }
    if(interval>0.0)
        xbg_logf("FRAMETIME presents=%llu interval_ms=%.1f submit_ms=%.1f plan_ms=%.1f present_ms=%.1f gpu_ms=%.1f rest_ms=%.1f cpu_pct=%.0f",
            (unsigned long long)n,interval,submit,plan,present,gpu,rest>0?rest:0.0,cpu_pct);
    last=now;last_submit=perf.submit_cpu_ns;last_gpu=perf.gpu_ns;last_plan=frameskip::plan_ns;++n;
}
}
static void xbg_present_texture_now(const win_t *window,const tex_t *texture){std::lock_guard<std::recursive_mutex> lock(mutex);preflight_cpu_reads();deathguard::frame_seal();{SiteScope site(SITE_PRESENT);cur_op=61;TimingRoot timing_root;CPU_SCOPE(Other);need(initialized&&cfg.present,"Game presentation callback not installed");
         // np662：**切后台不挂死**。表面销毁后上屏回调会在 wait_window() 无限等新表面 ⇒
         // 帧循环整个停住、看门狗 90 秒判死（死状与真卡点难辨）。无窗口时跳过本帧出图、
         // 游戏照常跑（回前台自动恢复出图）；提交侧照旧（冷却逻辑不变）。
         // np662: don't hang when backgrounded. After surface destruction the present callback blocks
         //   forever in wait_window() -> frame loop stalls -> watchdog kill (indistinguishable from a real
         //   stall). Without a window, skip presenting this frame and keep the game running (presenting
         //   resumes automatically when the surface returns); submission path unchanged.
         if(!mtl_has_window()){static int sn=0;if(sn++<3)xbg_logf("【后台】无上屏表面 —— 跳过出图，游戏继续跑（第 %d 次）",sn);return;}
         uint64_t present_begin=timing_clock();Image &im=get_image(texture,false);need(im.gpu_written,"Game tried to present a texture never rendered");
         // np665：上屏源图自检（见 present_probe_report 的注释）。只做一次，读回那张**正要上屏**的纹理。
         // np665: presented-source self-check (see present_probe_report). One shot, reads back the exact
         //   texture that is about to be presented.
         const int probe_n=present_probe_target();
         static bool probe_done=false;
         const bool probe_now = (probe_n>0 && !probe_done && stats.presents.load()>=uint64_t(probe_n)) || present_probe_due();
         const bool hash_now=present_hash_now(uint64_t(g_draws_since_present));   // np715 逐帧像素 hash（默认关）   // np715 逐帧像素 hash（默认关）
         Buffer probe_buf;
         Commands c;im.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
         if((probe_now||hash_now) && im.bpp>=4 && im.samples==VK_SAMPLE_COUNT_1_BIT && (im.aspects&VK_IMAGE_ASPECT_COLOR_BIT)){
             probe_buf=readback_buffer(size_t(im.width)*im.height*4);
             // np668：**自检的对照**。DM_PRESENT_PROBE_FILL=1 时先把源图整张涂绿再读回：
             //   读回是绿的 ⇒ 读回通路本身没问题，之前读到的全 0 是真的"源图黑"；
             //   读回还是 0 ⇒ 坏的是读回通路（copy 没生效/读错内存），之前的结论要作废。
             // np668: control for the probe. With DM_PRESENT_PROBE_FILL=1 the source image is cleared to green
             //   before the readback: green comes back => the readback path is sound and the zeros were real;
             //   still zeros => the readback path is broken and the earlier conclusion must be discarded.
             static int probe_fill=-1;if(probe_fill<0){const char *e=getenv("DM_PRESENT_PROBE_FILL");probe_fill=(e&&*e=='1')?1:0;if(probe_fill)xbg_logf("[上屏] 自检对照开启：读回前先涂绿");}
             if(probe_fill && probe_now){
                 VkImageSubresourceRange rr{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
                 VkImageMemoryBarrier tb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};tb.srcQueueFamilyIndex=tb.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;tb.image=im.image;tb.subresourceRange=rr;
                 tb.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;tb.newLayout=VK_IMAGE_LAYOUT_GENERAL;tb.srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT;tb.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
                 vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&tb);
                 VkClearColorValue gc{};gc.float32[1]=1.f;
                 vkCmdClearColorImage(c.command,im.image,VK_IMAGE_LAYOUT_GENERAL,&gc,1,&rr);
                 tb.oldLayout=VK_IMAGE_LAYOUT_GENERAL;tb.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;tb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;tb.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
                 vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&tb);
             }
             VkBufferImageCopy pr{};pr.bufferOffset=probe_buf.offset;
             pr.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};pr.imageExtent={im.width,im.height,1};
             vkCmdCopyImageToBuffer(c.command,im.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,probe_buf.buffer,1,&pr);
             VkBufferMemoryBarrier ph{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};ph.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;ph.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
             ph.srcQueueFamilyIndex=ph.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;ph.buffer=probe_buf.buffer;ph.offset=probe_buf.offset;ph.size=probe_buf.size;
             vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&ph,0,nullptr);
             if(probe_now)xbg_logf("[上屏] 自检启动：present=%llu 纹理 serial=%llu %ux%u fmt=%#x",(unsigned long long)stats.presents.load(),
                 (unsigned long long)texture->serial,im.width,im.height,unsigned(im.fmt));
         }
         c.finish();submit_slots(false);
         if((probe_now||hash_now) && probe_buf.buffer){gpu_sync(perf.sync_readback);probe_buf.invalidate();
             if(hash_now)present_frame_hash_report(probe_buf,im.width,im.height,4,texture->serial);
          if(probe_now){present_probe_report(probe_buf,im.width,im.height,4);
             xbg_logf("[上屏] 绘制落点：一帧 %llu 次绘制，打到 serial=%llu 的 %llu 次；最后一次目标 serial=%llu image=%p；上屏 image=%p 同一镜像=%d",
                 (unsigned long long)g_draws_since_present,(unsigned long long)texture->serial,
                 (unsigned long long)g_draws_to_present_serial,(unsigned long long)g_last_draw_serial,g_last_draw_image,
                 (void*)im.image,int(g_last_draw_image==(void*)im.image));
             xbg_logf("[上屏] 溯源：gpu_written=%d provenance_site=%u provenance_epoch=%llu last_write=%llu layout=%d checksub=%llu",
                 int(im.gpu_written),im.provenance_write_site,(unsigned long long)im.provenance_epoch,
                 (unsigned long long)im.last_write,int(im.layout),(unsigned long long)im.checked_submission);
             present_probe_other_serials();
             if(probe_n>0 && stats.presents.load()>=uint64_t(probe_n))probe_done=true;}
         }
         g_present_serial_hint=texture->serial;g_winfill_serial=texture->serial;
         {bool known=false;for(int i=0;i<g_win_n;i++)if(g_win_serials[i]==texture->serial)known=true;
          if(!known && g_win_n<8){g_win_serials[g_win_n++]=texture->serial;xbg_logf("[上屏] 登记窗口缓冲 serial=%llu（共 %d 个）",(unsigned long long)texture->serial,g_win_n);}
          frameskip::note_window_serial(texture->serial);}   // 让阶段 2 知道哪趟是写窗口的合成趟g_draws_since_present=0;g_draws_to_present_serial=0;
         int r;{CPU_SCOPE(Present);
         // np658：reset 恢复窗口内先安静（present 也走队列，同样会喂 render_service 的失败计数）。
         // np658: stay quiet during the recovery window (present also goes through the queue and feeds the counter).
         if(unsigned cd=lost_cooldown_ms()){usleep(cd*1000u);}
         r=cfg.present(cfg.userdata,window->native,im.image,im.fmt,im.width,im.height,window->crop,window->interval);}
         // np651：**present 也要兜底**。真 device lost 时 present 返回 -4，原来直接 `xbg_fatalf`
         //   ——np649/np650 两跑都死在这里（`FATAL: Game presentation failed: -4`），而同一份代码
         //   np648 能跑 236 秒，说明这是随机的 lost 穿透，不是确定性 bug。按提交路径同款重试；
         //   重试仍失败就**丢掉这一帧**（不 present、不加 presents），进程继续活。
         // np651: the present path needs the same fallback. On a real device loss present returns -4 and the
         //   code used to call xbg_fatalf -- np649/np650 both died here while the same build (np648) ran for
         //   236 s, i.e. a random lost, not a deterministic bug. Retry like the submit path; if it still fails,
         //   drop this frame (no present, no presents++) and keep the process alive.
         if(r && retry_submit_enabled()){
             static int pn=0;if(pn++<3)xbg_logf("【重试】present %d，开始退避",int(r));
             // np658：先睡掉冷却窗再敲 present（present 也会触发队列错误被 render_service 记账）。
             // np658: sleep through the cooldown before knocking on present (it also feeds the fail counter).
             if(unsigned cd=lost_cooldown_ms()){usleep(cd*1000u);}
             for(int a=0;a<6 && r;++a){usleep(3000u<<a);
                 if(unsigned cd2=lost_cooldown_ms()){usleep(cd2*1000u);}  // np661：present 重试同款。
                 r=cfg.present(cfg.userdata,window->native,im.image,im.fmt,im.width,im.height,window->crop,window->interval);}
             for(int a=0;a<20 && r;++a){usleep(100000u);
                 if(unsigned cd3=lost_cooldown_ms()){usleep(cd3*1000u);}  // np661：present 长宽限同款。
                 r=cfg.present(cfg.userdata,window->native,im.image,im.fmt,im.width,im.height,window->crop,window->interval);}
             static int okp=0;if(!r && okp++<3)xbg_logf("【宽限】present 重试成功");
         }
         if(r){static int dn=0;if(dn++<8)xbg_logf("【红】present 失败 %d —— 丢掉这一帧（第 %d 次）",int(r),dn);return;}
         frametime::report(timing_clock(),present_begin);
         ++stats.presents;if(pass_prof_enabled && stats.presents.load()==pass_prof_frame)xbg_logf("NVN GPASS boundary frame=%llu slots=%u skipped_slots=%u reserved_queries=%u max_queries=%u max_slots=%u",(unsigned long long)pass_prof_frame,pass_prof_slots,pass_prof_skipped_slots,pass_prof_queries,pass_prof_query_limit,pass_prof_slot_limit);perf.present_cpu_ns+=timing_clock()-present_begin;perf_frame();if(provenance_file && stats.presents.load()>=provenance.target)provenance_close("END");if(capture.trace && stats.presents==capture.frame){capture_log("END frame=%llu events=%u bytes=%llu presentedSerial=%llu image=%p",(unsigned long long)capture.frame,capture.events,(unsigned long long)capture.bytes,(unsigned long long)texture->serial,(void*)im.image);fclose(capture.trace);capture.trace=nullptr;capture.active=false;xbg_logf("NVN one-frame capture complete frame=%llu events=%u bytes=%llu",(unsigned long long)capture.frame,capture.events,(unsigned long long)capture.bytes);}}timing_boundary();}
extern "C" void xbg_legacy_read_texels(const tex_t *t,unsigned level,unsigned layer,const NVNcopyRegion *r,uint8_t *dst,size_t row,size_t image){gfx_drain();std::lock_guard<std::recursive_mutex> lock(mutex);preflight_cpu_reads();need(t&&r&&dst,"Invalid ReadTexels arguments");auto found=images.find(t->serial);if(found==images.end() || !found->second->gpu_written){int bw,bh,bpp=nvn_fmt_bpb(t->d.format,&bw,&bh);need(bpp && r->xoffset==0 && r->yoffset==0 && r->zoffset==0 && r->depth==1 && uint32_t(r->width)==std::max(1u,t->d.width>>level) && uint32_t(r->height)==std::max(1u,t->d.height>>level),"CPU ReadTexels partial region unsupported");need(!(t->d.flags&0x10),"Linear CPU ReadTexels not implemented");need(nvn_deswizzle(&t->d,level,layer,t->mem,dst,row)==0,"ReadTexels deswizzle failed");}
    else{p_copy_b2t p{};p.src=uint64_t(dst);p.tex=t;p.has_view=1;p.view.set_levels=p.view.set_layers=1;p.view.base_level=level;p.view.nlevels=1;p.view.min_layer=layer;p.view.nlayers=1;p.r=*r;int64_t old=state.copy_stride,old_image=state.copy_image_stride;state.copy_stride=row;state.copy_image_stride=image;copy_texture_buffer(p,true);state.copy_stride=old;state.copy_image_stride=old_image;}}

// ---- public entry points (queued in order on the graphics thread) ----
extern "C" void xbg_legacy_pool_add(void *p,size_t n){gfx_enqueue([p,n]{xbg_legacy_pool_add_now(p,n);});}
extern "C" void xbg_legacy_pool_del(void *p){gfx_enqueue([p]{xbg_legacy_pool_del_now(p);});}
extern "C" void xbg_register_texture(const dpool_t *p,int id,const tex_t *t,const view_t *v){
    if(!gfx_async){xbg_register_texture_now(p,id,t,v);return;}
    need(p&&t,"Invalid texture registration");dpool_t pool=*p;tex_t tex=*t;view_t view{};bool has=v!=nullptr;if(v)view=*v;
    gfx_enqueue([pool,id,tex,view,has]{xbg_register_texture_now(&pool,id,&tex,has?&view:nullptr);});}
extern "C" void xbg_register_sampler(const dpool_t *p,int id,const samp_par *s){
    if(!gfx_async){xbg_register_sampler_now(p,id,s);return;}
    need(p&&s,"Invalid sampler registration");dpool_t pool=*p;samp_par sp=*s;gfx_enqueue([pool,id,sp]{xbg_register_sampler_now(&pool,id,&sp);});}
extern "C" void xbg_legacy_tex_release(tex_t *t){if(!t)return;if(!gfx_async){xbg_legacy_tex_release_now(t);return;}tex_t copy=*t;gfx_enqueue([copy]()mutable{xbg_legacy_tex_release_now(&copy);});}
extern "C" void xbg_legacy_prog_release(prog_t *p){if(!p)return;if(!gfx_async){xbg_legacy_prog_release_now(p);return;}prog_t copy=*p;gfx_enqueue([copy]()mutable{xbg_legacy_prog_release_now(&copy);});}
extern "C" void xbg_texture_invalidate(const tex_t *t){if(!t)return;if(!gfx_async){xbg_texture_invalidate_now(t);return;}tex_t copy=*t;gfx_enqueue([copy]{xbg_texture_invalidate_now(&copy);});}
extern "C" void xbg_submit(int n,const uint64_t *handles){
    if(!gfx_async){xbg_submit_now(n,handles);return;}
    std::vector<uint64_t> list(handles,handles+n);++gfx_enqueued;
    gfx_enqueue([list]{xbg_submit_now(int(list.size()),list.data());gfx_executed.store(submitted);});}
extern "C" void xbg_present_texture(const win_t *window,const tex_t *texture){
    if(!gfx_async){xbg_present_texture_now(window,texture);return;}
    // At most two frames queued ahead of the graphics thread.
    {uint64_t t0=uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());std::unique_lock<std::mutex> l(gfx_job_mutex);gfx_idle_cv.wait(l,[]{return gfx_presents_queued.load()<2;});gfx_pace_ns+=uint64_t(std::chrono::steady_clock::now().time_since_epoch().count())-t0;}
    win_t w=*window;tex_t t=*texture;++gfx_presents_queued;
    gfx_enqueue([w,t]{xbg_present_texture_now(&w,&t);{std::lock_guard<std::mutex> l(gfx_job_mutex);--gfx_presents_queued;}gfx_idle_cv.notify_all();});}

extern "C" void xbg_legacy_register_texture(int id,const tex_t *t,const view_t *v){std::lock_guard<std::recursive_mutex> l(mutex);need(id>=0&&t,"Invalid legacy texture registration");if(register_texture_record(legacy_textures[id],*t,v))++memo_bump[0];}
extern "C" void xbg_legacy_register_sampler(int id,const samp_par *s){std::lock_guard<std::recursive_mutex> l(mutex);need(id>=0&&s,"Invalid legacy sampler registration");legacy_samplers[id]=*s;++memo_epoch;++memo_bump[1];}
