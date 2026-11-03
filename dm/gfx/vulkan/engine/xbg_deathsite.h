// np701：判死证据链的跨编译单元入口。xbg_game.cpp 实现登记表与监视线程；
// dm/android/vulkan_present.cpp（鸿蒙与安卓共用）用它给 acquire/submit/present/fence 挂守卫。
// 站点编号是两个编译单元之间的契约：只在文件尾追加，不改现有值。
// np701: the death-evidence chain's cross-TU entry. xbg_game.cpp owns the registry and the monitor;
// dm/android/vulkan_present.cpp (shared by HarmonyOS and Android) guards acquire/submit/present/fence
// with it. Site numbers are a contract between the two TUs: append only, never renumber.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
enum {
    XBG_SITE_NONE=0, XBG_SITE_SUBMIT, XBG_SITE_SUBMIT_SPLIT, XBG_SITE_FENCE_RETIRE, XBG_SITE_FENCE_POOLRESET,
    XBG_SITE_PIPE_GRAPHICS, XBG_SITE_PIPE_COMPUTE, XBG_SITE_PIPE_OTHER, XBG_SITE_QUEUE_WAIT_IDLE,
    XBG_SITE_ACQUIRE, XBG_SITE_PRESENT, XBG_SITE_PRESENT_SUBMIT, XBG_SITE_PRESENT_FENCE
};
void xbg_death_site_enter(unsigned site,unsigned long long ctx);
void xbg_death_site_leave(void);
#ifdef __cplusplus
}
struct XbgDeathSite{
    explicit XbgDeathSite(unsigned site,unsigned long long ctx=0){xbg_death_site_enter(site,ctx);}
    ~XbgDeathSite(){xbg_death_site_leave();}
};
#endif
