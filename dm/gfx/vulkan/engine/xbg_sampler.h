#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct xbg_nvn_sampler_desc {
    uint32_t min_filter,mag_filter,wrap_s,wrap_t,wrap_r,compare_mode,compare_func;
    float min_lod,max_lod,lod_bias,anisotropy,border[4];
} xbg_nvn_sampler_desc;
typedef struct xbg_sampler_limits {
    VkBool32 anisotropy_enabled,mirror_clamp_to_edge_enabled,integer_border;
    float max_anisotropy,max_lod_bias;
    // np602：VK_EXT_custom_border_color 真的可用（扩展已在设备上启用，且 customBorderColors 支持）。
    // 为假时非定点边框色仍然报 -7，不做近似。
    // np602: VK_EXT_custom_border_color is genuinely usable (extension enabled on the device and
    // customBorderColors supported). When false, a noncanonical border still fails with -7, never approximated.
    VkBool32 custom_border_color;
} xbg_sampler_limits;
/* Pure conversion of actual NVN state. No Vulkan call. Unknown legacy clamp
 * modes and noncanonical borders fail instead of being approximated.
 * When a border needs VK_EXT_custom_border_color, *out gets borderColor = *_CUSTOM_EXT and the four
 * components are written to custom_rgba (may be null; it is only written in that case), so the caller
 * can chain VkSamplerCustomBorderColorCreateInfoEXT with the format it actually knows.
 * Returns 0 on success and leaves output untouched on failure. */
int xbg_sampler_from_nvn(const xbg_nvn_sampler_desc *nvn,const xbg_sampler_limits *limits,VkSamplerCreateInfo *out,float custom_rgba[4]);
const char *xbg_sampler_error(int result);
#ifdef __cplusplus
}
#endif
