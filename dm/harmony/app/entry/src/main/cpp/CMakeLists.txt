# S2-b 的 CMake（np233）—— 在 S2-a 基础上接入 **上游 d2r 的宿主层**
#
# 为什么是 d2r 的宿主而不是 nxnative 的（数据见 `进度与待办-2026-10-23.md` §1.12）：
#   nxnative 宿主是"苹果专用"的（9 个 Apple 框架 + Mach-O 段嵌入），
#   d2r 宿主是"为跨平台写的"（同一份编 Apple / Android / HarmonyOS）：
#   编译普查 43 → 63（加上游自带 dm/android/compat/include）→ 67（-std=gnu11）
#   ⇒ 我们不需要自己搓 Apple 兼容层。
cmake_minimum_required(VERSION 3.22)
project(dms2 LANGUAGES C CXX ASM)     # ASM：宿主的 entry.S（未实现导入的记名桩表）

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../)
set(E ${ROOT}/dm/gfx/vulkan/engine)
set(H ${ROOT}/dm/host)

# ── 上游 Vulkan 引擎（20 个源，门 1 已验证）──
add_library(dmVulkanEngine STATIC
  ${E}/xbg_game.cpp ${E}/xbg_shader_key.c ${E}/xbg_command_clone.c
  ${E}/xbg_draw_plan.c ${E}/xbg_vertex.c ${E}/xbg_quad.c ${E}/xbg_stages.cpp
  ${E}/xbg_clear.cpp ${E}/xbg_blit.cpp ${E}/xbg_texture.c ${E}/xbg_depth.c
  ${E}/xbg_framebuffer.c ${E}/xbg_vulkan.cpp ${E}/xbg_nvn.cpp ${E}/xbg_sampler.cpp
  ${E}/xbg_spirv_quad.cpp ${E}/xbg_spirv_warp.cpp ${E}/xbg_spirv_ldc.cpp
  ${E}/xbg_spirv_temps.cpp ${E}/xbg_spirv_raster.cpp)
target_include_directories(dmVulkanEngine PUBLIC ${ROOT}/dm/gfx/vulkan
  PRIVATE ${ROOT}/dm/gfx/nvn/include ${ROOT}/gen ${ROOT}/dm/host)
target_compile_features(dmVulkanEngine PUBLIC cxx_std_17)
target_compile_options(dmVulkanEngine PRIVATE -O2 -fno-strict-aliasing -Wno-missing-field-initializers -Wno-format)
set_source_files_properties(${E}/xbg_texture.c PROPERTIES COMPILE_OPTIONS -fsigned-char)

# ── 宿主：d2r 的 dm/host，剔掉 6 个 D2R 专属/苹果专属文件 ──
# 剔除理由：D2R 专属功能（rift / cloudsync / cloudmerge / textbox / vault_ui / bright）
# 与苹果触摸（touch_mac）。
# 注：diag.c **要编**——它导出二十多个被通用代码引用的符号（probe_install / ticks_install /
#   dm_loadscreen_up / c2s_deliver / dm_prof_* …），去掉它就得写二十多条替身。
#   它要的 D2R 经验表用 BotW 壳替身顶上（gen/dm_hud_exp.h，全 0 表）。
#   ⚠ 行为边界见 gen/dm_hud_exp.h 头注释：players_apply() 只在有人请求 /players 时才动
#     （那会跳 D2R 硬编码地址，BotW 上必崩），所以**不要在 BotW 上调 dm_hud_request_players()**。
# ⚠ CONFIGURE_DEPENDS：往 dm/host 里**新增** .c 时不必再手改这份清单。
#   没有它，GLOB 只在 configure 时算一次 ⇒ 新文件不参与编译，链接期报 `undefined symbol`
#   （2026-10-28 加 hle_msgq.c 时踩到，白跑一次构建）。
# ⚠ CONFIGURE_DEPENDS: adding a new .c under dm/host no longer requires editing this list.
#   Without it the glob is evaluated only at configure time, so a new file is not compiled and the link
#   fails with `undefined symbol` (hit on 2026-10-28 when adding hle_msgq.c, one wasted build).
file(GLOB HOST_SRC CONFIGURE_DEPENDS ${H}/*.c)
foreach(bad dm_bright.c dm_cloudmerge.c dm_cloudsync.c dm_rift.c
            dm_textbox.c dm_vault_ui.c touch_mac.c)
  list(REMOVE_ITEM HOST_SRC ${H}/${bad})
endforeach()

# 上游 compat（Apple 头的替身实现）与 libdeflate（hle_tact.c 用）
file(GLOB COMPAT_SRC ${ROOT}/dm/android/compat/*.c)
file(GLOB DEFLATE_SRC ${ROOT}/dm/third_party/libdeflate/lib/*.c ${ROOT}/dm/third_party/libdeflate/lib/arm/*.c)
# 宿主的汇编：entry.S（未实现导入的记名桩表 + dm_gate 门入口 + dm_thread_enter 客体线程起点）
# ⚠ 上游这份是 Mach-O 风格（符号带前导下划线），我们已在树内做 ELF 修正（见 entry.S 头两行）
file(GLOB HOST_ASM ${H}/*.S)
# NVN 的 HLE 层（hle_nvn.c 提供 nvn_new_serial / dm_nvn_frame_count / front_display_refresh /
# dm_gfx_suspend 等，宿主要用）；nvn_dirty.c 是脏区跟踪；nvn_names.c 是生成的名字表
file(GLOB NVN_HLE_SRC ${ROOT}/dm/gfx/nvn/*.c)
# ⚠ gfx/nvn/nvn_vk.c 是 OHOS 回放层（依赖 dm/ohos 的 dm_log.h/vk_ctx_ohos.h）；上屏走
#   dm/gfx/vulkan/nvn_vk.c（下面显式列出）。glob 会误收它 ⇒ 排除（2026-10-31 收编工程壳后踩到）。
list(REMOVE_ITEM NVN_HLE_SRC ${ROOT}/dm/gfx/nvn/nvn_vk.c)
# gfx 根下的 HLE 层（hle_vi.c = nn::vi 显示层、hle_nv.c = nv:: 驱动胶水层）。
# ⚠ 这两份原在 fork 里、工作副本没拷过来，导致 nn::vi 无实现 ⇒ 客体 vi 初始化时 SDK panic
#   （np242 真机：六个模块构造器全跑完后 Break reason=0，回溯落在 main+0x9fc4bc 的图形初始化）。
file(GLOB GFX_HLE_SRC ${ROOT}/dm/gfx/*.c)
# 同目录还有两份汇编：nvn_thunks.S（每个 NVN 导入一个跳板）/ nvn_entry.S（未实现/打点入口）。
# ⚠ 上游这三份 .S 都是 Mach-O 风格（符号带前导下划线），我们已在树内做 ELF 修正，
#   原版留档为 *.S.upstream-orig
file(GLOB NVN_ASM ${ROOT}/dm/gfx/nvn/*.S)

# ── N-API 库：S2-a 的全部 + 宿主 + compat + libdeflate ──
add_library(dms2 SHARED
  napi_init.c
  ${HOST_SRC}
  ${GFX_HLE_SRC}
  ${HOST_ASM}
  ${NVN_HLE_SRC}
  ${NVN_ASM}
  ${COMPAT_SRC}
  ${DEFLATE_SRC}
  ${ROOT}/dm/gfx/vulkan/nvn_vk.c
  ${ROOT}/dm/gfx/tex/nvn_deswizzle.c
  ${ROOT}/dm/android/vulkan_present.cpp
  ${ROOT}/dm/harmony/native/front_ohos_min.c
  ${ROOT}/dm/harmony/native/botw_d2r_feature_stubs.c)

# 宿主要用 GNU 扩展（typeof() / memmem()）⇒ 宿主/compat 源按 gnu11 并显式开 _GNU_SOURCE；
# 引擎仍 C++17（它自己不需要 _GNU_SOURCE）。
# ⚠ 两个坑（本轮都踩过，别再犯）：
#   1) COMPILE_OPTIONS 后直接写两个 flag 会被 CMake 当成"属性名/值"交替解析 ⇒
#      `set_source_files_properties called with incorrect number of arguments`；
#      必须先把 flag 列表放进一个变量，再整体传。
#   2) flag 之间不能写成一个带空格的字符串（会被当成单个参数）。
set(HOST_FLAGS "-std=gnu11" "-D_GNU_SOURCE")
set_source_files_properties(${HOST_SRC} ${GFX_HLE_SRC} ${HOST_ASM} ${NVN_HLE_SRC} ${NVN_ASM} ${COMPAT_SRC} ${DEFLATE_SRC}
  PROPERTIES COMPILE_OPTIONS "${HOST_FLAGS}")

# 宿主/compat/deflate 源按 gnu11 + _GNU_SOURCE；**main.c 单独加 -Dmain=dm_main**
# （同上游安卓/iOS 的做法：命令行 dmhost 用原名 main，App 库里改名，避免与 N-API 库冲突）
set_source_files_properties(${H}/main.c PROPERTIES COMPILE_OPTIONS "-Dmain=dm_main")

target_compile_definitions(dms2 PRIVATE __OHOS__)
target_compile_options(dms2 PRIVATE -O2)
target_include_directories(dms2 PRIVATE
  ${ROOT}/gen                                   # game_profile.h / dm_hud_exp.h（壳替身）
  ${ROOT}/dm/android/compat/include             # 上游 Apple 头替身层
  ${ROOT}/dm/harmony/native/android_shim        # ANativeWindow → OHNativeWindow、VK_OHOS_surface
  ${ROOT}/dm/gfx/nvn/include ${ROOT}/dm/gfx/nvn ${ROOT}/dm/gfx/tex
  ${ROOT}/dm/gfx/vulkan ${ROOT}/dm/gfx ${ROOT}/dm/host ${ROOT}/dm/third_party/libdeflate)

target_link_libraries(dms2 PUBLIC dmVulkanEngine
  vulkan hitrace_ndk.z native_display_manager native_window native_vsync native_image hilog_ndk.z
  ohaudio.so          # 宿主的 hle_audio.c 用 OHAudio NDK（OH_AudioRenderer_* / OH_AudioStreamBuilder_*）
  ace_napi.z ace_ndk.z)
