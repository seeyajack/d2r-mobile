/**
 * libdms2 —— S2/S3 的 N-API 接口（实现：entry/src/main/cpp/napi_init.c）
 *
 * probe()  读表面接线状态：window/size/vk_has_window/screen_max_fps
 * setPad() 手柄状态推进宿主（dm_pad_set）。buttons 位图与 Switch Pro 一一对应：
 *           A=1<<0 B=1<<1 X=1<<2 Y=1<<3 LS=1<<4 RS=1<<5 L=1<<6 R=1<<7
 *           ZL=1<<8 ZR=1<<9 PLUS=1<<10 MINUS=1<<11 LEFT=1<<12 UP=1<<13 RIGHT=1<<14 DOWN=1<<15
 *           lx/ly/rx/ry 摇杆，量程 ±32767
 * start()  起 guest：<runtime>/out/ 放改写产物、<runtime>/shaders/ 放 .xgs；
 *           内部先 dm_android_vk_initialize(shaders) 再 dm_main(["dmhost", out, "main", "sdk"])
 */
export const probe: () => string;
export const setPad: (buttons: number, lx: number, ly: number, rx: number, ry: number) => void;
export const start: (runtimeDir: string, logPath?: string, markerPath?: string, envs?: string) => void;
export const readLog: (path: string, mode?: 'head' | 'tail' | 'all') => string;
export const listDir: (path: string) => string;
export const dirFileCount: (path: string) => number;   // 递归文件数；目录不存在返回 -1（原生 stat/readdir，绕开 ArkTS fs 的 np279 坑）
export const makeDir: (path: string) => number;        // 0 成功 / 1 已存在 / -errno 失败（np345d 诊断）
export const logLine: (line: string) => void;
export const writeFile: (path: string, data: ArrayBuffer) => number;
export const writeFileGz: (path: string, gzData: ArrayBuffer) => number;   // 解 gzip 落盘（libdeflate）；成功=字节数，失败=-errno
