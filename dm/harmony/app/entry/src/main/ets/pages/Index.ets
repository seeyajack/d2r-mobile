/**
 * S2-a/S2-c 页面（np233）
 *
 * 两件事：
 *  ① 给 XComponent 一块表面 → 原生侧接住 OHNativeWindow（出图目标）
 *  ② 收手柄的键/轴事件 → 推进宿主（dm_pad_set）
 *
 * 手柄走 ArkUI 的键事件 + 焦点轴事件（API 15+，官方 guide：
 *  "Handling Gamepad Input Events"）。需要一个**有焦点**的组件来收事件，
 *  这里用一个常驻焦点的按钮；方向键的默认焦点跳转会干扰游戏操作，所以套了
 *  focusScopeId + arrowStepOut=false（官方对"只用方向键做游戏操作"的建议做法）。
 */
import dms2 from 'libdms2.so';
import { KeyCode } from '@kit.InputKit';
import { common } from '@kit.AbilityKit';
import fs from '@ohos.file.fs';
import resourceManager from '@ohos.resourceManager';

// KeyType 与 AxisModel 在本机 SDK 的声明里找不到（ets/api 不含这两个 kit），
// 所以这里用**数值常量**而不是枚举：
//   KeyType：Down = 0，Up = 1（OHOS 标准取值）
//   轴 ID：就是 Linux 输入子系统的 ABS_* 码 —— ABS_X=0x00 ABS_Y=0x01 ABS_Z=0x02 ABS_RZ=0x0c
// 等声明补齐后再换回枚举（可读性更好），行为不变。
const KEY_DOWN: number = 0;
const KEY_UP: number = 1;
const ABS_X: number = 0x00;
const ABS_Y: number = 0x01;
const ABS_Z: number = 0x02;
const ABS_RZ: number = 0x0c;

// 宿主按键位图（见 types/libdms2/index.d.ts）。
// ⚠ ArkTS 的严格模式不支持对象字面量 / as const / 计算属性名，所以用类 + Map。
class HostBtn {
  static readonly A: number = 1 << 0;
  static readonly B: number = 1 << 1;
  static readonly X: number = 1 << 2;
  static readonly Y: number = 1 << 3;
  static readonly LS: number = 1 << 4;
  static readonly RS: number = 1 << 5;
  static readonly L: number = 1 << 6;
  static readonly R: number = 1 << 7;
  static readonly ZL: number = 1 << 8;
  static readonly ZR: number = 1 << 9;
  static readonly PLUS: number = 1 << 10;
  static readonly MINUS: number = 1 << 11;
  static readonly LEFT: number = 1 << 12;
  static readonly UP: number = 1 << 13;
  static readonly RIGHT: number = 1 << 14;
  static readonly DOWN: number = 1 << 15;
}

// 鸿蒙键码 → 宿主办。名字以官方 KeyCode 为准；不同品牌手柄的键码可能有差异，
// 这正是官方 guide 提醒"要做兼容适配"的地方——真机上逐键核对后再固化。
function buildKeyMap(): Map<number, number> {
  const m = new Map<number, number>();
  m.set(KeyCode.KEYCODE_BUTTON_A, HostBtn.A);
  m.set(KeyCode.KEYCODE_BUTTON_B, HostBtn.B);
  m.set(KeyCode.KEYCODE_BUTTON_X, HostBtn.X);
  m.set(KeyCode.KEYCODE_BUTTON_Y, HostBtn.Y);
  m.set(KeyCode.KEYCODE_DPAD_LEFT, HostBtn.LEFT);
  m.set(KeyCode.KEYCODE_DPAD_UP, HostBtn.UP);
  m.set(KeyCode.KEYCODE_DPAD_RIGHT, HostBtn.RIGHT);
  m.set(KeyCode.KEYCODE_DPAD_DOWN, HostBtn.DOWN);
  m.set(KeyCode.KEYCODE_BUTTON_L1, HostBtn.L);
  m.set(KeyCode.KEYCODE_BUTTON_R1, HostBtn.R);
  m.set(KeyCode.KEYCODE_BUTTON_L2, HostBtn.ZL);
  m.set(KeyCode.KEYCODE_BUTTON_R2, HostBtn.ZR);
  m.set(KeyCode.KEYCODE_BUTTON_THUMBL, HostBtn.LS);
  m.set(KeyCode.KEYCODE_BUTTON_THUMBR, HostBtn.RS);
  m.set(KeyCode.KEYCODE_BUTTON_SELECT, HostBtn.MINUS);
  m.set(KeyCode.KEYCODE_BUTTON_START, HostBtn.PLUS);
  return m;
}
const KEYMAP: Map<number, number> = buildKeyMap();

const AXIS_FULL = 32767;   // 摇杆推给宿主的满程值

// fs.accessSync 在路径不存在时是**抛异常**而不是返回 false，所以包一层
// ⚠ 不要用 fs.accessSync 判存在：本机 OHOS 上它对**不存在的路径也返回成功**（不抛异常），
//   害得"首次解出"被整段跳过、宿主打不开 main.fixups.tsv。改用 listFileSync：列不到就是不存在。
function pathExists(p: string): boolean {
  try {
    fs.listFileSync(p);
    return true;
  } catch {
    return false;
  }
}

// 安全列 rawfile 目录：getRawFileListSync 对**不存在**的目录会抛
// "Invalid relative path"（不是返回空表），所以统一包一层。
function rawList(rm: resourceManager.ResourceManager, dir: string): string[] {
  try {
    return rm.getRawFileListSync(dir);
  } catch {
    return [];
  }
}

// 幂等建目录：OHOS 的 fs.mkdirSync(path, true) 对**已存在**的目录会抛 13900015 File exists
// （Node 的 recursive:true 是静默成功的，所以容易踩）。这里统一吞掉——建目录的语义本来就是"确保存在"。
function ensureDir(p: string): void {
  try {
    fs.mkdirSync(p, true);
  } catch {
    // 已存在 / 权限问题都不该中断启动流程
  }
}

// 启动跟踪：每步都追加写进 filesDir/boot-trace.txt。
// 起 guest 会崩（宿主 die 是 abort），崩了屏幕就没了——这个文件留得下，重启后从界面回看。
function trace(ctx: common.UIAbilityContext, line: string): void {
  try {
    dms2.logLine(line);          // 后台随时可读（hilog 标签 dm-host，前缀 ark）
  } catch {
  }
  try {
    const p = ctx.filesDir + '/boot-trace.txt';
    const f: fs.File = fs.openSync(p, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE | fs.OpenMode.APPEND);
    fs.writeSync(f.fd, new Date().toISOString() + ' ' + line + '\n');
    fs.closeSync(f);
  } catch {
    // 跟踪写不写都不算错，忽略
  }
}

@Entry
@Component
struct Index {
  @State status: string = '（还没探测）';
  @State padStatus: string = '（还没收到手柄输入）';
  @State runtimeDir: string = '';   // 留空 = 用应用 filesDir（素材由 HAP rawfile 解出；hdc 写不进沙箱，/data/local/tmp 应用又读不了）
  @State ticks: number = 0;

  // 手柄状态（只有变化时才推进宿主，避免每帧刷）
  private buttons: number = 0;
  private lx: number = 0;
  private ly: number = 0;
  private rx: number = 0;
  private ry: number = 0;

  private pushPad(why: string) {
    this.ticks++;
    dms2.setPad(this.buttons, this.lx, this.ly, this.rx, this.ry);
    this.padStatus = `${why} #${this.ticks}  buttons=0x${(this.buttons & 0xffff).toString(16)} ` +
      `L(${this.lx},${this.ly}) R(${this.rx},${this.ry})`;
  }

  // 把 HAP rawfile 里的一个目录树解到应用 filesDir。
  //
  // 判「目录还是文件」用**先列目录**：getRawFileListSync 对目录返回条目、对文件抛
  // "Invalid relative path"。⚠ 不能用「读内容失败 ⇒ 当目录」——**0 字节文件**读出来是
  // 长度 0 的 Uint8Array（不抛异常），会被误判成目录，接着 getRawFileListSync(文件路径)
  // 抛出来就中止整个解出（np242 真机实测，元凶是 subsdk0/1.gates.tsv 这两个空文件）。
  //
  // 写文件走原生 dms2.writeFile（fopen/fwrite + mkdir -p）：ArkTS 的 fs.writeSync 写进
  // filesDir **子目录**的文件原生侧看不见；rawfile 内容由 ArkTS 递过来。
  private extractRaw(ctx: common.UIAbilityContext, rm: resourceManager.ResourceManager,
                     rawDir: string, dstDir: string): number {
    const names: string[] = rawList(rm, rawDir);   // 列不出 ⇒ 是文件
    if (names.length > 0) {
      let count = 0;
      for (const name of names) {
        const sub = rawDir.length === 0 ? name : rawDir + '/' + name;
        count += this.extractRaw(ctx, rm, sub, dstDir + '/' + name);
      }
      return count;
    }
    // 叶子：当文件写（0 字节也写，宿主会读 gates.tsv 这类空表）
    try {
      const buf = rm.getRawFileContentSync(rawDir);
      const w: number = dms2.writeFile(dstDir, buf.buffer);
      if (w >= 0) return 1;
      // np345：writeFile 现在如实报负数（-errno，fopen 失败）。只打前 10 条失败路径，防日志洪水；
      // 全量失败数由调用方（'extract …: X 个文件' 对比 rawList 总数）暴露。
      this.writeFails++;
      if (this.writeFails <= 10) trace(ctx, '写入失败(' + w + ')：' + rawDir);
    } catch (err) {
      this.writeFails++;
      if (this.writeFails <= 10) trace(ctx, '解出失败：' + rawDir + ' → ' + (err as Error).message);
    }
    return 0;
  }

  private writeFails: number = 0;   // np345：本轮解出的写入失败总数（extractRaw 里累加）

  // 一出现就 probe：起 guest 崩了之后重启 app，UI 树上直接就能读到 host.log 尾部
  onPageShow(): void {
    try {
      const ctx = getContext(this) as common.UIAbilityContext;
      this.status = dms2.probe() +
        '\n--- out/ ---\n' + dms2.listDir(ctx.filesDir + '/out') +
        '\n--- shaders/ ---\n' + dms2.listDir(ctx.filesDir + '/shaders') +
        '\n--- boot-trace ---\n' + dms2.readLog(ctx.filesDir + '/boot-trace.txt', 'tail') +
        '\n--- host.log(头) ---\n' + dms2.readLog(ctx.filesDir + '/host.log', 'head');
    } catch (e) {
      this.status = 'probe 抛异常：' + e;
    }
  }

  build() {
    Column({ space: 10 }) {
      Text('dm · S2 表面 + 手柄接线验证')
        .fontSize(16)
        .fontWeight(FontWeight.Bold)

      // 这块表面就是出图目标：ArkUI 创建 OHNativeWindow 后经 N-API 的 surface 回调交给我们。
      XComponent({ id: 'dms2vk', type: XComponentType.SURFACE, libraryname: 'dms2' })
        .width('100%')
        .height(200)

      Button('读一次状态（probe）')
        .width('100%')
        .onClick(() => {
          try {
            this.status = dms2.probe();
          } catch (e) {
            this.status = 'probe 抛异常：' + e;
          }
        })

      // 起 guest：把运行目录交给原生层（<dir>/out 改写产物、<dir>/shaders 放 .xgs）。
      // 默认用应用 filesDir；也可以手填别的路径（例如 /data/local/tmp/dm）。
      // ⚠ 应用沙箱：hdc 写不进 filesDir，所以素材要么放进 HAP、要么走系统选择器、
      //    要么放到 hdc 可写的位置（/data/local/tmp）——本条输入框就是为这个留的。
      TextInput({ placeholder: '运行目录（留空 = 应用 filesDir）', text: this.runtimeDir })
        .width('100%')
        .height(40)
        .fontSize(12)
        .onChange((v: string) => {
          this.runtimeDir = v;
        })

      Button('起 guest（start）')
        .width('100%')
        .backgroundColor('#2e4d2e')
        .fontColor('#ffffff')
        .onClick(() => {
          try {
            const ctx = getContext(this) as common.UIAbilityContext;
            // ① 首次运行：把 HAP rawfile 里的改写产物（dmout/）拷进应用自己的目录
            //    （应用沙箱 hdc 写不进，只能由应用自己从包内取；45MB，一次即可）
            const dst = ctx.filesDir + '/out';
            trace(ctx, 'enter click, dst=' + dst + ', exists=' + pathExists(dst));
            // ⚠ 不做存在性判断：本机 fs.listFileSync/accessSync 对不存在的路径都不抛异常
            //   （害得"首次解出"被整段跳过）。改为**每次点都重新解出**（45MB、一秒多，可接受）。
            // ⚠ 写文件走原生（dms2.writeFile）：ArkTS 的 fs.writeSync 写进 filesDir 子目录的
            //   文件原生侧看不到（宿主要读的素材必须由原生落盘）；ArkTS 只负责递 rawfile 内容。
            this.status = '正在从安装包解出改写产物…';
            const rm = ctx.resourceManager;
            trace(ctx, 'rawfile 列表：共 ' + rawList(rm, 'out').length + ' 项');
            const n = this.extractRaw(ctx, rm, 'out', dst);
            this.status = '解出完成：' + n + ' 个文件';
            trace(ctx, 'extract out/: ' + n + ' 个文件');
            // ② 以 filesDir 为运行目录（dmout/ 已在里面，shaders/ 缺省为空目录）
            // 不管有没有重新解出，都把 out/ 的现状（文件名=大小）记下来——解出空文件也会让宿主打不开
            try {
              const sizes: string[] = [];
              for (const nm of fs.listFileSync(dst)) {
                sizes.push(nm + '=' + fs.statSync(dst + '/' + nm).size);
              }
              trace(ctx, 'out/ 现状: ' + sizes.join(' '));
            } catch {
              trace(ctx, 'out/ 列不出');
            }
            // 着色器投递（门票③，np346）：rawfile 里是 <key>.xgs.gz（全量 728MB→约 215MB，
            // 装机一分钟级；未压缩 1.1GB 包在无线链路上装不动）。dirFileCount 够 17000 就跳过；
            // 解出走 writeFileGz（原生 libdeflate 解 gzip + mkdirs 修复后的落盘）。
            const dir = this.runtimeDir.trim().length > 0 ? this.runtimeDir.trim() : ctx.filesDir;
            const sdir = ctx.filesDir + '/shaders';
            ensureDir(sdir);
            const gzNames: string[] = rawList(rm, 'shaders').filter((s: string): boolean => s.endsWith('.gz'));
            if (gzNames.length > 0) {
              const have: number = dms2.dirFileCount(sdir);
              if (have < 17000) {
                this.status = '正在解出着色器包（17,308 个，一次性）…';
                this.writeFails = 0;
                let sn = 0;
                for (const nm of gzNames) {
                  try {
                    const buf = rm.getRawFileContentSync('shaders/' + nm);
                    const dst = sdir + '/' + nm.substring(0, nm.length - 3);   // 去掉 .gz
                    const w: number = dms2.writeFileGz(dst, buf.buffer);
                    if (w >= 0) sn++;
                    else { this.writeFails++; if (this.writeFails <= 10) trace(ctx, '着色器解出失败(' + w + ')：' + nm); }
                  } catch (err) {
                    this.writeFails++;
                    if (this.writeFails <= 10) trace(ctx, '着色器读取失败：' + nm + ' → ' + (err as Error).message);
                  }
                }
                trace(ctx, 'extract shaders(gz)/: ' + sn + '/' + gzNames.length + ' 个, 失败 ' + this.writeFails + ' → ' + sdir);
              } else {
                trace(ctx, 'shaders: 已有 ' + have + ' 个，跳过解出');
              }
            } else {
              const have: number = dms2.dirFileCount(sdir);
              trace(ctx, '瘦包：shaders 沙箱计数=' + have + (have >= 17000 ? '（够）' : '（不足 ⇒ 第一个管线创建会 FATAL）'));
            }
            // np548：增量补包。上面那段是"一次性全量"（`have < 17000` 才解，够数就整段跳过），
            //   而 hdc 写不进应用沙箱（shell 与 file send 都是 Permission denied），
            //   ⇒ 后来补的包只能走 HAP：放进 rawfile/shaders_extra/<键>.xgs.gz，在这里解出。
            //   为什么单独一个目录而不是并进 shaders/：全量那段的跳过条件会把新包一起无视。
            // ★ np548：原来这里**每次启动都无条件解一遍**，理由是"几个到几十个"。但设备上那套包是
            //   1.0.0 时代翻的，与 1.6.0 的新键只重叠 6,036 个（缺 9,626 个）⇒ 一次要解近万个，
            //   每次启动都解会明显拖慢。改成**沙箱里已经有同名 .xgs 就跳过**：
            //   只多出几十毫秒的 accessSync，且天然能自愈"上次解到一半被杀"。
            //   注意：这条规则意味着同一个键的包内容变了不会被覆盖（键是内容哈希，内容变了键就变，
            //   所以实际上不会发生）；真要换内容，删掉沙箱里那个 .xgs 即可。
            //   np548: this used to extract unconditionally because the count was tiny. The device's bundles were
            //   translated in the 1.0.0 era and only 6,036 keys overlap with 1.6.0 (9,626 missing), so a start would
            //   extract ~10k files. Now a bundle already present in the sandbox is skipped: costs tens of ms of
            //   accessSync and heals a half-finished extraction. (A key is a content hash, so a changed bundle is a
            //   different key; to force a rewrite, delete that .xgs.)
            {
              const extra: string[] = rawList(rm, 'shaders_extra').filter((s: string): boolean => s.endsWith('.gz'));
              if (extra.length > 0) {
                let ok = 0;
                let skip = 0;
                const t0: number = Date.now();
                for (const nm of extra) {
                  const dst = sdir + '/' + nm.substring(0, nm.length - 3);
                  try {
                    if (fs.accessSync(dst)) { skip++; continue; }
                  } catch (e) { }
                  try {
                    const buf = rm.getRawFileContentSync('shaders_extra/' + nm);
                    const w: number = dms2.writeFileGz(dst, buf.buffer);
                    if (w >= 0) ok++; else trace(ctx, '增量包解出失败(' + w + ')：' + nm);
                  } catch (err) {
                    trace(ctx, '增量包读取失败：' + nm + ' → ' + (err as Error).message);
                  }
                }
                trace(ctx, 'shaders_extra: 新解 ' + ok + ' 个、已存在跳过 ' + skip + ' 个（共 ' + extra.length +
                      '），耗时 ' + (Date.now() - t0) + ' ms → ' + sdir);
              }
            }
            const log = ctx.filesDir + '/host.log';
            this.status = 'start → ' + dir + '\nlog → ' + log;
            // 探针文件：原生侧会试读它，用来判断"ArkTS 写的文件原生到底能不能看到"
            const marker = ctx.filesDir + '/native-probe.txt';
            try {
              const mf: fs.File = fs.openSync(marker, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE | fs.OpenMode.TRUNC);
              fs.writeSync(mf.fd, 'PROBE-OK-' + Date.now());
              fs.closeSync(mf);
              trace(ctx, 'marker 已写：' + marker + ' 大小=' + fs.statSync(marker).size);
            } catch {
              trace(ctx, 'marker 写失败');
            }
            trace(ctx, 'calling start(' + dir + ', ' + log + ', ' + marker + ')');
            // 第四个参数：调试开关 + 内容目录（空格分隔 KEY=VALUE）。
            //  DM_IMPORTS=1  导入解析全打印（排障用）
            //  DM_ROMFS      客体 romfs 目录；HAP 里带了 romfs/ 才解出并指过去
            //  DM_TEMP       临时区（存档等）
            //  np429 第四轮：① main:a66238 = 0xa661e8 里 `bl 0xa08e94` 的返回点；
            //  ② main:a0d0f0 = 0xa08e94 成功后的尾调用（疑似"等完成"）。采样加密到 500ms。
            //  np430：@0 = 转储 [x0] 16 字节 ⇒ 取"上下文/管理器"的**虚表指针**（离线解虚表找消费函数）。
            //  上一轮用的 @88 只给了计数，没给类身份；这一轮换成 @0。采样放稀到 2000ms 减少噪声。
            //  np432b 零探针 A/B：本轮**不挂任何探针**，只留采样。
            //  为什么要 A/B：np432 的四个计数点给出了硬结论（120f264/a66050 各只进 1 次、a660c0 与 a671cc 各 0 次），
            //  但同轮的采样显示主线程 100% CPU 停在同一小段（0xa6608c/94/a8，134/149），而这段里没有任何分支指令。
            //  两种解释必须用一次对照来分开：
            //    ① 真的在这些指令上打转（例如同一条写指令被反复重试）⇒ 不挂探针也照样停在这儿；
            //    ② 是 brk 探针自己闹的（SIGTRAP 每次都触发系统崩溃 dump ⇒ 线程被拖住/被杀）⇒ 去掉探针后 pc 就会走开、且不再被杀。
            //  另外本轮能跑满：np432 那轮第一个探针命中就触发 signo(5)（si_code=1），系统 15 秒后按 Cpp Crash 杀掉，
            //  所以"稳态"从来没被观察到过；零探针这一轮是第一次有机会看到 30 秒以后的样子。
            //  np432e：采样提到 2ms（500Hz）+ DM_HIST=1（机内聚合，每 5 秒只打前 8 名）。
            //  100Hz 那两轮只拿到 150 个点、永远只命中 0xa66094 一条，分辨不出"一条指令被打转"还是"一小段被打转"。
            //  判读：直方图若还是单点 90%+ ⇒ 真的是一条指令在原地（那是内核/硬件层面在拖，不是客体循环）；
            //        若散在 0xa6608c..0xa660b8 之间 ⇒ 是一条 5~7 条指令的小循环，下一步就照这条循环去查条件。
            //  np434：查"谁把 NULL 传进来"。离线已定：名字是 `/Agl/agl_resource.%s.%s`、
            //  `/GSys/gsys_resource.%s.%s`、`/KSys/U-King.%s.%s`（%s = 语言/区域），拼好后交 `0xdb8a78` 查，
            //  查不到返回 0 ⇒ 记录 +0x20 为 NULL ⇒ ShaderHolder 构建时 `ldr x8,[x21]` 崩。
            //  两个探针（都避开 PC 相对指令：`bl 0xdb8a78` 与 `adrp` 不能挂）：
            //    · main:db8a78@0 —— 工厂入口，x1 就是**拼好的实际名字**（probe_str 会把它按字符串打出来）
            //    · main:db89a4@0 —— 调用返回后（`stur x0,[x29,#-0x80]`），x0 就是查询结果
            //  判读：名字里语言/区域为空 ⇒ 语言码没写进去（补丁/服务层的问题）；
            //        名字正常而返回 0 ⇒ romfs 里没有这份资源（缺素材），或查询表本身没建好。
            //  np434c：`@s8` = 把 [x0+8] 当 char* 打出来。名字对象是"vtable + 数据指针"布局，
            //  数据指针在 +8、内联缓冲在 +0x14；这样能拿到**完整名字**（尤其扩展名 .sarc / .ssarc）。
            //  np442：给 nnMain 铺路标，看实际执行到哪些点。**全部避开 bl/blr 站点**——
            //  np442 实测：把探针挂在 `bl 0x2128360`（oe::Initialize）上，HLE 桩拿到的 x30 变成跳板地址，
            //  寄存器随即全乱、pc 跑到宿主区（0x5a40ecfb10）当场崩。所以每个路标都往后挪一条：
            //    e53868  oe::Initialize 之后（mov w0,wzr）
            //    e553d8  "当前上下文"构造前的 malloc 之后（mov x24,x0）
            //    e5551c  发布"当前上下文"到全局 0x2cbf6e0（str x24,[x25,#0x6e0]）
            //    e562f8  崩的那次虚调用 `blr x8` 之后（ldr x20,[x8,#0xf90]）⇒ 命中说明 vcall 返回了
            //  np447 ★ 关键：1.6.0 要求 NVN API 主版本 = 53（1.0.0 是 52，见 hle_nvn.c np262 注释）。
            //  我们的缺省是 52 ⇒ nnMain 的初始化检查失败 ⇒ 跳过上下文构造 ⇒ 后面崩。用现成的开关覆盖即可。
            //  两个路标：e55318（三条检查全过之后）与 e5551c（发布当前上下文）。
            //  np447 验证通过：0xe55318 与 0xe5551c 都命中、无 FAULT ⇒ 补上 NVN API 主版本 53 后
            //  nnMain 的初始化检查通过、"当前上下文"发布成功。**探针全部撤掉**（brk 会被系统记成崩溃并查杀）。
            //  np450：memcpy 的 dst 是「经 GOT 的导入调用」的返回值（=0）。两针夹住那次调用：
            //    e 13ca2dc  csel：此时 x8=[x23]、x9=x8+x25、x25=偏移 ⇒ 看容器是否为 0
            //    e 13ca2e4  调用返回后：x0=返回值（0？）、x28 指向源指针槽
            //  np464 教训：0x1cfb704 是 `b`（PC 相对分支），挂探针必飞（实测 pc 跑到 OHOS::GetDirFiles）。
            //  改挂在调用点前的 `mov x1,x20`（非 PC 相对），此时 x0 = 那个对象。
            //  np477 修正：真正的分发点在 0x126dae8（br x2），前面那条是 mov。两针改挂 0x126dad0（查 vtable 前）与 0x126dae8 前一条 mov x1（非 PC 相对）。
            //  np477 判据修正：125cd50 是 SubmitJob（被 54 次调用=job 在投递）；摘除 0 次的真伪要看计数器本身。
            //  两针：0x1250a38（读计数的 ldr，@h0 看 x25 基址附近）计数模式 + 0x1250ab0（跳出等待的点）——
            //  若跳出点命中过，说明等到了；没命中就是真没等到。
            //  np481：探针收干净（0x1250a38/0x1250ab0 那对已使命完成）。保留 MQ 与 NVN 日志。
            //  np482：看 54 个 SubmitJob 的任务条目内容（x2 = 条目，@h0 转储头部）+ 首次调用站参数
            //  np482 验证假设：SubmitJob 的 x19（TLS 取的本地队列对象）在主线程和工作线程上各长什么样。
            //  两针：SubmitJob 入口 @h0（x0 侧）+ 0x125d018（blr [vtable+0xd0] 前的 mov x0,x19）@h38（x19+0x38 队列字段）
            //  np483 收口：工作线程分发点计数（126dad0=取 vtable 前，126daa8=拿到任务后）+ 主线程通知真身（125d014@h0 = vtable 指针）
            //  np485 定胜负：主循环 TryReceive(1cfd518) vs 收到消息分支(1cfd530)；再抓主循环队列对象 x21
            //  np486：对比两个队列。125d014@h0 = job 通知的队列指针（x0+0 链表头）；1cfd514@h0 = 主循环队列（x1）。
            //  只挂这两个 @h，避开 1cfd530（上轮它引发洞区野跳）。
            //  np487：工作线程执行路径两针——收到消息后调执行体之前(126db68) / 执行返回之后(126db78)
            //  np489：RootTask::calc（1.0.0=0x726838，阶段字节 [this+0x200]）。1.6.0 的对应函数引用日志名 0x261997f 于 0xe468d8。
            //  在它的 calc 入口挂 @c，再看有没有跑、跑在哪个阶段。
            //  np490：RootTask::calc = vtable[22] = 0x1254f60。挂 @h0 看阶段字节（1.0.0 在 [this+0x200]）与命中数。
            //  np491：RootTask 对象本身——@h0 看虚表是否写好、@h200 看阶段字节。构造器命中时 x0=对象。
            //  np492：挂在构造器尾声（e46d70，恢复寄存器处——x19/x20 仍持有 this 相关值）看构造完成的对象。
            //  另一路：既然 calc 不被调，先确认 RootTask 有没有进入 sead 任务树——挂构造器尾声读 [x0+0]（虚表）。
            //  np493：兄弟槽对照——vtable[21]=0xdecc00（@c）与 calc=1254f60（@c）。21 被调而 22 不被调 ⇒ 调度器只跑部分任务。
            //  np494：主循环里同构于 1.0.0 建任务工厂的调用（1835b78 blr [vtable+0x28]）。@h0 看对象、@c 计数。
            //  np495：撤 1835b78 探针（=0 证明那条链没跑到）。主线程 SleepThread 现场改周期打点，看真实主循环。
            //  np496：生命周期事件对象本体——@h0 读 [0x2c7fb38] 指向的队列对象头部（1cf5580 读的第一字节=事件计数）。
            //  @c 挂 1cf5580 入口统计重试频率。
            //  np497：撤探针——改看接管日志与主循环是否放出。
            //  np504：1cf2fb0 入口 @h0——看它读的三个半字候选是什么值（期望 0x2001）
            //  np505：0x995a3c blr 前——@h0 看对象、@s 看名字参数（x1=0x2c8ab08 附近的名字串）
            //  np507：置位函数 9955c0 入口 @c（看本轮是否被调）+ 第一个虚调用 99562c @c/@h0
            //  np520 纠错：上面两条 brk 探针（9955c0/99562c）撤掉——已证实 1cf5580 是 NintendoWare Atk 音频库初始化，
            //  "5ms 等待/tag 0x2001"属音频（snd），不是 nn::oe 生命周期。
            //  改走"不制造崩溃"的仪器：DM_SAMPLE=2（2ms 采主线程 pc）+ DM_HIST=1（机内聚合，每 5 秒只打前 8 名）。
            //  为什么不用 brk：高频点每命中一次都让系统记一条 Cpp Crash 并 fork processdump，约 15 秒后查杀进程。
            //  判据：直方图里有没有 main+0x1614a90 一带（每帧系统 tick）、有没有 main+0x1254f60（RootTask::calc）。
            //  np521（C1 取证）：1cf2fb0 在 node+0x14 / +0x20 / +0x2c 三处找半字 0x2001，
            //  三处都不等就把 x8 置 0，然后 `ldr w0,[x8,#8]` 崩在 addr=0x8。
            //  这两条探针只做一件事：**把节点字节打出来**，先知道差多少，再决定改哪里。
            //    main:1cf2fb0@h14 —— 函数入口（允许挂），x0=node，转储 node+0x14 的 16 字节（覆盖 0x14 与 0x20 两个标签）
            //    main:1cf2fc4@h2c —— 第一个标签不匹配才会走到这条 mov（x0 仍是 node），转储 node+0x2c
            //  同一站点不能挂两次，所以两个候选偏移拆到两个站点。
            //  np522（E1+E2）：节点全零，且全映像没有任何代码写 u16 0x2001 ⇒ 标签来自数据。
            //  于是问两个"有没有/给没给"的问题（都用 @c：头 3 次给全量含 x1 的 16 字节，之后每秒一行，不刷屏）：
            //    main:1cf5580@c —— Atk 初始化入口（函数入口是合法挂点）。判读：
            //        计数 0 ⇒ 根本没被调；x1=0 ⇒ 描述符是空；x1 非 0 ⇒ 拿到描述符了。
            //        注意：它开头是 `if (标志字节≠0) 直接 return 1`（不看参数），
            //        ⇒ "每秒都在被调"本身也可能只是这个短路，要看首 3 次的 x1。
            //    main:12f2da8@c —— 虚调用（0x12f2d94 blr x8）刚返回后的第一条指令，x0 = 描述符指针。
            //        判读：x0=0 ⇒ 引擎没给描述符（那就得先修这条供给，而不是修 Atk 初始化）。
            //  本轮撤掉 DM_SAMPLE/DM_HIST：进程 2.5 秒就死，凑不满直方图第一个 5 秒窗口，留着只是干扰。
            //  np531：验证"掩码键"是否稳定——去掉丢帧开关，让它照常 fatal，看两轮日志里的 masked 键是否相同。
            // np554：卡点从"崩"变成"命令流断流"（最后 OPTRACE=19，进程 17% CPU 存活）。
            //   现在要判**在等还是在转**：DM_SAMPLE=2（2 ms 采主线程 pc）+ DM_HIST=1（机内聚合，
            //   每 5 秒只打前 8 名）。DM_IMPORTS 关掉——它每行一条、会把日志预算吃光。
            //   TLS 已排除（slot=4 的返回值始终有效），不需要再开。
            // np554: the failure changed from a crash to a stalled command stream (last OPTRACE=19, process alive at
            //   17% CPU). Now decide waiting vs spinning: DM_SAMPLE=2 (sample the main thread's pc every 2 ms) +
            //   DM_HIST=1 (aggregate in-process, print the top 8 every 5 s). DM_IMPORTS is off: one line per call eats
            //   the whole log budget. TLS is already ruled out (slot=4 always returns a valid pointer).
            let envs = 'DM_NVN_LOG=1 DM_FS=1 DM_WATCH_STALL=8 DM_NVN_API_MAJOR=53 DM_DIAG_TICK=1 DM_SHADER_DUMP=1 DM_DRAW_TRACE=8 DM_CV_PROBE=1';
            // romfs 双路（d1 瘦包）：
            //   A. HAP 里带 romfs/（fat 包，首次装机用）→ 照旧**每次点都重新解出**。
            //      ⚠ 不能用 pathExists 判"解没解过"：本机 fs.listFileSync 对不存在的路径也不抛
            //      异常（np279）⇒ 恒为 true ⇒ romfs 一次都没解出 ⇒ rom:/ 全打不开。
            //   B. HAP 里不带（瘦包，日常迭代用，装机 361→57MB）→ 用**原生** dirFileCount 探测
            //      沙箱里上一版 fat 包解出的 romfs 还在不在（install -r 保留应用数据；
            //      存在性判断必须原生做，同 np279 的坑）。≥1000 个文件才算在（romfs 实测 8190）。
            if (rawList(rm, 'romfs').length > 0) {
              const rdir = ctx.filesDir + '/romfs';
              this.status = '正在从安装包解出内容（romfs）…';
              this.writeFails = 0;
              const rn = this.extractRaw(ctx, rm, 'romfs', rdir);
              trace(ctx, 'extract romfs/: ' + rn + ' 个文件, 写入失败 ' + this.writeFails + ' → ' + rdir);
              envs += ' DM_ROMFS=' + rdir;
            } else {
              const rdir = ctx.filesDir + '/romfs';
              const have: number = dms2.dirFileCount(rdir);
              trace(ctx, '瘦包：原生探测 romfs → ' + rdir + ' 文件数=' + have);
              if (have >= 1000) {
                envs += ' DM_ROMFS=' + rdir;
              } else {
                trace(ctx, '【提示】HAP 无 romfs 且沙箱里没有（' + have + '）⇒ 本轮不带内容；'
                  + '首次装机请用 fat 包（把 romfs 拷回 resources/rawfile/romfs 再构建）');
              }
            }
            envs += ' DM_TEMP=' + ctx.filesDir + '/temp';
            // np523：存档目录。宿主 `nn::fs::MountSaveData` 的接管**硬要求** DM_SAVE（未设就 die——
            //  那是 D2R 为了防止"静默落到没有角色的目录、把选角测量搞失真"而定的规矩）。
            //  BotW 在启动早期就会挂存档，所以必须给一个可写目录；mount_ex 会自己 mkdir 并 clonefile 出工作目录。
            //  真机实测（2.2KB 的 DummySound.bfsar 补进 romfs 之后）崩点从音频一路推到这一句：
            //  `FATAL 未设 DM_SAVE：请指向一份存档目录的可写副本`。
            envs += ' DM_SAVE=' + ctx.filesDir + '/save';
            // np345d 写入自测：定位"新建目录的 writeFile 静默失败"的根因。
            // 三级深度各写一个 4 字节文件，writeFile 现在失败返回 -errno（原生 fopen）；
            // 结果与 dirFileCount 一并 trace——一条自测分清"顶层/新建目录/嵌套新建"哪级失败、什么 errno。
            {
              const wt = ctx.filesDir + '/wtest';
              const pd = new Uint8Array([0x5a, 0xa5, 0x01, 0x7f]);
              const md: number = dms2.makeDir(wt);                       // mkdir 本身的 errno（np345d）
              const r1: number = dms2.writeFile(wt + '/a.txt', pd.buffer);
              const r2: number = dms2.writeFile(wt + '/sub1/b.txt', pd.buffer);
              const r3: number = dms2.writeFile(wt + '/sub1/sub2/c.txt', pd.buffer);
              const cnt: number = dms2.dirFileCount(wt);
              trace(ctx, 'wtest 自测: makeDir=' + md + ' a.txt=' + r1 + ' sub1/b.txt=' + r2
                + ' sub1/sub2/c.txt=' + r3 + ' dirFileCount=' + cnt);
            }
            dms2.start(dir, log, marker, envs);
            trace(ctx, 'start returned');
          } catch (e) {
            this.status = 'start 抛异常：' + e;
            trace(getContext(this) as common.UIAbilityContext, '❌ 抛异常：' + e);
          }
        })

      Text(this.status)
        .fontSize(12)
        .width('100%')
        .padding(8)
        .backgroundColor('#1b2838')
        .fontColor('#d6e4f0')

      // ---- 手柄：常驻焦点收键事件；方向键不外溢（官方对"只用方向键做游戏操作"的建议）----
      Button('手柄焦点区（连上手柄后按几下）')
        .defaultFocus(true)
        .width('100%')
        .focusScopeId('pad', true, false)
        .onKeyEvent((event: KeyEvent) => {
          if (!event) return;
          const bit = KEYMAP.get(event.keyCode);
          if (bit === undefined) return;
          if (event.type === KEY_DOWN) this.buttons |= bit;
          else if (event.type === KEY_UP) this.buttons &= ~bit;
          this.pushPad('键');
        })
        .onFocusAxisEvent((event: FocusAxisEvent) => {
          if (!event || !event.axisMap) return;
          // 逐个轴取：左摇杆 ABS_X/ABS_Y，右摇杆 ABS_Z/ABS_RZ（官方 guide 的对照写法）
          event.axisMap.forEach((value: number, axis: number) => {
            switch (axis) {
              case ABS_X: this.lx = Math.round(value * AXIS_FULL); break;
              case ABS_Y: this.ly = Math.round(value * AXIS_FULL); break;
              case ABS_Z: this.rx = Math.round(value * AXIS_FULL); break;
              case ABS_RZ: this.ry = Math.round(value * AXIS_FULL); break;
              default: break;
            }
          });
          this.pushPad('轴');
        })

      Text(this.padStatus)
        .fontSize(12)
        .width('100%')
        .padding(8)
        .backgroundColor('#22331f')
        .fontColor('#d8f0d6')
    }
    .padding(12)
    .width('100%')
    .height('100%')
  }
}
