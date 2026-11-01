#!/usr/bin/env python3
"""从真机日志里把"缺包现场"（ctlhex + datahex）拼回一条 dmsh_vk.py 能吃的记录。

为什么要它：有些着色器**不在任何文件里**（疑似 agl 运行时编译的变体），所以只能拿运行时
dump 到的 control + code 本身去翻译，产出以**运行时键**命名的 .xgs。

记录格式（与 shader_split.py 输出一致）：
    'BTCDBTCD' + 0x38 字节头（+0x28 控制段长度、+0x2c 代码段长度）+ 代码区 + 控制段(0x880)
map 行：package <TAB> index <TAB> stage <TAB> key <TAB> source

用法:
  python make_synthetic_pkg.py <日志> <输出目录> [--code-off 0x180] [--stage vertex]
输出:
  <输出目录>/pkg/<key>.bin、<输出目录>/shader_map.tsv
"""
import os
import re
import struct
import sys

CODE_MAGICS = (b'\x78\x56\x34\x12', b'\x79\x56\x34\x12')   # 图形 / 计算


def parse(log):
    txt = open(log, encoding='utf-8', errors='replace').read()
    key = re.search(r'shaderkey prog=([0-9a-f]{16})', txt)
    ctl = {}
    for m in re.finditer(r'ctlhex [0-9a-f]{16} \+([0-9a-f]{3}) ([0-9a-f]+)', txt):
        ctl[int(m.group(1), 16)] = bytes.fromhex(m.group(2))
    dat = {}
    # np625：偏移放宽到 1..8 位——ncode 到 0x8000 后偏移是 4 位（如 `+1a00`），旧正则只认 3 位，
    #   大着色器 0x1800 之后的段全部丢失（这就是"代码长度不足"的另一半根因）。
    # np625: allow 1..8 hex digits for the offset -- with ncode up to 0x8000 the offsets are 4 digits
    #   (e.g. `+1a00`); the old regex only took 3 and silently dropped every chunk past 0x1000.
    for m in re.finditer(r'datahex [0-9a-f]{16} \+([0-9a-f]{1,8}) ([0-9a-f]+)', txt):
        dat[int(m.group(1), 16)] = bytes.fromhex(m.group(2))
    c = bytearray(0x880)
    for o, b in ctl.items():
        c[o:o + len(b)] = b
    # np625：代码段大小不再按"最大偏移的那一段"算——大着色器（ncode 到 0x8000）的 dump 段是
    #   0x180 一段共几十段，`max(dat)+len(dat[max(dat)])` 会把缓冲切短，后面的段全被丢掉。
    #   直接取"最大偏移 + 该段长度"，与所有段的并集一致。
    # np625: do not size the code buffer from the single largest-offset chunk -- large shaders (ncode up to
    #   0x8000) dump as dozens of 0x180 chunks, and `max(dat)+len(dat[max(dat)])` truncated the buffer and
    #   dropped everything past it. Size from the union: max offset + its chunk length.
    d = bytearray((max(dat) + len(dat[max(dat)])) if dat else 0)
    for o, b in dat.items():
        d[o:o + len(b)] = b
    return (key.group(1) if key else None), bytes(c), bytes(d)


def real_key(control, code):
    """np619：**按引擎的口径算键**：sha256(control[0:nctl] || code[0:ncode])[:16]。
    日志里的 `shaderkey prog=` 是程序键、`ctl=` 是只哈希控制段的诊断值，两者都**不是** .xgs 的名字；
    引擎（xbg_shader_key.c）与 shader_split.py 用的是"控制段 + 代码段"一起哈希。
    这段照抄规则：nctl = u32(control+0x20)，仅当 0x760<=nctl<0x880 才用它，否则 0x880；
    ncode = u32(control+0x704)，要求 0 < ncode < 1<<24。
    """
    import hashlib
    nctl = struct.unpack_from('<I', control, 0x20)[0]
    if not (0x760 <= nctl < 0x880): nctl = 0x880
    ncode = struct.unpack_from('<I', control, 0x704)[0] if len(control) >= 0x708 else 0
    if not (0 < ncode < (1 << 24)) or len(code) < ncode: ncode = len(code)
    return hashlib.sha256(control[:nctl] + code[:ncode]).hexdigest()[:16], nctl, ncode


def main():
    log, out = sys.argv[1], sys.argv[2]
    code_off = 0x180
    stage = 'vertex'
    if '--code-off' in sys.argv:
        code_off = int(sys.argv[sys.argv.index('--code-off') + 1], 0)
    if '--stage' in sys.argv:
        stage = sys.argv[sys.argv.index('--stage') + 1]
    key, ctl, dat = parse(log)
    if not key or len(ctl) != 0x880 or not dat:
        print('日志里没找到完整的 ctlhex/datahex'); return 1
    own = struct.unpack_from('<I', ctl, 0x20)[0]
    nctl = own if 0x760 <= own < 0x880 else 0x880
    ncode = struct.unpack_from('<I', ctl, 0x704)[0]
    if code_off == 0:                      # 自己找代码魔数
        for k in range(0, min(len(dat) - 4, 0x400), 4):
            if dat[k:k + 4] in CODE_MAGICS:
                code_off = k; break
    code = dat[code_off:code_off + ncode]
    print(f'键 {key}  own={own:#x} nctl={nctl:#x} ncode={ncode:#x} 代码偏移={code_off:#x} 实得 {len(code)} 字节')
    if len(code) != ncode:
        print('代码长度不足（日志窗口不够）'); return 1
    # np619：日志里的键只是"程序键"，.xgs 的名字要按引擎口径重算（控制段+代码段一起哈希）
    log_key, key = key, real_key(ctl, code)[0]
    if log_key != key: print(f'  键按引擎口径重算：日志 {log_key} → {key}')
    hdr = bytearray(0x38)
    hdr[0:8] = b'BTCDBTCD'
    struct.pack_into('<I', hdr, 0x28, nctl)     # 控制段长度
    struct.pack_into('<I', hdr, 0x2c, ncode)    # 代码段长度
    rec = bytes(hdr) + code + ctl
    pkg = os.path.join(out, 'pkg')
    os.makedirs(pkg, exist_ok=True)
    _st = {'vertex': 'vs', 'pixel': 'ps', 'fragment': 'ps', 'compute': 'cs', 'vs': 'vs', 'ps': 'ps', 'cs': 'cs'}.get(stage, stage)
    name = f'{_st}_{key}.bin'   # np619：名字前两字符就是阶段（dmsh_vk 的 stage 兜底读它）
    open(os.path.join(pkg, name), 'wb').write(rec)
    with open(os.path.join(out, 'shader_map.tsv'), 'w', encoding='utf-8') as f:
        f.write('# kind: synthetic  (runtime dump → dmsh_vk)\n')
        f.write('package\tindex\tstage\tdm_ctl_sha16\tsource\n')
        f.write(f'{name}\t0\t{_st}\t{key}\truntime-dump\n')
    # np543 ★ 喂翻译器的切片要**再跳 0x30**（与 dmsh_vk.py 的 `code = code[0x30:]` 同一口径）。
    #   不跳的后果（np537–np542 实测）：翻译器把 0x30 字节程序头当指令读，顶点着色器被翻成
    #   Fragment（205 words、只剩 support_buffer），塞进 stage=0 的包里就是"顶点槽里放片元模块"。
    #   这里顺手把切片和 job.json 一起产出，避免下一次又手工拼错。
    # np543: the slice handed to the translator must skip another 0x30 bytes, the same convention as
    #   dmsh_vk.py's `code = code[0x30:]`. Without it the translator reads the 0x30-byte program header as
    #   instructions and a vertex shader comes out as Fragment (205 words, only support_buffer), which then
    #   sits in a stage=0 bundle. Emit the slice and the job here so it cannot be hand-mistyped again.
    job_stage = {'vertex': 'vs', 'pixel': 'ps', 'fragment': 'ps', 'compute': 'cs', 'vs': 'vs', 'ps': 'ps', 'cs': 'cs'}.get(stage, stage)
    xcode = code[0x30:]
    xpath = os.path.join(out, f'{key}.xcode.bin')
    open(xpath, 'wb').write(xcode)
    import json
    job = [{'key': key, 'stage': job_stage, 'code': os.path.abspath(xpath), 'tex': {}, 'img': {}, 'attr': {},
            'fout': {}, 'c1': '', 'debug': True, 'reduced': True}]
    jpath = os.path.join(out, f'job_{key}.json')
    json.dump(job, open(jpath, 'w'), ensure_ascii=False)
    print(f'已写 {os.path.join(pkg, name)}（{len(rec)} 字节）与 shader_map.tsv')
    print(f'翻译：① 切片 {xpath}（{len(xcode)} 字节 = code-0x30）')
    print(f'      ② job {jpath}（stage={job_stage}）')
    print(f'      ③ dotnet <dmsh_xlate.dll> {jpath} <工作目录>')
    print(f'      ④ 打包前跑 tools/check_shader_stage.py 校验"执行模型 = stage"')
    return 0


if __name__ == '__main__':
    sys.exit(main())
