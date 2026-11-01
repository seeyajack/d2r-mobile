#!/usr/bin/env python3
"""np621：真机 dump → 合成包 → 翻译 → .xgs.gz 的**一键流水线**。

用法（在 HarmonyOS NEXT Porting Project 根目录）：
  python tools/shader_replenish.py <真机日志> [包输出目录]

步骤：
  1. 按记录切日志（每个 `ctlhex <键> +000` 起一条；同一 prog 键会有多条记录，不能按 prog 合并）；
  2. 每条记录喂 make_synthetic_pkg.py --code-off 0（自动找代码魔数），键按引擎口径重算
     （sha256(control[0:nctl] ‖ code[0:ncode])[:16] —— 等于日志里的 `prog=`，**不是** `缺着色器 <x>`）；
  3. dmsh_vk.py 批量翻译（需先 set 环境：XLATE / SPIRV_OPT / SPIRV_VAL / MAIN / DMSH_MAP）；
  4. 成功的 .xgs 以 gzip 写进两份树的 rawfile/shaders_extra/（应用启动时自解）。

已出包的键自动跳过（读 batch2/batch3 的 vk 目录），所以每轮只会处理**新缺的键**。
失败分类会在末尾汇总（当前唯一卡点：动态存储图像 cbuf_slot=-1 的 handle 解不出槽）。
"""
import io, os, re, sys, json, shutil, subprocess, gzip, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = r'C://Users//Administrator//.workbuddy//binaries//python//envs//default//Scripts//python.exe'
DMTOOLS = r'C://Users//Administrator//dm-tools'
BASE = os.path.join(DMTOOLS, 'synth')
EXTRA_DSTS = [
    os.path.join(ROOT, 'vk-spike', 's2', 'app', 'entry', 'src', 'main', 'resources', 'rawfile', 'shaders_extra'),
    os.path.join(ROOT, 'd2r-mobile', 'dm', 'harmony', 'app', 'entry', 'src', 'main', 'resources', 'rawfile', 'shaders_extra'),
]

def existing_keys():
    keys = set()
    for f in glob.glob(os.path.join(BASE, 'batch*', 'vk', '*.xgs')):
        keys.add(os.path.splitext(os.path.basename(f))[0])
    return keys

def split_records(log_path):
    recs, cur, prog = [], [], None
    for l in io.open(log_path, encoding='utf-8', errors='replace'):
        m = re.search(r'shaderkey prog=([0-9a-f]{16})', l)
        if m:
            prog = m.group(1)
            continue
        if 'ctlhex' in l and '+000' in l:
            if cur:
                recs.append(cur)
            cur = [('shaderkey prog=%s' % (prog or '0')), l]
            continue
        if cur and ('ctlhex' in l or 'datahex' in l):
            cur.append(l)
    if cur:
        recs.append(cur)
    return recs

def main():
    log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'vk-spike', 's3', 'last_run.log')
    have = existing_keys()
    txt = io.open(log, encoding='utf-8', errors='replace').read()
    missing = set(re.findall(r'NVN missing shader ([0-9a-f]{16})', txt))
    todo = sorted(missing - have)
    print('缺失键 %d，已出包 %d，本轮待处理 %d' % (len(missing), len(have & missing), len(todo)))
    if not todo:
        return 0
    batch = os.path.join(BASE, 'auto_%s' % os.path.basename(log).replace('.', '_'))
    shutil.rmtree(batch, ignore_errors=True)
    os.makedirs(batch, exist_ok=True)
    recs = split_records(log)
    rows, seen, one = [], set(), os.path.join(batch, '_one')
    for r in recs:
        shutil.rmtree(one, ignore_errors=True)
        os.makedirs(one, exist_ok=True)
        p = os.path.join(one, 's.log')
        io.open(p, 'w', encoding='utf-8').write('\n'.join(r))
        subprocess.run([PY, os.path.join(ROOT, 'tools', 'make_synthetic_pkg.py'), p, one, '--code-off', '0'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
        mp = os.path.join(one, 'shader_map.tsv')
        if not os.path.exists(mp):
            continue
        line = [x for x in io.open(mp, encoding='utf-8').read().strip().splitlines() if not x.startswith('#')][-1]
        name, idx, st, key, src = line.split('\t')
        if key in seen or key not in todo:
            continue
        seen.add(key)
        shutil.copy(os.path.join(one, 'pkg', name), os.path.join(batch, name))
        rows.append(line)
    if not rows:
        print('没有新的可合成键'); return 0
    io.open(os.path.join(batch, 'shader_map.tsv'), 'w', encoding='utf-8').write(
        '# kind: synthetic\npackage\tindex\tstage\tdm_ctl_sha16\tsource\n' + '\n'.join(rows) + '\n')
    print('合成 %d 个包，开始翻译…' % len(rows))
    env = dict(os.environ,
               XLATE=os.path.join(DMTOOLS, 'xlate10.cmd'),
               SPIRV_OPT=os.path.join(DMTOOLS, 'spirv', 'bin', 'spirv-opt.exe'),
               SPIRV_VAL=os.path.join(DMTOOLS, 'val.cmd'),
               MAIN=r'F://switch-work//exefs160//main',
               DMSH_MAP=os.path.join(batch, 'shader_map.tsv'))
    tools = os.path.join(ROOT, 'd2r-mobile', 'dm', 'gfx', 'tools')
    r = subprocess.run([PY, os.path.join(tools, 'dmsh_vk.py'), batch, os.path.join(batch, 'vk')],
                       env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout.strip()[-600:] or r.stderr.strip()[-600:])
    n = 0
    for f in glob.glob(os.path.join(batch, 'vk', '*.xgs')):
        key = os.path.splitext(os.path.basename(f))[0]
        data = open(f, 'rb').read()
        for d in EXTRA_DSTS:
            os.makedirs(d, exist_ok=True)
            with gzip.open(os.path.join(d, key + '.xgs.gz'), 'wb') as g:
                g.write(data)
        n += 1
    print('已写入 %d 个 .xgs.gz 到 shaders_extra（重装/重跑后生效）' % n)
    return 0

if __name__ == '__main__':
    sys.exit(main())
