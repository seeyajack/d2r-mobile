#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""np701 判死证据的离线分诊。输入 gfx-death.txt / gfx-lastframe.txt（deathguard 落盘，沙箱 files/ 下）。

用法:
    python tools/death_triage.py <death.txt|lastframe.txt> [选项]

做什么（对应总纲待办 2 的离线半程）:
  1. death.txt: 判死线程与站点（卡在哪个引擎调用）、提交摘要环尾、管线计数三行（np700 口径）。
  2. lastframe.txt: pass 形状对照上限（Vulkan 规范最小保证 + 可选 --max-w/h/layers/att 覆盖）、
     建管线耗时榜（>1s 的编译是 spirv-dis 无界循环/驱动编译器嫌疑）、着色器键清单（喂 spvdis_min.py）。
  3. 交叉: 尾条提交摘要的 passes/draws 与清单行数对账（差得远 = 清单被环上限截断或串帧）。

只依赖标准库。
"""
import sys, argparse, re
from collections import Counter

SPEC_MIN = {"w": 4096, "h": 4096, "att": 8}   # Vulkan 规范最小保证：maxFramebufferWidth/Height、maxColorAttachment


def parse_lastframe(path):
    frames, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"\[(sealed|current)\] serial=(\d+) passes=(\d+) shaders=(\d+) builds=(\d+) builds_overflow=(\d+)", line)
        if m:
            cur = {"tag": m.group(1), "serial": int(m.group(2)), "pass_n": int(m.group(3)),
                   "shader_n": int(m.group(4)), "build_n": int(m.group(5)), "overflow": int(m.group(6)),
                   "passes": [], "shaders": [], "builds": []}
            frames[cur["tag"]] = cur
            continue
        if cur is None:
            continue
        m = re.match(r"P \w+ (\d+) w=(\d+) h=(\d+) att=(\d+) samples=(\d+) serial0=(\d+) depth=(\d+) draws=(\d+)", line)
        if m:
            cur["passes"].append(dict(idx=int(m.group(1)), w=int(m.group(2)), h=int(m.group(3)),
                                      att=int(m.group(4)), samples=int(m.group(5)),
                                      serial0=int(m.group(6)), depth=int(m.group(7)), draws=int(m.group(8))))
            continue
        m = re.match(r"S \w+ (\S+) stage=(\d+) n=(\d+)", line)
        if m:
            cur["shaders"].append((m.group(1), int(m.group(2)), int(m.group(3))))
            continue
        m = re.match(r"B \w+ (\S+) ms=(\d+)", line)
        if m:
            cur["builds"].append((m.group(1), int(m.group(2))))
    return frames


def parse_death(path):
    death = {"threads": [], "digest": [], "counts": None, "why": None, "lost": None, "end_cause": None}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = re.search(r"why=(线程 .*)$", line)
        if m and not death["why"]:
            death["why"] = m.group(1)
        m = re.match(r"T slot=(\d+) tid=(\d+) site=(\d+)\(([^)]*)\) age_ms=(\d+) ctx=([0-9a-f]+)", line)
        if m:
            death["threads"].append(dict(slot=int(m.group(1)), tid=int(m.group(2)), site=int(m.group(3)),
                                         name=m.group(4), age_ms=int(m.group(5)), ctx=m.group(6)))
            continue
        m = re.match(r"D serial=(\d+) draws=(\d+) passes=(\d+) disp=(\d+)", line)
        if m:
            death["digest"].append(dict(serial=int(m.group(1)), draws=int(m.group(2)),
                                        passes=int(m.group(3)), disp=int(m.group(4))))
            continue
        m = re.search(r"lost armed=(\w+) ago_ms=(-?\d+) cooldown_left_ms=(\d+)", line)
        if m:
            death["lost"] = m.groups()
            continue
        m = re.search(r"pass_end_cause:(.*)", line)
        if m:
            death["end_cause"] = m.group(1).strip()
            continue
        m = re.search(r"pipeline_counts hits=(\d+) misses=(\d+) bypass=(\d+) evictions=(\d+)", line)
        if m:
            death["counts"] = tuple(int(x) for x in m.groups())
    return death


def main():
    ap = argparse.ArgumentParser(description="np701 判死证据分诊")
    ap.add_argument("file")
    ap.add_argument("--max-w", type=int, default=SPEC_MIN["w"], help="设备 maxFramebufferWidth")
    ap.add_argument("--max-h", type=int, default=SPEC_MIN["h"], help="设备 maxFramebufferHeight")
    ap.add_argument("--max-att", type=int, default=SPEC_MIN["att"], help="设备 maxColorAttachments")
    ap.add_argument("--slow-build-ms", type=int, default=1000, help="建管线耗时报警阈值（默认 1000ms）")
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8", errors="replace").read()
    if "DM-DEATH" in text:
        d = parse_death(a.file)
        print(f"== 判死现场 {a.file} ==")
        if d["why"]:
            print(f"判死: {d['why']}")
        for t in d["threads"]:
            flag = "  <== 卡点" if t["age_ms"] > 15000 else ""
            print(f"  线程 tid={t['tid']} {t['name']} 卡 {t['age_ms']/1000:.1f}s ctx={t['ctx']}{flag}")
        if d["digest"]:
            tail = d["digest"][0]
            print(f"提交摘要环尾: serial={tail['serial']} draws={tail['draws']} passes={tail['passes']} disp={tail['disp']}（共 {len(d['digest'])} 条）")
        if d["lost"]:
            print(f"lost: armed={d['lost'][0]} ago={int(d['lost'][1])/1000:.1f}s cooldown_left={d['lost'][2]}ms")
        if d["counts"]:
            h, mi, by, ev = d["counts"]
            print(f"管线计数: hits={h} misses={mi} bypass={by} evictions={ev}" +
                  ("  <== bypass 风暴（np700 型回归）" if by > 1000 and mi == 0 else ""))
        print("（pass/shader 清单在同目录 gfx-lastframe.txt，用 death_triage.py 单独跑）")
        return
    if "DM-FRAME-MANIFEST" not in text:
        sys.exit("不是 deathguard 产物（缺 DM-FRAME-MANIFEST / DM-DEATH 头）")
    frames = parse_lastframe(a.file)
    for tag in ("current", "sealed"):
        f = frames.get(tag)
        if not f:
            print(f"[{tag}] 缺")
            continue
        print(f"== [{tag}] serial={f['serial']} 声明 passes={f['pass_n']} shaders={f['shader_n']} builds={f['build_n']} overflow={f['overflow']} ==")
        ps = f["passes"]
        print(f"  实际 pass 行 {len(ps)}" + ("（截断于环上限 4096）" if len(ps) >= 4096 else ""))
        bad = [p for p in ps if p["w"] > a.max_w or p["h"] > a.max_h or p["att"] > a.max_att]
        if bad:
            for p in bad[:8]:
                print(f"  ⚠ pass#{p['idx']} {p['w']}x{p['h']} att={p['att']} 超出设备上限（>{a.max_w}x>{a.max_h} att>{a.max_att}）")
        else:
            print(f"  pass 形状: 全部在 {a.max_w}x{a.max_h} att<={a.max_att} 内")
        if ps:
            dw = Counter((p["w"], p["h"], p["att"]) for p in ps)
            top = ", ".join(f"{w}x{h}a{at}x{n}" for (w, h, at), n in dw.most_common(6))
            print(f"  形状 Top6: {top}")
        slow = sorted([b for b in f["builds"] if b[1] >= a.slow_build_ms], key=lambda b: -b[1])
        if f["build_n"] and not f["builds"]:
            print("  ⚠ 本帧有建管线但环已截断")
        for k, ms in slow[:8]:
            print(f"  ⚠ 建管线 {ms}ms key={k[:32]}…（编译耗时嫌疑，喂 spvdis_min.py）")
        if f["overflow"]:
            print(f"  ⚠ 建管线环溢出 {f['overflow']} 条——bypass/重建风暴的读数（np700 型）")
        if f["shaders"]:
            print(f"  着色器 {len(f['shaders'])} 个（频次前 8）: " +
                  ", ".join(f"{k[:8]}…s{st}x{n}" for k, st, n in sorted(f["shaders"], key=lambda x: -x[2])[:8]))


if __name__ == "__main__":
    main()
