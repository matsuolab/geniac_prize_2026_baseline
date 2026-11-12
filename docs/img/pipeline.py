"""Draw docs/img/pipeline-{light,dark}.svg (the README pipeline figure): python docs/img/pipeline.py"""

from itertools import pairwise
from pathlib import Path

OUT = Path(__file__).parent
T = {
    "light": dict(
        text="#1f2328", muted="#59636e", line="#59636e", panel="#f6f8fa", pstroke="#d0d7de",
        model=("#ddf4ff", "#54aeff"), sim=("#ffffff", "#d0d7de"), data=("#fbefff", "#c297ff"),
        train=("#fff8c5", "#d4a72c"), out=("#dafbe1", "#4ac26b"),
    ),
    "dark": dict(
        text="#e6edf3", muted="#9198a1", line="#9198a1", panel="#151b23", pstroke="#3d444d",
        model=("#0c2d6b", "#4493f8"), sim=("#0d1117", "#3d444d"), data=("#271052", "#ab7df8"),
        train=("#3b2300", "#bb8009"), out=("#04260f", "#2ea043"),
    ),
}  # fmt: skip
W, H = 960, 500
FONT = "-apple-system, 'Segoe UI', 'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Noto Sans JP', sans-serif"


def svg(c: dict) -> str:
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c["line"]}"/></marker></defs>',
    ]

    def txt(x, y, s, size=12.5, color="muted", anchor="middle", bold=False):
        weight = ' font-weight="600"' if bold else ""
        o.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}"{weight} fill="{c[color]}">{s}</text>'
        )

    def box(x, y, w, h, kind, t1, t2=None):
        f, s = c[kind]
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{f}" stroke="{s}" stroke-width="1.5"/>')
        cy = y + h / 2
        if t2:
            txt(x + w / 2, cy - 6, t1, 15, "text", bold=True)
            txt(x + w / 2, cy + 15, t2)
        else:
            txt(x + w / 2, cy + 5, t1, 15, "text", bold=True)

    def arrow(d, label=None, lx=0, ly=0):
        o.append(f'<path d="{d}" fill="none" stroke="{c["line"]}" stroke-width="1.6" marker-end="url(#a)"/>')
        if label:
            txt(lx, ly, label)

    def panel(x, y, w, h, title):
        fill, stroke = c["panel"], c["pstroke"]
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}"/>')
        txt(x + 18, y + 26, title, 13.5, anchor="start", bold=True)

    # top: one trial
    panel(10, 10, 940, 200, "1 回の trial(1 task × 1 seed)")
    box(40, 80, 230, 76, "model", "Qwen3.8-27B", "+ LoRA adapter(学習するのはここ)")
    box(480, 80, 250, 76, "sim", "CaP-X が code を実行", "LIBERO-PRO・Robosuite の simulator")
    box(770, 80, 160, 76, "sim", "成功判定", "simulator の判定")
    arrow("M270,104 L478,104", "Python code", 374, 96)
    arrow("M480,134 L272,134", "stdout・stderr・観測の説明", 374, 152)
    arrow("M155,80 C155,50 850,50 850,78", "FINISH(終える)", 500, 48)
    txt(125, 186, "直す時は REGENERATE + 新しい code", anchor="start")
    # bottom: pipeline
    panel(10, 240, 940, 250, "ベースラインの流れ(番号は examples/ の script)")
    y, w, h = 300, 140, 70
    xs = [30, 215, 400, 585, 770]
    box(xs[0], y, w, h, "data", "0. 収集", "trial を保存")
    box(xs[1], y, w, h, "train", "2. SFT", "成功した trial")
    box(xs[2], y, w, h, "train", "4. DMPO", "SFT の adapter から")
    box(xs[3], y, w, h, "out", "1・5. 評価", "学習の前後、同じ seed")
    box(xs[4], y, w + 10, h, "out", "提出", "Hugging Face に上げる")
    box(xs[1] + 92, 400, 140, 62, "data", "3. pair", "成功と失敗の trial")
    arrow(f"M{xs[0] + w / 2},210 L{xs[0] + w / 2},{y - 2}")
    for a, b in pairwise(xs):
        arrow(f"M{a + w},{y + h / 2} L{b - 2},{y + h / 2}")
    arrow(f"M{xs[0] + w / 2},{y + h} L{xs[0] + w / 2},431 L{xs[1] + 90},431")
    arrow(f"M{xs[1] + 232},431 L{xs[2] + w / 2},431 L{xs[2] + w / 2},{y + h + 2}")
    txt(xs[4] + 75, y + h + 28, "評価は運営チームが行う")
    o.append("</svg>")
    return "\n".join(o)


for k, c in T.items():
    (OUT / f"pipeline-{k}.svg").write_text(svg(c))
