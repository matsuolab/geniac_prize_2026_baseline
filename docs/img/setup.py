"""Draw docs/img/setup-{light,dark}.svg (the installation layout figure): python docs/img/setup.py"""

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

    def arrow(d, label=None, lx=0, ly=0, anchor="middle"):
        o.append(f'<path d="{d}" fill="none" stroke="{c["line"]}" stroke-width="1.6" marker-end="url(#a)"/>')
        if label:
            txt(lx, ly, label, anchor=anchor)

    def panel(x, y, w, h, title, dashed=False, fill=None):
        fill, stroke = fill or c["panel"], c["pstroke"]
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}"{dash}/>')
        txt(x + 18, y + 26, title, 13.5, anchor="start", bold=True)

    # left: this repository, with the competition CaP-X cloned inside it
    panel(10, 10, 640, 480, "geniac_prize_2026_baseline/(このリポジトリ)")
    box(30, 50, 190, 76, "train", "geniac_baseline", "変換・学習・計画・採点")
    box(240, 50, 180, 76, "data", "configs/", "run spec と task config")
    box(440, 50, 190, 76, "out", "patches/", "setup.sh と patch(2 つ)")
    panel(30, 170, 600, 300, "geniac_prize_2026_code/(setup.sh が作る)", fill=c["sim"][0])
    box(50, 216, 260, 76, "sim", "capx/(コンペの CaP-X)", "決まった commit + patch")
    box(340, 216, 270, 76, "sim", "capx/third_party/", "submodule 7 つ(LIBERO-PRO ほか)")
    box(50, 316, 560, 76, "sim", "venv", ".venv(Robosuite)・.venv-libero(LIBERO-PRO)・.venv-serve(vLLM)")
    txt(330, 430, "収集と評価: このパッケージが書いた run.sh が、この venv で capx/envs/launch.py を呼ぶ")
    txt(330, 456, "シミュレータ、multi-turn の trial loop、skill library はすべてこちらにある")
    arrow("M125,126 L125,168", "run.sh が呼ぶ", 135, 152, anchor="start")
    arrow("M535,126 L535,168", "clone・patch・submodule", 525, 152, anchor="end")

    # right: where the code comes from, and the reproduction-only tree
    box(680, 50, 260, 76, "model", "GitHub", "matsuolab/geniac_prize_2026_code")
    arrow("M810,126 L810,196 L632,196", "決まった commit を取る", 820, 166, anchor="start")
    panel(670, 310, 280, 180, "ベースラインの数字の再現だけ", dashed=True)
    box(690, 352, 240, 76, "data", "ASPIRE", "patches/aspire/setup.sh")
    txt(810, 452, "venv はコンペの CaP-X の物を使う")
    txt(810, 474, "docs/reproduce_baseline.md")
    o.append("</svg>")
    return "\n".join(o)


for k, c in T.items():
    (OUT / f"setup-{k}.svg").write_text(svg(c))
