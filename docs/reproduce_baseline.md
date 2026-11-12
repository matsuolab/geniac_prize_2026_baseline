# Reproducing the baseline numbers (ASPIRE)

[results.md](results.md) の数字と、配布しているベースラインのデータと adapter は、ASPIRE に同梱された CaP-X で作った。
同じ数字を出し直す時だけ、このページの環境を使う。それ以外の収集と評価は、[installation.md](installation.md) の環境
(コンペの CaP-X + patch)で行う。2 つの環境の数字は比べない。

ベースラインのデータ(LIBERO の成功 trial すべて)は ASPIRE の追加 skill を呼ぶ。この skill はコンペの CaP-X にも
`patches/geniac_prize/skill-library.patch` で入れてある([setup.md](setup.md) の Skill library)ので、収集と評価は
コンペの CaP-X で行える。ASPIRE を使うのは、LIBERO-PRO の初期状態の扱いなど下の違いを含めて、数字を同じ条件で
出し直す時だけである。skill library を変えることは参加者の改善の範囲である。

## Tree

ASPIRE は次の 2 つを重ねた物である。`bash patches/aspire/setup.sh [target_dir]` が f4c8939 を clone して patch を当て、
変更した 9 file の sha256 を `patches/aspire/SHA256SUMS` と照合する。patch を当てた tree は、ベースラインの収集と評価に
使った tree と 9 file とも同じ内容である(sha256 で確認)。

| item | value |
| --- | --- |
| ASPIRE | [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) @ `f4c8939`(branch v0.1) |
| 追加の変更 | `patches/aspire/aspire-f4c8939-local-changes.patch`(9 file)。LIBERO 用の追加 skill 14 個、trial の実時間上限(`ASPIRE_TRIAL_TIMEOUT_S`、既定 2400 s)など。一覧は [setup.md](setup.md) の aspire にある |
| skill library | `aspire/sim/cap/integrations/franka/libero_reduced_skill_library.py`、sha256 `a5bda54c96971a3263f4e726e71a8baf79dc7498f0289455e6f99c7c56dac2b7` |

## Run spec

ASPIRE には venv を作らない。シミュレータと vLLM は [installation.md](installation.md) で作ったコンペの CaP-X の venv を
絶対 path で使う。ASPIRE の lock は torch 2.12.1+cu126 を入れ、これは B300(compute 10.3)の kernel を持たない。

`configs/` の run spec を次のように書き換え、タスクの config に `configs/capx/aspire/` を使う。launch は
`<ASPIRE root>/aspire/sim` で実行される(ベースラインと同じ)。

```yaml
capx_root: <ASPIRE root>
launch_script: aspire/sim/cap/envs/launch.py
pythonpath: <ASPIRE root>
sim_python: <geniac_prize_2026_code root>/.venv-libero/bin/python    # Robosuite は .venv/bin/python
serve_python: <geniac_prize_2026_code root>/.venv-serve/bin/python
```

ベースラインの数字は次の venv で測った。

| venv | torch | simulator |
| --- | --- | --- |
| `.venv`(Robosuite) | 2.13.0+cu130 | robosuite 1.5.1、mujoco 3.5.0 |
| `.venv-libero`(LIBERO-PRO) | 2.13.0+cu130 | robosuite 1.4.0、mujoco 3.5.0 |
| `.venv-serve`(vLLM) | 2.11.0 | |

## Differences from the competition CaP-X

ASPIRE の CaP-X は、コンペの CaP-X と主に次の点が違う。どちらの tree で測ったかで数字が変わるのは、主にこのためである。

| item | ASPIRE | コンペの CaP-X |
| --- | --- | --- |
| skill library | LIBERO 用の追加 skill 14 個がある | `skill-library.patch` で同じ 14 個を足す(Robosuite の skill library は元から同じ) |
| LIBERO の初期状態 | 保存された init state を読み込んだ後にもう一度 reset し、配置を引き直す。seed の間で配置が重ならない | 保存された init state をそのまま使う。seed `s` と `s + init state の数` が同じ配置になる(layout 単位の分割はこのためにある、[collect.md](collect.md)) |
| trial の実時間上限 | 既定 2400 s、時間切れなら 1 回やり直す | 既定 1000 s、時間切れなら 2 回やり直す。patch により `trial_timeout_s` で変えられる |
| 判断の語の読み方 | 回答のどこかに REGENERATE があれば regenerate | REGENERATE か FINISH で始まる最初の行で決める |
| 画像と VDM を使う条件 | config に加えて、モデルが `VLM_MODELS` にあること | config だけで決まる |
| VDM テキストのファイル | 書く | patch を当てた時だけ書く |
