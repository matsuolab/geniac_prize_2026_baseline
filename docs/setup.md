# Setup scripts and patches

`patches/` には、外部のリポジトリを決まった commit で取ってきて patch を当てる setup script が、tree ごとに 1 つずつある。
どちらも、clone、checkout、`git apply --check`、apply の順に行い、すでにある directory には書き込まない。

| directory | 対象 | 使う時 | 作る tree(既定) |
| --- | --- | --- | --- |
| `patches/geniac_prize/` | [matsuolab/geniac_prize_2026_code](https://github.com/matsuolab/geniac_prize_2026_code)(コンペの CaP-X) | 収集と評価のすべて | `./geniac_prize_2026_code` |
| `patches/aspire/` | [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) @ `f4c8939` | ベースラインの数字を出し直す時だけ([reproduce_baseline.md](reproduce_baseline.md)) | `./ASPIRE` |

どちらもリポジトリの root で実行する。

## geniac_prize

コンペの CaP-X をベースラインの収集と評価に使うための patch で、2 つある。このパッケージが対応する commit はこれだけである。

| patch | 中身 |
| --- | --- |
| `geniac-prize.patch` | 収集と評価が依存する修正([Fixes](#fixes)) |
| `skill-library.patch` | ASPIRE で足した LIBERO 用の skill 14 個([Skill library](#skill-library)) |

```bash
bash patches/geniac_prize/setup.sh [target_dir]  # clone、checkout、2 つの patch の git apply --check と apply、sha256 の表示、submodule
```

clone 元の既定は SSH(`git@github.com:matsuolab/geniac_prize_2026_code.git`)で、GitHub に SSH 鍵を登録してある必要がある。
リポジトリが private の間は HTTPS では認証なしに clone できないためで、公開後は HTTPS に戻す。`GENIAC_PRIZE_URL` で clone 元の URL を変えられる。
venv の作り方は [installation.md](installation.md) にある。
`configs/*.yaml` の既定はこの配置(`capx_root: ../geniac_prize_2026_code`、`.venv`、`.venv-libero`、`.venv-serve`)に
合わせてあり、タスクの config は `configs/capx/geniac_prize/` を使う。

`geniac-prize.patch` はコンペの CaP-X 本体に取り込んでもらう予定である。取り込まれた修正は patch から消す。
`skill-library.patch` は提出物の側(skill library は提出物の一部、ルールブック §4.1)の変更なので、本体には入れない。

固定する commit は `patches/geniac_prize/setup.sh` の `commit=` の 1 行だけに書いてある。commit を変える時はこの行を変え、
`setup.sh` を実行して patch が当たることを確かめる。

### Setting

ベースラインのデータと adapter は ASPIRE で作った([reproduce_baseline.md](reproduce_baseline.md))。
skill library は `skill-library.patch` で ASPIRE と同じになるが、LIBERO-PRO の初期状態の扱いなど他の違いが残るため、
ここで測った成功率はベースラインの数字とは別の設定の値である。比べる時は同じ環境で学習前と学習後を測る。
trial の実時間上限は run spec の `trial_timeout_s`(既定 2400 s、ベースラインの ASPIRE と同じ)が `CAPX_TRIAL_TIMEOUT_S` として渡る。retry の回数はコンペの CaP-X の
3 回で、ASPIRE の 2 回とは違う。

### Fixes

行番号は setup.sh が固定する commit(patch を当てる前)のものである。env var を設定しない場合の挙動は patch 前と同じである。

| # | fix | 対象 | env var | why |
| --- | --- | --- | --- | --- |
| 1 | VDM テキストを turn ごとに `differencing_feedback_XX.txt` として trial フォルダに保存する | `capx/envs/trial.py` の `_handle_multi_turn_step` と `_run_single_trial`、`capx/utils/launch_utils.py` の `_save_trial_artifacts` | なし | `convert rows` が FINISH の prompt を再構成するのに要る([data_format.md](data_format.md))。ファイル名と番号は ASPIRE 版と同じ |
| 2 | trial の実時間上限を env var で変える。reset 後の再設定(`capx/envs/trial.py:783-786`)も同じ値を使う | `capx/envs/runner.py:36` | `CAPX_TRIAL_TIMEOUT_S`(既定 1000) | 長い trial が途中で打ち切られる。retry の回数は変えない |
| 3 | perception サーバーの URL を env var から読む | `capx/integrations/vision/sam3.py:20`、`sam2.py:17`、`owlvit.py:16`、`capx/integrations/motion/pyroki.py:12`、`curobo.py:10` | `SAM3_SERVICE_URL`、`SAM2_SERVICE_URL`、`OWLVIT_SERVICE_URL`、`PYROKI_SERVICE_URL`、`CUROBO_SERVICE_URL` | 1 台で複数の run を並べる時にポートを動かせない。GraspNet と Molmo は元から env var を読む |
| 4 | 外部の LIBERO checkout を読み込む。`libero.libero` の構成の package も読める | `capx/integrations/libero/__init__.py:89-95` | `CAPX_LIBERO_ROOT` | 同梱の LIBERO-PRO だけを探し、`from libero import benchmark` の構成しか読めない |

### Skill library

`skill-library.patch` は `capx/integrations/franka/libero_reduced_skill_library.py` の
`FrankaLiberoApiReducedSkillLibrary` に、ASPIRE の patch(`patches/aspire/`)が足した skill 14 個をそのまま足す。
ベースラインの LIBERO-PRO のデータはこれらを呼ぶので、無い tree では学習したモデルのコードがその場で失敗する。

- 2 つの tree の元のファイルは、ASPIRE の license header と module の path を除いて同じである。足す行も ASPIRE の
  patch と 1 行ずつ同じで(`tests/test_patches.py` で確認)、patch 後のファイルは ASPIRE の物
  (sha256 `a5bda54c…`、[reproduce_baseline.md](reproduce_baseline.md))から header を除き path を `capx.` にした物と一致する。
- Robosuite の skill library(`control_reduced_skill_library.py`)は 2 つの tree で元から同じなので、変えない。
- `point_at_object` は `CAPX_VLM_URL` と `CAPX_VLM_MODEL` の VLM に物の位置を聞く。片方でも無ければ VLM は使わず、
  SAM3 の text prompt で探す。収集と評価の script は、Qwen の VDM サーバーをこの 2 つに入れる。
- `configs/capx/geniac_prize/libero/` の config がこの API を使う時、収集と評価の点検(`check_skill_library`)は
  14 個がすべて登録されているかを見て、無ければ止まる。

以前の setup.sh で作った tree には、次で足せる。

```bash
git -C geniac_prize_2026_code apply patches/geniac_prize/skill-library.patch
```

| skill | 役目 |
| --- | --- |
| `goto_pose_settled` | 同じ目標に `goto_pose` を数回出し、長い移動を最後まで届かせる |
| `point_at_object` | VLM に画像の中の物を指させ、pixel を返す |
| `locate_object_points` | VLM が指した点から SAM3 で切り出し、物の点群(world 座標)を返す |
| `top_down_grasp_for` | 立っている物(箱、瓶、缶)を上から掴む姿勢 |
| `bowl_rim_grasp_for` | 指より広い bowl を縁で掴む姿勢 |
| `bowl_place_pose_on` | 縁で持った bowl を目標の中心に置く gripper の位置 |
| `place_pose_on_top_of` | 物を目標の上に置く位置 |
| `survey_from_above` | 手首のカメラで上から物を探す |
| `refine_bowl_centre_from_above` | 手首のカメラで bowl の中心と縁の高さを測り直す |
| `yaw_quat` | 下向きの gripper を鉛直軸の周りに回した姿勢 |
| `open_drawer_by_handle` | 取っ手を掴んで引き出しを開ける |
| `place_pose_in_container` | 籠の中に落とす位置 |
| `wine_rack_place_pose` | 首を持ったワインの瓶をラックに置く姿勢 |
| `turn_stove_knob` | コンロのつまみを掴んで回す |

## aspire

ベースラインのデータ、adapter、数字を作った ASPIRE の tree を作る。収集と評価の既定には使わない。

```bash
bash patches/aspire/setup.sh [target_dir]  # clone, checkout f4c8939, git apply --check, apply, sha256 check
```

`ASPIRE_URL` で clone 元の URL を変えられる。patch を当てた後、変更した 9 file の sha256 を
`patches/aspire/SHA256SUMS` と照合する(ベースラインの収集と評価に使った tree と同じ内容であることの確認)。
venv は作らず、geniac_prize の venv を使う([reproduce_baseline.md](reproduce_baseline.md))。

### Changes

path は `aspire/sim/cap/` からの相対である。

| change | 対象 | env var |
| --- | --- | --- |
| `VLM_MODELS` に `Qwen/Qwen3.8-27B` を足す(画像と VDM を使うため) | `llm/client.py` | なし |
| LIBERO 用の skill を 14 個足す | `integrations/franka/libero_reduced_skill_library.py` | `CAPX_VLM_URL`、`CAPX_VLM_MODEL`(`point_at_object` が VLM に物の位置を聞く先。無ければ使わない) |
| trial の実時間上限を env var で変える(既定 2400 s) | `envs/runner.py` | `ASPIRE_TRIAL_TIMEOUT_S` |
| perception サーバーと Molmo の URL を env var から読む | `integrations/vision/sam3.py`、`integrations/vision/molmo.py`、`integrations/motion/pyroki.py` | `SAM3_SERVICE_URL`、`CAPX_MOLMO_URL`、`PYROKI_SERVICE_URL` |
| OmniGibson 3.9 に合わせる(BEHAVIOR 用。LIBERO と Robosuite では使わない) | `envs/simulators/og_compat.py`(新規)、`envs/simulators/r1pro_b1k.py`、`integrations/r1pro/utils.py` | `CAPX_CUROBO_TORCH_LBFGS`、`CAPX_CUROBO_CUDA_GRAPHS` |
