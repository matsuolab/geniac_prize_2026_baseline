# 7. データ収集

[Tutorial](README.md) | 前: [6. W&B 連携](06_wandb.md) | 次: [8. 学習実行](08_train.md)

学習前のモデル(Qwen3.8-27B)で CaP-X の multi-turn の trial を集め、[8 章](08_train.md)の学習データ
(`runs/data/split/` と `runs/data/pairs/`)に変換する。成功した trial は SFT に、失敗した trial は成功した trial と
組んで DMPO の pair に使う。`<...>` の意味は
[Tutorial](README.md#placeholders) の表にある。

| 工程 | 場所 | 中身 |
| --- | --- | --- |
| `collect plan` | ログインノード | run spec を点検し、`<output_dir>/run.sh`、seed の区間ごとの config、`sources.yaml` を書く。何も実行しない |
| `run.sh` | 計算ノード(GPU 1 枚) | vLLM サーバー 2 つ(Qwen3.8-27B、Molmo2-8B)を 1 つずつ立て、seed の区間ごとに CaP-X を走らせる |
| `convert`、`pairs` | 計算ノード(GPU 無し) | trial を行データにし、train / val に分け、DMPO の pair を作る |

## モデルの準備

1. 4 章の環境構築を終えていること。

2. ログインノードで Hugging Face にログインする。収集では SAM3 のサーバーが、最初の起動時に `facebook/sam3` の重みを
   `~/.cache/huggingface/hub` に取る。この repo は gated なので、先に Hugging Face のページ
   (https://huggingface.co/facebook/sam3)でアクセスを申請し、承認されたアカウントのトークンでログインしておく。
   fine-grained のトークンを使う場合は、gated の repo を読む権限(Read access to contents of all public gated repos
   you can access)を付ける。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   geniac_prize_2026_baseline/.venv/bin/hf auth login
   geniac_prize_2026_baseline/.venv/bin/hf auth whoami      # "Logged in" と "user: <アカウント名>" が出れば良い
   ```

3. モデルの重みを用意する。Qwen3.8-27B(約 52 GB)と Molmo2-8B(約 33 GB)の snapshot を、計算ノードから見える場所に
   置く。チームの共用ストレージに既にあれば、それを `<モデルのスナップショット>` と `<Molmo2-8B のスナップショット>`
   として使う(チームのリーダーに聞く)。Qwen3.8-27B は revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` の物を使う。
   snapshot が無ければ、ログインノードで取る。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   geniac_prize_2026_baseline/.venv/bin/hf download Qwen/Qwen3.8-27B \
       --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir models/Qwen3.8-27B
   geniac_prize_2026_baseline/.venv/bin/hf download allenai/Molmo2-8B --local-dir models/Molmo2-8B
   ```

   この場合は `<モデルのスナップショット>`、`<Molmo2-8B のスナップショット>`は`<共用ストレージ>/<ユーザー名>/models/Qwen3.8-27B` 、 `.../models/Molmo2-8B` となる。

4. ベースラインの chat template を作る([installation.md](../installation.md) の Model weights)。最後に sha256 が
   表示され、ベースラインの物と違えばエラーで止まる。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   .venv/bin/geniac-baseline template --model-dir <モデルのスナップショット> --out runs/qwen38_low_reasoning.jinja
   ```

## run spec を作る

1. run spec (設定ファイル) をコピーする。LIBERO-PRO は `configs/collect_libero.yaml`(30 task、seed 301-340 と 351-390)、Robosuite は
   `configs/collect.yaml`(7 task、seed 301-400)である。以降は LIBERO-PRO で書く。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   cp configs/collect_libero.yaml configs/my_collect_libero.yaml
   ```

   spec の中の相対パス(タスクの `config:`、`capx_root`、`output_dir`、`chat_template`)は spec のあるディレクトリから
   解決される。タスクの config は `configs/capx/` にあるので、コピーは `configs/` の中に置く。他の場所に置くと
   `collect plan` が `FileNotFoundError` で止まる。

2. 次の項目を書き換える。

   ```yaml
   model_path: <モデルのスナップショット>
   molmo_path: <Molmo2-8B のスナップショット>
   b300_workarounds: true    # B300 の GPU で動かす時は必ず true
   ```

   GMO クラウドの GPU は B300 なので、`b300_workarounds` を true にする。

3. 通常は以上の設定で動く。ただし、同じ計算ノードで他の人が既定のポートを使っている場合、衝突を避けるためポート番号をずらす必要がある。

   ```yaml
   port_offset: 300
   vllm_port_offset: 300
   ```

   また、小規模でテストしたい場合はtasksの項を編集する。

   - `tasks:`: 1 項目が 1 タスクである。

     ```yaml
     tasks:
       - task: libero_goal_bowl_on_stove                                 # タスク名
         config: capx/geniac_prize/libero/libero_goal_bowl_on_stove.yaml   # CaP-X のタスクの config(spec のあるディレクトリから)
         layouts: 50                                                     # LIBERO-PRO の init state の数
         seeds: ["301-340", "351-390"]                                   # 使う seed
     ```

     seed は trial の番号で、1 つの seed で 1 回 trial を流す。train と val のどちらになるか、使えるかどうかは、
     seed の値で決まる([data_format.md](../data_format.md) の Train / val split)。

     | | train | val | 使えない(評価用) |
     | --- | --- | --- | --- |
     | LIBERO-PRO | 301-340、351-380 | 381-390 | 341-350、391-400、301-400 の外 |
     | Robosuite | 301-390 | 391-400 | 301-400 の外 |

     - 使えない seed を入れると `collect plan` が止める。LIBERO-PRO の 341-350 と 391-400 は、init state が 50 の
       タスクで評価に使う layout 40-49 と同じ場面になる。
     - val に成功した trial が 1 つも無いと、8 章の SFT が止まる。少なく試す時も val の seed を入れる。
     - DMPO は同じタスクの成功と失敗を組むので、成功か失敗の片方しか出ないタスクは使えない。学習前のモデルの
       成功率は [results.md](../results.md) の左の列にある(例: `libero_goal_bowl_on_stove` は 27/50)。
     - `seeds` の 1 つの区間(例: `"301-340"`)ごとに CaP-X を 1 回走らせ、その中の trial を 8 つずつ並列に流す。
       結果は `<output_dir>/seg_<task>_<区間の最初の seed>/` に入る。

## 計画を書き出す

ログインノードで実行する。spec を点検し、`output_dir`(既定は `runs/collect_libero`)に書き出す。

```bash
cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
.venv/bin/geniac-baseline collect plan --spec configs/my_collect_libero.yaml
```

```text
[plan] wrote <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/runs/collect_libero/run.sh
[plan] after the run: geniac-baseline convert rows --sources .../runs/collect_libero/sources.yaml ...
[plan] run it on a GPU node: bash .../runs/collect_libero/run.sh
```

モデル、`save_multiturn_prompts`、seed、ポート、chat template のどれかに問題があれば、何も書かずに止まる。
`output_dir` を作り直す時は、古い物を消してから実行する。

## 収集を流す

1. 収集のジョブスクリプトを作る。`RUN_DIR` には `collect plan` が書いたディレクトリを渡す。`<時間>` はパーティションの
   上限以内にする。

   `VLLM_USE_FLASHINFER_SAMPLER=0` は、vLLM のサンプラーが起動時に `nvcc` でカーネルをコンパイルしないようにする。
   計算ノードに CUDA の toolkit が無いと、これが無い場合 vLLM は
   `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` で起動に失敗する。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   cat > collect_job.sh <<'EOF'
   #!/usr/bin/env bash
   #SBATCH --job-name=collect
   #SBATCH --partition=<パーティション>
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=16
   #SBATCH --mem=256G
   #SBATCH --time=<時間>
   #SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
   set -euo pipefail
   : "${RUN_DIR:?sbatch --export=ALL,RUN_DIR=<collect plan の出力> で渡す}"
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   hostname
   nvidia-smi --query-gpu=name,memory.total --format=csv
   export VLLM_USE_FLASHINFER_SAMPLER=0
   bash "$RUN_DIR/run.sh"
   EOF
   ```

   `--output` などの `#SBATCH` の行には `~` を使わない。Slurm は `~` を展開せず、投入したディレクトリの下に `~` という
   名前のディレクトリを作ってログを書く。

2. 投入する。

   ```bash
   sbatch --export=ALL,RUN_DIR=runs/collect_libero collect_job.sh
   ```

3. 進み具合を見る。ジョブのログ(`<共用ストレージ>/<ユーザー名>/logs/collect-<jobid>.out`)に、サーバーの起動と
   区間ごとの終了が出る。

   ```text
   [run] starting qwen on :8110
   [run] qwen ready after 250s
   [run] starting molmo on :8122
   [run] molmo ready after 120s
   [run] libero_goal_bowl_on_stove:301-308
   [run] libero_goal_bowl_on_stove:301-308 rc=0
   [run] libero_goal_bowl_on_stove:381-388
   [run] libero_goal_bowl_on_stove:381-388 rc=0
   [run] done; trial folders under .../runs/collect_libero/seg_*/outputs/
   ```

   サーバーが起動しない時は `FATAL: qwen died` の後に vLLM のログの末尾が出る。全文は
   `runs/collect_libero/vllm_qwen.log` にある。trial ごとの様子は `runs/collect_libero/seg_<task>_<seed>/<task>.log` に
   ある。LIBERO-PRO では、このログに `ModuleNotFoundError: No module named 'robosuite.controllers.composite'` が出るが、
   Robosuite 1.5 のタスクを登録しようとして失敗しただけ(`capx/envs/simulators/__init__.py` の try)で、問題ない。

   B300 1 枚で、サーバーの起動に Qwen が約 4 分、Molmo が約 2 分かかった。1 つの区間は 8 trial を並列に流し
   (`num_workers` 8)、8 seed の区間が 12〜15 分で終わった。

4. 結果を見る。trial ごとに次の名前のフォルダができる。`taskcompleted_1` が成功である。

   ```text
   runs/collect_libero/seg_<task>_<seed>/outputs/Qwen_Qwen3.8-27B/<task>/trial_<seed>_sandboxrc_<rc>_reward_<r>_taskcompleted_<0|1>/
   ```

   CaP-X はコードブロックを実行するたびにその時点の結果でフォルダを保存するので、同じ seed のフォルダが複数あることが
   ある。trial の結果は、`summary.txt` に `Reward:` の行があるフォルダ(trial 終了時の保存)である。フォルダの数を
   trial の数として数えない。変換(`convert rows`)はこのフォルダだけを使う。

## 学習データに変換する

1. 変換のジョブスクリプトを作る。GPU は要らない。`convert tag` は LLM 本体ではなく tokenizer だけを読む。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   cat > convert_job.sh <<'EOF'
   #!/usr/bin/env bash
   #SBATCH --job-name=convert
   #SBATCH --partition=<パーティション>
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --time=01:00:00
   #SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
   set -euxo pipefail
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   GB=.venv/bin/geniac-baseline
   SRC=runs/collect_libero/sources.yaml
   TOK=<モデルのスナップショット>
   "$GB" convert rows --sources $SRC --out runs/data/rows
   "$GB" convert rows --sources $SRC --include failure --out runs/data/fail_rows
   "$GB" convert tag --src runs/data/rows --out runs/data/tagged --tokenizer $TOK
   "$GB" convert tag --src runs/data/fail_rows --out runs/data/fail_tagged --tokenizer $TOK
   "$GB" convert split --src runs/data/tagged --out runs/data/split
   "$GB" convert validate runs/data/split/sft.jsonl runs/data/split/val.jsonl
   "$GB" pairs --success runs/data/split --failure runs/data/fail_tagged --out runs/data/pairs
   EOF
   sbatch convert_job.sh
   ```

   各段階は新しいディレクトリに書き、既存のディレクトリは上書きしない。やり直したいときは `runs/data/` のその段階が該当する
   ディレクトリを消す。複数の収集(別の `output_dir`)をまとめる時は、`sources.yaml` の `sources:` の項目を 1 つの
   ファイルに並べ、それを `SRC` に渡す。

2. 変換のジョブのログ(`<共用ストレージ>/<ユーザー名>/logs/convert-<jobid>.out`)で件数を確かめる。`convert split` が
   SFT の行数を、`pairs` が pair の数を出す。

   ```text
   {
    "libero|train": 33,
    "libero|val": 14
   }
   runs/data/split/sft.jsonl: 33 rows OK
   runs/data/split/val.jsonl: 14 rows OK
   {
    "train|task": 4,
    "val|task": 2
   }
   ```

   上は 2 task × 16 trial(成功 6、失敗 26)の時の値である。`libero|val` が無いと SFT が、`train|...` が無いと DMPO が
   止まる。その場合は、まだ使っていない seed か、成功率の違うタスクを足して収集し直す。val の seed は 1 タスクに
   10 個しかないので、足りない時はタスクを足す。

3. 完成物の配置を確かめる。[8 章](08_train.md)はこの 2 つを使う。

   | ディレクトリ | ファイル | 使う段階 |
   | --- | --- | --- |
   | `runs/data/split/` | `sft.jsonl`、`val.jsonl`、`manifest.json`(成功した trial) | SFT |
   | `runs/data/pairs/` | `pairs.jsonl`、`rows.jsonl`、`manifest.json`(成功と失敗の trial の pair) | DMPO |

次: [8. 学習実行](08_train.md)
