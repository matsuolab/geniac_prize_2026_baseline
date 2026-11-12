# 9. ローカル評価の実行

[Tutorial](README.md) | 前: [8. 学習実行](08_train.md) | 次: [10. Hugging Face へのアップロード](10_upload.md)

学習の効果を測るために、学習前のモデルと学習した adapter を、チームのクラスタ上で評価し、成功率を出す。
`<...>` の意味は [Tutorial](README.md#placeholders) の表にある。

ここで出る数字は、リーダーボードのスコアとは別の物である。リーダーボードは運営チームが提出 image を gateway 越しに
走らせて測る。この章の評価は、学習前と学習後を同じ seed と同じ CaP-X の tree で比べるための
物である。ここで測る値は、ASPIRE で測ったベースラインの数字([results.md](../results.md))とも設定が違う
([setup.md](../setup.md) の geniac_prize の Setting)。

流れは次のとおりである。ログインノードで評価の準備をし(GPU 無し)、計算ノードのジョブで実行と採点をする。

| 工程 | 場所 | 中身 |
| --- | --- | --- |
| `eval plan` | ログインノード | run spec を点検し、`<output_dir>/run.sh` と seed ごとの config を書く。何も実行しない |
| `run.sh` | 計算ノード(GPU 1 枚) | vLLM サーバー 3 つ(VDM、コード生成、Molmo)を 1 つずつ立て、seed の区間ごとに CaP-X を走らせる |
| `eval score` | 計算ノード(ジョブの最後) | trial フォルダの名前から成功を数え、`<output_dir>/score.json` に書く |

## 準備

[4 章](04_install.md)の環境構築と、[7 章](07_collect.md)の「[モデルの準備](07_collect.md#モデルの準備)」(snapshot と chat template)を終えていること。学習後の
評価には、[8 章](08_train.md)の DMPO の adapter(`runs/train_dmpo/adapter/`)が要る。

## 評価する

1. run spec を自分用にコピーして書き換える。既定の spec は `configs/` にあり、タスクの config を spec からの相対パスで
   指すので、コピーも `configs/` に置く。`configs/my_*.yaml` は `git status` に出るが commit しない(ホストに結びつく
   パスが入るため。[CONTRIBUTING.md](../../CONTRIBUTING.md) の Rules)。

   | 評価 | 元の spec | 出力(`output_dir`) |
   | --- | --- | --- |
   | 学習前、Robosuite 7 task × 50 seed | `configs/eval_base.yaml` | `runs/eval_base` |
   | 学習前、LIBERO-PRO 30 task × 50 seed | `configs/eval_base_libero.yaml` | `runs/eval_base_libero` |
   | 学習後(DMPO の adapter)、Robosuite | `configs/eval_trained.yaml` | `runs/eval_trained` |
   | 学習後、LIBERO-PRO | `configs/eval_trained_libero.yaml` | `runs/eval_trained_libero` |

   まず、1 task 2 seed だけの小さな評価で、サーバーの起動から採点までが通ることを確かめる。ここでは、学習前の
   Robosuite を例にする。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   cp configs/eval_base.yaml configs/my_eval_smoke.yaml
   ```

   `configs/my_eval_smoke.yaml` の次の行を書き換え、`tasks:` は `cube_lifting` の 1 つだけを残す。

   ```yaml
   model_path: <モデルのスナップショット>
   molmo_path: <Molmo2-8B のスナップショット>
   output_dir: ../runs/eval_smoke
   b300_workarounds: true        # B300 の GPU で動かす時は必ず true
   tasks:
     - task: cube_lifting
       config: capx/geniac_prize/robosuite/cube_lifting.yaml
       seeds: ["401-402"]
   ```

   - `model_path` と `molmo_path` には [7 章](07_collect.md#モデルの準備)で用意した snapshot を書く。`null` のままにすると、vLLM が計算ノードで
     Hugging Face から重みを取りに行く。
   - seed は評価用の物しか使えない(Robosuite は 401-450、LIBERO-PRO は [eval.md](../eval.md) の Seeds)。それ以外は
     次の `eval plan` が止める。
   - 評価のサーバーは port 8110、8111、8122、8113-8117 を使う。同じ計算ノードで他の人の評価や収集が動いている時は、
     [7 章](07_collect.md)の「[run spec を作る](07_collect.md#run-spec-を作る)」と同じように `port_offset` と `vllm_port_offset` をずらし、他の人と
     違う値にする。

2. ログインノードで計画と点検をする。`[plan] wrote` の行が出れば、`runs/eval_smoke/` に `run.sh`、`plan.json`、
   seed の区間ごとの `seg_*/` ができている。

   ```bash
   .venv/bin/geniac-baseline eval plan --spec configs/my_eval_smoke.yaml
   ```

   ```text
   [plan] wrote <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/runs/eval_smoke/run.sh
   [plan] run it on a GPU node: bash <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/runs/eval_smoke/run.sh
   ```

3. 評価のジョブスクリプトを作る。[7 章](07_collect.md#収集を流す)の収集のジョブスクリプトに、最後の採点の行を足した物である。`RUN_DIR` には `eval plan` が書いた
   ディレクトリを渡す。`<時間>` はパーティションの上限以内にする(所要時間は 4 番)。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   cat > eval_job.sh <<'EOF'
   #!/usr/bin/env bash
   #SBATCH --job-name=eval
   #SBATCH --partition=<パーティション>
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=16
   #SBATCH --mem=256G
   #SBATCH --time=<時間>
   #SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
   set -euo pipefail
   : "${RUN_DIR:?sbatch --export=ALL,RUN_DIR=<eval plan の出力> で渡す}"
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   hostname
   nvidia-smi --query-gpu=name,memory.total --format=csv
   export VLLM_USE_FLASHINFER_SAMPLER=0
   bash "$RUN_DIR/run.sh"
   .venv/bin/geniac-baseline eval score "$RUN_DIR" --json "$RUN_DIR/score.json"
   EOF
   ```

4. ジョブを投入し、終わるのを待つ。`run.sh` はサーバーを 1 つずつ立てて `/health` を待つので、CaP-X が始まるまでに
   数分から 10 分ほどかかる。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   sbatch --export=ALL,RUN_DIR=$PWD/geniac_prize_2026_baseline/runs/eval_smoke eval_job.sh
   squeue -u $USER
   tail -f logs/eval-<jobid>.out
   ```

   ログには次の行が順に出る。

   ```text
   [run] starting vdm on :8110
   [run] vdm ready after ...s
   [run] starting code on :8111
   [run] code ready after ...s
   [run] starting molmo on :8122
   [run] molmo ready after ...s
   [run] cube_lifting:401-402
   [run] cube_lifting:401-402 rc=0
   [run] done; trial folders under <output_dir>/seg_*/outputs/
   {"n": 2, "missing": 0, "success": ..., ...}
   ```

   止まった時に見る場所:

   | 症状 | 見る場所 | 対処 |
   | --- | --- | --- |
   | `[run] FATAL: vdm died` や `[run] FATAL: code never became ready` が出て、ジョブが終わる(サーバーが起動しない) | ジョブのログに出る vLLM のログの末尾。全文は `runs/eval_smoke/vllm_<vdm\|code\|molmo>.log` | GPU の memory 不足なら、spec の `qwen_gpu_util`、`molmo_gpu_util`、`max_model_len` を下げる |
   | `[run] cube_lifting:401-402 rc=1` のように `rc=` が 0 以外で、最後に `[run] FAILED segment ...` が出る(サーバーの起動後に CaP-X が trial の前に落ちたか、kill された。他の区間は流れるが、採点は行われない) | `runs/eval_smoke/seg_<task>_<区間の最初の seed>/<task>.log`(CaP-X の出力)の末尾の Traceback。`rc=137` で Traceback が無ければ、計算ノードの RAM 不足による kill | 原因を直し、その区間の seed だけを refill(「採点する」の 2 番)で流し直す。`rc=137` ならジョブスクリプトの `--mem` を増やす。小さな評価なら `runs/eval_smoke/` を消して 2 番からやり直してもよい(残っていると `eval plan` が `refusing to overwrite` で止まる) |

   所要時間の目安(B300 1 枚、ベースラインの実測)は、LIBERO-PRO 9 task × 50 trial で 3〜4 h、Robosuite 4 task × 50
   trial で 7〜14 h である。task の数で換算すると、全 task の評価は LIBERO-PRO(30 task)で約 10〜13 h、
   Robosuite(7 task)で約 12〜25 h になる。パーティションの時間制限に収まらない場合は、`tasks:` を分けた spec を
   複数作り(`output_dir` も別にする)、別々のジョブで流す。全体の採点は「採点する」の 1 番のようにまとめて行える。

5. 小さな評価が通ったら、同じ手順(1〜4 番)で本番の評価を流す。spec をコピーし、`model_path`、`molmo_path`、
   `b300_workarounds` を 1 番と同じに書き換える(`tasks:` と `output_dir` は変えない)。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   cp configs/eval_base.yaml configs/my_eval_base.yaml         # 学習前
   cp configs/eval_trained.yaml configs/my_eval_trained.yaml   # 学習後
   ```

   LIBERO-PRO は `*_libero.yaml` を使う(`sim_python` が `.venv-libero` を指す)。学習後の spec の `adapter` は
   既定で `../runs/train_dmpo/adapter` を指し、`lora_name`(`dmpo-lora`)の名前でコード生成のサーバーに載る。
   学習後の `run.sh` では、コード生成のサーバーに `--enable-lora --max-lora-rank <adapter の r> --max-loras 1
   --lora-modules dmpo-lora=<adapter>` が付き、CaP-X には `--model dmpo-lora` が渡る。

### 補足: examples で計画から採点までを 1 つのジョブにまとめる

本番の評価は、`examples/01_eval_base_cap_agent0.sh`(学習後は `examples/05_eval_trained_cap_agent0.sh`)を使うと、
2 番の計画、`run.sh` の実行、採点を 1 つのジョブで流せる。この場合は 2 番を行わず、3 番のジョブスクリプトの
代わりに次のスクリプトを使う。

```bash
cd <共用ストレージ>/<ユーザー名>
cat > eval_example_job.sh <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --partition=<パーティション>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=<時間>
#SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
set -euo pipefail
: "${SPEC:?sbatch --export=ALL,SPEC=<spec>,EXAMPLE=<example> で渡す}"
: "${EXAMPLE:?sbatch --export=ALL,SPEC=<spec>,EXAMPLE=<example> で渡す}"
cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
export VLLM_USE_FLASHINFER_SAMPLER=0
RUN_GPU=1 bash "$EXAMPLE"
EOF
sbatch --export=ALL,SPEC=configs/my_eval_base.yaml,EXAMPLE=examples/01_eval_base_cap_agent0.sh eval_example_job.sh         # 学習前
sbatch --export=ALL,SPEC=configs/my_eval_trained.yaml,EXAMPLE=examples/05_eval_trained_cap_agent0.sh eval_example_job.sh   # 学習後
```

- example は先に `eval plan` を行うので、2 番を済ませた spec を渡すと `refusing to overwrite` で止まる(その時は
  `output_dir` を消す)。
- spec の誤りに気づくのはジョブが始まってからになるため、動作確認のための小さな評価は 2〜4 番の手順で行うとよい。

## 採点する

1. 採点はジョブの最後に行われる。ログには 1 行目に全体、2 行目以降に task ごとの結果が出て、
   `<output_dir>/score.json` には同じ内容が `total` と `tasks` に入る。項目の意味は [eval.md](../eval.md) の
   Scoring にある。

   ```text
   {"n": 350, "missing": 0, "success": ..., "first_try": ..., ..., "mean_task_success": ...}
     cube_lifting                        .../50  first_try ...  turns ...  finish_unsolved ...  no_finish ...  code_no_kw ...
   ```

   学習前と学習後は、同じ seed の `success` と `mean_task_success` で比べる。

   `run.sh` が失敗した時や時間切れの時は、ジョブが採点の前に終わり、`score.json` はできない。その時や、spec を分けて
   流した run をまとめる時は、ログインノードで採点する。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   .venv/bin/geniac-baseline eval score runs/<run> [runs/<別の run> ...] --json runs/<run>/score.json
   ```

2. `missing` が 0 でない時(CaP-X が trial フォルダを残さずに落ちた seed)は、その seed だけを流し直す(refill)。
   `missing_seeds` に出た seed だけを `seeds` に書き、`output_dir` を別にした spec で「評価する」の 2〜4 番を行い、
   元の run と一緒に採点する。

   ```text
     cube_lifting                         .../50  ...  missing 1 [403]
   ```

   ```yaml
   output_dir: ../runs/eval_base_refill
   tasks:
     - task: cube_lifting
       config: capx/geniac_prize/robosuite/cube_lifting.yaml
       seeds: ["403-403"]
   ```

   ```bash
   .venv/bin/geniac-baseline eval score runs/eval_base runs/eval_base_refill --json runs/eval_base/score.json
   ```

   同じ seed が 2 つの run の両方にあるとエラーになる。refill でも結果が残らない seed は失敗として数える
   (ベースラインの数字と同じ扱い)。

次: [10. Hugging Face へのアップロード](10_upload.md)。[Tutorial](README.md) に戻る。
