# 8. 学習実行

[Tutorial](README.md) | 前: [7. データ収集](07_collect.md) | 次: [9. ローカル評価の実行](09_eval.md)

Qwen3.8-27B の LoRA adapter を、SFT、DMPO の順に学習する。

この章のコマンドは `<共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline` で実行する。学習はジョブとして計算ノードで
流す。

## 準備

4 章で学習用のパッケージ(`[train]`)を、[7 章](07_collect.md)でモデルの重み(Qwen/Qwen3.8-27B、
`<モデルのスナップショット>`)と学習データを用意してあるものとする。学習データは収集と変換
([collect.md](../collect.md)、[data_format.md](../data_format.md))の出力で、次の 2 つのディレクトリにある。

| ディレクトリ | ファイル | 使う段階 |
| --- | --- | --- |
| `runs/data/split/` | `sft.jsonl`、`val.jsonl`、`manifest.json`(成功した trial) | SFT |
| `runs/data/pairs/` | `pairs.jsonl`、`rows.jsonl`、`manifest.json`(成功と失敗の trial の pair) | DMPO |

学習のジョブスクリプトを作る。SFT と DMPO で共用する。

```bash
mkdir -p runs
cat > runs/train.sh <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --partition=<パーティション>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=<時間>
#SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
set -euo pipefail
cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
source .venv/bin/activate
export HF_HUB_OFFLINE=1
geniac-baseline train "$STAGE" --config "$CONFIG"
EOF
```

ベースラインのデータでは、B300 1 枚で SFT に約 62 h、DMPO に約 80 h かかる。学習時 checkpoint(`<output_dir>/ckpt-last`)
を SFT は `save_every` step ごとに、DMPO は毎 step 書く。そのため、ジョブが途中で止まった時は、config(`runs/train_sft.yaml` か
`runs/train_dmpo.yaml`)の `resume:` に `<output_dir>/ckpt-last` を書いて投入し直すと、続きから学習する。

また、学習を自分で複数回のジョブに分ける時は、config の `stop_after_min` に 1 回のジョブの分数を書く。その時間に達すると
checkpoint を書いて止まるので、上と同じように `resume:` を書いて投入し直すことができる。

学習が最後まで終わったら、`resume:` を `null` に戻す。書いたまま別の `output_dir` で新しく学習を始めると、その
checkpoint から始まってしまう。

## SFT

1. config をコピーし、次の項目を書き換える。

   ```bash
   cp configs/train_sft.yaml runs/train_sft.yaml
   ```

   ```yaml
   data: <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/runs/data/split
   model: <モデルのスナップショット>
   output_dir: <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/runs/train_sft

   wandb:
     enabled: true
     project: geniac-baseline
     name: sft
     mode: online
   ```

2. 投入する。

   ```bash
   sbatch --export=ALL,STAGE=sft,CONFIG=runs/train_sft.yaml runs/train.sh
   ```

   ベースラインのデータ(train 3711 行)では B300 1 枚で約 62 h かかる。ログは `<共用ストレージ>/<ユーザー名>/logs/train-<jobid>.out`、loss などは W&B の
   `geniac-baseline` project で見られる。終わると `runs/train_sft/` に epoch ごとの adapter と、val の loss が最小の epoch を書いた
   `best_epoch.json` ができる。

## DMPO

1. config をコピーし、次の項目を書き換える。

   ```bash
   cp configs/train_dmpo.yaml runs/train_dmpo.yaml
   ```

   ```yaml
   data: <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/runs/data/pairs
   model: <モデルのスナップショット>
   init_adapter: <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/runs/train_sft
   output_dir: <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/runs/train_dmpo

   wandb:
     enabled: true
     project: geniac-baseline
     name: dmpo
     mode: online
   ```

2. 投入する。

   ```bash
   sbatch --export=ALL,STAGE=dmpo,CONFIG=runs/train_dmpo.yaml runs/train.sh
   ```

   ベースラインのデータ(train 521 pair)では約 80 h かかる。終わると `runs/train_dmpo/adapter/` ができ、これを評価に使う。

   `init_adapter` は必須で、SFT を通さずに DMPO だけを学習することはできない。SFT が終わるのを待たずに投入する時は、
   `sbatch` の `--dependency` オプションを使う。`afterok:<SFT の jobid>` は「そのジョブが成功して終わってから始める」
   という指定で、SFT が失敗した時は DMPO は始まらない。

   ```bash
   sbatch --dependency=afterok:<SFT の jobid> --export=ALL,STAGE=dmpo,CONFIG=runs/train_dmpo.yaml runs/train.sh
   ```

## W&B で確認できるもの

W&B の `geniac-baseline` project に、run(`sft`、`dmpo`)ごとに値が step ごとに記録される。主な値は次のとおりである。

| 値 | 中身 |
| --- | --- |
| `train/loss` | 学習の loss(SFT は回答 token の cross-entropy、DMPO は DMPO の loss) |
| `train/lr`、`train/grad_norm` | 学習率、勾配のノルム |
| `val/loss_all` | SFT の val の loss。評価と DMPO に使う epoch は、これが最小の物 |
| `val/decision_acc` | SFT の val で、`REGENERATE` と `FINISH` の選択が正しかった割合 |
| `val/pref_acc`、`val/margin` | DMPO の val の pair で、成功した trial を高く評価できた割合と、その差 |
| `perf/sec`、`perf/max_mem_gb` | 1 step の秒数、GPU memory の最大使用量 |

run の Overview には学習の config、Logs にはジョブのログと同じ出力がある。

次: [9. ローカル評価の実行](09_eval.md)。[Tutorial](README.md) に戻る。
