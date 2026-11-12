# 6. W&B 連携

[Tutorial](README.md) | 前: [5. テスト実行](05_test.md) | 次: [7. データ収集](07_collect.md)

学習の loss などを Weights & Biases(W&B)で見られるように、W&B にログインし、計算ノードから記録が届くことを
確かめる。学習で使う設定は [8 章](08_train.md)で行う。`<...>` の意味は [Tutorial](README.md#placeholders) の表にある。

1. W&B のアカウントを用意する。

2. ログインノードで wandb を入れ、ログインする。`wandb login` が API key を聞いてくるので、W&B の User settings に
   ある key を貼り付ける。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   source .venv/bin/activate
   uv pip install -e ".[wandb]"
   wandb login
   ```

   key はホームディレクトリの `~/.netrc` に保存される。

3. テストとして、計算ノードから `x` に 0 から 9 を記録するジョブを流してみる。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   cat > wandb_ping.sh <<'EOF'
   #!/usr/bin/env bash
   #SBATCH --job-name=wandb-ping
   #SBATCH --partition=<パーティション>
   #SBATCH --mem=4G
   #SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
   set -euo pipefail
   source <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/.venv/bin/activate
   python - <<'PY'
   import wandb
   run = wandb.init(project="geniac-baseline", name="ping", dir="<共用ストレージ>/<ユーザー名>/logs")
   for i in range(0, 10):
       run.log({"x": i})
   print("URL:", run.url)
   run.finish()
   PY
   EOF
   sbatch wandb_ping.sh
   ```

   `<共用ストレージ>/<ユーザー名>/logs/wandb-ping-<jobid>.out` の最後の `URL: https://...` を開くと、`x` が0から9までふえるグラフが見られる。
   アカウントによっては、URL が `https://wandb.ai/...` ではなく `https://forge.coreweave.com/wandb/...` になる(W&B の
   ログイン先が `api.forge.coreweave.com` の場合)。どちらでも記録は同じように届く。

次: [7. データ収集](07_collect.md)
