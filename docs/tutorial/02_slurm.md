# 2. 基本的な Slurm コマンドの確認

[Tutorial](README.md) | 前: [1. GMO クラウドへのログイン](01_login.md) | 次: [3. 開発コードのクローン](03_clone.md)

前章ではログインノードへの接続まで完了した。
この章では、計算ノードを使うための Slurm のコマンドをいくつか紹介する。

| command | 用途 |
| --- | --- |
| `snodes` | ノードとパーティションの状態を見る |
| `squeue -u $USER` | 自分のジョブの一覧と状態(`PD` は待ち、`R` は実行中)を見る |
| `srun ... --pty bash` | 計算ノードに対話的に入る |
| `sbatch job.sh` | ジョブスクリプトを投入する。ジョブ ID が表示される |
| `scancel <jobid>` | ジョブを止める |
| `sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode` | 終わったジョブの結果を見る |

1. `snodes`コマンドを用いると使用可能なパーティション名を確認できる。今後はこのパーティション名を使っていく。

   ```bash
   snodes
   ```

2. 計算ノードに対話的に入り、`hostname` がログインノードのもの([1 章](01_login.md)で確かめた物)から変わった
   ことを確かめる。終わったら `exit` で抜ける(抜けるまで計算ノードを確保し続けるため注意)。

   ```bash
   srun --partition=<パーティション> --pty bash
   hostname
   exit
   ```

   GPU や CPU、時間を指定する時は `srun` にオプションを足す。GPU は `--gres=gpu:1`、
   CPU は `--cpus-per-task=16`、時間の上限は `--time=01:00:00` のように書く。
   その他のオプションについては適宜マニュアルを参照されたい。

3. 次に、実験として計算ノードのホスト名を書き出すだけのジョブを流す。
   チームの共用ストレージに自分用のディレクトリを作り、ログもそこに書く(以降の章もこのディレクトリで作業する)。

   ```bash
   mkdir -p <共用ストレージ>/<ユーザー名>/logs
   cd <共用ストレージ>/<ユーザー名>
   cat > hello.sh <<'EOF'
   #!/usr/bin/env bash
   #SBATCH --job-name=hello
   #SBATCH --partition=<パーティション>
   #SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
   set -euo pipefail
   hostname
   date
   EOF
   sbatch hello.sh
   squeue -u $USER
   ```

   終わったら `logs/hello-<jobid>.out` に計算ノードのホスト名と実行された時刻が書き出される。
   `srun` と同じく、オプションを `#SBATCH --gres=gpu:1` のように書くと、GPU を確保できる。

次: [3. 開発コードのクローン](03_clone.md)
