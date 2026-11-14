# 4. 依存関係のインストール

[Tutorial](README.md) | 前: [3. 開発コードのクローン](03_clone.md) | 次: [5. テスト実行](05_test.md)

ベースラインリポジトリの Python 環境と、収集と評価で呼ぶコンペの CaP-X(構成は
[installation.md](../installation.md) の図)を用意する。作業は [3 章](03_clone.md)で clone した
`<共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline` で行う。`<...>` の意味は [Tutorial](README.md#placeholders) の表に
ある。

作る環境は次の 4 つである。名前と場所は `configs/` の既定に合わせてあるので変えない。Robosuite と LIBERO-PRO はバージョンの
違う robosuite が要るため、CaP-X の venv を分ける。

| 環境(`geniac_prize_2026_baseline/` から) | Python | 中身 | 使う所 |
| --- | --- | --- | --- |
| `.venv` | 3.12 | このパッケージ、テスト、学習(torch、transformers、peft)、W&B、Hugging Face | 5〜10 章 |
| `geniac_prize_2026_code/.venv` | 3.10 | CaP-X、Robosuite 1.5.1 | Robosuite の収集と評価 |
| `geniac_prize_2026_code/.venv-libero` | 3.12 | CaP-X、LIBERO-PRO、Robosuite 1.4.0 | LIBERO-PRO の収集と評価 |
| `geniac_prize_2026_code/.venv-serve` | 3.12 | vLLM 0.26.0 | Qwen3.8-27B と Molmo2-8B のサーバー |

1. GPU 無しで計算ノードを確保する。5 番まではこの計算ノードで続けて行う。時間切れや接続の切断で計算ノードから
   抜けた時は、同じ `srun` で入り直し、続きの番号から行う。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   srun --partition=<パーティション> --cpus-per-task=4 --mem=8G --time=01:00:00 --pty bash
   ```

   `hostname` が計算ノードの物であることを確かめてから、[uv](https://docs.astral.sh/uv/) を入れる。Python 本体も
   uv が入れるので、クラスタの Python の module は使わない。`uv --version` でバージョンが表示されれば、uv を入れる
   必要はない。

   ```bash
   hostname                      # ログインノードの物なら、上の srun をやり直す
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source "$HOME/.local/bin/env"
   uv --version
   ```

   uv はダウンロードした物を `~/.cache/uv` に置く。ホームディレクトリの容量が小さい場合は、`~/.bashrc` に
   `export UV_CACHE_DIR=<共用ストレージ>/<ユーザー名>/.cache/uv` を書き、`source ~/.bashrc` で読み込む。

2. ベースラインの環境 `.venv` を作る。ダウンロードとインストールだけなので、GPU は要らない。
   `requirements-train.txt` が学習のライブラリをベースラインの学習と同じバージョン(torch 2.9.1、CUDA 13.0)に固定する。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   uv venv --python 3.12
   uv pip install -e ".[dev,train,wandb,hub]" -r requirements-train.txt \
       --extra-index-url https://download.pytorch.org/whl/cu130
   uv pip check                  # "All installed packages are compatible" と出れば良い
   ```

   extra の `dev` はテスト([5 章](05_test.md))、`train` は SFT と DMPO(8 章)、`wandb` は W&B 連携(6 章)、`hub` は
   Hugging Face(7 章と 10 章)に使う。GPU を使わない作業(テスト、変換、計画)だけなら、CI と同じ `uv pip install -e ".[dev]"`
   で足りる。`examples/` の script は `.venv/bin/geniac-baseline` を使うので、`.venv` を activate しなくてよい。

3. コンペの CaP-X を `geniac_prize_2026_baseline/geniac_prize_2026_code` に作る。`patches/geniac_prize/setup.sh` が
   [geniac_prize_2026_code](https://github.com/matsuolab/geniac_prize_2026_code) を clone して決まった commit に合わせ、
   [setup.md](../setup.md) の patch を当て、submodule のうちベースラインで使う 7 つを取る。BEHAVIOR の
   シミュレータは評価の時に別のサーバーで動くので、ここでは入れない。clone は [3 章](03_clone.md)と同じく
   SSH で行う(3 章の 1 で登録した鍵を使う)。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline    # 必ずここで実行する
   bash patches/geniac_prize/setup.sh                 # 途中で、patch で変わったファイルの sha256 が出る
   ```

   配置は次のようになる。`geniac_prize_2026_code/` はベースラインリポジトリの `.gitignore` に入っている。

   ```
   <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/
   ├── .venv/                  # 2 番
   ├── geniac_prize_2026_code/ # コンペの CaP-X(patch 済み)
   │   ├── capx/third_party/   # submodule
   │   └── .venv/  .venv-libero/  .venv-serve/    # 4 番で作る
   └── configs/  docs/  examples/  geniac_baseline/  tests/
   ```

4. CaP-X の venv を 3 つ作る。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/geniac_prize_2026_code
   uv python install 3.10 3.12

   # Robosuite: .venv(Python 3.10)。最後の python で Robosuite の設定ファイルを作る
   uv venv .venv --python 3.10
   source .venv/bin/activate
   uv sync --active --frozen --extra robosuite
   python capx/third_party/robosuite/robosuite/scripts/setup_macros.py
   deactivate

   # LIBERO-PRO: .venv-libero(Python 3.12)。最後の cp で Robosuite の設定ファイルを作る
   uv venv .venv-libero --python 3.12
   source .venv-libero/bin/activate
   uv sync --active --frozen --extra libero --extra contactgraspnet
   cp capx/third_party/libero_dependencies/robosuite/robosuite/{macros.py,macros_private.py}
   deactivate

   # vLLM: .venv-serve(Python 3.12)。--no-config でコンペの CaP-X の transformers<5.0 の制約を読まない
   uv venv .venv-serve --python 3.12
   source .venv-serve/bin/activate
   uv pip install --no-config --torch-backend=cu130 vllm==0.26.0 "transformers>=5" huggingface_hub
   deactivate
   ```

5. LIBERO-PRO の assets の場所を `~/.libero/config.yaml` に書く。このファイルが無いと、LIBERO は最初の import で
   設定を対話的に聞いてくるので、ジョブが止まる。最後の `exit` で計算ノードから抜ける。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/geniac_prize_2026_code
   L="$PWD/capx/third_party/LIBERO-PRO/libero/libero"
   mkdir -p "$HOME/.libero"
   cat > "$HOME/.libero/config.yaml" <<EOF
   assets: $L/assets
   bddl_files: $L/bddl_files
   benchmark_root: $L
   datasets: $L/../datasets
   init_states: $L/init_files
   EOF
   exit
   ```

   ログインノードに戻ったら、このシェルでも `uv` を使えるように PATH を通す。次のログインからは要らない。

   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   uv --version
   ```

6. GPU を 1 枚確保し、4 つの環境のライブラリが読み込めることと、GPU が見えることを確かめる。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   srun --partition=<パーティション> --gres=gpu:1 --cpus-per-task=2 --time=00:10:00 --pty bash
   .venv/bin/python -c "import torch, transformers, peft, accelerate; print('torch', torch.__version__, 'GPU:', torch.cuda.is_available())"
   cd geniac_prize_2026_code
   .venv/bin/python -c "import capx, robosuite; print('robosuite', robosuite.__version__)"
   .venv-libero/bin/python -c "import capx, robosuite; from libero import benchmark; print('robosuite', robosuite.__version__)"
   .venv-serve/bin/python -c "import torch, vllm; print('vllm', vllm.__version__, 'GPU:', torch.cuda.is_available())"
   exit
   ```

   それぞれの出力の最後が `torch 2.9.1+cu130 GPU: True`、`robosuite 1.5.1`、`robosuite 1.4.0`、
   `vllm 0.26.0 GPU: True` になれば良い。Robosuite の `WARNING` は出ても問題ない。

   CaP-X の tree が patch を当てた物かは、収集と評価の script が実行の前に点検する(`RUN_GPU=1` を付けない
   `bash examples/01_eval_base_cap_agent0.sh`)。この点検にはモデルの重みから作る chat template
   ([installation.md](../installation.md) の Model weights)が要るので、重みを用意してから行う。

次: [5. テスト実行](05_test.md)
