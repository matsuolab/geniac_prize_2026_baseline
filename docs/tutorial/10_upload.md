# 10. Hugging Face へのアップロード

[Tutorial](README.md) | 前: [9. ローカル評価の実行](09_eval.md)

リーダーボードの評価を受けるには、提出 image(`.sif`)と `submission.yaml` を、チームの Hugging Face のリポジトリに
push する。運営チームがそれを見つけ、gateway 越しに image を走らせて採点する。

image の作り方、手元での評価、提出の手順は、すべて
[geniac_prize_2026_code](https://github.com/matsuolab/geniac_prize_2026_code) にあるので、そちらに従う。
この章には、どの文書の何を読めばよいかと、参照実装 CaP-Agent0 を提出する手順と、それに 8 章の adapter を載せて
提出する手順を書く。

## geniac_prize_2026_code を clone する

```bash
cd <共用ストレージ>/<ユーザー名>
git clone git@github.com:matsuolab/geniac_prize_2026_code.git
```

[4 章](04_install.md)の `bash patches/geniac_prize/setup.sh` が作る `geniac_prize_2026_baseline/geniac_prize_2026_code/` も
同じリポジトリだが、[7 章](07_collect.md)〜[9 章](09_eval.md)のために決まった commit に patch を当てた物で、
提出には使わない([setup.md](../setup.md))。提出には、ここで `<共用ストレージ>/<ユーザー名>/` の直下に clone した物を使う。

## 読む文書

| やること | geniac_prize_2026_code の文書 |
| --- | --- |
| ルール、評価のしくみ、提出物の条件を知る | [PRELIM_RULEBOOK.md](https://github.com/matsuolab/geniac_prize_2026_code/blob/main/PRELIM_RULEBOOK.md)(実行環境は §5、提出は §6、使えるモデルは §7) |
| 手元で評価する | [docs/submission.md](https://github.com/matsuolab/geniac_prize_2026_code/blob/main/docs/submission.md) の 1 |
| image を作る | 同 2。|
| 提出用リポジトリを作り、登録する | 同 3(ルールブック §6.1) |
| 提出する | 同 4(ルールブック §6.2〜6.5) |

## 参照実装 CaP-Agent0 を提出する

参照実装の image を作り、チームの提出用リポジトリに push するまでの手順である。ルールブック §6.6 の「参照実装での
評価を少なくとも 1 回動かす」は、この提出で行える。チームの org と提出用リポジトリ(`<提出用リポジトリ>`)は作成済み、
登録(ルールブック §6.1)も済んでいるものとする。評価中に使うモデル(Qwen3.6-27B、SAM3 など。
`singularity/capx-submission-agent0.yaml` の `weights`)は、提出の前に申告する(ルールブック §7.2)。

1. image が使う submodule を取る(`docs/operations.md` の「2. 準備」の「イメージ」の表の `submission-agent0` の行)。

2. 計算ノードで image を build する。build はログインノードでは行えない(`singularity/build.sh` が止める)。
   ログは `capx-sif-build.<jobid>.log` に出て、最後に `[build] done: submission-agent0` が出れば、
   `singularity/images/capx-submission-agent0.sif`(約 10 GB)ができている。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_code
   sbatch -p <パーティション> singularity/build.sbatch submission-agent0
   ```

   <details>
   <summary>エラーが出る場合</summary>

   ログに `module: command not found` が出て止まった時は、次のジョブで singularity を先に読み込んでから
   `build.sh` を呼ぶ。ジョブスクリプトは geniac_prize_2026_code の外に置く。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   cat > build_agent0_job.sh <<'EOF'
   #!/usr/bin/env bash
   #SBATCH --job-name=capx-sif-build
   #SBATCH --partition=<パーティション>
   #SBATCH --cpus-per-task=16
   #SBATCH --mem=64G
   #SBATCH --time=<時間>
   #SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
   set -euo pipefail
   if ! command -v singularity > /dev/null; then
       set +u   # lmod は未定義の変数を読むため、読み込む間だけ set +u
       source /opt/share/apps/lmod/lmod/init/bash
       set -u
       export MODULEPATH=/opt/share/modules/modulefiles/latest/Linux:/opt/share/modules/modulefiles/latest/Core:/opt/share/apps/lmod/lmod/modulefiles/Core
       module load singularitypro/4.1
   fi
   bash <共用ストレージ>/<ユーザー名>/geniac_prize_2026_code/singularity/build.sh submission-agent0
   EOF
   sbatch build_agent0_job.sh
   ```

   </details>

3. ログインノードで Hugging Face にログインする。トークンは提出用リポジトリに書き込める物を使う。トークンはホームに
   保存されるので、次のジョブからも使える。

   ```bash
   <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/.venv/bin/hf auth login
   ```

4. image を push するジョブを流す。1 GB を超えるアップロードはログインノードでは行わない(ルールブック §10.3)。
   中身は `docs/submission.md` の 4 の 1 番目のコマンドと同じで、image を `agent.sif` の名前で置き、
   その commit の SHA をログに出す。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   cat > push_image_job.sh <<'EOF'
   #!/usr/bin/env bash
   #SBATCH --job-name=push-image
   #SBATCH --partition=<パーティション>
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --time=<時間>
   #SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
   set -euo pipefail
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_code
   <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline/.venv/bin/python - <<'PY'
   from huggingface_hub import HfApi
   info = HfApi().upload_file(
       path_or_fileobj="singularity/images/capx-submission-agent0.sif",
       path_in_repo="agent.sif",
       repo_id="<提出用リポジトリ>",
       commit_message="agent0 image",
   )
   print("image revision:", info.oid)
   PY
   EOF
   sbatch push_image_job.sh
   ```

   ジョブが終わると、ログの最後に次の行が出る。この 40 桁を次の `submission.yaml` に書く。

   ```text
   image revision: <40 桁の commit の SHA>
   ```

5. `submission.yaml` を作る。参照実装の設定 `singularity/capx-submission-agent0.yaml` に `weights` と `env` が
   書いてあるので、コピーして使う。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   mkdir -p submission
   cp geniac_prize_2026_code/singularity/capx-submission-agent0.yaml submission/submission.yaml
   ```

   `submission/submission.yaml` の `team_id` と `submission_id` を書き換え、`image` を足す(`weights` と `env` は
   そのまま)。`team_id` と `submission_id` はリーダーボードで他のチームにも見える(ルールブック §6.1)。

   ```yaml
   team_id: <team_id>
   submission_id: agent0
   image:
     repo: <提出用リポジトリ>
     revision: <4 番のログの SHA>
     path: agent.sif
   ```

6. `submission.yaml` を提出用リポジトリの `main` に push する。小さいファイルなので、ログインノードで行ってよい。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   geniac_prize_2026_baseline/.venv/bin/hf upload <提出用リポジトリ> submission/submission.yaml submission.yaml --commit-message "agent0"
   ```

   運営は `main` の `submission.yaml` だけを見る。評価されるのは中身が前の提出と違う時だけで、同じ image を
   もう一度評価してほしい時は `submission_id` を変える(ルールブック §6.4)。結果はリーダーボードに出る。

## adapter 込みのモデルを使う場合

CaP-Agent0 の構成のまま、[8 章](08_train.md)で学習した adapter(`runs/train_dmpo/adapter/`)を Qwen3.8-27B に
載せて提出する手順である。参照実装の image は `CAPX_ADAPTER_PATH` を読まないので、`submission.yaml` に adapter を
書くだけでは載らない(エラーにならず、学習前のモデルのまま評価される)。起動スクリプトに adapter を読む処理を足し、
前の節の 2 番で作った image の上に重ねた image を作る。geniac_prize_2026_code のファイルは書き換えず、コピーを
直す。前の節の 1〜3 番(submodule、image の build、Hugging Face へのログイン)を済ませていること。

1. adapter をチームの org の Hugging Face のリポジトリ(`<adapter のリポジトリ>`)に push する。大きさによっては 1 GB を
   超えるので、計算ノードのジョブで行う。

   ```bash
   cd <共用ストレージ>/<ユーザー名>
   cat > push_adapter_job.sh <<'EOF'
   #!/usr/bin/env bash
   #SBATCH --job-name=push-adapter
   #SBATCH --partition=<パーティション>
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --time=<時間>
   #SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
   set -euo pipefail
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   .venv/bin/python - <<'PY'
   from huggingface_hub import HfApi
   api = HfApi()
   api.create_repo("<adapter のリポジトリ>", private=True, exist_ok=True)
   info = api.upload_folder(
       folder_path="runs/train_dmpo/adapter",
       repo_id="<adapter のリポジトリ>",
       commit_message="DMPO adapter",
   )
   print("adapter revision:", info.oid)
   PY
   EOF
   sbatch push_adapter_job.sh
   ```

   ログの最後の `adapter revision: <40 桁の commit の SHA>` を、6 番の `submission.yaml` に書く。

2. 起動スクリプトをコピーし、adapter を読むように直す。作業用のディレクトリ `agent0-lora/` を作り、そこで行う。

   ```bash
   mkdir -p <共用ストレージ>/<ユーザー名>/agent0-lora
   cd <共用ストレージ>/<ユーザー名>/agent0-lora
   cp ../geniac_prize_2026_code/singularity/agent0-entrypoint.sh .
   ```

   直すのは次の 3 か所である。

   | 場所 | 変更 |
   | --- | --- |
   | `MODEL_PATH=` の行の後 | `CAPX_ADAPTER_PATH` があれば、vLLM に `--enable-lora --max-lora-rank <adapter の r> --max-loras 1 --lora-modules dmpo-lora=<adapter>` を付け、agent が使うモデル名を `dmpo-lora` にする。adapter は bf16 の重みで学習したので、この時は `--quantization fp8` を外す |
   | vLLM を起動する行 | `--quantization fp8` を上の設定に置き換え、LoRA の引数を足す |
   | `python -m capx.submission.run` の行 | `--model` を、agent が使うモデル名(adapter があれば `dmpo-lora`)にする |

   `CAPX_ADAPTER_PATH` が無い時は元のスクリプトと同じに動く。次の patch で直せる。

   ```bash
   patch agent0-entrypoint.sh <<'EOF'
   --- agent0-entrypoint.sh
   +++ agent0-entrypoint.sh
   @@ -31,6 +31,19 @@
    
    MODEL="${CAPX_MODEL:-Qwen/Qwen3.6-27B}"
    MODEL_PATH="${CAPX_MODEL_PATH:-${MODEL}}"
   +# LoRA adapter (CAPX_ADAPTER_PATH): served under CAPX_LORA_NAME, the name the agent asks for.
   +# The adapter was trained on bf16 weights, so the base is not quantized to fp8 then.
   +ADAPTER_PATH="${CAPX_ADAPTER_PATH:-}"
   +LORA_NAME="${CAPX_LORA_NAME:-dmpo-lora}"
   +AGENT_MODEL="${MODEL}"
   +QUANT_ARGS=(--quantization fp8)
   +LORA_ARGS=()
   +if [[ -n "${ADAPTER_PATH}" ]]; then
   +    LORA_RANK="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["r"])' "${ADAPTER_PATH}/adapter_config.json")"
   +    QUANT_ARGS=()
   +    LORA_ARGS=(--enable-lora --max-lora-rank "${LORA_RANK}" --max-loras 1 --lora-modules "${LORA_NAME}=${ADAPTER_PATH}")
   +    AGENT_MODEL="${LORA_NAME}"
   +fi
    RUNNER_CONFIG="${CAPX_RUNNER_CONFIG:-env_configs/gateway/runner.yaml}"
    [[ "${RUNNER_CONFIG}" = /* ]] || RUNNER_CONFIG="/workspace/capx/${RUNNER_CONFIG}"
    PORT="${CAPX_SERVE_PORT:-8110}"
   @@ -89,6 +102,7 @@
    export CAPX_MOLMO_MODEL="${CAPX_MOLMO_MODEL:-${MODEL}}"
    
    echo "[agent0] serving ${MODEL} from ${MODEL_PATH} on 127.0.0.1:${PORT}"
   +echo "[agent0] agent model: ${AGENT_MODEL} (adapter: ${ADAPTER_PATH:-none})"
    echo "[agent0] fallback localizer -> ${CAPX_MOLMO_URL} (${CAPX_MOLMO_MODEL})"
    # fp8 on the fly keeps a 27B inside a single card's budget alongside the sim's
    # rendering context; it needs compute capability >= 8.9 (Ada/Hopper/Blackwell).
   @@ -97,12 +111,13 @@
        --served-model-name "${MODEL}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
   -    --quantization fp8 \
   +    "${QUANT_ARGS[@]}" \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization "${GPU_UTIL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        ${EAGER_FLAG} \
   +    "${LORA_ARGS[@]}" \
        --reasoning-parser qwen3 \
        ${SERVE_EXTRA_ARGS} > "${LOGS}/vllm.log" 2>&1 &
    SERVE_PID=$!
   @@ -144,7 +159,7 @@
    # are the ones served above; flags passed to the container come after and win.
    python -m capx.submission.run \
        --config-path "${RUNNER_CONFIG}" \
   -    --model "${MODEL}" \
   +    --model "${AGENT_MODEL}" \
        --server-url "http://127.0.0.1:${PORT}/v1/chat/completions" \
        --output-dir "${OUT}" "$@" &
    RUNNER_PID=$!
   EOF
   bash -n agent0-entrypoint.sh && echo ok
   ```

   `patching file agent0-entrypoint.sh` の後に `ok` が出れば良い。`Hunk ... FAILED` が出た時は、geniac_prize_2026_code の
   起動スクリプトが変わっているので、上の表を見て手で直す。

3. 前の節の 2 番の image に、直した起動スクリプトを差し替え、学習時の skill library と task の prompt を入れる
   def ファイルを作る(`submission/install.sh` が行う。中身はその先頭のコメントにある)。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/agent0-lora
   cp -R ../geniac_prize_2026_baseline/submission .
   cp ../geniac_prize_2026_baseline/patches/geniac_prize/skill-library.patch submission/
   cat > capx-submission-agent0-lora.def <<'EOF'
   Bootstrap: localimage
   From: <共用ストレージ>/<ユーザー名>/geniac_prize_2026_code/singularity/images/capx-submission-agent0.sif

   %files
       agent0-entrypoint.sh /workspace/capx/entrypoint.sh
       submission /opt/baseline-submission

   %post
       chmod a+rx /workspace/capx/entrypoint.sh
       bash /opt/baseline-submission/install.sh /workspace/capx

   %runscript
       exec /workspace/capx/entrypoint.sh "$@"

   %labels
       capx.role submission-agent0-lora
   EOF
   ```

   `From:` には、前の節の 2 番でできた `.sif` の絶対パスを書く。

4. 計算ノードで image を build する。`singularity/build.sh` は geniac_prize_2026_code の中の git で管理された
   ファイルしか image に入れないので、使わずに `singularity build` を直接呼ぶ。中身は 1 つのファイルの差し替えだけなので、
   数分で終わる。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/agent0-lora
   cat > build_lora_job.sh <<'EOF'
   #!/usr/bin/env bash
   #SBATCH --job-name=build-agent0-lora
   #SBATCH --partition=<パーティション>
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --time=<時間>
   #SBATCH --output=<共用ストレージ>/<ユーザー名>/logs/%x-%j.out
   set -euo pipefail
   if ! command -v singularity > /dev/null; then
       set +u   # 計算ノードのジョブには module が無いので、lmod を読み込む(未定義の変数を読むため set +u)
       source /opt/share/apps/lmod/lmod/init/bash
       set -u
       export MODULEPATH=/opt/share/modules/modulefiles/latest/Linux:/opt/share/modules/modulefiles/latest/Core:/opt/share/apps/lmod/lmod/modulefiles/Core
       module load singularitypro/4.1
   fi
   export SINGULARITY_TMPDIR="/var/tmp/${USER}/sif-tmp"
   mkdir -p "${SINGULARITY_TMPDIR}"
   cd <共用ストレージ>/<ユーザー名>/agent0-lora
   singularity build --fakeroot --force capx-submission-agent0-lora.sif capx-submission-agent0-lora.def
   ls -lh capx-submission-agent0-lora.sif
   EOF
   sbatch build_lora_job.sh
   ```

5. 前の節の 4 番のジョブスクリプトで、`path_or_fileobj` を
   `"<共用ストレージ>/<ユーザー名>/agent0-lora/capx-submission-agent0-lora.sif"` に、`commit_message` を
   `"agent0 + DMPO adapter image"` に変えて流す。ログの `image revision:` の SHA を次の 6 番に書く。

6. `submission.yaml` を作る。前の節の 5 番の物と比べて、モデルを Qwen3.8-27B に替え、adapter を `weights` に足し、
   `env` で `CAPX_ADAPTER_PATH` を渡す。`CAPX_MODEL` は vLLM がベースのモデルを配信する名前で、agent は
   adapter を載せた `dmpo-lora` の名前で問い合わせる。

   ```yaml
   team_id: <team_id>
   submission_id: agent0-dmpo
   image:
     repo: <提出用リポジトリ>
     revision: <5 番のログの SHA>
     path: agent.sif
   weights:
     - repo: Qwen/Qwen3.8-27B
       revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
       path: /models/qwen3.8-27b
     - repo: <adapter のリポジトリ>
       revision: <1 番のログの SHA>
       path: /models/adapter
     - repo: facebook/sam3
       revision: 3c879f39826c281e95690f02c7821c4de09afae7
       path: /models/sam3
       gated: true
     - repo: google/owlv2-large-patch14-ensemble
       revision: 95e26936e865f87db1742128404b3c035d47d89d
       path: /models/owlv2-large-patch14-ensemble
     - repo: facebook/sam2.1-hiera-large
       revision: 665f8e2ad61cf5f53d65644ff27c8ee525124610
       path: /models/sam2.1-hiera-large
   env:
     CAPX_MODEL: Qwen/Qwen3.8-27B
     CAPX_MODEL_PATH: /models/qwen3.8-27b
     CAPX_ADAPTER_PATH: /models/adapter
     CAPX_SAM3_CHECKPOINT: /models/sam3/sam3.pt
     CAPX_VLM_URL: http://127.0.0.1:8110/v1    # LIBERO の skill が物の位置を聞く VLM(CAPX_MODEL と同じモデル)
     CAPX_VLM_MODEL: Qwen/Qwen3.8-27B
   ```

   - adapter のリポジトリは、`weights` に書くモデルとして提出の前に申告する(ルールブック §7.2)。

7. 前の節の 6 番と同じように `submission.yaml` を提出用リポジトリに push する。

なお、adapter は geniac_prize_2026_baseline の multi-turn のプロンプト(コード生成の
thinking を切った物)で学習したが、CaP-Agent0 は 1 回の問い合わせで 9 個の候補を作って合成するプロンプトで、
thinking も切っていない。そのため、7 章〜9 章で測った adapter の効果が、そのまま出るとは限らない。

これでチュートリアルは以上である。
今後このベースラインコードを用いて開発をする際には、[Documentation](https://github.com/matsuolab/geniac_prize_2026_baseline/tree/main#documentation) のセクションを参照して追加の実装をされたい。
