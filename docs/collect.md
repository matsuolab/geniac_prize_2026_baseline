# Data collection

1. run spec を選ぶ。Robosuite は `configs/collect.yaml`(7 task、seed 301-400)、LIBERO-PRO は
   `configs/collect_libero.yaml`(30 task、layout 0-39)。tree と venv の違いのため suite ごとに分けてある。
   タスクの行は次の形である。

   ```yaml
   tasks:
     - task: cube_lifting
       config: capx/geniac_prize/robosuite/cube_lifting.yaml   # must set save_multiturn_prompts: true
       seeds: ["301-400"]
     - task: libero_spatial_on_stove
       config: capx/geniac_prize/libero/libero_spatial_on_stove.yaml
       layouts: 50                                           # LIBERO-PRO only, required
       seeds: ["301-340", "351-390"]
   ```

2. `geniac-baseline collect plan --spec configs/collect.yaml` を実行する。spec を点検し、`output_dir` に `run.sh`、
   seed 区間ごとの config のコピー、`sources.yaml`、`plan.json` を書き出す。この時点では何も実行しない。
3. [GPU] GPU ホストで `bash <output_dir>/run.sh` を実行する。モデルサーバーを 1 つずつ起動し、seed 区間ごとに CaP-X の
   `envs/launch.py` を呼び、終了時にサーバーを止める。失敗した区間があれば、最後に一覧を出して exit 1 で終わる。
4. trial を行データに変換する。成功した trial は SFT に、失敗した trial は DMPO の pair に使う。

   ```bash
   geniac-baseline convert rows --sources runs/collect/sources.yaml --out runs/data/rows
   geniac-baseline convert rows --sources runs/collect/sources.yaml --include failure --out runs/data/fail_rows
   geniac-baseline convert tag --src runs/data/rows --out runs/data/tagged --tokenizer <model snapshot>
   geniac-baseline convert tag --src runs/data/fail_rows --out runs/data/fail_tagged --tokenizer <model snapshot>
   geniac-baseline convert split --src runs/data/tagged --out runs/data/split
   ```

`examples/00_collect_data.sh` は手順 2 から 4 をまとめて実行する(`SPEC=configs/collect_libero.yaml` で LIBERO-PRO)。

データは、提出するときと同じ skill library で集める。学習したモデルは、集めた時の skill の名前と使い方でコードを書くので、
提出時に無い skill を呼ぶと失敗する。Robosuite はどちらの環境でも skill library が同じなので、主に LIBERO-PRO の話である。
ベースラインの LIBERO-PRO のデータは ASPIRE の追加 skill を呼ぶので、`patches/geniac_prize/setup.sh` はこの skill を
コンペの CaP-X にも足す([setup.md](setup.md) の Skill library)。提出する image にも同じ skill library を入れる。

## Run spec paths

run spec の相対 path は、key によって基準が違う。

| key | 基準 |
| --- | --- |
| `capx_root`、`output_dir`、`chat_template`、`adapter`、`tasks[].config` | spec file のある directory |
| `launch_script`、`sim_python`、`serve_python` | `capx_root`(絶対 path を書けばそのまま使う) |
| `pythonpath` | 変換しない。絶対 path で書く |

## Preflight checks

`collect plan` は次の場合に計画を書き出さずに止まる。

| Condition | Reason |
| --- | --- |
| モデルが checkout の `VLM_MODELS` に無い | CaP-X が画像と VDM テキストを黙って使わなくなる |
| タスクの config に `save_multiturn_prompts: true` が無い | 判断プロンプトが保存されず、学習データにならない |
| seed が 301-400 の外、または LIBERO-PRO の seed が評価用 layout(40-49)にあたる | 評価用の seed と layout を学習データに入れない |
| LIBERO-PRO のタスクに `layouts` が無い | seed の layout が決まらない |
| config のポートが CaP-X のクライアントの既定値と合わない、または `port_offset` を checkout が扱えない | クライアントが別のサーバーに接続する |
| `chat_template` が無い(`allow_stock_chat_template: true` で解除) | ベースラインと異なるテンプレートで動く |

## LIBERO-PRO layouts

CaP-X は LIBERO の trial を `init_states[(seed - 1) % layouts]` から始める。コンペの CaP-X はその状態を
そのまま使うので、seed `s` と `s + layouts` は同じシーンになる。そのため学習と評価は seed ではなく layout で分ける。
ASPIRE は状態を設定した後にもう一度 reset するため、この重複は起きない([reproduce_baseline.md](reproduce_baseline.md))。ベースラインの
学習データには seed 401 以上と layout 40-49 の trial が無いことを確認した。

`layouts` はタスクの init state の数である。ベースラインで使った LIBERO-PRO の suite(`libero_spatial_swap`、
`libero_object_swap`、`libero_goal_with_mug`)はいずれも 50 である。シミュレータのホストで次のように確認する
(suite 名と task id はタスクの config のもの)。

```bash
# the competition CaP-X (from the geniac_prize_2026_code root)
.venv-libero/bin/python -c "from capx.integrations.libero import load_libero_task as L; \
print(len(L('libero_object_swap', 0).init_states))"
# ASPIRE: the same with `from aspire.sim.cap.integrations.libero import load_libero_task`, PYTHONPATH=<ASPIRE root>
```

## Default settings

`provenance` 列は、ASPIRE の収集 script(ベースライン)にある値を run、自分たちで選んだ値を ours とした。

| Setting | Value | provenance |
| --- | --- | --- |
| model | Qwen/Qwen3.8-27B。コード生成と VDM を 1 つの vLLM サーバーで扱う | run |
| serving | bf16、`--reasoning-parser qwen3`、gpu-memory-utilization 0.62 | run |
| max-model-len 262144、max-num-seqs 8 | 1 trial の会話全体が入る長さ | ours |
| reasoning effort | chat template の既定を `low` にした(`geniac-baseline template`)。`--reasoning-effort` は CaP-X が GPT 系にしか送らないため、Qwen には届かない | ours |
| max-tokens | 12288 | ours |
| temperature | 1.0。Robosuite の config は parallel ensemble を使い、その中で温度を固定するため効かない。LIBERO-PRO は 1 回の呼び出しなので効く | run |
| pointing model | allenai/Molmo2-8B(bf16、max-model-len 16384、gpu-memory-utilization 0.28) | ours |
| trial の実時間上限 | 2400 s(`trial_timeout_s`)。ASPIRE の既定値 | run |
| 画像 | LIBERO-PRO 800×512、Robosuite 512×512(ベースラインの trial の保存画像で確認) | run |
| B300 用の設定 | `b300_workarounds: true` で `--attention-backend TRITON_ATTN --mm-encoder-attn-backend TORCH_SDPA`、`TORCH_CUDA_ARCH_LIST=10.0`。既定は off | run |

GPU の割り当ては run spec の `qwen_gpu_util`、`molmo_gpu_util`、`max_model_len`、`tensor_parallel_size` で変える。

task config(`configs/capx/aspire/`)はベースラインが使った物で、ASPIRE の f4c8939 の config から次を変えてある。
Robosuite: `use_multimodel: false`、`num_workers: 8`。LIBERO-PRO: `use_img_differencing: true`、`api_servers`。
両方: `save_multiturn_prompts: true`。LIBERO-PRO の `max_steps` は 4000 である。
