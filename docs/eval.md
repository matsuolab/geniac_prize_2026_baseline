# Evaluation

1. run spec を選ぶ。学習前は `configs/eval_base.yaml`(Robosuite)と `configs/eval_base_libero.yaml`(LIBERO-PRO)、
   学習後は `configs/eval_trained.yaml` と `configs/eval_trained_libero.yaml`。学習後の spec には adapter
   (既定は DMPO の最後の adapter)と `lora_name` を書き、タスクと seed は学習前と揃える。
2. `geniac-baseline eval plan --spec configs/eval_base.yaml` を実行し、`output_dir/run.sh` を書き出す。
3. [GPU] `bash <output_dir>/run.sh` を実行する。
4. `geniac-baseline eval score <output_dir> --json score.json` で採点する。

spec の相対 path の基準は [collect.md](collect.md#run-spec-paths) の表のとおりである。

`examples/01_eval_base_cap_agent0.sh` と `examples/05_eval_trained_cap_agent0.sh` は手順 2 から 4 をまとめて実行する
(`SPEC=configs/eval_base_libero.yaml` などで suite を変える)。

## Seeds

使えるのは評価用の seed だけで、それ以外は点検で止まる。Robosuite は 401-450。LIBERO-PRO は 401-650 のうちタスクの
評価用 layout(init state が 50 なら layout 40-49)にあたるもので、441-450、491-500、541-550、591-600、641-650 になる。

## Settings

評価はベースラインの設定で行う。CaP-X の cap_agent0(`env_configs/libero/franka_libero_cap_agent0.yaml`)とは次が違う。

| item | baseline | cap_agent0 |
| --- | --- | --- |
| LIBERO-PRO の `max_steps` | 4000 | 8000 |
| LIBERO-PRO の parallel ensemble | なし(1 回の呼び出し) | あり |
| API | LIBERO 用の追加 skill 14 個を含む skill library(ASPIRE、またはコンペの CaP-X + `skill-library.patch`) | コンペの CaP-X の skill library |
| code writer | thinking なし(学習データと同じ形) | サーバーの既定(Qwen3.8 は thinking あり) |

| Server | Settings |
| --- | --- |
| VDM(port 8110) | 収集時と同じ(thinking あり、`--reasoning-parser qwen3`)。gpu-memory-utilization 0.30 |
| コード生成(port 8111) | 同じモデルで thinking なし(`--default-chat-template-kwargs {"enable_thinking": false}`)、reasoning parser なし。gpu-memory-utilization 0.30。adapter を指定すると `--enable-lora --max-lora-rank <adapter の r> --max-loras 1 --lora-modules <lora_name>=<adapter>` を付ける |
| Molmo(port 8122) | 収集時と同じ |

## Scoring

`eval score` はタスクごとに次を出力する。複数の run ディレクトリをまとめて採点でき、同じ seed が 2 回出るとエラーになる。
CaP-X はコードブロックを実行するたびにその時点の結果を名前にしたフォルダを保存するため、1 つの trial で同じ seed の
フォルダが複数できる。採点には trial 終了時の保存(`summary.txt` に Stdout / Stderr / Reward があるもの)だけを使う。

| Field | Meaning |
| --- | --- |
| `n` | 計画した seed の数(run の `plan.json` から。無ければ trial フォルダの数) |
| `missing` | trial フォルダが無い、または途中の保存しか無い(trial 終了前に止まった)seed の数。失敗として数え、`missing_seeds` に seed を出す |
| `success` | フォルダ名が `taskcompleted_1` かつ `sandboxrc_0` の trial の数 |
| `first_try` | REGENERATE なしで成功した数 |
| `turns_success` | 成功した trial のモデル呼び出し回数の平均 |
| `finish_unsolved` | 未達成のまま FINISH した数 |
| `no_finish` | FINISH に至らなかった数(ターン上限、クラッシュ、`missing`) |
| `code_no_keyword` | コードを含むが REGENERATE の語が無い回答の数(CaP-X は FINISH と扱う) |

## Refill

`missing` の seed は、その seed だけを `seeds` に書いた別の `output_dir` の spec で計画し直して実行する(refill)。
採点は元の run と refill の run を一緒に渡す(`eval score <run> <refill>`)。それでも結果が残らない seed は失敗のまま
数える。ベースラインの数字もこの扱いである。

コンペの gateway 経由の評価には使えない。その経路ではフォルダ名が常に `taskcompleted_0` になるため、gateway の
結果を使う。
