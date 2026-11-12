# Data format

```
CaP-X trial folders + sources.yaml --convert rows--> rows.jsonl --convert tag--> rows.jsonl (+ code_outcome)
                                   --convert split--> sft.jsonl / val.jsonl / manifest.json   (input of `train sft`)
failures: convert rows --include failure --> rows.jsonl --convert tag--> rows.jsonl
success split + failure rows --pairs--> pairs.jsonl / rows.jsonl / manifest.json            (input of `train dmpo`)
```

各段階は新しいディレクトリに書き出し、既存のディレクトリは上書きしない。

## Trial folder

CaP-X が書き出す。フォルダ名の trial 番号 `t` がそのまま seed である。

```
<output_dir>/<model name, "/" -> "_">/<task>/trial_<t>_sandboxrc_<rc>_reward_<r>_taskcompleted_<0|1>/
  all_responses.json            one entry per model call: initial_prompt + code_blocks (call 0);
                                decision "regenerate" + code_blocks + multi_turn_prompt (when save_multiturn_prompts);
                                decision "finish" + reasoning (CaP-X never stores this call's prompt)
  summary.txt                   final Stdout (last block), Stderr (join of every non-empty stderr), Reward
                                (a per-turn save writes only the program)
  code.py                       the final program
  differencing_feedback_XX.txt  VDM text after block XX+1 (written only by the ASPIRE copy of CaP-X)
```

CaP-X はコードブロックを実行するたびに、その時点の結果を名前にしたフォルダを保存する。そのため結果が途中で変わった
trial は、同じ seed のフォルダを複数残す。途中の保存の `summary.txt` には program しか無く、trial 終了時の保存だけが
Stdout / Stderr / Reward を持つ。`convert rows` と `eval score` は seed ごとにこの終了時の保存を使う
(`is_final_save`)。フォルダの数は trial の数ではない。

| Result | Condition |
| --- | --- |
| success | `taskcompleted_1` かつ `sandboxrc_0` かつ `code.py` が空でない |
| failure | `taskcompleted_0` かつ `all_responses.json` がある |
| どちらでもない | 使わない(件数だけ数える) |

## Sources file

`convert rows` の入力である。`collect plan` が自動で書き出す。同じ (task, seed) の trial が 2 つあるとエラーになる。

```yaml
sources:
  - run_tag: my_run_01            # becomes part of sample_id; no "__"
    task: cube_lifting
    trials_dir: runs/collect/seg_cube_lifting_301/outputs/Qwen_Qwen3.8-27B/cube_lifting
    config: runs/collect/seg_cube_lifting_301/cube_lifting.yaml   # the exact config the run used
  - run_tag: my_run_01
    task: libero_spatial_on_stove  # LIBERO-PRO task names start with "libero_"
    trials_dir: ...
    config: ...
    layouts: 50                    # LIBERO-PRO only, required: the task's number of init states
```

## Rows

モデル呼び出し 1 回が 1 行になる。turn 0 は初期プロンプトと最初のプログラム、turn 1 以降は判断プロンプトと
`FINISH` または `REGENERATE` + コードブロックである。次は `tests/fixtures/cube_lifting_real` の例である(テキストは省略)。

```json
{"id": "cube_lifting__fixture__trial302-t1",
 "messages": [
  {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that generates Python code to directly solve the task."}]},
  {"role": "user", "content": [{"type": "text", "text": "You are controlling a Franka Emika robot with API described below.\nGoal: pick up the red cube and lift it.\n... <API docs> ...The following code blocks were just executed:\n```python\nimport numpy as np\n...\n```\nThe current console stdout is:\n```\n...\n```\nThe current console stderr is:\n```\n\n```\n...\n\nIncluded below is the observed differences between the current state of the environment (afer the code above was executed) and the previous state ...:\n<VDM text>\nBased on the code output, potential error messages, and the observation made above, carefully reason about the following:\nPlease respond with EXACTLY ONE of the following:\n- The word 'REGENERATE' ...\n- The word 'FINISH' if the task appears to be complete\n"}]},
  {"role": "assistant", "content": [{"type": "text", "text": "REGENERATE\n```python\nimport numpy as np\n...\n```"}]}],
 "metadata": {"source": "capx_multiturn", "sample_id": "cube_lifting__fixture__trial302", "task": "cube_lifting",
              "condition": "cube_lifting", "seed": 302, "run_tag": "fixture", "n_llm_calls": 3, "success": true,
              "turn": 1, "kind": "decision", "feedback": "saved", "code_outcome": "final"}}
```

| Field | Meaning |
| --- | --- |
| `kind` | `first_turn`(turn 0)または `decision`(turn 1 以降) |
| `feedback` | `saved` は CaP-X が保存したプロンプト、`rebuilt` は再構成した FINISH のプロンプト、turn 0 では null |
| `code_outcome` | `tag` が付ける。`final`(このコードで trial が終わった)、`replaced_error`(traceback の後に置き換えられた)、`replaced_clean`(エラーなしで置き換えられた)、FINISH では null |
| `n_layouts` | LIBERO-PRO の行だけに付く。タスクの init state の数 |

`geniac-baseline convert validate` でスキーマを検査できる。

## Rebuilding FINISH prompts

CaP-X は FINISH を選んだ呼び出しのプロンプトを保存しないため、`convert rows` が再構成する。書き出す前に、保存済みの
REGENERATE プロンプトをすべて同じ材料から再構成してバイト単位で比較し、1 つでも一致しなければ何も書き出さない。
VDM テキストのファイル(`differencing_feedback_XX.txt`)が無い出力も拒否する。

## Train / val split

| suite | train | val | dropped | error |
| --- | --- | --- | --- | --- |
| LIBERO-PRO(`libero_*`) | 他のすべての seed | seed 381-390 | タスクの評価用 layout にあたる seed | seed >= 401 |
| Robosuite(その他) | 他のすべての seed | seed 391-400 | なし | seed >= 401 |

LIBERO-PRO の layout は `(seed - 1) % n_layouts` で、評価用 layout はタスクの最後の 10 個である(init state が 50 なら
layout 40-49)。コンペの CaP-X では
seed `s` と `s + n_layouts` が同じシーンになるため、layout で分けないと同じシーンが学習と評価の両方に入る。
`n_layouts` は推定せず、sources ファイルにタスクごとに書く(docs/collect.md)。

## Pairs

`geniac-baseline pairs --success runs/data/split --failure runs/data/fail_tagged --out runs/data/pairs` は、成功した
trial と失敗した trial を同じタスクの中で 1 対 1 に組み、`pairs.jsonl` と、組んだ trial の行を集めた `rows.jsonl` を
書き出す。

```json
{"pair_id": 0, "task": "cube_lifting", "match": "task", "split": "train", "chosen": "cube_lifting__fixture__trial302", "rejected": "cube_lifting__fixture__trial360", "chosen_turns": 3, "rejected_turns": 4}
```

- `chosen` は成功、`rejected` は失敗した trial の `sample_id` である。各 trial は 1 回だけ使う。
- LIBERO-PRO は同じ layout の trial を先に組み(`match: layout`)、残りをタスク内で組む(`match: task`)。Robosuite は
  タスク内で組む。
- pair の 2 つの trial は必ず train / val の同じ側に入る。分割の規則は上と同じで、評価用 layout の trial は除外し、
  seed が 401 以上ならエラーで止まる。
- 行が欠けた trial(turn が `0..T-1` でない)は pair に使わない。
- `--seed`(既定 0)が同じなら、同じ入力から同じ pair を作る。

