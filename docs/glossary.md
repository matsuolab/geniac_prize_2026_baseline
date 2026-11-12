# Glossary

| term | meaning |
| --- | --- |
| trial | 1 つの task を 1 つの seed で解く 1 回の試行。CaP-X が `trial_<seed>_..._taskcompleted_<0/1>` のフォルダに書く |
| turn | trial の中の 1 回のモデル呼び出し。turn 0 は最初の program、以降は直前の実行結果を見た判断 |
| REGENERATE / FINISH | 判断の turn の答え。REGENERATE は語の後ろに書き直した program を続け、FINISH は trial を終える |
| VDM | visual differencing model。実行前後の画像の違いを文章にして、次の turn の prompt に入れる。ここでは同じ Qwen3.8 が担う |
| cap_agent0 | CaP-X の LIBERO 用の設定(`env_configs/libero/franka_libero_cap_agent0.yaml`)。ベースラインの設定との違いは docs/eval.md |
| layout / init state | LIBERO の task が持つ開始状態の一覧。trial は `init_states[(seed - 1) % layouts]` から始まる。ベースラインの suite は 50 個 |
| evaluation layouts | 各 LIBERO-PRO task の最後の 10 layout(40-49)。学習データには入れない |
| model snapshot | Hugging Face のモデルを置いた local ディレクトリ(`config.json` と tokenizer のファイルがある所)。`convert tag` と学習は offline で読む |
| skill library | タスクのコードが呼べる関数の集まり(CaP-X の API)。ASPIRE 版は LIBERO 用の追加 skill 14 個を持ち、コンペの CaP-X には `skill-library.patch` で同じ物を足す([setup.md](setup.md)) |
| tree | 収集と評価に使う CaP-X の checkout。既定はコンペの CaP-X + patch(docs/installation.md)。ベースラインの数字の再現だけ ASPIRE(docs/reproduce_baseline.md) |

## Tasks

| suite | tasks |
| --- | --- |
| Robosuite(7) | cube_lifting, cube_stack, cube_restack, spill_wipe, nut_assembly, two_arm_handover, two_arm_lift |
| LIBERO-PRO goal(10、`libero_goal_with_mug`) | bowl_on_cabinet, bowl_on_plate, bowl_on_stove, cream_cheese_bowl, drawer, drawer_bowl, push_plate, turn_on_stove, wine_on_cabinet, wine_on_rack |
| LIBERO-PRO object(10、`libero_object_swap`) | alphabet_soup, bbq_sauce, butter, chocolate_pudding, cream_cheese, ketchup, milk, orange_juice, salad_dressing, tomato_sauce |
| LIBERO-PRO spatial(10、`libero_spatial_swap`) | bowl_plate, in_drawer, next_to_cookie_box, next_to_plate, next_to_ramekin, on_cabinet, on_cookie_box, on_ramekin, on_stove, table_center |

LIBERO-PRO の task 名は `libero_<suite>_<name>`(例: `libero_goal_drawer`)である。task ごとの config は
`configs/capx/<tree>/<suite>/<task>.yaml` にある。
