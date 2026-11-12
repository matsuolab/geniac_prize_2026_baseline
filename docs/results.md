# Baseline results

ベースラインの成功数である(1 task 50 trial、thinking off の code writer)。環境は ASPIRE([reproduce_baseline.md](reproduce_baseline.md))で、
コンペの CaP-X での値ではない。1 run ずつの値である。

| item | value |
| --- | --- |
| LIBERO-PRO | 30 task、学習に使っていない layout 40-49、seed 441-450 / 491-500 / 541-550 / 591-600 / 641-650 |
| Robosuite | 7 task、seed 401-450 |
| base | Qwen/Qwen3.8-27B(学習前) |
| DMPO | 34 task の成功 trial で SFT(epoch 2)した後、590 pair で DMPO した adapter(最終 epoch) |

| suite | task | base | DMPO |
| --- | --- | :-: | :-: |
| LIBERO-PRO | goal_bowl_on_cabinet | 15/50 | 47/50 |
| LIBERO-PRO | goal_bowl_on_plate | 19/50 | 31/50 |
| LIBERO-PRO | goal_bowl_on_stove | 27/50 | 49/50 |
| LIBERO-PRO | goal_cream_cheese_bowl | 0/50 | 17/50 |
| LIBERO-PRO | goal_drawer | 4/50 | 11/50 |
| LIBERO-PRO | goal_drawer_bowl | 0/50 | 1/50 |
| LIBERO-PRO | goal_push_plate | 0/50 | 0/50 |
| LIBERO-PRO | goal_turn_on_stove | 49/50 | 50/50 |
| LIBERO-PRO | goal_wine_on_cabinet | 10/50 | 25/50 |
| LIBERO-PRO | goal_wine_on_rack | 37/50 | 50/50 |
| LIBERO-PRO | object_alphabet_soup | 10/50 | 22/50 |
| LIBERO-PRO | object_bbq_sauce | 0/50 | 6/50 |
| LIBERO-PRO | object_butter | 15/50 | 39/50 |
| LIBERO-PRO | object_chocolate_pudding | 27/50 | 50/50 |
| LIBERO-PRO | object_cream_cheese | 10/50 | 42/50 |
| LIBERO-PRO | object_ketchup | 4/50 | 13/50 |
| LIBERO-PRO | object_milk | 33/50 | 39/50 |
| LIBERO-PRO | object_orange_juice | 36/50 | 48/50 |
| LIBERO-PRO | object_salad_dressing | 0/50 | 1/50 |
| LIBERO-PRO | object_tomato_sauce | 11/50 | 15/50 |
| LIBERO-PRO | spatial_bowl_plate | 27/50 | 50/50 |
| LIBERO-PRO | spatial_in_drawer | 0/50 | 1/50 |
| LIBERO-PRO | spatial_next_to_cookie_box | 2/50 | 3/50 |
| LIBERO-PRO | spatial_next_to_plate | 35/50 | 42/50 |
| LIBERO-PRO | spatial_next_to_ramekin | 31/50 | 46/50 |
| LIBERO-PRO | spatial_on_cabinet | 7/50 | 34/50 |
| LIBERO-PRO | spatial_on_cookie_box | 9/50 | 29/50 |
| LIBERO-PRO | spatial_on_ramekin | 23/50 | 38/50 |
| LIBERO-PRO | spatial_on_stove | 30/50 | 49/50 |
| LIBERO-PRO | spatial_table_center | 19/50 | 46/50 |
| Robosuite | cube_lifting | 39/50 | 50/50 |
| Robosuite | cube_restack | 9/50 | 38/50 |
| Robosuite | cube_stack | 27/50 | 50/50 |
| Robosuite | nut_assembly | 0/50 | 0/50 |
| Robosuite | spill_wipe | 45/50 | 46/50 |
| Robosuite | two_arm_handover | 0/50 | 0/50 |
| Robosuite | two_arm_lift | 2/50 | 1/50 |
| | **total** | **612/1850 (33.1%)** | **1079/1850 (58.3%)** |

## Conditions

数字は次の条件で測った。ASPIRE の環境での条件であり、ほかの環境で同じ数字が出るとは限らない。

| item | value |
| --- | --- |
| skill library | ASPIRE の skill library(LIBERO 用の追加 skill 14 個を含む、[reproduce_baseline.md](reproduce_baseline.md)) |
| 1 trial の step 上限 | LIBERO-PRO 4000(task config の `max_steps`)、Robosuite はシミュレータの既定(基底クラスでは 1500) |
| 1 trial の実時間上限 | 2400 s(`trial_timeout_s`、ASPIRE の `ASPIRE_TRIAL_TIMEOUT_S`)。時間切れの trial は 1 回だけやり直す |
| LIBERO-PRO の初期状態 | 保存された init state を読み込んだ後に、もう一度 reset する(配置がそこで引き直される) |
