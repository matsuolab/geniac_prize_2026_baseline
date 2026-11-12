# Training

学習は 2 段階で行う。SFT で warm start した LoRA adapter を、DMPO でさらに学習する。

## SFT

1. `configs/train_sft.yaml` の `data`(`convert split` の出力ディレクトリ)、`model`、`output_dir` を設定する。
2. `geniac-baseline train sft --config configs/train_sft.yaml --dry-run` を実行する。全行を encode し、token 数と
   1 epoch の step 数(`steps_per_epoch`)を表示する。GPU は不要だが `[train]` extra が必要である。
3. [GPU] `geniac-baseline train sft --config configs/train_sft.yaml` を実行する。epoch ごとに `epoch-N/adapter` を書き出し、
   val を測る。中断した場合は `--resume <output_dir>/ckpt-last` を付けて再開する。`output_dir` が空でなければ、
   `--resume` 無しでは始まらない。

## Epoch selection

評価と DMPO に使う SFT の adapter は、epoch の終わりの adapter のうち `val/loss_all`(val の全回答 token の
cross-entropy)が最小の物である。この規則はベースラインの評価結果を見る前に決めた。学習の最後に
`<output_dir>/best_epoch.json` に書き出す。ベースラインは 3711 行、grad_accum 8 で 1 epoch 464 step、val loss_all は
epoch 1 / 2 / 3 で 0.1182 / 0.1151 / 0.1162 であり、epoch 2 を使った。`val.jsonl` が無い場合は止まる
(`require_val: false` で val 無しの学習になる)。

`max_row_tokens`(既定 100000)より長い行は学習から外し、`[data]` の行に `dropped_long_rows` として row id と token 数を
出す。ベースラインでは 1 行(object_chocolate_pudding の seed 324 の turn 6、159,470 token)が B300 1 枚で memory が
足りず、この行だけを外した。

loss は回答の全 token に対する cross-entropy で、プロンプトには loss をかけない(`geniac_baseline/trainer/losses/sft_ce.py`)。
既定値はベースラインの SFT adapter の設定である(LoRA rank 32、alpha 32、lr 1e-4、grad_accum 8、3 epoch)。

## DMPO

1. `geniac-baseline pairs` で pair を作る(docs/data_format.md)。
2. `configs/train_dmpo.yaml` の `data`(pairs の出力ディレクトリ)、`model`、`init_adapter`、`output_dir` を設定する。
   `init_adapter` に SFT の run ディレクトリを渡すと、`best_epoch.json` の epoch の adapter を使う。
3. `geniac-baseline train dmpo --config configs/train_dmpo.yaml --dry-run` で pair 数、turn 数、step 数を確認する。
4. [GPU] `geniac-baseline train dmpo --config configs/train_dmpo.yaml` を実行する。最初に SFT の adapter で全 turn の
   log 確率を 1 回だけ計算して `<output_dir>/ref_logp.json` に保存し、その後に学習する。epoch ごとに `epoch-N/adapter`、
   最後に `adapter/` を書き出す。評価には最後の adapter を使う。中断した場合は `--resume <output_dir>/ckpt-last` を
   付けて再開する(`ref_logp.json` を再利用する。adapter か pair のデータが違えば止まる)。

ベースラインの DMPO は train 521 pair、val 69 pair、1 epoch 17 step(最後の step は 9 pair)、3 epoch で 51 step だった。
pair はベースモデル(学習前)で集めた成功と失敗の trial から作る。ETO と DMPO の論文は SFT 後の agent 自身の探索から
作るので、この点が異なる。

policy は SFT の adapter から始め、reference は同じ SFT の adapter を固定したものである。成功した trajectory
$\tau_w$($T_w$ turn)と失敗した trajectory $\tau_l$($T_l$ turn)の pair に対する loss は次の通り(DMPO, EMNLP 2024,
arXiv 2406.14868, Eq. 16)。$r_k$ は、turn $k$ の回答の全 token の log 確率の和について、policy と reference の差を取った
ものである。

$$
\phi(k,T)=\frac{\gamma^{k}\,(1-\gamma^{T-k})}{1-\gamma^{T}},\quad
r_k(\tau)=\log\pi_\theta(y_k\mid x_k)-\log\pi_{\mathrm{ref}}(y_k\mid x_k),\quad
m=\beta\Big(\sum_{k=0}^{T_w-1}\phi(k,T_w)\,r_k(\tau_w)-\sum_{k=0}^{T_l-1}\phi(k,T_l)\,r_k(\tau_l)\Big),\quad
\mathcal{L}=-\operatorname{mean}\log\sigma(m)
$$

勾配は 2 回のパスで正確に計算する。1 回目は勾配なしで $m$ を求め、2 回目は turn ごとに
$-(1-\sigma(m))\,\beta\,\phi\,\mathrm{sign}/B$ を係数として backward する(sign は chosen で +1、rejected で -1、$B$ は
step 内の pair 数)。1 つの step は `pairs_per_step` 個の pair を使い、epoch の最後の step は残りの pair だけを使う
(pair 数が `pairs_per_step` で割り切れない場合、最後の step の pair は少なくなる)。

## Default settings

| Setting | Default | Provenance |
| --- | --- | --- |
| `beta` | 0.5 | ETO(arXiv 2403.02502)が ALFWorld で使った値。DMPO の論文は {0.1, …, 0.9} を探索しており、0.5 はその範囲内 |
| `gamma` | 0.9 | ベースラインで選んだ値。DMPO の論文の探索範囲 {0.1, …, 0.9, 0.99} の範囲内 |
| `lr` | 1e-5 | ベースラインで選んだ値。ETO の 1e-6(全パラメータの学習)を LoRA 用に 10 倍した |
| `pairs_per_step` | 32 | DMPO の論文(Section 5.1)と ETO の batch size |
| `epochs` | 3 | ETO |
| `warmup_steps`, schedule | 1、cosine | ベースラインで選んだ値(SFT と同じ形のスケジュール) |
| LoRA, `init_adapter` | SFT と同じ LoRA、SFT の adapter | ベースラインの構成 |

wandb は `wandb.enabled: true` のときだけ使う(SFT と DMPO 共通)。
