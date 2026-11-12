# Third-party notices

このリポジトリには、次の第三者の著作物とその改変が含まれる。各ライセンスの全文は `licenses/` にある。

| files | origin | license |
| --- | --- | --- |
| `patches/geniac_prize/geniac-prize.patch`、`configs/capx/geniac_prize/` | [CaP-X](https://github.com/capgym/cap-x) とコンペ版の [geniac_prize_2026_code](https://github.com/matsuolab/geniac_prize_2026_code) | MIT(`licenses/CaP-X-LICENSE`) |
| `patches/geniac_prize/skill-library.patch`(コンペの CaP-X の LIBERO skill library に、ASPIRE の patch で足した skill を移した物) | CaP-X、geniac_prize_2026_code、[NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) | MIT と Apache-2.0 |
| `submission/gateway/`(`configs/capx/geniac_prize/` の task config から作った gateway の task config)、`submission/trial-state.patch`(geniac_prize_2026_code の `capx/submission/code_env.py` を変更する patch) | CaP-X、geniac_prize_2026_code、[NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) | MIT と Apache-2.0 |
| `tests/fixtures/`(CaP-X が書き出した trial、prompt、config) | CaP-X(ASPIRE 同梱版) | MIT と Apache-2.0 |
| `patches/aspire/aspire-f4c8939-local-changes.patch`(ASPIRE の 9 file を変更する patch。変更点は patch の内容の通り) | [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) | Apache-2.0(`licenses/ASPIRE-LICENSE`、`licenses/ASPIRE-NOTICE`) |
| `configs/capx/aspire/`(ASPIRE の task config を改変した物) | [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) | Apache-2.0(`licenses/ASPIRE-LICENSE`、`licenses/ASPIRE-NOTICE`) |

Qwen3.8-27B の chat template は同梱しない(`geniac-baseline template` が利用者の snapshot から作る)。
