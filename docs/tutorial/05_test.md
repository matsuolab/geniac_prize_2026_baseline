# 5. テスト実行

[Tutorial](README.md) | 前: [4. 依存関係のインストール](04_install.md) | 次: [6. W&B 連携](06_wandb.md)

[4 章](04_install.md)で作った `.venv` で、このパッケージのテストを流す。テストは CPU だけで動き、モデルの重みも、
シミュレータも、CaP-X の tree も要らない。データの変換、実行計画と点検、採点、学習の loss、提出の検証を、
`tests/fixtures/` の実際の trial と、テストの中で作る合成の trial で確かめる。`<...>` の意味は
[Tutorial](README.md#placeholders) の表にある。

1. ログインノードから GPU 無しで計算ノードを確保し、テストを流す。対象のディレクトリと import の設定は
   `pyproject.toml` にあるので、ファイル名は並べない。`TMPDIR=/tmp` で、テストの一時ファイルを計算ノードの `/tmp` に
   書かせる。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   srun --partition=<パーティション> --cpus-per-task=2 --mem=8G --time=00:10:00 \
       /usr/bin/env TMPDIR=/tmp .venv/bin/python -m pytest -q
   ```

2. 出力の末尾を確かめる。件数と時間は commit によって変わる。`failed` と `error` が無く、skip が次の 1 件だけなら
   良い。skip されるのは「torch が無くても trainer を import できる」ことのテストで、先に別のテストが torch を
   import した時は skip する。

   ```text
   SKIPPED [1] tests/test_train_cpu.py:37: torch already imported by another test
   171 passed, 1 skipped in 6.65s
   ```

   skip が `tests/test_dmpo_torch.py:7: could not import 'torch'` の場合は、`.venv` に torch が入っていない。
   [4 章](04_install.md)の 2 番をやり直す。

3. 失敗した場合は、そのテストファイルを指定して詳しく見る。`-v` はテストごとの結果を、`--tb=short` は失敗した箇所を
   表示し、`-x` は最初の失敗で止める。

   ```bash
   srun --partition=<パーティション> --cpus-per-task=2 --mem=8G --time=00:10:00 \
       /usr/bin/env TMPDIR=/tmp .venv/bin/python -m pytest tests/test_convert_rows.py -v -x --tb=short
   ```

4. ベースラインのコードを変えた時は、テストに加えて、CI と同じ lint を流す([CONTRIBUTING.md](../../CONTRIBUTING.md))。
   lint はログインノードで流してよい。

   ```bash
   cd <共用ストレージ>/<ユーザー名>/geniac_prize_2026_baseline
   uv run --no-sync ruff check .
   uv run --no-sync ruff format --check .
   ```

このテストは CaP-X の tree を動かさない。CaP-X の環境は [4 章](04_install.md)の 6 番と、GPU で収集や評価を流す章で
確かめる。

次: [6. W&B 連携](06_wandb.md)
