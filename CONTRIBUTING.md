# Contributing

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
```

## Before opening a pull request

CI と同じ次のコマンドを実行する。CPU だけで動き、モデルもシミュレータも要らない。

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

## Rules

- CaP-X を再実装しない。収集と評価は checkout の CaP-X エントリポイントを呼ぶ。CaP-X の挙動に依存する変更には、
  そのコードの docstring かコメントに、CaP-X のファイル名と行番号を出典として付ける。
- converter の厳密さを保つ。`convert rows` は、正確に再構成できない行を書き出さずに拒否する。
  `tests/test_convert_rows.py` は、ベースラインのデータを作った変換の出力と比較する。この比較を壊す変更には、
  pull request に理由を書く。
- 不明な値に既定値を置かない。値が分からないもの(タスクの LIBERO init state の数など)は、仮定せずに指定を必須に
  する。
- コード、config、ドキュメント、fixture に、特定のマシンや個人に結びつく情報を入れない。ホームディレクトリの絶対パス、
  ホスト名、ユーザー名、認証情報は書かない。config に秘密情報を置かず、各ツールは自身のログイン機能を使う。
- torch が必要なテストは、torch が入っていない環境では skip する(`pytest.importorskip("torch")`)。
