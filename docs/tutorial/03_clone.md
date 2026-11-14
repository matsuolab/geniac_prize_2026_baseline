# 3. 開発コードのクローン

[Tutorial](README.md) | 前: [2. 基本的な Slurm コマンドの確認](02_slurm.md) | 次: [4. 依存関係のインストール](04_install.md)

Slurmコマンドの使い方がわかったところで、早速ベースラインリポジトリを共用ストレージの自分用のディレクトリ
`<共用ストレージ>/<ユーザー名>/` にclone する。

1. 初めに、GitHub に SSH で接続できるようにする。ログインノードで GitHub 用の鍵を作り、公開鍵を GitHub の
   Settings → SSH and GPG keys に登録する。（[参考](https://docs.github.com/ja/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)）

   鍵はホームディレクトリの `~/.ssh` に置き、共用ストレージには置かない。鍵には必ずパスフレーズを付ける
   (`ssh-keygen` が聞いてくるので空にしない)。
   鍵を作ったら、`~/.ssh` と秘密鍵の権限を確かめる。
   それぞれの権限がグループメンバーなどもアクセスできるようになっていると危険なので必要に応じてchmodで適切に管理する。

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github
   ls -ld ~/.ssh ~/.ssh/id_ed25519_github    # drwx------ と -rw------- であることをチェック
   cat ~/.ssh/id_ed25519_github.pub          # GitHub に登録するためにコピーしておく
   cat >> ~/.ssh/config <<'EOF'
   Host github.com
       IdentityFile ~/.ssh/id_ed25519_github
       IdentitiesOnly yes
   EOF
   ssh -T git@github.com                     # "Hi <account>! You've successfully authenticated" と出れば良い
   ```

   以降の章で使う認証情報も同様にホーム領域に置き、グループ領域で共有しないようにする。

2. GPU 無しで計算ノードを確保し、このベースラインリポジトリを clone する。

   ```bash
   mkdir -p <共用ストレージ>/<ユーザー名>
   cd <共用ストレージ>/<ユーザー名>
   srun --partition=<パーティション> --cpus-per-task=1 --mem=2G --time=00:10:00 --pty bash
   git clone git@github.com:matsuolab/geniac_prize_2026_baseline.git
   exit
   ```

   ここまでで、ディレクトリの構造が以下のようになる。

   ```
   <共用ストレージ>/
   ├── <ユーザー名>/               # 自分用のディレクトリ
   │   ├── logs/                   # Slurm のログ(2 章)
   │   └── geniac_prize_2026_baseline/  # ベースラインリポジトリ
   │       └── configs/  docs/  examples/  geniac_baseline/  tests/ ...
   └── <他のメンバー>/
   ```

次: [4. 依存関係のインストール](04_install.md)
