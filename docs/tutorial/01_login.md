# 1. GMO クラウドへのログイン

[Tutorial](README.md) | 次: [2. 基本的な Slurm コマンドの確認](02_slurm.md)

まず初めに、GMOクラウドのログインノードにアクセスする方法を紹介する。
ログインノードでは計算負荷の高い作業は行わない。
Slurmジョブを投げる前の準備やバージョン管理を行うために使い、GPU演算を含むその他の処理は [2 章](02_slurm.md)の方法で計算ノードに投げる。
`<...>` で表記された箇所は [Tutorial](README.md#placeholders) を参照して各自で適切な値を埋めて実行する。

1. GMOクラウドに登録するために手元の PC で SSH の鍵を作る(既に使う鍵があれば不要)。

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_gmo
   ```

2. 作成された公開鍵 `~/.ssh/id_ed25519_gmo.pub` を cat コマンドなどで確認し、コピーする。
   GMOクラウドから届いている案内メールからユーザ登録を済ませて、「公開鍵設定」からコピーした公開鍵を設定する。
   秘密鍵(`.pub` の無い方)は誰とも共有することがないように手元の PC だけで保管する。
   参考までに、以下の動画でも同様の設定手順が紹介されている。
   https://youtu.be/9wkcIkLzmQA?t=172&si=IkCD0fnkjJ3pl79K

3. 設定が終わったら、`~/.ssh/config` に接続先の情報を記載する。
   GMOクラウドのNaviから `<ホスト名>`、`<ユーザー名>`、`<ポート番号>`を確認できる。

   ```
   Host gmo
       HostName <ホスト名>
       User <ユーザー名>
       Port <ポート番号>
       IdentityFile ~/.ssh/id_ed25519_gmo
       IdentitiesOnly yes
       ServerAliveInterval 60
   ```

4. ログインし、ログインノードに入れたことを確かめる。

   ```bash
   ssh gmo
   hostname
   ```

次: [2. 基本的な Slurm コマンドの確認](02_slurm.md)
