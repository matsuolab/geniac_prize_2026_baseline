#!/usr/bin/env bash
# Forward 127.0.0.1:8000 of this machine (the ASPIRE client) to the Qwen server of remote_qwen/serve.sh on a compute
# node, through the login node, and reconnect whenever the connection drops (docs/remote_qwen.md). It runs in a tmux
# session, so it keeps going after you log out.
#   bash remote_qwen/tunnel.sh <user>@<compute node>     # the node: squeue -h -j <job> -o %N
#   tmux attach -t qwen_tunnel                           # watch it;  tmux kill-session -t qwen_tunnel  stops it
# QWEN_GATEWAY (default gmo) is the ~/.ssh/config host of the login node. Its IdentityFile (no passphrase) is used
# for both hops and the SSH agent is not, so the tunnel does not depend on the machine you logged in from.
set -euo pipefail
[[ $# -eq 1 && "$1" == *@* ]] || { sed -n '5,6p' "$0" >&2; exit 2; }
host="$1"
gateway="${QWEN_GATEWAY:-gmo}"
port="${QWEN_PORT:-8000}"
session="${QWEN_TUNNEL_SESSION:-qwen_tunnel}"

if [[ "${QWEN_TUNNEL_LOOP:-0}" != 1 ]]; then
  command -v tmux >/dev/null || { echo "error: tmux is required" >&2; exit 1; }
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "error: tmux session $session exists; stop it first: tmux kill-session -t $session" >&2
    exit 1
  fi
  self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  tmux new-session -d -s "$session" \
    "$(printf '%q ' env QWEN_TUNNEL_LOOP=1 QWEN_GATEWAY="$gateway" QWEN_PORT="$port" bash "$self" "$host")"
  echo "tunnel to $host started in tmux session $session"
  echo "check: curl -s http://127.0.0.1:$port/v1/models"
  exit 0
fi

key="$(ssh -G "$gateway" | awk '$1 == "identityfile" { print $2; exit }')"
key="${key/#\~/$HOME}"
[[ -r "$key" ]] || { echo "error: no readable IdentityFile for $gateway in ~/.ssh/config ($key)" >&2; exit 1; }
while true; do
  echo "$(date -u +%FT%TZ) connecting 127.0.0.1:$port -> $host:$port via $gateway"
  rc=0
  env -u SSH_AUTH_SOCK ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o IdentitiesOnly=yes -i "$key" \
    -J "$gateway" -L "127.0.0.1:$port:127.0.0.1:$port" "$host" || rc=$?
  echo "$(date -u +%FT%TZ) ssh exited with $rc; retrying in 5 s"
  sleep 5
done
