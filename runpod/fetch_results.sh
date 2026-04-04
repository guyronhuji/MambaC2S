#!/usr/bin/env bash
# ============================================================
# Fetch experiment outputs from a running RunPod pod.
# Lists your running pods, lets you pick one, then rsyncs
# outputs/ to ./outputs/runpod/ on your local machine.
#
# Usage:
#   bash runpod/fetch_results.sh
# ============================================================

set -euo pipefail

LOCAL_DEST="./outputs/runpod"

# ── 1. List running pods ─────────────────────────────────────
echo "Fetching running pods ..."
PODS_JSON=$(runpodctl pod list 2>/dev/null) || { echo "ERROR: runpodctl pod list failed"; exit 1; }

# Extract running pod IDs
POD_IDS=$(echo "$PODS_JSON" | python3 -c "
import sys, json
pods = json.load(sys.stdin)
running = [p['id'] for p in pods if (p.get('desiredStatus','') or '').upper() == 'RUNNING']
print('\n'.join(running))
")

if [ -z "$POD_IDS" ]; then
    echo "No running pods found."
    exit 1
fi

# ── 2. Get details (incl. SSH) for each pod ──────────────────
ROWS=()
while IFS= read -r pod_id; do
    detail=$(runpodctl pod get "$pod_id" 2>/dev/null)
    row=$(echo "$detail" | python3 -c "
import sys, json
d = json.load(sys.stdin)
pod_id = d.get('id','?')
name   = d.get('name','?')
gpu    = d.get('machine',{}).get('gpuDisplayName') or '?'
cost   = d.get('costPerHr','?')
ssh    = d.get('ssh') or {}
# ssh.command is like: ssh root@1.2.3.4 -p 12345 -i ...
cmd    = ssh.get('command','')
err    = ssh.get('error','')
print(f'{pod_id}|{name}|{gpu}|\${cost}/hr|{cmd}|{err}')
")
    ROWS+=("$row")
done <<< "$POD_IDS"

# ── 3. Display table ─────────────────────────────────────────
echo ""
printf "%-4s %-20s %-12s %-22s %s\n" "#" "Pod ID" "Name" "GPU" "SSH status"
printf "%-4s %-20s %-12s %-22s %s\n" "---" "-------------------" "-----------" "---------------------" "----------"
for i in "${!ROWS[@]}"; do
    IFS='|' read -r pod_id name gpu cost ssh_cmd ssh_err <<< "${ROWS[$i]}"
    if [ -n "$ssh_cmd" ]; then
        ssh_status="ready"
    elif [ -n "$ssh_err" ]; then
        ssh_status="($ssh_err)"
    else
        ssh_status="(unknown)"
    fi
    printf "%-4s %-20s %-12s %-22s %s\n" "$((i+1))" "$pod_id" "$name" "$gpu  $cost" "$ssh_status"
done

TOTAL=${#ROWS[@]}

# ── 4. Select pod ────────────────────────────────────────────
echo ""
read -rp "Choose pod number (1-${TOTAL}): " CHOICE

if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt "$TOTAL" ]; then
    echo "Invalid choice."
    exit 1
fi

IFS='|' read -r POD_ID POD_NAME POD_GPU POD_COST SSH_CMD SSH_ERR <<< "${ROWS[$((CHOICE-1))]}"

# ── 5. Build rsync SSH args from the ssh.command ─────────────
if [ -z "$SSH_CMD" ]; then
    echo ""
    echo "SSH not ready for this pod yet ($SSH_ERR)."
    echo "Get it from: runpod.io/console/pods → Connect → 'SSH over exposed TCP'"
    echo "IMPORTANT: Use the 'root@IP -p PORT' format, NOT the ssh.runpod.io proxy URL"
    echo "Example: ssh root@213.173.x.x -p 12345 -i ~/.ssh/id_ed25519"
    read -rp "Paste the full SSH command: " SSH_CMD
fi

# Parse host and port out of the ssh command
SSH_USER_HOST=$(echo "$SSH_CMD" | grep -oE '[A-Za-z0-9._-]+@[^ ]+' | head -1 || true)
SSH_PORT=$(echo "$SSH_CMD" | grep -oE '\-p [0-9]+' | grep -oE '[0-9]+' | head -1 || true)

if [ -z "$SSH_USER_HOST" ]; then
    echo "Could not parse SSH command: $SSH_CMD"
    exit 1
fi

mkdir -p "$LOCAL_DEST"
echo ""
echo "Pod  : $POD_NAME ($POD_ID)"
echo "GPU  : $POD_GPU  $POD_COST"
echo "SSH  : $SSH_USER_HOST${SSH_PORT:+ port $SSH_PORT}"
echo "Dest : $LOCAL_DEST"
echo ""

TMP_TAR="/tmp/runpod_outputs_$$.tar.gz"

# ── RunPod community cloud: use runpodctl send/receive (croc protocol) ──
# No ports needed — works through the SSH proxy.
echo ""
echo "============================================================"
echo "  In your pod SSH session, run:"
echo ""
echo "    cd /workspace/MambaC2S && tar czf /tmp/outputs.tar.gz outputs/ && runpodctl send /tmp/outputs.tar.gz"
echo ""
echo "  It will print a code like:  8338-galileo-collect-fidel"
echo "============================================================"
echo ""
read -rp "Paste the code here: " TRANSFER_CODE

if [ -z "$TRANSFER_CODE" ]; then
    echo "No code entered."
    exit 1
fi

mkdir -p "$LOCAL_DEST"
echo ""
echo "Receiving ..."
cd "$LOCAL_DEST"
runpodctl receive "$TRANSFER_CODE"

# Extract if a tar.gz landed here
TAR_FILE=$(ls outputs.tar.gz 2>/dev/null || ls *.tar.gz 2>/dev/null | head -1 || true)
if [ -n "$TAR_FILE" ]; then
    echo "Extracting $TAR_FILE ..."
    tar xzf "$TAR_FILE" --strip-components=1 2>/dev/null || tar xzf "$TAR_FILE"
    rm -f "$TAR_FILE"
fi

echo ""
echo "Done — results saved to $LOCAL_DEST"
