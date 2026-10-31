#!/bin/sh
# Competition contract, upscaling track: /bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR
# (adapted from examples/competition_contenders/upscaling_example/run.sh of miner-vidaio)
#   - every regular, non-hidden file in INPUT_DIR is one item (H.264 MP4, sha256 name, no extension);
#   - INPUT_DIR/.vidaio-next-upscale-task-<name> holds {"target_width","target_height","upscale_factor"};
#   - write one MP4 per item into OUTPUT_DIR under the SAME name, exactly the target size,
#     the input's frame count and frame rate, BT.709 limited range tags;
#   - no network, read-only root filesystem: write only under OUTPUT_DIR and /tmp;
#   - exit 0 always: a failing item is logged and skipped (it scores zero on its own).
# The whole script is timed (model load included), so the model is loaded ONCE for the batch:
# upscale.py takes all items in one process.
set -u
umask 022
[ "$#" -eq 2 ] || { echo "usage: /app/run.sh INPUT_DIR OUTPUT_DIR" >&2; exit 64; }
input_dir=$1
output_dir=$2
app_dir=${CONTENDER_APP_DIR:-/app}
python=${PYTHON:-python3}
PATH=/usr/local/bin:/usr/bin:/bin${PATH:+:$PATH}
PYTHONDONTWRITEBYTECODE=1
export PATH PYTHONDONTWRITEBYTECODE
BATCH_BUDGET_SECONDS=${BATCH_BUDGET_SECONDS:-3000}   # manifest batch_timeout_seconds is 3600
have_timeout=0
command -v timeout >/dev/null 2>&1 && have_timeout=1
if [ "$have_timeout" -eq 1 ]; then
    timeout -k 15 "$BATCH_BUDGET_SECONDS" "$python" "$app_dir/upscale.py" --checkpoint "$app_dir/model.pth" \
        --budget-seconds "$BATCH_BUDGET_SECONDS" "$input_dir" "$output_dir"
else
    "$python" "$app_dir/upscale.py" --checkpoint "$app_dir/model.pth" --budget-seconds "$BATCH_BUDGET_SECONDS" \
        "$input_dir" "$output_dir"
fi
rc=$?
rm -f -- "$output_dir"/.partial-* 2>/dev/null
[ "$rc" -eq 0 ] || echo "upscale.py exited $rc; outputs written so far are kept" >&2
exit 0
