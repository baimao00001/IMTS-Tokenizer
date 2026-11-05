#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONIOENCODING=utf-8
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --output-dir|--output-dir=*)
      printf '%s\n' 'This launcher creates its own timestamp directory; do not pass --output-dir.' >&2
      exit 2
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
  esac
done

if [[ "$DRY_RUN" == 1 ]]; then
  exec python -B -u IMTSTokenizer/scripts/run_parallel.py "$@"
fi
python -B -u IMTSTokenizer/scripts/run_parallel.py --dry-run "$@" > /dev/null

RUN_DIR="$(python -B -c 'from pathlib import Path; from lib.run_paths import resolve_output_dir; p = resolve_output_dir(Path.cwd()); p.mkdir(parents=True, exist_ok=False); print(p.as_posix())')"
PYTHONDONTWRITEBYTECODE=1 nohup python -B -u IMTSTokenizer/scripts/run_parallel.py \
  --output-dir "$RUN_DIR" "$@" > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/launcher.pid"
printf 'PID: %s\nAll outputs: %s\n' "$PID" "$RUN_DIR"
