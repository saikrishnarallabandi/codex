#!/usr/bin/env bash
set -euo pipefail

# install python deps if needed and compile
pip install -q httpx openai requests
python3 -m py_compile scripts/call_gpt.py

# prepare sanitized input
TMP_INPUT=$(mktemp)
tail -n +2 codex-cli/log.in > "$TMP_INPUT"

if [ -n "${OPENAI_API_KEY:-}" ]; then
  python3 scripts/call_gpt.py < "$TMP_INPUT"
else
  echo "OPENAI_API_KEY not set; skipping call_gpt.py run"
fi

if command -v pnpm >/dev/null 2>&1; then
  if [ -d codex-cli/node_modules ]; then
    (cd codex-cli && pnpm test)
  else
    echo "node_modules missing; skipping codex-cli tests"
  fi
else
  echo "pnpm not found; skipping codex-cli tests"
fi

rm -f "$TMP_INPUT"
