#!/usr/bin/env bash
# Smoke-test every console view for runtime errors.
#
# `npm run build` does NOT catch this class of bug. A `const` read inside its temporal dead
# zone, a bad property access, a null deref - all compile fine and then throw at render, taking
# the whole view down to a blank page. That shipped twice here, both times because a change was
# made and only the view that "looked affected" was re-checked.
#
# When React throws during render and there is no error boundary, it unmounts the tree and
# #root goes empty. So: load each view, assert #root has real content, and assert a marker
# string unique to that view is present.
#
# Usage:  scripts/check_ui.sh [base-url]
set -uo pipefail

BASE="${1:-http://127.0.0.1:5173}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# view : marker that only renders when that view mounts successfully
VIEWS=(
  "overview:What the numbers rest on"
  "identify:Kill-chain matrix"
  "defend:Authorisation stream"
  "adapt:Recall on attacks the model has never seen"
)

fail=0
for entry in "${VIEWS[@]}"; do
  view="${entry%%:*}"
  marker="${entry#*:}"
  out="$TMP/$view.html"

  timeout 70 chromium --headless=new --no-sandbox --disable-gpu \
    --virtual-time-budget=9000 --dump-dom "$BASE/#$view" > "$out" 2>/dev/null

  size=$(python3 - "$out" <<'PY'
import re, sys, pathlib
html = pathlib.Path(sys.argv[1]).read_text(errors="ignore")
m = re.search(r'<div id="root">(.*?)</div>\s*<script', html, re.S)
print(len(m.group(1)) if m else 0)
PY
)

  if [ "$size" -lt 500 ]; then
    echo "  FAIL  #$view — #root is empty ($size chars): the view threw during render"
    fail=1
  elif ! grep -qF "$marker" "$out"; then
    echo "  FAIL  #$view — rendered but marker not found: '$marker'"
    fail=1
  else
    echo "  ok    #$view (${size} chars)"
  fi
done

if [ "$fail" -ne 0 ]; then
  echo ""
  echo "One or more views failed to render. Open the browser console - a blank panel is"
  echo "almost always an exception during render, not a data problem."
  exit 1
fi
echo ""
echo "All views render."
