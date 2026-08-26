#!/usr/bin/env bash
# Wait for a process to exit, then run a command.
#
# Takes a PID, deliberately, not a pattern. The pattern-matching version of this is a trap:
#
#     while pgrep -f "some.command" >/dev/null; do sleep 20; done
#
# `pgrep -f` matches full command lines, and the waiter's own argv contains the pattern - so it
# matches itself and loops forever while looking perfectly healthy. That cost 90 minutes of
# wall clock on this project. Trying to guard it with a `[s]ome` bracket glob does not help
# either, because the unguarded pattern is still sitting in argv, and any parent wrapper
# (timeout, nohup, the shell) carries it too.
#
# A PID is unambiguous and cannot match itself. Use `$!` from the job you launched.
#
# Usage:  scripts/wait_for.sh <pid> <command...>
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <pid> <command...>" >&2
  exit 2
fi

pid="$1"; shift
while kill -0 "$pid" 2>/dev/null; do
  sleep 15
done

exec "$@"
