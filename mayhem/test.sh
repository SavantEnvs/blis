#!/usr/bin/env bash
#
# mayhem/test.sh — run BLIS's own netlib BLAS test drivers (blastest/: ?blat1, ?blat2, ?blat3 for
# s/d/c/z), built by mayhem/build.sh against a normally-compiled libblis. Each driver computes every
# BLAS routine on generated inputs and compares against a reference; per routine it prints
# "PASSED THE COMPUTATIONAL TESTS" / "----- PASS -----" or a FAIL / "****" diagnostic.
# One CTRF test per routine result line; a driver that crashes, exits non-zero or prints no
# results counts as a failure.
set -uo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
: "${BLIS_CONFIG:=haswell}"
cd "$SRC"

emit_ctrf() {
  local tool="$1" passed="$2" failed="$3" skipped="${4:-0}" pending="${5:-0}" other="${6:-0}"
  local tests=$(( passed + failed + skipped + pending + other ))
  cat > "${CTRF_REPORT:-$SRC/ctrf-report.json}" <<JSON
{
  "results": {
    "tool": { "name": "$tool" },
    "summary": {
      "tests": $tests,
      "passed": $passed,
      "failed": $failed,
      "pending": $pending,
      "skipped": $skipped,
      "other": $other
    }
  }
}
JSON
  printf 'CTRF {"results":{"tool":{"name":"%s"},"summary":{"tests":%d,"passed":%d,"failed":%d,"pending":%d,"skipped":%d,"other":%d}}}\n' \
    "$tool" "$tests" "$passed" "$failed" "$pending" "$skipped" "$other"
  [ "$failed" -eq 0 ]
}

BIN="$SRC/mayhem/out-test/obj/$BLIS_CONFIG/blastest"
INPUT="$SRC/blastest/input"
if ! ls "$BIN"/sblat1.x >/dev/null 2>&1; then
  echo "test.sh: blastest drivers missing under $BIN — run mayhem/build.sh first" >&2
  emit_ctrf blis-blastest 0 1
  exit 1
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/blis-blastest.XXXXXX")
trap 'rm -rf "$work"' EXIT
cd "$work"

passed=0 failed=0
for p in s d c z; do
  for lvl in 1 2 3; do
    drv="${p}blat${lvl}"
    out="out.$drv"
    rm -f "$out"
    if [ "$lvl" = 1 ]; then
      "$BIN/$drv.x" > "$out" 2>&1; rc=$?
    else
      "$BIN/$drv.x" < "$INPUT/$drv.in" > "$drv.stdout" 2>&1; rc=$?
    fi
    if [ "$lvl" = 1 ]; then
      np=$(grep -c -- '- PASS -' "$out" 2>/dev/null || true)
    else
      np=$(grep -c 'PASSED THE COMPUTATIONAL TESTS' "$out" 2>/dev/null || true)
    fi
    nf=$(grep -c -E 'FAIL|\*\*\*\*' "$out" 2>/dev/null || true)
    np=${np:-0}; nf=${nf:-0}
    if [ "$rc" -ne 0 ] || [ "$np" -eq 0 ]; then
      echo "FAIL $drv: exit=$rc pass_lines=$np"
      nf=$(( nf + 1 ))
    else
      echo "ok   $drv: $np routines passed, $nf failed"
    fi
    [ "$nf" -gt 0 ] && grep -E -B2 -A2 'FAIL|\*\*\*\*' "$out" 2>/dev/null | head -20
    passed=$(( passed + np )); failed=$(( failed + nf ))
  done
done
emit_ctrf blis-blastest "$passed" "$failed"
