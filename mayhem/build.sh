#!/usr/bin/env bash
#
# mayhem/build.sh — build BLIS twice, out of tree under mayhem/:
#   1) mayhem/out-fuzz: the haswell config (AVX2 asm kernels, incl. the "sup" millikernels) with
#      ASan+UBSan+SanCov and DWARF 3, linked into each libFuzzer harness and its -standalone twin.
#   2) mayhem/out-test: the same config with BLIS's normal flags, plus the upstream netlib BLAS test
#      drivers (blastest/), which mayhem/test.sh runs.
# Threading is disabled (-t none) so every run is deterministic. Re-runnable and offline: configure
# only regenerates config files in the build dir, make is incremental.
set -euo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
: "${SANITIZER_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer}"
: "${DEBUG_FLAGS:=-g -gdwarf-3}"
: "${CC:=clang}" ; : "${CXX:=clang++}" ; : "${LIB_FUZZING_ENGINE:=-fsanitize=fuzzer}"
: "${MAYHEM_JOBS:=$(nproc)}"
: "${COVERAGE_FLAGS=}"
: "${BLIS_CONFIG:=haswell}"
export SANITIZER_FLAGS DEBUG_FLAGS CC CXX LIB_FUZZING_ENGINE MAYHEM_JOBS COVERAGE_FLAGS

cd "$SRC"
HARN="$SRC/mayhem/harnesses"
FUZZ_TARGETS="fuzz_cblas_l1 fuzz_cblas_l2 fuzz_cblas_l3 fuzz_blis_typed fuzz_blis_l1m fuzz_blis_obj"

# SanCov on the library too (not only the harness), so libFuzzer/Mayhem see library edges.
COV_FLAGS=""
[ -n "$LIB_FUZZING_ENGINE" ] && COV_FLAGS="-fsanitize=fuzzer-no-link"

# configure_blis <dir> <cflags>: (re)configure only when the flags changed, so re-runs are incremental.
configure_blis() {
  local dir="$1" cflags="$2"
  mkdir -p "$dir"
  if [ ! -f "$dir/config.mk" ] || [ "$(cat "$dir/.mayhem-cflags" 2>/dev/null)" != "$CC|$cflags" ]; then
    (cd "$dir" && CC="$CC" CFLAGS="$cflags" "$SRC/configure" --enable-cblas --disable-shared \
        -t none "$BLIS_CONFIG")
    echo "$CC|$cflags" > "$dir/.mayhem-cflags"
  fi
}

# UBSan's -fsanitize=function flags every call BLIS makes through its type-erased dispatch tables
# (void_fp function pointers cast back to the per-datatype signature, e.g. bli_zeqsc via
# bli_util_oapi.c, bli_sgetsc via bli_l0_oapi.c) — that is the framework's design, hit on nearly
# every input. Relax only that one check; ASan and the rest of UBSan stay on and halting.
UBSAN_RELAX=""
case "$SANITIZER_FLAGS" in *undefined*) UBSAN_RELAX="-fno-sanitize=function" ;; esac
SAN="$SANITIZER_FLAGS $UBSAN_RELAX"

# The haswell optimized kernels are inline asm that clobbers every GPR, rbp included. Any clang
# instrumentation (ASan, UBSan or SanCov) makes clang keep rbp as the frame pointer in those
# functions, so after the asm block the epilogue/argument reloads go through a clobbered rbp and the
# process dies with a bogus SEGV/"stack-overflow" on the first sup gemm call (verified: only
# -fno-sanitize=all removes the frame pointer). So the optimized-kernel set (BLIS's
# CKOPTFLAGS.<config>; reference kernels and the whole framework keep full instrumentation) is
# compiled uninstrumented — instrumentation cannot see inside inline asm anyway. Kernel faults still
# surface as SEGVs, and ASan checks every framework/harness access around them.
KERNEL_MAKEVARS=("CKOPTFLAGS.${BLIS_CONFIG}=-O2 -O3 -fomit-frame-pointer -fno-sanitize=all")

# 1) sanitized + instrumented library and harnesses
FZ="$SRC/mayhem/out-fuzz"
configure_blis "$FZ" "$SAN $COV_FLAGS $DEBUG_FLAGS"
make -C "$FZ" -j"$MAYHEM_JOBS" "${KERNEL_MAKEVARS[@]}" libs
LIB="$FZ/lib/$BLIS_CONFIG/libblis.a"
INC="-I$FZ/include/$BLIS_CONFIG -I$HARN"

$CC $SAN $DEBUG_FLAGS -c "$SRC/mayhem/lsan_off.c" -o "$FZ/lsan_off.o"
$CC $SAN $DEBUG_FLAGS -c "$STANDALONE_FUZZ_MAIN" -o "$FZ/standalone_main.o"
for t in $FUZZ_TARGETS; do
  $CC $SAN $DEBUG_FLAGS $INC -c "$HARN/$t.c" -o "$FZ/$t.o"
  $CC $SAN $DEBUG_FLAGS $LIB_FUZZING_ENGINE "$FZ/$t.o" "$FZ/lsan_off.o" "$LIB" \
      -lm -lpthread -o "/mayhem/$t"
  $CC $SAN $DEBUG_FLAGS "$FZ/standalone_main.o" "$FZ/$t.o" "$FZ/lsan_off.o" "$LIB" \
      -lm -lpthread -o "/mayhem/$t-standalone"
done

# 2) the functional oracle: normal flags, upstream BLAS test drivers (blastest/*.x)
TS="$SRC/mayhem/out-test"
(
  unset CFLAGS LDFLAGS
  configure_blis "$TS" "$COVERAGE_FLAGS"
  make -C "$TS" -j"$MAYHEM_JOBS" libs
  make -C "$TS" -j"$MAYHEM_JOBS" blastest-bin
)
ls "$TS"/obj/"$BLIS_CONFIG"/blastest/*.x >/dev/null
echo "build.sh: done"
