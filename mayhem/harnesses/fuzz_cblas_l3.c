/* CBLAS level-3 fuzzer: gemm symm/hemm syrk/herk syr2k/her2k trmm trsm gemmt, four datatypes,
 * both storage orders, small dims (0..40) so BLIS's small/unpacked "sup" path (haswell asm
 * millikernels, fix a822cb2e) and the conventional packed path are both reached.
 *
 * Input: [op][type][order][transa][transb][side][uplo][diag][M][N][K][slackA][slackB][slackC]
 * [alpha 16][beta 16] then matrix data. Valid enums only (cblas_xerbla exit()s otherwise);
 * matrices are exactly ld*(outer-1)+inner elements. */
#include "fuzz_common.h"
#include <cblas.h>

static enum CBLAS_TRANSPOSE tr_of(int v) { return v == 0 ? CblasNoTrans : v == 1 ? CblasTrans : CblasConjTrans; }
static size_t mat_len(enum CBLAS_ORDER o, int r, int c, int ld) {
  if (r <= 0 || c <= 0) return 0;
  return o == CblasColMajor ? (size_t)ld * (c - 1) + r : (size_t)ld * (r - 1) + c;
}
static int min_ld(enum CBLAS_ORDER o, int r, int c) { return imax(1, o == CblasColMajor ? r : c); }

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  rd_t r = {data, size};
  int op = rd_range(&r, 8), t = rd_range(&r, 4);
  enum CBLAS_ORDER o = rd_range(&r, 2) ? CblasColMajor : CblasRowMajor;
  enum CBLAS_TRANSPOSE ta = tr_of(rd_range(&r, 3)), tb = tr_of(rd_range(&r, 3));
  enum CBLAS_SIDE sd = rd_range(&r, 2) ? CblasRight : CblasLeft;
  enum CBLAS_UPLO up = rd_range(&r, 2) ? CblasLower : CblasUpper;
  enum CBLAS_DIAG dg = rd_range(&r, 2) ? CblasUnit : CblasNonUnit;
  int M = rd_range(&r, 41), N = rd_range(&r, 41), K = rd_range(&r, 41);
  int sa = rd_range(&r, 4), sb = rd_range(&r, 4), sc = rd_range(&r, 4);
  scal_t al, be;
  rd_scal(&r, &al);
  rd_scal(&r, &be);
  size_t es = ESZ[t];
  void *A = NULL, *B = NULL, *C = NULL;
  int lda, ldb, ldc;
  /* herk/her2k (and syrk/syr2k for real) only accept NoTrans or (Conj)Trans per their contract;
   * a complex syrk with ConjTrans is illegal -> parameter error path (prints, returns). */

  switch (op) {
  case 0: { /* gemm: C(MxN) = op(A) op(B) */
    int ar = ta == CblasNoTrans ? M : K, ac = ta == CblasNoTrans ? K : M;
    int br = tb == CblasNoTrans ? K : N, bc = tb == CblasNoTrans ? N : K;
    lda = min_ld(o, ar, ac) + sa; ldb = min_ld(o, br, bc) + sb; ldc = min_ld(o, M, N) + sc;
    A = rd_buf(&r, mat_len(o, ar, ac, lda), es);
    B = rd_buf(&r, mat_len(o, br, bc, ldb), es);
    C = rd_buf(&r, mat_len(o, M, N, ldc), es);
    if (t == 0) cblas_sgemm(o, ta, tb, M, N, K, al.f[0], A, lda, B, ldb, be.f[0], C, ldc);
    else if (t == 1) cblas_dgemm(o, ta, tb, M, N, K, al.d[0], A, lda, B, ldb, be.d[0], C, ldc);
    else if (t == 2) cblas_cgemm(o, ta, tb, M, N, K, &al, A, lda, B, ldb, &be, C, ldc);
    else cblas_zgemm(o, ta, tb, M, N, K, &al, A, lda, B, ldb, &be, C, ldc);
    break;
  }
  case 1: { /* gemmt: C(NxN) triangle = op(A)(NxK) op(B)(KxN) */
    int ar = ta == CblasNoTrans ? N : K, ac = ta == CblasNoTrans ? K : N;
    int br = tb == CblasNoTrans ? K : N, bc = tb == CblasNoTrans ? N : K;
    lda = min_ld(o, ar, ac) + sa; ldb = min_ld(o, br, bc) + sb; ldc = imax(1, N) + sc;
    A = rd_buf(&r, mat_len(o, ar, ac, lda), es);
    B = rd_buf(&r, mat_len(o, br, bc, ldb), es);
    C = rd_buf(&r, mat_len(o, N, N, ldc), es);
    if (t == 0) cblas_sgemmt(o, up, ta, tb, N, K, al.f[0], A, lda, B, ldb, be.f[0], C, ldc);
    else if (t == 1) cblas_dgemmt(o, up, ta, tb, N, K, al.d[0], A, lda, B, ldb, be.d[0], C, ldc);
    else if (t == 2) cblas_cgemmt(o, up, ta, tb, N, K, &al, A, lda, B, ldb, &be, C, ldc);
    else cblas_zgemmt(o, up, ta, tb, N, K, &al, A, lda, B, ldb, &be, C, ldc);
    break;
  }
  case 2: { /* symm / hemm: A is MxM (Left) or NxN (Right) */
    int ka = sd == CblasLeft ? M : N, herm = sa & 1;
    lda = imax(1, ka) + sa; ldb = min_ld(o, M, N) + sb; ldc = min_ld(o, M, N) + sc;
    A = rd_buf(&r, mat_len(o, ka, ka, lda), es);
    B = rd_buf(&r, mat_len(o, M, N, ldb), es);
    C = rd_buf(&r, mat_len(o, M, N, ldc), es);
    if (t == 0) cblas_ssymm(o, sd, up, M, N, al.f[0], A, lda, B, ldb, be.f[0], C, ldc);
    else if (t == 1) cblas_dsymm(o, sd, up, M, N, al.d[0], A, lda, B, ldb, be.d[0], C, ldc);
    else if (t == 2) (herm ? cblas_chemm : cblas_csymm)(o, sd, up, M, N, &al, A, lda, B, ldb, &be, C, ldc);
    else (herm ? cblas_zhemm : cblas_zsymm)(o, sd, up, M, N, &al, A, lda, B, ldb, &be, C, ldc);
    break;
  }
  case 3: { /* syrk / herk: C(NxN) = op(A) op(A)^T, A NxK (NoTrans) or KxN */
    int ar = ta == CblasNoTrans ? N : K, ac = ta == CblasNoTrans ? K : N, herm = sa & 1;
    lda = min_ld(o, ar, ac) + sa; ldc = imax(1, N) + sc;
    A = rd_buf(&r, mat_len(o, ar, ac, lda), es);
    C = rd_buf(&r, mat_len(o, N, N, ldc), es);
    if (t == 0) cblas_ssyrk(o, up, ta, N, K, al.f[0], A, lda, be.f[0], C, ldc);
    else if (t == 1) cblas_dsyrk(o, up, ta, N, K, al.d[0], A, lda, be.d[0], C, ldc);
    else if (t == 2) {
      if (herm) cblas_cherk(o, up, ta, N, K, al.f[0], A, lda, be.f[0], C, ldc);
      else cblas_csyrk(o, up, ta, N, K, &al, A, lda, &be, C, ldc);
    } else {
      if (herm) cblas_zherk(o, up, ta, N, K, al.d[0], A, lda, be.d[0], C, ldc);
      else cblas_zsyrk(o, up, ta, N, K, &al, A, lda, &be, C, ldc);
    }
    break;
  }
  case 4: { /* syr2k / her2k */
    int ar = ta == CblasNoTrans ? N : K, ac = ta == CblasNoTrans ? K : N, herm = sa & 1;
    lda = min_ld(o, ar, ac) + sa; ldb = min_ld(o, ar, ac) + sb; ldc = imax(1, N) + sc;
    A = rd_buf(&r, mat_len(o, ar, ac, lda), es);
    B = rd_buf(&r, mat_len(o, ar, ac, ldb), es);
    C = rd_buf(&r, mat_len(o, N, N, ldc), es);
    if (t == 0) cblas_ssyr2k(o, up, ta, N, K, al.f[0], A, lda, B, ldb, be.f[0], C, ldc);
    else if (t == 1) cblas_dsyr2k(o, up, ta, N, K, al.d[0], A, lda, B, ldb, be.d[0], C, ldc);
    else if (t == 2) {
      if (herm) cblas_cher2k(o, up, ta, N, K, &al, A, lda, B, ldb, be.f[0], C, ldc);
      else cblas_csyr2k(o, up, ta, N, K, &al, A, lda, B, ldb, &be, C, ldc);
    } else {
      if (herm) cblas_zher2k(o, up, ta, N, K, &al, A, lda, B, ldb, be.d[0], C, ldc);
      else cblas_zsyr2k(o, up, ta, N, K, &al, A, lda, B, ldb, &be, C, ldc);
    }
    break;
  }
  default: { /* trmm / trsm: B(MxN) = op(A) B or B op(A) */
    int ka = sd == CblasLeft ? M : N, solve = op & 1;
    lda = imax(1, ka) + sa; ldb = min_ld(o, M, N) + sb;
    A = rd_buf(&r, mat_len(o, ka, ka, lda), es);
    B = rd_buf(&r, mat_len(o, M, N, ldb), es);
    if (t == 0) (solve ? cblas_strsm : cblas_strmm)(o, sd, up, ta, dg, M, N, al.f[0], A, lda, B, ldb);
    else if (t == 1) (solve ? cblas_dtrsm : cblas_dtrmm)(o, sd, up, ta, dg, M, N, al.d[0], A, lda, B, ldb);
    else if (t == 2) (solve ? cblas_ctrsm : cblas_ctrmm)(o, sd, up, ta, dg, M, N, &al, A, lda, B, ldb);
    else (solve ? cblas_ztrsm : cblas_ztrmm)(o, sd, up, ta, dg, M, N, &al, A, lda, B, ldb);
    break;
  }
  }
  free(A);
  free(B);
  free(C);
  return 0;
}
