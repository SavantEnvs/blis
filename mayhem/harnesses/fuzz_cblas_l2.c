/* CBLAS level-2 fuzzer: gemv gbmv ger(u/c) symv/hemv sbmv/hbmv spmv/hpmv trmv trsv tbmv tbsv
 * tpmv tpsv syr/her spr/hpr syr2/her2 spr2/hpr2, all four datatypes, both storage orders.
 *
 * Input: [op][type][order][trans][uplo][diag][M][N][KL][KU][incX][incY][ldslack] [alpha 16][beta 16]
 * then matrix/vector data. Enum values are always valid (cblas_xerbla exit()s on an invalid enum,
 * which is the documented CBLAS contract, not a bug); dims include 0 (cff3af48: RowMajor ConjTrans
 * with M==0 looped forever in [cz]g[eb]mv). Buffers are exactly the size the BLAS contract allows
 * the routine to touch. */
#include "fuzz_common.h"
#include <cblas.h>

static enum CBLAS_ORDER ord_of(int v) { return v ? CblasColMajor : CblasRowMajor; }
static enum CBLAS_TRANSPOSE tr_of(int v) { return v == 0 ? CblasNoTrans : v == 1 ? CblasTrans : CblasConjTrans; }

/* elements of a logical r x c matrix stored with leading dimension ld in order o */
static size_t mat_len(enum CBLAS_ORDER o, int r, int c, int ld) {
  if (r <= 0 || c <= 0) return 0;
  return o == CblasColMajor ? (size_t)ld * (c - 1) + r : (size_t)ld * (r - 1) + c;
}
static int min_ld(enum CBLAS_ORDER o, int r, int c) { return imax(1, o == CblasColMajor ? r : c); }

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  rd_t r = {data, size};
  int op = rd_range(&r, 12), t = rd_range(&r, 4);
  enum CBLAS_ORDER o = ord_of(rd_range(&r, 2));
  enum CBLAS_TRANSPOSE tr = tr_of(rd_range(&r, 3));
  enum CBLAS_UPLO up = rd_range(&r, 2) ? CblasLower : CblasUpper;
  enum CBLAS_DIAG dg = rd_range(&r, 2) ? CblasUnit : CblasNonUnit;
  int M = rd_range(&r, 17), N = rd_range(&r, 17), KL = rd_range(&r, 5), KU = rd_range(&r, 5);
  int incX = rd_inc(&r, 1), incY = rd_inc(&r, 1), slack = rd_range(&r, 4);
  scal_t al, be;
  rd_scal(&r, &al);
  rd_scal(&r, &be);
  size_t es = ESZ[t];
  int cplx = t >= 2;
  void *A = NULL, *X = NULL, *Y = NULL;
  int lda;

  switch (op) {
  case 0: { /* gemv */
    int nx = tr == CblasNoTrans ? N : M, ny = tr == CblasNoTrans ? M : N;
    lda = min_ld(o, M, N) + slack;
    A = rd_buf(&r, mat_len(o, M, N, lda), es);
    X = rd_buf(&r, vec_len(nx, incX), es);
    Y = rd_buf(&r, vec_len(ny, incY), es);
    if (t == 0) cblas_sgemv(o, tr, M, N, al.f[0], A, lda, X, incX, be.f[0], Y, incY);
    else if (t == 1) cblas_dgemv(o, tr, M, N, al.d[0], A, lda, X, incX, be.d[0], Y, incY);
    else if (t == 2) cblas_cgemv(o, tr, M, N, &al, A, lda, X, incX, &be, Y, incY);
    else cblas_zgemv(o, tr, M, N, &al, A, lda, X, incX, &be, Y, incY);
    break;
  }
  case 1: { /* gbmv: band storage, lda >= KL+KU+1, (cols-1)*lda + KL+KU+1 elements */
    int nx = tr == CblasNoTrans ? N : M, ny = tr == CblasNoTrans ? M : N;
    int cols = o == CblasColMajor ? N : M;
    lda = KL + KU + 1 + slack;
    A = rd_buf(&r, (M > 0 && N > 0) ? (size_t)lda * (cols - 1) + KL + KU + 1 : 0, es);
    X = rd_buf(&r, vec_len(nx, incX), es);
    Y = rd_buf(&r, vec_len(ny, incY), es);
    if (t == 0) cblas_sgbmv(o, tr, M, N, KL, KU, al.f[0], A, lda, X, incX, be.f[0], Y, incY);
    else if (t == 1) cblas_dgbmv(o, tr, M, N, KL, KU, al.d[0], A, lda, X, incX, be.d[0], Y, incY);
    else if (t == 2) cblas_cgbmv(o, tr, M, N, KL, KU, &al, A, lda, X, incX, &be, Y, incY);
    else cblas_zgbmv(o, tr, M, N, KL, KU, &al, A, lda, X, incX, &be, Y, incY);
    break;
  }
  case 2: { /* ger / geru / gerc */
    lda = min_ld(o, M, N) + slack;
    X = rd_buf(&r, vec_len(M, incX), es);
    Y = rd_buf(&r, vec_len(N, incY), es);
    A = rd_buf(&r, mat_len(o, M, N, lda), es);
    int c = KL & 1;
    if (t == 0) cblas_sger(o, M, N, al.f[0], X, incX, Y, incY, A, lda);
    else if (t == 1) cblas_dger(o, M, N, al.d[0], X, incX, Y, incY, A, lda);
    else if (t == 2) (c ? cblas_cgerc : cblas_cgeru)(o, M, N, &al, X, incX, Y, incY, A, lda);
    else (c ? cblas_zgerc : cblas_zgeru)(o, M, N, &al, X, incX, Y, incY, A, lda);
    break;
  }
  case 3: { /* symv / hemv */
    lda = imax(1, N) + slack;
    A = rd_buf(&r, mat_len(o, N, N, lda), es);
    X = rd_buf(&r, vec_len(N, incX), es);
    Y = rd_buf(&r, vec_len(N, incY), es);
    if (t == 0) cblas_ssymv(o, up, N, al.f[0], A, lda, X, incX, be.f[0], Y, incY);
    else if (t == 1) cblas_dsymv(o, up, N, al.d[0], A, lda, X, incX, be.d[0], Y, incY);
    else if (t == 2) cblas_chemv(o, up, N, &al, A, lda, X, incX, &be, Y, incY);
    else cblas_zhemv(o, up, N, &al, A, lda, X, incX, &be, Y, incY);
    break;
  }
  case 4: { /* sbmv / hbmv, K = KL */
    int K = KL;
    lda = K + 1 + slack;
    A = rd_buf(&r, N > 0 ? (size_t)lda * (N - 1) + K + 1 : 0, es);
    X = rd_buf(&r, vec_len(N, incX), es);
    Y = rd_buf(&r, vec_len(N, incY), es);
    if (t == 0) cblas_ssbmv(o, up, N, K, al.f[0], A, lda, X, incX, be.f[0], Y, incY);
    else if (t == 1) cblas_dsbmv(o, up, N, K, al.d[0], A, lda, X, incX, be.d[0], Y, incY);
    else if (t == 2) cblas_chbmv(o, up, N, K, &al, A, lda, X, incX, &be, Y, incY);
    else cblas_zhbmv(o, up, N, K, &al, A, lda, X, incX, &be, Y, incY);
    break;
  }
  case 5: { /* spmv / hpmv: packed N(N+1)/2 */
    A = rd_buf(&r, (size_t)N * (N + 1) / 2, es);
    X = rd_buf(&r, vec_len(N, incX), es);
    Y = rd_buf(&r, vec_len(N, incY), es);
    if (t == 0) cblas_sspmv(o, up, N, al.f[0], A, X, incX, be.f[0], Y, incY);
    else if (t == 1) cblas_dspmv(o, up, N, al.d[0], A, X, incX, be.d[0], Y, incY);
    else if (t == 2) cblas_chpmv(o, up, N, &al, A, X, incX, &be, Y, incY);
    else cblas_zhpmv(o, up, N, &al, A, X, incX, &be, Y, incY);
    break;
  }
  case 6: { /* trmv / trsv */
    int s = KU & 1;
    lda = imax(1, N) + slack;
    A = rd_buf(&r, mat_len(o, N, N, lda), es);
    X = rd_buf(&r, vec_len(N, incX), es);
    if (t == 0) (s ? cblas_strsv : cblas_strmv)(o, up, tr, dg, N, A, lda, X, incX);
    else if (t == 1) (s ? cblas_dtrsv : cblas_dtrmv)(o, up, tr, dg, N, A, lda, X, incX);
    else if (t == 2) (s ? cblas_ctrsv : cblas_ctrmv)(o, up, tr, dg, N, A, lda, X, incX);
    else (s ? cblas_ztrsv : cblas_ztrmv)(o, up, tr, dg, N, A, lda, X, incX);
    break;
  }
  case 7: { /* tbmv / tbsv, K = KL */
    int s = KU & 1, K = KL;
    lda = K + 1 + slack;
    A = rd_buf(&r, N > 0 ? (size_t)lda * (N - 1) + K + 1 : 0, es);
    X = rd_buf(&r, vec_len(N, incX), es);
    if (t == 0) (s ? cblas_stbsv : cblas_stbmv)(o, up, tr, dg, N, K, A, lda, X, incX);
    else if (t == 1) (s ? cblas_dtbsv : cblas_dtbmv)(o, up, tr, dg, N, K, A, lda, X, incX);
    else if (t == 2) (s ? cblas_ctbsv : cblas_ctbmv)(o, up, tr, dg, N, K, A, lda, X, incX);
    else (s ? cblas_ztbsv : cblas_ztbmv)(o, up, tr, dg, N, K, A, lda, X, incX);
    break;
  }
  case 8: { /* tpmv / tpsv */
    int s = KU & 1;
    A = rd_buf(&r, (size_t)N * (N + 1) / 2, es);
    X = rd_buf(&r, vec_len(N, incX), es);
    if (t == 0) (s ? cblas_stpsv : cblas_stpmv)(o, up, tr, dg, N, A, X, incX);
    else if (t == 1) (s ? cblas_dtpsv : cblas_dtpmv)(o, up, tr, dg, N, A, X, incX);
    else if (t == 2) (s ? cblas_ctpsv : cblas_ctpmv)(o, up, tr, dg, N, A, X, incX);
    else (s ? cblas_ztpsv : cblas_ztpmv)(o, up, tr, dg, N, A, X, incX);
    break;
  }
  case 9: { /* syr / her (real alpha) */
    lda = imax(1, N) + slack;
    X = rd_buf(&r, vec_len(N, incX), es);
    A = rd_buf(&r, mat_len(o, N, N, lda), es);
    if (t == 0) cblas_ssyr(o, up, N, al.f[0], X, incX, A, lda);
    else if (t == 1) cblas_dsyr(o, up, N, al.d[0], X, incX, A, lda);
    else if (t == 2) cblas_cher(o, up, N, al.f[0], X, incX, A, lda);
    else cblas_zher(o, up, N, al.d[0], X, incX, A, lda);
    break;
  }
  case 10: { /* spr / hpr */
    X = rd_buf(&r, vec_len(N, incX), es);
    A = rd_buf(&r, (size_t)N * (N + 1) / 2, es);
    if (t == 0) cblas_sspr(o, up, N, al.f[0], X, incX, A);
    else if (t == 1) cblas_dspr(o, up, N, al.d[0], X, incX, A);
    else if (t == 2) cblas_chpr(o, up, N, al.f[0], X, incX, A);
    else cblas_zhpr(o, up, N, al.d[0], X, incX, A);
    break;
  }
  default: { /* syr2 / her2 and spr2 / hpr2 */
    int packed = KU & 1;
    lda = imax(1, N) + slack;
    X = rd_buf(&r, vec_len(N, incX), es);
    Y = rd_buf(&r, vec_len(N, incY), es);
    A = rd_buf(&r, packed ? (size_t)N * (N + 1) / 2 : mat_len(o, N, N, lda), es);
    if (packed) {
      if (t == 0) cblas_sspr2(o, up, N, al.f[0], X, incX, Y, incY, A);
      else if (t == 1) cblas_dspr2(o, up, N, al.d[0], X, incX, Y, incY, A);
      else if (t == 2) cblas_chpr2(o, up, N, &al, X, incX, Y, incY, A);
      else cblas_zhpr2(o, up, N, &al, X, incX, Y, incY, A);
    } else {
      if (t == 0) cblas_ssyr2(o, up, N, al.f[0], X, incX, Y, incY, A, lda);
      else if (t == 1) cblas_dsyr2(o, up, N, al.d[0], X, incX, Y, incY, A, lda);
      else if (t == 2) cblas_cher2(o, up, N, &al, X, incX, Y, incY, A, lda);
      else cblas_zher2(o, up, N, &al, X, incX, Y, incY, A, lda);
    }
    break;
  }
  }
  (void)cplx;
  free(A);
  free(X);
  free(Y);
  return 0;
}
