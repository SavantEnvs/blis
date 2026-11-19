/* CBLAS level-1 fuzzer: axpy axpby copy swap scal dot/dotu/dotc/sdsdot/dsdot nrm2 asum i?amax
 * rot rotm rotg rotmg, four datatypes, N 0..64, increments -3..3 (0 included: legal for some
 * routines, rejected or degenerate for others). Raw scalar/data bits so NaN/Inf/denormals reach
 * the overflow-safe scaling in nrm2 and the rotm/rotmg flag handling.
 *
 * Input: [op][type][N][incX][incY] [alpha 16][beta 16] [param 5 x 8] then vector data. */
#include "fuzz_common.h"
#include <cblas.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  rd_t r = {data, size};
  int op = rd_range(&r, 12), t = rd_range(&r, 4), N = rd_range(&r, 65);
  int incX = rd_inc(&r, 1), incY = rd_inc(&r, 1);
  scal_t al, be, res;
  double P[5];
  float Pf[5];
  rd_scal(&r, &al);
  rd_scal(&r, &be);
  rd_fill(&r, P, sizeof P);
  memcpy(Pf, P, sizeof Pf);
  size_t es = ESZ[t];
  void *X = rd_buf(&r, vec_len(N, incX), es);
  void *Y = rd_buf(&r, vec_len(N, incY), es);
  volatile double sink = 0;

  switch (op) {
  case 0: /* axpy */
    if (t == 0) cblas_saxpy(N, al.f[0], X, incX, Y, incY);
    else if (t == 1) cblas_daxpy(N, al.d[0], X, incX, Y, incY);
    else if (t == 2) cblas_caxpy(N, &al, X, incX, Y, incY);
    else cblas_zaxpy(N, &al, X, incX, Y, incY);
    break;
  case 1: /* axpby (BLIS extension) */
    if (t == 0) cblas_saxpby(N, al.f[0], X, incX, be.f[0], Y, incY);
    else if (t == 1) cblas_daxpby(N, al.d[0], X, incX, be.d[0], Y, incY);
    else if (t == 2) cblas_caxpby(N, &al, X, incX, &be, Y, incY);
    else cblas_zaxpby(N, &al, X, incX, &be, Y, incY);
    break;
  case 2: /* copy */
    if (t == 0) cblas_scopy(N, X, incX, Y, incY);
    else if (t == 1) cblas_dcopy(N, X, incX, Y, incY);
    else if (t == 2) cblas_ccopy(N, X, incX, Y, incY);
    else cblas_zcopy(N, X, incX, Y, incY);
    break;
  case 3: /* swap */
    if (t == 0) cblas_sswap(N, X, incX, Y, incY);
    else if (t == 1) cblas_dswap(N, X, incX, Y, incY);
    else if (t == 2) cblas_cswap(N, X, incX, Y, incY);
    else cblas_zswap(N, X, incX, Y, incY);
    break;
  case 4: /* scal, and the real-alpha complex forms */
    if (t == 0) cblas_sscal(N, al.f[0], X, incX);
    else if (t == 1) cblas_dscal(N, al.d[0], X, incX);
    else if (t == 2) { if (be.b[0] & 1) cblas_csscal(N, al.f[0], X, incX); else cblas_cscal(N, &al, X, incX); }
    else { if (be.b[0] & 1) cblas_zdscal(N, al.d[0], X, incX); else cblas_zscal(N, &al, X, incX); }
    break;
  case 5: /* dot family */
    if (t == 0) sink = (be.b[0] & 1) ? cblas_sdsdot(N, al.f[0], X, incX, Y, incY) : cblas_sdot(N, X, incX, Y, incY);
    else if (t == 1) sink = (be.b[0] & 1) ? cblas_dsdot(N, X, incX, Y, incY) : cblas_ddot(N, X, incX, Y, incY);
    else if (t == 2) { if (be.b[0] & 1) cblas_cdotc_sub(N, X, incX, Y, incY, &res); else cblas_cdotu_sub(N, X, incX, Y, incY, &res); }
    else { if (be.b[0] & 1) cblas_zdotc_sub(N, X, incX, Y, incY, &res); else cblas_zdotu_sub(N, X, incX, Y, incY, &res); }
    break;
  case 6: /* nrm2 */
    if (t == 0) sink = cblas_snrm2(N, X, incX);
    else if (t == 1) sink = cblas_dnrm2(N, X, incX);
    else if (t == 2) sink = cblas_scnrm2(N, X, incX);
    else sink = cblas_dznrm2(N, X, incX);
    break;
  case 7: /* asum */
    if (t == 0) sink = cblas_sasum(N, X, incX);
    else if (t == 1) sink = cblas_dasum(N, X, incX);
    else if (t == 2) sink = cblas_scasum(N, X, incX);
    else sink = cblas_dzasum(N, X, incX);
    break;
  case 8: /* i?amax */
    if (t == 0) sink = cblas_isamax(N, X, incX);
    else if (t == 1) sink = cblas_idamax(N, X, incX);
    else if (t == 2) sink = cblas_icamax(N, X, incX);
    else sink = cblas_izamax(N, X, incX);
    break;
  case 9: /* rot (c, s from alpha/beta) */
    if (t == 0) cblas_srot(N, X, incX, Y, incY, al.f[0], be.f[0]);
    else if (t == 1) cblas_drot(N, X, incX, Y, incY, al.d[0], be.d[0]);
    else if (t == 2) cblas_csrot(N, X, incX, Y, incY, al.f[0], be.f[0]);
    else cblas_zdrot(N, X, incX, Y, incY, al.d[0], be.d[0]);
    break;
  case 10: /* rotm: P[0] is the flag */
    if (t & 1) cblas_drotm(N, X, incX, Y, incY, P);
    else cblas_srotm(N, X, incX, Y, incY, Pf);
    break;
  default: { /* rotg / rotmg on scalars */
    scal_t a = al, b = be, s;
    float cf; double cd;
    if (t == 0) { if (N & 1) cblas_srotmg(&a.f[0], &a.f[1], &b.f[0], b.f[1], Pf); else cblas_srotg(&a.f[0], &b.f[0], &cf, &s.f[0]); }
    else if (t == 1) { if (N & 1) cblas_drotmg(&a.d[0], &a.d[1], &b.d[0], b.d[1], P); else cblas_drotg(&a.d[0], &b.d[0], &cd, &s.d[0]); }
    else if (t == 2) cblas_crotg(&a, &b, &cf, &s);
    else cblas_zrotg(&a, &b, &cd, &s);
    break;
  }
  }
  (void)sink;
  free(X);
  free(Y);
  return 0;
}
