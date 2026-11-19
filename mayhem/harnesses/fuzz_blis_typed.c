/* BLIS native typed-API fuzzer: bli_?gemm gemmt hemm symm herk her2k syrk syr2k trmm trmm3 trsm
 * and bli_?gemv ger hemv symv her syr her2 syr2 trmv trsv, four datatypes, with conj/trans
 * combinations and general (rs != 1 && cs != 1), row- and column-major strides that the
 * BLAS/CBLAS layer cannot express.
 *
 * Input: [op][type][transa][transb][conja][conjb][side][uplo][diag][m][n][k][stride modes 3]
 * [slacks 3][alpha 16][beta 16] then data. Operands are always conformal (BLIS error checking
 * aborts on non-conformal dims by design); each matrix buffer is exactly
 * (rows-1)*rs + (cols-1)*cs + 1 elements. */
#include "fuzz_common.h"
#include <blis.h>

typedef struct { inc_t rs, cs; size_t len; } st_t;

/* strides for a rows x cols matrix: 0 = column-major, 1 = row-major, 2 = general stride */
static st_t mk_st(int mode, dim_t rows, dim_t cols, int slack) {
  st_t s;
  if (mode == 0) { s.rs = 1; s.cs = (rows > 1 ? rows : 1) + slack; }
  else if (mode == 1) { s.cs = 1; s.rs = (cols > 1 ? cols : 1) + slack; }
  else { s.rs = 2; s.cs = 2 * (rows > 1 ? rows : 1) + 1 + slack; }
  s.len = (rows > 0 && cols > 0) ? (size_t)((rows - 1) * s.rs + (cols - 1) * s.cs + 1) : 0;
  return s;
}

#define GEN(ctype, ctype_r, ch)                                                                      \
static void run_##ch(rd_t *r, int op, trans_t ta, trans_t tb, conj_t ca, conj_t cb, side_t sd,      \
                     uplo_t up, diag_t dg, dim_t m, dim_t n, dim_t k, const int *md, const int *sl) { \
  ctype al, be;                                                                                      \
  ctype_r alr, ber;                                                                                  \
  rd_fill(r, &al, sizeof al); rd_fill(r, &be, sizeof be);                                            \
  memcpy(&alr, &al, sizeof alr); memcpy(&ber, &be, sizeof ber);                                      \
  ctype *A = NULL, *B = NULL, *C = NULL;                                                             \
  st_t sa, sb, sc;                                                                                   \
  size_t es = sizeof(ctype);                                                                         \
  int tra = bli_does_trans(ta), trb = bli_does_trans(tb);                                            \
  switch (op) {                                                                                      \
  case 0: /* gemm */                                                                                 \
    sa = mk_st(md[0], tra ? k : m, tra ? m : k, sl[0]);                                              \
    sb = mk_st(md[1], trb ? n : k, trb ? k : n, sl[1]);                                              \
    sc = mk_st(md[2], m, n, sl[2]);                                                                  \
    A = rd_buf(r, sa.len, es); B = rd_buf(r, sb.len, es); C = rd_buf(r, sc.len, es);                \
    bli_##ch##gemm(ta, tb, m, n, k, &al, A, sa.rs, sa.cs, B, sb.rs, sb.cs, &be, C, sc.rs, sc.cs);   \
    break;                                                                                           \
  case 1: /* gemmt / syr2k */                                                                        \
  case 2:                                                                                            \
    sa = mk_st(md[0], tra ? k : m, tra ? m : k, sl[0]);                                              \
    if (op == 1) sb = mk_st(md[1], trb ? m : k, trb ? k : m, sl[1]);                                 \
    else sb = mk_st(md[1], trb ? k : m, trb ? m : k, sl[1]);                                         \
    sc = mk_st(md[2], m, m, sl[2]);                                                                  \
    A = rd_buf(r, sa.len, es); B = rd_buf(r, sb.len, es); C = rd_buf(r, sc.len, es);                \
    if (op == 1) bli_##ch##gemmt(up, ta, tb, m, k, &al, A, sa.rs, sa.cs, B, sb.rs, sb.cs, &be, C, sc.rs, sc.cs); \
    else bli_##ch##syr2k(up, ta, tb, m, k, &al, A, sa.rs, sa.cs, B, sb.rs, sb.cs, &be, C, sc.rs, sc.cs);       \
    break;                                                                                           \
  case 3: /* her2k: A, B are m x k (or k x m) */                                                    \
    sa = mk_st(md[0], tra ? k : m, tra ? m : k, sl[0]);                                              \
    sb = mk_st(md[1], trb ? k : m, trb ? m : k, sl[1]);                                              \
    sc = mk_st(md[2], m, m, sl[2]);                                                                  \
    A = rd_buf(r, sa.len, es); B = rd_buf(r, sb.len, es); C = rd_buf(r, sc.len, es);                \
    bli_##ch##her2k(up, ta, tb, m, k, &al, A, sa.rs, sa.cs, B, sb.rs, sb.cs, &ber, C, sc.rs, sc.cs);\
    break;                                                                                           \
  case 4: /* herk / syrk */                                                                          \
    sa = mk_st(md[0], tra ? k : m, tra ? m : k, sl[0]);                                              \
    sc = mk_st(md[2], m, m, sl[2]);                                                                  \
    A = rd_buf(r, sa.len, es); C = rd_buf(r, sc.len, es);                                            \
    if (sl[1] & 1) bli_##ch##herk(up, ta, m, k, &alr, A, sa.rs, sa.cs, &ber, C, sc.rs, sc.cs);       \
    else bli_##ch##syrk(up, ta, m, k, &al, A, sa.rs, sa.cs, &be, C, sc.rs, sc.cs);                  \
    break;                                                                                           \
  case 5: { /* hemm / symm */                                                                        \
    dim_t ka = sd == BLIS_LEFT ? m : n;                                                              \
    sa = mk_st(md[0], ka, ka, sl[0]);                                                                \
    sb = mk_st(md[1], trb ? n : m, trb ? m : n, sl[1]);                                              \
    sc = mk_st(md[2], m, n, sl[2]);                                                                  \
    A = rd_buf(r, sa.len, es); B = rd_buf(r, sb.len, es); C = rd_buf(r, sc.len, es);                \
    if (k & 1) bli_##ch##hemm(sd, up, ca, tb, m, n, &al, A, sa.rs, sa.cs, B, sb.rs, sb.cs, &be, C, sc.rs, sc.cs); \
    else bli_##ch##symm(sd, up, ca, tb, m, n, &al, A, sa.rs, sa.cs, B, sb.rs, sb.cs, &be, C, sc.rs, sc.cs);       \
    break;                                                                                           \
  }                                                                                                  \
  case 6: { /* trmm / trsm (B in place) */                                                           \
    dim_t ka = sd == BLIS_LEFT ? m : n;                                                              \
    sa = mk_st(md[0], ka, ka, sl[0]);                                                                \
    sb = mk_st(md[1], m, n, sl[1]);                                                                  \
    A = rd_buf(r, sa.len, es); B = rd_buf(r, sb.len, es);                                            \
    if (k & 1) bli_##ch##trsm(sd, up, ta, dg, m, n, &al, A, sa.rs, sa.cs, B, sb.rs, sb.cs);          \
    else bli_##ch##trmm(sd, up, ta, dg, m, n, &al, A, sa.rs, sa.cs, B, sb.rs, sb.cs);                \
    break;                                                                                           \
  }                                                                                                  \
  case 7: { /* trmm3: C = op(A) op(B) */                                                             \
    dim_t ka = sd == BLIS_LEFT ? m : n;                                                              \
    sa = mk_st(md[0], ka, ka, sl[0]);                                                                \
    sb = mk_st(md[1], trb ? n : m, trb ? m : n, sl[1]);                                              \
    sc = mk_st(md[2], m, n, sl[2]);                                                                  \
    A = rd_buf(r, sa.len, es); B = rd_buf(r, sb.len, es); C = rd_buf(r, sc.len, es);                \
    bli_##ch##trmm3(sd, up, ta, dg, tb, m, n, &al, A, sa.rs, sa.cs, B, sb.rs, sb.cs, &be, C, sc.rs, sc.cs); \
    break;                                                                                           \
  }                                                                                                  \
  case 8: { /* gemv: x is n (or m), y is m (or n); vector incs from slack */                         \
    inc_t ix = 1 + sl[0], iy = 1 + sl[1];                                                            \
    sa = mk_st(md[0], m, n, sl[2]);                                                                  \
    A = rd_buf(r, sa.len, es);                                                                       \
    B = rd_buf(r, vec_len(tra ? m : n, ix), es); C = rd_buf(r, vec_len(tra ? n : m, iy), es);       \
    bli_##ch##gemv(ta, cb, m, n, &al, A, sa.rs, sa.cs, B, ix, &be, C, iy);                           \
    break;                                                                                           \
  }                                                                                                  \
  case 9: { /* ger */                                                                                \
    inc_t ix = 1 + sl[0], iy = 1 + sl[1];                                                            \
    sa = mk_st(md[0], m, n, sl[2]);                                                                  \
    B = rd_buf(r, vec_len(m, ix), es); C = rd_buf(r, vec_len(n, iy), es);                           \
    A = rd_buf(r, sa.len, es);                                                                       \
    bli_##ch##ger(ca, cb, m, n, &al, B, ix, C, iy, A, sa.rs, sa.cs);                                 \
    break;                                                                                           \
  }                                                                                                  \
  case 10: { /* hemv / symv */                                                                       \
    inc_t ix = 1 + sl[0], iy = 1 + sl[1];                                                            \
    sa = mk_st(md[0], m, m, sl[2]);                                                                  \
    A = rd_buf(r, sa.len, es); B = rd_buf(r, vec_len(m, ix), es); C = rd_buf(r, vec_len(m, iy), es);\
    if (k & 1) bli_##ch##hemv(up, ca, cb, m, &al, A, sa.rs, sa.cs, B, ix, &be, C, iy);               \
    else bli_##ch##symv(up, ca, cb, m, &al, A, sa.rs, sa.cs, B, ix, &be, C, iy);                    \
    break;                                                                                           \
  }                                                                                                  \
  case 11: { /* her / syr / her2 / syr2 */                                                           \
    inc_t ix = 1 + sl[0], iy = 1 + sl[1];                                                            \
    sa = mk_st(md[0], m, m, sl[2]);                                                                  \
    B = rd_buf(r, vec_len(m, ix), es); C = rd_buf(r, vec_len(m, iy), es); A = rd_buf(r, sa.len, es);\
    switch (k & 3) {                                                                                 \
    case 0: bli_##ch##her(up, ca, m, &alr, B, ix, A, sa.rs, sa.cs); break;                           \
    case 1: bli_##ch##syr(up, ca, m, &al, B, ix, A, sa.rs, sa.cs); break;                            \
    case 2: bli_##ch##her2(up, ca, cb, m, &al, B, ix, C, iy, A, sa.rs, sa.cs); break;                \
    default: bli_##ch##syr2(up, ca, cb, m, &al, B, ix, C, iy, A, sa.rs, sa.cs); break;               \
    }                                                                                                \
    break;                                                                                           \
  }                                                                                                  \
  default: { /* trmv / trsv */                                                                       \
    inc_t ix = 1 + sl[0];                                                                            \
    sa = mk_st(md[0], m, m, sl[2]);                                                                  \
    A = rd_buf(r, sa.len, es); B = rd_buf(r, vec_len(m, ix), es);                                    \
    if (k & 1) bli_##ch##trsv(up, ta, dg, m, &al, A, sa.rs, sa.cs, B, ix);                           \
    else bli_##ch##trmv(up, ta, dg, m, &al, A, sa.rs, sa.cs, B, ix);                                 \
    break;                                                                                           \
  }                                                                                                  \
  }                                                                                                  \
  free(A); free(B); free(C);                                                                         \
}

GEN(float, float, s)
GEN(double, double, d)
GEN(scomplex, float, c)
GEN(dcomplex, double, z)

static const trans_t TR[4] = {BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, BLIS_CONJ_NO_TRANSPOSE, BLIS_CONJ_TRANSPOSE};

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  rd_t r = {data, size};
  int op = rd_range(&r, 13), t = rd_range(&r, 4);
  trans_t ta = TR[rd_range(&r, 4)], tb = TR[rd_range(&r, 4)];
  conj_t ca = rd_range(&r, 2) ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE;
  conj_t cb = rd_range(&r, 2) ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE;
  side_t sd = rd_range(&r, 2) ? BLIS_RIGHT : BLIS_LEFT;
  uplo_t up = rd_range(&r, 2) ? BLIS_LOWER : BLIS_UPPER;
  diag_t dg = rd_range(&r, 2) ? BLIS_UNIT_DIAG : BLIS_NONUNIT_DIAG;
  dim_t m = rd_range(&r, 33), n = rd_range(&r, 33), k = rd_range(&r, 33);
  int md[3], sl[3];
  for (int i = 0; i < 3; i++) md[i] = rd_range(&r, 3);
  for (int i = 0; i < 3; i++) sl[i] = rd_range(&r, 3);
  switch (t) {
  case 0: run_s(&r, op, ta, tb, ca, cb, sd, up, dg, m, n, k, md, sl); break;
  case 1: run_d(&r, op, ta, tb, ca, cb, sd, up, dg, m, n, k, md, sl); break;
  case 2: run_c(&r, op, ta, tb, ca, cb, sd, up, dg, m, n, k, md, sl); break;
  default: run_z(&r, op, ta, tb, ca, cb, sd, up, dg, m, n, k, md, sl); break;
  }
  return 0;
}
