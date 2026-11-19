/* BLIS level-1m / level-1d / util typed-API fuzzer: bli_?addm copym subm axpym scal2m xpbym scalm
 * setm invscalm normfm norm1m normim mkherm mksymm mktrim, bli_?addd copyd subd axpyd scal2d xpbyd
 * setd scald shiftd invertd, and the vector norms/sumsqv. Exercises the structured (diagonal-offset,
 * uplo, unit-diagonal) matrix iteration and pruning that level-2/3 never reach.
 *
 * Input: [op][type][diagoff][diag][uplo][trans][m][n][stride modes 2][slacks 2][alpha 16][beta 16]
 * then data. Operands are conformal; each buffer is exactly (rows-1)*rs + (cols-1)*cs + 1. */
#include "fuzz_common.h"
#include <blis.h>

typedef struct { inc_t rs, cs; size_t len; } st_t;
static st_t mk_st(int mode, dim_t rows, dim_t cols, int slack) {
  st_t s;
  if (mode == 0) { s.rs = 1; s.cs = (rows > 1 ? rows : 1) + slack; }
  else if (mode == 1) { s.cs = 1; s.rs = (cols > 1 ? cols : 1) + slack; }
  else { s.rs = 2; s.cs = 2 * (rows > 1 ? rows : 1) + 1 + slack; }
  s.len = (rows > 0 && cols > 0) ? (size_t)((rows - 1) * s.rs + (cols - 1) * s.cs + 1) : 0;
  return s;
}

#define GEN(ctype, ctype_r, ch)                                                                     \
static void run_##ch(rd_t *r, int op, doff_t dof, diag_t dg, uplo_t up, trans_t tr, conj_t cj,    \
                     dim_t m, dim_t n, const int *md, const int *sl) {                              \
  ctype al, be;                                                                                     \
  ctype_r alr, nrm = 0, scale, sumsq;                                                               \
  rd_fill(r, &al, sizeof al); rd_fill(r, &be, sizeof be);                                           \
  memcpy(&alr, &al, sizeof alr); memcpy(&scale, &be, sizeof scale);                                 \
  memcpy(&sumsq, (char *)&be + sizeof(ctype) - sizeof sumsq, sizeof sumsq);                         \
  size_t es = sizeof(ctype);                                                                        \
  int trx = bli_does_trans(tr);                                                                     \
  st_t sx = mk_st(md[0], trx ? n : m, trx ? m : n, sl[0]), sy = mk_st(md[1], m, n, sl[1]);          \
  ctype *X = rd_buf(r, sx.len, es), *Y = NULL;                                                      \
  switch (op) {                                                                                     \
  case 0: case 1: case 2: case 3: case 4: case 5:                                                   \
    Y = rd_buf(r, sy.len, es);                                                                      \
    if (op == 0) bli_##ch##addm(dof, dg, up, tr, m, n, X, sx.rs, sx.cs, Y, sy.rs, sy.cs);           \
    else if (op == 1) bli_##ch##copym(dof, dg, up, tr, m, n, X, sx.rs, sx.cs, Y, sy.rs, sy.cs);     \
    else if (op == 2) bli_##ch##subm(dof, dg, up, tr, m, n, X, sx.rs, sx.cs, Y, sy.rs, sy.cs);      \
    else if (op == 3) bli_##ch##axpym(dof, dg, up, tr, m, n, &al, X, sx.rs, sx.cs, Y, sy.rs, sy.cs);\
    else if (op == 4) bli_##ch##scal2m(dof, dg, up, tr, m, n, &al, X, sx.rs, sx.cs, Y, sy.rs, sy.cs);\
    else bli_##ch##xpbym(dof, dg, up, tr, m, n, X, sx.rs, sx.cs, &be, Y, sy.rs, sy.cs);             \
    break;                                                                                          \
  case 6: case 7: case 8: case 9: case 10: case 11: case 12: case 13: case 14: case 15:             \
    Y = rd_buf(r, sy.len, es);                                                                      \
    if (op == 6) bli_##ch##addd(dof, dg, tr, m, n, X, sx.rs, sx.cs, Y, sy.rs, sy.cs);               \
    else if (op == 7) bli_##ch##copyd(dof, dg, tr, m, n, X, sx.rs, sx.cs, Y, sy.rs, sy.cs);         \
    else if (op == 8) bli_##ch##subd(dof, dg, tr, m, n, X, sx.rs, sx.cs, Y, sy.rs, sy.cs);          \
    else if (op == 9) bli_##ch##axpyd(dof, dg, tr, m, n, &al, X, sx.rs, sx.cs, Y, sy.rs, sy.cs);    \
    else if (op == 10) bli_##ch##scal2d(dof, dg, tr, m, n, &al, X, sx.rs, sx.cs, Y, sy.rs, sy.cs);  \
    else if (op == 11) bli_##ch##xpbyd(dof, dg, tr, m, n, X, sx.rs, sx.cs, &be, Y, sy.rs, sy.cs);   \
    else if (op == 12) bli_##ch##setd(cj, dof, m, n, &al, Y, sy.rs, sy.cs);                         \
    else if (op == 13) bli_##ch##scald(cj, dof, m, n, &al, Y, sy.rs, sy.cs);                        \
    else if (op == 14) bli_##ch##shiftd(dof, m, n, &al, Y, sy.rs, sy.cs);                           \
    else bli_##ch##invertd(dof, m, n, Y, sy.rs, sy.cs);                                             \
    break;                                                                                          \
  case 16: case 17: case 18:  /* in-place on an m x n matrix */                                    \
    Y = rd_buf(r, sy.len, es);                                                                      \
    if (op == 16) bli_##ch##scalm(cj, dof, dg, up, m, n, &al, Y, sy.rs, sy.cs);                     \
    else if (op == 17) bli_##ch##setm(cj, dof, dg, up, m, n, &al, Y, sy.rs, sy.cs);                 \
    else bli_##ch##invscalm(cj, dof, dg, up, m, n, &al, Y, sy.rs, sy.cs);                           \
    break;                                                                                          \
  case 19: case 20: case 21:                                                                        \
    Y = rd_buf(r, sy.len, es);                                                                      \
    if (op == 19) bli_##ch##normfm(dof, dg, up, m, n, Y, sy.rs, sy.cs, &nrm);                       \
    else if (op == 20) bli_##ch##norm1m(dof, dg, up, m, n, Y, sy.rs, sy.cs, &nrm);                  \
    else bli_##ch##normim(dof, dg, up, m, n, Y, sy.rs, sy.cs, &nrm);                                \
    break;                                                                                          \
  case 22: { /* mkherm / mksymm / mktrim on an m x m matrix */                                      \
    st_t sq = mk_st(md[1], m, m, sl[1]);                                                            \
    Y = rd_buf(r, sq.len, es);                                                                      \
    if (n % 3 == 0) bli_##ch##mkherm(up, m, Y, sq.rs, sq.cs);                                       \
    else if (n % 3 == 1) bli_##ch##mksymm(up, m, Y, sq.rs, sq.cs);                                  \
    else bli_##ch##mktrim(up, m, Y, sq.rs, sq.cs);                                                  \
    break;                                                                                          \
  }                                                                                                 \
  default: { /* vector norms / sumsqv: x is length m with increment 1+slack */                     \
    inc_t ix = 1 + sl[0];                                                                           \
    Y = rd_buf(r, vec_len(m, ix), es);                                                              \
    switch (n & 3) {                                                                                \
    case 0: bli_##ch##normfv(m, Y, ix, &nrm); break;                                                \
    case 1: bli_##ch##norm1v(m, Y, ix, &nrm); break;                                                \
    case 2: bli_##ch##normiv(m, Y, ix, &nrm); break;                                                \
    default: bli_##ch##sumsqv(m, Y, ix, &scale, &sumsq); break;                                     \
    }                                                                                               \
    break;                                                                                          \
  }                                                                                                 \
  }                                                                                                 \
  (void)alr;                                                                                        \
  free(X); free(Y);                                                                                 \
}

GEN(float, float, s)
GEN(double, double, d)
GEN(scomplex, float, c)
GEN(dcomplex, double, z)

static const trans_t TR[4] = {BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, BLIS_CONJ_NO_TRANSPOSE, BLIS_CONJ_TRANSPOSE};
static const uplo_t UP[3] = {BLIS_LOWER, BLIS_UPPER, BLIS_DENSE};

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  rd_t r = {data, size};
  int op = rd_range(&r, 24), t = rd_range(&r, 4);
  dim_t m = 0, n = 0;
  int dofb = rd_u8(&r);
  diag_t dg = rd_range(&r, 2) ? BLIS_UNIT_DIAG : BLIS_NONUNIT_DIAG;
  uplo_t up = UP[rd_range(&r, 3)];
  trans_t tr = TR[rd_range(&r, 4)];
  conj_t cj = rd_range(&r, 2) ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE;
  m = rd_range(&r, 25);
  n = rd_range(&r, 25);
  /* diagonal offset anywhere from -(m+1) to n+1, i.e. including fully pruned diagonals */
  doff_t dof = (doff_t)(dofb % (m + n + 3)) - (doff_t)(m + 1);
  int md[2], sl[2];
  for (int i = 0; i < 2; i++) md[i] = rd_range(&r, 3);
  for (int i = 0; i < 2; i++) sl[i] = rd_range(&r, 3);
  switch (t) {
  case 0: run_s(&r, op, dof, dg, up, tr, cj, m, n, md, sl); break;
  case 1: run_d(&r, op, dof, dg, up, tr, cj, m, n, md, sl); break;
  case 2: run_c(&r, op, dof, dg, up, tr, cj, m, n, md, sl); break;
  default: run_z(&r, op, dof, dg, up, tr, cj, m, n, md, sl); break;
  }
  return 0;
}
