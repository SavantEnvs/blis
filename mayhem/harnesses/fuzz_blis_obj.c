/* BLIS object-API fuzzer: obj_t operands with independently chosen datatypes, conj/trans bits
 * and structure (uplo, diag, diagonal offset) driving bli_gemm (incl. the mixed-domain /
 * mixed-precision path, frame/3/gemm/bli_gemm_md*.c), bli_castm (cross-datatype casts),
 * bli_hemm/symm, bli_herk/syrk, bli_trmm/trsm and bli_copym/addm/normfm.
 *
 * Input: [op][dtA][dtB][dtC][transA][transB][side][uplo][diag][m][n][k][stride modes 3][slacks 3]
 * [alpha 16][beta 16] then data. Every object is backed by an exactly-sized buffer
 * (bli_obj_create_with_attached_buffer); dims are always conformal and same-datatype operations
 * use one datatype, because BLIS's error checking bli_abort()s on a contract violation by design. */
#include "fuzz_common.h"
#include <blis.h>

static const num_t DT[4] = {BLIS_FLOAT, BLIS_DOUBLE, BLIS_SCOMPLEX, BLIS_DCOMPLEX};
static const trans_t TR[4] = {BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, BLIS_CONJ_NO_TRANSPOSE, BLIS_CONJ_TRANSPOSE};

typedef struct { obj_t o; void *buf; } mobj_t;

/* stored rows x cols matrix of datatype dt; 0 = col-major, 1 = row-major, 2 = general stride */
static void mk(rd_t *r, mobj_t *x, num_t dt, dim_t rows, dim_t cols, int mode, int slack) {
  inc_t rs, cs;
  if (mode == 0) { rs = 1; cs = (rows > 1 ? rows : 1) + slack; }
  else if (mode == 1) { cs = 1; rs = (cols > 1 ? cols : 1) + slack; }
  else { rs = 2; cs = 2 * (rows > 1 ? rows : 1) + 1 + slack; }
  size_t len = (rows > 0 && cols > 0) ? (size_t)((rows - 1) * rs + (cols - 1) * cs + 1) : 0;
  x->buf = rd_buf(r, len, bli_dt_size(dt));
  bli_obj_create_with_attached_buffer(dt, rows, cols, x->buf, rs, cs, &x->o);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  rd_t r = {data, size};
  int op = rd_range(&r, 7);
  num_t da = DT[rd_range(&r, 4)], db = DT[rd_range(&r, 4)], dc = DT[rd_range(&r, 4)];
  trans_t ta = TR[rd_range(&r, 4)], tb = TR[rd_range(&r, 4)];
  side_t sd = rd_range(&r, 2) ? BLIS_RIGHT : BLIS_LEFT;
  uplo_t up = rd_range(&r, 2) ? BLIS_LOWER : BLIS_UPPER;
  diag_t dg = rd_range(&r, 2) ? BLIS_UNIT_DIAG : BLIS_NONUNIT_DIAG;
  dim_t m = rd_range(&r, 25), n = rd_range(&r, 25), k = rd_range(&r, 25);
  int md[3], sl[3];
  for (int i = 0; i < 3; i++) md[i] = rd_range(&r, 3);
  for (int i = 0; i < 3; i++) sl[i] = rd_range(&r, 3);
  scal_t al, be;
  rd_scal(&r, &al);
  rd_scal(&r, &be);
  int tra = bli_does_trans(ta), trb = bli_does_trans(tb);
  mobj_t A = {0}, B = {0}, C = {0};
  obj_t alpha, beta;

  switch (op) {
  case 0: /* gemm, mixed domain/precision: C(dc) = op(A)(da) op(B)(db) */
    mk(&r, &A, da, tra ? k : m, tra ? m : k, md[0], sl[0]);
    mk(&r, &B, db, trb ? n : k, trb ? k : n, md[1], sl[1]);
    mk(&r, &C, dc, m, n, md[2], sl[2]);
    bli_obj_set_conjtrans(ta, &A.o);
    bli_obj_set_conjtrans(tb, &B.o);
    bli_obj_create_1x1_with_attached_buffer(dc, &al, &alpha);
    bli_obj_create_1x1_with_attached_buffer(dc, &be, &beta);
    bli_gemm(&alpha, &A.o, &B.o, &beta, &C.o);
    break;
  case 1: /* castm: B(db) = op(A)(da) */
    mk(&r, &A, da, tra ? n : m, tra ? m : n, md[0], sl[0]);
    mk(&r, &B, db, m, n, md[1], sl[1]);
    bli_obj_set_conjtrans(ta, &A.o);
    bli_castm(&A.o, &B.o);
    break;
  case 2: { /* hemm / symm, one datatype */
    dim_t ka = sd == BLIS_LEFT ? m : n;
    mk(&r, &A, dc, ka, ka, md[0], sl[0]);
    mk(&r, &B, dc, m, n, md[1], sl[1]);
    mk(&r, &C, dc, m, n, md[2], sl[2]);
    bli_obj_set_struc(k & 1 ? BLIS_HERMITIAN : BLIS_SYMMETRIC, &A.o);
    bli_obj_set_uplo(up, &A.o);
    bli_obj_set_conj(bli_extract_conj(ta), &A.o);
    bli_obj_create_1x1_with_attached_buffer(dc, &al, &alpha);
    bli_obj_create_1x1_with_attached_buffer(dc, &be, &beta);
    if (k & 1) bli_hemm(sd, &alpha, &A.o, &B.o, &beta, &C.o);
    else bli_symm(sd, &alpha, &A.o, &B.o, &beta, &C.o);
    break;
  }
  case 3: /* herk (real alpha/beta) / syrk: C(m x m) = op(A) op(A)^H */
    mk(&r, &A, dc, tra ? k : m, tra ? m : k, md[0], sl[0]);
    mk(&r, &C, dc, m, m, md[2], sl[2]);
    bli_obj_set_onlytrans(tra ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE, &A.o);
    bli_obj_set_struc(n & 1 ? BLIS_HERMITIAN : BLIS_SYMMETRIC, &C.o);
    bli_obj_set_uplo(up, &C.o);
    if (n & 1) {
      if (tra) bli_obj_set_onlytrans(BLIS_CONJ_TRANSPOSE, &A.o);
      bli_obj_create_1x1_with_attached_buffer(bli_dt_proj_to_real(dc), &al, &alpha);
      bli_obj_create_1x1_with_attached_buffer(bli_dt_proj_to_real(dc), &be, &beta);
      bli_herk(&alpha, &A.o, &beta, &C.o);
    } else {
      bli_obj_create_1x1_with_attached_buffer(dc, &al, &alpha);
      bli_obj_create_1x1_with_attached_buffer(dc, &be, &beta);
      bli_syrk(&alpha, &A.o, &beta, &C.o);
    }
    break;
  case 4: { /* trmm / trsm: B = op(A) B or B op(A) */
    dim_t ka = sd == BLIS_LEFT ? m : n;
    mk(&r, &A, dc, ka, ka, md[0], sl[0]);
    mk(&r, &B, dc, m, n, md[1], sl[1]);
    bli_obj_set_struc(BLIS_TRIANGULAR, &A.o);
    bli_obj_set_uplo(up, &A.o);
    bli_obj_set_diag(dg, &A.o);
    bli_obj_set_conjtrans(ta, &A.o);
    bli_obj_create_1x1_with_attached_buffer(dc, &al, &alpha);
    if (k & 1) bli_trsm(sd, &alpha, &A.o, &B.o);
    else bli_trmm(sd, &alpha, &A.o, &B.o);
    break;
  }
  default: { /* copym / addm / normfm on a structured, diagonal-offset view */
    doff_t dof = (doff_t)(k % (m + n + 1)) - m;
    mk(&r, &A, dc, tra ? n : m, tra ? m : n, md[0], sl[0]);
    mk(&r, &B, dc, m, n, md[1], sl[1]);
    bli_obj_set_struc(BLIS_TRIANGULAR, &A.o);
    bli_obj_set_uplo(up, &A.o);
    bli_obj_set_diag(dg, &A.o);
    bli_obj_set_diag_offset(dof, &A.o);
    bli_obj_set_conjtrans(ta, &A.o);
    if (op == 5) bli_copym(&A.o, &B.o);
    else {
      bli_addm(&A.o, &B.o);
      scal_t nrm;
      obj_t no;
      bli_obj_create_1x1_with_attached_buffer(bli_dt_proj_to_real(dc), &nrm, &no);
      bli_normfm(&B.o, &no);
    }
    break;
  }
  }
  free(A.buf);
  free(B.buf);
  free(C.buf);
  return 0;
}
