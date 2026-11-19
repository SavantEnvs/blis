/* Shared helpers for the BLIS fuzz harnesses.
 *
 * Every harness decodes an operation selector, enum flags, small dimensions and strides from the
 * input, then allocates EXACTLY the number of elements the BLAS/BLIS contract says the library may
 * touch (no padding), fills them from the remaining input bytes, and calls the public API. An
 * access outside these buffers is therefore a library bug, not a harness bug.  No state is kept
 * across inputs: every buffer is freed before returning. */
#ifndef FUZZ_COMMON_H
#define FUZZ_COMMON_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const uint8_t *p; size_t n; } rd_t;

static inline uint8_t rd_u8(rd_t *r) {
  if (!r->n) return 0;
  r->n--;
  return *r->p++;
}
/* value in [0, mod) */
static inline int rd_range(rd_t *r, int mod) { return mod <= 1 ? 0 : rd_u8(r) % mod; }
/* stride/increment in {-3..-1, 1..3}; 0 only when allow0 */
static inline int rd_inc(rd_t *r, int allow0) {
  static const int v[] = {1, 2, 3, -1, -2, -3, 0};
  return v[rd_u8(r) % (allow0 ? 7 : 6)];
}
/* fill nbytes of dst from the input (zero once the input is exhausted) */
static inline void rd_fill(rd_t *r, void *dst, size_t nbytes) {
  size_t k = nbytes < r->n ? nbytes : r->n;
  if (k) memcpy(dst, r->p, k);
  if (nbytes > k) memset((char *)dst + k, 0, nbytes - k);
  r->p += k;
  r->n -= k;
}
/* exactly-sized heap buffer of nelem elements of esz bytes, filled from the input */
static inline void *rd_buf(rd_t *r, size_t nelem, size_t esz) {
  void *b = malloc(nelem * esz ? nelem * esz : 1);
  if (nelem) rd_fill(r, b, nelem * esz);
  return b;
}
/* elements reachable by a length-n vector with increment inc */
static inline size_t vec_len(int n, int inc) {
  int a = inc < 0 ? -inc : inc;
  return n <= 0 ? 0 : (size_t)(1 + (n - 1) * a);
}
static inline int imax(int a, int b) { return a > b ? a : b; }

/* Scalar storage large enough for a dcomplex; filled from the input (raw bits: NaN/Inf/denormals). */
typedef union { float f[4]; double d[2]; uint8_t b[16]; } scal_t;
static inline void rd_scal(rd_t *r, scal_t *s) { rd_fill(r, s->b, sizeof s->b); }

static const size_t ESZ[4] = {4, 8, 8, 16}; /* s d c z */
#endif
