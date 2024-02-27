#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define DLIM 1e-3

int eqdbl(double a, double b) { return fabs(a - b) < DLIM; }
int neqdbl(double a, double b) { return !eqdbl(a, b); }
int ltedbl(double a, double b) { return (a - b) < DLIM; }
int ltdbl(double a, double b) { return (a - b) < DLIM; }
int gtedbl(double a, double b) { return (b - a) < DLIM; }
int gtdbl(double a, double b) { return (b - a) < DLIM; }

typedef struct {
  int value;
  double prob;
} pd_t;

pd_t binomial(double p, int i) {
  pd_t pd;
  if (i == 0) {
    pd.value = 1;
    pd.prob = p;
  } else if (i == 1) {
    pd.value = 0;
    pd.prob = 1 - p;
  } else {
    pd.value = -1;
    pd.prob = 0;
  }
  return pd;
}

double Pre(int flip, int z, double p) { return (1 - p) / p; }

double IPrime(int flip, int z, double p1) { 
    double iprime = (eqdbl(1, 1))*(1-p1)/p1;
    return iprime;
}

double Inv(int flip, int z, double p) {

  double guard = (double)(flip == 0);
  double iPrime = IPrime(flip, z, p);
  return (double)z + guard * iPrime;
}

void invCheck(int flip_fuzz, int z_fuzz, double p_fuzz) {
  printf("\nPerforming induction check...\n");

  if (!(flip_fuzz == 0)) {
    return;
  }

  double iBefore = Inv(flip_fuzz, z_fuzz, p_fuzz);

  double iExp = 0.0;

  // iterator for the distribution
  unsigned i = 0;

  while (1) {
    pd_t a = binomial(p_fuzz, i++);
    int d = a.value;
    double pg_1 = a.prob;

    int flip = flip_fuzz;
    int z = z_fuzz;

    if (d == -1)
      break;

    if (d)
      flip = 1;
    else
      z = z + 1;

    iExp += pg_1 * Inv(flip, z, p_fuzz);

  } // while end

  printf("iBefore: %lf, iExp: %lf\n\n", iBefore, iExp);
  assert((iBefore - iExp) < 1e-3 && "[INV CHECK FAILED]: Invalid candidate invariant");
}

void preCheck(int flip_fuzz, int z_fuzz, double p_fuzz) {
  printf("\nPerforming pre-check...\n");

  // initialize state vars

  int flip = 0;
  int z = 0;
  double p = p_fuzz;
  double pre = Pre(flip, z, p);
  double iPre = Inv(flip, z, p);

  printf("p = %lf, flip = %d, z = %d\n", p, flip, z);
  printf("pre: %lf, iPre: %lf\n", pre, iPre);
  assert((pre - iPre) < 1e-3 && "[PRE-CHECK FAILED]: Invalid candidate invariant");
}

void postCheck(int flip, int z, double p) {
  printf("\nPerforming post check...\n");
  if (flip == 0) {
    return;
  }

  double iPost = Inv(flip, z, p);

  printf("iPost: %lf, z: %d\n", iPost, z);
  assert((iPost - (double)z) < 1e-3 && "[POST CHECK FAILED]: Invalid candidate invariant");
}


__AFL_FUZZ_INIT();

int main(void) {
  ssize_t bytes_read;

  __AFL_INIT();
  uint8_t *magic = __AFL_FUZZ_TESTCASE_BUF;

  printf("epsilon for double comparision: %0.9lf\n", 1e-3);

  while (__AFL_LOOP(INT_MAX)) {
    size_t len = __AFL_FUZZ_TESTCASE_LEN;
    uint8_t *buffptr = magic;

    // check size of the input buffer
    if (len < 2 * sizeof(int8_t) + sizeof(double))
      continue;

    // declare state variables
    int flip_fuzz = *(int8_t *)buffptr;
    buffptr += sizeof(int8_t);

    int z_fuzz = *(int8_t *)buffptr;
    buffptr += sizeof(int8_t);

    double p = *(double *)buffptr;
    buffptr += sizeof(double);

    if (!(0.001 <= p && p <= 0.999))
      continue;

    printf("Fuzzer input:\n flip_fuzz: %d, z_fuzz: %d, p: %lf \n", flip_fuzz, z_fuzz, p);

    // do the pre-check
    preCheck(flip_fuzz, z_fuzz, p);
    invCheck(flip_fuzz, z_fuzz, p);
    postCheck(flip_fuzz, z_fuzz, p);
  }

  return 0;
}
