#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define DLIM 999.0


void save_crash_data(int flip, int z, int x, double p, double diff) {
  FILE *fp = fopen("crashing_data.csv", "w");
  fprintf(fp, "flip,%d\n", flip);
  fprintf(fp, "z,%d\n", z);
  fprintf(fp, "x,%d\n", x);
  fprintf(fp, "p,%lf\n", p);
  fprintf(fp, "diff,%lf\n", diff);
  fclose(fp);
}

/*
  fcmp returns 
  -1 if x1 is less than x2, 
   0 if x1 is equal to x2, 
   1 if x1 is greater than x2 
  (relative to the tolerance).
*/

int fcmp(double x1, double x2, double delta) {

  double difference = x1 - x2;

  if (difference > delta)
    return 1; /* x1 > x2 */
  else if (difference < -delta) 
    return -1;  /* x1 < x2 */
  else /* -delta <= difference <= delta */
    return 0;  /* x1 == x2 */
}

int eqdbl(double a, double b) { return fabs(a - b) < 0.001; }
int neqdbl(double a, double b) { return !eqdbl(a, b); }
int ltedbl(double a, double b) { return (a - b) < 0.001; }
int ltdbl(double a, double b) { return (a - b) < 0.001; }
int gtedbl(double a, double b) { return (b - a) < 0.001; }
int gtdbl(double a, double b) { return (b - a) < 0.001; }

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

double Pre(int flip, int z, int x, double p) { return (1 - p) / p; }

double IPrime(int flip, int z, int x, double p1) { 
    double iprime = 0;
    return iprime;
}

double Inv(int flip, int z, int x, double p) {

  double guard = (double)(flip == 0);
  double iPrime = IPrime(flip, z, x, p);
  return (double)z + guard * iPrime;
}

void invCheck(int flip_fuzz, int z_fuzz, int x_fuzz, double p_fuzz) {
  printf("\nPerforming induction check...\n");

  if (!(flip_fuzz == 0)) {
    return;
  }

  double iBefore = Inv(flip_fuzz, z_fuzz, x_fuzz, p_fuzz);

  double iExp = 0.0;

  // iterator for the distribution
  unsigned i = 0;

  while (1) {
    pd_t a = binomial(p_fuzz, i++);
    int d = a.value;
    double pg_1 = a.prob;

    int flip = flip_fuzz;
    int z = z_fuzz;
    int x = x_fuzz;

    if (d == -1)
      break;

    if (d)
      flip = 1;
    else {
      x = x * 2;
      z = z + 1;
    }

    iExp += pg_1 * Inv(flip, z, x, p_fuzz);

  } // while end

  printf("iBefore: %lf, iExp: %lf\n\n", iBefore, iExp);
  
  if (iBefore - iExp > DLIM) {
    save_crash_data(flip_fuzz, z_fuzz, x_fuzz, p_fuzz, (iBefore - iExp));
    assert(0);
  }
}

void preCheck(int flip_fuzz, int z_fuzz, int x_fuzz, double p_fuzz) {
  printf("\nPerforming pre-check...\n");

  // initialize state vars

  int flip = 0;
  int z = 0;
  int x = x_fuzz;

  double p = p_fuzz;
  double pre = Pre(flip, z, x, p);
  double iPre = Inv(flip, z, x, p);

  printf("p = %lf, flip = %d, z = %d, x = %d\n", p, flip, z, x);
  printf("pre: %lf, iPre: %lf\n", pre, iPre);

  if (pre - iPre > DLIM) {
    save_crash_data(flip, z, x, p, (pre - iPre));
    assert(0);
  }
}

void postCheck(int flip, int z, int x, double p) {
  printf("\nPerforming post check...\n");
  if (flip == 0) {
    return;
  }

  double iPost = Inv(flip, z, x,  p);

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
    if (len < 3 * sizeof(int8_t) + sizeof(double))
      continue;

    // declare state variables
    int flip_fuzz = *(int8_t *)buffptr;
    buffptr += sizeof(int8_t);
    flip_fuzz = flip_fuzz % 2;

    int z_fuzz = *(int8_t *)buffptr;
    buffptr += sizeof(int8_t);

    int x_fuzz = *(int8_t *)buffptr;
    buffptr += sizeof(int8_t);

    double p = *(double *)buffptr;

    if (!(0.001 <= p && p <= 0.999))
      continue;

    printf("Fuzzer input:\n flip_fuzz: %d, z_fuzz: %d, x_fuzz: %d, p: %lf \n", flip_fuzz, z_fuzz, x_fuzz, p);

    // do the pre-check
    preCheck(flip_fuzz, z_fuzz, x_fuzz, p);
    invCheck(flip_fuzz, z_fuzz, x_fuzz, p);
    // postCheck(flip_fuzz, z_fuzz, x_fuzz, p);
  }

  return 0;
}
