

#ifndef __FIAR_FUZZ__
#define __FIAR_FUZZ__

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

#endif
