#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NR params.nr


typedef struct Parameters {
    double alpha,h,ri,ro,gamma, mach, mth, mvisc, tvisc; 
    double nu0;
    int nr;
    int planet_torque;
    double bc_lam[2];
} Parameters;


typedef struct Planet {
    double a, omp, delta, G1, beta, mp;
    double rh, dep;
} Planet;


double *rc, *rmin, *lam, *dr;
double *mass, *ones;
Parameters params;
Planet planet; 


void set_params(void);
void set_planet(void);
void set_grid(void);
void free_grid(void);
double nu(double);
void init_lam(void);
void matvec(double *, double *, double *, double *, double *, int);
void trisolve(double *, double *, double *, double *, double *,int);
void crank_nicholson_step(double, double *, double *, double *, double *);
void test_matvec(void);
