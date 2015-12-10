#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NR params.nr
#define TRUE 1
#define FALSE 0 

typedef struct Parameters {
    double alpha,h,ri,ro,gamma, mach, mth, mvisc, tvisc; 
    double nu0;
    int nr,nt;
    double dt,nvisc;
    int planet_torque, move_planet;
    double bc_lam[2];
} Parameters;


typedef struct Planet {
    double a, omp, delta, G1, beta, mp,vs;
    double rh, dep;
    double c,eps;
    int onesided,gaussian;
    double T0;
} Planet;


double *rc, *rmin, *lam, *dr;
double *mass, *ones, *mdot;
double dlr;
Parameters params;
Planet planet; 


void set_params(void);
void set_planet(void);
void set_grid(void);
void free_grid(void);
double nu(double);
double scaleH(double);
void init_lam(void);
void matvec(double *, double *, double *, double *, double *, int);
void trisolve(double *, double *, double *, double *, double *,int);
void crank_nicholson_step(double, double *, double *, double *, double *);
void test_matvec(void);
double smoothing(double,double,double);
double dTr(double);
void move_planet(double);
double calc_drift_speed(void);
void calc_coeffs(double, double,double *, double *,int);
void set_mdot(int);
