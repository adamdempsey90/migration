#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NR params.nr
#define TRUE 1
#define FALSE 0 
#define WRITE_MATRIX
//#define INITIAL_NOISE
#define MAXITERATIONS 300
#define MAXSTRLEN 300
typedef struct Parameters {
    double alpha,h,ri,ro,gamma, mach, mth, mvisc, tvisc; 
    double nu0;
    int nr,nt;
    double dt,nvisc;
    int planet_torque, move_planet,move_planet_implicit;
    int read_initial_conditions;
    double bc_lam[2];
    double release_time;
    char outputname[MAXSTRLEN];
} Parameters;


typedef struct Planet {
    double a, omp, delta, G1, beta, mp,vs;
    double rh, dep;
    double c,eps;
    double onesided;
    int gaussian;
    double T0;
} Planet;

typedef struct TridDiagMat {
    int size;
    double *md, *ld, *ud, *fm;
} TriDiagMat;



double *rc, *rmin, *lam, *dr;
double *lrc, *lrmin;
double *mass, *ones, *mdot;
double dlr;
Parameters params;
Planet planet; 
TriDiagMat matrix; 

void set_params(void);
void set_planet(void);
void set_grid(void);
void free_grid(void);
double nu(double);
double scaleH(double);
void init_lam(void);
void matvec(double *, double *, double *, double *, double *, int);
void trisolve(double *, double *, double *, double *, double *,int);
void crank_nicholson_step(double,double,double *);
void test_matvec(void);
double smoothing(double,double,double);
double dTr(double,double);
void move_planet(double,double *, double *, double *);
double calc_drift_speed(double,double *);
void calc_coeffs(double, double,double *, double *,double,int);
void set_mdot(int);
void init_lam_from_file(void);
void set_matrix(void);
void free_matrix(void);
double secant_method(double (*function)(double,double *, double[]),double, double,double *, double,double[]); 
void move_planet_implicit(double, double *, double *, double *);
double planet_zero_function_euler(double, double *,double[]); 
void advance_system(double,double *,double);
void predictor_corrector(double , double *, double *, double *); 
void predict_step(double , double *, double *, double *); 
void correct_step(double , double *, double *, double *); 
void multi_step(double , double *, double *, double *); 
void set_bool(char *buff, int *val); 
void read_input_file(char *fname); 
