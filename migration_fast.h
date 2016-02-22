
#define H5_USE_16_API
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hdf5.h>
#include <string.h>

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif


#define ZEROTORQUE
#define NR params.nr
#define TRUE 1
#define FALSE 0 
#define WRITE_MATRIX
//#define INITIAL_NOISE
#define MAXITERATIONS 300
#define MAXSTRLEN 300
#define HDF5_INSERT_ERROR(status)  if (status < 0) printf("HDF5 error at line %d\n",__LINE__);
#define MALLOC_SAFE(ptr) if (ptr == NULL) printf("Malloc error at line %d!\n",__LINE__);

typedef struct Parameters {
    double alpha,h,ri,ro,gamma, mach, mth, mvisc, tvisc; 
    double nu0;
    int nr,nt;
    double dt,nvisc;
    int planet_torque, move_planet,move_planet_implicit;
    int read_initial_conditions;
    int explicit_stepper;
    int start_ss;
    int hs_visc;
    double bc_lam[2];
    double bc_mdot;
    double cfl;
    int flux_bc;
    double release_time;
    char outputname[MAXSTRLEN];
    int nonlocal_torque;
    int shock_dep;
} Parameters;


typedef struct param_t {
    int nr;
    double ri;
    double ro;
    double alpha;
    double gamma;
    double h;
    double bc_lam_inner;
    double bc_lam_outer;
    double bc_mdot;
    int flux_bc;
    double dt;
    double cfl;
    double nvisc;
    int nt;
    double release_time;
    int start_ss;
    int read_initial_conditions;
    int planet_torque;
    int explicit_stepper;
    int move_planet;
    int move_planet_implicit;
    int gaussian;
    int symmetric_torque;
    int nonlocal_torque;
    int shock_dep;
    int hs_visc;
    double one_sided;
    double a;
    double mp;
    double G1;
    double beta;
    double delta;
    double c;
    double eps;
    double xd;
} param_t; 

typedef struct Planet {
    double a, omp, delta, G1, beta, mp,vs;
    double rh, dep;
    double c,eps;
    double onesided;
    int gaussian;
    int symmetric_torque;
    double xd;
    double T0;
} Planet;

typedef struct TridDiagMat {
    int size;
    int nsol;
    int icol;
    double *md, *ld, *ud, *fm;
    double *u, *w;
} TriDiagMat;

typedef struct Field {
double *sol;
double *sol_mdot;
double *times;
double *lami;
double *mdoti;
double *avals;
double *vs; 
double *torque;
double *vs_ss;
double *mdot_ss;
double *ivals_ss;
double *kvals_ss;
double *sol_ss;
double *lamp;
double *lam0;
double *efficiency;
double *nu_grid;
} Field;


typedef struct SteadyStateField {
    double *lam;
    double *lamp;
    double mdot;
    double a;
    double vs;
    double *lam0;
    double mdot0;
    double *ivals;
    double *kvals;

} SteadyStateField;


double *rc, *rmin, *lam, *dr;
double *taumin, *tauc;
double *lrc, *lrmin;
double *mass, *ones, *mdot;
double dlr;
Parameters params;
Planet planet; 
TriDiagMat matrix; 
Field fld;
SteadyStateField fld_ss;
void set_params(char *);
void set_planet(void);
void set_grid(void);
void free_grid(void);
double nu(double);
double cs(double);
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
double calc_total_torque(double,double *);
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
void write_hdf5_double(double *data, hsize_t *dims, int ndims, hid_t group_path, char *name);
void write_hdf5_file(void);
void write_hdf5_params(hid_t *params_id); 
void allocate_steady_state_field(SteadyStateField *fld); 
void free_steady_state_field(SteadyStateField *fld) ;
void steadystate_config(SteadyStateField *fld, double a); 
void allocate_field(Field *tmpfld);
void free_field(Field *tmpfld);
int locate(double *, int , double);
void linear_interpolation(double *, double *, double *, double *, int, int);

void trisolve_sm(double *, double *, double *, double *,double *,double *, double *,int, int);
void trisolve_smn(double *, double *, double *, double *,double *,double *,int, int);
void crank_nicholson_step_nl(double,double,double *);
double tau_integrand(double );
void set_tau(double );
double calc_coeffs_nl(double,double,int);
double dTr_nl(double, int, double, int);
void set_weights( double *,double *,double , int, int);
void steadystate_config_nl(SteadyStateField *fld, double a); 
void explicit_step(double dt, double *aplanet, double *y);
double set_dt(void);
