#include "migration_fast.h"
int main(void) {
    int i,j;
    printf("Setting parameters...\n");
    set_params();
    printf("Setting up planet...\n");
    set_planet();
    printf("Setting up grid...\n");
    set_grid();
    printf("Initializing lambda...\n");
    init_lam();


    test_matvec();
    double *ld = (double *)malloc(sizeof(double)*(NR-1));
    double *md = (double *)malloc(sizeof(double)*NR);
    double *ud = (double *)malloc(sizeof(double)*(NR-1));
    double *fm = (double *)malloc(sizeof(double)*NR); 
    mass = (double *)malloc(sizeof(double)*NR);

   printf("Initializing matrices...\n"); 
    for(i=0;i<NR-1;i++) {
        ld[i] = 0;
        md[i] = 0;
        ud[i] = 0;
        fm[i] = 0;
        mass[i] = rmin[i+1]-rmin[i];
        
    }
    md[NR-1] = 1;
    fm[NR-1] = 0;
    mass[NR-1] = sqrt(rc[NR-1]*exp(log(params.ri) + NR*log(params.ro/params.ri)/((double)NR))) - rmin[NR-1];

    double total_mass = 0;
    for(i=0;i<NR;i++) total_mass += mass[i];

    double t_end = 500;
    double dt = .05;
    int nt = (int)(t_end/dt) + 2;
    double *times = (double *)malloc(sizeof(double)*nt);
    double *sol = (double *)malloc(sizeof(double)*NR*nt);

    for(i=0;i<NR;i++) {
        sol[i] = lam[i];
    }

    for(i=0;i<nt;i++) {
        times[i] = i * t_end /((double) nt);
    }

    printf("Starting Integration...\n");
    for(i=1;i<nt;i++) {
        //printf("t = %.2f\n", times[i]);
        crank_nicholson_step(dt,ld,md,ud,fm);
        
        for(j=0;j<NR;j++) {
            sol[j + NR*i] = lam[j];
        }
    
    }
    
    printf("Outputting results...\n");
    FILE *fname = fopen("results.dat","w");
    fprintf(fname, "%d",NR);
    for(j=0;j<NR;j++) {
        fprintf(fname, "\t%lg",rc[j]);
    }
    fprintf(fname,"\n");
    fprintf(fname,"%d",nt);
    for(j=0;j<NR;j++) {
        fprintf(fname, "\t%lg",rmin[j]);
    }
    fprintf(fname,"\n");
    fprintf(fname,"%.2e",total_mass);
    for(j=0;j<NR;j++) {
        fprintf(fname, "\t%lg",mass[j]);
    }
    fprintf(fname,"\n");


    for(i=0;i<nt;i++) {
        fprintf(fname,"%lg",times[i]);
        for(j=0;j<NR;j++) {
            fprintf(fname,"\t%lg",sol[j+NR*i]);
        }
        fprintf(fname,"\n");
    }

    printf("Cleaning up...\n");
    fclose(fname);

    free(times); free(sol);
    free(ld); free(md); free(ud);free(fm);
    free(mass);
    free_grid();
    return 1;
}


void matvec(double *ld, double *md, double *ud, double *a, double *b, int n) {
    int i;

    b[0] += md[0]*a[0] + ud[0]*a[1];

    for(i=1;i<n-1;i++) {
        b[i] += ld[i-1]*a[i-1] + md[i]*a[i] + ud[i]*a[i+1];

    }
    b[n-1] += ld[n-2]*a[n-2] + md[n-1] * a[n-1];

    return;
}

void test_matvec(void) {
   double c[10] = { 0.5,  0.32,  0.24,  0.81,  0.83,  0.25,  0.89,  0.66,  0.56,  0.};
   double md[10] = { 0.41,  0.12,  0.97,  0.52,  0.32,  0.28,  0.42,  0.58,  0.5 ,  0.51};  
   double ld[9] = {0.89,  0.2 ,  0.98,  0.9 ,  0.63,  0.19,  0.93,  0.79,  0.84};
   double ud[9] = {0.02,  0.19,  0.8 ,  0.4 ,  0.04,  0.11,  0.35,  0.79,  0.57};
   double a[10] = { 0.62,  0.98,  0.04,  0.85,  0.13,  0.87,  0.84,  0.77,  0.23,  0.43};
   double mvans[10]  = { 0.8314,  1.509 ,  0.9848,  1.8384,  1.1346,  1.5608,  1.4923,
               2.4229,  1.0314,  0.9004};
   double tans[10] = {1.90111459393,
    -13.9728491757,
    1.60394690777,
    1.84842666824,
    -4.30762459275,
    13.6213967067,
    -7.7289780403,
    4.42315829331,
    6.68673135109,
    -11.0134398724};

matvec(ld,md,ud,c,a,10);
   int i;
   double err = 0;
   printf("MatVec\n");
   for(i=0;i<10;i++) {
       printf("%.5lg\t%.5lg\n", mvans[i],a[i]);
       err += fabs( (mvans[i] - a[i])/mvans[i]);
    }
   printf("Matvec L1 error: %.2e\n", err);

    

   trisolve(ld,md,ud,c,a,10);

   err = 0;
   printf("TriSolve\n");
   for(i=0;i<10;i++) {
       printf("%.5lg\t%.5lg\n",tans[i],a[i]);
        err += fabs( (tans[i]-a[i])/tans[i]);
   }
    printf("TriSolve L1 error: %.2e\n", err);
   return;
}


void crank_nicholson_step(double dt,double *ld, double *md, double *ud, double *fm) {
    int i;
    double ap,am, bm,bp, rm, rp;

    for(i=0;i<NR-1;i++) {
        fm[i] = 0;
        md[i] = 0;
        ud[i] = 0;
        ld[i] = 0;
    }
    md[NR-1] = 0;
    fm[NR-1] = 0;

    md[0] = 1;
    ud[0] = 0;
    fm[0] *= dt;
    for(i=1;i<NR-1;i++) {
        rm = rmin[i];
        rp = rmin[i+1];
        am = 3*nu(rm) * (params.gamma - .5)/(2.*rm);
        ap = 3*nu(rp) * (params.gamma - .5)/(2*rp);
           
        bm = 3*nu(rm)/(rc[i]-rc[i-1]);
        bp = 3*nu(rp)/(rc[i+1]-rc[i]);
        
        //printf("%lg\t%lg\t%lg\t%lg\t%lg\n",rc[i],am,ap,bm,bp);

        md[i] = (ap-am + bm - bp)*dt/2.;
        ld[i-1] = (-am + bm)*dt/2.;
        ud[i] = (ap + bp)*dt/2.;
        fm[i] *= dt;
    }
    fm[NR-1] *= dt;
    md[NR-1] = 1;
//    ld[NR-2] = 0;
    
//    rm = rmin[NR-1];
    
//    ld[NR-2] = -3*nu(rm)*(params.gamma-.5)/(2*rm) + 3*nu(rm)/(rc[NR-1]-rc[NR-2]);

    matvec(ld,md,ud,lam,fm,NR);

    for(i=0;i<NR;i++) {
        fm[i] += mass[i]*lam[i];
        md[i] = mass[i] - md[i];
    }

    md[0] = 1;
    md[NR-1] = 1;
    ld[NR-2] = 0;
    ud[0] = 0;
    fm[0] = params.bc_lam[0];
    fm[NR-1] = params.bc_lam[1];

    trisolve(ld,md,ud,fm,lam,NR);


    return;
}

void trisolve(double *ld, double *md, double *ud, double *d,double *sol,int n) {
    int i;

    double *cp = (double *)malloc(sizeof(double)*(n-1));
    double *bp = (double *)malloc(sizeof(double)*n);
    double *dp = (double *)malloc(sizeof(double)*n);

    for(i=0;i<n-1;i++) {
        cp[i] = 0;
        bp[i] = 1;
        dp[i] = 0;
    }
    bp[n-1] = 0;
    dp[n-1] = 0;

    cp[0] = ud[0]/md[0];

    for(i=1;i<n-1;i++) {
        cp[i] = ud[i]/(md[i]- ld[i-1]*cp[i-1]);
    }
    dp[0] = d[0]/md[0];
    for(i=1;i<n;i++) {
        dp[i] = (d[i] - ld[i-1]*dp[i-1])/(md[i]-ld[i-1]*cp[i-1]);

    }

    sol[n-1] = dp[n-1];

    for(i=n-2;i>=0;i--) {
        sol[i] = dp[i] - cp[i]*sol[i+1];
    }

    free(cp); free(bp); free(dp);
    return;
}



void init_lam(void) {
    int i;
    double plaw = log(params.bc_lam[0]/params.bc_lam[1])/log(rc[0]/rc[NR-1]);

    for(i=0;i<NR;i++) {
        lam[i] = params.bc_lam[0] * pow(rc[i]/rc[0],plaw);
    }

    return;
}

void set_grid(void) {
    rc = (double *)malloc(sizeof(double)*params.nr);
    rmin = (double *)malloc(sizeof(double)*params.nr);
    lam = (double *)malloc(sizeof(double)*params.nr);


    int i;

    for (i=0;i<params.nr;i++) {
        rc[i] = exp( log(params.ri) + i*log(params.ro/params.ri)/((double) params.nr));
        lam[i] = 0;
    }
    rmin[0] = sqrt(exp( log(params.ri) - log(params.ro/params.ri)/((double) params.nr)) * rc[0]);
    for (i=1;i<params.nr;i++) {
        rmin[i] = sqrt(rc[i-1]*rc[i]);
    }

    return;

}

void free_grid(void) {
    free(rc); free(rmin); free(lam);
}

double nu(double r) {
    return params.nu0 * pow(r,params.gamma);
}
void set_params(void) {
    params.nr = 128;
    params.alpha = .1;
    params.gamma = 0.5;
    params.h = .1;
    params.ri = 1.;
    params.ro = 50.;
    params.mach = 1/params.h;
    params.nu0 = params.alpha * params.h*params.h;
    params.mth = params.h*params.h*params.h;
    params.mvisc = sqrt(27.*M_PI/8  * params.alpha * params.mach);
    params.tvisc = pow(params.ro,-params.gamma)/(params.alpha * params.h * params.h );

    params.planet_torque = 0;
    params.bc_lam[0] = 1e-3;
    params.bc_lam[1] = 1.;

    printf("Parameters:\n\tnr = %d\n\talpha = %.1e\n\th = %.2f\n\t(ri,ro) = (%lg,%lg)\n\tMach = %.1f\n\tm_th = %.2e\n\tm_visc = %.2e\n\tt_visc=%.2e\n",
            params.nr,
            params.alpha,
            params.h,
            params.ri,
            params.ro,
            params.mach,
            params.mth,
            params.mvisc,
            params.tvisc);

    return;
}

void set_planet(void) {
    planet.a  = 10.;
    planet.mp = 1.;
    planet.rh = pow( planet.mp * params.mth/3.,1./3) * planet.a;
    planet.omp = pow(planet.a,-1.5);
    planet.G1 = 0;
    planet.beta = 2./3;
    planet.delta = .1;
    planet.dep = params.h;
    return;
}
