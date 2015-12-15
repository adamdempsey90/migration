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
    if (params.read_initial_conditions == TRUE) {
        init_lam_from_file();
    }
    else {
        init_lam();
    }



//    test_matvec();
    set_matrix(); 

    double total_mass = 0;
    for(i=0;i<NR;i++) total_mass += dr[i]*lam[i];

    double t_end = params.nvisc * params.tvisc;
    double dt = params.dt;
    int nt = params.nt;
    double *times = (double *)malloc(sizeof(double)*nt);
    double *avals = (double *)malloc(sizeof(double)*nt);
    double *vs = (double *)malloc(sizeof(double)*nt);
    double *sol = (double *)malloc(sizeof(double)*NR*nt);
    double *sol_mdot = (double *)malloc(sizeof(double)*NR*nt);
    double *torque = (double *)malloc(sizeof(double)*NR);
    double *lami = (double *)malloc(sizeof(double)*NR);
    double *mdoti = (double *)malloc(sizeof(double)*NR);

    avals[0] = planet.a; vs[0] = planet.vs;

    for(i=0;i<NR*nt;i++) {
        sol[i] = 0;
        sol_mdot[i] = 0;
    }
    for(i=0;i<NR;i++) {
        lami[i] = lam[i];
        mdoti[i] = mdot[i];
    }


#pragma omp parallel for
    for(i=0;i<NR;i++) {
        torque[i] = dTr(rc[i],planet.a);
    }

    for(i=0;i<nt;i++) {
        times[i] = pow(10,i * log10(params.nvisc*params.tvisc) /((double) nt));
    }

    printf("Viscous time is %.1e...\n", params.tvisc); 
    printf("Starting Integration...\n");
    double t = 0;
    for(i=0;i<nt;i++) {
        //printf("t = %.2f\n", times[i]);
        while (t < times[i]) {
            advance_system(dt, &t, times[i]);
            if ((planet.a > params.ro) || (planet.a < params.ri)) {
                printf("\nPlanet has left the domain!\n");
            }
        }
        printf("\r t = %.2e = %.2e tvisc\t%02d%% complete...", t,t/params.tvisc,(int)(100* i/((float)nt)));
        fflush(stdout);
        set_mdot(params.planet_torque);     
        avals[i] = planet.a;
        vs[i] = planet.vs;
        for(j=0;j<NR;j++) {
            sol[j + NR*i] = lam[j];
            sol_mdot[j + NR*i] = mdot[j];
        }
    
    }
    printf("\n"); 
    printf("Outputting results...\n");
    FILE *fname = fopen("mg_results.dat","w");
    FILE *pname = fopen("planet.dat","w");
    for(i=0; i<nt; i++) {
        fprintf(pname, "%.12g\t%.12g\t%.12g\n",times[i],avals[i],vs[i]);
    }
    fclose(pname);
    
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
        fprintf(fname, "\t%lg",dr[j]);
    }
    fprintf(fname,"\n");
    fprintf(fname,"%lg",params.tvisc);
    for(j=0;j<NR;j++) {
        fprintf(fname, "\t%lg",torque[j]);
    }
    fprintf(fname,"\n");

    fprintf(fname, "%lg",params.bc_lam[0]);
    for(j=0;j<NR;j++) {
        fprintf(fname,"\t%lg",lami[j]);
    }
    fprintf(fname,"\n");

    fprintf(fname, "%lg",params.bc_lam[1]);
    for(j=0;j<NR;j++) {
        fprintf(fname,"\t%lg",mdoti[j]);
    }
    fprintf(fname,"\n");

    for(i=0;i<nt;i++) {
        fprintf(fname,"%lg",times[i]);
        for(j=0;j<NR;j++) {
            fprintf(fname,"\t%lg",sol[j+NR*i]);
        }
        fprintf(fname,"\n");
    }
    for(i=0;i<nt;i++) {
        fprintf(fname,"-1");
        for(j=0;j<NR;j++) {
            fprintf(fname,"\t%lg",sol_mdot[j+NR*i]);
        }
        fprintf(fname,"\n");
    }

    printf("Cleaning up...\n");
    fclose(fname);

    free(lami); free(mdoti);
    free(times); free(sol); free(avals); free(vs);
    free(sol_mdot);
    free(torque);
    free_grid();
    free_matrix();
    return 1;
}

void advance_system(double dt, double *t, double tend) { 

    double dt2 = tend-(*t);

    if (dt > dt2) {
        dt = dt2;
    }
 
    if ((params.move_planet) && (*t >= params.release_time)) {

        if (params.move_planet_implicit) {
     //       move_planet_implicit(dt,lam,&planet.vs, &planet.a);
      //      predictor_corrector(dt,lam, &planet.vs,&planet.a); 
            multi_step(dt,lam,&planet.vs,&planet.a);
        }
        else {
            crank_nicholson_step(dt,planet.a, lam);
            move_planet(dt, lam, &planet.vs, &planet.a); 
        }
    }
    else {
        crank_nicholson_step(dt,planet.a,lam);

    }
    *t += dt;

    return;
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


void calc_coeffs(double x, double dx,double *a, double *b,double rp,int planet_torque) {
    // dx = rc[i]-rc[i-1] or rc[i+1] - rc[i] or dr[i]
    *a = 3*nu(x) * (params.gamma - .5)/(x);
    *b = 3*nu(x)/(dx);

    if (planet_torque == TRUE) {
        *a -= dTr(x,rp)/(M_PI * sqrt(x));
    }
    *a /= 2;
    return;
}


void crank_nicholson_step(double dt, double aplanet, double *y) {
    int i;
    double ap,am, bm,bp, rm, rp;

    for(i=0;i<NR-1;i++) {
        matrix.fm[i] = 0;
        matrix.md[i] = 0;
        matrix.ud[i] = 0;
        matrix.ld[i] = 0;
    }
    matrix.md[NR-1] = 0;
    matrix.fm[NR-1] = 0;



#pragma omp parallel for private(rm,rp,am,ap,bm,bp) shared(matrix,rc,rmin)
    for(i=1;i<NR-1;i++) {
        rm = rmin[i];
        rp = rmin[i+1];

        calc_coeffs(rm,rc[i]-rc[i-1],&am,&bm,aplanet,params.planet_torque);
        calc_coeffs(rp,rc[i+1]-rc[i],&ap,&bp,aplanet,params.planet_torque);

        matrix.md[i] = (ap-am - bm - bp)*dt/2.;
        matrix.ld[i-1] = (-am + bm)*dt/2.;
        matrix.ud[i] = (ap + bp)*dt/2.;
      }
    
    matvec(matrix.ld,matrix.md,matrix.ud,lam,matrix.fm,NR);

#pragma omp parallel for shared(y,dr,matrix)
    for(i=0;i<NR;i++) {
        matrix.fm[i] += dr[i]*y[i];
        matrix.md[i] = dr[i] - matrix.md[i];
        matrix.ld[i] *= -1;
        matrix.ud[i] *= -1;
    }

    matrix.md[0] = 1;
    matrix.md[NR-1] = 1;
    matrix.ld[NR-2] = 0;
    matrix.ud[0] = 0;
    matrix.fm[0] = params.bc_lam[0];
    matrix.fm[NR-1] = params.bc_lam[1];


#ifdef WRITE_MATRIX
    FILE *fname = fopen("matrix.dat","w");

    for(i=0;i<NR;i++) {
        fprintf(fname,"%lg\t",matrix.md[i]);
    }
    fprintf(fname,"\n0\t");
    for(i=0;i<NR-1;i++) {
        fprintf(fname,"%lg\t",matrix.ld[i]);
    }
    fprintf(fname,"\n");
    for(i=0;i<NR-1;i++) {
        fprintf(fname,"%lg\t",matrix.ud[i]);
    }
    fprintf(fname,"0\n");
    for(i=0;i<NR;i++) {
        fprintf(fname,"%lg\t",matrix.fm[i]);
    }
    fprintf(fname,"\n");
    fclose(fname);
#endif
    trisolve(matrix.ld,matrix.md,matrix.ud,matrix.fm,y,NR);


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

void init_lam_from_file(void) {
    FILE *f = fopen("lambda_init.dat","r");

    double x,temp;
    int i=0;
    while (fscanf(f,"%lg\t%lg\n",&x,&temp) != EOF) {
          if (fabs(x -rc[i]) < 1e-4) {
                 lam[i] = temp;
          }
          else {
             printf("Grid is not the same! %.12e\t%.12lg\t.12%lg\n", fabs(x-rc[i]),rc[i],x);
          }
        i++;
    }
 
    fclose(f);

    set_mdot(params.planet_torque);
    return;

}

void init_lam(void) {
    int i;
//    double plaw = log(params.bc_lam[0]/params.bc_lam[1])/log(rc[0]/rc[NR-1]);

    double mdot0 = 1.5*params.nu0 * (params.bc_lam[1]*pow(params.ro,params.gamma - .5) - params.bc_lam[0]*pow(params.ri,params.gamma-.5))/(sqrt(params.ro)-sqrt(params.ri));
    double sig0 = 2*mdot0/(3*params.nu0);
    double x;
    for(i=0;i<NR;i++) {
        x = rc[i]/params.ri;
        lam[i] = params.bc_lam[0]*pow(x,.5-params.gamma) + sig0*pow(rc[i],1-params.gamma)*(1-sqrt(1/x));
    }

#ifdef INITIAL_NOISE
    double locs[7] = {.1,.6, 2, 5, 9, 20 , 30};
    int nlocs = 7; 
    int j;
    for(i=0;i<NR;i++) {
        for(j=0;j<nlocs;j++) {
            lam[i] += params.bc_lam[1] * exp( -(rc[i]-locs[j])*(rc[i]-locs[j])/.2);
        }
    }
#endif

    set_mdot(FALSE);
    return;
}

void set_matrix(void) {
    int i;
    matrix.size = NR;
    matrix.ld = (double *)malloc(sizeof(double)*(NR-1));
    matrix.md = (double *)malloc(sizeof(double)*NR);
    matrix.ud = (double *)malloc(sizeof(double)*(NR-1));
    matrix.fm = (double *)malloc(sizeof(double)*NR); 
 
    printf("Initializing matrices...\n"); 
    for(i=0;i<NR-1;i++) {
        matrix.ld[i] = 0;
        matrix.md[i] = 0;
        matrix.ud[i] = 0;
        matrix.fm[i] = 0;
                
    }
    matrix.md[NR-1] = 1;
    matrix.fm[NR-1] = 0;
    return;
}

void set_grid(void) {
    rc = (double *)malloc(sizeof(double)*NR);
    rmin = (double *)malloc(sizeof(double)*NR);
    lam = (double *)malloc(sizeof(double)*NR);
    mdot = (double *)malloc(sizeof(double)*NR);
    dr  = (double *)malloc(sizeof(double)*NR);
    dlr = log(params.ro/params.ri)/((double)NR);
    lrc = (double *)malloc(sizeof(double)*NR);
    lrmin = (double *)malloc(sizeof(double)*NR);


    int i;

    for (i=0;i<NR;i++) {
        lrc[i] = log(params.ri) + i*dlr;
        rc[i] = exp(lrc[i]);
        dr[i] = rc[i]*dlr;
        lam[i] = 0;
    }
    lrmin[0] = (log(params.ri) - dlr + lrc[0])*.5;
    rmin[0] = exp(lrmin[0]);//.5*(rc[0] + exp(log(params.ri)-dlr));
    for(i=1;i<NR;i++) {
        lrmin[i] = .5*(lrc[i]+lrc[i-1]);
        rmin[i] = exp(lrmin[i]);
    }

    set_mdot(params.planet_torque);
    return;

}

void free_grid(void) {
    free(rc); free(rmin); free(lam); free(dr);
    free(lrc); free(lrmin);
    free(mdot);
     return;
}

void free_matrix(void) {
    free(matrix.md); free(matrix.ld); free(matrix.ud); free(matrix.fm);
    return;
}
double nu(double x) {
    return params.nu0 * pow(x,params.gamma);
}
double scaleH(double x) {
    return params.h * x *  pow(x, (params.gamma -.5)/2);
}
void set_params(void) {
    params.nr = 512;
    params.alpha = .1;
    params.gamma = 0.5;
    params.h = .1;
    params.ri = .05;
    params.ro = 30.;
    params.dt = 10;
    params.nvisc = 1;
    params.nt = 100;
    params.mach = 1/params.h;
    params.nu0 = params.alpha * params.h*params.h;
    params.mth = params.h*params.h*params.h;
    params.mvisc = sqrt(27.*M_PI/8  * params.alpha * params.mach);
    params.tvisc = params.ro*params.ro/nu(params.ro); 
    params.planet_torque = TRUE;
    params.move_planet = TRUE;
    params.read_initial_conditions = FALSE;
    params.move_planet_implicit = TRUE;
    params.bc_lam[0] = 1e-12;
    params.bc_lam[1] = 1e-2;
    params.release_time = 0;

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
    planet.mp = 1;
    planet.rh = pow( planet.mp * params.mth/3.,1./3) * planet.a;
    planet.omp = pow(planet.a,-1.5);
    planet.G1 = 0;
    planet.beta = 2./3;
    planet.delta = .1;
    planet.dep = params.h*planet.a;
    planet.c = 2./3;
    planet.eps = 1;
    planet.gaussian = FALSE;
    planet.onesided = 0;
    planet.T0 = 2*M_PI*planet.a*planet.mp*planet.mp*params.mth/params.h;
    planet.vs = 0;

    printf("Planet properties:\n \
            \ta = %lg\n \
            \tmass = %lg mth = %.2e Mstar = %.2e mvisc\n \
            \trh = %lg = %lg h\n",
            planet.a, 
            planet.mp,planet.mp*params.mth,planet.mp/params.mvisc,
            planet.rh, planet.rh/params.h);




    return;
}

double dTr(double x,double a) {
    double left_fac, right_fac, smooth_fac; 
    double norm, xi,res;

    if (planet.gaussian == TRUE) {
        xi = (x-a)/planet.dep;
        left_fac = (xi-planet.beta)/planet.delta;
        right_fac = (xi+planet.beta)/planet.delta;
        left_fac = exp(-left_fac*left_fac);
        right_fac = exp(-right_fac*right_fac);

        norm = planet.T0/(planet.delta * sqrt(M_PI));

        res = norm*( (planet.G1+1)*right_fac - left_fac);
    }
    else {
        xi = (x-a) / scaleH(x);


        norm = planet.eps * a*M_PI*(planet.mp*params.mth)*(planet.mp*params.mth);
    
        right_fac = norm*pow(a/fmax(scaleH(x),fabs(x-a)),4);
    
        left_fac = -norm*pow(x/fmax(scaleH(x),fabs(x-a)),4);    
        
        left_fac *= (1-smoothing(xi, -planet.c, planet.delta));
        right_fac *= smoothing(xi,-planet.c, planet.delta)*smoothing(xi,planet.c,planet.delta);
        
        res = left_fac*(1-planet.onesided) + right_fac;


    
    }

    return res;
}

double smoothing(double x, double x0, double w) {
    return 0.5*(1 + tanh( (x-x0)/w));
}


double calc_drift_speed(double a, double *y) {
    int i;
    double res = 0;


#pragma omp parallel for reduction(+:res)    
    for(i=0;i<NR;i++) {
        res += dTr(rc[i],a)*y[i];
    }

    res *= -2*dlr*sqrt(a)/(planet.mp*params.mth);

    return res;

}

void move_planet(double dt, double *y, double *vs, double *a) {
    
    *vs = calc_drift_speed(*a,y);
    planet.a += dt*planet.vs;
    *a += dt*(*vs);

    return;
}

void set_mdot(int planet_torque) {
    int i;
    double ca, cb;
    
    for(i=0;i<NR;i++) {
        ca = 3*nu(rc[i])/rc[i];
        cb = ca*(params.gamma - .5);

        if (planet_torque == TRUE) {
            cb -= dTr(rc[i],planet.a)/(M_PI*sqrt(rc[i]));
        }  
        mdot[i] = cb*lam[i]; 
        if (i == 0){ 
            mdot[i] += ca*(lam[i+1]-params.bc_lam[0])/(2*dlr);
        }
        else {
            if (i==NR-1) {
                mdot[i] += ca*(params.bc_lam[1] - lam[i-1])/(2*dlr);
            }
            else {
                mdot[i] += ca*(lam[i+1]-lam[i-1])/(2*dlr);
            }
        }
    }



    return;
}



void linear_interpolation(double *x,double *y,double *xd, double *yd,int nx,int nd) {
/* Linear interpolation of the data xd,yd with nd data points onto the nx x-values 
 * stored in x. Result stored in y. Both x and xd should be in order.
*/
    int  i,j,jstart;
    
    jstart = 0;
    for(i=0;i<nx;i++) {
        for(j=jstart;j<nd;j++) {
            if (x[i] <= xd[j]) {
                y[i] = yd[j-1] + (yd[j]-yd[j-1])*(x[i]-xd[j-1])/(xd[j]-xd[j-1]);
            }
        }
    }

    return;
}




void move_planet_implicit(double dt, double *y, double *vs, double *a) {
    double tol = 1e-4;
    double a_old = *a;
    double args[2] = {dt,a_old};
    double a_new;
    double *lam_old = (double *)malloc(sizeof(double)*NR);
    memcpy(&lam_old[0],&y[0],sizeof(double)*NR);
    *a = secant_method(&planet_zero_function_euler,a_old, .9*a_old, lam_old,tol,args); 
    memcpy(&y[0],&lam_old[0],sizeof(double)*NR);
    *vs = calc_drift_speed(planet.a,y);
    free(lam_old);
    /*
    double vs = calc_drift_speed(*a, );
    double rhs = planet.a + .5*dt * vs;
    double args[2] = {dt,rhs};
    double a_old = planet.a;
    double tol = 1e-4;
    planet.a = secant_method(&planet_zero_function,a_old, .5*a_old,tol,args)
    */
    return;

}

double secant_method(double (*function)(double,double *,double[]),double x1, double x2,double *y,double tol, double args[]) {
    int i;

    
    double f1,f2, temp;

    f1 = (*function)(x1,y,args);
    for (i=0;i<MAXITERATIONS;i++) {
        f2 = (*function)(x2,y,args);
        if (fabs(f1-f2) <= tol) {
            printf("f1-f2 < tol:%lg\t%lg\t%lg\t%lg\n",x1,x2,f1,f2);
            break;
        }
        temp = x1;
        x1 -= f1*(x1-x2)/(f1-f2);
        if (fabs(x1-x2) <= tol) {
            printf("Converged to %lg in %d iterations\n",x1,i);
            return x1;
        }
       // printf("%lg\t%lg\t%lg\t%lg\n",x1,x2,f1,f2);
        x2 = temp;
        f1 =f2;
    }
    printf("Max iterations exceeded!\n");

    return x1;
}

double planet_zero_function_euler(double a, double *y,double args[]) {
    double  dt = args[0];
    double a_old = args[1];
    crank_nicholson_step(dt,a,y);
    double vs = calc_drift_speed(a,y);

    return a - a_old - dt*vs;
}

void predictor_corrector(double dt, double *y, double *vs, double *a) {

/*
    double vs1 = calc_drift_speed(*a, y);
    
    double a1 = *a + dt*vs1;

    crank_nicholson_step(dt,a1,y);
    
    *vs = calc_drift_speed(a1,y);
    *a += .5*dt*(vs1 + *vs);
*/

    predict_step(dt,y,vs,a);
    correct_step(dt,y,vs,a);
    predict_step(dt,y,vs,a);
    correct_step(dt,y,vs,a);


    return;
}

void predict_step(double dt, double *y,double *vs, double *a) {
    *vs = calc_drift_speed(*a,y);
    *a += dt * (*vs);
    return;
}

void correct_step(double dt, double *y, double *vs, double *a) {
    crank_nicholson_step(dt,*a,y);
    double vs1 = calc_drift_speed(*a,y);
    *a += .5*dt*(vs1 - (*vs));
    *vs = vs1;
    return;
}

void multi_step(double dt, double *y, double *vs, double *a) {
    double vs1 = calc_drift_speed(*a,y);
    *a += .5*dt*vs1;

    crank_nicholson_step(dt,*a,y);

    *vs = calc_drift_speed(*a,y);

    *a += .5*dt*(*vs);

    return;
}



/*
double planet_zero_function(double a, double args[2]) {
    planet.a = a;
    double dt = args[0];
    double vs;
    double rhs = args[1];
    crank_nicholson_step(dt);
    vs = calc_drift_speed(a,);
    double lhs =  a - .5*dt*vs;
    return lhs - rhs;
}
*/
