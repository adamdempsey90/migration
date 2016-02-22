#include "migration_fast.h"

int main(int argc, char *argv[]) {
    
    int i,j;
    char parfile[MAXSTRLEN];
    if (argc < 2) {
        strcpy(parfile,"params.in");
    }
    else {
        strcpy(parfile,argv[1]);
    }
#ifdef NU
    printf("Code compiled with CONSTANT VISCOSITY.\n");
#endif
    printf("Reading parameters from %s...\n",parfile);
    set_params(parfile);
    
        printf("Setting up grid...\n");
    set_grid();
    printf("Initializing lambda...\n");
    if (params.read_initial_conditions == TRUE) {
        init_lam_from_file();
    }
    else {
        init_lam();
    }

    allocate_field(&fld);
    for(i=0;i<NR;i++) {
        fld.nu_grid[i] = nu(rc[i]);
    }
    allocate_steady_state_field(&fld_ss);
    
       
//    test_matvec();
    set_matrix(); 

    double total_mass = 0;
    for(i=0;i<NR;i++) total_mass += dr[i]*lam[i];

//    double t_end = params.nvisc * params.tvisc;
    double dt = params.dt;
    int nt = params.nt;
    fld.avals[0] = planet.a; fld.vs[0] = planet.vs;

    for(i=0;i<NR*nt;i++) {
        fld.sol[i] = 0;
        fld.sol_mdot[i] = 0;
    }
   
    if (params.planet_torque && params.nonlocal_torque && params.shock_dep) {
        steadystate_config_nl(&fld_ss,planet.a);

    }
    else {
        steadystate_config(&fld_ss,planet.a);
    }
//    steadystate_config_nl(&fld_ss,planet.a);

    fld.vs_ss[0] = fld_ss.vs;
    fld.mdot_ss[0] = fld_ss.mdot;
    fld.efficiency[0] = fld_ss.mdot/fld_ss.mdot0;

    if (params.start_ss) {
        for(i=0;i<NR;i++) {
            lam[i] = fld_ss.lam[i];
            mdot[i] = fld_ss.mdot;
        }
    }


    for(i=0;i<NR;i++) {
        fld.lami[i] = lam[i];
        fld.mdoti[i] = mdot[i];
        fld.sol_ss[i] = fld_ss.lam[i];
        fld.lamp[i] = fld_ss.lamp[i];
        fld.lam0[i] = fld_ss.lam0[i];
        fld.ivals_ss[i] = fld_ss.ivals[i];
        fld.kvals_ss[i] = fld_ss.kvals[i];
    }


#pragma omp parallel for private(i)
    for(i=0;i<NR;i++) {
        if (params.nonlocal_torque) {
            fld.torque[i] = dTr_nl(rc[i],i,planet.a,TRUE);
        }
        else {
            fld.torque[i] = dTr(rc[i],planet.a);
        }
    }

    for(i=0;i<nt;i++) {
        fld.times[i] = pow(10,i * log10(params.nvisc*params.tvisc) /((double) nt));
    }

    printf("Viscous time is %.1e...\n", params.tvisc); 
    printf("Starting Integration...\n"); 
    printf("dt = %.3e\n",set_dt());
    double t = 0;
    for(i=0;i<nt;i++) {
        //printf("t = %.2f\n", fld.times[i]);
        while (t < fld.times[i]) {
            advance_system(dt, &t, fld.times[i]);
            if ((planet.a > params.ro) || (planet.a < params.ri)) {
                printf("\nPlanet has left the domain!\n");
            }
        }
        printf("\r t = %.2e = %.2e tvisc\t%02d%% complete...", t,t/params.tvisc,(int)(100* i/((float)nt)));
        fflush(stdout);
        set_mdot(params.planet_torque);     
        fld.avals[i] = planet.a;
        fld.vs[i] = planet.vs;
        

        if (params.planet_torque && params.nonlocal_torque && params.shock_dep) {
            steadystate_config_nl(&fld_ss,planet.a);
        }
        else {
            steadystate_config(&fld_ss,planet.a);
        }
        fld.vs_ss[i] = fld_ss.vs;
        fld.mdot_ss[i] = fld_ss.mdot;
        fld.efficiency[i] = fld_ss.mdot/fld_ss.mdot0;

        for(j=0;j<NR;j++) {
            fld.sol[j + NR*i] = lam[j];
            fld.sol_mdot[j + NR*i] = mdot[j];
            if (params.nonlocal_torque) {
                fld.torque[j + NR*i] = dTr_nl(rc[j],j,planet.a,TRUE);
            }
            else {
                fld.torque[j + NR*i] = dTr(rc[j],planet.a);
            }
            fld.sol_ss[j + NR*i] = fld_ss.lam[j];
            fld.lamp[j+NR*i] = fld_ss.lamp[j];
            fld.lam0[j+NR*i] = fld_ss.lam0[j];
            fld.ivals_ss[j + NR*i] = fld_ss.ivals[j];
            fld.kvals_ss[j + NR*i] = fld_ss.kvals[j];
        }
    
    }
    printf("\n"); 
    printf("Outputting results...\n");
    write_hdf5_file();
    
    free_field(&fld);
    free_grid();
    free_matrix();
    free_steady_state_field(&fld_ss);
    return 1;
}

void allocate_field( Field *tmpfld) {
     int i;
     MALLOC_SAFE( (tmpfld->times = (double *)malloc(sizeof(double)*params.nt)));
     MALLOC_SAFE( (tmpfld->avals =  (double *)malloc(sizeof(double)*params.nt)));
     MALLOC_SAFE( (tmpfld->vs = (double *)malloc(sizeof(double)*params.nt)));
     MALLOC_SAFE( (tmpfld->sol = (double *)malloc(sizeof(double)*NR*params.nt)));
     MALLOC_SAFE( (tmpfld->sol_mdot =  (double *)malloc(sizeof(double)*NR*params.nt)));
     MALLOC_SAFE( (tmpfld->torque = (double *)malloc(sizeof(double)*NR*params.nt)));
     MALLOC_SAFE( (tmpfld->lami =  (double *)malloc(sizeof(double)*NR)));
     MALLOC_SAFE( (tmpfld->mdoti = (double *)malloc(sizeof(double)*NR)));

     MALLOC_SAFE( (tmpfld->mdot_ss =  (double *)malloc(sizeof(double)*params.nt)));
     MALLOC_SAFE( (tmpfld->vs_ss = (double *)malloc(sizeof(double)*params.nt)));
     MALLOC_SAFE( (tmpfld->sol_ss = (double *)malloc(sizeof(double)*NR*params.nt)));
     MALLOC_SAFE( (tmpfld->ivals_ss = (double *)malloc(sizeof(double)*NR*params.nt)));
     MALLOC_SAFE( (tmpfld->kvals_ss = (double *)malloc(sizeof(double)*NR*params.nt)));

    MALLOC_SAFE( (tmpfld->lamp = (double *)malloc(sizeof(double)*NR*params.nt)));
    MALLOC_SAFE( (tmpfld->lam0 = (double *)malloc(sizeof(double)*NR*params.nt)));
     MALLOC_SAFE( (tmpfld->efficiency = (double *)malloc(sizeof(double)*params.nt)));

     MALLOC_SAFE( (tmpfld->nu_grid =  (double *)malloc(sizeof(double)*NR)));
    for(i=0;i<NR;i++) {
        tmpfld->lami[i] = 0;
        tmpfld->mdoti[i] = 0;
        tmpfld->nu_grid[i] = 0;
    }

    for(i=0;i<params.nt;i++) {
        tmpfld->times[i] = 0;
        tmpfld->avals[i] = 0;
        tmpfld->vs[i] = 0;
        tmpfld->vs_ss[i] = 0;
        tmpfld->mdot_ss[i] = 0;
        tmpfld->efficiency[i] = 0;

    }

    for(i=0;i<NR*params.nt;i++) {
        tmpfld->sol[i]= 0;
        tmpfld->sol_mdot[i]=0;
        tmpfld->lamp[i] = 0;
        tmpfld->lam0[i] = 0;
        tmpfld->sol_ss[i] = 0;
        tmpfld->ivals_ss[i] = 0;
        tmpfld->kvals_ss[i] = 0;

    }
    return;
}

void free_field( Field *tmpfld) {

    free(tmpfld->times);
    free(tmpfld->avals);
    free(tmpfld->vs);
    free(tmpfld->sol);
    free(tmpfld->sol_mdot);
    free(tmpfld->torque);
    free(tmpfld->lami);
    free(tmpfld->mdoti);
    free(tmpfld->mdot_ss);
    free(tmpfld->vs_ss);
    free(tmpfld->sol_ss);
    free(tmpfld->lamp);
    free(tmpfld->lam0);
    free(tmpfld->efficiency);
    free(tmpfld->nu_grid);
    free(tmpfld->ivals_ss);
    free(tmpfld->kvals_ss);

    return;
}

double set_dt(void) {
    int i;
    double dt1,dt2,dt3;
    double dt_min=params.tvisc;
    for(i=0;i<NR;i++) {
        dt1 = dlr/(rc[i]*cs(rc[i]));
        dt2 = dlr*dlr/nu(rc[i]);
        if (dt1 < dt_min) {
            dt_min = dt1;
        }
        if (dt2 < dt_min) {
            dt_min = dt2;
        }
    }

    return dt_min*params.cfl;
}


void advance_system(double dt, double *t, double tend) { 

    double dt2 = tend-(*t);

    if (params.explicit_stepper) {
        dt = set_dt();
    }

    if (dt > dt2) {
        dt = dt2;
    }
 
    if (params.explicit_stepper) {
        explicit_step(dt,&planet.a,lam);
        planet.vs = calc_drift_speed(planet.a,lam);
    }
    else {
    if ((params.move_planet) && (*t >= params.release_time)) {

        if (params.move_planet_implicit) {
     //       move_planet_implicit(dt,lam,&planet.vs, &planet.a);
      //      predictor_corrector(dt,lam, &planet.vs,&planet.a); 
            multi_step(dt,lam,&planet.vs,&planet.a);
        }
        else {
            if (params.nonlocal_torque) {
                crank_nicholson_step_nl(dt,planet.a,lam);
             }
            else {
                crank_nicholson_step(dt,planet.a,lam);
              }
            move_planet(dt, lam, &planet.vs, &planet.a); 
        }
    }
    else {
        if (params.nonlocal_torque) {
            crank_nicholson_step_nl(dt,planet.a,lam);
         }
        else {
            crank_nicholson_step(dt,planet.a,lam);
          }
        planet.vs = calc_drift_speed(planet.a,lam);
    }
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
        *a -= dTr(x,rp)/( sqrt(x));
    }
    *a /= 2;
    return;
}

double dTr_nl(double x, int i, double rp, int centered) {

    if (params.shock_dep) {

        double tsh = 1.89 + .53/planet.mp;
        double f0 = planet.eps; //.45;
        double T0 = (planet.mp*params.mth)*(planet.mp*params.mth)/params.mth;
        double norm = sqrt(tsh) * f0 * T0 * .5/M_PI;
        norm /= rp;
        
        double tauval;
        if (centered) {
            tauval = tauc[i];
        }
        else {
            tauval= taumin[i];
        }


        if (tauval <= tsh) {
            return 0;
        }
        if (x < rp) {
            return -norm*tau_integrand(x/rp)*pow(tauval,-1.5);//sqrt(x);
        }
        else {
            return norm*tau_integrand(x/rp)*pow(tauval,-1.5);//sqrt(x);
        } 
    }
    else {
///        double fac = dTr(x,rp)/x;
        double nlfac=0;
        double xd = planet.xd;
        double hp = params.h*rp;
        double left_region = (xd-1)*hp/2;
        double right_region = (xd+1)*hp/2;
        if ( (fabs(x-rp) > left_region) && (fabs(x-rp) < right_region)) {
       //     printf("%lg inside %lg and %lg\n",(x-rp)/hp,left_region,right_region);
            nlfac = 1./hp;
        }
        return nlfac;

    }
}


double calc_coeffs_nl(double xm,double rp,int i) {

/*
    double tsh = 1.89 + .53/planet.mp;
    double f0 = .45;
    double T0 = (planet.mp*params.mth)*(planet.mp*params.mth)*pow(params.h,-3.);
    double norm = -sqrt(tsh) * f0 * T0 * .5/M_PI;
    if (taumin[i] <= tsh) {
        return 0;
    }
    if (xm < rp) {
        return -norm*tau_integrand(xm/rp)*pow(taumin[i],-1.5); ///sqrt(xm);
    }
    else {
        return norm*tau_integrand(xm/rp)*pow(taumin[i],-1.5); ///sqrt(xm);
    } 
*/
    return -dTr_nl(xm,i,rp,FALSE)*sqrt(xm)*2;

}

double tau_integrand(double x) {
    double norm = 2 * pow(params.h,-2.5)*pow(2.,-1.25);
    double p1 = params.gamma;
    double p2 = -.5*(params.gamma - 1.5);
    
    return norm* pow(x,(5*p2+p1)*.5 - 11./4)*pow(fabs(pow(x,1.5)-1),1.5);
}

void set_tau(double a) {
    /* Set the value of tau
     * This assumes that matrix.icol has already been set to the 
     * correct index!
     */
    int i;
    int icol = matrix.icol;

    /* taumin[icol] is tau at the inner zone edge that holds the planet */
    taumin[icol] = 0;
    tauc[icol] = 0;
    /* Outer disk */
    taumin[icol+1] = .5*dlr*tau_integrand(rmin[icol+1]/a)*rmin[icol+1]/a;
    tauc[icol+1] = .5*dlr*tau_integrand(rc[icol+1]/a)*rc[icol+1]/a;
    for(i=icol+2;i<NR-1;i++) {
        taumin[i] = taumin[i-1] + dlr*tau_integrand(rmin[i]/a)*rmin[i]/a;
        tauc[i] = tauc[i-1] + dlr*tau_integrand(rc[i]/a)*rc[i]/a;

    }
    taumin[NR-1] = taumin[NR-2] + .5*dlr*tau_integrand(rmin[NR-1]/a)*rmin[NR-1]/a;
    tauc[NR-1] = tauc[NR-2] + .5*dlr*tau_integrand(rc[NR-1]/a)*rc[NR-1]/a;
    /* Inner Disk */
    taumin[icol-1] = .5*dlr*tau_integrand(rmin[icol-1]/a)*rmin[icol-1]/a;
    tauc[icol-1] = .5*dlr*tau_integrand(rc[icol-1]/a)*rc[icol-1]/a;
    for(i=icol-2;i>0;i--) {
        taumin[i] = taumin[i+1] + .5*dlr*tau_integrand(rmin[i]/a)*rmin[i]/a;
        tauc[i] = tauc[i+1] + .5*dlr*tau_integrand(rc[i]/a)*rc[i]/a;
    }
    taumin[0] = taumin[1] + .5*dlr*tau_integrand(rmin[1]/a)*rmin[1]/a;
    tauc[0] = tauc[1] + .5*dlr*tau_integrand(rc[1]/a)*rc[1]/a;
    return;
}


void explicit_step(double dt, double *aplanet, double *y) {
/* TVD RK3 
 */
    int i;
    double a1, a2;
    double ap,am,bp,bm,rm,rp;
    double *y1 = (double *)malloc(sizeof(double)*NR);
    double *y2 = (double *)malloc(sizeof(double)*NR);
    
    a1 = *aplanet; a2 = *aplanet;

#pragma omp parallel for private(i)
    for(i=0;i<NR;i++) {
        y1[i] = 0;
        y2[i] = 0;
    }

#pragma omp parallel for private(i,rm,rp,am,ap,bm,bp) shared(rc,rmin,y,y1,dt)
    for(i=0;i<NR;i++) {
        if (i==0) {
            am=0; bm=0;
        }
        else {
            rm = rmin[i];
            calc_coeffs(rm,rc[i]-rc[i-1],&am,&bm,*aplanet,params.planet_torque);
        }
        if (i!= NR-1) {
            rp = rmin[i+1];
            calc_coeffs(rp,rc[i+1]-rc[i],&ap,&bp,*aplanet,params.planet_torque);
        }
        else {
            ap = 0; bp =0;
        }
        y1[i] = (ap-am-bm-bp)*y[i];
        if (i!=0) {
            y1[i] += (bm-am)*y[i-1];
        }
        else {
            y1[i] -= params.bc_mdot;
        }
        if (i!=NR-1) {
            y1[i] += (ap+bp)*y[i+1];
        }
        else {
            y1[i] += params.bc_mdot;
        }
        y1[i] = y[i] + dt*y1[i]/dr[i];
    }



    if (params.move_planet) {
        a1 = *aplanet + dt*calc_drift_speed(*aplanet,y);
        
    }

#pragma omp parallel for private(i,rm,rp,am,ap,bm,bp) shared(rc,rmin,y,y2,dt)
    for(i=0;i<NR;i++) {
        if (i==0) {
            am=0; bm=0;
        }
        else {
            rm = rmin[i];
            calc_coeffs(rm,rc[i]-rc[i-1],&am,&bm,*aplanet,params.planet_torque);
        }
        if (i!= NR-1) {
            rp = rmin[i+1];
            calc_coeffs(rp,rc[i+1]-rc[i],&ap,&bp,*aplanet,params.planet_torque);
        }
        else {
            ap = 0; bp =0;
        }

        y2[i] = (ap-am-bm-bp)*y1[i];
        if (i!=0) {
            y2[i] += (bm-am)*y1[i-1];
        }
        else {
            y2[i] -= params.bc_mdot;
        }
        if (i!=NR-1) {
            y2[i] += (ap+bp)*y1[i+1];
        }
        else {
            y2[i] += params.bc_mdot;
        }
        y2[i] = .75*y[i] + .25*y1[i] + .25*dt*y2[i]/dr[i];
    }
    if (params.move_planet) {
        a2 = .75*(*aplanet) + .25*a1 + .25*dt*calc_drift_speed(a1,y1);
    }

#pragma omp parallel for private(i,rm,rp,am,ap,bm,bp) shared(rc,rmin,y,y1,y2,dt)
    for(i=0;i<NR;i++) {
        if (i==0) {
            am=0; bm=0;
        }
        else {
            rm = rmin[i];
            calc_coeffs(rm,rc[i]-rc[i-1],&am,&bm,*aplanet,params.planet_torque);
        }
        if (i!= NR-1) {
            rp = rmin[i+1];
            calc_coeffs(rp,rc[i+1]-rc[i],&ap,&bp,*aplanet,params.planet_torque);
        }
        else {
            ap = 0; bp =0;
        }
    
        y[i] = (1./3)*y[i] + (2./3)*y2[i]; 
        y[i] += (2./3)*dt*((ap-am-bm-bp)*y2[i])/dr[i];
        
        if (i != 0) {
            y[i] += (2./3)*dt*(bm-am)*y2[i-1]/dr[i];
        }
        else {
            y[i] -= (2./3)*dt*params.bc_mdot/dr[i];
        }
        if (i != NR-1) {
            y[i] += (2./3)*dt*(ap+bp)*y2[i+1]/dr[i];
        }
        else {
            y[i] += (2./3)*dt*params.bc_mdot/dr[i];
        }
    }
    if (params.move_planet) {
        *aplanet = (1./3)*(*aplanet) + (2./3)*a2 + (2./3)*dt*calc_drift_speed(a2,y2);
    }
    

    free(y1); free(y2);
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



#pragma omp parallel for private(i,rm,rp,am,ap,bm,bp) shared(matrix,rc,rmin)
    for(i=1;i<NR-1;i++) {
        rm = rmin[i];
        rp = rmin[i+1];

        calc_coeffs(rm,rc[i]-rc[i-1],&am,&bm,aplanet,params.planet_torque);
        calc_coeffs(rp,rc[i+1]-rc[i],&ap,&bp,aplanet,params.planet_torque);

        matrix.md[i] = (ap-am - bm - bp)*dt/2.;
        matrix.ld[i-1] = (-am + bm)*dt/2.;
        matrix.ud[i] = (ap + bp)*dt/2.;
      }
    if (params.flux_bc) {
        calc_coeffs(rmin[NR-1],rc[NR-1]-rc[NR-2],&am,&bm,aplanet,params.planet_torque);
        calc_coeffs(rmin[1],rc[1]-rc[0],&ap,&bp,aplanet,params.planet_torque);
        matrix.md[0] = (ap-bp)*dt/2.;
        matrix.ud[0] = (ap+bp)*dt/2.;
        matrix.md[NR-1] = (-am-bm)*dt/2.;
        matrix.ld[NR-2] = (-am+bm)*dt/2.;
        matrix.fm[0] = -params.bc_mdot*dt;
        matrix.fm[NR-1] = params.bc_mdot*dt;
    }

    matvec(matrix.ld,matrix.md,matrix.ud,y,matrix.fm,NR);

#pragma omp parallel for private(i) shared(y,dr,matrix)
    for(i=0;i<NR;i++) {
        matrix.fm[i] += dr[i]*y[i];
        matrix.md[i] = dr[i] - matrix.md[i];
        if (i<NR-1) {
            matrix.ld[i] *= -1;
            matrix.ud[i] *= -1;
        }
    }
/*
    if (params.flux_bc) {
        matrix.fm[0] = -params.bc_mdot;
        matrix.fm[NR-1] = params.bc_mdot;
        matrix.md[0] = 0;
        matrix.md[NR-1] = 0;
        matrix.ld[NR-2] = 0;
        matrix.ud[0] = 0;
    }
    else {
*/
    if (!params.flux_bc) {
        matrix.md[NR-1] = 1;
        matrix.ld[NR-2] = 0;
        matrix.fm[NR-1] = params.bc_lam[1];

        matrix.md[0] = 1;
        matrix.ud[0] = 0;
        matrix.fm[0] = params.bc_lam[0];
    }
    else {
        matrix.md[0] = 1;
        matrix.ud[0] = 0;
        matrix.fm[0] = params.bc_mdot*2*rc[0]/(3*nu(rc[0]));

/*
        if (params.bc_lam[0] == 0) {
            matrix.md[0] = 1;
            matrix.ud[0] = 0;
            matrix.fm[0] = params.bc_lam[0];
        }
*/
    }
    trisolve(matrix.ld,matrix.md,matrix.ud,matrix.fm,y,NR);


    return;
}

int in_region(double x, double a, double leftr, double rightr) {

        double dist = fabs(x-a);
        if ((dist >= leftr) && (dist <= rightr)) {
            return TRUE;
        }
        else {
            return FALSE;
        }
}

double dep_func(double x, double a, double xd, double w) {
    double dist = x-a;
/*
    if (dist < 0) dist += xd*w;
    else        dist -= xd*w;
    dist *= dist;
    return exp(- dist * 2*M_PI*M_PI/(w*w))/(w);
*/
    double leftr = (xd-.5)*w;
    double rightr = (xd+.5)*w;
        
    double norm = 1./w;
    double res = smoothing(x,a+leftr,params.h/2)*(1-smoothing(x,a+rightr,params.h/2));
    res += smoothing(x,a-rightr,params.h/2)*(1-smoothing(x,a-leftr,params.h/2));
    return norm*res;




}

double dTr_kana(double x, double a) {
    
    double norm = planet.eps*.798*pow(planet.mp*params.mth,2)*pow(a,4);
    double dist = x-a;

    if (fabs(dist) < params.h*a*1.3) {
        return 0;
    }
    else {
        if (dist<0) norm *= -1;
        return norm/(x*pow(fabs(x-a),4));
    }


}

void set_weights( double *w,double *u,double a, int ia, int n) {
    int i;

    double hp = params.h * a;
    double xd = planet.xd;

    double leftr = (xd-.5)*hp;
    double rightr = (xd+.5)*hp;

    double normL,normR;

/* Lower then upper */
//#pragma omp parallel for private(i) shared(a,rc,dr,ia)
   for(i=0;i<ia;i++) {
        
        w[i] = dlr*dTr_kana(rc[i],a);  // w lower
        w[i+n] = 0;                       // w upper

//        normL = 1/hp ? in_region(rmin[i],a,leftr,rightr) : 0; 
//        normR = 1/hp ? in_region(rmin[i+1],a,leftr,rightr) : 0;
        normL = dep_func(rmin[i],a,xd,hp);
        normR = dep_func(rmin[i+1],a,xd,hp);
   
        if (i==0) {
            u[i+n] = -2*sqrt(rmin[i+1])*normR;
        }
        else {
            u[i+n] = -2*sqrt(rmin[i+1])*normR;  // u lower 
            u[i+n] -= -2*sqrt(rmin[i])*normL;
        }
        //printf("%lg\t%lg\n",(rmin[i]-a)/(params.h*a),dTr_nl(rmin[i+1],a,ia,TRUE));
        u[i+n*2] = 0;                     // u upper
    }


//#pragma omp parallel for private(i) shared(a,rc,dr)
    for(i=ia;i<n;i++) { 
        w[i+n] = dlr*dTr_kana(rc[i],a);  // w upper
        w[i] = 0;                           // w lower
//        normL = 1/hp ? in_region(rmin[i],a,leftr,rightr) : 0; 
//        normR = 1/hp ? in_region(rmin[i+1],a,leftr,rightr) : 0;
        normL = dep_func(rmin[i],a,xd,hp);
        normR = dep_func(rmin[i+1],a,xd,hp);
        if (i==n-1) {
            u[i+n*2] = 2*sqrt(rmin[i])*normL;
        }
        else {
            u[i+n*2] =  -2*sqrt(rmin[i+1])*normR;
            u[i+n*2] -= -2*sqrt(rmin[i])*normL;   // u upper
        }
        u[i+n] = 0;                           // u lower
    }

    if (params.flux_bc) {
        u[n] = 0; //-2*sqrt(rmin[1])*dTr_nl(rmin[1],i,a,TRUE);
        u[2*n+NR-1] =0;// 2*sqrt(rmin[NR-1])*dTr_nl(rmin[NR-1],i,a,TRUE);
    }
    return;
}

void crank_nicholson_step_nl(double dt, double aplanet, double *y) {
    int i;
    int need_set_flag = TRUE;
    double ap,am, bm,bp, rm, rp;
    
    for(i=0;i<NR-1;i++) {
        matrix.fm[i] = 0;
        matrix.md[i] = 0;
        matrix.ud[i] = 0;
        matrix.ld[i] = 0;
        matrix.u[i] = 0;
        if (!params.shock_dep) {
            matrix.u[i+NR] = 0;
            matrix.u[i+2*NR] = 0;
        }
        matrix.w[i] = 0;
        if (rmin[i] >= aplanet) {
            if (need_set_flag) { 
                matrix.icol = i;
                if (params.shock_dep) {
                    matrix.w[matrix.icol] = 1;
                }
                need_set_flag = FALSE;
            }
        }
    }



    if (need_set_flag) {
        printf("Couldnt find planet in domain, a=%lg\n Setting planet to last grid point at r=%lg\n",aplanet,rc[NR-1]);
        matrix.icol = NR-1;
        if (params.shock_dep) {
             matrix.w[matrix.icol] = NR-1;
         }

    }
    matrix.md[NR-1] = 0;
    matrix.fm[NR-1] = 0;
    
    FILE *fout;

    if (params.shock_dep) {
        set_tau(aplanet);
    }
    else {
        set_weights(matrix.w,matrix.u,aplanet,matrix.icol,NR);
       /*
        fout=fopen("matrix_test.dat","w");
        for(i=0;i<NR;i++) {
            fprintf(fout,"%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",rc[i],y[i],matrix.w[i],matrix.w[i+NR],matrix.u[i+NR],matrix.u[i+NR*2],dep_func(rmin[i],aplanet,planet.xd,params.h*aplanet));
        }
      fclose(fout);
      */
    }

#pragma omp parallel for private(i,rm,rp,am,ap,bm,bp) shared(matrix,rc,rmin)
    for(i=1;i<NR-1;i++) {
        rm = rmin[i];
        rp = rmin[i+1];

        calc_coeffs(rm,rc[i]-rc[i-1],&am,&bm,aplanet,FALSE);
        calc_coeffs(rp,rc[i+1]-rc[i],&ap,&bp,aplanet,FALSE);

        matrix.md[i] = (ap-am - bm - bp)*dt/2.;
        matrix.ld[i-1] = (-am + bm)*dt/2.;
        matrix.ud[i] = (ap + bp)*dt/2.;

        if (params.planet_torque && params.shock_dep) {
            am  = calc_coeffs_nl(rm,aplanet,i);
            ap = calc_coeffs_nl(rp,aplanet,i+1);
            matrix.u[i] = (ap-am)*dt/2.;
        }

      }
    if (params.flux_bc) {
        calc_coeffs(rmin[NR-1],rc[NR-1]-rc[NR-2],&am,&bm,aplanet,FALSE);
        calc_coeffs(rmin[1],rc[1]-rc[0],&ap,&bp,aplanet,FALSE);
        matrix.md[0] = (ap-bp)*dt/2.;
        matrix.ud[0] = (ap+bp)*dt/2.;
        matrix.md[NR-1] = (-am-bm)*dt/2.;
        matrix.ld[NR-2] = (-am+bm)*dt/2.;
        matrix.fm[0] = -params.bc_mdot*dt;
        matrix.fm[NR-1] = params.bc_mdot*dt;

        if (params.planet_torque && params.shock_dep) {
            am=calc_coeffs_nl(rmin[NR-1],aplanet,NR-1);
            ap=calc_coeffs_nl(rmin[1],aplanet,1);
            matrix.u[0] = ap*dt/2.;
            matrix.u[NR-1] = -am*dt/2.;
        }
    }
    
    if (params.planet_torque && !params.shock_dep) {
        for(i=0;i<matrix.nsol*NR;i++) {
            matrix.u[i] *= dt/2.;
        }
    }
    
    matvec(matrix.ld,matrix.md,matrix.ud,y,matrix.fm,NR);
    
    double nlfac=0;
    double nlfac2 = 0;
    if (params.shock_dep) {
        nlfac = y[matrix.icol];
    }
    else {
#pragma omp parallel for private(i) reduction(+:nlfac,nlfac2)
        for(i=0;i<NR;i++) {
            nlfac += matrix.w[i]*y[i];
            nlfac2 += matrix.w[i+NR]*y[i];
        }
    }

//    printf("Total torques\nOuter = %.2e\nInner = %.2e\n",nlfac2,nlfac);

#pragma omp parallel for private(i) shared(nlfac,nlfac2,y,dr,matrix)
    for(i=0;i<NR;i++) {
        matrix.fm[i] += dr[i]*y[i];
        if (!params.shock_dep) {
            matrix.fm[i] += matrix.u[i+NR*2]*nlfac2 + matrix.u[i+NR]*nlfac;
            matrix.u[i+NR] *= -1;
            matrix.u[i+2*NR] *= -1;
        }
        else {
            matrix.fm[i] += matrix.u[i]*nlfac;
            matrix.u[i] *= -1;
        }
        matrix.md[i] = dr[i] - matrix.md[i];
        if (i<NR-1) {
            matrix.ld[i] *= -1;
            matrix.ud[i] *= -1;
        }
    }
/*
    if (params.flux_bc) {
        matrix.fm[0] = -params.bc_mdot;
        matrix.fm[NR-1] = params.bc_mdot;
        matrix.md[0] = 0;
        matrix.md[NR-1] = 0;
        matrix.ld[NR-2] = 0;
        matrix.ud[0] = 0;
    }
    else {
*/
    if (!params.flux_bc) {
        matrix.md[NR-1] = 1;
        matrix.ld[NR-2] = 0;

        if (params.shock_dep) {
            matrix.u[NR-1] = 0;
            matrix.u[0] = 0;
        }
        else {
            matrix.u[NR-1+NR*2] = 0;
            matrix.u[0 + NR] = 0;
        }
        matrix.fm[NR-1] = params.bc_lam[1];

        matrix.md[0] = 1;
        matrix.ud[0] = 0;
        matrix.fm[0] = params.bc_lam[0];
    }
    else {

        if (params.bc_lam[0] == 0) {
            matrix.md[0] = 1;
            matrix.ud[0] = 0;
            if (params.shock_dep) {
                matrix.u[0] = 0;
            }
            else {
                matrix.u[0+NR] = 0;
            }
            matrix.fm[0] = params.bc_lam[0];
        }
    }


    if (params.shock_dep) {
       trisolve_sm(matrix.ld,matrix.md,matrix.ud,matrix.fm,y,matrix.u,matrix.w,matrix.icol,NR);
    }
    else {
        for (i=0;i<NR;i++) {
            matrix.u[i] = matrix.fm[i];
        }
       trisolve_smn(matrix.ld,matrix.md,matrix.ud,matrix.u,y,matrix.w,matrix.nsol,NR);

    }

    return;
}

void trisolve_smn(double *ld, double *md, double *ud, double *d,double *sol, double *wn, int nsol, int n) {
    /* Thomas algorithm using Shermin-Morrison formula for an extra matrix A + u w^T, col, at index icol
     */
     /* only works for nsol = 3 for now */


    int i,j;
    double *cp, *bp, *dp;
    double *cp2;
    double *sol2;
     cp =  (double *)malloc(sizeof(double)*(n-1)*nsol);
    bp =  (double *)malloc(sizeof(double)*n*nsol);
    dp =  (double *)malloc(sizeof(double)*n*nsol);
    
    cp2 = (double *)malloc(sizeof(double)*nsol*(nsol-1));

     sol2 =  (double *)malloc(sizeof(double)*n*nsol);
  
     int indx,cindx;
    for(j=0;j<nsol;j++) {
        indx = j*n;
        cindx = j*(n-1);
        for(i=0;i<n-1;i++) { 
            cp[i+cindx] = 0;
            bp[i+indx] = 1;
            dp[i+indx] = 0;
        }
        bp[n-1 + indx] = 0;
        dp[n-1 + indx] = 0;
    

    
        cp[0+cindx] = ud[0]/md[0];


        for(i=1;i<n-1;i++) {
            cp[i+cindx] = ud[i]/(md[i]- ld[i-1]*cp[i-1+cindx]);
        }
        dp[0+indx] = d[0+indx]/md[0];
        for(i=1;i<n;i++) {
            dp[i+indx] = (d[i+indx] - ld[i-1]*dp[i-1+indx])/(md[i]-ld[i-1]*cp[i-1+cindx]);
        }

        sol2[n-1+indx] = dp[n-1+indx];
        for(i=n-2;i>=0;i--) {
            sol2[i+indx] = dp[i+indx] - cp[i+cindx]*sol2[i+1+indx];
        }
    }

/* Sol contains the solutions to the nsol rhs 
 * Now we combine them using the nsol weights
 */

    int k;
    double res;

    for(i=0;i<nsol-1;i++) {
        for(j=0;j<nsol;j++) {
            /* Compute w_i dot sol2_j */
            res = 0;
#pragma omp parallel for private(i,j,k) reduction(+:res)
            for(k=0;k<n;k++) {
                res += wn[k + n*i] * sol2[k+j*n]; 
            }
            cp2[j+nsol*i] = res;
        }
    }
    double fac1,fac2,fac3;
    fac1=cp2[0]/(1+cp2[1]);
    fac2=cp2[2]/(1+cp2[1]);
    fac3=(-cp2[nsol]+fac1*cp2[nsol+1])/(1+cp2[nsol+2]-fac2*cp2[nsol+1]);
    
    
#pragma omp parallel for private(i)
    for(i=0;i<n;i++) {
        sol[i] = sol2[i]- (fac1+fac2*fac3)*sol2[i+n] + fac3*sol2[i+n*2];

    }

    free(cp); free(bp); free(dp);
    free(cp2);
    free(sol2);
    return;
}

void trisolve_sm(double *ld, double *md, double *ud, double *d,double *sol,double *u, double *w, int icol, int n) {
    /* Thomas algorithm using Shermin-Morrison formula for an extra matrix A + u w^T, col, at index icol
     */

    int i;
    double *cp, *bp, *dp;
    double *cp2, *bp2, *dp2;
    double *sol2;
     cp =  (double *)malloc(sizeof(double)*(n-1));
    bp =  (double *)malloc(sizeof(double)*n);
    dp =  (double *)malloc(sizeof(double)*n);

    cp2 =  (double *)malloc(sizeof(double)*(n-1));
    bp2 =  (double *)malloc(sizeof(double)*n);
    dp2 =  (double *)malloc(sizeof(double)*n);
     sol2 =  (double *)malloc(sizeof(double)*n);
  
     
    for(i=0;i<n-1;i++) {
        cp[i] = 0;
        bp[i] = 1;
        dp[i] = 0;
        cp2[i] = 0;
        bp2[i] = 1;
        dp2[i] = 0;
    }
    bp[n-1] = 0;
    dp[n-1] = 0;

    cp[0] = ud[0]/md[0];

    bp2[n-1] = 0;
    dp2[n-1] = 0;

    cp2[0] = ud[0]/md[0];
    for(i=1;i<n-1;i++) {
        cp[i] = ud[i]/(md[i]- ld[i-1]*cp[i-1]);
        cp2[i] = ud[i]/(md[i]- ld[i-1]*cp2[i-1]);
    }
    dp[0] = d[0]/md[0];
    dp2[0] = u[0]/md[0];
    for(i=1;i<n;i++) {
        dp[i] = (d[i] - ld[i-1]*dp[i-1])/(md[i]-ld[i-1]*cp[i-1]);
        dp2[i] = (u[i] - ld[i-1]*dp2[i-1])/(md[i]-ld[i-1]*cp2[i-1]);
    }

    sol[n-1] = dp[n-1];
    sol2[n-1] = dp2[n-1];
    for(i=n-2;i>=0;i--) {
        sol[i] = dp[i] - cp[i]*sol[i+1];
        sol2[i] = dp2[i] - cp2[i]*sol2[i+1];
    }

    double fac1 = 0;
    double fac2 = 0;
#pragma omp parallel for private(i) reduction(+:fac1,fac2) 
    for(i=0;i<n;i++) {
        fac1 += sol[i]*w[i];
        fac2 += sol2[i]*w[i]; 
    }
    fac1 /= (1+fac2);
    
    for(i=0;i<n;i++) {
        sol[i]  -= fac1*sol2[i];
    }

    free(cp); free(bp); free(dp);
    free(cp2); free(bp2); free(dp2);
    free(sol2);
    return;
}
void trisolve(double *ld, double *md, double *ud, double *d,double *sol,int n) {
    int i;
    double *cp, *bp, *dp;
    MALLOC_SAFE(( cp =  (double *)malloc(sizeof(double)*(n-1))));
    MALLOC_SAFE(( bp =  (double *)malloc(sizeof(double)*n)));
    MALLOC_SAFE(( dp =  (double *)malloc(sizeof(double)*n)));

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
    if (f == NULL) {
        printf("Can't find the data file lambda_init.dat!");
    }
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
#ifdef NU
    for(i=0;i<NR;i++) {
        lam[i] = 2*rc[i]/(3*nu(rc[i]));
    }
#else
    double mdot0, sig0, x;

    if (params.flux_bc) {
        for(i=0;i<NR;i++) {
     //       lam[i] = params.bc_mdot*2*rc[i]*(1 - sqrt(params.ri/rc[i]))/(3*nu(rc[i]));
            lam[i] = params.bc_mdot*2*rc[i]/(3*nu(rc[i]));
        }
    }
    else {
         mdot0 = 1.5*params.nu0 * (params.bc_lam[1]*pow(params.ro,params.gamma - .5) - params.bc_lam[0]*pow(params.ri,params.gamma-.5))/(sqrt(params.ro)-sqrt(params.ri));
     sig0 = 2*mdot0/(3*params.nu0);

     for(i=0;i<NR;i++) {
         x = rc[i]/params.ri;
         lam[i] = params.bc_lam[0]*pow(x,.5-params.gamma) + sig0*pow(rc[i],1-params.gamma)*(1-sqrt(1/x));
     }
    }
#endif
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
    if (params.nonlocal_torque && !params.shock_dep) {
        matrix.nsol = 3;

        MALLOC_SAFE(( matrix.u =  (double *)malloc(sizeof(double)*NR*matrix.nsol))); 
        MALLOC_SAFE(( matrix.w =  (double *)malloc(sizeof(double)*NR*(matrix.nsol-1)))); 
    }
    else {
        matrix.nsol = 1;
        MALLOC_SAFE(( matrix.u =  (double *)malloc(sizeof(double)*NR))); 
        MALLOC_SAFE(( matrix.w =  (double *)malloc(sizeof(double)*NR))); 

    }
    
    matrix.icol = 0;
    MALLOC_SAFE(( matrix.ld = (double *)malloc(sizeof(double)*(NR-1))));
    MALLOC_SAFE(( matrix.md = (double *)malloc(sizeof(double)*NR)));
    MALLOC_SAFE(( matrix.ud =  (double *)malloc(sizeof(double)*(NR-1))));
    MALLOC_SAFE(( matrix.fm =  (double *)malloc(sizeof(double)*NR)));  

    printf("Initializing matrices...\n"); 
    for(i=0;i<NR-1;i++) {
        matrix.ld[i] = 0;
        matrix.md[i] = 0;
        matrix.ud[i] = 0;
        matrix.fm[i] = 0;
        matrix.u[i] = 0;
        matrix.w[i] = 0;
                
    }
    matrix.u[NR-1] = 0;
    matrix.w[NR-1] = 0;
    matrix.md[NR-1] = 1;
    matrix.fm[NR-1] = 0;
    return;
}

void set_grid(void) {
    MALLOC_SAFE(( rc =  (double *)malloc(sizeof(double)*NR)));
    MALLOC_SAFE((rmin =  (double *)malloc(sizeof(double)*NR)));
    MALLOC_SAFE((lam =  (double *)malloc(sizeof(double)*NR)));
    MALLOC_SAFE((mdot = (double *)malloc(sizeof(double)*NR)));
    MALLOC_SAFE((dr  =  (double *)malloc(sizeof(double)*NR)));
    dlr = log(params.ro/params.ri)/((double)(NR-1));
    MALLOC_SAFE((lrc = (double *)malloc(sizeof(double)*NR)));
    MALLOC_SAFE((lrmin = (double *)malloc(sizeof(double)*NR)));

    MALLOC_SAFE(( taumin =  (double *)malloc(sizeof(double)*NR)));
    MALLOC_SAFE(( tauc =  (double *)malloc(sizeof(double)*NR)));

    int set_flag = TRUE;
    int i;

    for (i=0;i<NR;i++) {
        lrc[i] = log(params.ri) + i*dlr;
        rc[i] = exp(lrc[i]);
        dr[i] = rc[i]*dlr;
        lam[i] = 0;
        taumin[i] = 0;
        tauc[i] = 0;
    }
    lrmin[0] = (log(params.ri) - dlr + lrc[0])*.5;
    rmin[0] = exp(lrmin[0]);//.5*(rc[0] + exp(log(params.ri)-dlr));
    for(i=1;i<NR;i++) {
        lrmin[i] = .5*(lrc[i]+lrc[i-1]);
        rmin[i] = exp(lrmin[i]);
    }
   
    printf("Grid spacing is dlr = %.3e\n",dlr);    

    set_mdot(params.planet_torque);
    return;

}

void free_grid(void) {
    free(rc); free(rmin); free(lam); free(dr);
    free(lrc); free(lrmin);
    free(mdot); free(taumin);
    free(tauc);
     return;
}

void free_matrix(void) {
    free(matrix.md); free(matrix.ld); free(matrix.ud); free(matrix.fm);
    free(matrix.u); free(matrix.w);
    return;
}
double cs(double x) {
    return params.h*pow(x,(2*params.gamma - 3.)/4.);

}
double nu(double x) {
#ifdef NU
    double res = params.nu0*sqrt(planet.a);
#else
      double res = params.nu0 * pow(x,params.gamma);

      if (params.hs_visc) {
          double rh = pow(planet.mp*params.mth/3.,1./3) * planet.a;
          if (fabs(x-planet.a) <= rh) {
            res += 1.5*pow(rh/planet.a,3) * sqrt(planet.a)/(2*M_PI);
          }

      }
#endif

      return res;
}
double scaleH(double x) {
    return params.h * x *  pow(x, (params.gamma -.5)/2);
}
void set_params(char *parfile) {
    read_input_file(parfile);
    
    params.mach = 1/params.h;
    params.nu0 = params.alpha * params.h*params.h;
    params.mth = params.h*params.h*params.h;
    params.mvisc = sqrt(27.*M_PI/8  * params.alpha * params.mach);
    params.tvisc = params.ro*params.ro/nu(params.ro); 
    planet.rh = pow( planet.mp * params.mth/3.,1./3) * planet.a;
    planet.omp = pow(planet.a,-1.5);
    planet.dep = params.h*planet.a;
    planet.T0 = 2*M_PI*planet.a*planet.mp*planet.mp*params.mth/params.h;
    planet.vs = 0;

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

     printf("Planet properties:\n \
            \ta = %lg\n \
            \tmass = %lg mth = %.2e Mstar = %.2e mvisc\n \
            \tK = %.2e\n \
            \trh = %lg = %lg h\n",
            planet.a, 
            planet.mp,planet.mp*params.mth,planet.mp/params.mvisc,
            planet.mp*params.h/params.alpha,
            planet.rh, planet.rh/params.h);
     printf("\tplanet_torque = %s\n \
             \tmove_planet = %s\n \
             \tmove_planet_implicit = %s\n \
             \tgaussian = %s\n \
             \tread_initial_conditions = %s\n \
             \toutputname = %s\n",
             params.planet_torque ? "TRUE" : "FALSE", 
             params.move_planet ? "TRUE" : "FALSE", 
             params.move_planet_implicit ? "TRUE" : "FALSE", 
             planet.gaussian ? "TRUE" : "FALSE", 
             params.read_initial_conditions ? "TRUE" : "FALSE", 
             params.outputname);
    return;
}

double dTr(double x,double a) {
    double left_fac, right_fac; 
    double norm, xi,res;

    if (planet.gaussian == TRUE) {
        xi = (x-a)/planet.dep;
        left_fac = (xi-planet.beta)/planet.delta;
        right_fac = (xi+planet.beta)/planet.delta;
        left_fac = exp(-left_fac*left_fac);
        right_fac = exp(-right_fac*right_fac);

        norm = planet.T0/(planet.delta * sqrt(M_PI));

        res = -norm*( (planet.G1+1)*right_fac - left_fac);
    }
   
    else {
        xi = (x-a) / scaleH(a);

 //       norm = planet.eps * a*M_PI*(planet.mp*params.mth)*(planet.mp*params.mth); 
        norm = planet.eps * (planet.mp*params.mth)*(planet.mp*params.mth);
        if (planet.symmetric_torque) {
            right_fac = norm*pow(a/fmax(scaleH(x),fabs(x-a)),4);
        }
        else {
            right_fac = norm*pow(x/fmax(scaleH(x),fabs(x-a)),4);
        }
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


double calc_total_torque(double a, double *y) {
    int i;
    double res = 0;

#pragma omp parallel for private(i) reduction(+:res) 
    for(i=1;i<NR-1;i++) {
        if (params.nonlocal_torque) {
            res += dTr_nl(rc[i],i,a,TRUE)*y[i]/2;  // Factor of 1/2 from torque normalization

        }
        else {
            res += dTr(rc[i],a)*y[i]/2;  // Factor of 1/2 from torque normalization
        }
    }
    if (params.nonlocal_torque) {
        res += .5*(dTr_nl(rc[0],0,a,TRUE)*y[0] + dTr_nl(rc[NR-1],NR-1,a,TRUE)*y[NR-1])/2;
    }
    else {
        res += .5*(dTr(rc[0],a)*y[0] + dTr(rc[NR-1],a)*y[NR-1])/2;
    }
//    res *= -2*dlr*sqrt(a)/(planet.mp*params.mth);

    return res*dlr;

}

double calc_drift_speed(double a,double *y) {
    double T = calc_total_torque(a,y);
    return -2*sqrt(a)*T/(planet.mp*params.mth);
}


void move_planet(double dt, double *y, double *vs, double *a) {
    double T = calc_total_torque(*a,y);
    double q = planet.mp*params.mth;

    double L0 = sqrt(*a) * q;
    double L1;
    L1 = L0 - dt*T;
    
    *vs = -2 * (*a) * T / L0;
    *a = pow(L1/q,2);
/*
    *vs = calc_drift_speed(*a,y);
    planet.a += dt*(*vs);
    *a += dt*(*vs);
*/
    return;
}

void set_mdot(int planet_torque) {
    int i;
    double ca, cb;
    
    for(i=0;i<NR;i++) {
        ca = 3*nu(rc[i])/rc[i];
        cb = ca*(params.gamma - .5);

        if (planet_torque == TRUE) {
            if (params.nonlocal_torque) {
                cb -= 2*sqrt(rc[i])*dTr_nl(rc[i],i,planet.a,TRUE);
            }
            else {
                cb -= dTr(rc[i],planet.a)/(sqrt(rc[i]));
            }
        }  
        mdot[i] = cb*lam[i]; 
        if (i == 0){ 
            mdot[i] += ca*(lam[i+1]-params.bc_lam[0])/(dlr);
        }
        else {
            if (i==NR-1) {
                if (params.flux_bc) {
                    mdot[i] = params.bc_mdot;
                }
                else {
                    mdot[i] += ca*(params.bc_lam[1] - lam[i-1])/(2*dlr);
                }
            }
            else {
                mdot[i] += ca*(lam[i+1]-lam[i-1])/(2*dlr);
            }
        }
    }



    return;
}


int locate(double *xx, int nx, double x) {
/* Bisection lookup routine. 
 * Given the ordered abcissas xx[0:nx-1], find the value of index j
 * such that x lies between xx[j] and xx[j+1].
 * Return *j=-1 or *j=nx if x lies outside the range of xx. 
*/
    
    int ju, jm, jl;
    int ascnd;

    jl = -1;
    ju = nx;

    ascnd = (xx[nx-1] >= xx[0]);

    while (ju-jl > 1) {
        jm = (ju + jl) >> 1;
        if (x >= xx[jm] == ascnd) {
            jl = jm;
        }
        else {
            ju = jm;
        }
    }
    if (x == xx[0]) {
        return 0;
    }
    else {
        if (x == xx[nx-1]) {
            return  nx-2;
        }
        else {
            return  jl;
        }
    }
}

void linear_interpolation(double *x,double *y,double *xd, double *yd,int nx,int nd) {
/* Linear interpolation of the data xd,yd with nd data points onto the nx x-values 
 * stored in x. Result stored in y. Both x and xd should be in order.
*/
    int  i,j;
  
    for(i=0;i<nx;i++) {
        j = locate(xd,nd,x[i]);
        if (j==-1) {
            printf("Grid point %lg below interpolation range %lg\n",x[i],xd[0]);
            y[i] = 0;
        }
        else {
            if (j==nd) {
                printf("Grid point %lg above interpolation range %lg\n",x[i],xd[nd-1]);
                y[i] = 0;
            }
            else {
                y[i] = yd[j] + (yd[j+1] - yd[j])*(x[i]-xd[j])/(xd[j+1]-xd[j]);
            }
        }

    }
    return;
}




void move_planet_implicit(double dt, double *y, double *vs, double *a) {
    double tol = 1e-4;
    double a_old = *a;
    double args[2] = {dt,a_old};
    
    double *lam_old;
     MALLOC_SAFE(( lam_old =(double *)malloc(sizeof(double)*NR)));
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
    planet.a = secant_method*(&planet_zero_function,a_old, .5*a_old,tol,args)
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
    if (params.nonlocal_torque) {
        crank_nicholson_step_nl(dt,a,y);
    }
    else {
        crank_nicholson_step(dt,a,y);
    }
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
    if (params.nonlocal_torque) {
        crank_nicholson_step_nl(dt,*a,y);
    }
    else {
        crank_nicholson_step(dt,*a,y);
    }
    double vs1 = calc_drift_speed(*a,y);
    *a += .5*dt*(vs1 - (*vs));
    *vs = vs1;
    return;
}

void multi_step(double dt, double *y, double *vs, double *a) {
    double vs1 = calc_drift_speed(*a,y);
    *a += .5*dt*vs1;

    if (params.nonlocal_torque) {
        crank_nicholson_step_nl(dt,*a,y);
    }
    else {
        crank_nicholson_step(dt,*a,y);
    }

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


/*  
   Grid Parameters:
    nr = 256;
    ri = 0.05;
    ro = 30;
   
    Disk Parameters:
    alpha = .1;
    gamma = 0.5;
    h = .1;
   
    Boundary Conditions:

    bc_lam[0] = 1e-12;
    bc_lam[1] = 1e-2;

    Time Parameters:
    dt = 3;
    nvisc = 1;
    nt = 100;
    release_time = 0;
    read_initial_conditions = FALSE;

  
    Planet Properties:

    planet_torque = TRUE;
    move_planet = TRUE;
    move_planet_implicit = TRUE;
    gaussian = FALSE;
    onesided = 0;
    a  = 10.;
    mp = 1;
    G1 = 0;
    beta = 2./3;
    delta = .1;
    c = 2./3;
    eps = 1;
*/    


void read_input_file(char *fname) {
    char garbage[100],tmpstr[MAXSTRLEN];
    char *gchar;
    int read_res;
    FILE *f;




    f = fopen(fname,"r");

    if (f==NULL) printf("\n\nERROR Can't Find Input File!\n\n");
  
	gchar=fgets(garbage,sizeof(garbage),f);	// Grid Parameters

    read_res=fscanf(f,"nr = %d \n",&NR);
    read_res=fscanf(f,"ri = %lg \n",&params.ri);
    read_res=fscanf(f,"ro = %lg \n",&params.ro);   
    gchar=fgets(garbage,sizeof(garbage),f); //	Disk Parameters:
    read_res=fscanf(f,"alpha = %lg \n",&params.alpha); 
    read_res=fscanf(f,"gamma = %lg \n",&params.gamma);
    read_res=fscanf(f,"h = %lg \n",&params.h);
   
    gchar=fgets(garbage,sizeof(garbage),f);	//Boundary Conditions:

    read_res=fscanf(f,"bc_lam_inner = %lg \n",&params.bc_lam[0]);
    read_res=fscanf(f,"bc_lam_outer = %lg \n",&params.bc_lam[1]);
    read_res=fscanf(f,"bc_mdot = %lg \n",&params.bc_mdot);
    read_res=fscanf(f,"flux_bc = %s \n",tmpstr);
    set_bool(tmpstr,&params.flux_bc);
    
    gchar=fgets(garbage,sizeof(garbage),f); //	Time Parameters:
    read_res=fscanf(f,"dt = %lg \n",&params.dt);
    read_res=fscanf(f,"cfl = %lg \n",&params.cfl);
    read_res=fscanf(f,"nvisc = %lg \n",&params.nvisc);
    read_res=fscanf(f,"nt = %d \n",&params.nt);
    read_res=fscanf(f,"release_time = %lg \n",&params.release_time);
    read_res=fscanf(f,"start_ss = %s \n",tmpstr);
    set_bool(tmpstr,&params.start_ss);
    read_res=fscanf(f,"read_initial_conditions = %s \n",tmpstr);
    set_bool(tmpstr,&params.read_initial_conditions);
  
    gchar=fgets(garbage,sizeof(garbage),f);	// Planet Properties:

    read_res=fscanf(f,"planet_torque = %s \n",tmpstr);
    set_bool(tmpstr,&params.planet_torque);
    read_res=fscanf(f,"explicit_stepper = %s \n",tmpstr);
    set_bool(tmpstr,&params.explicit_stepper);
    read_res=fscanf(f,"move_planet = %s \n",tmpstr);
    set_bool(tmpstr,&params.move_planet);
    read_res=fscanf(f,"move_planet_implicit = %s \n",tmpstr);
    set_bool(tmpstr,&params.move_planet_implicit);
    read_res=fscanf(f,"gaussian = %s \n",tmpstr);
    set_bool(tmpstr,&planet.gaussian);
    
    read_res=fscanf(f,"symmetric_torque = %s \n",tmpstr);
    set_bool(tmpstr,&planet.symmetric_torque);

    read_res=fscanf(f,"nonlocal_torque = %s \n",tmpstr);
    set_bool(tmpstr,&params.nonlocal_torque);
    read_res=fscanf(f,"shock_dep = %s \n",tmpstr);
    set_bool(tmpstr,&params.shock_dep);
    read_res=fscanf(f,"hs_visc = %s \n",tmpstr);
    set_bool(tmpstr,&params.hs_visc);

    read_res=fscanf(f,"one_sided = %lg \n",&planet.onesided);
    read_res=fscanf(f,"a  = %lg \n",&planet.a);
    read_res=fscanf(f,"mp = %lg \n",&planet.mp);
    read_res=fscanf(f,"G1 = %lg \n",&planet.G1);
    read_res=fscanf(f,"beta = %lg \n",&planet.beta);
    read_res=fscanf(f,"delta = %lg \n",&planet.delta);
    read_res=fscanf(f,"c = %lg \n",&planet.c);
    read_res=fscanf(f,"eps = %lg \n",&planet.eps);
    read_res=fscanf(f,"xd = %lg \n",&planet.xd);
    
    read_res=fscanf(f,"outputname = %s\n",params.outputname); 
    fclose(f);
    printf("Outputting results to %s...\n",params.outputname);

/*
    rm_sub_string(fname,"params.in");
    char *dirname = fname;
    sprintf(outputname,"%s%s.hdf5",dirname,inputstr);
*/
// Leave me alone compiler
    read_res += 1; garbage[0] = gchar[0];


  return;

}

void set_bool(char *buff, int *val) {
    if ( (!strncmp(buff,"TRUE",MAXSTRLEN)) || (!strncmp(buff,"true",MAXSTRLEN)) || (!strncmp(buff,"True",MAXSTRLEN)) || (!strncmp(buff,"T",MAXSTRLEN)) || (!strncmp(buff,"T",MAXSTRLEN)) || (!strncmp(buff,"1",MAXSTRLEN))) {
        *val = TRUE;
            }
    else {
        *val = FALSE;
            }
    return;

}

void write_hdf5_double(double *data, hsize_t *dims, int ndims, hid_t group_path, char *name) {
  hid_t dspc_id, dset_id;
 
  dspc_id = H5Screate_simple(ndims,dims,NULL);
  dset_id = H5Dcreate(group_path,name,H5T_NATIVE_DOUBLE,dspc_id,H5P_DEFAULT);

    HDF5_INSERT_ERROR( H5Dwrite(dset_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,data) );
	
    HDF5_INSERT_ERROR( H5Sclose(dspc_id));
   HDF5_INSERT_ERROR( H5Dclose(dset_id));



  return;


}
void write_hdf5_file(void) {
  
  printf("Outputting Results to %s...\n",params.outputname);
  
  hid_t file_id = H5Fcreate(params.outputname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t root_id = H5Gcreate(file_id,"/Migration",0);
  hid_t mesh_id = H5Gcreate(root_id,"Mesh",0);
  hid_t solution_id = H5Gcreate(root_id,"Solution",0);
  hid_t matrix_id = H5Gcreate(root_id,"Matrix",0);
  hid_t params_id = H5Gcreate(root_id,"Parameters",0);
    hid_t steadystate_id = H5Gcreate(root_id,"SteadyState",0);


    hsize_t dims1[1] = {NR};
    hsize_t dims1_t[1]= {params.nt};
    hsize_t dims1_small[1] = {NR-1};
    hsize_t dims2[2] = {params.nt,NR};

// Write Mesh data
    write_hdf5_double(rc,dims1,1,mesh_id,"rc");
    write_hdf5_double(dr,dims1,1,mesh_id,"dr");
      write_hdf5_double(rmin,dims1,1,mesh_id,"rmin");
      write_hdf5_double(fld.lami,dims1,1,mesh_id,"lami");
      write_hdf5_double(fld.mdoti,dims1,1,mesh_id,"mdoti");
      write_hdf5_double(fld.nu_grid,dims1,1,mesh_id,"nu_grid");
    write_hdf5_double(tauc,dims1,1,mesh_id,"tauc");
    write_hdf5_double(taumin,dims1,1,mesh_id,"taumin");
// Write Matrix
   write_hdf5_double(matrix.md,dims1,1,matrix_id,"md");
    write_hdf5_double(matrix.ld,dims1_small,1,matrix_id,"ld");
    write_hdf5_double(matrix.ud,dims1_small,1,matrix_id,"ud");
   write_hdf5_double(matrix.fm,dims1,1,matrix_id,"fm");

// Write Solution
    write_hdf5_double(fld.sol,dims2,2,solution_id,"lam");
    write_hdf5_double(fld.torque,dims2,2,solution_id,"torque");
    write_hdf5_double(fld.sol_mdot,dims2,2,solution_id,"mdot");
    write_hdf5_double(fld.times,dims1_t,1,solution_id,"times");

     write_hdf5_double(fld.avals,dims1_t,1,solution_id,"avals");
     write_hdf5_double(fld.vs,dims1_t,1,solution_id,"vs");
  
// Steady State Solution
    write_hdf5_double(fld.sol_ss,dims2,2,steadystate_id,"lam_ss");
    write_hdf5_double(fld.lamp,dims2,2,steadystate_id,"lamp");
    write_hdf5_double(fld.lam0,dims2,2,steadystate_id,"lam0");
    write_hdf5_double(fld.mdot_ss,dims1_t,1,steadystate_id,"mdot_ss");
    write_hdf5_double(fld.vs_ss,dims1_t,1,steadystate_id,"vs_ss");
    write_hdf5_double(fld.efficiency,dims1_t,1,steadystate_id,"eff");
    write_hdf5_double(fld.ivals_ss,dims1_t,1,steadystate_id,"ivals_ss");
    write_hdf5_double(fld.kvals_ss,dims1_t,1,steadystate_id,"kvals_ss");

    write_hdf5_params(&params_id);

    HDF5_INSERT_ERROR(H5Gclose(mesh_id));
  HDF5_INSERT_ERROR(H5Gclose(matrix_id));
  HDF5_INSERT_ERROR(H5Gclose(solution_id));
 HDF5_INSERT_ERROR(H5Gclose(params_id));
 HDF5_INSERT_ERROR(H5Gclose(steadystate_id));

  HDF5_INSERT_ERROR(H5Gclose(root_id));
  HDF5_INSERT_ERROR(H5Fclose(file_id));


  return;


}
void write_hdf5_params(hid_t *params_id) {
  hid_t memtype,dspc_id, dset_id;
  hsize_t dims[1] = {1};




    param_t out_par;
    out_par.nr = params.nr;
    out_par.ri = params.ri;
    out_par.ro = params.ro;
    out_par.alpha= params.alpha;
    out_par.gamma= params.gamma;
    out_par.h = params.h;
    out_par.bc_lam_inner =  params.bc_lam[0];
    out_par.bc_lam_outer = params.bc_lam[1];
    out_par.bc_mdot = params.bc_mdot;
    out_par.flux_bc = params.flux_bc;
    out_par.dt= params.dt;
    out_par.cfl = params.cfl;
    out_par.nvisc = params.nvisc;
    out_par.nt = params.nt;
    out_par.release_time = params.release_time;
    out_par.start_ss = params.start_ss;
    out_par.read_initial_conditions = params.read_initial_conditions;
    out_par.planet_torque = params.planet_torque;
    out_par.explicit_stepper = params.explicit_stepper;
    out_par.move_planet = params.move_planet;
    out_par.move_planet_implicit = params.move_planet_implicit;
    out_par.gaussian = planet.gaussian;
    out_par.symmetric_torque = planet.symmetric_torque;
    out_par.nonlocal_torque = params.nonlocal_torque;
    out_par.shock_dep = params.shock_dep;
    out_par.hs_visc = params.hs_visc;
    out_par.one_sided = planet.onesided;
    out_par.a = planet.a;
    out_par.mp = planet.mp;
    out_par.G1 = planet.G1;
    out_par.beta = planet.beta;
    out_par.delta = planet.delta;
    out_par.c = planet.c;
    out_par.eps = planet.eps;
    out_par.xd = planet.xd;

    memtype = H5Tcreate (H5T_COMPOUND, sizeof (param_t));
     HDF5_INSERT_ERROR(H5Tinsert (memtype, "nr", HOFFSET (param_t, nr), H5T_NATIVE_INT));
 
    HDF5_INSERT_ERROR(H5Tinsert (memtype, "ri", HOFFSET (param_t, ri), H5T_NATIVE_DOUBLE));

    HDF5_INSERT_ERROR(H5Tinsert (memtype, "ro", HOFFSET (param_t, ro), H5T_NATIVE_DOUBLE));

   HDF5_INSERT_ERROR(H5Tinsert (memtype, "alpha", HOFFSET (param_t, alpha), H5T_NATIVE_DOUBLE));

   HDF5_INSERT_ERROR(H5Tinsert (memtype, "gamma", HOFFSET (param_t, gamma), H5T_NATIVE_DOUBLE));

   HDF5_INSERT_ERROR(H5Tinsert (memtype, "h", HOFFSET (param_t, h), H5T_NATIVE_DOUBLE));

   HDF5_INSERT_ERROR(H5Tinsert (memtype, "bc_lam_inner", HOFFSET (param_t, bc_lam_inner), H5T_NATIVE_DOUBLE));

   HDF5_INSERT_ERROR(H5Tinsert (memtype, "bc_lam_outer", HOFFSET (param_t, bc_lam_outer), H5T_NATIVE_DOUBLE));

   HDF5_INSERT_ERROR(H5Tinsert (memtype, "bc_mdot", HOFFSET (param_t, bc_mdot), H5T_NATIVE_DOUBLE));
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "flux_bc", HOFFSET (param_t, flux_bc), H5T_NATIVE_INT));

   HDF5_INSERT_ERROR(H5Tinsert (memtype, "dt", HOFFSET (param_t, dt), H5T_NATIVE_DOUBLE));
   HDF5_INSERT_ERROR(H5Tinsert (memtype, "cfl", HOFFSET (param_t, cfl), H5T_NATIVE_DOUBLE));

   HDF5_INSERT_ERROR(H5Tinsert (memtype, "nvisc", HOFFSET (param_t, nvisc), H5T_NATIVE_DOUBLE));

      HDF5_INSERT_ERROR(H5Tinsert (memtype, "nt", HOFFSET (param_t, nt), H5T_NATIVE_INT));
 
      HDF5_INSERT_ERROR(H5Tinsert (memtype, "release_time", HOFFSET (param_t, release_time), H5T_NATIVE_DOUBLE));
 
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "start_ss", HOFFSET (param_t, start_ss), H5T_NATIVE_INT));
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "read_initial_conditions", HOFFSET (param_t, read_initial_conditions), H5T_NATIVE_INT));
 
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "planet_torque", HOFFSET (param_t, planet_torque), H5T_NATIVE_INT));
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "explicit_stepper", HOFFSET (param_t, explicit_stepper), H5T_NATIVE_INT));
 
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "move_planet", HOFFSET (param_t, move_planet), H5T_NATIVE_INT));
 
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "move_planet_implicit", HOFFSET (param_t, move_planet_implicit), H5T_NATIVE_INT));
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "gaussian", HOFFSET (param_t, gaussian), H5T_NATIVE_INT));
 
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "symmetric_torque", HOFFSET (param_t, symmetric_torque), H5T_NATIVE_INT));
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "nonlocal_torque", HOFFSET (param_t, nonlocal_torque), H5T_NATIVE_INT));
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "shock_dep", HOFFSET (param_t, shock_dep), H5T_NATIVE_INT));
 HDF5_INSERT_ERROR(H5Tinsert (memtype, "hs_visc", HOFFSET (param_t, hs_visc), H5T_NATIVE_INT));

    HDF5_INSERT_ERROR(H5Tinsert (memtype, "one_sided", HOFFSET (param_t, one_sided), H5T_NATIVE_DOUBLE));
      HDF5_INSERT_ERROR(H5Tinsert (memtype, "a", HOFFSET (param_t, a), H5T_NATIVE_DOUBLE));
      HDF5_INSERT_ERROR(H5Tinsert (memtype, "mp", HOFFSET (param_t, mp), H5T_NATIVE_DOUBLE));
      HDF5_INSERT_ERROR(H5Tinsert (memtype, "G1", HOFFSET (param_t, G1), H5T_NATIVE_DOUBLE));
      HDF5_INSERT_ERROR(H5Tinsert (memtype, "beta", HOFFSET (param_t,beta), H5T_NATIVE_DOUBLE));
      HDF5_INSERT_ERROR(H5Tinsert (memtype, "delta", HOFFSET (param_t, delta), H5T_NATIVE_DOUBLE));
      HDF5_INSERT_ERROR(H5Tinsert (memtype, "c",HOFFSET (param_t,c), H5T_NATIVE_DOUBLE));
        HDF5_INSERT_ERROR(H5Tinsert (memtype, "eps",HOFFSET (param_t,eps), H5T_NATIVE_DOUBLE));
        HDF5_INSERT_ERROR(H5Tinsert (memtype, "xd",HOFFSET (param_t,xd), H5T_NATIVE_DOUBLE));

  printf("%lg\n\n\n",out_par.xd); 


  dspc_id = H5Screate_simple(1,dims,NULL);
  dset_id = H5Dcreate(*params_id,"Parameters",memtype,dspc_id,H5P_DEFAULT);

  HDF5_INSERT_ERROR(H5Dwrite(dset_id,memtype,H5S_ALL,H5S_ALL,H5P_DEFAULT,&out_par));

  HDF5_INSERT_ERROR(H5Sclose(dspc_id));
  HDF5_INSERT_ERROR(H5Dclose(dset_id));

    return;
}

void allocate_steady_state_field(SteadyStateField *tmpfld) {
    int i;
    MALLOC_SAFE( ( tmpfld->lam = (double *)malloc(sizeof(double)*NR)) );
    MALLOC_SAFE( ( tmpfld->lam0 = (double *)malloc(sizeof(double)*NR)) );
    MALLOC_SAFE( ( tmpfld->lamp = (double *)malloc(sizeof(double)*NR)) );
    MALLOC_SAFE(( tmpfld->ivals = (double *)malloc(sizeof(double)*NR)));
    MALLOC_SAFE(( tmpfld->kvals = (double *)malloc(sizeof(double)*NR)));


    for(i=0;i<NR;i++) {
        tmpfld->lam[i] = 0;
        tmpfld->lamp[i] = 0;
        tmpfld->lam0[i] = 0;
        tmpfld->ivals[i] = 0;
        tmpfld->kvals[i] = 0;
    }
    return;
}

void free_steady_state_field(SteadyStateField *tmpfld) {
    free(tmpfld->lam);
    free(tmpfld->lam0);
    free(tmpfld->lamp);
    free(tmpfld->ivals); 
    free(tmpfld->kvals);

    return;
}
void steadystate_config(SteadyStateField *tmpfld, double a) {
    int i;
    double res;

    tmpfld->a = a;



    tmpfld->ivals[0] = 0;
    for(i=1;i<NR;i++) {
        if (params.nonlocal_torque) {
            res = -dlr*dTr_nl(rc[i],i,a,TRUE)*rc[i]*sqrt(rc[i])/(3*nu(rc[i]));

        }
        else {
            res = -dlr*dTr(rc[i],a)*sqrt(rc[i])/(3*nu(rc[i]));
        }
        tmpfld->ivals[i] = tmpfld->ivals[i-1] + res;
    }

#pragma omp parallel for private(i)
    for(i=0;i<NR;i++) {
        tmpfld->ivals[i] = exp(tmpfld->ivals[i]); // * pow(rc[i]/params.ri,params.gamma-.5);
    }


    tmpfld->kvals[0] = 0;
    for(i=1;i<NR;i++) {
//        res = dr[i]*tmpfld->ivals[i]/(3*nu(rc[i]));
        res = dlr * (tmpfld->ivals[i])*sqrt(rc[i])/2;
        tmpfld->kvals[i] = tmpfld->kvals[i-1] + res;
    }

    if (params.flux_bc) {
        tmpfld->mdot = params.bc_mdot;
    }
    else {
        tmpfld->mdot = (params.bc_lam[1]*tmpfld->ivals[NR-1] - params.bc_lam[0])/tmpfld->kvals[NR-1];
    }

    if (params.flux_bc) {
        
#pragma omp parallel for private(i)
        for(i=0;i<NR;i++) {
         tmpfld->lam[i] =  (tmpfld->mdot * tmpfld->kvals[i])/tmpfld->ivals[i];
         tmpfld->lam[i] += tmpfld->mdot * sqrt(params.ri)/tmpfld->ivals[i];
         tmpfld->lam[i] *= 2*sqrt(rc[i])/(3*nu(rc[i]));

         }
    }
    else {
#pragma omp parallel for private(i)
        for(i=0;i<NR;i++) {
         tmpfld->lam[i] = (params.bc_lam[0] + tmpfld->mdot * tmpfld->kvals[i])/tmpfld->ivals[i];

         }
    }
    if (params.flux_bc) {
        tmpfld->mdot0 = params.bc_mdot;
    }
    else {
        tmpfld->mdot0 = 1.5*params.alpha*params.h*params.h;
        tmpfld->mdot0  *= (params.bc_lam[1]*pow(params.ro,params.gamma-.5)-params.bc_lam[0]*pow(params.ri,params.gamma-.5))/(sqrt(params.ro)-sqrt(params.ri));
    }
/*
    tmpfld->vs = (tmpfld->lam[NR-1])*nu(params.ro)*1.5/sqrt(params.ro)
                     -(tmpfld->mdot)*(sqrt(params.ro)-sqrt(params.ri));
    tmpfld->vs *=  -2*sqrt(planet.a)/(planet.mp*params.mth);
*/
    /*
    tmpfld->vs = -1.5*params.alpha*params.h*params.h*2*sqrt(a)*(params.bc_lam[1]-params.bc_lam[0]) *(1 - (tmpfld->mdot)/(tmpfld->mdot0));

    tmpfld->vs /= (planet.mp*params.mth);
    */
#pragma omp parallel for private(i)
    for(i=0;i<NR;i++) {
        
        tmpfld->lam0[i] = params.bc_mdot*2*rc[i]/(3*nu(rc[i])); 
            //        tmpfld->lam0[i] += params.bc_lam[0] /pow(rc[i]/params.ri,params.gamma-.5);
     
        if (tmpfld->lam0[i] == 0) {
            tmpfld->lamp[i] = 0;
        }
        else {
            tmpfld->lamp[i] = (tmpfld->lam[i] - tmpfld->lam0[i])/(tmpfld->lam0[i]);
        }

    }
    tmpfld->vs = tmpfld->lamp[NR-1]*sqrt(rc[NR-1]);
    tmpfld->vs *= -params.bc_mdot * 2*sqrt(a)/(planet.mp*params.mth);
    return;

}

void steadystate_config_nl(SteadyStateField *tmpfld, double a) {
    int i;
    double res;

    tmpfld->a = a;

    double k = pow(planet.mp*params.mth,2)/(params.alpha*pow(params.h,5));
    k /= (3*M_PI);
    double f = 0;
    double fac = 0;
    double fp = planet.eps;
    double tauval = 0;
    double rval;
    double tsh = 1.89 + .53/planet.mp;
    set_tau(a);

    if (params.flux_bc) {
        tmpfld->mdot = params.bc_mdot;
        tmpfld->mdot0 = params.bc_mdot;
    }
#pragma omp parallel for private(i)
    for(i=0;i<NR;i++) {
        
        tmpfld->lam0[i] = (tmpfld->mdot0) * 2*rc[i] *( 1- sqrt(params.ri/rc[i]))/(3*nu(rc[i]));
//        tmpfld->lam0[i] += params.bc_lam[0] /pow(rc[i]/params.ri,params.gamma-.5);
       

    }

    for(i=0;i<NR;i++) {
        tauval = tauc[i];
        rval = rc[i];
        if (tauval <= tsh) {
            f = fp;
        }
        else {
            f = fp*sqrt(tsh/tauval);
        }

        fac = (sqrt(a)-sqrt(params.ri))/(sqrt(rval)-sqrt(params.ri));


        tmpfld->lamp[i] = -(k*f*fac/(1+k*fp));
        tmpfld->lam[i] = tmpfld->lam0[i]*(1+tmpfld->lamp[i]);
    
    }

    tmpfld->vs = tmpfld->lamp[NR-1];
    tmpfld->vs *= -2*sqrt(planet.a) * params.bc_mdot * (sqrt(rc[NR-1])-sqrt(params.ri));
    tmpfld->vs /= (planet.mp*params.mth);

    
    return;

}



