from scipy.integrate import simps
from scipy.interpolate import interp1d
from numpy import *
from matplotlib.pyplot import *
import thomas
import multiprocessing as mpi

class Status():
    def __init__(self,t,r,**kwargs):
        nr = len(r)
        nt = len(t)

        self.alpha = kwargs['alpha']
        self.gamma = kwargs['gamma']
        self.h = kwargs['h']
        self.mp = kwargs['mp']

        self.lam = zeros((nr,nt))
        self.dlam = zeros((nr,nt))
        self.sigma = zeros((nr,nt))
        self.vr = zeros(self.sigma.shape)
        self.mdot = zeros(self.sigma.shape)
        self.dTr = zeros(self.sigma.shape)
        self.a = zeros(t.shape)
        self.vs = zeros(t.shape)

        self.t = t
        self.r = r
        self.set_vnorm()
        self.rr = zeros(self.lam.shape)
        self.vv = zeros(self.lam.shape)
        for i in range(nt):
            self.rr[:,i] = self.r
            self.vv[:,i] = self.vnorm
    def nu(self,r):
        return self.alpha*self.h**2 * r**(self.gamma)
    def set_vnorm(self):
        self.vnorm = -1.5*self.nu(self.r)/self.r

    def animate(self,q='lam',ref_line=None,ylims=None,logx=False,logy=False,linetype='-',plot_planet=False,multiplier=1):

        if q=='lam':
            dat = self.lam
            ystr = '$\\lambda$'
        if q=='mdot':
            dat = self.mdot
            ystr = '$\\dot{M}$'
        if q=='vr':
            dat = self.vr/self.vv
            ystr = '$v_r$'
        if q=='dTr':
            dat = self.dTr
            ystr = '$\\frac{d T}{d r}$'
        if q == 'dlam':
            dat = self.dlam
            ystr = '$\\frac{d \\lambda}{d r}$'

        fig=figure()
        ax=fig.add_subplot(111)
        line, = ax.plot(self.r,dat[:,0]*multiplier,linetype)
        ax.plot(self.r,dat[:,0]*multiplier,color='grey')
        if plot_planet:
            line2, = ax.plot(self.a[0],1,'o',markersize=10)

        if ref_line != None:
            ax.plot(self.r,ref_line,'--k')

        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

        if ylims != None:
            ax.set_ylim(ylims)
        ax.set_xlabel('$r$',fontsize=20)
        ax.set_ylabel(ystr,fontsize=20)
        ax.set_title('t = 0',fontsize=20)
        fig.canvas.draw()

        for i,ti in enumerate(self.t[1:],start=1):
            ax.set_title('t = %.1e' % ti)
            line.set_ydata(dat[:,i]*multiplier)
            if plot_planet:
                line2.set_xdata(self.a[i])
            fig.canvas.draw()

    def plot(self,ivals=0,q='lam',logx=False,logy=False):
        if type(ivals)  == int:
            ivals = [ivals]
        fig=figure();
        ax=fig.add_subplot(111)
        if q=='lam':
            dat = self.lam
        if q=='vr':
            dat = self.vr
        if q=='sigma':
            dat = self.sigma
        if q=='mdot':
            dat = -self.lam*self.vr

        for i in ivals:
            ax.plot(self.r,dat[:,i])

        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

class Simulation():
    def __init__(self,alpha=.06,k=1,h=.1,mp=1,planet_a=10,beta=2./3,mdot=1e-6,mdot0=1e-6,G1=0,delta=.3,ri=4.0,ro=30.,nr=1024,move_planet=False,planet_torque=True,logarithmic_spacing=False,uniform_spacing=False,bc_lam=[1e-3,1]):
        self.alpha = alpha
        self.k = k
        self.h = h
        self.mp = mp
        self.planet_a = planet_a
        self.beta = beta
        self.rh = (mp*h**3/3)**(1./3) * planet_a
        self.dep = max(self.h,self.rh)
        self.mdot = mdot
        self.mdot0 = mdot0
        self.G1 = G1
        self.delta = delta
        self.ri = ri
        self.ro = ro
        self.Nr = nr

        self.move_planet= move_planet
        self.planet_torque = planet_torque
        self.logarithmic_spacing = logarithmic_spacing
        self.uniform_spacing = uniform_spacing
        self.verbose_output = False

        self.gamma = -k+1.5
        self.mach = 1/h
        self.mth = h**3
        self.mvisc = sqrt( 27*pi/8 * self.alpha*self.mach)
        self.tvisc = self.ro**2/self.nu(self.ro)
        self.min_dt = 1e-2
        self.tol = 1e-6
        self.safety_fac = .95


        self.extrapolate_inner = False
        self.fixed_mdot = False
        self.fixed_lam_inner = True
        self.fixed_lam_outer = True
        self.bc_lam_value = bc_lam

        self.zero_mdot_outer = False

        self.rc,self.dr = self.set_up_grid()
        self.r = self.rc
        self.lam0 = self.initial_cond(self.rc,self.dr)
        self.lam = copy(self.lam0)
        self.lam_final = (2./3)*self.mdot*self.r/self.nu(self.r)

        if planet_torque:
            print 'Mp = %.2e Mth' % self.mp
            print 'Mp = %.2e Mvisc' % (self.mp/self.mvisc)
            print 'Mach # = %.1f' % self.mach

        print 'Viscous time = %.2e' % self.tvisc
        print 'R bounds: %lg\t%lg' % (self.rc[0],self.rc[-1])
    def add_bc(self,bcs,bc_val=(None,None)):
        if type(bcs) == str:
            bcs = [bcs]
        print bcs
        for bc in bcs:
            if bc == 'fixed_lam_inner':
                print 'Changing Inner B.C to fixed lambda = %lg' % bc_val[0]
                self.fixed_lam_inner = True
                self.bc_lam_value[0] = bc_val[0]
                self.extrapolate_inner  = False

            if bc == 'fixed_lam_outer':
                print 'Changing Outer B.C to fixed lambda = %lg' % bc_val[1]
                self.fixed_lam_outer = True
                self.bc_lam_value[1] = bc_val[1]
                self.fixed_mdot = False
                self.zero_mdot_outer = False

            if bc == 'fixed_mdot':
                print 'Changing Outer B.C to fixed Mdot = %e' % self.mdot
                self.fixed_mdot = True
                self.zero_mdot_outer = False
                self.fixed_lam_outer = False
            if bc == 'extrapolate_inner':
                print 'Changing Inner B.C to steady state extrapolation'
                self.extrapolate_inner = True
                self.fixed_lam_inner = False
            if bc == 'zero_mdot_outer':
                print 'Changing Inner B.C to Mdot = 0'
                self.zero_mdot_outer = True
                self.fixed_lam_outer = False
                self.fixed_mdot = False
    def calc_mdot(self,lam,rc,curr_a):
        A, B = self.calc_coeff(rc,curr_a)
        dlam = zeros(lam.shape)
        dlam[1:-1] = (lam[2:] - lam[:-2])/(rc[2:]-rc[:-2])
        dlam[0] =(lam[1]-lam[0])/(rc[1]-rc[0])
        dlam[-1] = (lam[-1] - lam[-2])/(rc[-1]-rc[-2])
        dtr = self.Tfunc(rc,curr_a)/(pi*sqrt(rc))
        fac1 = 3*self.nu(rc)*dlam
        fac2 = dtr*lam
        mdot = A*lam + B*dlam
        return mdot,fac1 #,fac2,lam,dlam
    def calc_vr(self,lam,rc,curr_a):
        return self.calc_mdot(lam,rc,curr_a)/(-lam)

    def Tfunc(self,rc,curr_a):
        xv = (rc-curr_a)/self.dep
        norm = self.delta*sqrt(pi)
        fac1 = (xv-self.beta)/self.delta
        fac2 = (xv+self.beta)/self.delta
        res = (self.G1+1)*exp(-fac1**2) - exp(-fac2**2)
    #    sigp = interp(sigma,planet_a)
        Tnorm = 2*pi * curr_a*self.mp**2 *self.mach*self.mth
        return Tnorm*res/norm * int(self.planet_torque)

    def nu(self,rc):
        return self.alpha* self.h**2 * rc**self.gamma

    def mass_to_lam(self,rc,dr,mass):
        return mass/dr

    def calc_coeff(self,x,curr_a):
        Bc = 3*self.nu(x)
        Ac = Bc*(self.gamma-.5)/x
        if self.planet_torque:
            dtr = self.Tfunc(x,curr_a)
            Ac -= dtr/(pi*sqrt(x))
        return Ac,Bc



    def set_bc(self,rc,dr,curr_a,md,ld,ud,Fmat):
        print 'Entered bc'
        rm1 = rc[0]-dr[0]
        rp1 = (rc[0]+rc[1])/2
        num1 = self.nu(rm1)
        am1,bm1 = self.calc_coeff(rm1,curr_a)
        ap1,bp1 = self.calc_coeff(rp1,curr_a)

        if self.extrapolate_inner:
            c0 = (ap1/2 - bp1/(rc[1]-rc[0]))
            c1 = (ap1/2 + bp1/(rc[1]-rc[0]))

            c0 -= (-1.5*self.nu(rc[0]))/(rc[1]-rc[0])
            c1 -= (1.5*self.nu(rc[1]))/(rc[1]-rc[0])



            md[0] = c0
            ud[0] = c1

    #    md[0] = 0; ud[0] = 0;

#        if self.fixed_lam_inner:
#            lam0 = self.bc_lam_value[0]
#            md[0] = lam0
#            ud[0] = 0



        am1,bm1 = self.calc_coeff(.5*(rc[-1]+rc[-2]),curr_a)

        if self.fixed_mdot:
            Fmat[-1] = self.mdot
            md[-1] = -bm1/((rc[-1]-rc[-2])) -am1/2
            ld[-1] =  bm1/((rc[-1]-rc[-2])) -am1/2

    #        md[-1] = 0; ld[-1] = 0
        if self.zero_mdot_outer:
            md[-1] = 0;
            ld[-1] = 0;

#        if self.fixed_lam_outer:
#            lamN = self.bc_lam_value[-1]
#            delta_r = rc[-1]-rc[-2]
#            Fmat[-1]  = lamN*(am1/2 + bm1/delta_r)
#            md[-1] = 0
#            ld[-1] = (am1/2 - bm1/delta_r)

    def calc_flux(self,rc,dr,curr_a):
        r_face= (rc[:-1]+rc[1:])/2
        A_face,B_face = self.calc_coeff(r_face,curr_a)

        ld = zeros(len(rc)-1)
        md = zeros(len(rc))
        ud = zeros(len(rc)-1)
        Fmat = zeros(rc.shape)

        for i in range(1,len(rc)-1):
            md[i] = .5*(A_face[i]-A_face[i-1])
            md[i] += -(B_face[i]/(rc[i+1]-rc[i])+B_face[i-1]/(rc[i]-rc[i-1]))

            ld[i-1] = -A_face[i-1]/2 + B_face[i-1]/(rc[i]-rc[i-1])
            ud[i] = A_face[i]/2 + B_face[i]/(rc[i+1]-rc[i])




#        rm1 = rc[0]-dr[0]
 #       num1 = self.nu(rm1)
  #      am1,bm1 = self.calc_coeff(rm1,curr_a)
   #     ap1,bp1 = A_face[0],B_face[0]
#
 #       c0 = (ap1/2 - bp1/(rc[1]-rc[0]))
  #      c1 = (ap1/2 + bp1/(rc[1]-rc[0]))
#
 #       c0 -= (-1.5*self.nu(rc[0]))/(rc[1]-rc[0])
  #      c1 -= (1.5*self.nu(rc[1]))/(rc[1]-rc[0])



   #     md[0] = c0
    #    ud[0] = c1

    #    md[0] = 0; ud[0] = 0;


#        self.set_bc(rc,dr,curr_a,md,ld,ud,Fmat)
#        print 'Calling bc'
 #       self.set_bc(rc,dr,curr_a,md,ld,ud,Fmat)
  #      print 'End bc'
#        am1,bm1 = self.calc_coeff(.5*(rc[-1]+rc[-2]),curr_a)

 #       if self.fixed_mdot:
#            Fmat[-1] = self.mdot
 #           md[-1] = -bm1/((rc[-1]-rc[-2])) -am1/2
   #         ld[-1] =  bm1/((rc[-1]-rc[-2])) -am1/2

    #        md[-1] = 0; ld[-1] = 0
   #     else:
    #        md[-1] = 0;
     #       ld[-1] = 0;

 #       Amat = diag(md,0)+diag(ud,1)+diag(ld,-1)

 #       Mmat = diag(dr,0)

        return ld,md,ud,Fmat


    def take_step_crank(self,dt,rc,dr,lam,curr_a=1,Mmat=None,Amat=None,Fmat=None):

 #       if Mmat == None and Amat == None and Fmat == None:
        ld,md,ud,Fmat = self.calc_flux(rc,dr,curr_a)

 #       lhs = Mmat - .5*dt*Amat
  #      rhs = dot(Mmat+.5*dt*Amat,lam) + dt*Fmat

        md *= .5*dt
        ld *= .5*dt
        ud *= .5*dt

        rhs = (dr+md)*lam + dt*Fmat
        rhs[1:] += ld*lam[:-1]
        rhs[:-1] += ud*lam[1:]

        md = dr - md
        ld *= -1
        ud *= -1




        if self.fixed_lam_inner:
#            lhs[0,:] = zeros(lhs[0,:].shape)
            ud[0] = 0
            md[0] = 1.

#            lhs[-1,:] = zeros(lhs[-1,:].shape)
            ld[-1] = 0
            md[-1] = 1.
            rhs[0] = self.bc_lam_value[0]
            rhs[-1] = self.bc_lam_value[-1]

#        md = diag(lhs,0)
#        ud = diag(lhs,1)
#        ld = diag(lhs,-1)

        return thomas.trisolve((ld,md,ud),rhs)


    def take_step_rk4(self,dt,rc,dr,curr_t,curr_lam,curr_a):

        k1 = self.calc_drift_speed(rc,curr_lam,curr_a)

        a2 = curr_a + .5*dt*k1
        lam2 = self.take_step_crank(.5*dt,rc,dr,curr_lam,a2)
        k2 = self.calc_drift_speed(rc,lam2,a2)

        a3 = curr_a + .5*dt*k2
        lam3 = self.take_step_crank(.5*dt,rc,dr,curr_lam,a3)
        k3 = calc_drift_speed(rc,lam3,a3)

        a4 = curr_a + dt*k3
        lam4 = self.take_step_crank(dt,rc,dr,curr_lam,a4)
        k4 = calc_drift_speed(rc,lam4,a4)

        curr_a += (dt/6)*(k1 + 2*(k2+k3) + k4)
        curr_lam = self.take_step_crank(dt,rc,dr,curr_lam,curr_a)

        return curr_lam,curr_a

    def take_step_rk45ck(self,dt,rc,dr,curr_t,curr_lam,curr_a):
        if dt < self.min_dt:
            print 'Time step below minimum dt of %.1e!' % self.min_dt
            cflag=False
            return curr_a, dt, dt,cflag
        k1 = dt*self.calc_drift_speed(rc,curr_lam,curr_a)

        a2 = curr_a + k1/5.
        lam2 = self.take_step_crank(dt/5.,rc,dr,curr_lam,a2)
        k2 = dt*self.calc_drift_speed(rc,lam2,a2)

        a3 = curr_a + k1*3./40 + k2*9./40
        lam3 = self.take_step_crank(dt*3./10,rc,dr,curr_lam,a3)
        k3 = dt*self.calc_drift_speed(rc,lam3,a3)

        a4 = curr_a + k1*3./10  + k2*-9./10 + k3*6./5
        lam4 = self.take_step_crank(dt*3./5,rc,dr,curr_lam,a4)
        k4 = dt*self.calc_drift_speed(rc,lam4,a4)

        a5 = curr_a + k1*-11./54 + k2*5./2 + k3*-70./27 + k4*35./27
        lam5 = self.take_step_crank(dt,rc,dr,curr_lam,a5)
        k5 = dt*self.calc_drift_speed(rc,lam5,a5)


        a6 = curr_a + k1*1631./55296 + k2*175./512 + k3*575./13824 + k4*44275./110592 + k5*253./4096
        lam6 = self.take_step_crank(dt*7./8,rc,dr,curr_lam,a6)
        k6 = dt*self.calc_drift_speed(rc,lam6,a6)

        y5 = curr_a + k1*37./378 + k3*250./621 + k4*125./594 + k6*512./1771
        y4 = curr_a + k1*2825./27648 + k3*18575./48384 + k4*13525./55296 + k5*277./14336 + k6*1./4

        try:
            err = abs(self.tol/(y5-y4))
        except ZeroDivisionError:
            err = 10000/(self.safety_fac**5)
        if err >= 1:
            new_dt = min([2*dt,dt*self.safety_fac * err**.2])
            if self.verbose_output:
                print 'Increasing step at t=%f with dt=%.3e' % (curr_t,new_dt)
            cflag = True
            curr_dt = dt
        else:
            new_dt = max(.5*dt,dt* self.safety_fac *  err**.25)
            cflag = False
            if self.verbose_output:
                print 'Retrying step at t=%f with dt=%.3e' % (curr_t,new_dt)
            y5, curr_dt, new_dt, cflag = self.take_step_rk45ck(new_dt,rc,dr,curr_t,curr_lam,curr_a)


        return y5, curr_dt, new_dt,cflag

    def take_step(self,dt,rc,dr,curr_t,curr_lam,curr_a):
        new_a,curr_dt,new_dt,cflag = self.take_step_rk45ck(dt,rc,dr,curr_t,curr_lam,curr_a)
        if cflag:
            new_lam = self.take_step_crank(curr_dt,rc,dr,curr_lam,new_a)
            curr_t += curr_dt
        else:
            print 'Stepper could not converge!'
            return new_a, curr_lam, curr_t, -1

        if self.verbose_output:
            print 'Current time %f' % curr_t
        return new_a, new_lam, curr_t, new_dt




    def calc_drift_speed(self,rc,curr_lam,curr_a):
        dtr = self.Tfunc(rc,curr_a)
        res = simps(dtr*curr_lam/(rc),x=rc)
        norm = -sqrt(curr_a)/(pi*self.mp*self.mth)
        return norm*res

    def evolve(self,t,dt):
        params = {'alpha':self.alpha,
                    'h':self.h,
                    'mp':self.mp,
                    'gamma':self.gamma}


        res = Status(t,self.rc,**params)

        res.a[0] = self.planet_a
        res.vs[0] = 0
        res.dlam[:,0] = zeros(self.lam.shape)
        res.lam[:,0] = self.lam
        res.sigma[:,0] = self.lam/(2*pi*self.rc)
        res.mdot[:,0],junk = self.calc_mdot(self.lam,self.rc,self.planet_a)
        res.vr[:,0] = res.mdot[:,0]/(-self.lam)
        curr_t = t[0]
        rc = copy(self.rc)
        dr = copy(self.dr)


#        if not self.move_planet:
#            Mmat1,Amat1,Fmat1 = self.calc_flux(rc,dr,self.planet_a)
#        else:
#            Mmat1,Amat1,Fmat1 = None,None,None
        breakflag = False

        for i,ti in enumerate(t[1:],start=1):
            print '\tEvolving to %.2f' % ti
            while curr_t < ti and not breakflag:
                # if grad_bc:
                #     f0 = (curr_f[2]-curr_f[1])/(x[2]-x[1]) * x[0]
                # else:
                #     f0 = (2./3)*mdot*x[0]
                try:
                    if self.move_planet:
                        self.planet_a,self.lam,curr_t,dt = self.take_step(dt,rc,dr,curr_t,self.lam,self.planet_a)
                        if dt == -1:
                            print 'Exiting at time %f' % curr_t
                            breakflag = True
                        if self.verbose_output:
                            print dt
                        if self.planet_a < self.ri:
                            self.move_planet = False
                            self.planet_torque = False
                            print 'Planet left inner edge!'
                        if self.planet_a > self.ro:
                            self.move_planet=False
                            self.planet_torque = False
                            print 'Planet left outer edge'
        #                lam_interp = interp1d(rc,lam)
        #                rc,dr,_,_,_=set_up_grid(ri,ro,nr,planet_a)
        #                lam = lam_interp(rc)
                    else:
                        self.lam = self.take_step_crank(dt,rc,dr,self.lam,self.planet_a)
                        curr_t += dt
                except KeyboardInterrupt:
                    print '\n\nCaught KeyboardInterrupt!\n\n'
                    return res
                # if any(lam<0):
                #     print 'Surface density has fallen below zero!'
                #     breakflag = True


            res.lam[:,i] = self.lam
            res.dlam[:,i] = (self.lam-res.lam[:,0])/res.lam[:,0]
            res.sigma[:,i] = self.lam/(2*pi*self.rc)
            res.mdot[:,i],junk = self.calc_mdot(self.lam,self.rc,self.planet_a)
            res.vr[:,i] = res.mdot[:,i]/(-self.lam)
            if self.move_planet:
                res.a[i] = self.planet_a
                res.vs[i] = self.calc_drift_speed(rc,self.lam,self.planet_a)
            if breakflag:
                return res

        return res

    def set_up_grid(self):
        if self.uniform_spacing:
            if self.logarithmic_spacing:
                rc = exp(linspace(log(self.ri),log(self.ro),self.Nr))
                dr  = diff(log(rc))[0]*rc
            else:
                rc = linspace(self.ri,self.ro,self.Nr)
                dr = diff(rc)[0]*ones(rc.shape)
        else:
            if self.planet_torque:
                nsmall = self.Nr/2

                dr_small = 2*(self.beta+2*self.delta)/(nsmall/2-1)


                xp = dr_small/2 + arange(nsmall/2)*dr_small
                xp = hstack((-xp[::-1],xp))

                xp  = self.dep*xp+self.planet_a

                if self.logarithmic_spacing:
                    dr_large_r = (log(self.ro)-log(xp[-1]))/(self.Nr/4-1)
                    dr_large_l = (log(xp[0])-log(self.ri))/(self.Nr/4-1)

                    xr = log(xp[-1]) + arange(1,nsmall/2)*dr_large_r
                    xl = log(xp[0]) - arange(1,nsmall/2)*dr_large_l
                    xr = exp(xr)
                    xl = exp(xl)
                else:
                    dr_large_r = (self.ro-xp[-1])/(self.Nr/4-1)
                    dr_large_l = (xp[0]-self.ri)/(self.Nr/4-1)

                    xr = xp[-1] + arange(1,nsmall/2)*dr_large_r
                    xl = xp[0] - arange(1,nsmall/2)*dr_large_l

                rc=hstack((xl[::-1],xp,xr))
                dr = zeros(rc.shape)
                dr[1:-1] = (rc[2:]-rc[:-2])*.5
                dr[0] = rc[1]-rc[0]
                dr[-1] = rc[-1]-rc[-2]
            else:
        #    dr= diff(rc)
        #    dr = hstack((dr[0],dr))
                if self.logarithmic_spacing:
                    dlr = (log(self.ro/self.ri))/self.Nr
                    lrc = log(self.ri) + arange(self.Nr)*dlr
                    rc = exp(lrc)
                    dr = dlr*rc

                else:
                    rc = linspace(self.ri,self.ro,self.Nr)
                    dr = diff(rc)[0]*ones(rc.shape)
                xl=zeros(rc.shape)
                xr=zeros(rc.shape)
                xp = zeros(rc.shape)
        return rc,dr #,xl,xr,xp

    def input_init_cond(self,rc,x,y):
        return interp1d(x,y)(rc)

    def initial_cond(self,rc,dr):
            #    lami = (2./3)*self.mdot0*rc/self.nu(rc)
#        lami = 2*pi*rc*exp(-(rc-1.8)**2/.01)
        if self.fixed_lam_outer:
#            a = log(self.bc_lam_value[0]/self.bc_lam_value[1]) / log(rc[0]/rc[-1])
 #           lami =  self.bc_lam_value[1] * pow(rc/rc[-1],a)
            sig0 = (1-sqrt(rc[0]/rc[-1]))*sqrt(rc[-1])
            lami = sqrt(rc)*(1-sqrt(rc[0]/rc))/sig0
            fac = (rc-rc[0])/(rc[-1]-rc[0])
            sig0 = sin(fac[-1]*pi/2)
            lami = self.bc_lam_value[0] + (self.bc_lam_value[1] -self.bc_lam_value[0])* sin(fac**4 *pi/2)
    #    lami = lami/10 *  (rc/rc[-1])**10
    #    lami = exp(-(rc-1)**2/(2*.2**2))
        return lami

def calc_mdot(lam,rc,curr_a):
    A, B = calc_coeff(rc,curr_a)
    dlam = zeros(lam.shape)
    dlam[1:-1] = (lam[1:] - lam[:-1])/(rc[1:]-rc[:-1])
    dlam[0] =(lam[1]-lam[0])/(rc[1]-rc[0])
    dlam[-1] = (lam[-1] - lam[-2])/(rc[1]-rc[0])

    mdot = A*lam + B*dlam
    return mdot
def calc_vr(lam,rc,curr_a):
    return calc_mdot(lam,rc,curr_a)/(-lam)


def test_suite(ro_list,tend=1e4,steps=10,alpha=.06,nr=1024,res0=None,sim0=None,logarithmic_spacing=True):
    if res0 == None:
        sim0 = Simulation(ro=ro_list[-1],nr=nr,alpha=alpha,planet_torque=False,logarithmic_spacing=logarithmic_spacing)
        times = linspace(0,tend,200)
        res0 = sim0.evolve(times,steps)

    lam0 = res0.lam[:,-1]
    rc0 = res0.r

    times = linspace(0,tend,200)

    res_list = [res0]
    sim_list = [sim0]
    args = [(r,rc0,lam0,nr,alpha,times,steps,logarithmic_spacing) for r in ro_list]
    for i in args:
        s,r = run_ro_one(i)
        sim_list.append(s)
        res_list.append(r)

    return sim_list,res_list

def test_suite_a(a_list,tend=1e4,steps=10,ro=30,nr=1024,res0=None,sim0=None,logarithmic_spacing=True):
    if res0 == None:
        sim0 = Simulation(ro=ro,nr=nr,alpha=a_list[0],planet_torque=False,logarithmic_spacing=logarithmic_spacing)
        times = linspace(0,tend,200)
        res0 = sim0.evolve(times,steps)

    lam0 = res0.lam[:,-1]
    rc0 = res0.r

    times = linspace(0,tend,200)

    res_list = [res0]
    sim_list = [sim0]
    args = [(ro,rc0,lam0,nr,a,times,steps,logarithmic_spacing) for a in a_list]
    for i in args:
        s,r = run_ro_one(i)
        sim_list.append(s)
        res_list.append(r)

    return sim_list,res_list


def run_ro_one((r,rc0,lam0,nr,alpha,times,steps,log_spacing)):
    temp = Simulation(ro=r,nr=nr,alpha=alpha,planet_torque=True,logarithmic_spacing=log_spacing)
    temp.lam = temp.input_init_cond(temp.rc,rc0,lam0)
    temp.add_bc('fixed_lam_outer',(temp.lam[0],temp.lam[-1]))
    res  = temp.evolve(times,steps)
    return temp,res


def plot_summary(sim_list,res_list,logy=True,logx=True):
    fig,axes = subplots(2,2,figsize=(15,10))

###################################
###################################
##   sig,r     ##   delsig,h     ##
##   (0,0)     ##   (0,1)        ##
##             ##                ##
###################################
##             ##                ##
##   mdot,r    ##   delmdot,h    ##
##   (1,0)     ##     (1,1)      ##
###################################
###################################

    x = (res_list[0].r-sim_list[0].planet_a)/sim_list[0].dep
    axes[0,0].plot(res_list[0].r,res_list[0].lam[:,-1],color='grey')
    axes[0,1].axhline(0,color='grey')
    axes[1,0].plot(res_list[0].r,res_list[0].mdot[:,-1],color='grey')
    axes[1,1].axhline(0,color='grey')

    if logx:
        axes[0,0].set_xscale('log')
        axes[1,0].set_xscale('log')
    if logy:
        axes[0,0].set_yscale('log')
    #axes[0,0].set_title('Thermal mass, alpha = 0.06, h=0.1, different r_outer')
   # axes[1,0].set_yscale('log')
    ilam0 = interp1d(res_list[0].r,res_list[0].lam[:,-1])
    imdot0 = interp1d(res_list[0].r,res_list[0].mdot[:,-1])

    for s,r in zip(sim_list[1:],res_list[1:]):
        lam0 = ilam0(r.r)
        mdot0 = imdot0(r.r)
        x = (r.r - s.planet_a)/s.dep
        axes[0,0].plot(r.r,r.lam[:,-1])
        axes[0,1].plot(x,(r.lam[:,-1]-lam0)/lam0)
        axes[1,0].plot(r.r,r.mdot[:,-1])
        axes[1,1].plot(x,(r.mdot[:,-1]-mdot0)/mdot0)

    axes[0,1].set_xlim(-5,5)
    axes[1,1].set_xlim(-5,5)
#    axes[1,0].set_xlim(8,12)
    axes[1,1].set_ylim(-2,1)
#    axes[1,0].set_ylim(1e-4,4e-4)
#    axes[0,0].set_ylim(1e-1,1)
    axes[1,0].set_xlabel('$r$',fontsize=20)
    axes[1,1].set_xlabel('$\\frac{r-r_p}{max(h,r_h)}$',fontsize=20)
    axes[0,0].set_ylabel('$\\lambda$',fontsize=20)
    axes[0,1].set_ylabel('$\\frac{\\lambda-\\lambda_0}{\\lambda_0}$',fontsize=20)
    axes[1,0].set_ylabel('$\\dot{M}$',fontsize=20)
    axes[1,1].set_ylabel('$\\frac{\\dot{M}-\\dot{M}_0}{\\dot{M}_0}$',fontsize=20)

    for ax in axes.flatten():
        ax.minorticks_on()

