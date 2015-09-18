from numpy import *
from scipy.optimize import fsolve
from scipy import integrate
from matplotlib.pyplot import *


class Status():
    def __init__(self,t,r,nr,nt):
        self.sigma = zeros((nr,nt))
        self.vr = zeros(self.sigma.shape)
        self.dTr = zeros(self.sigma.shape)
        self.a = zeros(t.shape)
        self.vs = zeros(t.shape)
        self.t = t
        self.r = r

    def animate(self,q='sigma',tend=None):
        qvals={'sigma':self.sigma,
                'vr':self.vr,
                'dTr':self.dTr}

        if tend == None:
            tend = self.t[-1]
        if q not in qvals.keys():
            q = 'sigma'
        dat = qvals[q]

        fig=figure();
        ax=fig.add_subplot(111)
        line,=ax.plot(self.r,dat[:,0])

        if q == 'sigma':
            ax.set_ylim(dat[dat>0].min(),dat[dat>0].max())
        else:
            ax.set_ylim(dat.min(),dat.max())

        for i,t in enumerate(self.t[self.t<=tend]):
            line.set_ydata(dat[:,i])
            fig.canvas.draw()


class Field():
    def __init__(self,r,k,l,hor,alpha,a,mp,mud):
        self.r = r
        self.dr = diff(r)[0]
        self.omk = r**(-1.5)
        self.omk2 = self.omk**2
        self.hor = hor
        self.mud = mud
        self.c2 = hor**2 * r**(-k)
        self.sigma = (mud/(a*a*pi))*r**(-l)
        self.sig0 = copy(self.sigma)
        self.mdisk = trapz(self.sigma*self.r,x=self.r)*2*pi
        self.k = k
        self.l = l
        self.omegap2 = -(k+l)*self.c2/r**2
        self.omega2 = self.omegap2 + self.omk2
        self.omega = sqrt(self.omega2)
        self.nu = alpha*self.c2/self.omk
        self.dr = diff(self.r)[0]
        self.a = a
        self.mp = mp
        self.vs = 0
        self.oms = a**(-1.5)
        self.set_kappa2()
        self.set_m()


    def grad(self,q):
         dq = zeros(q.shape)
         dq[1:-1] = .5*(q[2:]-q[:-2])
         dq[0] = -1.5*q[0] + 2*q[1] - .5*q[2]
         dq[-1] = 1.5*q[-1] - 2*q[-2] +.5*q[-3]
         return dq/self.dr
    def grad2(self,q):
         d2q = zeros(q.shape)
         d2q[1:-1] = q[2:] - 2*q[1:-1] + q[:-2]
         d2q[0] = 2*q[0] -5*q[1] +4*q[2]-q[3]
         d2q[-1] = 2*q[-1] - 5*q[-2] + 4*q[-3] - q[-4]
         return d2q/self.dr/self.dr

    def set_m(self):
         D = (self.omega - self.oms)**2 - self.c2/self.r**2
         self.m = array([sqrt(kap/d) if d>0 and kap>0 else 0 for d,kap in zip(D,self.kappa2)])
         self.xi = self.m * sqrt(self.c2)/(self.r*sqrt(self.kappa2))

    def set_kappa2(self):
        self.kappa2 = self.grad(self.r**4 * self.omega2)/self.r**3

    def set_omega(self):
        l = -self.r*self.grad(self.sigma)/self.sigma
        self.omegap2 = -(self.k+l)*self.c2/self.r**2
        self.omega2 =  self.omk2 + self.omegap2
        self.omega = sqrt(self.omega2)

    def set_all(self,sigma):
        self.sigma = sigma
        self.set_omega()
        self.set_kappa2()
        self.set_m()

class Simulation():
    intdict={'quad':integrate.quad,
            'fixed_quad':integrate.fixed_quad,
            'quadrature':integrate.quadrature,
            'romberg':integrate.romberg}



    def __init__(self,x=linspace(-4,4,200),k=1,l=0,hor=.07,alpha=1e-3,a=1,mp=3e-6,mud=1,integrator='quad',solver='rk4',softening=None,steady_state=False,calc_lr=True,fixed_torque=False,smoothed=True,nsmooth=20):
        self.fld = Field(x*hor + a ,k,l,hor,alpha,a,mp,mud)
        self.calc_lr = calc_lr
        self.integrator = integrator
        self.solver = solver
        self.softening = softening
        self.set_integrator(integrator)
        self.set_solver(solver)
        self.fixed_torque = fixed_torque
        self.smoothed = smoothed
        self.nsmooth = nsmooth
        self.dt = 1
        self.fld.set_all(self.fld.sigma)
        self.set_torques()
        self.set_vs()
        self.set_vr()


        if steady_state:
            self.steadystate=True
            self.solver = 'steady_state'
            self.set_solver(self.solver)
        else:
            self.steadystate = False


    def set_integrator(self,integrator):
        self.integrate = self.intdict[integrator]

    def set_solver(self,solver):
        solvedict={'rk2':self.take_step_rk2,
                'rk4':self.take_step_rk4,
                'rk45':self.take_step_rk45,
                'crank':self.take_step_crank_nicholson,
                'feuler':self.take_step_forward_euler,
                'beuler':self.take_step_backward_euler,
                'steady_state':self.take_step_inf}

        self.take_step = solvedict[solver]

    def laplace(self):
        x = self.fld.r/self.fld.a

        b = zeros(x.shape)
        db = zeros(x.shape)
        mpsi = zeros(x.shape)

        if self.softening != None:
            rs = self.softening
        else:
            rs = 0

        for i,(a,m) in enumerate(zip(x,self.fld.m)):
            lfunc = lambda x: cos(m*x)/sqrt(1-2*a*cos(x)+a*a+rs*rs)
            dlfunc = lambda x: cos(m*x)*(cos(x)-a) * (1-2*a*cos(x)+a*a+rs*rs)**(-1.5)
            ans = self.integrate(lfunc,0,pi)
            if type(ans) == tuple or type(ans) == list or type(ans) == ndarray:
                ans = ans[0]*2/pi
            b[i] = ans*2/pi
            ans = self.integrate(dlfunc,0,pi)
            if type(ans) == tuple or type(ans) == list or type(ans) == ndarray:
                ans = ans[0]*2/pi
            db[i] = ans*2/pi
            mpsi[i] = .5*pi*(abs(db[i]) + 2*sqrt(1+self.fld.xi[i]**2)*m*b[i])


        self.lap = b
        self.dlap = db
        self.mpsi = mpsi


    def set_torques(self):

        if self.calc_lr:
            self.laplace()

            eps = array([-1 if r<self.fld.a else 1 for r in self.fld.r])

            self.Tnorm = self.fld.mp**2 * self.fld.mud * (self.fld.a*self.fld.oms)**2 / (self.fld.hor**3)

            val = self.Tnorm * (2*self.fld.a**2/self.fld.mud)*self.fld.hor**3 * self.fld.m**2 * self.mpsi**2
            val *= eps*(self.fld.oms**2/self.fld.kappa2)
            val /= (self.fld.r*(1 + 4*self.fld.xi**2))


            if self.smoothed:
                self.dTr = convolve(val,ones(self.nsmooth)/self.nsmooth)[self.nsmooth/2-1:-self.nsmooth/2]
            else:
                self.dTr = val
        else:
            self.dTr = zeros(self.fld.r.shape)
            self.Tnorm = ones(self.fld.r.shape)



    def set_vr(self):
        g = 3*pi*self.fld.nu*self.fld.r**2 * self.fld.omk * self.fld.sigma
        self.dGr = -self.fld.grad(g)
        self.vr = (self.dGr + self.fld.sigma*self.dTr)/(2*pi*self.fld.r**2 * self.fld.sigma*self.fld.omk)

    def set_dt(self,dt):
        self.dt = dt

    def calc_rhs(self,sigma):
        if not self.fixed_torque:
            self.fld.set_all(sigma)
            self.set_torques()
            self.set_vs()
        else:
            self.fld.sigma = sigma
        self.set_vr()

        ds = self.fld.grad(sigma)
        rhs = ds*self.fld.vs-self.fld.grad(self.fld.r*self.vr*self.fld.sigma)/self.fld.r

        return rhs

    def set_bc(self,sigma):
        # Zero gradient
        #sigma[0] =  (2*sigma[1] - .5*sigma[2])/1.5

        # Const total Mdisk
        # res = sigma*self.fld.r*2*pi
        # sigma[-1] = 2*self.fld.mdisk/self.fld.dr - res[0]-2*sum(res[1:-1])
        # sigma[-1] /= (2*pi*self.fld.r[-1])
#        sigma[0] = self.fld.sig0[0]
#        sigma[-1] = self.fld.sig0[-1]
        return sigma

    def set_vs(self):
        norm = -2/(self.fld.mp*self.fld.a*self.fld.oms)
        self.fld.vs = integrate.simps(self.fld.sigma*self.dTr,x=self.fld.r)*norm
    def move_planet(self):
        self.calc_vs()
        self.fld.a += self.dt * self.fld.vs

    def take_step_forward_euler(self,dt=None):
        if dt != None:
            self.set_dt(dt)

        self.fld.sigma += self.dt*self.calc_rhs(self.fld.sigma)
        self.fld.sigma = self.set_bc(self.fld.sigma)
#        self.move_planet()

    def take_step_backward_euler(self,dt=None):
        if dt != None:
            self.set_dt(dt)

        sig0 = self.fld.sigma
        f0 = self.fld.sigma + self.dt*self.calc_rhs(self.fld.sigma)

        self.fld.sigma = fsolve( lambda x: x - sig0 - dt*self.calc_rhs(x),f0)
        self.fld.sigma = self.set_bc(self.fld.sigma)
#        self.move_planet()

    def take_step_rk2(self,dt=None):
        if dt != None:
            self.set_dt(dt)

        sig0 = self.fld.sigma
        k1 = self.calc_rhs(sig0)
        k2 = self.calc_rhs(sig0 + .5*self.dt*k1)
        self.fld.sigma = sig0 + k2
        self.fld.sigma = self.set_bc(self.fld.sigma)
#        self.move_planet()

    def take_step_rk4(self,dt=None):
        if dt != None:
            self.set_dt(dt)
        sig0 = self.fld.sigma
        k1 = self.calc_rhs(sig0)
        k2 = self.calc_rhs(sig0 + .5*self.dt*k1)
        k3 = self.calc_rhs(sig0 + .5*self.dt*k2)
        k4 = self.calc_rhs(sig0 + self.dt*k3)

        self.fld.sigma = sig0 + (self.dt/6)*(k1+2*(k2+k3)+k4)
        self.fld.sigma = self.set_bc(self.fld.sigma)
#        self.move_planet()

    def take_step_rk45(self,dt=None):
        tol = 1e-7
        sig0 = self.fld.sigma
        k1 = self.calc_rhs(sig0)
        k2 = self.calc_rhs(sig0 + .25*self.dt*k1)
        k3 = self.calc_rhs(sig0 + 3./32*self.dt*k1 + 9./32*self.dt*k2)
        k4 = self.calc_rhs(sig0 +  1932./2197*self.dt*k1 - 7200./2197*self.dt*k2 +7296./2197* self.dt*k3)
        k5 = self.calc_rhs(sig0 +  439./216*self.dt*k1 - 8*self.dt*k2 + 3680./513* self.dt*k3 - 845/4104* self.dt*k4)
        k6 = self.calc_rhs(sig0 - 8./27*self.dt*k1 + 2*self.dt*k2 - 3544./2565* self.dt*k3 + 1859./4104* self.dt*k4 - 11./40* self.dt*k5)

        y4 = sig0 + self.dt*(25./216 * k1 + 1408./2565*k3 + 2197./4104*k4 - .2*k5)
        y5 = sig0 + self.dt*(16./135*k1 + 6656./12825*k3 + 28561./56430*k4-9./50*k5 + 2./55*k6)

        err = abs(y4-y5).max()

        if err > tol:
            self.dt /= 2
            print 'Decreasing step size to %f' % self.dt
            yf = self.take_step_rk45()
        else:
            yf = y5

        yf = self.set_bc(yf)
        return yf

    def take_step_crank_nicholson(self,dt=None):
        if dt != None:
            self.set_dt(dt)

        f0 = self.fld.sigma + .5*self.dt*self.calc_rhs(self.fld.sigma)

        self.fld.sigma = fsolve( lambda x: x- (f0 + .5*self.dt*self.calc_rhs(x)),f0)
        self.fld.sigma = self.set_bc(self.fld.sigma)
#        self.move_planet()

    def take_step_inf(self):
            f0 = self.fld.sigma
            self.fld.sigma = fsolve(lambda x: self.calc_rhs(x),f0)

    def evolve(self,times,dt):
        self.dt = dt
        self.max_steps = 1e3
        nt = len(times)
        nr = len(self.fld.r)
        status = Status(times,self.fld.r,nr,nt)

        status.sigma[:,0] = self.fld.sigma
        status.a[0] = self.fld.a
        status.vs[0] = self.fld.vs
        status.vr[:,0] = self.vr
        status.dTr[:,0] = self.dTr

        current_t = times[0]
        step_count = 0
        end_flag=False
        for i,end_t in enumerate(times[1:],start=1):
            while current_t < end_t and step_count < self.max_steps and end_flag==False:
                if current_t + self.dt > end_t:
                    self.dt = end_t - current_t
                    self.take_step()
                    self.dt = dt
                    current_t = end_t
                else:
                    self.take_step()
                    current_t += self.dt

                step_count += 1
                if step_count > self.max_steps:
                    'Print Exceeded number of substeps'
                    end_flag = True
                if nan in self.fld.sigma or self.fld.vs==nan:
                    'Print NaN detected, exiting.'
                    end_flag = True

            print 'Saving at t=%f' % current_t

            status.sigma[:,i] = self.fld.sigma
            status.a[i] = self.fld.a
            status.vs[i] = self.fld.vs
            status.vr[:,i] = self.vr
            status.dTr[:,i] = self.dTr

            if end_flag:
                return status

        print 'Finished!'
        return status

def linblad(r,k,l,hor,m_max):
    mvals = arange(m_max+1)
    dr = diff(r)[0]
    omk = r**(-1.5)
    sigma=  r**(-k)
    c2 = r**(-l)*hor**2
    omega2 = omk*omk - (k+l)*c2/(r*r)
    omega = sqrt(omega2)
    kappa2 = gradient(r**4 *omega2,dr)/r**3

    D = lambda x: kappa2 - x*x*(omega - 1)**2
    Ds = lambda x: D(x) + x*x*c2/(r*r)

    Dfunc = [Ds(m) for m in mvals]

    lr = [r[sign(d[1:])-sign(d[:-1]) != 0] for d in Dfunc]

    ilr = zeros(mvals.shape)
    olr = zeros(mvals.shape)

    for i,l in enumerate(lr):
        if len(l)==0:
            ilr[i] = 0
            olr[i] = 0
        elif len(l) == 1:
            if l < 1:
                ilr[i] = l
                olr[i] = 0
            else:
                olr[i] = l
                ilr[i] = 0
        else:
            ilr[i] = l[0]
            olr[i] = l[1]


    clr = zeros(r.shape)
    for i in range(len(r)):
        denom = (omega[i] -1)**2 - c2[i]/(r[i]*r[i])
        if denom <= 0:
            clr[i] = 0
        else:
            clr[i] = sqrt(kappa2[i]/denom)


    return mvals,ilr,olr,clr
