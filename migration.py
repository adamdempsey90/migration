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


class Field():
    def __init__(self,r,k,l,hor,alpha,a,mp,mud):
        self.r = r
        self.omk = r**(-1.5)
        self.omk2 = self.omk**2
        self.hor = hor
        self.mud = mud
        self.c2 = hor**2 * r**(-k)
        self.sigma = (mud/(a*a*pi))*r**(-l)
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
        l = self.r*self.grad(self.sigma)/self.sigma
        self.omegap2 = -(self.k+l)*self.c2/self.r**2
        self.omega2 =  self.omk + self.omegap2
        self.omega = sqrt(self.omega2)

    def set_all(self,sigma):
        self.sigma = sigma
        self.set_omega()
        self.set_kappa2()
        self.set_m()

class Simulation():
    intdict={'quad':integrate.quad,
            'dblquad':integrate.dblquad,
            'tplquad':integrate.tplquad,
            'fixed_quad':integrate.fixed_quad,
            'quadrature':integrate.quadrature,
            'romberg':integrate.romberg}



    def __init__(self,x=linspace(-4,4,200),k=1,l=0,hor=.07,alpha=1e-3,a=1,mp=3e-6,mud=1,integrator='quad',solver='rk4'):
        self.fld = Field(x*hor + a ,k,l,hor,alpha,a,mp,mud)
        self.integrator = integrator
        self.solver = solver
        self.set_integrator(integrator)
        self.set_solver(solver)
        self.dt = 1
        self.set_vr(self.fld.sigma)

    def set_integrator(self,integrator):
        self.integrate = self.intdict[integrator]

    def set_solver(self,solver):
        solvedict={'rk2':self.take_step_rk2,
                'rk4':self.take_step_rk4,
                'crank':self.take_step_crank_nicholson,
                'feuler':self.take_step_forward_euler,
                'beuler':self.take_step_backward_euler}
        self.take_step =solvedict[solver]

    def laplace(self):
        x = self.fld.r/self.fld.a

        b = zeros(x.shape)
        db = zeros(x.shape)
        psi = zeros(x.shape)

        for i,(a,m) in enumerate(zip(x,self.fld.m)):
            lfunc = lambda x: cos(m*x)/sqrt(1-2*a*cos(x)+a*a)
            dlfunc = lambda x: cos(m*x)*(cos(x)-a) * (1-2*a*cos(x)+a*a)**(-1.5)
            ans = 2*self.integrate(lfunc,0,pi)
            if type(ans) == tuple or type(ans) == list or type(ans) == ndarray:
                ans = ans[0]
            b[i] = ans
            ans = 2*self.integrate(dlfunc,0,pi)
            if type(ans) == tuple or type(ans) == list or type(ans) == ndarray:
                ans = ans[0]
            db[i] = ans
            if m > 0:
                psi[i] = .5*pi*(abs(db[i])/m + 2*sqrt(1+self.fld.xi[i]**2)*b[i])
            else:
                psi[i] = 0

        self.lap = b
        self.dlap = db
        self.psi = psi


    def set_torques(self):
        self.laplace()

        Tnorm = self.fld.mp**2 * self.fld.mud * (self.fld.a*self.fld.oms)**2 / (self.fld.hor**3)

        val = Tnorm * (2*self.fld.sigma*self.fld.a**2/self.fld.mud)*self.fld.hor**3 * self.fld.m**4 * self.psi**2
        val *= (self.fld.oms**2/self.fld.kappa2)
        val /= (self.fld.r*(1 + 4*self.fld.xi**2))

        self.dTr = array([ v if r>=self.fld.a else -v for r,v in zip(self.fld.r,val)])

        g = 3*pi*self.fld.nu*self.fld.r**2 * self.fld.omk * self.fld.sigma

        self.dGr = -self.fld.grad(g)

    def set_vr(self,sigma):
        self.fld.set_all(sigma)
        self.set_torques()
        self.vr = (self.dGr + self.dTr)/(2*pi*self.fld.r**2 * self.fld.sigma*self.fld.omk)

    def set_dt(self,dt):
        self.dt = dt

    def calc_rhs(self,sigma):
        self.set_vr(sigma)
        rhs = -self.fld.grad(self.fld.r*self.vr*self.fld.sigma)/self.fld.r
        return rhs

    def move_planet(self):
        norm = -2/(self.fld.mp*self.fld.a*self.fld.oms)
        self.fld.vs = trapz(self.dTr,x=self.fld.r)*norm
        self.fld.a += self.dt * self.fld.vs

    def take_step_forward_euler(self,dt=None):
        if dt != None:
            self.set_dt(dt)
        self.fld.sigma += self.dt*self.calc_rhs(self.fld.sigma)
        self.move_planet()

    def take_step_backward_euler(self,dt=None):
        if dt != None:
            self.set_dt(dt)

        sig0 = self.fld.sigma
        f0 = self.fld.sigma + self.dt*self.calc_rhs(self.fld.sigma)

        self.fld.sigma = fsolve( lambda x: x - sig0 - dt*self.calc_rhs(x),f0)
        self.move_planet()

    def take_step_rk2(self,dt=None):
        if dt != None:
            self.set_dt(dt)

        sig0 = self.fld.sigma
        k1 = self.calc_rhs(sig0)
        k2 = self.calc_rhs(sig0 + .5*self.dt*k1)
        self.fld.sigma = sig0 + k2
        self.move_planet()

    def take_step_rk4(self,dt=None):
        if dt != None:
            self.set_dt(dt)
        sig0 = self.fld.sigma
        k1 = self.calc_rhs(sig0)
        k2 = self.calc_rhs(sig0 + .5*self.dt*k1)
        k3 = self.calc_rhs(sig0 + .5*self.dt*k2)
        k4 = self.calc_rhs(sig0 + self.dt*k3)

        self.fld.sigma = sig0 + (self.dt/6)*(k1+2*(k2+k3)+k4)
        self.move_planet()

    def take_step_crank_nicholson(self,dt=None):
        if dt != None:
            self.set_dt(dt)

        f0 = self.fld.sigma + .5*self.dt*self.calc_rhs(self.fld.sigma)

        self.fld.sigma = fsolve( lambda x: x- (f0 + .5*dt*self.calc_rhs(x)),f0)
        self.move_planet()

    def evolve(self,times,dt):
        self.dt = dt
        self.max_steps = 1e3
        nt = len(times)
        nr = len(self.fld.r)
        status = Status(times,self.fld.r,nr,nt)

        status.sigma[:,0] = self.fld.sigma
        status.a[0] = self.fld.a
        status.vs[0] = self.fld.vs
        status.vr[:,0] = zeros(self.fld.sigma.shape)
        status.dTr[:,0] = self.dTr

        current_t = times[0]
        step_count = 0
        end_flag=False
        for i,end_t in enumerate(times[1:],start=1):
            while current_t < end_t and step_count < self.max_steps:
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

            print 'Saving at t=%f' % current_t

            status.sigma[:,i] = self.fld.sigma
            status.a[i] = self.fld.a
            status.vs[i] = self.fld.vs
            status.vr[:,i] = self.vr
            status.dTr[:,0] = self.dTr

            if end_flag:
                return status

        print 'Finished!'
        return status
