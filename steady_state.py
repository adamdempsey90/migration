from scipy.integrate import simps, ode
from matplotlib.pyplot import *
from numpy import *
class SteadyState():
    def __init__(self,x=linspace(.2,1.4,500),alpha = 1e-1,k=1,h = .1,delta = .1, mp = 1, a = 1, mdot = 1e-6,beta=1,G1=0):
        self.x = x
        self.dx = diff(x)[0]
        self.alpha = alpha
        self.k = k
        self.h = h
        self.delta = delta
        self.mp = mp
        self.planet_a = a
        self.mdot = mdot
        self.beta = beta
        self.G1 = G1
        self.mth = h**3
        self.mach = 1/h
        self.mvisc = sqrt(10.24*self.alpha*self.mach)

        self.Tnorm = 2*pi * self.planet_a* self.mp**2 *self.mth/ self.h

    def state(self):
        print 'Simulation state:'
        print '\tMp = %.1e Mth = %.1e Mvisc' % (self.mp,self.mp/self.mvisc)
        print '\talpha = %.1e' % self.alpha
        print '\th = %.2f' % self.h
        print '\tMach = %.1f' % self.mach
        print '\tMth = %.2e' % self.mth
        print '\tMvisc = %.2e' %  (self.mvisc*self.mth,)
        print '\tTemp index = -%.1f' % self.k
        print '\ta = %.3f' % self.planet_a
        print '\tMdot = %.1e' % self.mdot
        print '\tGrid parameters:'
        print '\t\t inner-x = %.1e' % self.x[0]
        print '\t\t outer-x = %.1e ' % self.x[-1]
        print '\t\t dx = %.2e' % self.dx
        print '\tTorque parameters:'
        print '\t\tWidth = %.2f' % self.delta
        print '\t\tDeposition at %.2f scale heights' % self.beta
        print '\t\tTnorm = %.2e' % self.Tnorm
        print '\t\tAsymmetry = %.3f' % self.G1

    def nu(self,x):
        return self.alpha*self.h**2 * x

    def Tfunc(self,x):
        r = x**2
        xv = (r-self.planet_a)/self.h
        norm = self.delta*sqrt(pi)
        fac1 = (xv-self.beta)/self.delta
        fac2 = (xv+self.beta)/self.delta
        res = (self.G1+1)*exp(-fac1**2) - exp(-fac2**2)

        return self.Tnorm*res/norm

    def dTfunc(self,x):
        r = x**2
        xv = (r-self.planet_a)/self.h
        norm = self.delta**2 * sqrt(pi)
        fac1 = (xv-beta)/self.delta
        fac2 = (xv+beta)/self.delta
        res = fac2*exp(-fac2**2) -(self.G1+1)*fac1*exp(-fac1**2)
    #    sigp = interp(sigma,planet_a)
        return self.Tnorm*2*res/norm

    def rhs(self,x,f):
        return (2./3)*self.mdot + (2./3)*f*self.Tfunc(x)/(pi*self.nu(x))
    def jac(self,x,f):
        return (2./3) * self.Tfunc(x)/(pi*self.nu(x))

    def solve(self,dx):
        fN = (2./3)*self.mdot*self.x[-1]
        od = ode(self.rhs,self.jac).set_integrator('dopri5')
        #set_integrator('vode', method='bdf', with_jacobian=True)
        od.set_initial_value(fN,self.x[-1])
        dt = -dx
        self.xf=[self.x[-1]]
        self.yf=[fN]

        while od.successful() and od.t > self.x[0]:
            od.integrate(od.t+dt)
            self.xf.append(od.t)
            self.yf.append(od.y[0])

        self.xf = array(self.xf[::-1])
        self.yf = array(self.yf[::-1])
        self.sf = self.yf/(2*pi*self.xf*self.nu(self.xf))
        self.yi = (2./3)*self.mdot*self.xf
        self.si = self.yi/(2*pi*self.xf*self.nu(self.xf))
        self.ds = (self.sf - self.si)/self.si
        self.dyf = (self.yf - self.yi)/self.yi
    def plot(self,linetype='.',ax=None):
        if ax == None:
            fig,axes = subplots(2,2)
            axes[0,0].loglog(self.xf,self.yf,linetype)
            axes[0,1].loglog(self.xf,self.sf,linetype)
            axes[1,0].plot(self.xf,(self.yf-self.yi)/self.yi,linetype)
            axes[1,1].plot(self.xf,(self.sf-self.si)/self.si,linetype)
            axes[0,0].set_ylabel('F')
            axes[0,1].set_ylabel('$\\Sigma$')
            axes[1,0].set_ylabel('$\\frac{\\Delta F}{F}$')
            axes[1,1].set_ylabel('$\\frac{\\Delta \\Sigma}{\\Sigma}$')
            fig.canvas.draw()
        else:
            ax.plot(self.xf,(self.sf-self.si)/self.si,linetype)
    def plot_one(q,label,linetype='.',ax=None):
        draw_flag = False
        # if ax==None:
        #     draw_flag = True
        #     fig=figure()
        #     ax=fig.add_subplot(111)

        if q=='dsig':
            ax.plot(self.xf,self.ds,linetype,label=label)
        if q=='dflux':
            ax.plot(self.xf,self.dyf,linetype,label=label)
        if q =='sigma':
            ax.loglog(self.xf,self.sf,linetype,label=label)
        if q == 'flux':
            ax.loglog(self.xf,self.yf,linetype,label=label)
        if draw_flag:
            fig.canvas.draw()

def plot_all(sims,lbl,q='ds',logx=False,logy=False,ylims=None,xlims=None):

    if lbl=='mp':
        labels = ['M_p=%.1eM_{th},M_p=%.1eM_\\nu'%(s.mp,s.mp/s.mvisc) for s in sims]
        tstr = ''
    if lbl=='mdot':
        labels = ['$\\dot{M}=%.1e$'%s.mdot for s in sims]
        tstr =''
    if lbl=='alpha':
        tstr ='$M_p = %.1eM_{th}$' % sims[0].mp
        labels = ['$\\alpha=%.1e, M_p=%.1eM_\\nu$'%(s.alpha,s.mp/s.mvisc) for s in sims]

    fig=figure()
    ax=fig.add_subplot(111)
    for s,lb in zip(sims,labels):
        if q=='dsig':
            ax.plot(s.xf,s.ds,label=lb)
        if q=='dflux':
            ax.plot(s.xf,s.dyf,label=lb)
        if q=='sigma':
            ax.plot(s.xf,s.sf,label=lb)
        if q=='flux':
            ax.plot(s.xf,s.yf,label=lb)


    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('symlog',linthreshy=1e-7)

    if ylims != None:
        ax.set_ylim(ylims)
    if xlims != None:
        ax.set_xlim(xlims)
    ax.set_title(tstr)

    ax.legend(loc='best')
