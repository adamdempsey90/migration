import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.interpolate as interp
from subprocess import call
#
# pdict_def={'PlanetFlag': 1,
# 'a' : 10,
# 'gamma' :0.5,
# 'mu' : 1,
# 'h' : 0.1,
# 'alpha' : .01,
# 'beta' :  2./3,
# 'delta': 1,
# 'G1' : 0,
# 'ri' : 1,
# 'ro' : 100,
# 'lami': 1e-3,
# 'lamo': 1,
# 'f' : 1,
# 'c' : 2./3,#2./3,
# 'w' :.1,
# 'UseGaussian' : 0,
# 'nr' : 1e4,
# 'silent': True,
# 'lamfloor':1e-8,
# 'OneSided': False}

def smoothing(x,x0,w):
    return .5*(1 + np.tanh( (x-x0)/w))


class Simulation():

    key_order = ['PlanetFlag','a','gamma','mu','h','alpha','beta','delta',
        'G1','ri','ro','lami','lamo','f','c','w','UseGaussian','nr','onesided']
    def __init__(self,**kargs):
        self.pdict={'PlanetFlag': 1,
        'a' : 10,
        'gamma' :0.5,
        'mu' : 1,
        'h' : 0.1,
        'alpha' : .1,
        'beta' :  1,
        'delta': 1,
        'G1' : 0,
        'ri' : .05,
        'ro' : 30,
        'lami': 1e-12,
        'lamo': 1e-2,
        'f' : 1,
        'c' : 2./3,
        'w' :.1,
        'UseGaussian' : 0,
        'nr' : 512,
        'silent': True,
        'lamfloor':1e-8,
        'onesided': False}
        self.success = True
        for key,val in self.pdict.items():
            if type(val) == bool:
                val = int(val)
            setattr(self,key,val)
        for key,val in kargs.items():
            if not self.silent:
                print 'Setting ', key, 'to ', val
            if type(val) == bool:
                val = int(val)
            self.pdict[key] = val
            setattr(self, key, val)
        self.nu0 = self.alpha*self.h**2
        self.nr = int(self.nr)
        self.mth = self.h**3
        self.flaringindex = (self.gamma-.5)/2
        self.r = np.exp(np.linspace(np.log(self.ri-1e-12),np.log(self.ro),self.nr))
    def run(self):
        arglist = ['./steadystate2.m']
        for key in self.key_order:
            arglist.append('%.12f' % getattr(self,key))
        print ' '.join(arglist)
        call(arglist)
        dat = np.loadtxt('results.dat')
        self.r = dat[:,0]
        self.lam = dat[:,1]
        self.vr = dat[:,2]
        self.dTr = dat[:,3]
        self.mdot = -self.lam[10]*self.vr[10]
        self.lam0,self.mdot0 = self.adisk(self.r)
        self.vr0 = -self.mdot0/self.lam0
        self.lammin = self.lam.min()/self.lam0[self.lam == self.lam.min()]
        if abs(self.mdot) < 1e-10:
            self.success = False
            print 'No steady state found'
    #     if (self.lam<=self.lamfloor).any() or (self.lam<0).any():
#             self.success = False
#             print 'No steady state found'


    def run_migration(self,ai,ao,na):

        self.ai = ai
        self.ao = ao
        self.na = na
        arglist = ['./steadystate_migration.m']
        for key in self.key_order:
            arglist.append('%.12f' % getattr(self,key))
        arglist.append(str(ai))
        arglist.append(str(ao))
        arglist.append(str(na))
        print ' '.join(arglist)
        call(arglist)
        dat = np.loadtxt('results.dat')

        self.alist = np.zeros(na)
        self.adots = np.zeros(na)
        self.lamps = np.zeros(na)
        self.mdots = np.zeros(na)
        self.lams = np.zeros((self.nr,na))
        self.dTrs = np.zeros((self.nr,na))
        self.vrs = np.zeros((self.nr,na))
        self.r = dat[-1,-self.nr:]


        for i in range(na):
            self.alist[i] = dat[i,0]
            self.adots[i] = dat[i,1]
            self.mdots[i] = dat[i,2]
            self.lamps[i],_ = self.adisk(self.alist[i])
   #         self.lamps[i] = dat[i,3]
            self.lams[:,i] = dat[i,-2*self.nr:-self.nr]
            self.dTrs[:,i] = dat[i,-self.nr:]
            self.vrs = -self.lams[:,i]/self.mdots[i]


  #      self.dTr = dat[:,3]

        self.lam0,self.mdot0 = self.adisk(self.r)
        self.vr0 = -self.mdot0/self.lam0
        self.adot0 = -1.5*self.nu0/np.sqrt(self.alist)
        self.success = True

    def nu(self,r):
        alpha = self.alpha
        h = self.h
        gam = self.gamma
        return alpha * h**2 * r**(gam)
    def adot_pred(self,a,mdot):
        return -1.5 * self.nu0 * 2*np.sqrt(a) *(self.lamo-self.lami)*(1-mdot/self.mdot0)/(self.h**3 * self.mu)
    def adisk(self,r):
        li = self.lami
        lo = self.lamo
        ri = self.ri
        ro = self.ro
        nu = self.nu(r)
        nui = self.nu(ri)
        nuo  = self.nu(ro)
        mdot0 = 1.5*(lo-li)/(ro/nuo-ri/nui)
        return li + (2*mdot0/3)*(r/nu - ri/nui),mdot0

    def summary(self,logx=True,logy=True,savefig=None,fig=None,axes=None):
        a = self.a
        h = self.h

        if savefig != None:
            figsize = (20,15)
        else:
            figsize = (10,5)

        if fig == None and axes == None:
            newfig = True
            fig,axes = plt.subplots(2,2,figsize=figsize)
            tstr = '$\\dot{M}$ = %.2e, $\\dot{M}_0$ = %.2e' %(self.mdot,self.mdot0)
        else:
            newfig = False
            tstr = ''


        axes[0,0].plot(self.r,self.lam,'-',linewidth=3)
        axes[0,0].plot(self.r,self.lam0,'--',linewidth=2)
        axes[1,1].plot(self.r,self.vr,'-',linewidth=3)
        axes[1,1].plot(self.r,self.vr0,'--',linewidth=3)
        axes[1,0].plot((self.r-a)/(h*self.r),(self.lam-self.lam0)/self.lam0,'-',linewidth=3)
        axes[0,1].plot(self.r,self.dTr,linewidth=3)
#         axes[0,1].semilogx(self.r,self.vr,'-')
#         axes[0,1].semilogx(self.r,self.vr0,'--')
#         axes[0,1].set_xlabel('$r$',fontsize=20)
#         axes[0,1].set_ylabel('$v_r$',fontsize=20)


        if logx:
            axes[0,0].set_xscale('log')
            axes[1,1].set_xscale('log')
        if logy:
            axes[0,0].set_yscale('log')

        if newfig:
            axes[0,0].set_xlabel('$r$',fontsize=20)
            axes[0,0].set_ylabel('$\\lambda$',fontsize=20)
            axes[0,0].set_title(tstr)
           # axes[1,1].set_yscale('log')
            axes[1,1].set_xlabel('$r$',fontsize=20)
            axes[1,1].set_ylabel('$\\frac{v_r}{v_{r,0}}$',fontsize=20)
            axes[0,1].set_xlabel('$r$',fontsize=20)
            axes[0,1].set_ylabel('$\\Lambda(r)$',fontsize=20)
            axes[1,0].set_xlim(-20,20)
            axes[1,0].axhline(0)
            axes[1,1].axhline(1)
            axes[1,0].set_xlabel('$\\frac{r - a}{H}$',fontsize=20)
            axes[1,0].set_ylabel('$\\frac{\\Delta \\lambda}{\\lambda}$',fontsize=20)
            for ax in axes.flatten():
                ax.minorticks_on()




        if savefig != None:
            fig.savefig(savefig+'.pdf')

    def scaleH(self,x):
        return self.h*x * x**(self.flaringindex)
    def dTr(self,x,a):
        xi = (x-a)/self.scaleH(x)
        norm = self.f * a * np.pi * (self.mu*self.mth)**2
        delta = np.maximum(self.scaleH(x),np.abs(x-a))
        right_fac = norm*(a/delta)**4
        left_fac = -norm*(x/delta)**4
        left_fac *= (1-smoothing(xi,-self.c,self.w))
        right_fac *= smoothing(xi,-self.c,self.w)*smoothing(xi,self.c,self.w)
        return left_fac*(1-self.onesided) + right_fac

    def I_integrand(self,x,a):
        return self.dTr(x,a)/(3*self.nu(x)*np.pi*np.sqrt(x))

    def I_int(self,x,a):
        if type(x) != np.ndarray:
            if type(x) != list:
                x = [x]
            x = np.array(x)


        res = np.zeros(x.shape)


        res[0],err = integ.quad(self.I_integrand,self.ri,x[0],args=(a,))
        if len(x) > 1:
            for i,xi in enumerate(x[1:],start=1):
                ans,err=integ.quad(self.I_integrand,x[i-1],xi,args=(a,))
                res[i] = res[i-1] + ans
        return (x/self.ri)**(self.gamma-.5) * np.exp(-res)

    def K_integrand(self,x,a,interpolated_I=None):
        if interpolated_I == None:
            res = self.I_int(x,a)
        else:
            res = interpolated_I(x)
        return res/(3*self.nu(x))
    def K_int(self,x,a,interpolated_I = None):
        if type(x)  != np.ndarray:
            if type(x) != list:
                x = [x]
            x = np.array(x)
        res = np.zeros(x.shape)
        res[0],err = integ.quad(self.K_integrand,self.ri,x[0],args=(a,interpolated_I))
        if len(x) > 1:
            for i,xi in enumerate(x[1:],start=1):
                ans,err=integ.quad(self.K_integrand,x[i-1],xi,args=(a,interpolated_I))
                res[i] = res[i-1] + ans

        return res
    def python_calculate(self,rc):
        interp_I = interp.interp1d(rc,self.I_int(rc,self.a))
        interp_K = interp.interp1d(rc,self.K_int(rc,self.a,interpolated_I = interp_I))

        mdot = (self.lamo * interp_I(self.ro) - self.lami)/interp_K(self.ro)
        lam = self.lami + mdot*interp_K(rc)
        lam /= interp_I(rc)
        return mdot,lam

    def adot_integ(self,rc,lam_func,a,interpolated=True):
        if interpolated:
            return  self.dTr(rc,a) * lam_func(rc)/rc
        else:
            return self.dTr(rc,a) * lam_func/rc

    def calc_adot(self,rc,lam,a):
        lam_func = interp.interp1d(rc,lam)
        res,err=integ.quad(self.adot_integ,self.ri,self.ro,args=(lam_func,a))
        res *= -2*np.sqrt(a)/(self.mu*self.mth)
        return res

    def get_interpolated_functions(self,rc,a):
        ifunc = interp.interp1d(rc,self.I_int(rc,a))
        kfunc = interp.interp1d(rc,self.K_int(rc,a,ifunc))
        return ifunc,kfunc
