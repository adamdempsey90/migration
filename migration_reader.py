import numpy as np
import matplotlib.pyplot as plt

class Sim():
    def __init__(self,fname = 'results.dat', pname = 'planet.dat'):
        dat_p = np.loadtxt(pname);
        self.t = dat_p[:,0]
        self.nt = len(self.t)
        self.at = dat_p[:,1]
        self.vst = dat_p[:,2]

        dat = np.loadtxt(fname)
        self.nr = dat[0,0]
        self.rc = dat[0,1:]
        self.rm = dat[1,1:]
        self.disk_mass = dat[2,0]
        self.dr = dat[2,1:]
        self.tvisc = dat[3,0]
        self.dTr = dat[3,1:]
        self.lami = dat[4,0]
        self.lam0 = dat[4,1:]
        self.lamo = dat[5,0]
        self.mdot0 = dat[5,1:]
        self.lams = dat[6:-self.nt,1:].transpose()
        self.mdots = dat[-self.nt:,1:].transpose()
        self.vr = -self.mdots/self.lams
        self.vr0 = -self.mdot0/self.lam0
        self.lamp = np.zeros(self.lams.shape)
        for i in range(self.lams.shape[1]):
            self.lamp[:,i] = (self.lams[:,i]-self.lam0)/self.lam0


    def animate(self,tend,skip,tstart=0,q='lam'):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        inds = (self.t <= tend)&(self.t >= tstart)
        ax.set_xlabel('$r$',fontsize=20);

        if q == 'lam':
            ax.set_ylabel('$\\lambda$',fontsize=20)
            line, = ax.loglog(self.rc,self.lam0)
            ax.loglog(self.rc,self.lam0,'--k')
            dat = self.lams[:,inds]
            dat = dat[:,::skip]
        elif q == 'mdot':
            ax.set_ylabel('$\\dot{M}$',fontsize=20)
            line, = ax.semilogx(self.rc,self.mdot0/self.mdot0)
            ax.axhline(1,color='k',linestyle='--')
            dat = self.mdots[:,inds]
            dat = dat[:,::skip]
            for i in range(dat.shape[1]):
                dat[:,i] /= self.mdot0
        else:
            print  'q=%s is not a valid option' % q
            return

        ax.set_ylim((dat.min(),dat.max()))

        times = self.t[inds][::skip]


        for i,t in enumerate(times):
            line.set_ydata(dat[:,i])
            ax.set_title('t = %.2e viscous times' % (t/self.tvisc))
            fig.canvas.draw()

    def time_series(self,axes=None,fig=None):
        if fig == None:
            fig,axes = plt.subplots(2,1,sharex=True)
        fig.subplots_adjust(hspace=0)
        axes[0].semilogx(self.t,self.at,'.-')
        axes[1].semilogx(self.t,self.vst,'.-')

        if (self.at >= self.rc[-1]).any():
            axes[0].axhline(self.rc[-1],color='k')
        if (self.at <= self.rc[0]).any():
            axes[0].axhline(self.rc[0],color='k')

        fig.canvas.draw()
        return axes,fig
