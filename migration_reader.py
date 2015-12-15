import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
class Sim():
    def __init__(self,fname = 'mg_results.dat', pname = 'planet.dat'):
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


    def animate(self,tend,skip,tstart=0,q='lam',logx = True,logy=True):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        inds = (self.t <= tend)&(self.t >= tstart)
        ax.set_xlabel('$r$',fontsize=20);

        if q == 'lam':
            ax.set_ylabel('$\\lambda$',fontsize=20)
            line, = ax.plot(self.rc,self.lam0)
            linep, = ax.plot(self.at[0],self.lam0[self.rc>=self.at[0]][0],'o',markersize=10)
            ax.plot(self.rc,self.lam0,'--k')
            dat = self.lams[:,inds]
            dat = dat[:,::skip]
            if logy:
                ax.set_yscale('log')
        elif q == 'mdot':
            ax.set_ylabel('$\\dot{M}$',fontsize=20)
            line, = ax.plot(self.rc,self.mdot0/self.mdot0)
            linep, = ax.plot(self.at[0],1,'o',markersize=10)

            ax.axhline(1,color='k',linestyle='--')
            dat = self.mdots[:,inds]
            dat = dat[:,::skip]
            for i in range(dat.shape[1]):
                dat[:,i] /= self.mdot0
        else:
            print  'q=%s is not a valid option' % q
            return

        if logx:
            ax.set_xscale('log')
        ax.set_ylim((dat.min(),dat.max()))

        times = self.t[inds][::skip]

        avals = self.at[inds][::skip]

        for i,t in enumerate(times):
            line.set_ydata(dat[:,i])
            linep.set_xdata(avals[i])
            linep.set_ydata(dat[:,i][self.rc>=avals[i]][0])
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
    def write_lam_to_file(self,r,lam,interpolate=False):
        if interpolate:
            lam = interp1d(r,lam)(self.rc)
            r = self.rc

        with open("lambda_init.dat","w") as f:
            lines = ["%.12g\t%.12g" %(x,l) for x,l in zip(r,lam)]
            f.write('\n'.join(lines))

    def load_steadystate(self,directory=''):
        if len(directory)>0 and directory[-1] != '/':
            directory += '/'
        dat = np.loadtxt(directory+'results.dat')
        self.ss_r = dat[:,0]
        self.ss_lam = dat[:,1]
        self.ss_vr = dat[:,2]
        self.ss_dTr = dat[:,3]
        self.ss_mdot = -dat[10,1]*dat[10,2]

