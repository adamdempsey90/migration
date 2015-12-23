import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import h5py
from collections import OrderedDict
class Parameters():
    key_vals=OrderedDict([
        ('nr',int),
        ('ri',float),
        ('ro',float),
        ('alpha',float),
        ('gamma',float),
        ('h',float),
        ('bc_lam_inner',float),
        ('bc_lam_outer',float),
        ('dt',float),
        ('nvisc',float),
        ('nt',int),
        ('release_time',float),
        ('read_initial_conditions',bool),
        ('planet_torque',bool),
        ('move_planet',bool),
        ('gaussian',bool),
        ('one_sided',float),
        ('a',float),
        ('mp',float),
        ('G1',float),
        ('beta',float),
        ('delta',float),
        ('c',float),
        ('eps', float),
        ('outputname',str)])

    comment_lines = {'nr':'#Grid Parameters:', 'alpha':'\n\nDisk Parameters:','bc_lam_inner':'\n\n#Boundary Conditions:', 'dt':'\n\n#Time Parameters:','planet_torque':'\n\n#Planet Properties:'}

    def __init__(self,dataspace,fname='results.hdf5'):
           # We read in the from an hdf5 file
        setattr(self,'outputname',fname)
        for key,datatype in self.key_vals.items()[:-1]:
            try:
                setattr(self,key,dataspace[key].astype(datatype)[0])
            except AttributeError:
                setattr(self,key,getattr(dataspace,key))

    def dump_params(self,fname='params_py.in',**kargs):

        lines =[]
        for key in self.key_vals.keys():
            try:
                lines.append(self.comment_lines[key])
            except KeyError:
                pass

            if key in kargs.keys():
                val = kargs[key]
            else:
                val = getattr(self,key)

            lines.append(str(key) + ' = ' + str(val))

        with open(fname,'w') as f:
            f.write('\n'.join(lines))


class Sim(Parameters):
    def __init__(self,fname = 'results.hdf5'):

        f = h5py.File(fname,'r')

        Parameters.__init__(self,f['/Migration/Parameters']['Parameters'],fname=fname)

        sols = f['/Migration/Solution']
        mesh = f['/Migration/Mesh']
        mat = f['/Migration/Matrix']
        ss = f['/Migration/SteadyState']
        self.t = sols['times'][:]
        self.at =sols['avals'][:]
        self.vs = sols['vs'][:]
        self.lam = sols['lam'][:].transpose()
        self.mdot = sols['mdot'][:].transpose()
        self.torque= sols['torque'][:].transpose()

        self.rc = mesh['rc'][:]
        self.dr = mesh['dr'][:]
        self.rmin=mesh['rmin'][:]
        self.lam0 = mesh['lami'][:]
        self.mdot0 = mesh['mdoti'][:]

        self.main_diag = mat['md'][:]
        self.lower_diag = mat['ld'][:]
        self.upper_diag = mat['ud'][:]
        self.rhs = mat['fm'][:]


        self.lam_ss = ss['lam_ss'][:].transpose()
        self.lamp = ss['lamp'][:].transpose()
        self.mdot_ss = ss['mdot_ss'][:]
        self.vs_ss = ss['vs_ss'][:]
        self.eff = ss['eff'][:]
        self.mdot0 = self.eff/self.mdot_ss


        f.close()
#
#        self.t = dat_p[:,0]
#        self.nt = len(self.t)
#        self.at = dat_p[:,1]
#        self.vst = dat_p[:,2]
#
#        dat = np.loadtxt(fname)
#        self.nr = dat[0,0]
#        self.rc = dat[0,1:]
#        self.rm = dat[1,1:]
        self.mth = self.h**3
        self.disk_mass = np.dot(self.dr,self.lam0)
#        self.dr = dat[2,1:]
        self.tvisc = self.ro**2/(self.nu(self.ro))
        self.lami = self.bc_lam_inner
        self.lamo = self.bc_lam_outer

        self.B = 2*self.at*(self.lamo-self.lami)/(self.mp*self.mth)
        self.Bfac = (np.sqrt(self.ro)-np.sqrt(self.ri))/(np.sqrt(self.at)-np.sqrt(self.ri))
        self.freduc  = self.B * self.Bfac * (1 - self.eff)
#        self.dTr = dat[3,1:]
#        self.lami = dat[4,0]
#        self.lam0 = dat[4,1:]
#        self.lamo = dat[5,0]
#        self.mdot0 = dat[5,1:]
#        self.lams = dat[6:-self.nt,1:].transpose()
#        self.mdots = dat[-self.nt:,1:].transpose()
        self.vr = -self.mdot/self.lam
#        self.vr0 = -self.mdot0/self.lam0
#        self.lamp = np.zeros(self.lam.shape)
#        for i in range(self.lam.shape[1]):
#            self.lamp[:,i] = (self.lam[:,i]-self.lam0)/self.lam0

    def nu(self,x):
        return self.alpha*self.h*self.h * pow(x,self.gamma)
    def vr_nu(self,x):
        return -1.5 * self.nu(x)/x

    def animate(self,tend,skip,tstart=0,q='lam',logx = True,logy=True):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        inds = (self.t <= tend)&(self.t >= tstart)
        ax.set_xlabel('$r$',fontsize=20);

        if q == 'lam':
            ax.set_ylabel('$\\lambda$',fontsize=20)
            line, = ax.plot(self.rc,self.lam0)
            linep, = ax.plot(self.at[0],self.lam0[self.rc>=self.at[0]][0],'o',markersize=10)
            ax.plot(self.rc,self.lam_ss[:,-1],'--r')
            ax.plot(self.rc,self.lam0,'--k')
            dat = self.lam[:,inds]
            dat = dat[:,::skip]
            if logy:
                ax.set_yscale('log')
        elif q == 'mdot':
            ax.set_ylabel('$\\dot{M}$',fontsize=20)
            line, = ax.plot(self.rc,self.mdot0/self.mdot0)
            linep, = ax.plot(self.at[0],1,'o',markersize=10)

            ax.axhline(1,color='k',linestyle='--')
            dat = self.mdot[:,inds]
            dat = dat[:,::skip]
            for i in range(dat.shape[1]):
                dat[:,i] /= self.mdot0
        elif q == 'torque':
            ax.set_ylabel('$ \\Lambda(r)$',fontsize=20)
            line, = ax.plot(self.rc,self.torque[:,0])
            linep, = ax.plot(self.at[0],self.torque[:,0][self.rc>=self.at[0]][0],'o',markersize=10)
            dat = self.torque[:,inds][:,::skip]

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

    def time_series(self,axes=None,fig=None,scale=True):
        if fig == None:
            fig,axes = plt.subplots(2,2,sharex='col')
        fig.subplots_adjust(hspace=0)
        axes[0,0].semilogx(self.t,self.at,'.-')

        if scale:
            dat = self.vs / (-1.5*self.nu(self.at)/self.at)
        else:
            dat = self.vs

        axes[1,0].semilogx(self.t,dat,'.-')

        axes[1,0].set_yscale('log')
        axes[0,0].set_ylabel('$a$',fontsize=20)
        axes[1,0].set_ylabel('$\\dot{a}$',fontsize=20)
        axes[1,0].set_xlabel('$t$',fontsize=20)

        if (self.at >= self.rc[-1]).any():
            axes[0,0].axhline(self.rc[-1],color='k')
        if (self.at <= self.rc[0]).any():
            axes[0,0].axhline(self.rc[0],color='k')

        axes[1,1].plot(self.at,dat,'.-')
        axes[1,1].set_xlabel('$a$',fontsize=20)
        axes[1,1].set_ylabel('$\\dot{a}$',fontsize=20)
        if scale:
            axes[1,1].set_yscale('log')
            axes[1,1].plot(self.at,self.freduc,'+r')
        else:
            axes[1,1].plot(self.at,self.freduc*self.vr_nu(self.at),'+r')

        axes[0,1].plot(self.at,1-self.eff,'.-',label='f_mdot')
        axes[0,1].plot(self.at,self.B,'.-',label='Mdisk/Mplanet')
        axes[0,1].legend(loc='best')
        axes[0,1].set_yscale('log')

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

