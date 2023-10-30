#https://gala.adrian.pw/en/latest/potential/scf-examples.html#computing-expansion-coefficients-from-particle-positions

import argparse,pickle,pymp,pynbody,sys,warnings
import numpy as np
import gala.potential as gp
import matplotlib.pylab as plt
from scipy.optimize import curve_fit as cf
warnings.filterwarnings("ignore")
def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)

def Hernquist(r,rho,r_s):
    if not isinstance(r,np.ndarray): r = np.array(r)
    return( rho/( (r/r_s) * (1+r/r_s)**3 ) )

parser = argparse.ArgumentParser(description='Collect Gala SCF Fits')
parser.add_argument('-f','--feedback',choices=['BW','SB'],default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',choices=['cptmarvel','elektra','storm','rogue','h148','h229','h242','h329'],required=True,help='Simulation to analyze')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-i','--image',action='store_true',help='Plot Density Profiles')
parser.add_argument('-v','--verbose',action='store_true',help='Print halo IDs being analyzed')
args = parser.parse_args()

SimInfo = pickle.load(open(f'SimulationInfo.{args.feedback}.pickle','rb'))
simpath = SimInfo[args.simulation]['path']
halos = SimInfo[args.simulation]['halos']

print(f'Loading {args.simulation}')
sim = pynbody.load(simpath)
sim.physical_units()
h = sim.halos()
SCFData = pymp.shared.dict()
myprint(f'{args.simulation} loaded.',clear=True)

prog=pymp.shared.array((1,),dtype=int)
print(f'\tRunning SCF Fits: 0.00%')
with pymp.Parallel(args.numproc) as pl:
    for i in pl.xrange(len(halos)):
        hid = halos[i]
        halo = h[hid]
        current = {}

        #SCF fit params
        n,l = 16,1
        current['n,l'] = (16,1)

        pynbody.analysis.angmom.faceon(halo)

        if args.image:
            f,ax = plt.subplots(1,1)
            ax.semilogx()
            ax.semilogy()
            ax.set_xlabel('R [kpc]',fontsize=15)
            ax.set_ylabel(r'$\rho$ [M$_{\odot}$ kpc$^{-3}$]',fontsize=15)
            ax.plot([0,0],[0,0],c='k',label='DM Particles')
            ax.plot([0,0],[0,0],c='r',label='Star Particles')
            ax.plot([0,0],[0,0],c='b',label='Gala Model')
            ax.legend(loc='upper right',prop={'size':12})

        for i in [0,1]:
            hobj = [halo.d,halo.s][i]
            mass,xyz = hobj['mass'],hobj['pos']

            #Determine Scale Radius
            p = pynbody.analysis.profile.Profile(hobj,min=.25,max=int(max(hobj['r'])),nbins=500,ndim=3)
            r,rho = p['rbins'],p['density']
            par,ign = cf(Hernquist,r,rho)
            r_s = par[1]

            S,T = gp.scf.compute_coeffs_discrete(xyz,mass=mass,r_s=r_s,nmax=16,lmax=1)
            key = ['_DM','_Stellar'][i]
            current[f'S{key}'] = S
            current[f'T{key}'] = T
            SCFData[str(hid)] = current

            if args.image:
                pot = gp.SCFPotential(m=1.,r_s=r_s,Snlm=S,Tnlm=T)
                if i==0: ax.set_ylim([1e2,10**(int(np.log10(p['density'][0]))+1)])
                xyz2 = np.zeros((3, len(r)))
                xyz2[0] = r
                ax.plot(r,rho,c=['k','r'][i])
                ax.plot(r, pot.density(xyz2),c='b')
                f.savefig(f'../Images/Gala/{args.simulation}.{args.feedback}/{hid}.png',bbox_inches='tight',pad_inches=.1)
            
        prog[0]+=1
        myprint(f'\tRunning SCF Fits: {round(prog[0]/len(halos)*100,2)}%',clear=True)

OutFile = {}
for halo in halos:
    OutFile[str(halo)] = SCFData[str(halo)]
pickle.dump(OutFile,open(f'../Data/{args.simulation}.{args.feedback}.SCFData.pickle','wb'))
