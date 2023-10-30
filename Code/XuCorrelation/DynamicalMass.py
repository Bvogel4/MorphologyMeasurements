import argparse,os,pickle,pymp,pynbody,sys,time,warnings
import numpy as np
import matplotlib.pylab as plt
from pynbody.plot.sph import image
from scipy.optimize import curve_fit
def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Calculated Dynamical Masses within 1 Reff')
parser.add_argument('-f','--feedback',choices=['BW','SB'],default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',choices=['cptmarvel','elektra','storm','rogue','h148','h229','h242','h329'],required=True,help='Simulation to analyze')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
args = parser.parse_args()

SimInfo = pickle.load(open(f'../SimulationInfo.{args.feedback}.pickle','rb'))
simpath = SimInfo[args.simulation]['path']
halos = SimInfo[args.simulation]['goodhalos']

print(f'Loading {args.simulation}')
sim = pynbody.load(simpath)
sim.physical_units()
h = sim.halos()
MassData = pymp.shared.dict()
myprint(f'{args.simulation} loaded.',clear=True)

prog=pymp.shared.array((1,),dtype=int)
print(f'\tCalculating Mdyn: {round(prog[0]/len(halos)*100,2)}%')
with pymp.Parallel(args.numproc) as pl:
    for i in pl.xrange(len(halos)):
        hid = halos[i]
        halo = h[hid]

        pynbody.analysis.angmom.faceon(halo)
        Rhalf = pynbody.analysis.luminosity.half_light_r(halo)

        prof = pynbody.analysis.profile.Profile(halo,type='lin',min=.25,max=5*Rhalf,ndim=2,nbins=int((5*Rhalf)/0.1))
        indeff = np.argmin(np.abs(prof['rbins']-Rhalf))
        veff = prof['v_circ'][indeff]
        MassData[str(hid)] = ( (Rhalf*1e3)*veff**2)/(4.3009172706e-3)
        prog[0]+=1
        print(f'\tCalculating Mdyn: {round(prog[0]/len(halos)*100,2)}%')

OutFile = {}
for halo in halos:
    OutFile[str(halo)] = MassData[str(halo)]
pickle.dump(OutFile,open(f'../../Data/{args.simulation}.{args.feedback}.DynamicalMasses.pickle','wb'))
