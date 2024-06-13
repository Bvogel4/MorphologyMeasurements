import argparse,pickle,pymp,sys,warnings
import numpy as np
import matplotlib.pylab as plt
from math import pi,degrees
from skimage.measure import EllipseModel
from numpy import sin,cos
from matplotlib.patches import Ellipse
plt.rcParams.update({'text.usetex':False})
warnings.filterwarnings("ignore")
def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)
def RMS(res):
    if not isinstance(res,np.ndarray): res = np.array(res)
    return( np.sqrt(sum(res**2)/len(res)) )
def pix2kpc(pix,width):
    return(pix/1000.*width)

parser = argparse.ArgumentParser(description='Collect images of all resolved halos from a given simulation. Images will be generated across all orientations.')
parser.add_argument('-f','--feedback',choices=['BW','SB','RDZ'],default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',required=True,help='Simulation to analyze')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-v','--verbose',action='store_true')
args = parser.parse_args()

SimInfo = pickle.load(open(f'../SimulationInfo.{args.feedback}.pickle','rb'))
halos = SimInfo[args.simulation]['goodhalos']
Images = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Images.pickle','rb'))
Profiles = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Profiles.pickle','rb'))

ShapeData = pymp.shared.dict()
MaskData = pymp.shared.dict()
prog=pymp.shared.array((1,),dtype=int)
if not args.verbose: print(f'Fitting Isophotes for {args.simulation}: 0.000%')
with pymp.Parallel(args.numproc) as pl:
    for i in pl.xrange(len(halos)):
        if args.verbose: print(f'Analyzing {args.simulation}-{halos[i]}...')
        hid = halos[i]
        current_shape = {}
        current_mask = {}
        for x in np.arange(0,180,30):
            for y in np.arange(0,360,30):
                rbins = Profiles[str(hid)][f'x{x:03d}y{y:03d}']['rbins']
                Rhalf = Profiles[str(hid)][f'x{x:03d}y{y:03d}']['Reff']
                if np.isnan(Rhalf): Rhalf = Profiles[str(hid)][f'x{x:03d}y{y:03d}']['Rhalf']
                ind_eff = np.argmin(abs(rbins-Rhalf))
                v = Profiles[str(hid)][f'x{x:03d}y{y:03d}']['v_lum_den'][ind_eff]
                im = Images[str(hid)][f'x{x:03d}y{y:03d}']
                im = np.flip(im,0)
                iso,tolerance = [[[],[]],.01]
                if not np.isnan(v):
                    while len(iso[0])==0 and tolerance<.1:
                        iso = np.where((im>v*(1-tolerance)) & (im<v*(1+tolerance)))
                        tolerance+=.01
                
                f,ax = plt.subplots(1,1)
                LogImage = plt.imread(f'../../Figures/Images/{args.simulation}.{args.feedback}/{hid}/{hid}.x{x:03d}.y{y:03d}.png')
                ax.imshow(LogImage)
                ax.set_xlim([0,1e3])
                ax.set_ylim([1e3,0])
                ax.scatter(500,500,c='k',marker='+')
                
                if len(iso[0])>0:
                    #Plot isophote
                    ax.scatter(iso[1],iso[0],c='r',marker='.',s=1)
                    try:
                        #Fit Ellipse to Isophote
                        xy=np.zeros((len(iso[0]),2))
                        for idx in range(len(iso[0])):
                            xy[idx]=[iso[1][idx],iso[0][idx]]
                        #Fit ellipse
                        E = EllipseModel()
                        E.estimate(np.array(xy))
                        params = E.params
                        cen = np.array([params[0],params[1]])
                        phi = params[4]
                        a,b = params[2],params[3]
                        #a = max([params[2],params[3]])
                        #b = min([params[2],params[3]])
                        residual = E.residuals(np.array(xy))
                        rms = RMS(residual)
                        #Plot Ellipse and Fit Results
                        ax.add_patch(Ellipse(cen,2*a,2*b,angle=degrees(phi),facecolor='None',edgecolor='orange'))
                        plt.plot([-a*cos(phi)+cen[0],a*cos(phi)+cen[0]],[-a*sin(phi)+cen[1],a*sin(phi)+cen[1]],color='orange')
                        plt.plot([-b*cos(phi+pi/2)+cen[0],b*cos(phi+pi/2)+cen[0]],[-b*sin(phi+pi/2)+cen[1],b*sin(phi+pi/2)+cen[1]],color='orange')
                        atrue,btrue = max([a,b]),min([a,b])
                        ax.set_title(f'b/a: {round(btrue/atrue,3)}  RMS: {round(rms,3)}  Manual: False',fontsize=15)
                        current_shape[f'x{x:03d}y{y:03d}'] = {}
                        current_shape[f'x{x:03d}y{y:03d}']['b/a'] = btrue/atrue
                        current_shape[f'x{x:03d}y{y:03d}']['a'] = pix2kpc(atrue,6*Rhalf)
                        current_shape[f'x{x:03d}y{y:03d}']['b'] = pix2kpc(btrue,6*Rhalf)
                        current_mask[f'x{x:03d}y{y:03d}'] = False if rms<20 else True
                    except:
                        ax.set_title(f'b/a: NaN  RMS: NaN  Manual: False',fontsize=15)
                        current_shape[f'x{x:03d}y{y:03d}'] = {}
                        current_shape[f'x{x:03d}y{y:03d}']['b/a'] = np.NaN
                        current_shape[f'x{x:03d}y{y:03d}']['a'] = np.NaN
                        current_shape[f'x{x:03d}y{y:03d}']['b'] = np.NaN
                        current_mask[f'x{x:03d}y{y:03d}'] = True
                else:
                    ax.set_title(f'b/a: NaN  RMS: NaN  Manual: False',fontsize=15)
                    current_shape[f'x{x:03d}y{y:03d}'] = {}
                    current_shape[f'x{x:03d}y{y:03d}']['b/a'] = np.NaN
                    current_shape[f'x{x:03d}y{y:03d}']['a'] = np.NaN
                    current_shape[f'x{x:03d}y{y:03d}']['b'] = np.NaN
                    current_mask[f'x{x:03d}y{y:03d}'] = False
                
                f.savefig(f'../../Figures/Images/{args.simulation}.{args.feedback}/{hid}/{hid}.x{x:03d}.y{y:03d}.Isophote.png',bbox_inches='tight',pad_inches=.1)
                plt.close()
                prog[0]+=1
                if not args.verbose: myprint(f'Fitting Isophotes for {args.simulation}: {round(prog[0]/(len(halos)*72)*100,3)}%',clear=True)
        ShapeData[str(hid)] = current_shape
        MaskData[str(hid)] = current_mask
        if args.verbose: print(f'{args.simulation}-{halos[i]} done.')

#Pymp shared dictionaries 'corrupt' when saving, so transfer to standerd dict objects
ShapeFile,MaskFile = {},{}
for hid in halos:
    ShapeFile[str(hid)] = ShapeData[str(hid)]
    MaskFile[str(hid)] = MaskData[str(hid)]
pickle.dump(ShapeFile,open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle','wb'))
pickle.dump(MaskFile,open(f'../../Data/{args.simulation}.{args.feedback}.Masking.pickle','wb'))