import argparse,pickle,pymp,sys,warnings
import numpy as np
from math import pi
from numpy import degrees
from numpy import sin,cos
from matplotlib.patches import Ellipse
import matplotlib.pylab as plt
sys.path.append('..')
from ShapeFunctions.Functions import Ellipsoid,Project,kpc2pix,pix2kpc,Project_OLD
plt.rcParams.update({'text.usetex':False})
warnings.filterwarnings("ignore")
def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)

parser = argparse.ArgumentParser(description='Collect images of all resolved halos from a given simulation. Images will be generated across all orientations.')
parser.add_argument('-f','--feedback',choices=['BW','SB'],default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',choices=['cptmarvel','elektra','storm','rogue','h148','h229','h242','h329'],required=True,help='Simulation to analyze')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-p','--plot',action='store_true')
#parser.add_argument('-o','--overwrite',action='store_true',help='Overwrite existing images')
args = parser.parse_args()

SimInfo = pickle.load(open(f'../SimulationInfo.{args.feedback}.pickle','rb'))
halos = SimInfo[args.simulation]['halos']
Images = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Images.pickle','rb'))
Profiles = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Profiles.pickle','rb'))
Shapes = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.3DShapes.pickle','rb'))

# smooth = {
#     'cptmarvel':[1,2,3,5,6,7,10,11,13],
#     'elektra':[2,3,4,5,9,10,12,17,36],
#     'storm':[2,3,4,5,6,7,8,14,15,23,31,44],
#     'rogue':[7,8,10,11,12,15,16,17,28,31],
#     'h148':[3,4,6,7,11,12,13,15,20,23,27,28,33,34,38,65,114],
#     'h229':[2,3,6,14,18,22,49,92],
#     'h242':[8,10,21,26,30,34,38],
#     'h329':[7,29,115,117]
# }

ShapeData = pymp.shared.dict()
prog=pymp.shared.array((1,),dtype=int)
print(f'Projecting Ellipsoids for {args.simulation}: 0.000%')
with pymp.Parallel(args.numproc) as pl:
    for i in pl.xrange(len(halos)):
        hid = halos[i]
        current_shape = {}
        for xrot in np.arange(0,180,30):
            for yrot in np.arange(0,360,30):
                #Get 3D Ellipsoid Projection
                try:
                    rbins = Shapes[str(hid)]['rbins']
                    Rhalf = Profiles[str(hid)][f'x{xrot:03d}y{yrot:03d}']['Reff']
                    if np.isnan(Rhalf): Rhalf = Profiles[str(hid)][f'x{xrot:03d}y{yrot:03d}']['Rhalf']
                    ind_eff = np.argmin(abs(rbins-Rhalf))
                    a,ba,ca,Es = [Shapes[str(hid)]['a'][ind_eff],Shapes[str(hid)]['ba'][ind_eff],
                                  Shapes[str(hid)]['ca'][ind_eff],Shapes[str(hid)]['Es'][ind_eff]]
                    if 'ba_smooth' in Shapes[str(hid)]:
                        ba = float(Shapes[str(hid)]['ba_smooth'](Rhalf))
                        ca = float(Shapes[str(hid)]['ca_smooth'](Rhalf))
                    x,y,z = Ellipsoid(a,ba,ca,Es,xrot,yrot)
                    x = -x
                    ap,bp,cen,phi = Project_OLD(x,y,z)
                    atrue,btrue = max([ap,bp]),min([ap,bp])
                    current_shape[f'x{xrot:03d}y{yrot:03d}'] = {}
                    current_shape[f'x{xrot:03d}y{yrot:03d}']['b/a'] = btrue/atrue
                    current_shape[f'x{xrot:03d}y{yrot:03d}']['a'] = atrue
                    current_shape[f'x{xrot:03d}y{yrot:03d}']['b'] = btrue
                except:
                    ap,bp,cen,phi = 0,0,[500,500],0
                    current_shape[f'x{xrot:03d}y{yrot:03d}'] = {}
                    current_shape[f'x{xrot:03d}y{yrot:03d}']['b/a'] = np.NaN
                    current_shape[f'x{xrot:03d}y{yrot:03d}']['a'] = np.NaN
                    current_shape[f'x{xrot:03d}y{yrot:03d}']['b'] = np.NaN

                if args.plot:
                    f,ax = plt.subplots(1,1)
                    LogImage = plt.imread(f'../../Figures/Images/{args.simulation}.{args.feedback}/{hid}/{hid}.x{xrot:03d}.y{yrot:03d}.png')
                    ax.imshow(LogImage)
                    ax.set_xlim([0,1e3])
                    ax.set_ylim([1e3,0])
                    ax.scatter(500,500,c='k',marker='+')

                    #Get Isophote
                    try:
                        Rhalf = Profiles[str(hid)][f'x{xrot:03d}y{yrot:03d}']['Reff']
                        rbins = Profiles[str(hid)][f'x{xrot:03d}y{yrot:03d}']['rbins']
                        ind_eff = np.argmin(abs(rbins-Rhalf))
                        v = Profiles[str(hid)][f'x{xrot:03d}y{yrot:03d}']['v_lum_den'][ind_eff]
                        im = Images[str(hid)][f'x{xrot:03d}y{yrot:03d}']
                        im = np.flip(im,0)
                        iso,tolerance = [[[],[]],.01]
                        if not np.isnan(v):
                            while len(iso[0])==0 and tolerance<.1:
                                iso = np.where((im>v*(1-tolerance)) & (im<v*(1+tolerance)))
                                tolerance+=.01
                        ax.scatter(iso[1],iso[0],c='r',marker='.',s=1)
                    except:
                        continue
                    
                    #Get Projected Ellipse
                    if ap>0:
                        x,y = kpc2pix(x,6*Rhalf)+500,kpc2pix(y,6*Rhalf)+500
                        cen = [cen[0]+500,cen[1]+500]
                        ap,bp = kpc2pix(ap,6*Rhalf),kpc2pix(bp,6*Rhalf)
                        ax.add_patch(Ellipse(cen,2*ap,2*bp,angle=degrees(phi),facecolor='None',edgecolor='orange',linewidth=2))
                        plt.plot([-ap*cos(phi)+cen[0],ap*cos(phi)+cen[0]],[-ap*sin(phi)+cen[1],ap*sin(phi)+cen[1]],color='orange')
                        plt.plot([-bp*cos(phi+pi/2)+cen[0],bp*cos(phi+pi/2)+cen[0]],[-bp*sin(phi+pi/2)+cen[1],bp*sin(phi+pi/2)+cen[1]],color='orange')
                        ax.set_title(f'Projected b/a: {round(btrue/atrue,3)}',fontsize=15)
                    else:
                        ax.set_title(f'Projected b/a: {np.NaN}',fontsize=15)
                    f.savefig(f'../../Figures/Images/{args.simulation}.{args.feedback}/{hid}/{hid}.x{xrot:03d}.y{yrot:03d}.Intrinsic.png',bbox_inches='tight',pad_inches=.1)
                    plt.close()

                prog[0]+=1
                myprint(f'Projecting Ellipsoids for {args.simulation}: {round(prog[0]/(len(halos)*72)*100,3)}%',clear=True)
        ShapeData[str(hid)] = current_shape

#Pymp shared dictionaries 'corrupt' when saving, so transfer to standerd dict objects
ShapeFile = {}
for hid in halos:
    ShapeFile[str(hid)] = ShapeData[str(hid)]
pickle.dump(ShapeFile,open(f'../../Data/{args.simulation}.{args.feedback}.ProjectedData.pickle','wb'))
myprint(f'{args.simulation} done.',clear=True)