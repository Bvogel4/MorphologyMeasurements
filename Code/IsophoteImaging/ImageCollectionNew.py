import argparse,os,pickle,pymp,pynbody,sys,time,warnings
import numpy as np
import matplotlib.pylab as plt
from pynbody.plot.sph import image
from scipy.optimize import curve_fit
import os
from pathlib import Path
import traceback

# Current script directory (IsophoteImaging)
current_dir = Path(__file__).parent
# Root directory (one level up from IsophoteImaging)
root_dir = current_dir.parent.parent
code_dir= root_dir / 'Code'
# Absolute path for Figures/Images directory
images_dir = root_dir / 'Figures' / 'Images'
print(root_dir,images_dir)

def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
    print(string)
def sersic(r, mueff, reff, n):
    return mueff + 2.5*(0.868*n-0.142)*((r/reff)**(1./n) - 1)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Collect images of all resolved halos from a given simulation. Images will be generated across all orientations.')
parser.add_argument('-f','--feedback',choices=['BW','SB','RDZ'],default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',required=True,help='Simulation to analyze')
parser.add_argument('-i','--image',action='store_true',help='Generate images in addition to profile data')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-o','--overwrite',action='store_true',help='Overwrite existing images')
parser.add_argument('-v','--verbose',action='store_true',help='Print halo IDs being analyzed')
args = parser.parse_args()


#SimInfo = pickle.load(open(root_dir/f'SimulationInfo.{args.feedback}.pickle','rb'))
SimInfo = pickle.load(open(code_dir / f'SimulationInfo.{args.feedback}.pickle', 'rb'))

simpath = SimInfo[args.simulation]['path']
halos = SimInfo[args.simulation]['goodhalos']
dx,dy = 30,30 #Angular resolution of rotations
#Check if all halos in sim have been completed
image_path = images_dir / f'{args.simulation}.{args.feedback}' / str(halos[-1])
if f'{halos[-1]}.x{180-dx:03d}.y{360-dy}.png' in os.listdir(image_path) and not args.overwrite:
    print(f'{args.simulation}.{args.feedback} completed.')
    sys.exit(0)

print(f'Loading {args.simulation}')
tstart = time.time()
if os.path.exists(simpath):
    try:
        # Attempt to load the simulation
        sim = pynbody.load(simpath)
        print("Simulation loaded successfully.")
    except Exception as e:
        # Handle exceptions raised by pynbody.load
        print(f"Failed to load simulation: {e}")
else:
    print(f"Path does not exist: {simpath}")

sim.physical_units()
h = sim.halos()
if args.image: ImageData = pymp.shared.dict()
SBData = pymp.shared.dict()
myprint(f'{args.simulation} loaded.',clear=True)


def generate_image_and_sb(halo,hid, xrotation, yrotation, dx, dy, width, ImageSpace, args, images_dir, SBData):
    current_sb = {}
    current_image = {}
    halo.rotate_x(xrotation)
    halo.rotate_x(yrotation)

    # Find V-band SB at Reff
    current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}'] = {}
    current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['Rhalf'] = Rhalf
    try:
        prof = pynbody.analysis.profile.Profile(halo.s, type='lin', min=.25, max=5*Rhalf, ndim=2, nbins=int((5*Rhalf)/0.1))

        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['sb,v'] = prof['sb,v']
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['v_lum_den'] = prof['v_lum_den']
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['rbins'] = prof['rbins']
        binarea, binlum = prof._binsize.in_units('pc^2'), 10.0 ** (-0.4 * prof['magnitudes,v'])
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['lum_den'] = binlum / binarea
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['mags,v'] = prof['magnitudes,v']
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['binarea'] = binarea
    except:
        if args.verbose:
            print(f'Error in profiling {hid} x{xrotation*dx:03d}y{yrotation*dy:03d}')
            #print traceback
            print(traceback.format_exc())
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['sb,v'] = np.NaN
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['v_lum_den'] = np.NaN
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['rbins'] = np.NaN
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['lum_den'] = np.NaN
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['mags,v'] = np.NaN
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['binarea'] = np.NaN

    if not isinstance(current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['sb,v'], float):
        try:
            vband = prof['sb,v']
            smooth = np.nanmean(np.pad(vband.astype(float), (0, 3-vband.size%3), mode='constant', constant_values=np.nan).reshape(-1, 3), axis=1)
            x = np.arange(len(smooth)) * 0.3 + 0.15
            x[0] = .05
            if True in np.isnan(smooth):
                x = np.delete(x, np.where(np.isnan(smooth)==True))
                y = np.delete(smooth, np.where(np.isnan(smooth)==True))
            else:
                y = smooth
            r0 = x[int(len(x)/2)]
            m0 = np.mean(y[:3])
            par, ign = curve_fit(sersic, x, y, p0=(m0, r0, 1), bounds=([10, 0, 0.5], [40, 100, 16.5]))
            current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['Reff'] = par[1]
        except:
            if args.verbose:
                print(f'Error in fitting {hid} x{xrotation*dx:03d}y{yrotation*dy:03d}')
                #print traceback
                print(traceback.format_exc())
            current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['Reff'] = np.NaN
    else:
        current_sb[f'x{xrotation*dx:03d}y{yrotation*dy:03d}']['Reff'] = np.NaN

    if args.image:
        # Generate V-band SB image
        f = plt.figure(frameon=False)
        f.set_size_inches(10, 10)
        ax = plt.Axes(f, [0., 0., 1., 1.])
        ax.set_axis_off()
        f.add_axes(ax)
        im = image(sim[ImageSpace].s, qty='v_lum_den', width=width, subplot=ax, units='kpc^-2', resolution=1000, show_cbar=False)
        figpath = images_dir / f'{args.simulation}.{args.feedback}/{hid}/{hid}.x{xrotation*dx:03d}.y{yrotation*dy:03d}.png'
        f.savefig(figpath)
        plt.close()
        current_image[f'x{xrotation*dx:03d}y{yrotation*dy:03d}'] = im


    return current_sb

prog=pymp.shared.array((1,),dtype=int)
print(f'\tGenerating images: {round(prog[0]/len(halos)*100,2)}%')
x_rotation = np.arange(0,180,dx)
y_rotation = np.arange(0,360,dy)
# create 2d array to hold all pairs of x and y rotations
rotations = np.array(np.meshgrid(x_rotation, y_rotation)).T.reshape(-1, 2)


for i in range(len(halos)):
    t_start_current = time.time()
    if args.verbose: print(f'\tAnalyzing {halos[i]}...')
    hid = halos[i]
    halo = h[hid]
    #print(halo.star.loadable_keys())
    pynbody.analysis.angmom.faceon(halo)
    Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
    width = 6*Rhalf
    ImageSpace = pynbody.filt.Sphere(width*np.sqrt(2)*1.01)
    current_image = {}
    current_sb = {}
    #parralelize over all unique combinations of x and y rotations
    #with pymp.Parallel(args.numproc) as p:
         #for rotation in p.iterate(rotations):
    for rotation in rotations:
            xrotation = rotation[0]
            yrotation = rotation[1]

            #Find V-band SB at Reff
            current_sb[f'x{xrotation:03d}y{yrotation:03d}'] = generate_image_and_sb(halo,hid, xrotation, yrotation,
                                    dx, dy, width, ImageSpace, args,images_dir, SBData)
            break
            prog[0]+=1
    SBData[str(hid)] = current_sb
    t_end_current = time.time()
    if args.verbose: print(f'\t\t{hid} done in {round((t_end_current-t_start_current)/60,2)} minutes.')

image_file_path = root_dir / f'Data/{args.simulation}.{args.feedback}.Images.pickle'
sb_file_path = root_dir / f'Data/{args.simulation}.{args.feedback}.Profiles.pickle'

try:
    with image_file_path.open('rb') as file:
        ImageFile = pickle.load(file)
    with sb_file_path.open('rb') as file:
        SBFile = pickle.load(file)
except:
    ImageFile,SBFile = {},{}

data_dir = root_dir / 'Data'

# Loop through halos and update dictionaries
for halo in halos:
    if args.image:
        ImageFile[str(halo)] = ImageData[str(halo)]
    SBFile[str(halo)] = SBData[str(halo)]

# Save ImageFile, if applicable
if args.image:
    image_file_path = data_dir / f'{args.simulation}.{args.feedback}.Images.pickle'
    with image_file_path.open('wb') as file:
        pickle.dump(ImageFile, file)

# Save SBFile
sb_file_path = data_dir / f'{args.simulation}.{args.feedback}.Profiles.pickle'
with sb_file_path.open('wb') as file:
    pickle.dump(SBFile, file)

# Timing and logging
tstop = time.time()
myprint(f'\t{args.simulation} imaged in {round((tstop-tstart)/60,2)} minutes.', clear=True)

