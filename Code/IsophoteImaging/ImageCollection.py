import argparse,os,pickle,pymp,pynbody,sys,time,warnings
import numpy as np
import matplotlib.pylab as plt
from pynbody.plot.sph import image
from scipy.optimize import curve_fit
import os
from pathlib import Path
import traceback

# Current script directory (IsophoteImaging)
current_script_path = os.path.abspath(__file__)
current_dir = Path(current_script_path).parent

# Root directory (two level up from IsophoteImaging)
root_dir = current_dir.parent.parent
code_dir= root_dir / 'Code'
Siminfo_dir = code_dir / 'PickleFiles'
# Absolute path for Figures/Images directory
images_dir = root_dir / 'Figures' / 'Images'
print(f"__file__: {__file__}")
print(f"current_dir: {current_dir}")
print(f"root_dir: {root_dir}")
print(f"code_dir: {code_dir}")
print(f"Siminfo_dir: {Siminfo_dir}")
print(f"images_dir: {images_dir}")

def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
    print(string)
def sersic(r, mueff, reff, n):
    return mueff + 2.5*(0.868*n-0.142)*((r/reff)**(1./n) - 1)
warnings.filterwarnings("ignore")


def process_rotation(halo, hid, xrotation, yrotation, dx, dy, current_image, current_sb, ImageSpace, width):
    # Find V-band SB at Reff
    current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}'] = {}
    current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['Rhalf'] = Rhalf
    prof = pynbody.analysis.profile.Profile(halo.s, type='lin', min=.25, max=5 * Rhalf, ndim=2,
                                            nbins=int((5 * Rhalf) / 0.1))
    try:
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['sb,v'] = prof['sb,v']
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['v_lum_den'] = prof['v_lum_den']
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['rbins'] = prof['rbins']
        binarea, binlum = prof._binsize.in_units('pc^2'), 10.0 ** (-0.4 * prof['magnitudes,v'])
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['lum_den'] = binlum / binarea
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['mags,v'] = prof['magnitudes,v']
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['binarea'] = binarea
    except:
        if args.verbose:
            print(f'Error in fitting {hid} x{xrotation * dx:03d}y{yrotation * dy:03d}')
            # print traceback
            print(traceback.format_exc())
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['sb,v'] = np.NaN
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['v_lum_den'] = np.NaN
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['rbins'] = np.NaN
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['lum_den'] = np.NaN
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['mags,v'] = np.NaN
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['binarea'] = np.NaN
    if not type(current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['sb,v']) is float:
        try:
            vband = prof['sb,v']
            smooth = np.nanmean(
                np.pad(vband.astype(float), (0, 3 - vband.size % 3), mode='constant', constant_values=np.nan).reshape(
                    -1, 3), axis=1)
            x = np.arange(len(smooth)) * 0.3 + 0.15
            x[0] = .05
            if True in np.isnan(smooth):
                x = np.delete(x, np.where(np.isnan(smooth) == True))
                y = np.delete(smooth, np.where(np.isnan(smooth) == True))
            else:
                y = smooth
            r0 = x[int(len(x) / 2)]
            m0 = np.mean(y[:3])
            par, ign = curve_fit(sersic, x, y, p0=(m0, r0, 1), bounds=([10, 0, 0.5], [40, 100, 16.5]))
            current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['Reff'] = par[1]
        except:
            current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['Reff'] = np.NaN
    else:
        current_sb[f'x{xrotation * dx:03d}y{yrotation * dy:03d}']['Reff'] = np.NaN
    if args.image:
        # Generate V-band SB image
        f = plt.figure(frameon=False)
        f.set_size_inches(10, 10)
        ax = plt.Axes(f, [0., 0., 1., 1.])
        ax.set_axis_off()
        f.add_axes(ax)
        im = image(sim[ImageSpace].s, qty='v_lum_den', width=width, subplot=ax, units='kpc^-2', resolution=1000,
                   show_cbar=False)
        # f.savefig(image_path/f'/{args.simulation}.{args.feedback}/{hid}/{hid}.x{xrotation*dx:03d}.y{yrotation*dy:03d}.png')
        figpath = images_dir / f'{args.simulation}.{args.feedback}/{hid}/{hid}.x{xrotation * dx:03d}.y{yrotation * dy:03d}.png'
        print(figpath)
        f.savefig(figpath)

        plt.close()
        # Store data
        current_image[f'x{xrotation * dx:03d}y{yrotation * dy:03d}'] = im
    # Progress to next orientation

    if not args.verbose: myprint(f'\tGenerating images: {round(prog[0] / (len(halos) * 72) * 100, 2)}%', clear=True)
    return current_image, current_sb

parser = argparse.ArgumentParser(description='Collect images of all resolved halos from a given simulation. Images will be generated across all orientations.')
parser.add_argument('-f','--feedback',default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',required=True,help='Simulation to analyze')
parser.add_argument('-i','--image',action='store_true',help='Generate images in addition to profile data')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-o','--overwrite',action='store_true',help='Overwrite existing images')
parser.add_argument('-v','--verbose',action='store_true',help='Print halo IDs being analyzed')
args = parser.parse_args()


#SimInfo = pickle.load(open(root_dir/f'SimulationInfo.{args.feedback}.pickle','rb'))
SimInfo = pickle.load(open(Siminfo_dir / f'SimulationInfo.{args.feedback}.pickle', 'rb'))

simpath = SimInfo[args.simulation]['path']
halos = SimInfo[args.simulation]['goodhalos']
dx,dy = 90,90 #Angular resolution of rotations
#dx,dy = 30,30 #Angular resolution of rotations
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
if args.image:
    ImageData = pymp.shared.dict()
SBData = pymp.shared.dict()

myprint(f'{args.simulation} loaded.',clear=True)

halos_list = pymp.shared.list()

prog=pymp.shared.array((1,),dtype=int)
print(f'\tGenerating images: {round(prog[0]/len(halos)*100,2)}%')
with pymp.Parallel(args.numproc) as pl:
    for i in pl.xrange(len(halos)):
#for i in range(len(halos)):
        t_start_current = time.time()
        if args.verbose: print(f'\tAnalyzing {halos[i]}...')
        hid = halos[i]
        halo = h[hid]
        pynbody.analysis.angmom.faceon(halo)
        Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
        width = 6*Rhalf
        ImageSpace = pynbody.filt.Sphere(width*np.sqrt(2)*1.01)
        current_image = {}
        current_sb = {}
        xrotation = 0
        while xrotation*dx<180:
            yrotation = 0
            while yrotation*dy<360:
                current_sb, current_image = process_rotation(halo, hid, xrotation, yrotation, dx, dy, current_image, current_sb, ImageSpace, width)
                if xrotation == 0 and yrotation == 0:
                    halos_list.append(hid)
                prog[0] += 1
                halo.rotate_y(dy)
                yrotation+=1
            halo.rotate_x(dx)
            xrotation+=1
        if args.image:
            ImageData[str(hid)] = current_image
        SBData[str(hid)] = current_sb
        t_end_current = time.time()
        if args.verbose: print(f'\t\t{hid} done in {round((t_end_current-t_start_current)/60,2)} minutes.')


import pickle
from pathlib import Path


def load_data(file_path):
    try:
        with file_path.open('rb') as file:
            return pickle.load(file)
    except (FileNotFoundError, pickle.PickleError):
        return {}


def save_data(data, file_path):
    with file_path.open('wb') as file:
        pickle.dump(data, file)

print(SBData)
# Convert shared dict to regular dict

image_data = dict(ImageData)
sb_data = dict(SBData)
print(sb_data)

data_dir = root_dir / 'Data'
image_file_path = data_dir / f'{args.simulation}.{args.feedback}.Images.pickle'
sb_file_path = data_dir / f'{args.simulation}.{args.feedback}.Profiles.pickle'

image_file = load_data(image_file_path) if not args.overwrite else {}
sb_file = load_data(sb_file_path) if not args.overwrite else {}

# Loop through halos and update dictionaries
for halo in halos:
    halo_str = str(halo)
    if halo_str not in sb_data:
        print(f'{halo} not in SBData')
        print(f'{halo} has been processed? {halo in halos_list}')
    else:
        print(f'{halo} in SBData')

        if args.image:
            try:
                image_file[halo_str] = image_data[halo_str]
            except KeyError:
                print(f'Error in Image dict {args.simulation} {halo}')
        try:
            sb_file[halo_str] = sb_data[halo_str]
        except KeyError:
            print(f'Error in Profile dict {args.simulation} {halo}')

# Save ImageFile, if applicable
if args.image:
    save_data(image_file, image_file_path)

# Save SBFile
save_data(sb_file, sb_file_path)

# Timing and logging
tstop = time.time()
myprint(f'\t{args.simulation} imaged in {round((tstop-tstart)/60,2)} minutes.', clear=True)
