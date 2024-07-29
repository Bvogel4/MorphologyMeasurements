import argparse
import os
import pickle
import pymp
import pynbody
import sys
import time
import warnings
import numpy as np
import matplotlib.pylab as plt
from pynbody.plot.sph import image
from scipy.optimize import curve_fit
from pathlib import Path
import traceback
import gc

# Global variables and constants
dx, dy = 30, 30  # Angular resolution of rotations


def setup_directories():
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent.parent
    code_dir = root_dir / 'Code'
    Siminfo_dir = code_dir / 'PickleFiles'
    images_dir = root_dir / 'Figures' / 'Images'
    return root_dir, images_dir, Siminfo_dir


def myprint(string, clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
    print(string)


def sersic(r, mueff, reff, n):
    return mueff + 2.5 * (0.868 * n - 0.142) * ((r / reff) ** (1. / n) - 1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Collect images of all resolved halos from a given simulation. Images will be generated across all orientations.')
    parser.add_argument('-f', '--feedback', default='BW', help='Feedback Model')
    parser.add_argument('-s', '--simulation', required=True, help='Simulation to analyze')
    parser.add_argument('-i', '--image', action='store_true', help='Generate images in addition to profile data')
    parser.add_argument('-n', '--numproc', type=int, required=True, help='Number of processors to use')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing images')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print halo IDs being analyzed')
    return parser.parse_args()


def load_simulation(simpath):
    print(f'Loading simulation from {simpath}')
    if os.path.exists(simpath):
        try:
            sim = pynbody.load(simpath)
            print("Simulation loaded successfully.")
            return sim
        except Exception as e:
            print(f"Failed to load simulation: {e}")
            return None
    else:
        print(f"Path does not exist: {simpath}")
        return None


def process_halo(halo, hid, args, image_path):
    pynbody.analysis.angmom.faceon(halo)
    Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
    width = 6 * Rhalf
    ImageSpace = pynbody.filt.Sphere(width * np.sqrt(2) * 1.01)
    current_image = {}
    current_sb = {}

    for xrotation in range(0, 180, dx):
        with pymp.Parallel(args.numproc) as p:
            for yrotation in p.range(0, 360, dy):
                key = f'x{xrotation:03d}y{yrotation:03d}'
                orientation_data = process_orientation(halo, Rhalf, ImageSpace, width, xrotation, yrotation, args,
                                                       image_path, hid)
                current_sb[key] = {k: v for k, v in orientation_data.items() if k != 'image'}
                if args.image:
                    current_image[key] = orientation_data.get('image')

                halo.rotate_y(dy)
        halo.rotate_x(dx)

    return current_image, current_sb


def process_orientation(halo, Rhalf, ImageSpace, width, xrotation, yrotation, args, image_path, hid):
    orientation_data = {'Rhalf': Rhalf}
    prof = pynbody.analysis.profile.Profile(halo.s, type='lin', min=.25, max=5 * Rhalf, ndim=2,
                                            nbins=int((5 * Rhalf) / 0.1))

    try:
        orientation_data.update({
            'sb,v': prof['sb,v'].copy(),
            'v_lum_den': prof['v_lum_den'].copy(),
            'rbins': prof['rbins'].copy(),
            'lum_den': (10.0 ** (-0.4 * prof['magnitudes,v']) / prof._binsize.in_units('pc^2')).copy(),
            'mags,v': prof['magnitudes,v'].copy(),
            'binarea': prof._binsize.in_units('pc^2').copy()
        })
        orientation_data['Reff'] = fit_sersic_profile(prof)
    except Exception as e:
        if args.verbose:
            print(f'Error in fitting {hid} x{xrotation:03d}y{yrotation:03d}')
            print(traceback.format_exc())
        orientation_data.update(
            {k: np.NaN for k in ['sb,v', 'v_lum_den', 'rbins', 'lum_den', 'mags,v', 'binarea', 'Reff']})

    if args.image:
        orientation_data['image'] = generate_image(halo[ImageSpace].s, width, image_path, hid, xrotation, yrotation)

    del prof
    gc.collect()
    return orientation_data


def fit_sersic_profile(prof):
    vband = prof['sb,v']
    smooth = np.nanmean(
        np.pad(vband.astype(float), (0, 3 - vband.size % 3), mode='constant', constant_values=np.nan).reshape(-1, 3),
        axis=1)
    x = np.arange(len(smooth)) * 0.3 + 0.15
    x[0] = .05
    y = smooth[~np.isnan(smooth)]
    x = x[~np.isnan(smooth)]
    r0 = x[int(len(x) / 2)]
    m0 = np.mean(y[:3])
    par, _ = curve_fit(sersic, x, y, p0=(m0, r0, 1), bounds=([10, 0, 0.5], [40, 100, 16.5]))
    return par[1]


def generate_image(stars, width, image_path, hid, xrotation, yrotation):
    f = plt.figure(frameon=False)
    f.set_size_inches(10, 10)
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.set_axis_off()
    f.add_axes(ax)
    im = image(stars, qty='v_lum_den', width=width, subplot=ax, units='kpc^-2', resolution=1000, show_cbar=False)
    figpath = image_path / f'{hid}/{hid}.x{xrotation:03d}.y{yrotation:03d}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    f.savefig(figpath)
    plt.close(f)
    return im


def save_data(root_dir, args, ImageData, SBData):
    data_dir = root_dir / 'Data'
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_file_path = data_dir / f'{args.simulation}.{args.feedback}.Images.pickle'
        try:
            with image_file_path.open('rb') as file:
                ImageFile = pickle.load(file)
        except:
            ImageFile = {}
        ImageFile.update(ImageData)
        with image_file_path.open('wb') as file:
            pickle.dump(ImageFile, file)
        del ImageFile
        gc.collect()

    sb_file_path = data_dir / f'{args.simulation}.{args.feedback}.Profiles.pickle'
    try:
        with sb_file_path.open('rb') as file:
            SBFile = pickle.load(file)
    except:
        SBFile = {}
    SBFile.update(SBData)
    with sb_file_path.open('wb') as file:
        pickle.dump(SBFile, file)


def main():
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    root_dir, images_dir, Siminfo_dir = setup_directories()

    with open(Siminfo_dir / f'SimulationInfo.{args.feedback}.pickle', 'rb') as f:
        SimInfo = pickle.load(f)

    simpath = SimInfo[args.simulation]['path']
    halos = SimInfo[args.simulation]['goodhalos']

    image_path = images_dir / f'{args.simulation}.{args.feedback}'
    if not args.overwrite and (image_path / str(halos[-1]) / f'{halos[-1]}.x{180 - dx:03d}.y{360 - dy}.png').exists():
        print(f'{args.simulation}.{args.feedback} completed.')
        sys.exit(0)

    sim = load_simulation(simpath)
    if sim is None:
        sys.exit(1)

    sim.physical_units()
    h = sim.halos()

    prog = pymp.shared.array((1,), dtype=int)
    print(f'\tGenerating images: {round(prog[0] / len(halos) * 100, 2)}%')

    tstart = time.time()

    for i in range(len(halos)):
        t_start_current = time.time()
        if args.verbose:
            print(f'\tAnalyzing {halos[i]}...')
        hid = halos[i]
        halo = h[hid]

        current_image, current_sb = process_halo(halo, hid, args, image_path)

        # Save data for each halo immediately
        save_data(root_dir, args, {str(hid): current_image}, {str(hid): current_sb})

        del current_image, current_sb
        gc.collect()

        prog[0] += 72  # 6 x-rotations * 12 y-rotations
        if not args.verbose:
            myprint(f'\tGenerating images: {round(prog[0] / (len(halos) * 72) * 100, 2)}%', clear=True)

        t_end_current = time.time()
        if args.verbose:
            print(f'\t\t{hid} done in {round((t_end_current - t_start_current) / 60, 2)} minutes.')

    tstop = time.time()
    myprint(f'\t{args.simulation} imaged in {round((tstop - tstart) / 60, 2)} minutes.', clear=True)

if __name__ == "__main__":
    main()