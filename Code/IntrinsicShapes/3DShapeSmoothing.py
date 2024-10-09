import os,pickle
import matplotlib.pylab as plt
from scipy.interpolate import UnivariateSpline as Smooth
import numpy as np
import argparse
import traceback
import pathlib
import sys
parser = argparse.ArgumentParser(description='Smoothly interpolate Radial Bins for Shapes')
args = parser.parse_args()
verbose = False


from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


def gaussian_smooth(x, y, distance=0.2, sigma=6):
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Create a new array for smoothed y values
    y_smooth = np.zeros_like(y_sorted)

    for i, xi in enumerate(x_sorted):
        # Find indices of points within the specified distance
        mask = np.abs(x_sorted - xi) <= distance
        x_window = x_sorted[mask]
        y_window = y_sorted[mask]

        # Apply Gaussian smoothing to the window
        if len(y_window) > 1:
            y_smooth[i] = gaussian_filter1d(y_window, sigma)[0]
        else:
            y_smooth[i] = y_window[0]

    # Clip values between 0 and 1
    y_smooth = np.clip(y_smooth, 0, 1)

    # Create an interpolation function
    f = interp1d(x_sorted, y_smooth, kind='cubic', fill_value='extrapolate')

    # Display a warning if y values are not between 0 and 1
    if np.any(y_smooth < 0) or np.any(y_smooth > 1):
        print(
            'Warning: Some smoothed values are not in the physical range [0, 1]')

    # # Return a callable function that clips the output between 0 and 1
    # return lambda x_new: np.clip(f(x_new), 0, 1)
    return f


def nanfunction(x):
    return x*0 +10

from scipy.interpolate import UnivariateSpline
def smooth_and_filter_data(rbins, ba, ca, k=5, s_factor=0.01, residual_threshold=0.3, jump_threshold=0.3, jump_percentage=0.2):
    """
    Smooth and filter data, removing outliers and checking for large jumps.

    Parameters:
    rbins: array-like, x coordinates
    ba, ca: array-like, y coordinates for two datasets
    k: int, degree of the smoothing spline (default 5)
    s_factor: float, smoothing factor as a fraction of len(rbins) (default 0.01)
    residual_threshold: float, threshold for removing points based on residuals (default 0.3)
    jump_threshold: float, threshold for detecting large jumps (default 0.3)
    jump_percentage: float, percentage of data allowed to have large jumps (default 0.1)

    Returns:
    rbins, ba, ca: filtered arrays
    ba_s, ca_s: smoothed spline functions
    """


    # Check for large jumps
    ba_diff = np.abs(np.diff(ba))
    ca_diff = np.abs(np.diff(ca))
    large_jumps = (ba_diff > jump_threshold) | (ca_diff > jump_threshold)

    if np.sum(large_jumps) > (len(rbins) - 1) * jump_percentage:
        print(f"Warning: More than {jump_percentage*100}% of the data has successive jumps greater than {jump_threshold}")

        return rbins, ba, ca, nanfunction, nanfunction

    
    def Smooth(x, y, k, s):
        return UnivariateSpline(x, y, k=k, s=s)

    # Initial smoothing
    s = len(rbins) * s_factor
    ba_s, ca_s = Smooth(rbins, ba, k=k, s=s), Smooth(rbins, ca, k=k, s=s)

    # Calculate residuals and create mask
    res_ba = np.abs(ba_s(rbins) - ba)
    res_ca = np.abs(ca_s(rbins) - ca)
    mask = (res_ba < residual_threshold) & (res_ca < residual_threshold)

    # Apply mask
    rbins, ba, ca = rbins[mask], ba[mask], ca[mask]



    # Recalculate the smoothed line
    ba_s, ca_s = Smooth(rbins, ba, k=k, s=s), Smooth(rbins, ca, k=k, s=s)

    return rbins, ba, ca, ba_s, ca_s

# loop,remake=True,False
# while loop:
#     rem = input('Remake Image Directory: Figures/3DShapes/- (y/n): ')
#     if rem in ['y','n']:
#         loop = False
#         if rem=='y': remake = True
# if remake: 
#     #os.system('rmdir -f ../Figures/3DShapes')
#     #os.system('mkdir ../figures/3DShapes')
#     for sim in SimInfo:
#         os.system(f'mkdir ../../Figures/3DShapes/{sim}.{feedback}/')
windows = {
    '3DShapes':{
    'cptmarvel-10':[.65,2],
    'elektra-3':[.4,.8],
    'storm-2':[5,40],
    'storm-8':[3,10],
    'rogue-7':[.8,1],
    'rogue-8':[0,1],
    },
    'DMShapes':{
    
    }
}
sys.path.append(str(pathlib.Path(__file__).parent.parent))
# Import the SimInfoDicts package
from SimInfoDicts.sim_type_name import sim_type_name


for t in ['DMShapes','3DShapes']:
    print(f'Smoothing {t}')
    for feedback, use_sim in sim_type_name.items():
        if use_sim:
            print(f'Calculating masses for {feedback} feedback type.')
            pickle_path = f'../PickleFiles/SimulationInfo.{feedback}.pickle'
            if os.path.exists(pickle_path):
                SimInfo = pickle.load(open(pickle_path, 'rb'))
                #add a check to see if the mass file already exists, if it does, we need to make sure it has all the current sims
                for sim in SimInfo:
                    try:
                        Shapes = pickle.load(open(f'../../Data/{sim}.{feedback}.{t}.pickle','rb'))
                    except FileNotFoundError:
                        print(f'Error loading {sim}.{feedback}.{t}.pickle')
                    try:
                        Profiles = pickle.load(open(f'../../Data/{sim}.{feedback}.Profiles.pickle','rb'))
                    except FileNotFoundError:
                        print(f'Error loading {sim}.{feedback}.Profiles.pickle')
                    for hid in SimInfo[sim]['goodhalos']:
                        try:
                            rbins,ba,ca=Shapes[str(hid)]['rbins'],Shapes[str(hid)]['ba'], Shapes[str(hid)]['ca']
                            reffs = []
                            if len(rbins)>0:
                                #print('halo',hid)

                                for angle in Profiles[str(hid)]:
                                    try:
                                        reffs.append(Profiles[str(hid)][angle]['Reff'])
                                    except IndexError:
                                        if verbose:
                                            print(f'IndexError angle {angle} for halo {hid} in {sim}.{feedback}.Profiles.pickle')
                                
                                f,ax=plt.subplots(1,1,figsize=(6,2))
                                ax.set_xlim([0,max(rbins)])
                                ax.set_ylim([0,1])
                                ax.set_xlabel('R [kpc]',fontsize=15)
                                ax.set_ylabel('Axis Ratio',fontsize=15)
                                ax.tick_params(which='both',labelsize=10)
                                rbins, ba, ca, ba_s, ca_s = smooth_and_filter_data(
                                    rbins, ba, ca)
                                ax.plot(rbins,ba,c='k',label='B/A')
                                ax.plot(rbins,ca,c='k',linestyle='--',label='C/A')
                                # if f'{sim}-{hid}' in windows[t]:
                                #     w = windows[t][f'{sim}-{hid}']
                                #     ba = ba[(rbins<w[0])|(rbins>w[1])]
                                #     ca = ca[(rbins<w[0])|(rbins>w[1])]
                                #     rbins = rbins[(rbins<w[0])|(rbins>w[1])]
                                    # Assuming you have a rough idea of how much you want to smooth the data
            
                                #ba_s,ca_s = Smooth(rbins,ba,k=3),Smooth(rbins,ca,k=3)
                                #ba_s,ca_s = gaussian_smooth(rbins,ba,distance=0.2,sigma=6),gaussian_smooth(rbins,ca,distance=0.2,sigma=6)
                                #if sim == 'r468':
                                # Assuming you have a rough idea of how much you want to smooth the data
                                    #smoothing_factor = 1/50  # some value representing the trade-off between smoothness and fitting
                                    #ba_s, ca_s = Smooth(rbins, ba, k=3, s=smoothing_factor), Smooth(rbins, ca, k=3, s=smoothing_factor)
            
                                    #print('knots',ba_s.get_knots())
            
                                ax.plot(rbins,ba_s(rbins),c='r')
                                ax.plot(rbins,ca_s(rbins),c='r',linestyle='--')
                                #ax.axvspan(min(reffs),max(reffs),color='k',alpha=0.3,label=r'R$_{eff}$')
                                for reff in reffs:
                                    ax.axvline(reff,c='k',alpha=.05)
                                ax.axvline(reffs[0],c='k',alpha=.05,label=r'R$_{eff}$')
                
                                ax.legend(loc='lower right',prop={'size':12})
                                fname='Stars' if t=='3DShapes' else 'Dark'
                                #create folder if it doesn't exist
                                if not os.path.exists(f'../../Figures/3DShapes/{sim}.{feedback}/'):
                                    os.makedirs(f'../../Figures/3DShapes/{sim}.{feedback}/')
                                filename = f'../../Figures/3DShapes/{sim}.{feedback}/{fname}.{hid}.png'
                                print(f'Saving {filename}')
                                f.savefig(filename,bbox_inches='tight',pad_inches=.1)
                                plt.close()
                
                                Shapes[str(hid)]['ba_smooth'] = ba_s
                                Shapes[str(hid)]['ca_smooth'] = ca_s
                        except Exception as e:
                            print(traceback.format_exc())                
                            print(f"An error occurred during smoothing halo {hid}: {e}")
                        pickle.dump(Shapes,open(f'../../Data/{sim}.{feedback}.{t}.pickle','wb'))
print('Smoothing Done')