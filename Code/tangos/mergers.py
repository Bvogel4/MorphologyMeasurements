import numpy as np
import tangos
from tangos.examples.mergers import get_mergers_of_major_progenitor

def calculate_total_merger_ratio(merger_ratios):
    """
    Calculate the total ratio Ma/M_total given a list of merger ratios Ma/Mi.

    :param merger_ratios: List of merger ratios (Ma/Mi) where Mi is always the smaller galaxy
    :return: The total ratio Ma/M_total
    """
    # Calculate Ma (assuming Mi = 1 for each merger)
    merger_ratios = np.array(merger_ratios)
    # all mergers should be greater than 1, but if not, we simply invert the value
    merger_ratios[merger_ratios < 1] = 1 / merger_ratios[merger_ratios < 1]

    Ma = 1
    M_total = Ma + sum(1 / (merger_ratios))
    total_ratio = Ma / M_total
    return 1- total_ratio



def get_merger_dict(redshifts, ratios, snaps, times):
    merger_dict = {'redshift': [], 'ratio': [], 'time': [], 'snap': []}
    for redshift, ratio, time, snap in zip(redshifts, ratios, times, snaps):

        if redshift not in merger_dict['redshift']:
            merger_ratios = []
            merger_dict['redshift'].append(redshift)
            i = merger_dict['redshift'].index(redshift)
            merger_dict['ratio'].append('')
            merger_ratios.append(ratio)
            merger_dict['time'].append(time)
            merger_dict['snap'].append(snap)
        else:
            merger_ratios.append(ratio)

        #add total ratio once for each redshfit to the dictionary

        merger_dict['ratio'][i] = calculate_total_merger_ratio(merger_ratios)

            # if ratio > merger_dict['ratio'][i]:
            #     merger_dict['ratio'][i] = ratio

    return merger_dict




def largest_merger_ratios(sim):
    # recreate this kind of dict {1: {'ratio': [0.018803793185661853], 'time': [4.298635569270686], 'snap': ['001280'], 'grps': [SimArray([ 1, 53])]}, 2: {'ratio': [0.11465486785842333, 0.08268792674930038], 'time': [3.869872473372461, 4.298635569270686], 'snap': ['001152', '001280'], 'grps': [SimArray([ 2, 14]), SimArray([ 2, 17])]}, 3: {'ratio': [0.09404809180521696, 0.14801435041365774, 0.030147238179703544, 0.03172785482936007, 0.017176853819010903, 0.018749636316129668, 0.010836041713394936], 'time': [5.584924856992905, 6.013687952908708, 6.87121414472478, 7.299977240632852, 7.728740336534077, 8.157503432432808, 10.301318911979886], 'snap': ['001664', '001792', '002048', '002176', '002304', '002432', '003072'], 'grps': [SimArray([ 3, 18]), SimArray([ 3, 14]), SimArray([ 3, 29]), SimArray([ 3, 49]), SimArray([ 3, 80]), SimArray([ 3, 75]), SimArray([ 4, 93])]}, 5: {'ratio': [0.04178299208546208], 'time': [2.5835831856697413], 'snap': ['000768'], 'grps': [SimArray([ 2, 37])]}, 6: {'ratio': [], 'time': [], 'snap': [], 'grps': []}, 7: {'ratio': [], 'time': [], 'snap': [], 'grps': []}, 10: {'ratio': [], 'time': [], 'snap': [], 'grps': []}, 11: {'ratio': [], 'time': [], 'snap': [], 'grps': []}, 13: {'ratio': [], 'time': [], 'snap': [], 'grps': []}}

    # get arrays for ratios, times, snaps, and grps for each halo
    merger_ratios = {}
    timesteps = sim.timesteps
    # create an array holding snaps, times, and redshifts for each timestep
    # size of array is number of timesteps by 3
    snaps = []

    for ts in timesteps:
        snap = ts.extension.split('.')[-1].split('/')[0]
        snaps.append(snap)
    #snaps = snaps[::-1]

    times_all = timesteps[0].time_gyr_cascade
    redshifts_all = timesteps[0].redshift_cascade


    #print('Only processing halo 1')
    if sim.basename == 'r1023.romulus25.3072g1HsbBH':
        halo = sim.timesteps[-1].halos[2]
    elif sim.basename == 'r707.romulus25.3072g1HsbBH':
        halo = sim.timesteps[-1].halos[2]
    elif sim.basename == 'r968.romulus25.3072g1HsbBH':
        halo = sim.timesteps[-1].halos[3]
    else:
        halo = sim.timesteps[-1].halos[1]
    redshifts, ratios, halos = get_mergers_of_major_progenitor(halo)

    #remove index of times_all and snaps that are not in redshifts
    #redshifts_all has all redshifts, but redshifts only has the redshifts of the mergers
    #so we need to remove the indices of the redshifts that are not in redshifts_all

    #get indices of redshifts that are in redshifts_all
    indices = [redshifts_all.index(redshift) for redshift in redshifts]
    times = [times_all[i] for i in indices]
    snaps = [snaps[i] for i in indices]
    redshift_check = [redshifts_all[i] for i in indices]
    #check if redshift is the same as redshift_check

    for i in [0, 1, 2,-2, -1]:
        print(f'Redshifts: {redshifts[i]:.4f}, {redshift_check[i]:.4f}')


    print(len(redshifts), len(snaps), len(times))
    for redshift, ratio, h in zip(redshifts[0:i], ratios[0:i], halos[0:i]):
        print(f"Redshift: {redshift}, Ratio: {ratio}")
        #at some point maybe add grps to the dictionary
    # get ratios for each redshift and store in an array
    if len(redshifts)>0:
        sim_dict ={1: get_merger_dict(redshifts, ratios, snaps, times)}
    else:
        return

    return sim_dict


def get_grps(halos):
    #halos is a list of tangos halo objects, the function returns a list of the halo numbers from the mergers
    grps = []
