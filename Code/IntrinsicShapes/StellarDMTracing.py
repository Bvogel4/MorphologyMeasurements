import pickle
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
import traceback
import matplotlib.gridspec as gridspec
import seaborn as sns
from astropy import units as u
from astropy import constants as const
from scipy.optimize import curve_fit
import warnings


'''
Obviously need SimInfo pickles from Config_dir
Update the newest Merians from update_merians.ipynb
what needs to be run to make this work
Reff and Profiles from ImageCollection, dont think I need to run isophote masking, except maybe for x00y00
StShapes, DMShapes from 3DShapes
smoothed shapes from 3DShapesSmoothing
halotypes from halotypes.py, in the case of Merians they are all simply centrals
masses from get_masses.py
try/except blocks should catch any mismatches, so don't need to worry too much about everything being perfect. 
'''
#change colors to be assigned bases on dictionary
colors = {'MerianCDM':'r','BWMDC':'b'}


def power_law(x, a, b, c):
    return a * x**b + c


def fit_and_plot(r, m, Reff, sim, hid):
    merging_galaxies = {'r615': 1, 'r618': 1, 'elektra': 3, 'storm': 4, 'rogue': 1, 'h148': 3}

    # Sort arrays by r
    r, m = zip(*sorted(zip(r, m)))
    r = np.array(r)
    m = np.array(m)

    # Fit the power-law function to the data points
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            params, pcov = curve_fit(power_law, r, m, p0=[1e8, 0.5, 0], maxfev=10000)
    except (RuntimeError) as e:
        print(f'Warning: Error fitting power-law for sim {sim} halo {hid}: {str(e)}')
        return None, None

    # Generate fitted data
    rfit = np.linspace(min(r), max(r), 100)
    m_fit = power_law(rfit, *params)

    # Guess mass at x*Reff
    x = 10
    m_vir_value = power_law(x * Reff, *params)

    # if sim in merging_galaxies.keys() and hid == merging_galaxies[sim]:
    #     # Plotting
    #     fig, ax = plt.subplots()
    #     ax.scatter(r, m, label='Data')
    #     ax.plot(rfit, m_fit, label=f'Fitted Curve (a={params[0]:.2e}, b={params[1]:.2f}, c={params[2]:.2e})')
    #     ax.scatter(x * Reff, m_vir_value, label='5*Reff')
    #     ax.set_xlabel('Radius')
    #     ax.set_ylabel('Mass')
    #     ax.legend()
    #     plt.title(f'Power-law fit for sim={sim}, hid={hid}')
    #     plt.show()

    return params, m_vir_value



def calculate_dynamical_time(r_vir, M_halo):
    r_vir = r_vir * u.kpc
    M_halo = M_halo * u.solMass
    t_dyn = np.sqrt(r_vir ** 3 / (const.G * M_halo))
    return t_dyn.to(u.Gyr).value


def T(ba, ca):
    return ((1 - ba ** 2) / (1 - ca ** 2))


def LoadSimData(feedbacks, reff_multi=1, return_sims=False, fixed_r=False):

    # Initialize paths
    SimFilePath = []
    MassPath = []
    HaloTypePath = []
    merger_ratios = []


    for feedback in feedbacks:
        SimFilePath.append(f'../PickleFiles/SimulationInfo.{feedback}.pickle')
        MassPath.append(f'../../Data/BasicData/{feedback}.Masses.pickle')
        HaloTypePath.append(f'../../Data/BasicData/HaloTypes.{feedback}.txt')
    #MassPath[0] = f'../../Data/BasicData/Marvel_DCJL.Masses.pickle'
    print(SimFilePath)
    T_s, T_d = [], []
    B_s, C_s, B_d, C_d = [], [], [], []
    Es_smoothed, Ed_smoothed = [], []
    masses, htype, mb, reff, m_vir = [], [], [], [], []
    sims, hids, feedback_type = [], [], []
    merger = []
    mb_reffs = []
    mb_10rvirs = []
    rvir = []
    diffs_at_Reff = []
    jz_jcirc_avgs = []
    t_dyn = []

    for i in range(len(SimFilePath)):
        SimInfo = pickle.load(open(SimFilePath[i], 'rb'))
        try:
            mass_data = pickle.load(open(MassPath[i], 'rb'))
        except:
            print(f'Error loading mass data for {feedbacks[i]}')
        #print(mass_data)
        


        # try:
        #     merger_ratios = pickle.load(open(f'../../Data/BasicData/Mergers.{feedbacks[i]}.pickle','rb'))
        # except:
        #     print(f'Error loading merger ratios for {feedbacks[i]}')
        #for merger_ratios seperator is a comma
        try:
            dataframe = pd.read_csv(HaloTypePath[i], sep=r'\s+')
            types = {}
            for _, row in dataframe.iterrows():
                types[(row['Volume'], str(row['HaloGRP@z0']))] = row['HaloType']
        except:
            print(f'Error loading halo types for {feedbacks[i]}')
        #print(SimInfo)
        

        for sim in SimInfo:
            try:
                StShapes = pickle.load(open(f'../../Data/{sim}.{feedbacks[i]}.3DShapes.pickle', 'rb'))
                DMShapes = pickle.load(open(f'../../Data/{sim}.{feedbacks[i]}.DMShapes.pickle', 'rb'))
                Profiles = pickle.load(open(f'../../Data/{sim}.{feedbacks[i]}.Profiles.pickle', 'rb'))
                # if feedback == 'SBMarvel':
                #     Profiles = pickle.load(
                #         open(f'../../Data/{sim}.BWMDC.Profiles.pickle',
                #              'rb'))
            except:
                print(traceback.format_exc())
                print(f'error loading data in in sim {sim}')
                continue

            for hid in SimInfo[sim]['goodhalos']:
                if sim == 'cptmarvel' and hid == 7:
                    continue
                if sim == 'elektra' and hid == 10:
                    continue
                #if sim == 'r618' or sim == 'r618.romulus25si2s50v35':
                    #print(f'ignoring sim {sim}')
                    #continue

                #print(f'Loading data for sim {sim} halo {hid}')

                try:

                    rbins, rd, ba_s_func, ca_s_func, ba_d_func, ca_d_func = [
                        StShapes[(hid)]['rbins'], DMShapes[(hid)]['rbins'],
                        StShapes[(hid)]['ba_smooth'], StShapes[(hid)]['ca_smooth'],
                        DMShapes[(hid)]['ba_smooth'], DMShapes[(hid)]['ca_smooth']
                    ]
                    Reff = Profiles[str(hid)]['x000y000']['Reff'] * reff_multi
                    if fixed_r:
                        Reff = reff_multi

                    # Get values for the current halo
                    try:
                        ba_s_value = ba_s_func(Reff)
                        ca_s_value = ca_s_func(Reff)
                        ba_d_value = ba_d_func(Reff)
                        ca_d_value = ca_d_func(Reff)
                    except:
                        print(f'Error loading shape data for sim {sim} halo {hid}')
                        ba_s_value, ca_s_value, ba_d_value, ca_d_value = np.nan, np.nan, np.nan, np.nan

                    try:
                        assert ba_s_value > 0 and ca_s_value > 0 and ba_d_value > 0 and ca_d_value > 0 and ba_s_value <= 1 and ca_s_value <= 1 and ba_d_value <= 1 and ca_d_value <= 1
                    except AssertionError:
                        print(f'sim {sim} halo {hid} has invalid shape values')
                        print(f'ba_s: {ba_s_value}, ca_s: {ca_s_value}, ba_d: {ba_d_value}, ca_d: {ca_d_value}')
                        ba_s_value, ca_s_value, ba_d_value, ca_d_value = np.nan, np.nan, np.nan, np.nan
                        

                    try:
                        diff_at_Reff = StShapes[(hid)]['diffs_at_Reff']
                    except KeyError:
                        print(f'Error loading diff_at_Reff for sim {sim} halo {hid}')
                    #diff_at_Reff = np.nan

                    # if ba_s_value < 0.25:
                    #     if ca_s_value < 0.25:
                    #         print(f'stellar b/a and c/a unusually low in sim {sim} halo {hid}')
                    T_s_value = T(ba_s_value, ca_s_value)
                    T_d_value = T(ba_d_value, ca_d_value)

                    try:
                        m_vir_value = mass_data[sim][str(hid)]['Mvir']
                        sm = mass_data[sim][str(hid)]['Mstar']
                        if sm == 0:
                            print(f'Mstar is 0 in sim {sim} halo {hid}')
                            masses_value = np.nan
                        else:
                            masses_value = np.log10(sm)
                        #mb_value = (mass_data[sim][str(hid)]['Mb/Mtot_within_reff'])
                        #print(mass_data[sim][str(hid)].keys())
                        
                        mb_value = (mass_data[sim][str(hid)]['Mb/Mtot'])
                        mb_reff = (mass_data[sim][str(hid)]['Mb/Mtot_within_reff'])
                        mb_10rvir = (mass_data[sim][str(hid)]['Mb/Mtot_within_tenth_rvir'])
                        rvir_value = mass_data[sim][str(hid)]['Rvir']
                        #calculate dynamical time at effective raidius, virial radius, and 10th virial radius
                        t_dyn_vir = calculate_dynamical_time(rvir_value, m_vir_value)
                        m_vir_eff = mass_data[sim][str(hid)]['Mvir_within_reff']
                        m_vir_within_10rvir = mass_data[sim][str(hid)]['Mvir_within_tenth_rvir']
                        t_dyn_eff = calculate_dynamical_time(Reff, m_vir_eff)
                        t_dyn_10rvir = calculate_dynamical_time(rvir_value*.1, m_vir_within_10rvir)
                        #create arrays holding rvir, reff, and 10th rvir




                        # Plot for merging galaxies


                        # Example data points
                        r = np.array([rvir_value, Reff, rvir_value * 0.1])
                        m = np.array([m_vir_value, m_vir_eff, m_vir_within_10rvir])
                        #sort arrays by r
                        r,m = zip(*sorted(zip(r,m)))
                        params, m_vir_at_5Reff = fit_and_plot(r, m,Reff, sim, hid)




                        t_dyn_value = calculate_dynamical_time(5*Reff,m_vir_value)

                    


                        try:
                            jz_jcirc_avg = mass_data[sim][str(hid)]['jz_jcirc_avg']
                        except:
                            print(f'Error loading jz_jcirc_avg for sim {sim} halo {hid}')
                            jx_jcirc_avg = np.nan

                        # print(f'Loading masses for sim {sim} halo {hid}')
                        # print(#f"Reff: {mass_data[sim][str(hid)]['Reff']:.1f}, "
                        #       #f"Rvir: {mass_data[sim][str(hid)]['Rvir']:.0f}, "
                        #       #f"Reff/Rvir: {mass_data[sim][str(hid)]['Reff']/mass_data[sim][str(hid)]['Rvir']:.3f}, "
                        #       f"Mb/Mtot: {mass_data[sim][str(hid)]['Mb/Mtot']:.3f}, "
                        #       f"Mb/Mtot_within_reff: {mass_data[sim][str(hid)]['Mb/Mtot_within_reff']:.3f}, "
                        #       #f"Mb/Mtot_within_reff/Mb/Mtot: {mass_data[sim][str(hid)]['Mb/Mtot_within_reff']/mass_data[sim][str(hid)]['Mb/Mtot']:.3f}",
                        #       #f"Mvir_within_10rvir: {mass_data[sim][str(hid)]['Mvir_within_10rvir']:.3f}",
                        #       #f"Mstar_within_10rvir: {mass_data[sim][str(hid)]['Mstar_within_10rvir']:.3f}",
                        #       #f"Mb_within_10rvir: {mass_data[sim][str(hid)]['Mb_within_10rvir']:.3f}",
                        #       f"Mb/Mtot_within_10rvir: {mass_data[sim][str(hid)]['Mb/Mtot_within_10rvir']:.3f}")
                    except:
                        print(f'Error loading masses for sim {sim} halo {hid}')
                        print(traceback.format_exc())
                        m_vir_value = np.nan
                        masses_value = np.nan
                        mb_value = np.nan
                        mb_reff = np.nan
                        mb_10rvir = np.nan
                        rvir_value = np.nan
                        jz_jcirc_avg = np.nan
                        t_dyn_value = np.nan

                    # Get halo type
                    key = (sim, str(hid))
                    if key in types and types[key] in ['Central', 'Backsplash', 'Satellite']:
                        htype_value = 'o' if types[key] in ['Central'] else 'v'
                        #print(f'htype: {htype_value}')
                    else:
                        htype_value = np.nan

                    '''
                    # Load and smooth the Es matrices
                    Euler_s_smooth = StShapes[str(hid)]['Euler_s_smooth']
                    phi_s = Euler_s_smooth[0](Reff)
                    theta_s = Euler_s_smooth[1](Reff)
                    psi_s = Euler_s_smooth[2](Reff)
                    Es_at_Reff = np.array((phi_s,theta_s,psi_s))
                    #print('contructing array',Es_at_Reff)
                    #return


                    Euler_d_smooth = DMShapes[str(hid)]['Euler_d_smooth']
                    phi_d = Euler_d_smooth[0](Reff)
                    theta_d = Euler_d_smooth[1](Reff)
                    psi_d = Euler_d_smooth[2](Reff)
                    Ed_at_Reff = np.array((phi_d,theta_d,psi_d))

                    #print(phi_s-phi_d,theta_s,theta_d,psi_s,psi_d)
                    '''

                    ba_s_func(rbins)

                    # Check condition and assign NaN if needed
                    if (ba_s_value > 1 or ca_s_value > 1 or ba_d_value > 1 or ca_d_value > 1 or
                            ba_s_value < 0.01 or ca_s_value < 0.01 or ba_d_value < 0.01 or ca_d_value < 0.01):
                            # ):  # or ( np.log10(10**masses_value/m_vir_value) < -3):
                            #     # print(f"sim: {sim}, hid: {hid}, ba_s: {ba_s_value:.2f}, ca_s: {ca_s_value:.2f}, ba_d: {ba_d_value:.2f}, ca_d: {ca_d_value:.2f}")
                    
                        print(f"sim: {sim}, hid: {hid}, ba_s: {ba_s_value:.2f}, ca_s: {ca_s_value:.2f}, ba_d: {ba_d_value:.2f}, ca_d: {ca_d_value:.2f}")
                        #ba_s_value, ca_s_value, ba_d_value, ca_d_value = np.nan, np.nan, np.nan, np.nan
                    #     T_s_value, T_d_value, masses_value, mb_value, reff_value, m_vir_value = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    #     htype_value = np.nan
                    #     # Es_at_Reff,Ed_at_Reff = np.ones((3))*np.nan,np.ones((3))*np.nan

                    # Append values to lists
                    B_s.append(ba_s_value)
                    C_s.append(ca_s_value)
                    T_s.append(T_s_value)
                    B_d.append(ba_d_value)
                    C_d.append(ca_d_value)
                    T_d.append(T_d_value)
                    diffs_at_Reff.append(diff_at_Reff)
                    m_vir.append(m_vir_value)
                    masses.append(masses_value)
                    mb.append(mb_value)
                    mb_reffs.append(mb_reff)
                    mb_10rvirs.append(mb_10rvir)
                    rvir.append(rvir_value)
                    t_dyn.append(t_dyn_value)
                    htype.append(htype_value)
                    reff.append(Reff)
                    sims.append(sim)
                    hids.append(hid)
                    feedback_type.append(feedbacks[i])
                    jz_jcirc_avgs.append(jz_jcirc_avg)

                    #append merger ratios

                    #if merger ratios could not be loaded, append nan


                    try:
                        merger_row = merger_ratios.loc[(merger_ratios['sim'] == sim ) & (merger_ratios['halo'] == hid)]
                        merger_values = merger_row[['004096','004032']].values
                    except:
                        merger_values = np.array([[np.nan, np.nan]])
                    #print(f'sim,halo,merger values',sim,hid,merger_values)

                    #check if merger_values is an empty array
                    if len(merger_values) == 0:
                        #print(f'No merger values found for sim {sim} halo {hid}')
                        merger.append(np.array([[np.nan, np.nan]]))
                    else:
                        merger.append(merger_values)




                            # Es_smoothed.append(np.ones((3)))
                        # Ed_smoothed.append(np.ones((3)))
                        # Es_smoothed.append(Es_at_Reff)
                        # Ed_smoothed.append(Ed_at_Reff)
                        # print('after checks and assignment',Es_at_Reff)


                except KeyError:
                    print(f'Key error {KeyError} in sim, halo {sim},{hid}')
                    print(traceback.format_exc())
                    # for key, value in StShapes[str(hid)].items():
                    # Construct the full key path
                    # current_key = f"{key}"
                    # print(current_key)
                    # return

                    # continue  # Assume you want to skip this entry

                except Exception as e:
                    print(f'Error: {e}')  # Print other types of exceptions
                    # continue

    B_s = np.array(B_s)
    C_s = np.array(C_s)
    T_s = np.array(T_s)
    B_d = np.array(B_d)
    C_d = np.array(C_d)
    T_d = np.array(T_d)
    try:
        diffs_at_Reff = np.array(diffs_at_Reff)
    except:
        diffs_at_Reff = np.copy(B_s)*np.nan
    masses = np.array(masses)
    mb = np.array(mb)
    mb_reffs = np.array(mb_reffs)
    mb_10rvirs = np.array(mb_10rvirs)
    rvir = np.array(rvir)
    jz_jcirc_avg = np.array(jz_jcirc_avgs)
    t_dyn = np.array(t_dyn)

    htype = np.array(htype)
    reff = np.array(reff)
    m_vir = np.array(m_vir)
    sims = np.array(sims)
    hid = np.array(hids)
    # Es_smoothed = np.array(Es_smoothed)
    # Ed_smoothed = np.array(Ed_smoothed)
    feedback_type = np.array(feedback_type)
    #print(merger)
    mergers = np.array(merger)

    #get index of sim 'r431' and halo 1
    #i = (np.where((sims == 'r431') & (hid == 1)))
    #print 0th element of each array
   # print(B_s[i], C_s[i], T_s[i], B_d[i], C_d[i], T_d[i], masses[i], mb[i], htype[i], reff[i], m_vir[i], feedback_type[i], sims[i], hid[i], mergers[i])

    # Create a mask to filter out NaNs
    mask = mask = (~np.isnan(B_s) & ~np.isnan(C_s) & ~np.isnan(T_s) & ~np.isnan(B_d) & ~np.isnan(C_d) & ~np.isnan(T_d)  \
                   & ~np.isnan(masses) & ~np.isnan(mb)
                   & ~(htype == 'nan') & ~np.isnan(reff)  & ~np.isnan(m_vir))
    #mask = np.ones(len(B_s), dtype=bool)

    #mask mergers using mask, mergers is a 2d array with 2 columns
    #mergers = mergers[mask]

    B_s = B_s[mask]
    C_s = C_s[mask]
    T_s = T_s[mask]
    B_d = B_d[mask]
    C_d = C_d[mask]
    T_d = T_d[mask]
    diffs_at_Reff = diffs_at_Reff[mask]
    masses = masses[mask]
    mb = mb[mask]
    mb_reffs = mb_reffs[mask]
    mb_10rvirs = mb_10rvirs[mask]
    rvir = rvir[mask]
    jz_jcirc_avg = jz_jcirc_avg[mask]
    t_dyn = t_dyn[mask]


    htype = htype[mask]
    reff = reff[mask]
    m_vir = m_vir[mask]
    feedback_type = feedback_type[mask]
    sims = sims[mask]
    hid = hid[mask]
    mergers = mergers



    if return_sims:
        return B_s, C_s, T_s, B_d, C_d, T_d, \
            masses, mb,mb_reffs,mb_10rvirs, htype, reff, m_vir, feedback_type, sims, hid, mergers, rvir, diffs_at_Reff, jz_jcirc_avg, t_dyn
    # if return_sims:
    #     return B_s[mask], C_s[mask], T_s[mask], B_d[mask], C_d[mask], T_d[mask], \
    #         masses, mb, htype[mask], reff[mask], m_vir, feedback_type[mask], sims[mask], hid[mask], mergers

    # Apply the mask to each array
    return B_s[mask], C_s[mask], T_s[mask], B_d[mask], C_d[mask], T_d[mask], \
        masses[mask], mb[mask], htype[mask], reff[mask], m_vir[mask], feedback_type[mask], mergers

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

#from mergers get major minor mergers from ratio. Mergers has shape (46, 1, 2)
def get_major_minor_mergers(mergers,threshold = 1/4):
    #return a mask for major and minor mergers
    # arrays with shape (46,1,2)
    major_mask = []
    minor_mask = []
    for merger in mergers:
    # check for minor or major mergers in either column of mergers
    #if a major merger is found, it should overwrite a minor merger
    #only print if one value is not nans

        major,minor = False,False
        if merger[0][0] <= threshold or merger[0][1] <= threshold: #check minor
            major=(False)
            minor=(True)
        if merger[0][0] >= threshold or merger[0][1] >= threshold: #check major
            major=(True)
            minor=(False)
        if not np.isnan(merger[0][0]) or not np.isnan(merger[0][1]):
            print(f'{merger[0][0] <= threshold }, {merger[0][1] <= threshold } \n'
                  f' {merger[0][0] >= threshold }, {merger[0][1] >= threshold }')
            print(f'Merger ratios: {merger[0][0]:.2f}, {merger[0][1]:.2f}')
            print(f'Major: {major}, Minor: {minor}')
        major_mask.append(major)
        minor_mask.append(minor)

    return major_mask,minor_mask

def filter_data(B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir, condition):
    """
    Apply filtering condition to the data.

    Parameters:
    All input arrays (array-like): Data arrays to filter.
    condition (array-like): Boolean array of the same length as input arrays, specifying the filter condition.

    Returns:
    tuple: Filtered data arrays.
    """
    # print(condition,B_s[condition])
    return B_s[condition], C_s[condition], T_s[condition], B_d[condition], C_d[condition], T_d[condition], masses[
        condition], mb[condition], htype[condition], reff[condition], mvir[condition]


def analyze_distances(T_diff, distances, condition=None):
    if condition is not None:
        T_diff = T_diff[condition]
        distances = distances[condition]

    T_diff = (T_diff)

    # Compute mean and standard deviation
    mean_T_diff = np.nanmean(T_diff)
    std_T_diff = np.nanstd(T_diff)
    mean_distances = np.nanmean(distances)
    std_distances = np.nanstd(distances)

    print(
        f"Mean T_diff = {mean_T_diff:.3f}, Std T_diff = {std_T_diff:.2f}, Mean Distances = {mean_distances:.2f}, Std Distances = {std_distances:.2f}")
    return mean_T_diff, std_T_diff, mean_distances, std_distances


def T_vs_Tdm(T_d, T_s,masses,htype,feedback_type,legend,label):
    # T* vs Tdm
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.fill_between([0, 1], [-1 / 3, 2 / 3], [1 / 3, 4 / 3], color='0.75', alpha=.3)
    ax.plot([0, 1], [0, 1], c='0.5', linestyle='--')
    ax.set_ylabel(r'T$_*$', fontsize=15)
    ax.set_xlabel(r'T$_{DM}$', fontsize=15)

    norm = plt.Normalize(int(min(masses)), int(max(masses)) + .1)
    p = ax.scatter(T_d[htype == 'o'], T_s[htype == 'o'], marker='o', c=masses[htype == 'o'], cmap='viridis', norm=norm,label = 'Centrals')
    ax.scatter(T_d[htype == 'v'], T_s[htype == 'v'], marker='v', c=masses[htype == 'v'], cmap='viridis', norm=norm,
               label='Satellites')
    cbar = f.colorbar(p, cax=f.add_axes([.91, .11, .03, .77]))
    cbar.set_label(r'Log(M$_*$/M$_\odot$)', fontsize=15)

    ax.legend(loc='upper left', prop={'size': 12})
    f.savefig(f'../../Figures/3DShapes/T_Comparison.png', bbox_inches='tight', pad_inches=.1)

    f.savefig(f'../../Figures/3DShapes/TvsTDM{label}.png', bbox_inches='tight', pad_inches=.1, dpi=100)  #
    plt.close()

    #make another plot, colored by feedback type
    #get number of feedback types
    feedback_types = np.unique(feedback_type)

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 4, height_ratios=[1, 3, 3, 3], width_ratios=[3, 3, 3, 1])

    # Main scatter plot
    ax_main = plt.subplot(gs[1:, 0:3])

    # x histogram
    ax_histx = plt.subplot(gs[0, 0:3], sharex=ax_main)

    # y histogram
    ax_histy = plt.subplot(gs[1:, 3], sharey=ax_main)



    ax_main.set_xlim([0, 1])
    ax_main.set_ylim([0, 1])
    ax_main.fill_between([0, 1], [-1 / 3, 2 / 3], [1 / 3, 4 / 3], color='0.75', alpha=.3)
    ax_main.plot([0, 1], [0, 1], c='0.5', linestyle='--')
    ax_main.set_ylabel(r'T$_*$', fontsize=15)
    ax_main.set_xlabel(r'T$_{DM}$', fontsize=15)
    for i,feedback in enumerate(feedback_types):
        mask = feedback_type == feedback
        ht = htype[mask]
        ax_main.scatter(T_d[mask][ht=='o'], T_s[mask][ht=='o'], marker='o',c=colors[feedback])
        ax_main.scatter(T_d[mask][ht=='v'], T_s[mask][ht=='v'], marker='^',c=colors[feedback])
    #dummy points for legend
    #satellites and centrals
    ax_main.scatter(-1,-1,marker='o',label='Centrals',c = 'k')
    ax_main.scatter(-1,-1,marker='^',label='Satellites',c = 'k')
    #merians and marvel
    ax_main.scatter(-1,-1,marker='o',label='Merians',c=colors['MerianCDM'])
    ax_main.scatter(-1,-1,marker='o',label='Marvel+DCJL',c=[colors['BWMDC']])


    #add histogram of Triaxiality like in plot_data
    #share bins between all data
    bins = np.linspace(0,1,12)


    ax_histx.hist(T_d[feedback_type== 'MerianCDM'], bins=bins, density=True, histtype='step', color=colors['MerianCDM'], lw=1.5)
    ax_histx.hist(T_d[feedback_type== 'BWMDC'], bins=bins, density=True, histtype='step', color=colors['BWMDC'], lw=1.5)
    ax_histy.hist(T_s[feedback_type== 'MerianCDM'], bins=bins, density=True, histtype='step', orientation='horizontal', color=colors['MerianCDM'], lw=1.5)
    ax_histy.hist(T_s[feedback_type== 'BWMDC'], bins=bins, density=True, histtype='step', orientation='horizontal', color=colors['BWMDC'], lw=1.5)

    ax_main.legend(loc='upper left', prop={'size': 12})
    fig.savefig(f'../../Figures/3DShapes/T_Comparison.png', bbox_inches='tight', pad_inches=.1,dpi=300)
    #plt.close()

import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
# T vs Mstar
def T_vs_Mstar(T_d, T_s, masses, mb, htype, feedback):
    f, ax = plt.subplots(2, 1, figsize=(15, 6))
    plt.subplots_adjust(hspace=0)

    # triaxial markers
    for i in [0, 1]:
        ax[i].set_xlim([5.77, 9.5])
        ax[i].set_ylim([0, 1])
        ax[i].plot([4, 9.5], [1 / 3, 1 / 3], c='.75', linestyle='--', zorder=0)
        ax[i].plot([4, 9.5], [2 / 3, 2 / 3], c='.75', linestyle='--', zorder=0)
        ax[i].tick_params(which='both', labelsize=15)
        ax[i].text(5.8, 1 / 6, 'Oblate', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
        ax[i].text(5.8, 3 / 6, 'Triaxial', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
        ax[i].text(5.8, 5 / 6, 'Prolate', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
    ax[0].set_ylabel('T', fontsize=20)
    ax[1].set_ylabel(r'T$_*$', fontsize=20)
    ax[0].set_yticks([0, .5, 1])
    ax[1].set_yticks([0, .5])
    ax[1].set_xlabel(r'Log(M$_*$/M$_\odot$)', fontsize=20)
    # ax[0].xaxis.set_tick_params(labelbottom='False')
    ax[0].set_xticks([])

    # T*-TDM vs Mass
    # top
    for i in np.arange(len(masses)):
        ax[0].axvline(masses[i], ymin=min([T_d[i], T_s[i]]), ymax=max([T_d[i], T_s[i]]), c='.5', zorder=0)  # links

    color = []
    for i in range(len(feedback)):
        if feedback[i] == 'MerianCDM':
            color.append(colors['MerianCDM'])
        elif feedback[i] == 'BWMDC':
            color.append(colors['BWMDC'])
    color = np.array(color)
    
    ax[0].scatter(0, 0, c=colors['MerianCDM'], marker='o', label='Stellar Merians')
    ax[0].scatter(0, 0, c=colors['BWMDC'], marker='o', label='Stellar Marvel+DCJL')

    ax[0].scatter(masses[htype == 'o'], T_d[htype == 'o'], c='k', label='Dark Matter', marker='o')
    ax[0].scatter(masses[htype == 'o'], T_s[htype == 'o'], c=color[htype=='o'], marker='o')
    ax[0].scatter(masses[htype == 'v'], T_d[htype == 'v'], c='k', marker='v')
    ax[0].scatter(masses[htype == 'v'], T_s[htype == 'v'], c=color[htype=='v'], marker='v')
    ax[0].scatter(0, 0, c='.5', marker='v', label='Satellites')
    #ax[0].scatter(0, 0, c='k', marker='o', label='Centrals')

    ax[0].legend(
        prop={'size': 15},
        ncol=2,
        loc='center left',
        bbox_to_anchor=(0.02, 0.1),
        columnspacing=0.8,  # Adjust this value to decrease spacing between columns
        handlelength=2,  # Adjust this value to decrease the handle length
    )
    # T* vs Mass colored by the ratio of M(baryonic) / virial(mass) within the effetive radius
    # bottom

    vmin = np.min(mb)
    vmax = np.max(mb)
    norm = plt.Normalize(vmin, vmax)
    p = ax[1].scatter(masses[htype == 'o'], T_s[htype == 'o'], c=mb[htype == 'o'], cmap='viridis', norm=norm,
                      marker='o')
    ax[1].scatter(masses[htype == 'v'], T_s[htype == 'v'], c=mb[htype == 'v'], cmap='viridis', norm=norm, marker='v')
    cbar = f.colorbar(p, cax=f.add_axes([.91, .11, .03, .77]))
    cbar.set_label(r'M$_{bary}$/M$_{vir}(<$R$_{eff}$)', fontsize=25)
    #set ticks to every step size only within the range that satisify
    
    #cbar.set_ticks(np.arange(tick_min, tick_max + step, step))
    #cbar.set_ticks([-3, -2.5, -2, -1.5, -1])
    cbar.ax.tick_params(labelsize=15)

    f.savefig(f'../../Figures/3DShapes/TvsMstar.png', bbox_inches='tight', pad_inches=.1, dpi=100)  #
    #plt.close()


def plot_data(B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir, feedback_type, show_scatter = True, show_lines = True, condition=None):
    """
    Plot the data on the given axis.

    Parameters:
    ax : matplotlib axis object
    All other parameters: array-like, Data arrays to plot.

    Returns:
    None
    """
    #major_mask,minor_mask = get_major_minor_mergers(mergers)




    # Apply filter if condition is provided
    if condition is not None:
        # B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir = filter_data(B_s, C_s, T_s, B_d, C_d, T_d, masses,
        #                                                                           mb, htype, reff, mvir,
        #                                                                           condition=condition)
        B_s = B_s[condition]
        C_s = C_s[condition]
        T_s = T_s[condition]
        B_d = B_d[condition]
        C_d = C_d[condition]
        T_d = T_d[condition]
        #masses = masses[condition]
        #mb = mb[condition]
        htype = htype[condition]
        #reff = reff[condition]
        #mvir = mvir[condition]
        feedback_type = feedback_type[condition]



    # Setup the figure and grids for main and side histograms
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(5, 5, height_ratios=[1, 1, 3, 3, 3], width_ratios=[3, 3, 3, 1, 1])  # Adjusted ratios
    # Main scatter plot
    ax_main = plt.subplot(gs[2:5, 0:3])
    # First x histogram
    ax_histx = plt.subplot(gs[1, 0:3], sharex=ax_main)
    # Second x histogram
    ax_histx1 = plt.subplot(gs[0, 0:3], sharex=ax_main)
    # First y histogram
    ax_histy = plt.subplot(gs[2:5, 3], sharey=ax_main)
    # Second y histogram
    ax_histy1 = plt.subplot(gs[2:5, 4], sharey=ax_main)

    # Plot the main scatter plot
    # define colors by feedback type list
    color = ['r' if feedback == 'MerianCDM' else 'b' for feedback in feedback_type]
    color = np.array(color)
    lw = 1.5
    bins = 10
    # label feedback color types
    size = 30
    if show_scatter:
        ax_main.scatter(B_d[htype == 'o'], C_d[htype == 'o'], c=color[htype == 'o'], marker='o', s=size)
        ax_main.scatter(B_s[htype == 'o'], C_s[htype == 'o'], c=color[htype == 'o'], marker='*', s=size)
        ax_main.scatter(B_d[htype == 'v'], C_d[htype == 'v'], c=color[htype == 'v'], marker='v', s=size)
        ax_main.scatter(B_s[htype == 'v'], C_s[htype == 'v'], c=color[htype == 'v'], marker='x', s=size)



        

        #mass_range = np.max(masses) - np.min(masses)

        # norm = Normalize(np.min(masses)+mass_range/3, np.max(masses)-mass_range/3)
        # ax_main.scatter(B_s, C_s, c=masses, cmap='viridis', marker='o', norm=norm, s=size/4)

        # ax_main.scatter(B_s, C_s, c='k', marker='o', s=size/3,label = 'Merians log(M*) > 8.5')
        ax_main.scatter(-1, -1, c='r', marker='o', label='Merians', s=size)
        ax_main.scatter(-1, -1, c='b', marker='o', label='Marvel+DCJL', s=size)
        ax_main.scatter(-1, -1, c='gray', marker='o', label='DM Central', s=size)
        ax_main.scatter(-1, -1, c='gray', marker='v', label='DM Satellite', s=size)
        ax_main.scatter(-1, -1, c='gray', marker='*', label='Stellar Central', s=size)
        ax_main.scatter(-1, -1, c='gray', marker='x', label='Stellar Satellite', s=size)

    if show_lines:
        for i in np.arange(len(B_s)):
            ax_main.plot([B_s[i], B_d[i]], [C_s[i], C_d[i]], c='.5', zorder=0, lw=lw / 2)

    # Add labels and title to main plot
    # ax_main.set_title('Stellar vs. Dark Matter Halo Axis Ratios (B/A and C/A) in Galaxies', fontsize=20)
    ax_main.grid(True)

    #share bins
    bins = np.linspace(0, 1, 16)


    # Plot normalized histograms on the attached axes
    ax_histx.hist(B_s[feedback_type == 'MerianCDM'], bins=bins, density=True, histtype='step', color='red', lw=lw)
    ax_histx.hist(B_s[feedback_type == 'BWMDC'], bins=bins, density=True, histtype='step', color='blue', lw=lw)
    ax_histy.hist(C_s[feedback_type == 'MerianCDM'], bins=bins, density=True, histtype='step', orientation='horizontal',
                  color='red', lw=lw)
    ax_histy.hist(C_s[feedback_type == 'BWMDC'], bins=bins, density=True, histtype='step', orientation='horizontal',
                  color='blue', lw=lw)
    # ax_histx.set_title('Stellar')
    # ax_histy.set_title('Stellar')
    ax_histx.hist([], bins=1, density=True, histtype='step', orientation='horizontal', color='k', lw=lw,
                  label='Stellar')
    ax_histx.legend()

    ls = '--'
    ax_histx1.hist(B_d[feedback_type == 'MerianCDM'], bins=bins, density=True, histtype='step', color='red', lw=lw,
                   ls=ls)
    ax_histx1.hist(B_d[feedback_type == 'BWMDC'], bins=bins, density=True, histtype='step', color='blue', lw=lw, ls=ls)
    ax_histy1.hist(C_d[feedback_type == 'MerianCDM'], bins=bins, density=True, histtype='step',
                   orientation='horizontal', color='red', lw=lw, ls=ls)
    ax_histy1.hist(C_d[feedback_type == 'BWMDC'], bins=bins, density=True, histtype='step', orientation='horizontal',
                   color='blue', lw=lw, ls=ls)
    ax_histx1.hist([], bins=1, density=True, histtype='step', orientation='horizontal', color='k', lw=lw,
                   label='Dark Matter', ls=ls)
    ax_histx1.legend()
    # ax_histx1.set_title('Dark Matter')
    # ax_histy1.set_title('Dark Matter')

    # ax_histx.hist(B_d, bins=20, density=True, histtype='step', color='blue')
    # ax_histx.hist(B_s, bins=20, density=True, histtype='step', color='red')
    # ax_histy.hist(C_d, bins=20, density=True, histtype='step', orientation='horizontal', color='blue')
    # ax_histy.hist(C_s, bins=20, density=True, histtype='step', orientation='horizontal', color='red')

    # Adjust the height of the top histogram and the width of the right histogram
    # gs.update(height_ratios=[1, 5, 5, 5], width_ratios=[5, 5, 5, 1])  # Adjusting grid ratio

    # Remove tick marks
    ax_histx.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # No x-tick marks
    ax_histy.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # No y-tick marks

    ax_histy.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # No x-tick marks
    ax_histx.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    ax_histx1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # No x-tick marks
    ax_histy1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # No y-tick marks

    ax_histy1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # No x-tick marks
    ax_histx1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # Set up the grid, labels, and legends
    handles, labels = ax_main.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_main.legend(by_label.values(), by_label.keys())

    ax_main.set_xlim([0, 1])
    ax_main.set_ylim([0, 1])
    ax_main.plot([0, 1], [0, 1], c='0.5', linestyle='--')
    ax_main.set_xlabel(r'$S = B/A$', fontsize=20)
    ax_main.set_ylabel(r'$Q = C/A$', fontsize=20)
    ax_main.tick_params(which='both', labelsize=15)
    plt.savefig(f'../../Figures/3DShapes/Q_S_hist.png', bbox_inches='tight', pad_inches=.1, dpi=300)  #
    # Plot lines between Baryonic and Dark matter positions

    # norm = Normalize(vmin=-3, vmax=-1)

    # size = 15
    # # Mapping htype to markers
    # # Scatter plots with different shapes and colors by mb
    # size = 15
    # ax.scatter(B_d[htype == 'o'], C_d[htype == 'o'], c=mb[htype == 'o'], cmap='viridis', marker='o', label='DM Central', s=size, norm=norm)
    # ax.scatter(B_s[htype == 'o'], C_s[htype == 'o'], c=mb[htype == 'o'], cmap='viridis', marker='*', label='Stellar Central', s=size, norm=norm)
    #
    # ax.scatter(B_d[htype == 'v'], C_d[htype == 'v'], c=mb[htype == 'v'], cmap='viridis', marker='v', label='DM Satellite', s=size, norm=norm)
    # ax.scatter(B_s[htype == 'v'], C_s[htype == 'v'], c=mb[htype == 'v'], cmap='viridis', marker='^', label='Stellar Satellite', s=size, norm=norm)
    # ax.set_aspect('equal')
    # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax_main)
    # #cbar.set_label('log(M$_{Bary}$/M$_{vir}$)', fontsize=15)
    # cbar.set_label('Log(M*)', fontsize=15)
    # cbar.ax.tick_params(labelsize=15)
    # ax_main.legend(loc='upper left')

    return


def plot_data_with_mergers(B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir, feedback_type, mergers=None, condition=None,
              show_lines=False):
    """
    Plot the data on the given axis.

    Parameters:
    ax : matplotlib axis object
    All other parameters: array-like, Data arrays to plot.

    Returns:
    None
    """
    major_mask, minor_mask = get_major_minor_mergers(mergers)

    show_scatter = True
    # show_lines = True

    # Apply filter if condition is provided
    if condition is not None:
        B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir = filter_data(B_s, C_s, T_s, B_d, C_d, T_d, masses,
                                                                                  mb, htype, reff, mvir,
                                                                                  condition=condition)

    # Setup the figure and grids for main and side histograms
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(5, 5, height_ratios=[1, 1, 3, 3, 3], width_ratios=[3, 3, 3, 1, 1])  # Adjusted ratios
    # Main scatter plot
    ax_main = plt.subplot(gs[2:5, 0:3])
    # First x histogram
    ax_histx = plt.subplot(gs[1, 0:3], sharex=ax_main)
    # Second x histogram
    ax_histx1 = plt.subplot(gs[0, 0:3], sharex=ax_main)
    # First y histogram
    ax_histy = plt.subplot(gs[2:5, 3], sharey=ax_main)
    # Second y histogram
    ax_histy1 = plt.subplot(gs[2:5, 4], sharey=ax_main)

    # Plot the main scatter plot
    # define colors by feedback type list
    color = ['r' if feedback == 'MerianCDM' else 'b' for feedback in feedback_type]
    color = np.array(color)
    lw = 1.5
    bins = 10
    # label feedback color types
    size = 30
    if show_scatter:
        ax_main.scatter(B_d[htype == 'o'], C_d[htype == 'o'], c=color[htype == 'o'], marker='o', s=size)
        ax_main.scatter(B_s[htype == 'o'], C_s[htype == 'o'], c=color[htype == 'o'], marker='*', s=size)
        ax_main.scatter(B_d[htype == 'v'], C_d[htype == 'v'], c=color[htype == 'v'], marker='v', s=size)
        ax_main.scatter(B_s[htype == 'v'], C_s[htype == 'v'], c=color[htype == 'v'], marker='x', s=size)

        # plot major and minor mergers
        # check if mask is all false
        # create marker style for major and minor mergers with no face color

        marker_style_major = dict(linestyle='none', marker='o', markerfacecolor="none", markeredgecolor="k",
                                  markeredgewidth=1,
                                  label='Major Mergers', markersize=size / 3)
        marker_style_minor = dict(linestyle='none', marker='o', markerfacecolor="none", markeredgecolor="k",
                                  markeredgewidth=3,
                                  label='Minor Mergers', markersize=size / 2)
        if sum(major_mask) > 0:
            ax_main.plot(B_d[major_mask], C_d[major_mask], **marker_style_major)
            ax_main.plot(B_s[major_mask], C_s[major_mask], **marker_style_major)
        if sum(minor_mask) > 0:
            ax_main.plot(B_d[minor_mask], C_d[minor_mask], **marker_style_minor)
            ax_main.plot(B_s[minor_mask], C_s[minor_mask], **marker_style_minor)

        # mass_range = np.max(masses) - np.min(masses)

        # norm = Normalize(np.min(masses)+mass_range/3, np.max(masses)-mass_range/3)
        # ax_main.scatter(B_s, C_s, c=masses, cmap='viridis', marker='o', norm=norm, s=size/4)

        # ax_main.scatter(B_s, C_s, c='k', marker='o', s=size/3,label = 'Merians log(M*) > 8.5')
        ax_main.scatter(-1, -1, c='r', marker='o', label='Merians', s=size)
        ax_main.scatter(-1, -1, c='b', marker='o', label='Marvel+DCJL', s=size)
        ax_main.scatter(-1, -1, c='gray', marker='o', label='DM Central', s=size)
        ax_main.scatter(-1, -1, c='gray', marker='v', label='DM Satellite', s=size)
        ax_main.scatter(-1, -1, c='gray', marker='*', label='Stellar Central', s=size)
        ax_main.scatter(-1, -1, c='gray', marker='x', label='Stellar Satellite', s=size)

    if show_lines:
        for i in np.arange(len(B_s)):
            ax_main.plot([B_s[i], B_d[i]], [C_s[i], C_d[i]], c='.5', zorder=0, lw=lw / 2)

    # Add labels and title to main plot
    # ax_main.set_title('Stellar vs. Dark Matter Halo Axis Ratios (B/A and C/A) in Galaxies', fontsize=20)
    ax_main.grid(True)

    # share bins
    bins = np.linspace(0, 1, 16)

    # Plot normalized histograms on the attached axes
    ax_histx.hist(B_s[feedback_type == 'MerianCDM'], bins=bins, density=True, histtype='step', color='red', lw=lw)
    ax_histx.hist(B_s[feedback_type == 'BWMDC'], bins=bins, density=True, histtype='step', color='blue', lw=lw)
    ax_histy.hist(C_s[feedback_type == 'MerianCDM'], bins=bins, density=True, histtype='step', orientation='horizontal',
                  color='red', lw=lw)
    ax_histy.hist(C_s[feedback_type == 'BWMDC'], bins=bins, density=True, histtype='step', orientation='horizontal',
                  color='blue', lw=lw)
    # ax_histx.set_title('Stellar')
    # ax_histy.set_title('Stellar')
    ax_histx.hist([], bins=1, density=True, histtype='step', orientation='horizontal', color='k', lw=lw,
                  label='Stellar')
    ax_histx.legend()

    ls = '--'
    ax_histx1.hist(B_d[feedback_type == 'MerianCDM'], bins=bins, density=True, histtype='step', color='red', lw=lw,
                   ls=ls)
    ax_histx1.hist(B_d[feedback_type == 'BWMDC'], bins=bins, density=True, histtype='step', color='blue', lw=lw, ls=ls)
    ax_histy1.hist(C_d[feedback_type == 'MerianCDM'], bins=bins, density=True, histtype='step',
                   orientation='horizontal', color='red', lw=lw, ls=ls)
    ax_histy1.hist(C_d[feedback_type == 'BWMDC'], bins=bins, density=True, histtype='step', orientation='horizontal',
                   color='blue', lw=lw, ls=ls)
    ax_histx1.hist([], bins=1, density=True, histtype='step', orientation='horizontal', color='k', lw=lw,
                   label='Dark Matter', ls=ls)
    ax_histx1.legend()
    # ax_histx1.set_title('Dark Matter')
    # ax_histy1.set_title('Dark Matter')

    # ax_histx.hist(B_d, bins=20, density=True, histtype='step', color='blue')
    # ax_histx.hist(B_s, bins=20, density=True, histtype='step', color='red')
    # ax_histy.hist(C_d, bins=20, density=True, histtype='step', orientation='horizontal', color='blue')
    # ax_histy.hist(C_s, bins=20, density=True, histtype='step', orientation='horizontal', color='red')

    # Adjust the height of the top histogram and the width of the right histogram
    # gs.update(height_ratios=[1, 5, 5, 5], width_ratios=[5, 5, 5, 1])  # Adjusting grid ratio

    # Remove tick marks
    ax_histx.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # No x-tick marks
    ax_histy.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # No y-tick marks

    ax_histy.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # No x-tick marks
    ax_histx.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    ax_histx1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # No x-tick marks
    ax_histy1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # No y-tick marks

    ax_histy1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # No x-tick marks
    ax_histx1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # Set up the grid, labels, and legends
    handles, labels = ax_main.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_main.legend(by_label.values(), by_label.keys())

    ax_main.set_xlim([0, 1])
    ax_main.set_ylim([0, 1])
    ax_main.plot([0, 1], [0, 1], c='0.5', linestyle='--')
    ax_main.set_xlabel(r'$S = B/A$', fontsize=20)
    ax_main.set_ylabel(r'$Q = C/A$', fontsize=20)
    ax_main.tick_params(which='both', labelsize=15)
    plt.savefig(f'../../Figures/3DShapes/Q_S_hist.png', bbox_inches='tight', pad_inches=.1, dpi=300)  #
    # Plot lines between Baryonic and Dark matter positions

    # norm = Normalize(vmin=-3, vmax=-1)

    # size = 15
    # # Mapping htype to markers
    # # Scatter plots with different shapes and colors by mb
    # size = 15
    # ax.scatter(B_d[htype == 'o'], C_d[htype == 'o'], c=mb[htype == 'o'], cmap='viridis', marker='o', label='DM Central', s=size, norm=norm)
    # ax.scatter(B_s[htype == 'o'], C_s[htype == 'o'], c=mb[htype == 'o'], cmap='viridis', marker='*', label='Stellar Central', s=size, norm=norm)
    #
    # ax.scatter(B_d[htype == 'v'], C_d[htype == 'v'], c=mb[htype == 'v'], cmap='viridis', marker='v', label='DM Satellite', s=size, norm=norm)
    # ax.scatter(B_s[htype == 'v'], C_s[htype == 'v'], c=mb[htype == 'v'], cmap='viridis', marker='^', label='Stellar Satellite', s=size, norm=norm)
    # ax.set_aspect('equal')
    # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax_main)
    # #cbar.set_label('log(M$_{Bary}$/M$_{vir}$)', fontsize=15)
    # cbar.set_label('Log(M*)', fontsize=15)
    # cbar.ax.tick_params(labelsize=15)
    # ax_main.legend(loc='upper left')

    return
def create_histograms(B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir, conditions, reff_multi):
    """
    Create histograms for the given parameters with multiple conditions and display statistics.

    Parameters:
    B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir (array-like): Data arrays.
    conditions (list of tuples): List of (label, condition) tuples to filter and label the data.

    Returns:
    None
    """
    T_diff = np.array((T_d - T_s))
    distances = np.sqrt((B_d - B_s) ** 2 + (C_d - C_s) ** 2)
    parameters = {
        'S_*': B_s,
        'Q_*': C_s,
        'T_*': T_s,
        'S_DM': B_d,
        'Q_DM': C_d,
        'T_DM': T_d,
        'Log M*': masses,
        'log(M$_{Bary}$/M$_{vir}$)': mb,
        'Halo type': htype,
        'Effective Radius (Kpc)': reff,
        'Halo Mass': np.log10(mvir),
        'T_diff': (T_diff),
        r'$\Delta D$': distances,
        r'$S_{DM} - S_*$': B_d - B_s,
        r'$Q_{DM} - Q_*$': C_d - C_s
    }

    # Define consistent bin edges for Q, S, and T properties
    qst_bin_edges = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    delta_bin_edges = np.linspace(-.5, .5, 11)  # 10 bins from -1 to 1 for differences

    bin_edges = {}
    for param_name, param_data in parameters.items():
        if param_name == 'Halo type':
            bin_edges[param_name] = np.array([0, 1, 2])  # Assuming 'o' and 'v' will be mapped to 0 and 1
        elif param_name in ['S_*', 'Q_*', 'T_*', 'S_DM', 'Q_DM', 'T_DM']:
            bin_edges[param_name] = qst_bin_edges
        elif param_name in ['$S_{DM} - S_*$', '$Q_{DM} - Q_*$']:
            bin_edges[param_name] = delta_bin_edges
        else:
            bin_edges[param_name] = np.histogram_bin_edges(param_data, bins=10)

    # DataFrame to store statistics
    stats_df = pd.DataFrame(columns=['Condition', 'Parameter', 'Mean', 'Std'])

    fig, axes = plt.subplots(6, 3, figsize=(10, 18))
    axes = axes.flatten()

    stats_dict = {}

    # Loop through each parameter
    for ax, (param_name, param_data) in zip(axes, parameters.items()):
        bins = bin_edges.get(param_name)

        # Loop through each condition
        for label, condition in conditions:
            if condition is not None:
                filtered_data = param_data[condition]
            else:
                filtered_data = param_data
            # Plot histogram
            ax.hist(filtered_data, bins=bins, histtype='step', label=label)
            ax.set_xlabel(param_name)
            ax.set_ylabel('Frequency')

            if param_name == 'Halo type':
                continue
            # Calculate statistics
            mean_value = np.nanmean(filtered_data)
            std_value = np.nanstd(filtered_data)

            # Organize data in dictionary
            if label not in stats_dict:
                stats_dict[label] = {}

            stats_dict[label].update({
                f'{param_name} Mean': mean_value,
                f'{param_name} Std': std_value
            })
            if ax == axes[0]:
                ax.legend()

    # Convert the dictionary into a DataFrame
    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index').reset_index().rename(columns={'index': 'Condition'})
    # Hide any unused subplots
    for ax in axes[len(parameters):]:
        ax.set_visible(False)

    # plt.tight_layout()
    plt.savefig(f'../../Figures/3DShapes/Histogram.{reff_multi}.png', dpi=100)
    plt.close()

    # Save the dataframe to a CSV file
    stats_df.to_csv(f'../../Figures/3DShapes/Statistics.{reff_multi}.csv', index=False)

    # Print the dataframe
    # print(stats_df)



def plot_DM_S_axes_diffs(ax,c_diff,masses,feedback_type,ylabel):

    ax.scatter(masses[feedback_type == 'MerianCDM'], c_diff[feedback_type == 'MerianCDM'], label='Merians',
               c=colors['MerianCDM'])
    ax.scatter(masses[feedback_type == 'BWMDC'], c_diff[feedback_type == 'BWMDC'], label='Marvel+DCJL',
               c=colors['BWMDC'])
    # plot y = 0 line
    ax.axhline(1, c='k', linestyle='-', alpha=.5)
    ax.set_xlabel(r'Log(M$_*$/M$_\odot$)', fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.legend(loc='upper left', prop={'size': 20})
    # linear fit and show error on plot
    # increase tick size and font
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    # sns.regplot(x=masses[feedback_type=='MerianCDM'],y=c_diff[feedback_type=='MerianCDM'],ax=ax,scatter=False)
    # sns.regplot(x=masses[feedback_type=='BWMDC'],y=c_diff[feedback_type=='BWMDC'],ax=ax,scatter=False)
    sns.regplot(x=masses, y=c_diff, ax=ax, scatter=False, color='k', label='Linear Fit')
    # get linear fit parameters using scipy.stats.linregress
    from scipy.stats import linregress
    LinregressResult = linregress(masses, c_diff)
    slope = LinregressResult.slope
    intercept = LinregressResult.intercept
    rv = LinregressResult.rvalue
    pval = LinregressResult.pvalue
    stderr = LinregressResult.stderr
    intercept_stderr = LinregressResult.intercept_stderr
    print(
        f'slope: {slope}, intercept: {intercept}, r: {rv}, pval: {pval}, stderr: {stderr}, intercept_stderr: {intercept_stderr}')

    # plot linear fit
    x = np.linspace(6, 9.5, 100)
    y = slope * x + intercept
    # ax.plot(x,y,c='k',linestyle='--',label='Linear Fit')
    # add confidence interval from both slope error and intercept error

    return ax
def plot_SIDM_CDM(B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir, feedback_type, sims):
    # create new data arrays for each CDM and SIDM with matching string in sims, with the same order and same length

    # Create masks for CDM and SIDM
    mask_CDM = feedback_type == 'MerianCDM'
    mask_SIDM = feedback_type == 'MerianSIDM'

    # Create CDM data sets
    B_s_CDM = B_s[mask_CDM]
    C_s_CDM = C_s[mask_CDM]
    T_s_CDM = T_s[mask_CDM]
    B_d_CDM = B_d[mask_CDM]
    C_d_CDM = C_d[mask_CDM]
    T_d_CDM = T_d[mask_CDM]
    masses_CDM = masses[mask_CDM]
    mb_CDM = mb[mask_CDM]
    htype_CDM = htype[mask_CDM]
    reff_CDM = reff[mask_CDM]
    mvir_CDM = mvir[mask_CDM]
    feedback_type_CDM = feedback_type[mask_CDM]
    sims_CDM = sims[mask_CDM]

    # Create SIDM data sets
    B_s_SIDM = B_s[mask_SIDM]
    C_s_SIDM = C_s[mask_SIDM]
    T_s_SIDM = T_s[mask_SIDM]
    B_d_SIDM = B_d[mask_SIDM]
    C_d_SIDM = C_d[mask_SIDM]
    T_d_SIDM = T_d[mask_SIDM]
    masses_SIDM = masses[mask_SIDM]
    mb_SIDM = mb[mask_SIDM]
    htype_SIDM = htype[mask_SIDM]
    reff_SIDM = reff[mask_SIDM]
    mvir_SIDM = mvir[mask_SIDM]
    feedback_type_SIDM = feedback_type[mask_SIDM]
    sims_SIDM = sims[mask_SIDM]

    # let make 3 plots
    # 1. T vs Mstar
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # seperate SIDM vs CDM by shape
    # seperate stellar and dark matter by color
    # ax.scatter(masses[feedback_type == 'MerianCDM'],T_s[feedback_type == 'MerianCDM'],c='r')
    # ax.scatter(masses[feedback_type == 'MerianSIDM'],T_s[feedback_type == 'MerianSIDM'],c='b')
    # ax.scatter(masses[feedback_type == 'MerianCDM'],T_d[feedback_type == 'MerianCDM'],c='k')
    # ax.scatter(masses[feedback_type == 'MerianSIDM'],T_d[feedback_type == 'MerianSIDM'],c='k')
    shared_sims = set(sims_CDM) & set(sims_SIDM)
    mask_CDM_shared = np.isin(sims_CDM, list(shared_sims))
    mask_SIDM_shared = np.isin(sims_SIDM, list(shared_sims))
    ax.scatter(masses_CDM[mask_CDM_shared], T_s_CDM[mask_CDM_shared], c='r')
    ax.scatter(masses_SIDM[mask_SIDM_shared], T_s_SIDM[mask_SIDM_shared], c='b')
    ax.scatter(masses_CDM[mask_CDM_shared], T_d_CDM[mask_CDM_shared], c='k')
    ax.scatter(masses_SIDM[mask_SIDM_shared], T_d_SIDM[mask_SIDM_shared], c='k')
    stellar_dark_matter_link_color = 'green'
    for sim_name in shared_sims:
        # Get indices of this sim name in CDM and SIDM data
        idx_CDM = np.where(sims_CDM == sim_name)[0]
        idx_SIDM = np.where(sims_SIDM == sim_name)[0]

        # Draw lines between corresponding points
        for i in range(len(idx_CDM)):
            print(sims_CDM[idx_CDM[i]], sims_SIDM[idx_SIDM[i]])
            ax.plot([masses_CDM[idx_CDM[i]], masses_SIDM[idx_SIDM[i]]],
                    [T_s_CDM[idx_CDM[i]], T_s_SIDM[idx_SIDM[i]]],
                    c='gray', alpha=0.5, zorder=0)
            ax.plot([masses_CDM[idx_CDM[i]], masses_SIDM[idx_SIDM[i]]],
                    [T_d_CDM[idx_CDM[i]], T_d_SIDM[idx_SIDM[i]]],
                    c='gray', alpha=0.5, zorder=0)
        # Plot links between stellar and dark matter components in a different color
        ax.plot([masses_CDM[idx_CDM[i]], masses_CDM[idx_CDM[i]]],
                [T_s_CDM[idx_CDM[i]], T_d_CDM[idx_CDM[i]]],
                c=stellar_dark_matter_link_color, alpha=0.5, zorder=0)
        ax.plot([masses_SIDM[idx_SIDM[i]], masses_SIDM[idx_SIDM[i]]],
                [T_s_SIDM[idx_SIDM[i]], T_d_SIDM[idx_SIDM[i]]],
                c=stellar_dark_matter_link_color, alpha=0.5, zorder=0)
    ax.scatter(-1, -1, c='r', label='Stellar CDM')
    ax.scatter(-1, -1, c='b', label='Stellar SIDM')
    ax.scatter(-1, -1, c='k', label='Dark Matter')

    ax.set_xlim([np.min(masses_SIDM) - .1, np.max(masses_SIDM) + .1])
    ax.set_ylim([0, 1])
    # plot links between stellar and dark matter
    # for i in np.arange(len(B_s)):
    # ax.plot([masses[i],masses[i]],[T_s[i],T_d[i]],c='.5',zorder=0)
    ax.set_xlabel(r'Log(M$_*$/M$_\odot$)', fontsize=15)
    ax.set_ylabel('T', fontsize=15)
    ax.legend()
    ax.set_title('linking SIDM and CDM')
    plt.savefig(f'../../Figures/3DShapes/TvsMstar.SIDM_CDM.png', bbox_inches='tight', pad_inches=.1, dpi=300)

    # plot Triaxiality in dark matter CDM vs SIDM
    # T_d_CDM vs T_d_SIDM for matched sims
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    print('Triaxality')
    print(f'Length of idx_CDM: {len(idx_CDM)}')
    print(f'Contents of idx_CDM: {idx_CDM}')
    for sim_name in shared_sims:
        # Get indices of this sim name in CDM and SIDM data
        idx_CDM = np.where(sims_CDM == sim_name)[0]
        idx_SIDM = np.where(sims_SIDM == sim_name)[0]
        # Draw lines between corresponding points
        for i in range(len(idx_CDM)):
            ax.scatter(T_d_CDM[idx_CDM[i]], T_d_SIDM[idx_SIDM[i]], c='k', alpha=.5)

    ax.set_xlabel('T CDM', fontsize=15)
    ax.set_ylabel('T SIDM', fontsize=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot([0, 1], [0, 1], c='k', linestyle='--')
    ax.set_aspect('equal')
    plt.savefig(f'../../Figures/3DShapes/DMTriaxiality.CDM_SIDM.png', bbox_inches='tight', pad_inches=.1, dpi=300)

    # cdm marker = 'o', sidm marker = '*'
    # plot b/a vs c/a for CDM vs SIDM for matched sims Dark matter
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for sim_name in shared_sims:
        # Get indices of this sim name in CDM and SIDM data
        idx_CDM = np.where(sims_CDM == sim_name)[0]
        idx_SIDM = np.where(sims_SIDM == sim_name)[0]
        # Draw lines between corresponding points
        for i in range(len(idx_CDM)):
            ax.scatter(B_d_CDM[idx_CDM[i]], C_d_CDM[idx_CDM[i]], c='r', alpha=1, marker='o')
            ax.scatter(B_d_SIDM[idx_SIDM[i]], C_d_SIDM[idx_SIDM[i]], c='b', alpha=1, marker='*')
            # plot links between CDM and SIDM
            ax.plot([B_d_CDM[idx_CDM[i]], B_d_SIDM[idx_SIDM[i]]], [C_d_CDM[idx_CDM[i]], C_d_SIDM[idx_SIDM[i]]],
                    c='gray', alpha=.5, zorder=0)
    # dummy points for legend
    ax.scatter(-1, -1, c='r', label='CDM', marker='o')
    ax.scatter(-1, -1, c='b', label='SIDM', marker='*')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot([0, 1], [0, 1], c='k', linestyle='--')
    ax.set_xlabel(r'B/A', fontsize=15)
    ax.set_ylabel(r'C/A', fontsize=15)
    ax.set_aspect('equal')
    ax.set_title('Dark Matter')
    ax.legend()
    plt.savefig(f'../../Figures/3DShapes/CvB.DM_CDM_SIDM.png', bbox_inches='tight', pad_inches=.1, dpi=300)

    # plot b/a vs c/a for CDM vs SIDM for matched sims Stellar
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for sim_name in shared_sims:
        # Get indices of this sim name in CDM and SIDM data
        idx_CDM = np.where(sims_CDM == sim_name)[0]
        idx_SIDM = np.where(sims_SIDM == sim_name)[0]
        # Draw lines between corresponding points
        for i in range(len(idx_CDM)):
            ax.scatter(B_s_CDM[idx_CDM[i]], C_s_CDM[idx_CDM[i]], c='r', alpha=1, marker='o')
            ax.scatter(B_s_SIDM[idx_SIDM[i]], C_s_SIDM[idx_SIDM[i]], c='b', alpha=1, marker='*')
            # plot links between CDM and SIDM
            ax.plot([B_s_CDM[idx_CDM[i]], B_s_SIDM[idx_SIDM[i]]], [C_s_CDM[idx_CDM[i]], C_s_SIDM[idx_SIDM[i]]],
                    c='gray', alpha=.5, zorder=0)
    # dummy points for legend
    ax.scatter(-1, -1, c='r', label='CDM', marker='o')
    ax.scatter(-1, -1, c='b', label='SIDM', marker='*')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot([0, 1], [0, 1], c='k', linestyle='--')
    ax.set_xlabel(r'B/A', fontsize=15)
    ax.set_ylabel(r'C/A', fontsize=15)
    ax.set_aspect('equal')
    ax.set_title('Stellar')
    ax.legend()
    plt.savefig(f'../../Figures/3DShapes/CvB.St_CDM_SIDM.png', bbox_inches='tight', pad_inches=.1, dpi=300)

    # create 2 subplots, top Triaxiality linked stellar and Dark matter for cdm, bottom for sidm

    f, ax = plt.subplots(2, 1, figsize=(15, 6))
    plt.subplots_adjust(hspace=0)

    # triaxial markers
    for i in [0, 1]:
        ax[i].set_xlim([7.77, 9.5])
        ax[i].set_ylim([0, 1])
        ax[i].plot([6, 9.5], [1 / 3, 1 / 3], c='.75', linestyle='--', zorder=0)
        ax[i].plot([6, 9.5], [2 / 3, 2 / 3], c='.75', linestyle='--', zorder=0)
        ax[i].tick_params(which='both', labelsize=15)
        ax[i].text(7.8, 1 / 6, 'Oblate', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
        ax[i].text(7.8, 3 / 6, 'Triaxial', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
        ax[i].text(7.8, 5 / 6, 'Prolate', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
    ax[0].set_ylabel('T', fontsize=20)
    ax[1].set_ylabel(r'T$_*$', fontsize=20)
    ax[0].set_yticks([0, .5, 1])
    ax[1].set_yticks([0, .5])
    ax[1].set_xlabel(r'Log(M$_*$/M$_\odot$)', fontsize=20)
    # ax[0].xaxis.set_tick_params(labelbottom='False')
    ax[0].set_xticks([])

    # T*-TDM vs Mass CDM
    # top
    for i in np.arange(len(masses_CDM)):
        ax[0].axvline(masses_CDM[i], ymin=min([T_d_CDM[i], T_s_CDM[i]]), ymax=max([T_d_CDM[i], T_s_CDM[i]]), c='.5',
                      zorder=0)  # links

    ax[0].scatter(masses_CDM, T_d_CDM, c='k', label='Dark Matter', marker='o')
    ax[0].scatter(masses_CDM, T_s_CDM, c='r', label='Stellar', marker='o')
    ax[0].legend(
        prop={'size': 15},
        ncol=3,
        loc='center left',
        bbox_to_anchor=(0.02, 0.1),
        columnspacing=0.8,  # Adjust this value to decrease spacing between columns
        handlelength=2,  # Adjust this value to decrease the handle length
    )
    # T*-TDM vs Mass SIDM
    # bottom

    for i in np.arange(len(masses_SIDM)):
        ax[1].axvline(masses_SIDM[i], ymin=min([T_d_SIDM[i], T_s_SIDM[i]]), ymax=max([T_d_SIDM[i], T_s_SIDM[i]]),
                      c='.5', zorder=0)
    ax[1].scatter(masses_SIDM, T_d_SIDM, c='k', label='Dark Matter', marker='o')
    ax[1].scatter(masses_SIDM, T_s_SIDM, c='r', label='Stellar', marker='o')
    ax[1].legend(
        prop={'size': 15},
        ncol=3,
        loc='center left',
        bbox_to_anchor=(0.02, 0.1),
        columnspacing=0.8,  # Adjust this value to decrease spacing between columns
        handlelength=2,  # Adjust this value to decrease the handle length
    )

    f.savefig(f'../../Figures/3DShapes/TvsMstarCDM-SIDM.png', bbox_inches='tight', pad_inches=.1, dpi=100)  #
    plt.close()


def plot_data_with_disky(B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir, feedback_type, disky_mask,
                         mergers=None, show_lines=False, show_scatter=True):
    """
    Plot the data on the given axis, with different colors for disky and non-disky galaxies.

    Parameters:
    B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir : array-like
        Data arrays to plot.
    feedback_type : array-like
        Array of feedback types for each galaxy (not used for coloring).
    disky_mask : array-like
        Boolean mask where True indicates a disky galaxy.
    mergers : array-like, optional
        Merger information for galaxies.
    condition : function, optional
        A function to filter the data.
    show_lines : bool, optional
        Whether to show lines connecting stellar and dark matter components.
    show_scatter : bool, optional
        Whether to show scatter points. If False, only lines will be shown.

    Returns:
    None
    """
    #filter nan values in disky_mask
    condition = ~np.isnan(disky_mask)
    #print(f'condition: {condition}')
    B_s, C_s, T_s, B_d, C_d, T_d, htype, reff, feedback_type, disky_mask = B_s[condition], C_s[condition],\
        T_s[condition], B_d[condition], C_d[condition], T_d[condition],\
        htype[condition], reff[condition], feedback_type[condition], disky_mask[condition]


    # Setup the figure and grids
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(5, 5, height_ratios=[1, 1, 3, 3, 3], width_ratios=[3, 3, 3, 1, 1])
    ax_main = plt.subplot(gs[2:5, 0:3])
    ax_histx = plt.subplot(gs[1, 0:3], sharex=ax_main)
    ax_histx1 = plt.subplot(gs[0, 0:3], sharex=ax_main)
    ax_histy = plt.subplot(gs[2:5, 3], sharey=ax_main)
    ax_histy1 = plt.subplot(gs[2:5, 4], sharey=ax_main)

    # Define colors for disky and non-disky galaxies
    disky_color = 'green'
    non_disky_color = 'k'

    # Plot the main scatter plot or lines
    size = 30
    for is_disky in [True, False]:
        color = disky_color if is_disky else non_disky_color
        mask = disky_mask == is_disky
        dm_color = lighten_color(color, 1.5)

        if show_scatter:
            ax_main.scatter(B_d[mask], C_d[mask], c=color, marker='o', s=size, alpha=1)
            ax_main.scatter(B_s[mask], C_s[mask], c=color, marker='*', s=size, alpha=1)

        if show_lines or not show_scatter:
            for i in np.where(mask)[0]:
                if show_scatter:
                    ax_main.plot([B_s[i], B_d[i]], [C_s[i], C_d[i]], zorder=0, lw=0.5, c='k', alpha=0.5)
                else:
                    ax_main.plot([B_s[i], B_d[i]], [C_s[i], C_d[i]], zorder=0, lw=1, c=color, alpha=0.7)

    # Set up legends, labels, and grid
    if show_scatter:
        ax_main.scatter(-1, -1, c=disky_color, marker='o', label='Disky', s=size)
        ax_main.scatter(-1, -1, c=non_disky_color, marker='o', label='Non-Disky', s=size)
        ax_main.scatter(-1, -1, c='gray', marker='o', label='Dark Matter', s=size)
        ax_main.scatter(-1, -1, c='gray', marker='*', label='Stellar', s=size)
    else:
        ax_main.plot([], [], c=disky_color, label='Disky')
        ax_main.plot([], [], c=non_disky_color, label='Non-Disky')

    ax_main.legend(loc='upper left', fontsize=10)
    ax_main.set_xlabel(r'$S = B/A$', fontsize=20)
    ax_main.set_ylabel(r'$Q = C/A$', fontsize=20)
    ax_main.set_xlim([0, 1])
    ax_main.set_ylim([0, 1])
    ax_main.plot([0, 1], [0, 1], c='0.5', linestyle='--')
    ax_main.grid(True)
    ax_main.tick_params(which='both', labelsize=15)

    # Plot histograms
    bins = np.linspace(0, 1, 16)
    lw = 1.5

    for is_disky, color in [(True, disky_color), (False, non_disky_color)]:
        mask = disky_mask == is_disky

        ax_histx.hist(B_s[mask], bins=bins, density=True, histtype='step', color=color, lw=lw)
        ax_histy.hist(C_s[mask], bins=bins, density=True, histtype='step', orientation='horizontal', color=color, lw=lw)
        ax_histx1.hist(B_d[mask], bins=bins, density=True, histtype='step', color=color, lw=lw, ls='--')
        ax_histy1.hist(C_d[mask], bins=bins, density=True, histtype='step', orientation='horizontal', color=color,
                       lw=lw, ls='--')

    # Remove tick marks for histograms
    for ax in [ax_histx, ax_histx1, ax_histy, ax_histy1]:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                       labelleft=False)

    # Add legends for histograms
    ax_histx.hist([], bins=1, density=True, histtype='step', color='k', lw=lw, label='Stellar')
    ax_histx1.hist([], bins=1, density=True, histtype='step', color='k', lw=lw, ls='--', label='Dark Matter')
    ax_histx.legend(fontsize=8)
    ax_histx1.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('../../Figures/3DShapes/Q_S_hist_with_disky.png', bbox_inches='tight', pad_inches=.1, dpi=300)

    # calculate shape diff for disky and non-disky galaxies
    c_diff = C_d - C_s
    b_diff = B_d - B_s
    t_diff = T_d - T_s

    c_diff = C_d/C_s
    b_diff = B_d/B_s
    t_diff = T_d/T_s
    Clabel = f'Q_D/Q_*'
    Blabel = f'S_D/S_*'
    Tlabel = f'T_D/T_*'

    #print avg and std of shape differences for disky and non-disky galaxies to 2 decimal places
    print(f'Disky galaxies:     {Clabel}  mean: {np.mean(c_diff[disky_mask]):.2f}, std: {np.std(c_diff[disky_mask]):.2f}')
    print(f'Non-Disky galaxies: {Clabel}  mean: {np.mean(c_diff[~disky_mask]):.2f}, std: {np.std(c_diff[~disky_mask]):.2f}')
    print(f'Disky galaxies:     {Blabel}  mean: {np.mean(b_diff[disky_mask]):.2f}, std: {np.std(b_diff[disky_mask]):.2f}')
    print(f'Non-Disky galaxies: {Blabel}  mean: {np.mean(b_diff[~disky_mask]):.2f}, std: {np.std(b_diff[~disky_mask]):.2f}')
    print(f'Disky galaxies:     {Tlabel}  mean: {np.mean(t_diff[disky_mask]):.2f}, std: {np.std(t_diff[disky_mask]):.2f}')
    print(f'Non-Disky galaxies: {Tlabel}  mean: {np.mean(t_diff[~disky_mask]):.2f}, std: {np.std(t_diff[~disky_mask]):.2f}')
    
    #recreate T vs Mstar plot with disky and non-disky galaxies
    f, ax = plt.subplots(2, 1, figsize=(15, 6))
    plt.subplots_adjust(hspace=0)

    # triaxial markers
    for i in [0, 1]:
        ax[i].set_xlim([5.77, 9.5])
        ax[i].set_ylim([0, 1])
        ax[i].plot([4, 9.5], [1 / 3, 1 / 3], c='.75', linestyle='--', zorder=0)
        ax[i].plot([4, 9.5], [2 / 3, 2 / 3], c='.75', linestyle='--', zorder=0)
        ax[i].tick_params(which='both', labelsize=15)
        ax[i].text(5.8, 1 / 6, 'Oblate', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
        ax[i].text(5.8, 3 / 6, 'Triaxial', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
        ax[i].text(5.8, 5 / 6, 'Prolate', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
    ax[0].set_ylabel('T', fontsize=20)
    ax[1].set_ylabel(r'T$_*$', fontsize=20)
    ax[0].set_yticks([0, .5, 1])
    ax[1].set_yticks([0, .5])
    ax[1].set_xlabel(r'Log(M$_*$/M$_\odot$)', fontsize=20)
    # ax[0].xaxis.set_tick_params(labelbottom='False')
    ax[0].set_xticks([])

    # T*-TDM vs Mass
    # top
    for i in np.arange(len(masses)):
        ax[0].axvline(masses[i], ymin=min([T_d[i], T_s[i]]), ymax=max([T_d[i], T_s[i]]), c='.5', zorder=0)  # links


    ax[0].scatter(0, 0, c=disky_color, marker='o', label='Disky')
    ax[0].scatter(0, 0, c=non_disky_color, marker='o', label='Non-Disky')

    ax[0].scatter(masses[disky_mask], T_d[disky_mask], c=disky_color, label='Dark Matter', marker='o')
    ax[0].scatter(masses[disky_mask], T_s[disky_mask], c=disky_color, label='Stellar', marker='v')
    ax[0].scatter(masses[~disky_mask], T_d[~disky_mask], c=non_disky_color, marker='o')
    ax[0].scatter(masses[~disky_mask], T_s[~disky_mask], c=non_disky_color, marker='v')



    ax[0].legend()
    # ax[0].legend(
    #     prop={'size': 15},
    #     ncol=2,
    #     loc='center left',
    #     bbox_to_anchor=(0.02, 0.1),
    #     columnspacing=0.8,  # Adjust this value to decrease spacing between columns
    #     handlelength=2,  # Adjust this value to decrease the handle length
    # )
    # T* vs Mass colored by the ratio of M(baryonic) / virial(mass) within the effetive radius
    # bottom

    vmin = np.min(mb)
    vmax = np.max(mb)
    norm = plt.Normalize(vmin, vmax)
    p = ax[1].scatter(masses[disky_mask], T_s[disky_mask], cmap='viridis', c=mb[disky_mask], norm=norm, marker='o',s=100,edgecolors='k',label='Disky')
    ax[1].scatter(masses[~disky_mask], T_s[~disky_mask], c=mb[~disky_mask], cmap='viridis', norm=norm, marker='s',label='Non-Disky')
    cbar = f.colorbar(p, cax=f.add_axes([.91, .11, .03, .77]))
    cbar.set_label(r'M$_{bary}$/M$_{vir}(<$R$_{eff}$)', fontsize=25)
    # set ticks to every step size only within the range that satisify

    # cbar.set_ticks(np.arange(tick_min, tick_max + step, step))
    # cbar.set_ticks([-3, -2.5, -2, -1.5, -1])
    cbar.ax.tick_params(labelsize=15)
    ax[1].legend()


    #plt.close(fig)


def MvC_diffBdiff_disks(B_s, C_s, T_s, B_d, C_d, T_d, masses, mb, htype, reff, mvir, feedback_type, disky_mask,
                         mergers=None, show_lines=False, show_scatter=True):

    #plot masses vs C_diff and B_diff for disky and non-disky galaxies

    fig, ax = plt.subplots(1,1 , figsize=(10, 10))
    plt.subplots_adjust(hspace=0)
    plt.grid()

    ax.set_ylabel(r'$Q_{D}/Q_*$', fontsize=20)
    ax.set_xlabel(r'Log(M$_*$/M$_\odot$)', fontsize=20)
    for is_disky in [True, False]:
        mask = disky_mask == is_disky
        color = 'green' if is_disky else 'k'
        ax.scatter(masses[mask], C_d[mask]/C_s[mask], c=color, label='Disky' if is_disky else 'Non-Disky')

        #linear fits with sns.regplot
        sns.regplot(x=masses[mask], y=C_d[mask]/C_s[mask], ax=ax, scatter=False, color=color)
    #add line off plot to show linear fit
    ax.plot([0, 1], [0, 1], c='k', linestyle='-', label='Linear Fit')
    #add horizontal line at y=1
    ax.axhline(1, c='k', linestyle='--')
    ax.grid(True)
    ax.tick_params(which='both', labelsize=15)
    ax.set_ylim([0.5, 2.4])
    ax.set_xlim(min(masses)-.03, max(masses)+.03)

        
    ax.legend()
    plt.savefig('../../Figures/3DShapes/Mass_vs_C_diff_B_diff.png', bbox_inches='tight', pad_inches=.1, dpi=300)
    


# #Convert lists to arrays for htype indexing
# T_d,T_s,masses,mb,htype = np.array(T_d),np.array(T_s),np.array(masses),np.array(mb),np.array(htype)
# B_s,C_s,B_d,C_d = np.array(B_s),np.array(C_s),np.array(B_d),np.array(C_d)
#
# #T* vs Tdm
# f,ax = plt.subplots(1,1,figsize=(5,5))
# ax.set_xlim([0,1])
# ax.set_ylim([0,1])
# ax.fill_between([0,1],[-1/3,2/3],[1/3,4/3],color='0.75',alpha=.3)
# ax.plot([0,1],[0,1],c='0.5',linestyle='--')
# ax.set_ylabel(r'T$_*$',fontsize=15)
# ax.set_xlabel(r'T$_{DM}$',fontsize=15)
#
# norm = plt.Normalize(int(min(masses)),int(max(masses))+.1)
# p = ax.scatter(T_d[htype=='o'],T_s[htype=='o'],marker='o',c=masses[htype=='o'],cmap='viridis',norm=norm)
# ax.scatter(T_d[htype=='v'],T_s[htype=='v'],marker='v',c=masses[htype=='v'],cmap='viridis',norm=norm,label='Satellites')
# cbar = f.colorbar(p,cax=f.add_axes([.91,.11,.03,.77]))
#
# cbar.set_label(r'Log(M$_*$/M$_\odot$)',fontsize=15)
#
# ax.legend(loc='lower left',prop={'size':12})
# f.savefig(f'../../Figures/3DShapes/T_Comparison.png',bbox_inches='tight',pad_inches=.1)
#
#
# #T vs Mstar
# f,ax=plt.subplots(2,1,figsize=(12,6))
# plt.subplots_adjust(hspace=0)
#
# for i in [0,1]:
#     ax[i].set_xlim([5.8,9.5])
#     ax[i].set_ylim([0,1])
#     ax[i].plot([4,9.5],[1/3,1/3],c='.75',linestyle='--',zorder=0)
#     ax[i].plot([4,9.5],[2/3,2/3],c='.75',linestyle='--',zorder=0)
#     ax[i].tick_params(which='both',labelsize=15)
#     ax[i].text(5.83,1/6,'Oblate',fontsize=17,rotation='vertical',verticalalignment='center',c='.5')
#     ax[i].text(5.83,3/6,'Triaxial',fontsize=17,rotation='vertical',verticalalignment='center',c='.5')
#     ax[i].text(5.83,5/6,'Prolate',fontsize=17,rotation='vertical',verticalalignment='center',c='.5')
# ax[0].set_ylabel('T',fontsize=15)
# ax[1].set_ylabel(r'T$_*$',fontsize=15)
# ax[0].set_yticks([0,.5,1])
# ax[1].set_yticks([0,.5])
# ax[1].set_xlabel(r'Log(M$_*$/M$_\odot$)',fontsize=25)
# #ax[0].xaxis.set_tick_params(labelbottom='False')
# ax[0].set_xticks([])
#
# for i in np.arange(len(masses)):
#     ax[0].axvline(masses[i],ymin=min([T_d[i],T_s[i]]),ymax=max([T_d[i],T_s[i]]),c='.5',zorder=0)
# ax[0].scatter(masses[htype=='o'],T_d[htype=='o'],c='k',label='Dark Matter',marker='o')
# ax[0].scatter(masses[htype=='o'],T_s[htype=='o'],c='r',label='Stellar',marker='o')
# ax[0].scatter(masses[htype=='v'],T_d[htype=='v'],c='k',marker='v')
# ax[0].scatter(masses[htype=='v'],T_s[htype=='v'],c='r',marker='v')
# ax[0].scatter(0,0,c='.5',marker='v',label='Satellites')
# ax[0].legend(prop={'size':15},ncol=3,loc='center left', bbox_to_anchor=(0.02,0.1))
#
# norm = plt.Normalize(int(min(mb))-1,-.75)
# p = ax[1].scatter(masses[htype=='o'],T_s[htype=='o'],c=mb[htype=='o'],cmap='viridis',norm=norm,marker='o')
# ax[1].scatter(masses[htype=='v'],T_s[htype=='v'],c=mb[htype=='v'],cmap='viridis',norm=norm,marker='v')
# cbar = f.colorbar(p,cax=f.add_axes([.91,.11,.03,.77]))
# cbar.set_label(r'Log(M$_{bary}$/M$_{vir}(<$R$_{eff}$))',fontsize=25)
# cbar.set_ticks([-3,-2.5,-2,-1.5,-1])
# cbar.ax.tick_params(labelsize=15)
#
# f.savefig(f'../../Figures/3DShapes/TvsMstar.png',bbox_inches='tight',pad_inches=.1)
#
#
# #B/A vs C/A links
# f,ax = plt.subplots(1,1,figsize=(5,5))
# ax.set_xlim([0,1])
# ax.set_ylim([0,1])
# #ax.fill_between([0,1],[-.1,.9],[.1,1.1],color='0.75',alpha=.3)
# ax.plot([0,1],[0,1],c='0.5',linestyle='--')
# ax.set_xlabel(r'$S$',fontsize=20)
# ax.set_ylabel(r'$Q$',fontsize=20)
# ax.tick_params(which='both',labelsize=15)
# ax.scatter(-1,-1)
#
# for i in np.arange(len(B_s)):
#     ax.plot([B_s[i],B_d[i]],[C_s[i],C_d[i]],c='.5',zorder=0)
#
# f.savefig(f'../../Figures/3DShapes/CvB.LinksOnly.png',bbox_inches='tight',pad_inches=.1)
#
# ax.scatter(B_d[htype=='o'],C_d[htype=='o'],c='k',label='Dark Matter')
# ax.scatter(B_d[htype=='v'],C_d[htype=='v'],c='k',marker='v')
# ax.scatter(B_s[htype=='o'],C_s[htype=='o'],c='r',label='Stellar')
# ax.scatter(B_s[htype=='v'],C_s[htype=='v'],c='r',marker='v')
# ax.scatter(-1,-1,c='.5',marker='v',label='Satellites')
#
# ax.legend(loc='upper left',prop={'size':15})
# f.savefig(f'../../Figures/3DShapes/CvB.Links.png',bbox_inches='tight',pad_inches=.1)
