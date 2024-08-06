import os
import pickle
import sys
import pynbody
import numpy as np
import pathlib
import traceback
import gc


# Add the path to the directory containing the SimInfoDicts package to the system path
sys.path.append(str(pathlib.Path(__file__).parent.parent))
# Import the SimInfoDicts package
from SimInfoDicts.sim_type_name import sim_type_name

sim_type_name = {
    'MerianSIDM': False,
    'DOstormCDM': False,
    'DOstormSIDM': False,
    'MerianCDM': False,
    'SBMarvel': False,
    'BWMDC': True,
    'Storm': False
}

from astropy import units as u
from astropy import constants as const
import numpy as np

def calculate_dynamical_time(r_vir, M_halo):
    # Convert input parameters to Astropy quantities
    r_vir = r_vir * u.kpc
    M_halo = M_halo * u.solMass
    r_vir = r_vir.to(u.kpc)
    M_halo = M_halo.to(u.solMass)

    # Calculate dynamical time
    t_dyn = np.sqrt(r_vir**3 / (const.G * M_halo))

    return t_dyn.to(u.Gyr).value


def mass_properties_within_r(halo, r):
    # halo should be in physcial units, but just in case
    halo.physical_units()

    sphere_filter = pynbody.filt.Sphere(r)
    sphere = halo[sphere_filter]

    m_tot = (sphere['mass'].sum().in_units('Msol'))
    m_gas = (sphere.gas['mass'].sum().in_units('Msol'))
    m_star = (sphere.star['mass'].sum().in_units('Msol'))
    m_dm = (sphere.dm['mass'].sum().in_units('Msol'))
    m_vir_within_r = m_gas + m_star + m_dm
    # assert that all of these values are positive, and not close to 0 they are stored as pynbody SimArrays in units of solar masses
    assert m_star > 1, f"Star mass is {m_star}"
    assert m_dm > 1, f"DM mass is {m_dm}"
    # assert that m_tot is the sum of the other masses within floating point error
    assert np.isclose(m_tot, m_vir_within_r,
                      rtol=1e-10), f"Total mass is {m_tot}, sum of components is {m_gas + m_star + m_dm}"

    Mb_within_r = m_gas + m_star
    mb_mvir_within_r = Mb_within_r / m_vir_within_r
    try:
        HI_within_r = (sphere.gas['HI'].sum().in_units('Msol'))
    except:
        HI_within_r = np.copy(m_vir_within_r) * np.nan

    return m_vir_within_r, m_star, m_gas, mb_mvir_within_r, HI_within_r


# generaize to work with any SimInfo
# Load simulation info from the specific RDZ pickle file
def main():
    verbose = False
    for feedback, use_sim in sim_type_name.items():
        # see if mass file already exists
        if os.path.exists(f'../../Data/BasicData/{feedback}.Masses.pickle'):
            print(f'Mass file already exists for {feedback} feedback type.')
            # load it
            # all_sim_masses = pickle.load(open(f'../../Data/BasicData/{feedback}.Masses.pickle', 'rb'))
            # check if it has all the sims

        if use_sim:
            print(f'Calculating masses for {feedback} feedback type.')
            pickle_path = f'../PickleFiles/SimulationInfo.{feedback}.pickle'
            if os.path.exists(pickle_path):
                SimInfo = pickle.load(open(pickle_path, 'rb'))
                all_sim_masses = {}  # This will hold mass data for all simulations
                # add a check to see if the mass file already exists, if it does, we need to make sure it has all the current sims

                for simulation in SimInfo:

                    # load profiles to get reff
                    profile_path = f'../../Data/{simulation}.{feedback}.Profiles.pickle'
                    if os.path.exists(profile_path):
                        Profiles = pickle.load(open(profile_path, 'rb'))
                        compute_Rhalf = False
                    else:
                        print(f"No profile file found for {feedback} feedback type in location {profile_path}.")
                        continue
                    # check if simulation is in all_sim_masses
                    # and all good halos have been processed
                    if simulation in all_sim_masses:
                        if len(all_sim_masses[simulation]) == len(SimInfo[simulation]['goodhalos']):
                            ##check that all the masses are there
                            if all('Mb/Mtot_within_reff' in all_sim_masses[simulation][str(hid)] for hid in
                                   SimInfo[simulation]['goodhalos']):
                                if verbose:
                                    print(f"Masses already calculated for {simulation}.")
                                # continue

                    simpath = SimInfo[simulation]['path']
                    halos = SimInfo[simulation]['goodhalos']
                    # Check if simulation path exists
                    if not os.path.exists(simpath):
                        if verbose:
                            print(f"Path does not exist: {simpath}")
                        continue
                    sim = pynbody.load(simpath)
                    sim.physical_units()
                    if verbose:
                        print(f"Simulation {simulation} loaded successfully.")

                    # Prepare to calculate masses
                    h = sim.halos()
                    try:
                        ahf_halos = pynbody.halo.ahf.AHFCatalogue(sim)
                        ahf_halos.physical_units()
                    except:
                        print(f"Failed to load AHF halos for simulation {simulation}.")
                        print(traceback.format_exc())
                        ahf_halos = None
                    # get list of halo ids AmigaGrpCatalogue' object

                    masses = {}  # This will hold mass data for the current simulation

                    # Sequentially calculate masses
                    for hid in halos:
                        # check if sim, hid already in all_sim_masses
                        # if simulation in all_sim_masses:
                        # if str(hid) in all_sim_masses[simulation]:
                        # check that all the masses are there
                        # if 'Mb/Mtot_within_reff' in all_sim_masses[simulation][str(hid)]:
                        # if verbose:
                        #    print(f"Mass already calculated for halo {hid} in simulation {simulation}.")
                        # continue

                        try:
                            halo = h[hid]

                            if verbose:
                                print(f"Centering halo {hid} in simulation {simulation}.")
                            try:
                                try:
                                    pynbody.analysis.halo.center(halo, mode='hyb')
                                except:
                                    print('Failed to center halo with mode hyb')
                                    pynbody.analysis.halo.center(halo, mode='ssc')
                            except:
                                print(traceback.format_exc())
                                print(f"Failed to center halo {hid} in simulation {simulation}.")
                                continue
                            # calculate virial radius
                            try:
                                rvir = max(halo['r'])
                                mvir = (halo['mass'].sum())
                                mstar = (halo.star['mass'].sum())
                                mgas = (halo.gas['mass'].sum())
                                mb_mvir = (mstar + mgas) / mvir


                                rvir, Mvir, Mstar, Mgas = pynbody.array.SimArray(rvir, 'kpc'), pynbody.array.SimArray(
                                    mvir, 'Msol'), pynbody.array.SimArray(mstar, 'Msol'), pynbody.array.SimArray(mgas,
                                                                                                                 'Msol')
                                HI = np.copy(Mvir) * np.nan
                            except:
                                print(f"Failed to load virial radius for halo {hid} in simulation {simulation}.")
                                print(traceback.format_exc())
                                array = np.ndarray([0]) * np.nan
                                nan_array = pynbody.array.SimArray(array)
                                Mvir, Mstar, Mgas, mb_mvir, HI = nan_array, nan_array, nan_array, nan_array, nan_array

                            # compare ahf Rvir to pynbody Rvir

                            # Mvir, Mstar, Mgas, mb_mvir,HI = mass_properties_within_r(halo, rvir)
                            # create subsnap to get mass within reff

                            Reff = Profiles[str(hid)]['x000y000']['Reff']
                            Reff = pynbody.array.SimArray(Reff, 'kpc')
                            if verbose:
                                print(f"Calculating mass within Reff for halo {hid} in simulation {simulation}.")
                            try:
                                Mvir_within_reff, Mstar_within_reff, Mgas_within_reff, mb_mvir_within_reff, HI_within_reff = mass_properties_within_r(
                                    halo, Reff)
                            except:
                                print(
                                    f"Failed to calculate mass within Reff for halo {hid} in simulation {simulation}.")
                                print(traceback.format_exc())
                                Mvir_within_reff, Mstar_within_reff, Mgas_within_reff, mb_mvir_within_reff, HI_within_reff = np.nan, np.nan, np.nan, np.nan, np.nan
                            #print(
                                #f'Baryon fraction within Reff: {mb_mvir_within_reff} for halo {hid} in simulation {simulation}.')

                            #add masses at .1Rvir
                            try:
                                Mvir_within_tenth_rvir, Mstar_within_tenth_rvir, Mgas_within_tenth_rvir, mb_mvir_within_tenth_rvir, HI_within_tenth_rvir = mass_properties_within_r(
                                    halo, rvir/10)
                            except:
                                print(
                                    f"Failed to calculate mass within .1Rvir for halo {hid} in simulation {simulation}.")
                                print(traceback.format_exc())
                                Mvir_within_tenth_rvir, Mstar_within_tenth_rvir, Mgas_within_tenth_rvir, mb_mvir_within_tenth_rvir, HI_within_tenth_rvir = np.nan, np.nan, np.nan, np.nan, np.nan


                            #dynamical timescales
                            halo = h[hid]
                            pynbody.analysis.angmom.faceon(halo)
                            pynbody.analysis.decomp(halo)
                            # get Disk-to-Total Ratio (D/T) for each halo
                            mdisk = sum(halo.s['mass'][halo.s['decomp'] == 1])
                            mthick = sum(halo.s['mass'][halo.s['decomp'] == 4])
                            dt = (mdisk + mthick) / mstar
                            print(dt)

                            # get dynamical time for each halo at virial radius, and 10th largest stellar radius
                            t_dyn_rvir = calculate_dynamical_time(rvir, Mvir)
                            rstar = np.sort(halo.s['r'])[-10]
                            M_within_star = halo['mass'][halo['r'] < rstar].sum()
                            t_dyn_rstar = calculate_dynamical_time(rstar, M_within_star)
                            # store all values into dict



                            # Store computed values in the masses dictionary
                            masses[str(hid)] = {
                                'Mvir': Mvir.tolist(),  # Converting np.ndarray to list for JSON compatibility
                                'Mstar': Mstar.tolist(),
                                'Mgas': Mgas.tolist(),
                                # 'HI': HI.tolist(),
                                'Mb/Mtot': float(mb_mvir),
                                'HI': HI.tolist(),
                                'Mvir_within_reff': Mvir_within_reff.tolist(),
                                'Mstar_within_reff': Mstar_within_reff.tolist(),
                                'Mgas_within_reff': Mgas_within_reff.tolist(),
                                'Mb/Mtot_within_reff': float(mb_mvir_within_reff),
                                'Reff': Reff,
                                'Rvir': rvir,
                                'HI_within_reff': HI_within_reff.tolist(),
                                'Mvir_within_tenth_rvir': Mvir_within_tenth_rvir.tolist(),
                                'Mstar_within_tenth_rvir': Mstar_within_tenth_rvir.tolist(),
                                'Mgas_within_tenth_rvir': Mgas_within_tenth_rvir.tolist(),
                                'Mb/Mtot_within_tenth_rvir': float(mb_mvir_within_tenth_rvir),
                                'HI_within_tenth_rvir': HI_within_tenth_rvir.tolist(),
                                'dt': dt,
                                't_dyn_rvir': t_dyn_rvir,
                                'rstar': rstar,
                                'M_within_star': M_within_star,
                                't_dyn_rstar': t_dyn_rstar


                            }
                            if verbose:
                                print(f"Mass calculated for halo {hid} in simulation {simulation}.")
                        except AssertionError as e:
                            print(f"Failed to calculate mass for halo {hid} in simulation {simulation}: {e}")
                            print(traceback.format_exc())
                        except Exception as e:
                            print(f"Failed to calculate mass for halo {hid} in simulation {simulation}: {e}")
                            if verbose:
                                print(traceback.format_exc())
                    del halo
                    del h
                    del sim
                    gc.collect()


                    # Store the simulation's mass data in the all_sim_masses container
                    all_sim_masses[simulation] = masses

                # print(all_sim_masses)

                # Save the comprehensive results
                with open(f"../../Data/BasicData/{feedback}.Masses.pickle", 'wb') as f:
                    pickle.dump(all_sim_masses, f)

                if verbose:
                    print(f"Finished calculating masses for all simulations runs {feedback}. Results saved")
            else:
                print(f"No pickle file found for {feedback} feedback type.")


if __name__ == '__main__':
    main()