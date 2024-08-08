import os
import pickle
import sys
import pynbody
import numpy as np
import pathlib
import traceback
import gc
from contextlib import contextmanager
from astropy import units as u
from astropy import constants as const
import pymp

import os
import pickle
import sys
import pynbody
import numpy as np
import pathlib
import traceback
import gc
from contextlib import contextmanager
from astropy import units as u
from astropy import constants as const
import pymp

# Add the path to the directory containing the SimInfoDicts package to the system path
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from SimInfoDicts.sim_type_name import sim_type_name

import signal
import time

def timeout_handler(signum, frame):
    raise TimeoutError("Decomposition analysis timed out")


"""

decomp
======

Tools for bulge/disk/halo decomposition

"""

import logging

import numpy as np

from pynbody import filt, array, util, config
from pynbody.analysis import angmom, profile

logger = logging.getLogger('pynbody.analysis.decomp')


def decomp(h, aligned=False, j_disk_min=0.8, j_disk_max=1.1, E_cut=None, j_circ_from_r=False,
           cen=None, vcen=None, log_interp=False, angmom_size="3 kpc"):
    """
    Creates an array 'decomp' for star particles in the simulation, with an
    integer specifying a kinematic decomposition. The possible values are:

    1 -- thin disk

    2 -- halo

    3 -- bulge

    4 -- thick disk

    5 -- pseudo bulge

    This routine is based on an original IDL procedure by Chris Brook.


    **Parameters:**

    *h* -- the halo to work on

    *j_disk_min* -- the minimum angular momentum as a proportion of
                  the circular angular momentum which a particle must
                  have to be part of the 'disk'

    *j_disk_max* -- the maximum angular momentum as a proportion of
                  the circular angular momentum which a particle can
                  have to be part of the 'disk'

    *E_cut* -- the energy boundary between bulge and spheroid. If
             None, this is taken to be the median energy of the stars.

    *aligned* -- if False, the simulation is recenterd and aligned so
               the disk is in the xy plane as required for the rest of
               the analysis.

    *cen* -- if not None, specifies the center of the halo. Otherwise
           it is found.  This has no effect if aligned=True

    *vcen* -- if not None, specifies the velocity center of the
            halo. Otherwise it is found.  This has no effect if
            aligned=True

    *j_circ_from_r* -- if True, the maximum angular momentum is
    determined as a function of radius, rather than as a function of
    orbital energy. Default False (determine as function of energy).

    *angmom_size* -- the size of the gas sphere used to determine the
     plane of the disk

    """

    import scipy.interpolate as interp
    global config

    # Center, eliminate proper motion, rotate so that
    # gas disk is in X-Y plane
    if not aligned:
        angmom.faceon(h, cen=cen, vcen=vcen, disk_size=angmom_size)

    # Find KE, PE and TE
    ke = h['ke']
    pe = h['phi']

    h['phi'].convert_units(ke.units)  # put PE and TE into same unit system

    te = ke + pe
    h['te'] = te
    te_star = h.star['te']

    te_max = te_star.max()

    # Add an arbitrary offset to the PE to reflect the idea that
    # the group is 'fully bound'.
    te -= te_max
    logger.info("te_max = %.2e" % te_max)

    h['te'] -= te_max

    logger.info("Making disk rotation curve...")

    # Now make a rotation curve for the disk. We'll take everything
    # inside a vertical height of eps*3

    d = h[filt.Disc('1 Mpc', h['eps'].min() * 3)]

    try:

        # attempt to load rotation curve off disk
        r, v_c = np.loadtxt(h.ancestor.filename + ".rot." +
                            str(h.properties["halo_id"]), skiprows=1, unpack=True)

        pro_d = profile.Profile(d, nbins=100, type='log')
        r_me = pro_d["rbins"].in_units("kpc")
        r_x = np.concatenate(([0], r, [r.max() * 2]))
        v_c = np.concatenate(([v_c[0]], v_c, [v_c[-1]]))
        v_c = interp.interp1d(r_x, v_c, bounds_error=False)(r_me)

        logger.info(" - found existing rotation curve on disk, using that")

        v_c = v_c.view(array.SimArray)
        v_c.units = "km s^-1"
        v_c.sim = d

        v_c.convert_units(h['vel'].units)

        pro_d._profiles['v_circ'] = v_c
        pro_d.v_circ_loaded = True

    except Exception:
        pro_d = profile.Profile(d, nbins=100, type='log')  # .D()
        # Nasty hack follows to force the full halo to be used in calculating the
        # gravity (otherwise get incorrect rotation curves)
        pro_d._profiles['v_circ'] = profile.v_circ(pro_d, h)

    pro_phi = pro_d['phi']
    # import pdb; pdb.set_trace()
    # offset the potential as for the te array
    pro_phi -= te_max

    # (will automatically be reflected in E_circ etc)
    # calculating v_circ for j_circ and E_circ is slow

    if j_circ_from_r:
        pro_d.create_particle_array("j_circ", out_sim=h)
        pro_d.create_particle_array("E_circ", out_sim=h)
    else:

        if log_interp:
            j_from_E = interp.interp1d(
                np.log10(-pro_d['E_circ'].in_units(ke.units))[::-1], np.log10(pro_d['j_circ'])[::-1],
                bounds_error=False)
            h['j_circ'] = 10 ** j_from_E(np.log10(-h['te']))
        else:
            #            j_from_E  = interp.interp1d(-pro_d['E_circ'][::-1], (pro_d['j_circ'])[::-1], bounds_error=False)
            j_from_E = interp.interp1d(
                pro_d['E_circ'].in_units(ke.units), pro_d['j_circ'], bounds_error=False)
            h['j_circ'] = j_from_E(h['te'])

        # The next line forces everything close-to-unbound into the
        # spheroid, as per CB's original script ('get rid of weird
        # outputs', it says).
        h['j_circ'][np.where(h['te'] > pro_d['E_circ'].max())] = np.inf

        # There are only a handful of particles falling into the following
        # category:
        h['j_circ'][np.where(h['te'] < pro_d['E_circ'].min())] = pro_d[
            'j_circ'][0]

    h['jz_by_jzcirc'] = h['j'][:, 2] / h['j_circ']
    h_star = h.star

    if 'decomp' not in h_star:
        h_star._create_array('decomp', dtype=int)
    disk = np.where(
        (h_star['jz_by_jzcirc'] > j_disk_min) * (h_star['jz_by_jzcirc'] < j_disk_max))

    h_star['decomp', disk[0]] = 1
    # h_star = h_star[np.where(h_star['decomp']!=1)]

    # Find disk/spheroid angular momentum cut-off to make spheroid
    # rotational velocity exactly zero.

    V = h_star['vcxy']
    JzJcirc = h_star['jz_by_jzcirc']
    te = h_star['te']

    logger.info("Finding spheroid/disk angular momentum boundary...")

    j_crit = util.bisect(0., 5.0,
                         lambda c: np.mean(V[np.where(JzJcirc < c)]))
    # print(j_crit)

    logger.info("j_crit = %.2e" % j_crit)

    if j_crit > j_disk_min:
        logger.warning(
            "!! j_crit exceeds j_disk_min. This is usually a sign that something is going wrong (train-wreck galaxy?)")
        logger.warning("!! j_crit will be reset to j_disk_min=%.2e" % j_disk_min)
        j_crit = j_disk_min

    sphere = np.where(h_star['jz_by_jzcirc'] < j_crit)

    if E_cut is None:
        E_cut = np.median(h_star['te'])

    logger.info("E_cut = %.2e" % E_cut)

    halo = np.where((te > E_cut) * (JzJcirc < j_crit))
    bulge = np.where((te <= E_cut) * (JzJcirc < j_crit))
    pbulge = np.where((te <= E_cut) * (JzJcirc > j_crit)
                      * ((JzJcirc < j_disk_min) + (JzJcirc > j_disk_max)))
    thick = np.where((te > E_cut) * (JzJcirc > j_crit)
                     * ((JzJcirc < j_disk_min) + (JzJcirc > j_disk_max)))

    # h_star['decomp', disk] = 1
    h_star['decomp', halo] = 2
    h_star['decomp', bulge] = 3
    h_star['decomp', thick] = 4
    h_star['decomp', pbulge] = 5

    # Return profile object for informational purposes
    return pro_d, j_crit


def check_halo_data_completeness(halo_data):
    required_keys = [
        'Mvir', 'Mstar', 'Mgas', 'Mb/Mtot', 'HI',
        'Mvir_within_reff', 'Mstar_within_reff', 'Mgas_within_reff', 'Mb/Mtot_within_reff', 'HI_within_reff',
        'Mvir_within_tenth_rvir', 'Mstar_within_tenth_rvir', 'Mgas_within_tenth_rvir', 'Mb/Mtot_within_tenth_rvir', 'HI_within_tenth_rvir',
        'Reff', 'Rvir', 'dt_decomp', 'dt_star', 'dt_gas', 'dt_total',
        't_dyn_rvir', 'rstar', 'M_within_star', 't_dyn_rstar', 'jz_jcirc_avg', 'j_crit',
    ]
    return all(key in halo_data and halo_data[key] is not None and not np.isnan(halo_data[key]) for key in required_keys)

def calculate_dynamical_time(r_vir, M_halo):
    r_vir = r_vir * u.kpc
    M_halo = M_halo * u.solMass
    t_dyn = np.sqrt(r_vir ** 3 / (const.G * M_halo))
    return t_dyn.to(u.Gyr).value


def mass_properties_within_r(halo, r):
    halo.physical_units()
    sphere = halo[pynbody.filt.Sphere(r)]

    m_tot = sphere['mass'].sum().in_units('Msol')
    m_gas = sphere.gas['mass'].sum().in_units('Msol')
    m_star = sphere.star['mass'].sum().in_units('Msol')
    m_dm = sphere.dm['mass'].sum().in_units('Msol')
    m_vir_within_r = m_gas + m_star + m_dm

    assert m_star > 1, f"Star mass is {m_star}"
    assert m_dm > 1, f"DM mass is {m_dm}"
    assert np.isclose(m_tot, m_vir_within_r,
                      rtol=1e-10), f"Total mass is {m_tot}, sum of components is {m_vir_within_r}"

    Mb_within_r = m_gas + m_star
    mb_mvir_within_r = Mb_within_r / m_vir_within_r
    try:
        HI_within_r = sphere.gas['HI'].sum().in_units('Msol')
    except:
        HI_within_r = np.nan

    return m_vir_within_r, m_star, m_gas, mb_mvir_within_r, HI_within_r


@contextmanager
def load_simulation(simpath):
    sim = pynbody.load(simpath)
    sim.physical_units()
    try:
        yield sim
    finally:
        del sim
        gc.collect()


def process_halo(halo, hid, Profiles, timeout=600):  # 5 minutes timeout by default
    try:
        pynbody.analysis.halo.center(halo, mode='hyb')
    except:
        try:
            pynbody.analysis.halo.center(halo, mode='ssc')
        except:
            print(f"Failed to center halo {hid}")
            return None

    rvir = max(halo['r'])
    mvir = halo['mass'].sum()
    mstar = halo.star['mass'].sum()
    mgas = halo.gas['mass'].sum()
    mb_mvir = (mstar + mgas) / mvir

    rvir, Mvir, Mstar, Mgas = (pynbody.array.SimArray(x, 'kpc' if i == 0 else 'Msol')
                               for i, x in enumerate([rvir, mvir, mstar, mgas]))

    Reff = pynbody.array.SimArray(Profiles[str(hid)]['x000y000']['Reff'], 'kpc')

    mass_calculations = [
        ('', rvir),
        ('_within_reff', Reff),
        ('_within_tenth_rvir', rvir / 10)
    ]

    results = {}
    for suffix, r in mass_calculations:
        try:
            Mvir, Mstar, Mgas, mb_mvir, HI = mass_properties_within_r(halo, r)
            results.update({
                f'Mvir{suffix}': Mvir.tolist(),
                f'Mstar{suffix}': Mstar.tolist(),
                f'Mgas{suffix}': Mgas.tolist(),
                f'Mb/Mtot{suffix}': float(mb_mvir),
                f'HI{suffix}': HI.tolist() if not np.isnan(HI) else np.nan
            })
        except Exception as e:
            print(f"Failed to calculate mass{suffix} for halo {hid}: {e}")
            results.update({k: np.nan for k in
                            [f'Mvir{suffix}', f'Mstar{suffix}', f'Mgas{suffix}', f'Mb/Mtot{suffix}', f'HI{suffix}']})

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        start_time = time.time()

        pynbody.analysis.angmom.faceon(halo)
        results['faceon_time'] = time.time() - start_time

        angmom_size = str(Reff * 3) + ' kpc'
        pro_d, jcrit = decomp(halo, aligned=True, j_disk_min=.7, j_disk_max=1.2, angmom_size=angmom_size)
        results['decomp_time'] = time.time() - start_time

        mdisk = sum(halo.s['mass'][halo.s['decomp'] == 1])
        mthick = sum(halo.s['mass'][halo.s['decomp'] == 4])
        dt_decomp = (mdisk + mthick) / Mstar

        mdisk_star = (halo.s['mass'][halo.s['jz_by_jzcirc'] > jcrit]).sum()
        mdisk_gas = (halo.g['mass'][halo.g['jz_by_jzcirc'] > jcrit]).sum()

        mdisk_star_counter = (halo.s['mass'][halo.s['jz_by_jzcirc'] < -jcrit]).sum()
        mdisk_gas_counter = (halo.g['mass'][halo.g['jz_by_jzcirc'] < -jcrit]).sum()
        mdisk_star = mdisk_star - mdisk_star_counter
        mdisk_gas = mdisk_gas - mdisk_gas_counter

        dt_star = (mdisk) / mstar
        dt_gas = (mdisk_gas) / mgas

        dt_total = (mdisk_star + mdisk_gas) / (mstar + mgas)

        t_dyn_rvir = calculate_dynamical_time(rvir, Mvir)
        rstar = np.sort(halo.s['r'])[-10]
        M_within_star = halo['mass'][halo['r'] < rstar].sum()
        t_dyn_rstar = calculate_dynamical_time(rstar, M_within_star)

        jz_jcirc = halo.s['jz_by_jzcirc']
        jz_jcirc = jz_jcirc[np.isfinite(jz_jcirc)]
        jz_jcirc_avg = np.mean(jz_jcirc)

        results.update({
            'Reff': Reff,
            'Rvir': rvir,
            'dt_decomp': dt_decomp,
            'dt_star': dt_star,
            'dt_gas': dt_gas,
            'dt_total': dt_total,
            't_dyn_rvir': t_dyn_rvir,
            'rstar': rstar,
            'M_within_star': M_within_star,
            't_dyn_rstar': t_dyn_rstar,
            'jz_jcirc_avg': jz_jcirc_avg,
            'j_crit': jcrit
        })


    except TimeoutError as e:
        print(f"Timeout occurred for halo {hid}: {e}")
        print(f"Last completed step: {max(results.keys(), key=lambda k: results[k] if isinstance(results[k], (int, float)) else 0)}")
        for key in ['dt_decomp', 'dt_star', 'dt_gas', 'dt_total', 't_dyn_rvir', 'rstar', 'M_within_star', 't_dyn_rstar', 'jz_jcirc_avg']:
            if key not in results:
                results[key] = np.nan
    finally:
        signal.alarm(0)  # Disable the alarm

    return results

def save_progress(simulation, masses, feedback):
    checkpoint_file = f"../../Data/BasicData/{feedback}.{simulation}.Masses.checkpoint.pickle"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(masses, f)


def load_progress(simulation, feedback):
    checkpoint_file = f"../../Data/BasicData/{feedback}.{simulation}.Masses.checkpoint.pickle"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return {}


def process_simulation(simulation, SimInfo, Profiles, feedback):
    simpath = SimInfo[simulation]['path']
    halos = SimInfo[simulation]['goodhalos']

    if not os.path.exists(simpath):
        print(f"Path does not exist: {simpath}")
        return {}

    masses = load_progress(simulation, feedback)

    with load_simulation(simpath) as sim:
        h = sim.halos()

        with pymp.Parallel(num_threads=2) as p:
            for hid in p.range(len(halos)):
                halo_id = str(halos[hid])
                if halo_id in masses and check_halo_data_completeness(masses[halo_id]):
                    print(f"Skipping already processed and complete halo {halo_id} in simulation {simulation}")
                    continue
                try:
                    halo = h[halos[hid]]
                    result = process_halo(halo, halos[hid], Profiles)
                    if result and check_halo_data_completeness(result):
                        masses[halo_id] = result
                        save_progress(simulation, masses, feedback)
                    else:
                        print(f"Incomplete data for halo {halo_id} in simulation {simulation}. Will reprocess next time.")
                    del halo
                except Exception as e:
                    del halo
                    print(f"Failed to process halo {halos[hid]} in simulation {simulation}: {e}")
                    print(traceback.format_exc())
        del h


def main():
    for feedback, use_sim in sim_type_name.items():
        if feedback == 'MerianCDM':
            continue

        if not use_sim:
            continue

        print(f'Calculating masses for {feedback} feedback type.')
        pickle_path = f'../PickleFiles/SimulationInfo.{feedback}.pickle'

        if not os.path.exists(pickle_path):
            print(f"No pickle file found for {feedback} feedback type.")
            continue

        with open(pickle_path, 'rb') as f:
            SimInfo = pickle.load(f)

        all_sim_masses = {}

        for simulation in SimInfo:
            profile_path = f'../../Data/{simulation}.{feedback}.Profiles.pickle'
            if not os.path.exists(profile_path):
                print(f"No profile file found for {feedback} feedback type in location {profile_path}.")
                continue

            with open(profile_path, 'rb') as f:
                Profiles = pickle.load(f)

            all_sim_masses[simulation] = process_simulation(simulation, SimInfo, Profiles, feedback)

        with open(f"../../Data/BasicData/{feedback}.Masses.pickle", 'wb') as f:
            pickle.dump(all_sim_masses, f)

        print(f"Finished calculating masses for all simulations runs {feedback}. Results saved")


if __name__ == '__main__':
    #wait 4 hours
    #print current time
    print(time.ctime())
    #print projected start time
    #print(time.ctime(time.time() + 4*60*60))
    #time.sleep(4*60*60)
    main()