import numpy as np
import pynbody
from SimInfoDicts.MerianCDM import Sims
#from SimInfoDicts.BWMDC import Sims
import traceback


sim = Sims['r431']
s = pynbody.load(sim['path'])
s.physical_units()
h = s.halos()

halo = h[1]
pynbody.analysis.angmom.faceon(halo)

Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
prof = pynbody.analysis.profile.Profile(halo.s, type='lin', min=.25, max=5 * Rhalf, ndim=2,
                                        nbins=int((5 * Rhalf) / 0.1))
r = prof['rbins']


profile_tests = ['sb,V']

for test in profile_tests:
    try:
        sb = prof[test]
        print(f'{test} done')
    except:
        print(f'{test} failed')