#dictonary for BW sims
marvel_path = '/myhome2/users/munshi/dwarf_volumes/'
dcjl_path = '/myhome2/users/munshi/e12gals/'
z_romulus_path = '/data/REPOSITORY/e11Gals'

Sims = {
	#Sim name for image/file naming purposes
    'cptmarvel' : {
        #Path to sim data
        'path' : marvel_path+'cptmarvel.cosmo25cmb.4096g5HbwK1BH/cptmarvel.cosmo25cmb.4096g5HbwK1BH.004096/cptmarvel.cosmo25cmb.4096g5HbwK1BH.004096',
        #IDs of resolved halos in sim
        'halos' : [1,2,3,5,6,7,10,11,13,14,24],
        #IDs of halos that are valid for all morphology methods
        #These are the halos that will have saved data/images and contribute to figures
        'goodhalos' : [1,2,3,5,6,7,10]
    },
    'elektra' : {
        'path' : marvel_path+'elektra.cosmo25cmb.4096g5HbwK1BH/elektra.cosmo25cmb.4096g5HbwK1BH.004096/elektra.cosmo25cmb.4096g5HbwK1BH.004096',
        'halos' : [1,2,3,4,5,8,9,10,11,12,17,36,64],
        'goodhalos' : [1,2,3,4,5,9,10]
    },
    'storm' : {
        'path' : marvel_path+'storm.cosmo25cmb.4096g5HbwK1BH/storm.cosmo25cmb.4096g5HbwK1BH.004096/storm.cosmo25cmb.4096g5HbwK1BH.004096',
        'halos' : [1,2,3,4,5,6,7,8,10,11,12,14,15,22,23,31,37,44,48,55,118],
        'goodhalos' : [1,2,3,4,5,6,7,8,14,31]
    },
    'rogue' : {
        'path' : marvel_path+'rogue.cosmo25cmb.4096g5HbwK1BH/rogue.cosmo25cmb.4096g5HbwK1BH.004096/rogue.cosmo25cmb.4096g5HbwK1BH.004096',
        'halos' : [1,3,7,8,10,11,12,15,16,17,28,31,37,58,116],
        'goodhalos' : [1,3,7,8,10,11,12,15,28]
    },
    'h148' : {
        'path' : dcjl_path+'h148.cosmo50PLK.3072g3HbwK1BH/h148.cosmo50PLK.3072g3HbwK1BH.004096/h148.cosmo50PLK.3072g3HbwK1BH.004096',
        'halos' : [2,3,4,6,7,11,12,13,15,20,23,27,28,29,33,34,37,38,41,43,51,59,65,75,86,94,109,114,122],
        'goodhalos' : [2,3,4,6,7,11,12,23,27,38,65]
    },
    'h229' : {
        'path' : dcjl_path+'h229.cosmo50PLK.3072gst5HbwK1BH/h229.cosmo50PLK.3072gst5HbwK1BH.004096/h229.cosmo50PLK.3072gst5HbwK1BH.004096',
        'halos' : [2,3,6,14,15,18,20,22,25,33,47,48,49,52,57,62,89,92,127],
        'goodhalos' : [2,3,6,18]
    },
    'h242' : {
        'path' : dcjl_path+'h242.cosmo50PLK.3072gst5HbwK1BH/h242.cosmo50PLK.3072gst5HbwK1BH.004096/h242.cosmo50PLK.3072gst5HbwK1BH.004096',
        'halos' : [8,10,21,26,30,34,38,42,44,45,63,70,81,138],
        'goodhalos' : [8,10]
    },
    'h329' : {
        'path' : dcjl_path+'h329.cosmo50PLK.3072gst5HbwK1BH/h329.cosmo50PLK.3072gst5HbwK1BH.004096/h329.cosmo50PLK.3072gst5HbwK1BH.004096',
        'halos' : [7,9,29,30,37,53,115,117,127],
        'goodhalos' : [7]
    }
}
