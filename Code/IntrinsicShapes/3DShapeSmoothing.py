import os,pickle
import matplotlib.pylab as plt
from scipy.interpolate import UnivariateSpline as Smooth

SimInfo = pickle.load(open('../SimulationInfo.BW.pickle','rb'))

# loop,remake=True,False
# while loop:
#     rem = input('Remake Image Directory: Images/3DShapes/- (y/n): ')
#     if rem in ['y','n']:
#         loop = False
#         if rem=='y': remake = True
# if remake: 
#     os.system('rmdir -f ../Images/3DShapes')
#     os.system('mkdir ../Images/3DShapes')
#     for sim in SimInfo:
#         os.system(f'mkdir ../Images/3DShapes/{sim}.BW/')
windows = {
    '3D':{
    'cptmarvel-10':[.65,2],
    'elektra-3':[.4,.8],
    'storm-2':[5,40],
    'storm-8':[3,10],
    'rogue-7':[.8,1],
    'rogue-8':[0,1],
    },
    'DM':{
    
    }
}


for t in ['3D','DM']:
    for sim in SimInfo:
        Shapes = pickle.load(open(f'../../Data/{sim}.BW.{t}Shapes.pickle','rb'))
        Profiles = pickle.load(open(f'../../Data/{sim}.BW.Profiles.pickle','rb'))

        for hid in SimInfo[sim]['halos']:
            rbins,ba,ca=Shapes[str(hid)]['rbins'],Shapes[str(hid)]['ba'], Shapes[str(hid)]['ca']
            reffs = []
            if len(rbins)>0:
                for angle in Profiles[str(hid)]:
                    reffs.append(Profiles[str(hid)][angle]['Reff'])
                
                f,ax=plt.subplots(1,1,figsize=(6,2))
                ax.set_xlim([0,max(rbins)])
                ax.set_ylim([0,1])
                ax.set_xlabel('R [kpc]',fontsize=15)
                ax.set_ylabel('Axis Ratio',fontsize=15)
                ax.tick_params(which='both',labelsize=10)

                ax.plot(rbins,ba,c='k',label='B/A')
                ax.plot(rbins,ca,c='k',linestyle='--',label='C/A')
                if f'{sim}-{hid}' in windows[t]:
                    w = windows[t][f'{sim}-{hid}']
                    ba = ba[(rbins<w[0])|(rbins>w[1])]
                    ca = ca[(rbins<w[0])|(rbins>w[1])]
                    rbins = rbins[(rbins<w[0])|(rbins>w[1])]
                ba_s,ca_s = Smooth(rbins,ba,k=3),Smooth(rbins,ca,k=3)
                ax.plot(rbins,ba_s(rbins),c='r')
                ax.plot(rbins,ca_s(rbins),c='r',linestyle='--')
                #ax.axvspan(min(reffs),max(reffs),color='k',alpha=0.3,label=r'R$_{eff}$')
                for reff in reffs:
                    ax.axvline(reff,c='k',alpha=.05)
                ax.axvline(reffs[0],c='k',alpha=.05,label=r'R$_{eff}$')

                ax.legend(loc='lower right',prop={'size':12})
                fname='Stars' if t=='3D' else 'Dark'
                f.savefig(f'../../Figures/3DShapes/{sim}.BW/{hid}.{fname}.png',bbox_inches='tight',pad_inches=.1)
                plt.close()

                Shapes[str(hid)]['ba_smooth'] = ba_s
                Shapes[str(hid)]['ca_smooth'] = ca_s

        pickle.dump(Shapes,open(f'../../Data/{sim}.BW.{t}Shapes.pickle','wb'))