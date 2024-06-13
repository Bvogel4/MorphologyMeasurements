import pickle
import numpy as np
import matplotlib.pylab as plt
import argparse



parser = argparse.ArgumentParser(description='Compare Stellar and Dark Matter Distributions')
parser.add_argument('-f','--feedback',choices=['BW','SB','RDZ'],default='RDZ',help='Feedback Model')
args = parser.parse_args()


SimInfo = pickle.load(open('../SimulationInfo.RDZ.pickle','rb'))

mass_data = pickle.load(open('../../Data/BasicData/RDZ.Masses.pickle','rb'))
print(mass_data)
#print(mass_data)

# with open('../../Data/BasicData/HaloTypes.txt') as f:
#     halotype = f.readlines()
#     del halotype[0]
# types = {}
# for line in halotype:
#     l = line.split('\t')
#     if l[0] not in types: types[l[0]] = {}
#     types[l[0]][l[1]] = l[-2]



def T(ba,ca):
    return( (1-ba**2)/(1-ca**2) )
T_s,T_d = [],[]
B_s,C_s,B_d,C_d = [],[],[],[]
masses,htype,mb = [],[],[]

for sim in SimInfo:
    StShapes = pickle.load(open(f'../../Data/{sim}.{args.feedback}.3DShapes.pickle','rb'))
    DMShapes = pickle.load(open(f'../../Data/{sim}.{args.feedback}.DMShapes.pickle','rb'))
    Profiles = pickle.load(open(f'../../Data/{sim}.{args.feedback}.Profiles.pickle','rb'))

    for hid in SimInfo[sim]['goodhalos']:
        try:
            rbins,rd,ba_s,ca_s,ba_d,ca_d = [StShapes[str(hid)]['rbins'],DMShapes[str(hid)]['rbins'],
                                            StShapes[str(hid)]['ba_smooth'],StShapes[str(hid)]['ca_smooth'],
                                            DMShapes[str(hid)]['ba_smooth'],DMShapes[str(hid)]['ca_smooth']]
            Reff = Profiles[str(hid)]['x000y000']['Reff']
            #indeff = np.argmin(np.abs(rd-Reff))
            B_s.append(ba_s(Reff))
            C_s.append(ca_s(Reff))
            T_s.append(T(ba_s(Reff),ca_s(Reff)))
            B_d.append(ba_d(Reff))
            C_d.append(ca_d(Reff))
            T_d.append(T(ba_d(Reff),ca_d(Reff)))
            masses.append(np.log10(mass_data[sim][str(hid)]['Mstar']))
            print('test',mass_data[sim][str(hid)]['Mstar'])
            mb.append(np.log10(mass_data[sim][str(hid)]['Mb/Mtot']))
        except Exception as e:
            print(f'error {e}')
        # if sim not in types:
        #     htype.append('o')
        # elif str(hid) not in types[sim]:
        #     htype.append('o')
        # else:
        #     if types[sim][str(hid)] in ['Central','Backsplash']: htype.append('o')
        #     else: htype.append('v')
        # Other code processing each halo...
        if str(hid) == '1':
            htype.append('o')  # Central or Backsplash
        else:
            htype.append('v')  # Satellite


#Convert lists to arrays for htype indexing
T_d,T_s,masses,mb,htype = np.array(T_d),np.array(T_s),np.array(masses),np.array(mb),np.array(htype)
B_s,C_s,B_d,C_d = np.array(B_s),np.array(C_s),np.array(B_d),np.array(C_d)

#T* vs Tdm
f,ax = plt.subplots(1,1,figsize=(5,5))
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.fill_between([0,1],[-1/3,2/3],[1/3,4/3],color='0.75',alpha=.3)
ax.plot([0,1],[0,1],c='0.5',linestyle='--')
ax.set_ylabel(r'T$_*$',fontsize=15)
ax.set_xlabel(r'T$_{DM}$',fontsize=15)

norm = plt.Normalize(int(min(masses)),int(max(masses))+.1)
p = ax.scatter(T_d[htype=='o'],T_s[htype=='o'],marker='o',c=masses[htype=='o'],cmap='viridis',norm=norm)
ax.scatter(T_d[htype=='v'],T_s[htype=='v'],marker='v',c=masses[htype=='v'],cmap='viridis',norm=norm,label='Satellites')
cbar = f.colorbar(p,cax=f.add_axes([.91,.11,.03,.77]))
cbar.set_label(r'Log(M$_*$/M$_\odot$)',fontsize=15)

ax.legend(loc='lower left',prop={'size':12})
f.savefig(f'../../Figures/3DShapes/T_Comparison.png',bbox_inches='tight',pad_inches=.1)


#T vs Mstar
f,ax=plt.subplots(2,1,figsize=(12,6))
plt.subplots_adjust(hspace=0)

for i in [0,1]:
    ax[i].set_xlim([5.8,9.5])
    ax[i].set_ylim([0,1])
    ax[i].plot([4,9.5],[1/3,1/3],c='.75',linestyle='--',zorder=0)
    ax[i].plot([4,9.5],[2/3,2/3],c='.75',linestyle='--',zorder=0)
    ax[i].tick_params(which='both',labelsize=15)
    ax[i].text(5.83,1/6,'Oblate',fontsize=17,rotation='vertical',verticalalignment='center',c='.5')
    ax[i].text(5.83,3/6,'Triaxial',fontsize=17,rotation='vertical',verticalalignment='center',c='.5')
    ax[i].text(5.83,5/6,'Prolate',fontsize=17,rotation='vertical',verticalalignment='center',c='.5')
ax[0].set_ylabel('T',fontsize=25)
ax[1].set_ylabel(r'T$_*$',fontsize=25)
ax[0].set_yticks([0,.5,1])
ax[1].set_yticks([0,.5])
ax[1].set_xlabel(r'Log(M$_*$/M$_\odot$)',fontsize=25)
#ax[0].xaxis.set_tick_params(labelbottom='False')
ax[0].set_xticks([])

for i in np.arange(len(masses)):
    ax[0].axvline(masses[i],ymin=min([T_d[i],T_s[i]]),ymax=max([T_d[i],T_s[i]]),c='.5',zorder=0)
ax[0].scatter(masses[htype=='o'],T_d[htype=='o'],c='k',label='Dark Matter',marker='o')
ax[0].scatter(masses[htype=='o'],T_s[htype=='o'],c='r',label='Stellar',marker='o')
ax[0].scatter(masses[htype=='v'],T_d[htype=='v'],c='k',marker='v')
ax[0].scatter(masses[htype=='v'],T_s[htype=='v'],c='r',marker='v')
ax[0].scatter(0,0,c='.5',marker='v',label='Satellites')
ax[0].legend(prop={'size':15},ncol=3,loc='center left', bbox_to_anchor=(0.02,0.1))

norm = plt.Normalize(int(min(mb))-1,-.75)
p = ax[1].scatter(masses[htype=='o'],T_s[htype=='o'],c=mb[htype=='o'],cmap='viridis',norm=norm,marker='o')
ax[1].scatter(masses[htype=='v'],T_s[htype=='v'],c=mb[htype=='v'],cmap='viridis',norm=norm,marker='v')
cbar = f.colorbar(p,cax=f.add_axes([.91,.11,.03,.77]))
cbar.set_label(r'Log(M$_{bary}$/M$_{vir}(<$R$_{eff}$))',fontsize=25)
cbar.set_ticks([-3,-2.5,-2,-1.5,-1])
cbar.ax.tick_params(labelsize=15)

f.savefig(f'../../Figures/3DShapes/TvsMstar.png',bbox_inches='tight',pad_inches=.1)



#B/A vs C/A links
f,ax = plt.subplots(1,1,figsize=(5,5))
ax.set_xlim([0,1])
ax.set_ylim([0,1])
#ax.fill_between([0,1],[-.1,.9],[.1,1.1],color='0.75',alpha=.3)
ax.plot([0,1],[0,1],c='0.5',linestyle='--')
ax.set_xlabel(r'$S$',fontsize=20)
ax.set_ylabel(r'$Q$',fontsize=20)
ax.tick_params(which='both',labelsize=15)
ax.scatter(-1,-1)

for i in np.arange(len(B_s)):
    ax.plot([B_s[i],B_d[i]],[C_s[i],C_d[i]],c='.5',zorder=0)

f.savefig(f'../../Figures/3DShapes/CvB.LinksOnly.png',bbox_inches='tight',pad_inches=.1)

ax.scatter(B_d[htype=='o'],C_d[htype=='o'],c='k',label='Dark Matter')
ax.scatter(B_d[htype=='v'],C_d[htype=='v'],c='k',marker='v')
ax.scatter(B_s[htype=='o'],C_s[htype=='o'],c='r',label='Stellar')
ax.scatter(B_s[htype=='v'],C_s[htype=='v'],c='r',marker='v')
ax.scatter(-1,-1,c='.5',marker='v',label='Satellites')

ax.legend(loc='upper left',prop={'size':15})
f.savefig(f'../../Figures/3DShapes/CvB.Links.png',bbox_inches='tight',pad_inches=.1)

