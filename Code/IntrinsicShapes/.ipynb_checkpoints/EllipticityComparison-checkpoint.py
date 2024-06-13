import pickle
import numpy as np
import matplotlib.pylab as plt

IndPlot = True
PaperFig = True

mass_data = pickle.load(open('../../Data/BasicData/Marvel_DCJL.Masses.pickle','rb'))
SimInfo = pickle.load(open(f'../SimulationInfo.BW.pickle','rb'))
cens,sats,bsps = [],[],[]
with open('../../Data/BasicData/HaloTypes.txt') as f:
    lines = f.readlines()
for line in lines:
    l = line.split('\t')
    if l[0] in SimInfo:
        if l[-2]=='Central' and int(l[1]) in SimInfo[l[0]]['goodhalos']: cens.append((l[0],l[1]))
        elif l[-2]=='Satellite' and int(l[1]) in SimInfo[l[0]]['goodhalos']: sats.append((l[0],l[1]))
        elif l[-2]=='Backsplash' and int(l[1]) in SimInfo[l[0]]['goodhalos']: bsps.append((l[0],l[1]))



xang = np.arange(0,180,30)
yang = np.arange(0,360,30)
Yedge = np.arange(-15,185,30)
Xedge = np.arange(-15,375,30)
plot_diff = np.zeros((len(xang),len(yang)))
plot_delta = np.zeros((len(xang),len(yang)))

pr_f_c,ob_f_c,pr_s_c,ob_s_c = [],[],[],[]
pr_f_s,ob_f_s,pr_s_s,ob_s_s = [],[],[],[]
for x in np.arange(len(xang)):
    for y in np.arange(len(yang)):

        projected,observed,col = [],[],[]
        deltas,diffs = [],[]
        for sim in SimInfo:
            obs_data = pickle.load(open(f'../../Data/{sim}.BW.ShapeData.pickle','rb'))
            prj_data = pickle.load(open(f'../../Data/{sim}.BW.ProjectedData.pickle','rb'))

            for halo in SimInfo[sim]['goodhalos']:
                col.append(np.log10(mass_data[sim][str(halo)]['Mstar']))
                pr = prj_data[str(halo)][f'x{xang[x]:03d}y{yang[y]:03d}']['b/a']
                projected.append(pr)
                ob = obs_data[str(halo)][f'x{xang[x]:03d}y{yang[y]:03d}']['b/a']
                observed.append(ob)
                diffs.append(pr-ob)
                deltas.append(abs(pr-ob)/np.sqrt(2))
                if int(xang[x])==0 and int(yang[y])==0:
                    if (sim,str(halo)) in sats:
                        pr_f_s.append(pr)
                        ob_f_s.append(ob)
                    else:
                        pr_f_c.append(pr)
                        ob_f_c.append(ob)
                elif int(xang[x])==90 and int(yang[y])==0:
                    if (sim,str(halo)) in sats:
                        pr_s_s.append(pr)
                        ob_s_s.append(ob)
                    else:
                        pr_s_c.append(pr)
                        ob_s_c.append(ob)

        if IndPlot:
            f,ax = plt.subplots(1,1,figsize=(5,5))
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.fill_between([0,1],[-.1,.9],[.1,1.1],color='0.75',alpha=.3)
            ax.plot([0,1],[0,1],c='0.5',linestyle='--')
            ax.set_xlabel('Projected b/a',fontsize=15)
            ax.set_ylabel('Isophote b/a',fontsize=15)
            norm = plt.Normalize(int(min(col)),int(max(col))+1)
            p = ax.scatter(projected,observed,c=col,cmap='viridis',norm=norm)
            cbar = f.colorbar(p,cax=f.add_axes([.91,.11,.03,.77]))
            cbar.set_label(r'Log(M$_*$/M$_\odot$)',fontsize=15)
            f.savefig(f'../../Figures/EllipticityComparison/Individual/EllipticityComparison.x{xang[x]:03d}.y{yang[y]:03d}.png',bbox_inches='tight',pad_inches=.1)
            plt.close()

        plot_diff[x][y] = np.nanmean(diffs)
        plot_delta[x][y] = np.nanmean(deltas)


plots,legend,norms,fname =[[plot_diff,plot_delta],[r'$q_{proj}-q_{iso}$',r'$\Delta_{1-1}$'],
                           [plt.Normalize(0,.2),plt.Normalize(0,.2)],['Differentials','Deltas']]

for i in [0,1]:
    f,ax = plt.subplots(1,1,figsize=(12,5))
    ax.tick_params(labelsize=15)
    ax.set_xticks(np.arange(0,375,30))
    ax.set_yticks(np.arange(0,195,30))
    ax.set_xlim([-15,345])
    ax.set_ylim([-15,165])
    ax.set_xlabel(r'$\phi$-rotation [$^o$]',fontsize=30)
    ax.set_ylabel(r'$\theta$-rotation [$^o$]',fontsize=30)
    ax.tick_params(which='both',labelsize=20)

    norm = plt.Normalize(-1,1)
    p = ax.pcolor(Xedge,Yedge,plots[i],cmap='viridis',edgecolors='k',norm=norms[i])
    c = f.colorbar(p,ax=ax,pad=0.01,aspect=15)
    c.set_label(legend[i],fontsize=30)
    c.ax.tick_params(labelsize=20)
    c.set_ticks([0,.05,.1,.15,.2])

    c.ax.plot([0,1],[np.amin(plots[i]),np.amin(plots[i])],c='w') 
    c.ax.plot([0,1],[np.amax(plots[i]),np.amax(plots[i])],c='w')

    f.savefig(f'../../Figures/EllipticityComparison/{fname[i]}Grid.png',bbox_inches='tight',pad_inches=.1)
    plt.close()



if PaperFig:
    f,ax = plt.subplots(1,1,figsize=(5,5))
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.fill_between([0,1],[-.1,.9],[.1,1.1],color='0.75',alpha=.3,zorder=0)
    ax.plot([0,1],[0,1],c='0.5',linestyle='--',zorder=0)
    ax.set_xlabel(r'$q_{proj}$',fontsize=20)
    ax.set_ylabel(r'$q_{iso}$',fontsize=20)
    ax.tick_params(which='both',labelsize=15)

    ax.scatter(pr_f_c,ob_f_c,c='k',label=r'$(\theta,\phi)=(0^o,0^o)$')
    ax.scatter(pr_f_s,ob_f_s,c='k',marker='v')
    ax.scatter(pr_s_c,ob_s_c,c='r',label=r'$(\theta,\phi)=(90^o,0^o)$')
    ax.scatter(pr_s_s,ob_s_s,c='r',marker='v')
    ax.scatter(-1,-1,c='.5',marker='v',label='Satellites')

    ax.legend(loc=(.01,.63),prop={'size':15})
    f.savefig(f'../../Figures/EllipticityComparison/EllipticityComparison.png',bbox_inches='tight',pad_inches=.1)
    plt.close()