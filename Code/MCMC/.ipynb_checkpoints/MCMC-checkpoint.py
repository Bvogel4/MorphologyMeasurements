import logging
from math import sqrt,sin,cos,acos,log,factorial
import emcee,pynbody,pickle,sys,warnings,argparse,os,multiprocessing
import matplotlib.pyplot as plt
import corner
import numpy as np
import scipy.optimize as opt
from numpy.random import uniform as unirand
from numpy.random import normal as normrand

import importlib.util
from pathlib import Path

sys.path.append('../')
from scipy.interpolate import griddata
from ShapeFunctions.Functions import RegularGridInterpolator as RGI
from ShapeFunctions.Functions import Ellipsoid,Project,kpc2pix,pix2kpc,Project_OLD



# Setup logging
logging.basicConfig(filename='error_log.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

# Instead of ignoring warnings, log them
import warnings
def warn_with_logging(message, category, filename, lineno, file=None, line=None):
    logger.warning(f'{category.__name__}: {message} at {filename}:{lineno}')
warnings.showwarning = warn_with_logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)



def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)
plt.rcParams['text.usetex']=False

#MCMC Formalism from Kado-Fong et al. 2020 - https://iopscience.iop.org/article/10.3847/1538-4357/abacc2

# q = b/a = func(B/A,C/A,Theta,Phi) [eq (1) - (4)]
def q(B,C,theta,phi):
    f = sqrt( (C*sin(theta)*cos(phi))**2 + (B*C*sin(theta)*sin(phi))**2 + (B*cos(theta))**2 )
    g = cos(phi)**2 + (cos(theta)*sin(phi))**2 + B**2*(sin(phi)**2+(cos(theta)*cos(phi))**2) + (C*sin(theta))**2
    h = sqrt( (g-2*f)/(g+2*f) )
    return( (1-h)/(1+h) )

#Priors for mean,std of norm dist over B/A and C/A [eq (7) - (9)]
def priors(alpha):
    mu_B,mu_C,sig_B,sig_C = alpha
    p_mu_B = 1 if 0<mu_B<1 else 0
    p_mu_C = 1 if (0<mu_C<1 and mu_C<=mu_B) else 0
    p_sig_B = 1 if 0<sig_B<.5 else 0
    p_sig_C = 1 if 0<sig_C<.5 else 0
    return ( p_mu_B * p_mu_C * p_sig_B * p_sig_C )

#Predicted distribution of b/a when sampling Theta,Phi for a fixed B/A,C/A [m_i in eq (6)]
def predicted_dist(alpha,n_itr=1e3):
    mu_B,mu_C,sig_B,sig_C = alpha
    q_dist = np.zeros(int(n_itr))
    for i in np.arange(n_itr):
        B,C = normrand(mu_B,sig_B),normrand(mu_C,sig_C)
        theta = acos(unirand(-1,1))
        phi = unirand(0,2*np.pi)
        q_dist[int(i)] = q(B,C,theta,phi)
    return q_dist

#Functions for observed distribution [n_i in eq (6)]
def randomorientation():
    phi = np.degrees(np.random.uniform(0,np.pi*2))
    theta = np.degrees(np.arccos(np.random.uniform(-1,1)))
    return( (theta,phi) )
def observed_dist(halo_inter,n_itr=1e3):
    q_dist = np.zeros(int(n_itr))
    for i in np.arange(n_itr):
        angle = randomorientation()
        q_dist[int(i)] = float(halo_inter(angle))
    return q_dist

# Ln of prob(q | alpha) [eq (6)]
def ln_prob_q_alpha(alpha,observed_dist,bin_size=0.04):
    n = observed_dist
    m = predicted_dist(alpha,len(n))
    bins,prob = np.linspace(0,1,int(1/bin_size+1)),0
    for i in np.arange(len(bins)-1):
        ni = len( n[(bins[i]<n) & (n<=bins[i+1])] )
        mi = len( m[(bins[i]<m) & (m<=bins[i+1])] )
        if mi>0:
            prob+= ni*log(mi)-mi-log(factorial(ni))
    return(prob)

#Full prob function for emcee [eq(6) * priors]
def ln_prob_fn(alpha,observed_dist,bin_size):
    if priors(alpha)==1:
        return ln_prob_q_alpha(alpha,observed_dist,bin_size)
    else:
        return -np.inf

#Initial position for MCMC walkers
def initial(nwalk=32,B_true=.75,C_true=.5):
    pos,i = np.zeros((nwalk,4)),0
    while i<nwalk:
        #test = (normrand(0.75,.2),normrand(.5,.2),normrand(.1,.05),normrand(.1,.05))
        test = (normrand(B_true,.15),normrand(C_true,.15),normrand(.05,.01),normrand(.05,.01))
        if priors(test)==1:
            pos[i]=test
            i+=1
    return pos
    


parser = argparse.ArgumentParser(description='Collect images of all resolved halos from a given simulation. Images will be generated across all orientations.')
parser.add_argument('-f','--feedback',choices=['BW','SB','RDZ'],default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',required=True,help='Simulation to analyze')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-o','--overwrite',action='store_true',help='Overwrite existing data')
args = parser.parse_args()


#Load in MCMC file, or create new one
try:
    MCMCData = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.MCMC.pickle','rb'))
except:
    MCMCData = {}
if args.overwrite: MCMCData = {}
SimInfo = pickle.load(open(f'../SimulationInfo.{args.feedback}.pickle','rb'))
halos = SimInfo[args.simulation]['goodhalos']
ProjectedData = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.ProjectedData.pickle','rb'))
ProfileData = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Profiles.pickle','rb'))
IsophoteData = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.InterpolationFunctions.pickle','rb'))
TrueData = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.3DShapes.pickle','rb'))


haloprog = 0
#print(f'Running {args.simulation}: 0.00%')
for halo in halos:
    print(f'Running {args.simulation} - {halo}')
    #if not args.overwrite and halo in MCMCData:continue
    MCMCData[str(halo)] = {}
    for type in ['Isophote','Projected']:
        alpha = {}
        for i in [16,50,84]:
            alpha[f'alpha_{i}'] = (np.NaN,np.NaN,np.NaN,np.NaN)
        MCMCData[str(halo)][type] = alpha
    
    #Create observed distributions from isophote ellipses and projected instrinsic ellipsoids
    if isinstance(IsophoteData[str(halo)],float): continue
    types = []
    observed_dist_isophote = observed_dist(IsophoteData[str(halo)])
    if len(observed_dist_isophote[np.isnan(observed_dist_isophote)])/len(observed_dist_isophote)<.5:
        types.append(0)
    observed_dist_projected = []#observed_dist(ProjectedData[str(halo)])
    #if len(observed_dist_projected[np.isnan(observed_dist_projected)])/len(observed_dist_projected)<.5:
    #    types.append(1)

    #Find the "True" B/A,C/A for comparison distribution
    try:
        Reff = ProfileData[str(halo)]['x000y000']['Rhalf']
        ind_eff = np.argmin(abs(TrueData[str(halo)]['rbins']-Reff))
        alpha_true = (TrueData[str(halo)]['ba'][ind_eff],TrueData[str(halo)]['ca'][ind_eff],0,0)
        true_dist = predicted_dist(alpha_true)
    except Exception as e:
        logger.error(f'Error {e} occured in halo {halo}')
        alpha_true = (np.NaN,np.NaN,np.NaN,np.NaN)
        true_dist = np.array([])

    #Run emcee on 10 cores for Isophote and Projected Data
    nwalk,ndim,nstep,burnin = 32,4,3000,100
    dists,titles = [observed_dist_isophote,observed_dist_projected],['Isophote','Projected']
    for type in types:
        try:
            pool = multiprocessing.Pool(args.numproc)
            sampler = emcee.EnsembleSampler(nwalk,ndim,ln_prob_fn,args=(dists[type],.04),pool=pool)
            start_pos = initial(nwalk) if np.isnan(alpha_true[0]) else initial(nwalk,alpha_true[0],alpha_true[1])
            sampler.run_mcmc(start_pos,nstep,progress=True)
            samples = sampler.get_chain()
            flat_samples = sampler.get_chain(discard=burnin,flat=True)


            #Plot the walkers
            f,axes = plt.subplots(4, figsize=(10, 7), sharex=True)
            labels = [r'$\mu_B$', r"$\mu_C$", r"$\sigma_B$",r"$\sigma_C$"]
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                if i in [0,1]: ax.set_ylim([0,1])
                else: ax.set_ylim([0,.5])
            axes[-1].set_xlabel("step number")
            os.makedirs(f'../../Figures/MCMC/{args.simulation}.{args.feedback}', exist_ok=True)
            f.savefig(f'../../Figures/MCMC/{args.simulation}.{args.feedback}/{halo}.{titles[type]}.Walkers.png',bbox_inches='tight',pad_inches=.1)


            #Generate Corner Plots
            f = corner.corner(flat_samples[:,:2],labels=labels[:2],truths=alpha_true[:2],quantiles=[0.16, 0.5, 0.84],
                                show_titles=True,title_kwargs={"fontsize": 17},label_kwargs={"fontsize": 17})
            axes = np.array(f.axes).reshape((2,2))
            for yi in range(2):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.tick_params(which='both',labelsize=12)
            for i in range(2):
                axes[i,i].tick_params(which='both',labelsize=12)
            f.savefig(f'../../Figures/MCMC/{args.simulation}.{args.feedback}/{halo}.{titles[type]}.Corner.Zoom.png',bbox_inches='tight',pad_inches=.1)
            
            f = corner.corner(flat_samples[:,:2],labels=labels[:2],truths=alpha_true[:2],quantiles=[0.16, 0.5, 0.84],
                                show_titles=True,title_kwargs={"fontsize": 17},label_kwargs={"fontsize": 17})
            axes = np.array(f.axes).reshape((2,2))
            for yi in range(2):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.set_xlim([0,1])
                    ax.set_ylim([0,1])
                    ax.set_xticks([.25,.5,.75,1])
                    ax.set_yticks([.25,.5,.75,1])
                    ax.tick_params(which='both',labelsize=12)
            for i in range(2):
                axes[i,i].set_xlim([0,1])
                axes[i,i].set_xticks([.25,.5,.75,1])
                axes[i,i].tick_params(which='both',labelsize=12)
            f.savefig(f'../../Figures/MCMC/{args.simulation}.{args.feedback}/{halo}.{titles[type]}.Corner.png',bbox_inches='tight',pad_inches=.1)


            #Write out alpha data
            alpha = {}
            for i in [16,50,84]:
                alpha[f'alpha_{i}'] = (np.percentile(flat_samples[:, 0],i),
                                    np.percentile(flat_samples[:, 1],i),
                                    np.percentile(flat_samples[:, 2],i),
                                    np.percentile(flat_samples[:, 3],i))
            

            #Plot distribtions
            bins = np.linspace(0,1,26)
            f,ax = plt.subplots(1,1)
            ax.hist(dists[type],bins,histtype='step',edgecolor='k',linewidth=2,density=True,label='Observed')
            ax.hist(predicted_dist(alpha['alpha_50']),bins,histtype='step',edgecolor='r',linewidth=2,density=True,label='Predicted')
            ax.hist(predicted_dist(alpha_true),bins,histtype='step',edgecolor='b',linewidth=2,density=True,label='"True"')
            ax.set_ylabel('Normalized Distribution',fontsize=17)
            ax.set_xlabel(r'b/a',fontsize=17)
            ax.tick_params(which='both',labelsize=12)
            ax.set_xlim([0,1])
            ax.legend(loc='upper left',prop={'size':12})
            f.savefig(f'../../Figures/MCMC/{args.simulation}.{args.feedback}/{halo}.{titles[type]}.Distriburions.png',bbox_inches='tight',pad_inches=.1)

        except Exception as e:
            logger.error(f'error in plotting or saving {e}')
            err = 1

        #Write out data after each halo
        MCMCData[str(halo)][titles[type]] = alpha
        pickle.dump(MCMCData,open(f'../../Data/{args.simulation}.{args.feedback}.MCMC.pickle','wb'))

    haloprog+=1
    #myprint(f'Running {args.simulation}: {round(haloprog/len(halos)*100,2)}%',clear=True)