import os
import pickle

#load 'SimulationInfo.RDZ.pickle' file and print the halos in 'goodhalos
sim_dict = pickle.load(open('SimulationInfo.RDZ.pickle', 'rb'))
#sim_dict = pickle.load(open('../Data/r613.RDZ.Masking.pickle', 'rb'))
for sims in sim_dict:
    print(f'{sims} has good halos: {sim_dict[sims]["goodhalos"]}')
#print(sim_dict)





