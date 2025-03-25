import os
import pickle

#import dictonaries from folder SimInfoDicts, and place them into a list to be iterated over in the main function
#this is done to avoid having to manually add each dictionary to the main function
#they are stored as .py files
#from each file import the dictionary named Sims

#run this to initialize the directories and save the pickle files, as well as image directories


sim_dicts = {}
for file in os.listdir('SimInfoDicts'):
    if file.endswith('.py'):
        # ignore file called sim_type_name.py
        if file == 'sim_type_name.py':
            continue
        # make name of dictionary entry the name of the file without the .py extension
        sim_name = file[:-3]
        # import the dictionary from the file
        exec(f'from SimInfoDicts.{sim_name} import Sims')
        # add the dictionary to the sim_dicts dictionary
        #print(sim_name)
        sim_dicts[sim_name] = Sims

import traceback
# Function to initialize directories
def init_directories(sim_dict, sim_type):
    for sim in sim_dict:
        try:
            #create folders for figures, if they exist, ignore
            os.makedirs(f'../Figures/Images/{sim}.{sim_type}', exist_ok=True)
            for hid in sim_dict[sim]['goodhalos']:
                os.makedirs(f'../Figures/Images/{sim}.{sim_type}/{hid}',exist_ok=True)
        except:
            print(f'Error creating directories for {sim}.{sim_type}')
            print(traceback.format_exc())


#write function to save pickle files
def save_pickle(sim_dict, sim_type_name, init=False):

    for name,use_sim in sim_type_name.items():
        if use_sim:
            if init:
                init_directories(sim_dict[name], name)
            with open(f'PickleFiles/SimulationInfo.{name}.pickle', 'wb') as f:
                pickle.dump(sim_dict[name], f)
            print(f'File saved: SimulationInfo.{name}.pickle')

def main():

    # create a dictionary that holds the names of the simulation types and whether they are to be used
    # from sim_dicts, get the names of the simulation types
    sim_type_name = {}
    #check if sim_type_name.py exists
    #if not create it, if it does, read it
    if 'sim_type_name.py' in os.listdir('SimInfoDicts'):
        #import the dictionary from the file path
        from SimInfoDicts.sim_type_name import sim_type_name
        print(f'Config file found at /SimInfoDicts/sim_type_name.py')
        print(sim_type_name)
    else:
        for sim in sim_dicts:
            sim_type_name[sim] = False
            print('Config file not found, creating new one, please edit \n /SimInfoDicts/sim_type_name.py')
        #save to file
        #add line breaks after each entry

        with open('SimInfoDicts/sim_type_name.py', 'w') as f:
            f.write('sim_type_name = {\n')
            for key in sim_type_name:
                f.write(f'    "{key}": False,\n')
            f.write('}')




    # Initialize image directory
    while True:
        init = input('Initialize Image Directory? (y,n): ')
        if init == 'y' or init == 'n':
            break
        else:
            print('Invalid input')

    if init == 'y':
        init = True
    else:
        init = False
    save_pickle(sim_dicts, sim_type_name, init=init)



if __name__ == '__main__':
    main()
