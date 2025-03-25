
import pickle
import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
# Import the SimInfoDicts package
from SimInfoDicts.sim_type_name import sim_type_name
# Generate the placeholder text for HaloTypes.txt
def main():
    verbose = True

    for feedback, use_sim in sim_type_name.items():
        # do not look at feedback 'BWMDC'
        if feedback != 'BWMDC' and use_sim:
            pickle_path = f'../PickleFiles/SimulationInfo.{feedback}.pickle'
            if os.path.exists(pickle_path):
                SimInfo = pickle.load(open(pickle_path, 'rb'))

            types = []
            #fix this to load pickle file instead!!!!!
            #only applicable to feedback strings with the keyword Merian
            #feedback might look someting like MerianCDM or MerianSIDM
            key = 'Merian'
            if key in feedback:
                for sim in SimInfo:
                    for hid in SimInfo[sim]['goodhalos']:
                        halo_type = 'Central' if hid == 1 else 'Satellite'
                        types.append(f"{sim}\t{hid}\t{halo_type}\n")


            # Join all lines into a single string to represent the file content
            placeholder_content = ''.join(types)

            # Write content to a file
            output_path = f'../../Data/BasicData/HaloTypes.{feedback}.txt'
            # Adjust the path as necessary
            header = 'Volume\tHaloGRP@z0\tHaloType\n'
            with open(output_path, 'w') as file:
                file.write(header)
                file.write(placeholder_content)

            print(f"File written to {output_path}")
if __name__ == '__main__':
    main()