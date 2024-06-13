
import pickle

# Generate the placeholder text for HaloTypes.txt
placeholder_lines = []
#fix this to load pickle file instead!!!!!
SimInfo = pickle.load(open('SimulationInfo.RDZ.pickle','rb'))
for sim in SimInfo:
    for hid in SimInfo[sim]['goodhalos']:
        halo_type = 'Central' if hid == 1 else 'Satellite'
        placeholder_lines.append(f"{sim}\t{hid}\t{halo_type}\n")

# Join all lines into a single string to represent the file content
placeholder_content = ''.join(placeholder_lines)

# Write the placeholder content to a file
output_path = '../Data/BasicData/HaloTypes.RDZ.txt'  # Adjust the path as necessary
header = 'Volume\tHaloGRP@z0\tHaloType\n'
with open(output_path, 'w') as file:
    file.write(header)
    file.write(placeholder_content)

print(f"File written to {output_path}")
