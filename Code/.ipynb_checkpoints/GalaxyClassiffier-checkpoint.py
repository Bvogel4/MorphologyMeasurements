import os
import matplotlib.pyplot as plt
import IPython.display as display
import pickle

# Base path where your images are stored
image_base_path = '../Figures/halo_check_images'

# Assuming all your images are in a single folder and follow the 'simr431h1.png' naming convention
image_files = [f for f in os.listdir(image_base_path) if f.endswith('.png')]

# Prepare an empty dictionary for classifications
halo_types_dict = {}

def classify_halo(image_path):
    """
    Displays the image for classification.
    This function should be adapted based on your interactive environment.
    """
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # For a script, this would be manual classification,
    # but in an interactive environment like Jupyter Notebook,
    # you could use input() to classify each image.
    classification = input("Enter classification (Central, Backsplash, Satellite): ")
    return classification

# Iterate through each image file
for image_file in image_files:
    # Extract simulation and halo ID from the filename
    parts = image_file.split('h')
    simulation_id = parts[0]  # e.g., 'simr431'
    halo_id = parts[1].split('.')[0]  # e.g., '1'

    image_path = os.path.join(image_base_path, image_file)

    # Call the function to display the image and classify
    classification = classify_halo(image_path)

    # Store the classification in the dictionary
    if simulation_id not in halo_types_dict:
        halo_types_dict[simulation_id] = {}
    halo_types_dict[simulation_id][halo_id] = classification

    # Clear the output to prepare for the next image
    display.clear_output(wait=True)

# Once classification is done, convert the dictionary to the format expected by 'HaloTypes.txt'
with open('HaloTypes.txt', 'w') as file:
    file.write("SimulationID\tHaloID\tType\n")  # Writing header
    for sim, halos in halo_types_dict.items():
        for hid, type in halos.items():
            file.write(f"{sim}\t{hid}\t{type}\n")

print("Classification completed and saved to HaloTypes.txt.")