import pickle
import numpy as np

def load_pickle_data(filepath):
    """Loads a pickle file from the specified filepath."""
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def print_halo_data(halo_data, halo_id):
    """Prints the data for a specified halo."""
    specific_halo_data = halo_data.get(halo_id, {})
    print(f"Data for halo {halo_id}:")
    for key, value in specific_halo_data.items():
        # Convert numpy.float64 to Python native float for printing
        if isinstance(value, np.float64):
            value = float(value)
        print(f"{key}: {value}")

if __name__ == "__main__":
    # Path to the pickle file - adjust this based on the file you wish to load
    # For RDZ file
    feedback = 'Marvel_DCJL'
    #feedback = 'RDZ'
    filepath = f'Data/BasicData/{feedback}.Masses.pickle'
    # For Marvel file
    # filepath = 'Data/BasicData/Marvel_DCJL.Masses.pickle'
    
    # Load the data
    data = load_pickle_data(filepath)
    #print(data)
    # Determine which dataset to use ('cptmarvel' or 'r431')
    if 'cptmarvel' in data:
        halo_data = data['cptmarvel']
        print_halo_data(halo_data, '1')
    elif 'r431' in data:
        halo_data = data['r431']
        print_halo_data(halo_data, 1)
    else:
        print("Expected data not found in the file.")
