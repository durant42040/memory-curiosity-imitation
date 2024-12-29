import numpy as np
import os

def inspect_npy_file(file_path):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Load the .npy file
        array = np.load(file_path, allow_pickle=True)
        
        # Print information about the array
        print(f"Array type: {array.dtype}")
        print(f"Array shape: {array.shape}")
        print(f"Total elements: {array.size}")
        print("\nFirst few elements of the array:")
        print(len(array.flatten()[0][0])) 
        print(len(array.flatten()))
        
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
file_path = '../traj_data/real_traj.npy'
inspect_npy_file(file_path)
