import os
import pandas as pd
from pathlib import Path
import argparse

# conda activate sleap

def convert_h5_to_csv(root_folder):
    # Loop through all subdirectories and files
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.h5'):
                # Full path to the .h5 file
                hdf5_filepath = os.path.join(dirpath, filename)
                
                # Load the DataFrame from the HDF5 file
                dataframe = pd.read_hdf(hdf5_filepath)
                
                # Define the output CSV file path in the same directory
                csv_filepath = Path(hdf5_filepath).with_suffix('.csv')
                
                # Save the DataFrame to the CSV file
                dataframe.to_csv(csv_filepath, index=False)
                
                print(f'Data has been exported to {csv_filepath}')

def main():
    # Define the root folder to start the search
    root_folder = r"D:/Context Data/PFC Last/Raw Data/PFC alone/Raw Data"  # Replace this with your desired root folder path

    # Call the conversion function with the defined root folder
    convert_h5_to_csv(root_folder)

if __name__ == '__main__':
    main()