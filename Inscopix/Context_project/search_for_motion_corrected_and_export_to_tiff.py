import isx 
import os, glob
from pathlib import Path


def find_motion_corrected_paths(root_path, endswith: str):
    motion_corrected_files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    motion_corrected_files_filtered = []
    for motion_corrected_file in motion_corrected_files:
        dir_path = os.path.dirname(motion_corrected_file)
        motion_corrected_file_exists = [f for f in os.listdir(dir_path) if f.endswith("motion_corrected.tiff")]

        if not (motion_corrected_file_exists):
            motion_corrected_files_filtered.append(motion_corrected_file)
    return motion_corrected_files_filtered

def main():

    root_path = Path(r"E:\Context Data\PFC Last\Raw Data\PFC alone\Raw Data")
    # find your file paths containing the motion_corrected.isxd ending
    
    print(root_path)
    endswith = "motion_corrected.isxd"
    motion_corrected_files_filtered = find_motion_corrected_paths(root_path, endswith)


    if motion_corrected_files_filtered is None:
        print("No new unprocessed motion_corrected.isxd files found.")
    else:
        print(f"Found {len(motion_corrected_files_filtered)} unprocessed motion_corrected.isxd files.")
        for count, motion_corrected_file in enumerate(motion_corrected_files_filtered):           
            print(f"INPUT: {motion_corrected_file}")          
            replacement_path = motion_corrected_file.replace(".isxd", ".tiff")
            isx.export_movie_to_tiff(motion_corrected_file, replacement_path)
            print(f"{count} files left to be processed.")

    #for file_path in my_list:
        #replacement_path = file_path.replace(".gpio", "_gpio.csv")
        #isx.export_gpio_set_to_csv(file_path, replacement_path, inter_isxd_file_dir='/tmp', time_ref='start')

if __name__ == "__main__":
    main()