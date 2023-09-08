import os, glob
from pathlib import Path
import subprocess



directory_path = r"D:\SLEAP\RRD76\RDT OPTO CHOICE\test"
model_path_1 = r"D:\SLEAP\Photometry_and_Inscopix_Model\220201_133640.centroid.n=2688"
model_path_2 = r"D:\SLEAP\Photometry_and_Inscopix_Model\220201_141815.centered_instance.n=2688"

for root, dirs, files in os.walk(directory_path):
    # Exclude subfolders containing the exclusion string
    dirs[:] = [d for d in dirs if "not in final dataset" not in d]      
    mp4_files = [f for f in files if f.endswith('merged_resized_grayscaled.MP4')]
    # Check if any .slp file exists in the directory
    slp_files = glob.glob(os.path.join(root, '*.slp'))
    if slp_files:
        print(f"Skipping {root} directory as .slp file already exists.")
        continue
    if not mp4_files:
        print(f"There are no correctly named files to predict on")
    else:
        for mp4_file in mp4_files:
            video_path = os.path.join(root, mp4_file)
            cmd = ['sleap-track', video_path, '-m', model_path_1, '-m', model_path_2]
            subprocess.run(cmd)

