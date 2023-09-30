import os, glob
from pathlib import Path
import subprocess



directory_path = r"D:\Behavior Videos\BLA-NAcShell ArchT vs eYFP\RRD116\SHOCK TEST"
model_path_1 = r"C:\Python_Analyses\Python-Analyses-Holmes-Lab\SLEAP\SLEAP_models\Opto_Model\220708_114639.centroid.n=204"
model_path_2 = r"C:\Python_Analyses\Python-Analyses-Holmes-Lab\SLEAP\SLEAP_models\Opto_Model\220708_120742.centered_instance.n=204"

instance_count = 1

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
            'sleap-track', '--tracking.clean_instance_count', instance_count , video_path, '-m', model_path_1, '-m', model_path_2, '--tracking.tracker simple'
            

