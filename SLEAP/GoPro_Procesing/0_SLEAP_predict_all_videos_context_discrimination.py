import os, glob
from pathlib import Path
import subprocess



directory_path = r"E:\Context Data\PFC Last\Raw Data\PFC alone\Raw Data\B51618\B51618_Day4_Test"
model_path_1 = r"E:\SLEAP\models\240919_103321.single_instance.n=100"
#model_path_2 = r"C:\Python_Analyses\Python-Analyses-Holmes-Lab\SLEAP\SLEAP_models_for_github\Opto_Model_v2\231016_153043.centered_instance.n=1185"

for root, dirs, files in os.walk(directory_path):
    # Exclude subfolders containing the exclusion string
    dirs[:] = [d for d in dirs if "not in final dataset" not in d]      

    # Ignore folders named "other_data"
    if "other_data" in root:
        print(f"Skipping {root} directory as it is 'other_data'.")
        continue

    mp4_files = [f for f in files if f.endswith('.avi') and f != "freeze_video.avi"]
    # Check if any .slp file exists in the directory
    slp_files = glob.glob(os.path.join(root, '**', '*.slp'), recursive=True) #added the '**' and recursive=True to get the code to check multiple subfolders for the .slp files, since once they are extracted they get moved out of the main directory
    if slp_files:
        print(f"Skipping {root} directory as .slp file already exists.")
        continue
    if not mp4_files:
        print(f"There are no correctly named files to predict on")
    else:
        for mp4_file in mp4_files:
            video_path = os.path.join(root, mp4_file)
            cmd = ['sleap-track', video_path, '-m', model_path_1, '-n', "1"]
            subprocess.run(cmd)

