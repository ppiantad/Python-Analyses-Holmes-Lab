import os, glob
from pathlib import Path
import subprocess


def new_main():

    directory_path = r"D:\MATLAB\TDTbin2mat\SLEAP Data_processed\D1-eOP-6\Pre-RDT RM"
    model_path_1 = r"C:\Python_Analyses\Python-Analyses-Holmes-Lab\SLEAP\SLEAP_models_for_github\Photometry_and_Inscopix_Model\220201_133640.centroid.n=2688"
    model_path_2 = r"C:\Python_Analyses\Python-Analyses-Holmes-Lab\SLEAP\SLEAP_models_for_github\Photometry_and_Inscopix_Model\220201_141815.centered_instance.n=2688"

    for root, dirs, files in os.walk(directory_path):
        dirs[:] = [d for d in dirs if "not in final dataset" not in d]  
        avi_files = [f for f in files if f.endswith('.avi')]
        slp_files = [f for f in files if f.endswith('.slp')]
        for count, avi_file in enumerate(avi_files):
            if any(avi_file[:-4] in slp_file for slp_file in slp_files):
                print(f"Skipping {avi_file} because corresponding .slp file already exists")
                continue
            if not avi_files:
                print(f"There are no correctly named files to predict on")
            else:
                for count, avi_file in enumerate(avi_files):
                    video_path = os.path.join(root, avi_file)
                    #subprocess.call(['sleap-track', video_path, '-m', model_path_1, 'm', model_path_2])
                    cmd = ['sleap-track', video_path, '-m', model_path_1, '-m', model_path_2, '-n', "1"]
                    subprocess.run(cmd)



if __name__ == "__main__":
    #main()
    new_main()
    #one_vid()