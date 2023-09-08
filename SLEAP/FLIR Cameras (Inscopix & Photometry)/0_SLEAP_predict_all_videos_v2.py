import os, glob
from pathlib import Path
import subprocess

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def main():
    ROOT = "D:\SLEAP\Photometry\Analyzed_2"
    avis = find_paths_endswith(ROOT, ".avi")
    #model = "/media/rory/Padlock_DT/SLEAP/models/220204_110756.centroid.n=2835"
    #model2 = "/media/rory/Padlock_DT/SLEAP/models/220204_114501.centered_instance.n=2835"
    model = "D:/SLEAP/Photometry_and_Inscopix_Model/220201_133640.centroid.n=2688"
    model2 = "D:/SLEAP/Photometry_and_Inscopix_Model/220201_141815.centered_instance.n=2688"

    for avi in avis:
        slp = f"{avi}.predictions.slp"
        slp = Path(slp)
        if slp.is_file():
            pass
        else: # start where you left off
            print(avi)
            cmd = f"sleap-track {avi} -m {model} -m {model2}"
            print(cmd)
            os.system(cmd)


def new_main():

    directory_path = r"H:\MATLAB\TDTbin2mat\Photometry\RRD367"
    model_path_1 = "D:/SLEAP/Photometry_and_Inscopix_Model/220201_133640.centroid.n=2688"
    model_path_2 = "D:/SLEAP/Photometry_and_Inscopix_Model/220201_141815.centered_instance.n=2688"

    for root, dirs, files in os.walk(directory_path):
        dirs[:] = [d for d in dirs if "not in final dataset" not in d]  
        avi_files = [f for f in files if f.endswith('.avi')]
        slp_files = [f for f in files if f.endswith('.slp')]
        for count, avi_file in enumerate(avi_files):
            if any(avi_file[:-4] in slp_file for slp_file in slp_files):
                print(f"Skipping {avi_file} because corresponding .slp file already exists")
                continue
            for count, avi_file in enumerate(avi_files):
                video_path = os.path.join(root, avi_file)
                #subprocess.call(['sleap-track', video_path, '-m', model_path_1, 'm', model_path_2])
                cmd = ['sleap-track', video_path, '-m', model_path_1, '-m', model_path_2]
                subprocess.run(cmd)






def one_vid():
    
    video = "D:/SLEAP/Photometry/RRD256/Pre-RDT_RM/RRD256_PRERDT_RM_2022-09-09T11_25_32.avi"
    model = "D:/SLEAP/Photometry_and_Inscopix_Model/220201_133640.centroid.n=2688"
    model2 = "D:/SLEAP/Photometry_and_Inscopix_Model/220201_141815.centered_instance.n=2688"

    cmd = f"sleap-track {video} -m {model} -m {model2}"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    #main()
    new_main()
    #one_vid()