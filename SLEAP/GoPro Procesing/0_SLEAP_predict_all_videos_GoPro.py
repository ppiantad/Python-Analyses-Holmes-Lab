import os, glob
from pathlib import Path
import subprocess

def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def main():
    ROOT = r"/media/rory/RDT VIDS/BORIS_merge/BATCH_2"
    avis = find_paths_endswith(ROOT, "merged_resized_grayscaled.mp4")
    model = "/media/rory/Padlock_DT/Opto_Speed_Analysis/models/220708_114639.centroid.n=204"
    model2 = "/media/rory/Padlock_DT/Opto_Speed_Analysis/models/220708_120742.centered_instance.n=204"

    for avi in avis:
        avi_parent = Path(avi).parents[0]
        slp_out_path = avi.replace(".mp4", ".mp4.predictions.slp")
        
        slp_file = find_paths_endswith(avi_parent, ".slp")
        
        if len(slp_file) == 0: 
            #only perform process if there arent any sleap files where merged mp4 was found
            cmd = f"sleap-track '{avi}' -m '{model}' -m '{model2}'"
            print(cmd)
            os.system(cmd)


def new_main():

    directory_path = r"F:\BA-NAc CRISPR CNR1\BA-NA-Con-1"
    model_path_1 = r"D:\SLEAP\Opto_Model\220708_114639.centroid.n=204"
    model_path_2 = r"D:\SLEAP\Opto_Model\220708_120742.centered_instance.n=204"

    for root, dirs, files in os.walk(directory_path):
        mp4_files = [f for f in files if f.endswith('merged_resized_grayscaled.MP4')]
        # Check if any .slp file exists in the directory
        slp_files = glob.glob(os.path.join(root, '*.slp'))
        if slp_files:
            print(f"Skipping {root} directory as .slp file already exists.")
            continue
        for mp4_file in mp4_files:
            video_path = os.path.join(root, mp4_file)
            cmd = ['sleap-track', video_path, '-m', model_path_1, '-m', model_path_2]
            subprocess.run(cmd)

def one_vid():
    
    video = "D:/SLEAP/RRD76/RRD76_RDT_OPTO_CHOICE_0.1_mA_10222019_3-03_merged_resized_grayscaled.mp4"
    model = "D:/SLEAP/220708_114639.centroid.n=204"
    model2 = "D:/SLEAP/220708_120742.centered_instance.n=204"

    cmd = f"sleap-track {video} -m {model} -m {model2}"
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    #main()
    new_main()
    #one_vid()