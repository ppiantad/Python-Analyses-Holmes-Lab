from pathlib import Path
import os
from typing import List
#from preprocess import preprocess
from preprocess_no_cnmfe import preprocess
#from preprocess_no_cnmfe_dual_dynamic import preprocess
import glob
import shutil
import isx 


def get_videos(root_path, endswith)-> list:
    
    
    vids = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )
    
    filtered_vids = []
    #for dirpath, dirnames, filenames in os.walk(root_path):
        #for filename in filenames:
            #if filename.endswith(endswith):
                #vids.append(os.path.join(dirpath, filename))
    


    for vid in vids:
        dir_path = os.path.dirname(vid)
        di_files = [f for f in os.listdir(dir_path) if f.endswith("deinterleaved.isxd")]
        mc_files = [f for f in os.listdir(dir_path) if f.endswith("motion_corrected.isxd")]
        ds_files = [f for f in os.listdir(dir_path) if f.endswith("downsampled.isxd")]
        sf_files = [f for f in os.listdir(dir_path) if f.endswith("spatial_filtered.isxd")]
        tiff_files = [f for f in os.listdir(dir_path) if f.endswith("motion_corrected.tiff")]
        gpio_intermediate_files = [f for f in os.listdir(dir_path) if f.endswith("gpio.isxd")]
        if not (di_files or mc_files or ds_files or sf_files or tiff_files or gpio_intermediate_files):
            filtered_vids.append(vid)

            
    return filtered_vids
    
            # process the video here


def main() -> None:
    root_path = Path(r"D:\Inscopix\BLA-Insc-8\RM D1")
    # ^Change this path to where your videos are stored

    print(root_path)
    endswith = ".isxd"
    filtered_vids = get_videos(root_path, endswith)
    print(filtered_vids)
    # process only videos that meet the condition
    if filtered_vids is None:
        print("No video files found.")
    else:
        print(f"Found {len(filtered_vids)} video files.")
        for count, i in enumerate(filtered_vids):           
            print(f"INPUT: {i}")          
            preprocess(in_path=Path(i), out_dir=None) # output is input dir
            print(f"{count} files left to be processed.")




if __name__ == "__main__":
    main()
