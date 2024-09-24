import os, glob
from pathlib import Path
import pandas as pd
import numpy as np

from mfreeze.oop_interface import FreezeDetector, LoctionTracker
from mfreeze.utils import crop_set_same

# Change these!

start_frame = 0
# higher values = more sensitive? 
freeze_threshold = 1000



file_extention = ".avi"

report_dir_label = "freeze_vid"

directory_path = r"E:\Context Data\PFC Last\Raw Data\PFC alone\Raw Data"

for root, dirs, files in os.walk(directory_path):
    # Exclude subfolders containing the exclusion string
    dirs[:] = [d for d in dirs if "not in final dataset" not in d]      

    # Ignore folders named "other_data"
    if "other_data" in root:
        print(f"Skipping {root} directory as it is 'other_data'.")
        continue

    mp4_files = [f for f in files if f.endswith('.avi') and f != "freeze_video.avi"]

    if not mp4_files:
        print(f"There are no correctly named files to predict on")
    else:
        for mp4_file in mp4_files:
            try:
                current_video = str(mp4_file)
                print(f"Current Video:\n\t{current_video}")
                print(root)
                video_path = os.path.join(root, current_video)
                report_dir = os.path.join(root, report_dir_label)
                os.makedirs(report_dir, exist_ok=True)
                detector = FreezeDetector(video_path, 
                          save_video_dir=report_dir,
                          freeze_threshold=freeze_threshold, 
                          start_frame=start_frame,
                          med_filter_size=5)
                
                detector.detect_motion()
                detector.detect_freezes()
                detector.save_video()
                dff = detector.generate_report()
                file_name = f"{Path(current_video).stem}_{freeze_threshold}_detector_report.csv"
                path_for_csv = os.path.join(report_dir, file_name)
                print(path_for_csv)
                dff.to_csv(path_for_csv, index=False)
            except IndexError:
                print("No more videos left to analyse!", "\n")
                print("Check the contents of `done_vids` to make "
                    "sure all of the videos to be analysed have been.")










# do not run until fixing how this works?

# i, c  = detector.interactive_crop(frame=start_frame)
# i




#tracker = LoctionTracker(
    #current_video,
    #start_frame=start_frame, 
    #save_video_dir=report_dir,
    #reference_num_frames=1000,
#)
#tracker = crop_set_same(detector, tracker)
#tracker.track_location()
#tracker.save_video()


# Generate reports from detector and tracker




# Save the detector report to CSV




