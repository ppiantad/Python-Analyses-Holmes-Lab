
from pathlib import Path
import pandas as pd
import numpy as np

from mfreeze.oop_interface import FreezeDetector, LoctionTracker
from mfreeze.utils import crop_set_same



file_extention = ".avi"
input_dir = Path(r"E:\Context Data\PFC Last\Raw Data\PFC alone\Raw Data\B51618\B51618_Day3_Noon\Behavior\raw_bonsai_data")
report_dir = Path(r"E:\Context Data\PFC Last\Raw Data\PFC alone\Raw Data\B51618\B51618_Day3_Noon\Behavior\raw_bonsai_data\freeze_vid")


report_dir.mkdir(exist_ok=True)
done_vids = []
videos = list(input_dir.glob(f"*{file_extention}"))
video_index = 0



try:
    current_video = str(videos[video_index])
    print(f"Current Video:\n\t{current_video}")
except IndexError:
    print("No more videos left to analyse!", "\n")
    print("Check the contents of `done_vids` to make "
          "sure all of the videos to be analysed have been.")


# Change these!

start_frame = 0
# higher values = more sensitive? 
freeze_threshold = 1000



detector = FreezeDetector(current_video, 
                          save_video_dir=report_dir,
                          freeze_threshold=freeze_threshold, 
                          start_frame=start_frame,
                          med_filter_size=5)



# do not run until fixing how this works?

# i, c  = detector.interactive_crop(frame=start_frame)
# i


detector.detect_motion()
detector.detect_freezes()
detector.save_video()

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
dff = detector.generate_report()



# Save the detector report to CSV
dff.to_csv(report_dir / f"{Path(current_video).stem}_detector_report.csv", index=False)



