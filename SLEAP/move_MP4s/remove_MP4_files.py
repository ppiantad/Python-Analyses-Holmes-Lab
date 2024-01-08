import os
import shutil

def move_video_file(folder_path):
    # Check if the folder contains the required files
    h5_file = None
    slp_file = None
    video_file = None

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".h5"):
            h5_file = os.path.join(folder_path, file_name)
            #print(h5_file)
        elif file_name.endswith(".slp"):
            slp_file = os.path.join(folder_path, file_name)
            #print(slp_file)
        elif file_name.endswith("merged_resized_grayscaled.MP4"):
            video_file = os.path.join(folder_path, file_name)
            #print(video_file)

    # Move the video file up one level if all required files are present
    if h5_file and slp_file and video_file:
        target_folder = os.path.dirname(folder_path)
        #new_video_path = os.path.join(target_folder, "merged_resized_grayscaled.MP4")
        shutil.move(video_file, target_folder)
        print(f"Moved {video_file} to {target_folder}")

# Specify the root folder
root_folder = r"G:\Behavior Videos\BLA stGtACR vs EYFP\RRD319"

# Loop through all subfolders
for subdir, dirs, files in os.walk(root_folder):
    for folder in dirs:
        folder_path = os.path.join(subdir, folder)
        move_video_file(folder_path)
