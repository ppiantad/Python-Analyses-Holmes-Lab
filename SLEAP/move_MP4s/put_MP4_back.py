import os
import shutil

def move_video_file_back(folder_path):
    # Check if the folder contains the required files
    h5_file = None
    slp_file = None

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".h5"):
            h5_file = file_path
        elif file_name.endswith(".slp"):
            slp_file = file_path

    # Move the video file back into the corresponding folder
    if h5_file and slp_file:
        parent_folder = os.path.dirname(folder_path)
        
        # Find the video file based on the suffix
        video_suffix = "merged_resized_grayscaled.MP4"
        video_file = next((f for f in os.listdir(parent_folder) if f.endswith(video_suffix)), None)
        
        if video_file:
            video_file_path = os.path.join(parent_folder, video_file)
            new_video_path = os.path.join(folder_path, video_file)
            shutil.move(video_file_path, new_video_path)
            print(f"Moved {video_file_path} back to {new_video_path}")

# Specify the root folder
root_folder = r"G:\Behavior Videos\BLA stGtACR vs EYFP"

# Loop through all subfolders
for subdir, dirs, files in os.walk(root_folder):
    for folder in dirs:
        folder_path = os.path.join(subdir, folder)
        move_video_file_back(folder_path)
