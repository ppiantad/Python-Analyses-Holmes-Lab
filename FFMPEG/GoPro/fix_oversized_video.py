import os
import json
import subprocess

# Define the root directory where the search should start
root_dir = r"E:\Inscopix Mice 072025"

# Define the target filename to look for
target_filename = "context_switch_two.avi"

# Define the correct session duration in seconds
correct_duration = 840

def get_video_duration(video_path):
    """Runs ffprobe to extract the video duration."""
    command = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=duration", 
        "-of", "json", 
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        metadata = json.loads(result.stdout)
        return float(metadata["streams"][0]["duration"])
    except (KeyError, IndexError, ValueError):
        print(f"Could not get duration for {video_path}")
        return None

def process_video(video_path):
    """Processes the video by adjusting its speed using FFmpeg."""
    duration = get_video_duration(video_path)
    if duration is None or duration <= 0:
        return
    
    # Calculate the speed correction factor
    correction_factor = correct_duration / duration
    print(f"Processing {video_path} with correction factor {correction_factor:.4f}")

    # Define output file path
    output_path = video_path.replace(".avi", "_fixed.avi")

    # Check if the fixed file already exists
    if os.path.exists(output_path):
        print(f"Skipping {video_path}, fixed version already exists.")
        return

    # FFmpeg command to adjust video speed
    command = [
        "ffmpeg",
        "-i", video_path,
        "-filter:v", f"setpts={correction_factor:.4f}*PTS",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]

    subprocess.run(command, check=True)
    print(f"Saved fixed video to {output_path}")

# Walk through the directory structure and find target files
for root, _, files in os.walk(root_dir):
    # Skip processing if a _fixed.avi file is already in this folder
    if any(file.endswith("_fixed.avi") for file in files):
        print(f"Skipping folder {root}, fixed file already exists.")
        continue
    
    merged_files = [f for f in files if f.endswith("context_switch_two.avi")]
    
    for video_file in merged_files:
        video_path = os.path.join(root, video_file)
        process_video(video_path) 

    
  #  if target_filename in files:
   #     video_path = os.path.join(root, target_filename)
    #    process_video(video_path)
