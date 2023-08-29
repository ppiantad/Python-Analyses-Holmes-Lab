import os
from pathlib import Path
from typing import List
import glob
import re
import subprocess
from pathlib import Path

meta_folder_path = Path(r"D:\Behavior Videos\BLA-NAcShell ArchT vs eYFP")




def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )
    return files

def resize_video(video_path, new_w, new_h, out_path):
    #ffmpeg -i input.mp4 -vf scale=$w:$h <encoding-parameters> output.mp4
    command = ['ffmpeg', '-i', video_path, '-vf', f'scale={new_w}:{new_h}', os.path.join(video_path, out_path)]
    #cmd = f"ffmpeg -i {video_path} -vf scale={new_w}:{new_h} -preset slow -crf 18 {out_path}"
    subprocess.run(command)

def grayscale_video(video_path, out_path):
    #cmd = f"ffmpeg -i {video_path} -vf hue=s=0 {out_path}"
    command = ['ffmpeg', '-i', video_path, '-vf', f'hue=s={0}', os.path.join(video_path, out_path)]
    subprocess.run(command)




for root, dirs, files in os.walk(meta_folder_path):
        # Exclude subfolders containing the exclusion string
        dirs[:] = [d for d in dirs if "not in final dataset" not in d]        
        
        # look for .MP4 files in each subfolder

        mp4_files = [f for f in files if f.endswith('.MP4')]

        filtered_vids = []
        for mp4_file in mp4_files:
            #dir_path = os.path.dirname(mp4_file)
            merged_files = [f for f in files if f.endswith("merged.MP4")]
            resized_files = [f for f in files if f.endswith("_resized.MP4")]
            resized_greyscaled_files = [f for f in files if f.endswith("_grayscaled.MP4")]

            if not (merged_files or resized_files or resized_greyscaled_files):
                filtered_vids.append(mp4_file)

                
        #return filtered_vids


        if len(filtered_vids) == 0:
            print ("No MP4 files found to be processed.")
            # create a text file containing the list of video files in order
        else:
            print(f"Found {len(filtered_vids)} video files to be processed.")
            # sort the .MP4 files by creation time
            filtered_vids.sort(key=lambda f: os.path.getctime(os.path.join(root, f)))
            # create mylist.txt
            mylist_file = os.path.join(root, 'mylist.txt')
            with open(mylist_file, 'w') as f:
                for filtered_vid in filtered_vids:
                    f.write("file '{}'\n".format(os.path.join(root, filtered_vid)))
            input_files = []
            for f in filtered_vids:
                if ' ' in f:
                    f = f.replace(' ', '_')
                input_files.append(f)

            if len(filtered_vids) > 1:
                #if there are multiple files, remove the (1) from the first file to create the final "output_file" name
                # some files might end with (1).MP4 - use a regular expression to identify these, then change the output file name accordingly (remove the (1).MP4 and add merged.MP4)
                if re.search(r' \(\d+\)\.MP4$', input_files[0]):
                    output_file = os.path.join(root, f"{os.path.join(root, input_files[0].split('(')[0].replace(' ', ''))}merged.MP4")
                # some files might end with _1.MP4 - use a regular expression to identify these, then change the output file name accordingly (remove the _1.MP4 and add _merged.MP4)
                elif re.search(r'_\d+\.MP4$', input_files[0]):
                    output_file = os.path.join(root, f"{os.path.join(root, input_files[0].rsplit('_', 1)[0].replace(' ', ''))}_merged.MP4")
                # renaming catch all in case there are other types of videos that I haven't accounted for
                else:
                    output_file = os.path.join(root, f"{os.path.join(root, input_files[0].rsplit('_', 1)[0].replace(' ', ''))}_merged.MP4")
            elif len(filtered_vids) == 1:
                #if there is only one file (i.e. hot plate) remove the .MP4 from the end and add append _merged.MP4
                output_file = os.path.join(root, f"{os.path.join(root, input_files[0].split('.')[0].replace('.MP4', ''))}_merged.MP4")

            
            print(output_file)
            subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', mylist_file, '-c', 'copy', output_file])
            os.remove(mylist_file)

            print("Video files merged successfully!")

        video_paths = find_paths_endswith(root, "_merged.MP4")

        for video_path in video_paths:
            resize_out_path = video_path.replace(".MP4", "_resized.MP4")
            resize_video(video_path, 800, 600, resize_out_path)
            grayscale_out_path = resize_out_path.replace(".MP4", "_grayscaled.MP4")
            grayscale_video(resize_out_path, grayscale_out_path)
            #delete the raw merged file because it is very large and likely won't be used
            if video_path.endswith("_merged.MP4"):
                os.remove(os.path.join(root, video_path))