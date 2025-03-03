"""
Goals are the following:
    1) To extract h5 files from slp files
    2) Automate this conversion for 219 .slp files
    3) Extract this h5 file to its respective folder.

"""

import os
import glob
import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pickle
import pandas as pd
import shutil
import cv2
import sys

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from decimal import Decimal, getcontext


def find_paths_endswith(root_path, endswith) -> list:

    files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    return files


def slp_file_parsing(slp_filename: str):
    #slp_filename = slp_filename.split("/")[-1]
    mouse = slp_filename.split("_")[0]
    #print(mouse)
    session = "_".join(slp_filename.split("_")[1:]).replace(
        ".avi.predictions.slp", "")
    #print(session)

    try:
        session_mod_1 = session.split("20")[0]

        if "." in session_mod_1:
            session_mod_1 = session_mod_1.replace(".", "")
        if "_" in session_mod_1:
            session_mod_1 = session_mod_1.replace("_", " ")
            session_mod_1 = session_mod_1.strip() #added by PTP to get rid of leading and closing whitespace
            #session_mod_1 = session_mod_1.replace("_", "") #old version
        #print(f"{mouse}: {session_mod_1}")

        return mouse, session_mod_1
        
    except Exception as e:
        print(f"Session {mouse}: {session} can't be renamed!")
        print(e)
        pass



def send_all_other_files_somewhere(other_slp_files: list):
    for i in other_slp_files:
        slp_file_parsing(i)


def slp_to_h5_old(in_path, out_path):
    if " " in in_path:
            in_path = in_path.replace(" ", "/ ")
    if " " in out_path:
            out_path = out_path.replace(" ", "/ ")
    cmd = f"sleap-convert --format analysis -o {out_path} {in_path}"
    os.system(cmd)


def slp_to_h5(in_path, out_path):
    #cmd = f"sleap-convert --format analysis -o {out_path} {in_path}"
    #os.system(cmd)
    cmd = ['sleap-convert', '--format', 'analysis', '-o', out_path, in_path]
    subprocess.run(cmd)



def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(
            mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def meta_data(h5_filename):

    with h5py.File(h5_filename, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    print("===filename===")
    print(h5_filename)

    print("===HDF5 datasets===")
    print(dset_names)

    print("===locations data shape===")
    print(locations.shape)

    print("===nodes===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")


def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    return node_vel


def pix_to_cm(i):
    # There are 96 pix in 2.54 cm
    return i * (2.54/96)


def pix_per_frames_to_cm_per_s(i, fps):
    return i * (2.54/96) * (fps/1)


def export_to_csv(out_path, **kwargs):

    df = pd.DataFrame.from_dict(kwargs)

    df.to_csv(out_path, index=False)


def track_one_node(node_name, node_loc, track_map_out):

    plt.figure(figsize=(7, 7))
    plt.plot(node_loc[:, 0, 0], node_loc[:, 1, 0], 'y', label='mouse-0')
    plt.legend()

    plt.xlim(0, 1024)
    plt.xticks([])

    plt.ylim(0, 1024)
    plt.yticks([])
    plt.title(f'{node_name} tracks')
    plt.savefig(track_map_out)
    plt.close()


""""Will no longer call export to csv, another func will do that."""


def visualize_velocity_one_node(node_name, x_axis_time, x_coord_cm, y_coord_cm, vel_mouse, vel_mouse_to_cm_s, coord_vel_graphs_out):

    fig = plt.figure(figsize=(15, 7))
    fig.tight_layout(pad=10.0)

    ax1 = fig.add_subplot(211)
    # the format is (x,y,**kwargs)
    ax1.plot(x_axis_time, x_coord_cm, 'k', label='x')
    ax1.plot(x_axis_time, y_coord_cm, 'r', label='y')
    ax1.legend()

    ax1.set_xticks([i for i in range(0, len(vel_mouse), 600)])

    ax1.set_title(f'{node_name} X-Y Dynamics')
    ax1.set_ylabel("Coordinate (cm)")

    ax2 = fig.add_subplot(212, sharex=ax1)
    """For getting a heatmap version of the velocity."""
    #ax2.imshow(body_vel_mouse[:,np.newaxis].T, aspect='auto', vmin=0, vmax=10)

    ax2.plot(x_axis_time, vel_mouse_to_cm_s, label='Forward Speed')
    ax2.set_yticks([i for i in range(0, 28, 4)])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (cm/s)")
    # ax2.legend()

    plt.savefig(coord_vel_graphs_out)
    plt.close()


"""Will call on export to csv for every node"""


def export_sleap_data_mult_nodes(h5_filepath, session_root_path,mouse,session, video_length_seconds,fps):
    with h5py.File(h5_filepath, "r") as f:
        dset_names = list(f.keys())
        locations = fill_missing(f["tracks"][:].T)
        node_names = [n.decode() for n in f["node_names"][:]]

    for i, name in enumerate(node_names):
        print(f"Working on... {name}")
        # make a dir of this curr node
        #node_folder = os.path.join(session_root_path, f"{mouse}_{session}_{name}")
        node_folder = os.path.join(session_root_path, f"{name}").replace("\\","/")
        #print(node_folder)
        os.makedirs(node_folder, exist_ok=True)
        os.chdir(node_folder)
        #print(os.getcwd())
        

        INDEX = i
        node_loc = locations[:, INDEX, :, :]

        vel_mouse = smooth_diff(node_loc[:, :, 0]).tolist()
        print(len(vel_mouse))
        vel_mouse_to_cm_s = [pix_per_frames_to_cm_per_s(i, fps) for i in vel_mouse]
        print(len(vel_mouse_to_cm_s))
        fig = plt.figure(figsize=(15, 7))
        fig.tight_layout(pad=10.0)

        x_coord_pix = node_loc[:, 0, 0]
        print(len(x_coord_pix))
        y_coord_pix = node_loc[:, 1, 0]
        print(len(y_coord_pix))
        x_coord_cm = [(pix_to_cm(i)) for i in node_loc[:, 0, 0].tolist()]
        print(len(x_coord_cm))
        y_coord_cm = [(pix_to_cm(i)) for i in node_loc[:, 1, 0].tolist()]
        print(len(y_coord_cm))
        range_min = 0
        range_max = Decimal(len(vel_mouse)) / Decimal(str(fps))
        print("Range_max: ", range_max)
        frames = [i for i in range(1, len(vel_mouse)+1)]
        print("Frames: ", max(frames))
        # step is 30 Hz, so 0.033 s in 1 frame
        # step = float(1/fps)
        # works - adjusted x_axis_time so it starts at step, not at 0. also adjusted frames because the range starts at 1, so it needs to go to the length
        # of vel_mouse + 1
        # seems to work as of 2024-03-13 by changing to max(frames) instead of what it was before
        step = Decimal('1') / Decimal(str(fps))
        x_axis_time = [
            Decimal(str(step)) * i for i in range(1, int(max(frames)) + 1)
        ]
        print(len(x_axis_time))


        ######## EXPORTING CSV, COORD, VEL, TRACKS GRAPHS FOR EACH NODE ########
        csv_out = f"{mouse}_{session}_{name}_sleap_data.csv"
        #print(csv_out)
        #csv_out = os.path.join(node_folder,csv_out).replace("\\","/")
        #csv_out = "/".join([node_folder,csv_out])
        #print(csv_out)
        export_to_csv(csv_out,
                    idx_time=x_axis_time,
                    idx_frame=frames,
                    x_pix=x_coord_pix,
                    y_pix=y_coord_pix,
                    x_cm=x_coord_cm,
                    y_cm=y_coord_cm,
                    vel_f_p=vel_mouse,
                    vel_cm_s=vel_mouse_to_cm_s)

        coord_vel_graphs_out = f"{mouse}_{session}_{name}_coord_vel.png"
        #print(coord_vel_graphs_out)
        visualize_velocity_one_node(name,
                                    x_axis_time,
                                    x_coord_cm,
                                    y_coord_cm,
                                    vel_mouse,
                                    vel_mouse_to_cm_s,
                                    coord_vel_graphs_out)


        track_map_out = f"{mouse}_{session}_{name}_tracks.png"
        track_one_node(name, node_loc, track_map_out)
        



#This workflow has been optimized for a Windows file structure. It assumes you have your .avi files saved in a standardized way (see documentation) 
#It also assumes your data are organized with a folder for each mouse, and then a folder for each session, with one .avi and one .slp file in each folder.
def new_main():
    getcontext().prec = 28
    ROOT = r"F:\MATLAB\TDTbin2mat\Photometry\RRD341\RRD341-230417-134252"

    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if "not in final dataset" not in d]  
        slp_files = [f for f in files if f.endswith('.slp')]
        print(slp_files)
        h5_files = [a for a in files if a.endswith('.h5')]
        for count, slp_file in enumerate(slp_files):
            if any(slp_file[:-4] in h5_file for h5_file in h5_files):
                print(f"Skipping {slp_file} because corresponding .h5 file already exists")
                continue
            old_slp_path = os.path.join(root, slp_file).replace("\\","/")
            
            print("Current sleap_file", slp_file)
            slp_filename = slp_file.split("/")[-1]
            mouse, session = slp_file_parsing(slp_filename)
            print(f"mouse: {mouse} || session: {session}")
            SESSION_ROOT = os.path.join(root, session).replace("\\","/")
            print(f"SESSION_ROOT: {SESSION_ROOT}")

            new_slp_path = os.path.join(SESSION_ROOT,slp_filename).replace("\\","/")
            h5_path = new_slp_path.replace(".slp", ".h5")
            print(f"h5_path: {h5_path}")

            movie_filename = "_".join(slp_filename.split("_")[0:]).replace(
                ".avi.predictions.slp", ".avi")

            movie_path = os.path.join(root, movie_filename).replace("\\","/")
            # read the video file
            cap = cv2.VideoCapture(movie_path)
            # get the frames per second (FPS) of the video
            movie_fps = cap.get(cv2.CAP_PROP_FPS)
            # print the FPS
            print("FPS of video file: ", movie_fps)
            # release the video capture object and close all windows

            # Get the total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the length of the video in seconds
            video_length_seconds = total_frames / movie_fps

            # Print the FPS and length of the video
            print("FPS of video file:", movie_fps)
            print("Length of video file (seconds):", video_length_seconds)
            cap.release()
            #cv2.destroyAllWindows()

            new_movie_path = os.path.join(SESSION_ROOT, movie_filename).replace("\\","/")

            try:
                print(f"Processing {count + 1}/{len(slp_files)}")

                # 1) move the slp file
                os.makedirs(SESSION_ROOT, exist_ok=True)
                print(f"old path: {old_slp_path} || new path: {new_slp_path}")

                shutil.move(old_slp_path, new_slp_path)
                shutil.move(movie_path, new_movie_path)
                # 2) Convert .slp to .h5
                slp_to_h5(new_slp_path, h5_path)

                # 3) Extract speed
                #meta_data(h5_path)
                
                export_sleap_data_mult_nodes(h5_path, SESSION_ROOT, mouse, session, video_length_seconds, fps=movie_fps)
                #export_sleap_data_mult_nodes_body(h5_path, SESSION_ROOT,mouse, fps=30)
            except Exception as e:
                print(e)
                pass






if __name__ == "__main__":
    #main()
    new_main()
    #one_slp_file()
    #main_just_extract_vel()
