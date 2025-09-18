import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
import glob
from pathlib import Path
import random
import shutil

class MouseBehaviorAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.combined_data = None
        self.touchscreen_lines = {'left': None, 'right': None}
        self.body_parts = ['snout', 'left_ear', 'right_ear', 'body', 'tail']
        self.stretch_events = []
        self.video_path = None
        self.video_fps = 30  # Default FPS, will be updated when video is found
        
    def find_video_file(self):
        """Find the main video file for clip extraction"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        
        for ext in video_extensions:
            video_files = glob.glob(os.path.join(self.base_path, f'**/*{ext}'), recursive=True)
            if video_files:
                # Sort by file size (largest first) to get main video
                video_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                self.video_path = video_files[0]
                
                # Get video properties
                cap = cv2.VideoCapture(self.video_path)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                print(f"Found video: {self.video_path}")
                print(f"Video FPS: {self.video_fps}")
                print(f"Total frames: {total_frames}")
                
                return self.video_path
        
        print("Warning: No video file found for clip extraction")
        return None
    
    def frame_to_timestamp(self, frame_number):
        """Convert frame number to timestamp in seconds"""
        if self.video_fps > 0:
            return frame_number / self.video_fps
        return frame_number  # Return frame number if FPS unknown
    

    def load_existing_stretch_events(self, file_path=None):
        """Load existing stretch events from CSV file"""
        if file_path is None:
            file_path = os.path.join(self.base_path, 'stretch_events.csv')
        
        if os.path.exists(file_path):
            print(f"Found existing stretch events file: {file_path}")
            try:
                df = pd.read_csv(file_path)
                self.stretch_events = df.to_dict('records')
                print(f"Loaded {len(self.stretch_events)} existing stretch events")
                return True
            except Exception as e:
                print(f"Error loading stretch events file: {e}")
                return False
        else:
            print("No existing stretch events file found")
            return False

    def find_csv_files(self):
        """Find all CSV files ending with *_sleap_data in subdirectories"""
        csv_files = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('_sleap_data.csv'):
                    csv_files.append(os.path.join(root, file))
        return csv_files
    
    def combine_csv_data(self):
        """Combine all CSV files into a single DataFrame with body part labels"""
        csv_files = self.find_csv_files()
        if not csv_files:
            raise ValueError("No CSV files found matching pattern *_sleap_data.csv")
        
        combined_data = None
        
        for csv_file in csv_files:
            # Use subfolder name as the body part identifier
            subfolder_name = os.path.basename(os.path.dirname(csv_file))
            body_part = subfolder_name
            
            print(f"Processing {body_part} from {csv_file}")
            print(f"  Subfolder: {subfolder_name}")
            
            try:
                df = pd.read_csv(csv_file)
                
                # Check for required columns
                if 'x_pix' not in df.columns or 'y_pix' not in df.columns:
                    print(f"Warning: Required columns not found in {csv_file}")
                    print(f"  Available columns: {list(df.columns)}")
                    continue
                
                # Rename columns with subfolder name prefix
                df_renamed = df.rename(columns={
                    'x_pix': f'{body_part}_x_pix',
                    'y_pix': f'{body_part}_y_pix'
                })
                
                print(f"  Renamed columns: {body_part}_x_pix, {body_part}_y_pix")
                
                # Keep idx_time from first file
                if combined_data is None:
                    if 'idx_time' in df_renamed.columns:
                        combined_data = df_renamed[['idx_time', f'{body_part}_x_pix', f'{body_part}_y_pix']].copy()
                    else:
                        # Create idx_time if not present
                        df_renamed['idx_time'] = range(len(df_renamed))
                        combined_data = df_renamed[['idx_time', f'{body_part}_x_pix', f'{body_part}_y_pix']].copy()
                    print(f"  Initialized combined_data with {len(combined_data)} rows")
                else:
                    # Merge on idx_time if available, otherwise on index
                    if 'idx_time' in df_renamed.columns:
                        combined_data = combined_data.merge(
                            df_renamed[['idx_time', f'{body_part}_x_pix', f'{body_part}_y_pix']], 
                            on='idx_time', 
                            how='outer'
                        )
                    else:
                        # Merge on index
                        temp_df = df_renamed[[f'{body_part}_x_pix', f'{body_part}_y_pix']].copy()
                        temp_df['idx_time'] = range(len(temp_df))
                        combined_data = combined_data.merge(temp_df, on='idx_time', how='outer')
                    print(f"  Merged data, now {len(combined_data)} rows")
                        
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        if combined_data is None:
            raise ValueError("No valid CSV data could be processed")
        
        self.combined_data = combined_data.sort_values('idx_time').reset_index(drop=True)
        print(f"Combined data shape: {self.combined_data.shape}")
        print(f"Final columns: {list(self.combined_data.columns)}")
        return self.combined_data
    
    def save_combined_data(self, output_path=None):
        """Save combined data to CSV"""
        if self.combined_data is None:
            raise ValueError("No combined data available. Run combine_csv_data() first.")
        
        if output_path is None:
            output_path = os.path.join(self.base_path, 'combined_sleap_data.csv')
        
        self.combined_data.to_csv(output_path, index=False)
        print(f"Combined data saved to: {output_path}")
        return output_path
    
    def get_random_frame_image(self, frame_range=(1000, 2000)):
        """Get a random frame image for touchscreen annotation"""
        # Look for video files or image sequences in the base path
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        # Try to find video files first
        for ext in video_extensions:
            video_files = glob.glob(os.path.join(self.base_path, f'**/*{ext}'), recursive=True)
            if video_files:
                video_path = video_files[0]
                cap = cv2.VideoCapture(video_path)
                
                # Get random frame
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                random_frame = random.randint(
                    min(frame_range[0], total_frames-1), 
                    min(frame_range[1], total_frames-1)
                )
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    return frame
        
        # If no video, try image sequences
        for ext in image_extensions:
            image_files = glob.glob(os.path.join(self.base_path, f'**/*{ext}'), recursive=True)
            if image_files:
                # Sort and get random image from range
                image_files.sort()
                if len(image_files) > frame_range[0]:
                    start_idx = min(frame_range[0], len(image_files) - 1)
                    end_idx = min(frame_range[1], len(image_files) - 1)
                    random_idx = random.randint(start_idx, end_idx)
                    return cv2.imread(image_files[random_idx])
        
        # Create a blank image if no video/images found
        print("Warning: No video or image files found. Creating blank canvas for annotation.")
        return np.zeros((480, 640, 3), dtype=np.uint8) + 50
    
    def annotate_touchscreens(self):
        """Interactive annotation of touchscreen positions"""
        frame = self.get_random_frame_image()
        if frame is None:
            raise ValueError("Could not load reference frame")
        
        # Create annotation interface
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title("Click and drag to draw lines in front of left and right touchscreens\n(Left screen first, then right screen)")
        
        self.lines_drawn = []
        self.current_line = None
        
        def on_press(event):
            if event.inaxes != ax:
                return
            self.start_point = (event.xdata, event.ydata)
        
        def on_release(event):
            if event.inaxes != ax or self.start_point is None:
                return
            
            end_point = (event.xdata, event.ydata)
            
            if len(self.lines_drawn) < 2:
                # Draw line
                line_x = [self.start_point[0], end_point[0]]
                line_y = [self.start_point[1], end_point[1]]
                
                color = 'red' if len(self.lines_drawn) == 0 else 'blue'
                label = 'Left Screen' if len(self.lines_drawn) == 0 else 'Right Screen'
                
                ax.plot(line_x, line_y, color=color, linewidth=3, label=label)
                ax.legend()
                
                self.lines_drawn.append({
                    'start': self.start_point,
                    'end': end_point,
                    'screen': 'left' if len(self.lines_drawn) == 0 else 'right'
                })
                
                fig.canvas.draw()
                
                if len(self.lines_drawn) == 2:
                    ax.set_title("Touchscreen annotation complete. Close window to continue.")
        
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        
        plt.tight_layout()
        plt.show()
        
        if len(self.lines_drawn) < 2:
            raise ValueError("Both touchscreen lines must be drawn")
        
        # Store touchscreen line coordinates
        for line in self.lines_drawn:
            self.touchscreen_lines[line['screen']] = line
            
        print("Touchscreen annotation complete:")
        print(f"Left screen line: {self.touchscreen_lines['left']}")
        print(f"Right screen line: {self.touchscreen_lines['right']}")
        
        return self.touchscreen_lines
    
    def calculate_direction_vector(self, snout, body, tail):
        """Calculate direction vector from tail to snout through body"""
        if any(pd.isna(coord) for coord in [snout[0], snout[1], body[0], body[1], tail[0], tail[1]]):
            return None
        
        # Vector from tail to body
        tail_to_body = np.array([body[0] - tail[0], body[1] - tail[1]])
        # Vector from body to snout
        body_to_snout = np.array([snout[0] - body[0], snout[1] - body[1]])
        
        # Average direction vector
        direction = (tail_to_body + body_to_snout) / 2
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        
        return direction / norm
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
        point_vec = np.array([point[0] - line_start[0], point[1] - line_start[1]])
        
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj_length = max(min(proj_length, line_len), 0)
        proj = line_unitvec * proj_length
        
        return np.linalg.norm(point_vec - proj)
    
    def determine_facing_screen(self, snout, body, tail):
        """Determine which screen the mouse is facing based on body alignment"""
        direction = self.calculate_direction_vector(snout, body, tail)
        if direction is None:
            return None
        
        # Calculate which touchscreen line the mouse is facing
        left_line = self.touchscreen_lines['left']
        right_line = self.touchscreen_lines['right']
        
        # Get line center points
        left_center = np.array([
            (left_line['start'][0] + left_line['end'][0]) / 2,
            (left_line['start'][1] + left_line['end'][1]) / 2
        ])
        right_center = np.array([
            (right_line['start'][0] + right_line['end'][0]) / 2,
            (right_line['start'][1] + right_line['end'][1]) / 2
        ])
        
        # Vector from body to each screen
        body_pos = np.array([body[0], body[1]])
        to_left = left_center - body_pos
        to_right = right_center - body_pos
        
        # Normalize vectors
        to_left_norm = to_left / (np.linalg.norm(to_left) + 1e-8)
        to_right_norm = to_right / (np.linalg.norm(to_right) + 1e-8)
        
        # Dot product with direction vector
        left_alignment = np.dot(direction, to_left_norm)
        right_alignment = np.dot(direction, to_right_norm)
        
        # Return screen with better alignment (threshold for facing)
        if max(left_alignment, right_alignment) > 0.3:  # Adjustable threshold
            return 'left' if left_alignment > right_alignment else 'right'
        
        return None
    
    def detect_stretch_events(self, stretch_threshold=30, retract_threshold=20, min_duration=10):
        """Detect stretch and retract events"""
        if self.combined_data is None:
            raise ValueError("No combined data available. Run combine_csv_data() first.")
        
        if not self.touchscreen_lines['left'] or not self.touchscreen_lines['right']:
            raise ValueError("Touchscreen lines not annotated. Run annotate_touchscreens() first.")
        
        events = []
        
        # Find available body parts in data
        available_parts = {}
        for part in self.body_parts:
            x_col = f'{part}_x_pix'
            y_col = f'{part}_y_pix'
            if x_col in self.combined_data.columns and y_col in self.combined_data.columns:
                available_parts[part] = (x_col, y_col)
        
        print(f"Available body parts: {list(available_parts.keys())}")
        
        if 'snout' not in available_parts or 'body' not in available_parts or 'tail' not in available_parts:
            print("Warning: Missing required body parts (snout, body, tail) for full analysis")
        
        for i in range(len(self.combined_data)):
            row = self.combined_data.iloc[i]
            
            # Get coordinates for key body parts
            coords = {}
            for part, (x_col, y_col) in available_parts.items():
                coords[part] = (row[x_col], row[y_col])
            
            # Skip if essential parts are missing
            if 'snout' not in coords or 'body' not in coords or 'tail' not in coords:
                continue
            
            # Determine facing direction
            facing_screen = self.determine_facing_screen(
                coords['snout'], coords['body'], coords['tail']
            )
            
            if facing_screen is None:
                continue
            
            # Calculate distance from snout to target screen
            screen_line = self.touchscreen_lines[facing_screen]
            snout_to_screen = self.point_to_line_distance(
                coords['snout'],
                screen_line['start'],
                screen_line['end']
            )
            
            # Calculate body extension (distance from body center to snout)
            if not any(pd.isna(coord) for coord in coords['snout'] + coords['body']):
                body_extension = euclidean(coords['snout'], coords['body'])
            else:
                continue
            
            # Store frame data for event detection
            frame_data = {
                'frame': i,
                'time': row.get('idx_time', i),
                'facing_screen': facing_screen,
                'snout_to_screen_distance': snout_to_screen,
                'body_extension': body_extension,
                'coords': coords.copy()
            }
            
            # Simple stretch detection logic (can be enhanced)
            # Look for: approach -> extend -> retract pattern
            if i > min_duration and i < len(self.combined_data) - min_duration:
                # Get sliding window data
                window_data = []
                for j in range(i - min_duration, i + min_duration):
                    if j >= 0 and j < len(self.combined_data):
                        w_row = self.combined_data.iloc[j]
                        w_coords = {}
                        for part, (x_col, y_col) in available_parts.items():
                            w_coords[part] = (w_row[x_col], w_row[y_col])
                        
                        if all(part in w_coords for part in ['snout', 'body', 'tail']):
                            w_facing = self.determine_facing_screen(
                                w_coords['snout'], w_coords['body'], w_coords['tail']
                            )
                            if w_facing == facing_screen:
                                w_dist = self.point_to_line_distance(
                                    w_coords['snout'],
                                    screen_line['start'],
                                    screen_line['end']
                                )
                                w_ext = euclidean(w_coords['snout'], w_coords['body'])
                                window_data.append({
                                    'frame': j,
                                    'distance': w_dist,
                                    'extension': w_ext
                                })
                
                if len(window_data) >= min_duration:
                    # Analyze pattern
                    distances = [d['distance'] for d in window_data]
                    extensions = [d['extension'] for d in window_data]
                    
                    # Find local minima (closest approach) and check for extension pattern
                    mid_idx = len(window_data) // 2
                    min_dist_idx = np.argmin(distances)
                    
                    # Check if there's an approach -> extend -> retract pattern
                    if (min_dist_idx > 2 and min_dist_idx < len(window_data) - 3):
                        before_dist = np.mean(distances[:min_dist_idx-2])
                        closest_dist = distances[min_dist_idx]
                        after_dist = np.mean(distances[min_dist_idx+3:])
                        
                        before_ext = np.mean(extensions[:min_dist_idx-2])
                        peak_ext = np.max(extensions[min_dist_idx-1:min_dist_idx+2])
                        after_ext = np.mean(extensions[min_dist_idx+3:])
                        
                        # Check for stretch pattern
                        if (before_dist - closest_dist > stretch_threshold and
                            after_dist - closest_dist > retract_threshold and
                            peak_ext > before_ext + 10 and
                            peak_ext > after_ext + 10):
                            
                            # Calculate timestamps
                            start_time = self.frame_to_timestamp(window_data[0]['frame'])
                            peak_time = self.frame_to_timestamp(i)
                            end_time = self.frame_to_timestamp(window_data[-1]['frame'])
                            duration_seconds = end_time - start_time
                            # Convert seconds to hh:mm:ss format
                            def seconds_to_hms(seconds):
                                hours = int(seconds // 3600)
                                minutes = int((seconds % 3600) // 60)
                                secs = seconds % 60
                                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
                            
                            event = {
                                'start_frame': window_data[0]['frame'],
                                'peak_frame': i,
                                'end_frame': window_data[-1]['frame'],
                                'start_time_sec': start_time,
                                'peak_time_sec': peak_time,
                                'end_time_sec': end_time,
                                'start_time_hms': seconds_to_hms(start_time),
                                'peak_time_hms': seconds_to_hms(peak_time),
                                'end_time_hms': seconds_to_hms(end_time),
                                'duration_frames': len(window_data),
                                'duration_seconds': duration_seconds,
                                'duration_hms': seconds_to_hms(duration_seconds),
                                'facing_screen': facing_screen,
                                'closest_distance': closest_dist,
                                'max_extension': peak_ext,
                                'approach_distance': before_dist - closest_dist,
                                'retract_distance': after_dist - closest_dist
                            }
                            
                            # Avoid duplicate events
                            if not any(abs(e['peak_frame'] - event['peak_frame']) < min_duration 
                                     for e in events):
                                events.append(event)
        
        self.stretch_events = events
        print(f"Detected {len(events)} stretch events")
        
        return events
    
    def create_event_clips(self, num_clips=20, clip_duration_sec=5, output_folder=None):
        """Create video clips for a subset of detected stretch events"""
        if not self.stretch_events:
            print("No stretch events found. Run detect_stretch_events() first.")
            return []
        
        if not self.video_path:
            self.find_video_file()
            if not self.video_path:
                print("No video file found for clip creation")
                return []
        
        if output_folder is None:
            output_folder = os.path.join(self.base_path, 'event_clips')
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Select events for clipping (random sample or best events)
        events_to_clip = self.stretch_events.copy()
        
        # Sort by some quality metric (e.g., approach distance) and take diverse sample
        events_to_clip.sort(key=lambda x: x['peak_frame'])
        
        # Take a mix: best events + random sampling
        num_clips = min(num_clips, len(events_to_clip))
        if len(events_to_clip) > num_clips:
            best_events = events_to_clip[:num_clips//2]  # Top half based on approach distance
            remaining_events = events_to_clip[num_clips//2:]
            
            # Calculate how many random events we need
            num_random_needed = num_clips - len(best_events)
            num_random_available = len(remaining_events)
            num_random_to_take = min(num_random_needed, num_random_available)
            
            if num_random_to_take > 0:
                random_events = random.sample(remaining_events, num_random_to_take)
                events_to_clip = best_events + random_events
            else:
                events_to_clip = best_events
        
        # Final check to ensure we don't exceed requested number
        events_to_clip = events_to_clip[:num_clips]
        
        print(f"Creating {len(events_to_clip)} video clips...")
        
        # Simple video opening
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return []
        current_frame = 0
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video info: {total_frames} frames, {fps} FPS")
        
        created_clips = []
        
        for i, event in enumerate(events_to_clip):
            print(f"Processing event {i+1}/{len(events_to_clip)}...")
            
            try:
                # Calculate clip timing - make it simpler
                peak_frame = event['peak_frame']
                clip_frames = min(int(clip_duration_sec * fps), 150)  # Cap at 150 frames (~5 sec)
                
                start_frame = max(0, peak_frame - clip_frames // 2)
                end_frame = min(total_frames - 1, start_frame + clip_frames)
                
                print(f"  Clip frames: {start_frame} to {end_frame} ({end_frame - start_frame} frames)")
                
                # Test frame reading first
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                ret, test_frame = cap.read()
                if not ret:
                    print(f"  Warning: Cannot read start frame {start_frame}")
                    continue
                
                height, width = test_frame.shape[:2]
                print(f"  Frame size: {width}x{height}")
                
                # Create simple filename
                screen = event['facing_screen']
                clip_filename = f"event_{i+1:02d}_{screen}_frame_{peak_frame}.avi"
                clip_path = os.path.join(output_folder, clip_filename)
                
                # Use most compatible codec
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Most compatible
                out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    print(f"  Error: Cannot create video writer for {clip_filename}")
                    continue
                
                print(f"  Writing {end_frame - start_frame} frames...")
                
                # Write frames with progress tracking
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frames_written = 0
                max_frames = end_frame - start_frame
                
                for frame_idx in range(max_frames):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"  Warning: Could not read frame at index {frame_idx}")
                        break
                    
                    # Simple annotation (or skip for now)
                    try:
                        current_frame = start_frame + frame_idx
                        annotated_frame = self.annotate_frame(frame, event, current_frame)
                        out.write(annotated_frame)
                    except:
                        # If annotation fails, just write original frame
                        out.write(frame)
                    
                    frames_written += 1
                    
                    # Progress every 30 frames
                    if frames_written % 30 == 0:
                        print(f"    Progress: {frames_written}/{max_frames} frames")
                    
                    # Safety timeout - if we've written enough frames, stop
                    if frames_written >= max_frames:
                        break
                
                out.release()
                print(f"  Finished writing {frames_written} frames")
                
                # Verify file was created
                if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
                    created_clips.append({
                        'event_index': i,
                        'clip_path': clip_path,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'frames_written': frames_written,
                        'event_data': event
                    })
                    print(f"  ✓ Successfully created: {clip_filename} ({os.path.getsize(clip_path)} bytes)")
                else:
                    print(f"  ✗ Failed to create valid clip: {clip_filename}")
                    if os.path.exists(clip_path):
                        os.remove(clip_path)
                
            except Exception as e:
                print(f"  Error processing event {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        cap.release()
        
        # Create summary file
        if created_clips:
            summary_data = []
            for clip_info in created_clips:
                event = clip_info['event_data']
                summary_data.append({
                    'clip_filename': os.path.basename(clip_info['clip_path']),
                    'event_peak_frame': event['peak_frame'],
                    'event_peak_time_sec': event.get('peak_time_sec', 'N/A'),
                    'facing_screen': event['facing_screen'],
                    'approach_distance': event['approach_distance'],
                    'retract_distance': event['retract_distance'],
                    'duration_seconds': event.get('duration_seconds', 'N/A'),
                    'max_extension': event['max_extension'],
                    'clip_start_frame': clip_info['start_frame'],
                    'clip_end_frame': clip_info['end_frame'],
                    'frames_written': clip_info['frames_written']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_folder, 'clip_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"Clip summary saved to: {summary_path}")
        
        print(f"\n=== Clip Creation Complete ===")
        print(f"Successfully created {len(created_clips)} out of {len(events_to_clip)} attempted clips")
        return created_clips
    
    def create_simple_clips(self, num_clips=20):
        """Create simple clips without complex processing - fallback method"""
        print("Using simple clip creation method...")
        
        if not self.stretch_events:
            print("No stretch events found.")
            return []
        
        if not self.video_path:
            self.find_video_file()
            if not self.video_path:
                print("No video file found")
                return []
        
        output_folder = os.path.join(self.base_path, 'event_clips')
        os.makedirs(output_folder, exist_ok=True)
        
        # Select and sort events by peak frame for sequential processing
        events_to_clip = self.stretch_events[:num_clips]
        events_to_clip.sort(key=lambda x: x['peak_frame'])  # Sort by frame number
        print(f"Creating {len(events_to_clip)} simple clips...")
        
        cap = cv2.VideoCapture(self.video_path, cv2.CAP_MSMF)
        if not cap.isOpened():
            print("Cannot open video")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        created = []
        current_frame_pos = 0
        
        for i, event in enumerate(events_to_clip):
            print(f"Creating clip {i+1}/{len(events_to_clip)}...")
            
            peak_frame = event['peak_frame']
            start_frame = max(0, peak_frame - 60)
            clip_frames = 120
            
            screen = event['facing_screen']
            filename = f"event_{i+1:02d}_{screen}_frame_{peak_frame}.avi"
            filepath = os.path.join(output_folder, filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"  Cannot create writer for clip {i+1}")
                continue
            
            # Read sequentially from current position to start_frame
            while current_frame_pos < start_frame:
                cap.read()
                current_frame_pos += 1
            
            # Now create clip (already at correct position)
            written = 0
            for frame_idx in range(clip_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.putText(frame, f"Event {i+1} - {screen}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out.write(frame)
                written += 1
                current_frame_pos += 1
                
                if written % 30 == 0:
                    print(f"    {written} frames written")
            
            out.release()
            
            if os.path.exists(filepath) and os.path.getsize(filepath) > 5000:
                created.append(filepath)
                print(f"  ✓ Created: {filename}")
            else:
                print(f"  ✗ Failed: {filename}")
        
        cap.release()
        print(f"Created {len(created)} simple clips in {output_folder}")
        return created
    def annotate_frame(self, frame, event, current_frame):
        """Add annotations to frame showing event information"""
        annotated = frame.copy()
        
        # Add text overlay
        text_lines = [
            f"Frame: {current_frame}",
            f"Event Peak: {event['peak_frame']}",
            f"Screen: {event['facing_screen']}",
            f"Approach: {event['approach_distance']:.1f}px"
        ]
        
        # Add text background for better readability
        y_offset = 30
        for line in text_lines:
            cv2.rectangle(annotated, (10, y_offset - 20), (300, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(annotated, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Highlight current frame relative to event
        if current_frame == event['peak_frame']:
            cv2.rectangle(annotated, (5, 5), (annotated.shape[1]-5, annotated.shape[0]-5), (0, 255, 0), 3)
        elif event['start_frame'] <= current_frame <= event['end_frame']:
            cv2.rectangle(annotated, (5, 5), (annotated.shape[1]-5, annotated.shape[0]-5), (0, 255, 255), 2)
        
        return annotated       
     
    def save_stretch_events(self, output_path=None):
        """Save detected stretch events to CSV"""
        if not self.stretch_events:
            print("No stretch events to save")
            return
        
        if output_path is None:
            output_path = os.path.join(self.base_path, 'stretch_events.csv')
        
        df_events = pd.DataFrame(self.stretch_events)
        df_events.to_csv(output_path, index=False)
        print(f"Stretch events saved to: {output_path}")
        
        return output_path
    
    def run_full_analysis(self, create_clips=True, num_clips=20, simple_clips=True):
        """Run the complete analysis pipeline"""
        print("=== Mouse Behavior Analysis Pipeline ===")
        
        # Check for existing stretch events first
        print("\n0. Checking for existing analysis...")
        events_loaded = self.load_existing_stretch_events()
        
        if not events_loaded:
            # Run full analysis
            print("\n1. Finding video file...")
            self.find_video_file()
            
            print("\n2. Combining CSV data...")
            self.combine_csv_data()
            self.save_combined_data()
            
            print("\n3. Annotating touchscreen positions...")
            self.annotate_touchscreens()
            
            print("\n4. Detecting stretch events...")
            events = self.detect_stretch_events()
            
            print("\n5. Saving results...")
            self.save_stretch_events()
        else:
            # Just find video for clip creation
            self.find_video_file()
            events = self.stretch_events
            print("Using existing stretch events for analysis.")
        
        # Create video clips (whether events were loaded or just created)
        if create_clips and self.video_path and events:
            if simple_clips:
                print(f"\n6. Creating {num_clips} simple video clips...")
                self.create_simple_clips(num_clips)
            else:
                print(f"\n6. Creating {num_clips} video clips for validation...")
                try:
                    self.create_event_clips(num_clips=num_clips)
                except Exception as e:
                    print(f"Complex clip creation failed: {e}")
                    print("Falling back to simple clip creation...")
                    self.create_simple_clips(num_clips=num_clips)
        elif create_clips:
            print("\n6. Skipping video clips (no video file or no events found)")
        
        # Rest of your existing summary code...
                
            # Print summary
            if events:
                left_events = sum(1 for e in events if e['facing_screen'] == 'left')
                right_events = sum(1 for e in events if e['facing_screen'] == 'right')
                
                print(f"Left screen events: {left_events}")
                print(f"Right screen events: {right_events}")
                print(f"Average duration: {np.mean([e['duration_seconds'] for e in events]):.2f} seconds")
                print(f"Average approach distance: {np.mean([e['approach_distance'] for e in events]):.1f} pixels")

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    base_path = r"C:\Users\dicbr\Desktop\D2-9\RDT D1"
    
    analyzer = MouseBehaviorAnalyzer(base_path)
    
    try:
        # Option 1: Run with simple clips (recommended for troubleshooting)
        analyzer.run_full_analysis(create_clips=True, simple_clips=True)
        
        # Option 2: Run without clips
        # analyzer.run_full_analysis(create_clips=False)
        
        # Option 3: Run with complex clips (if simple works)
        # analyzer.run_full_analysis(create_clips=True, num_clips=20, simple_clips=False)
        
        # Optional: Access results programmatically
        print("\nFirst few stretch events:")
        for i, event in enumerate(analyzer.stretch_events[:3]):
            print(f"Event {i+1}: Frame {event['peak_frame']} ({event['peak_time_sec']:.2f}s), "
                  f"Screen: {event['facing_screen']}, "
                  f"Duration: {event['duration_seconds']:.2f} seconds")
                  
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()