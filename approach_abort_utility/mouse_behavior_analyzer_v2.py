import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import glob
import random

class ApproachAbortDetector:
    def __init__(self, base_path):
        self.base_path = base_path
        self.combined_data = None
        self.touchscreen_lines = {'left': None, 'right': None}
        self.video_path = None
        self.video_fps = 30
        self.approach_events = []
        self.houselight_onset_time = None
        
    def find_video_file(self):
        """Find the main video file"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        
        for ext in video_extensions:
            video_files = glob.glob(os.path.join(self.base_path, f'**/*{ext}'), recursive=True)
            if video_files:
                video_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                self.video_path = video_files[0]
                
                cap = cv2.VideoCapture(self.video_path)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                print(f"Found video: {os.path.basename(self.video_path)}")
                print(f"Video FPS: {self.video_fps}")
                print(f"Total frames: {total_frames}")
                return self.video_path
        
        print("Warning: No video file found")
        return None
    
    def frame_to_timestamp(self, frame_number):
        """Convert frame number to timestamp in seconds"""
        return frame_number / self.video_fps if self.video_fps > 0 else frame_number
    
    def load_existing_data(self):
        """Load existing combined data and events if available"""
        combined_file = os.path.join(self.base_path, 'combined_sleap_data.csv')
        events_file = os.path.join(self.base_path, 'approach_events.csv')
        
        combined_loaded = False
        events_loaded = False
        
        if os.path.exists(combined_file):
            try:
                self.combined_data = pd.read_csv(combined_file)
                print(f"Loaded existing combined data: {self.combined_data.shape}")
                combined_loaded = True
            except Exception as e:
                print(f"Error loading combined data: {e}")
        
        if os.path.exists(events_file):
            try:
                df = pd.read_csv(events_file)
                self.approach_events = df.to_dict('records')
                print(f"Loaded existing approach events: {len(self.approach_events)}")
                events_loaded = True
            except Exception as e:
                print(f"Error loading approach events: {e}")
        
        return combined_loaded, events_loaded
    
    def combine_csv_data(self):
        """Combine SLEAP CSV files"""
        csv_files = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('_sleap_data.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            raise ValueError("No SLEAP CSV files found")
        
        combined_data = None
        
        for csv_file in csv_files:
            subfolder_name = os.path.basename(os.path.dirname(csv_file))
            print(f"Processing {subfolder_name}")
            
            try:
                df = pd.read_csv(csv_file)
                if 'x_pix' not in df.columns or 'y_pix' not in df.columns:
                    continue
                
                df_renamed = df.rename(columns={
                    'x_pix': f'{subfolder_name}_x_pix',
                    'y_pix': f'{subfolder_name}_y_pix'
                })
                
                if combined_data is None:
                    if 'idx_time' in df_renamed.columns:
                        combined_data = df_renamed[['idx_time', f'{subfolder_name}_x_pix', f'{subfolder_name}_y_pix']].copy()
                    else:
                        df_renamed['idx_time'] = range(len(df_renamed))
                        combined_data = df_renamed[['idx_time', f'{subfolder_name}_x_pix', f'{subfolder_name}_y_pix']].copy()
                else:
                    if 'idx_time' in df_renamed.columns:
                        combined_data = combined_data.merge(
                            df_renamed[['idx_time', f'{subfolder_name}_x_pix', f'{subfolder_name}_y_pix']], 
                            on='idx_time', how='outer'
                        )
                        
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        self.combined_data = combined_data.sort_values('idx_time').reset_index(drop=True)
        
        # Save combined data
        output_path = os.path.join(self.base_path, 'combined_sleap_data.csv')
        self.combined_data.to_csv(output_path, index=False)
        print(f"Combined data saved: {self.combined_data.shape}")
        
        return self.combined_data
    
    def annotate_touchscreens(self):
        """Interactive touchscreen annotation"""
        # Get random frame for annotation
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame = random.randint(1000, min(2000, total_frames-1))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Could not load reference frame")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title("Click two points to define LEFT screen area, then two points for RIGHT screen")
        
        self.screen_points = []
        
        def onclick(event):
            if event.inaxes != ax:
                return
            
            self.screen_points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
            
            if len(self.screen_points) == 2:
                ax.plot([self.screen_points[0][0], self.screen_points[1][0]], 
                       [self.screen_points[0][1], self.screen_points[1][1]], 'r-', linewidth=3, label='Left Screen')
                ax.legend()
                ax.set_title("Now click two points to define RIGHT screen area")
            elif len(self.screen_points) == 4:
                ax.plot([self.screen_points[2][0], self.screen_points[3][0]], 
                       [self.screen_points[2][1], self.screen_points[3][1]], 'b-', linewidth=3, label='Right Screen')
                ax.legend()
                ax.set_title("Annotation complete. Close window to continue.")
            
            fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        
        if len(self.screen_points) < 4:
            raise ValueError("Need to define both screen areas")
        
        # Store screen definitions
        self.touchscreen_lines['left'] = {
            'start': self.screen_points[0],
            'end': self.screen_points[1]
        }
        self.touchscreen_lines['right'] = {
            'start': self.screen_points[2],
            'end': self.screen_points[3]
        }
        
        print("Screen annotation complete")
        return self.touchscreen_lines
    
    def detect_houselight_onset(self, search_frames=2000, min_change_threshold=5):
        """
        Detect houselight onset with interactive selection from ALL abrupt brightness changes
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        
        print(f"Analyzing brightness in first {search_frames} frames...")
        
        brightness_values = []
        
        for frame_num in range(min(search_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
        
        cap.release()
        
        print(f"Analyzed {len(brightness_values)} frames")
        
        # Calculate frame-to-frame brightness changes
        brightness_changes = np.diff(brightness_values)
        
        print(f"Calculated {len(brightness_changes)} brightness changes")
        print(f"Change range: {brightness_changes.min():.1f} to {brightness_changes.max():.1f}")
        
        # Find ALL changes above threshold - EVERY SINGLE ONE
        all_change_indices = []
        all_change_magnitudes = []
        
        for i in range(len(brightness_changes)):
            change = brightness_changes[i]
            if abs(change) >= min_change_threshold:
                all_change_indices.append(i)
                all_change_magnitudes.append(change)
        
        print(f"Found {len(all_change_indices)} changes above threshold ±{min_change_threshold}")
        
        if len(all_change_indices) == 0:
            print(f"No changes found above threshold {min_change_threshold}. Lowering threshold to 1...")
            min_change_threshold = 1
            for i in range(len(brightness_changes)):
                change = brightness_changes[i]
                if abs(change) >= min_change_threshold:
                    all_change_indices.append(i)
                    all_change_magnitudes.append(change)
            print(f"With threshold {min_change_threshold}: found {len(all_change_indices)} changes")
        
        if len(all_change_indices) == 0:
            print("No brightness changes detected at all")
            return None
        
        # Create change data for plotting
        all_changes = []
        for i, change_idx in enumerate(all_change_indices):
            frame_num = change_idx + 1  # +1 because diff shifts indices
            time_sec = self.frame_to_timestamp(frame_num)
            magnitude = all_change_magnitudes[i]
            
            all_changes.append({
                'frame': frame_num,
                'time': time_sec,
                'magnitude': magnitude,
                'type': 'increase' if magnitude > 0 else 'decrease',
                'index': change_idx
            })
        
        # Sort by time
        all_changes.sort(key=lambda x: x['time'])
        
        print(f"\nDetected {len(all_changes)} brightness changes:")
        for i, change in enumerate(all_changes[:10]):  # Show first 10
            print(f"  {i+1}. Frame {change['frame']} ({change['time']:.2f}s): {change['magnitude']:+.1f}")
        if len(all_changes) > 10:
            print(f"  ... and {len(all_changes)-10} more")
        
        # Create interactive plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        
        # Plot brightness over time
        frame_times = [self.frame_to_timestamp(i) for i in range(len(brightness_values))]
        ax.plot(frame_times, brightness_values, 'b-', linewidth=1, alpha=0.8, label='Brightness')
        
        # Add a vertical line for EVERY SINGLE brightness change
        change_lines = []
        
        print(f"Plotting {len(all_changes)} vertical lines...")
        
        for i, change in enumerate(all_changes):
            time_sec = change['time']
            magnitude = change['magnitude']
            
            # Color based on increase/decrease
            if magnitude > 0:
                color = 'green'
                alpha = 0.5
            else:
                color = 'red'
                alpha = 0.5
            
            # Create clickable vertical line for EVERY change
            line = ax.axvline(x=time_sec, color=color, linestyle='-', linewidth=1, 
                             alpha=alpha, picker=True, pickradius=3)
            line.change_data = change  # Attach change info to line
            line.original_alpha = alpha
            line.original_linewidth = 1
            change_lines.append(line)
        
        print(f"Plotted {len(change_lines)} vertical lines")
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Average Brightness', fontsize=12)
        ax.set_title(f'Brightness Over Time - {len(all_changes)} Changes Plotted\n' + 
                    'Every vertical line = one brightness change. Click any line to select houselight onset', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Add instruction text
        instruction_text = (
            'INSTRUCTIONS:\n'
            f'• {len(all_changes)} vertical lines plotted\n'
            f'• Threshold used: ±{min_change_threshold}\n'
            '• Green = increases, Red = decreases\n'
            '• Click ANY line to select\n'
            '• Close window after selection'
        )
        ax.text(0.02, 0.98, instruction_text, transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        # Track selected change
        selected_change = {'value': None}
        
        def on_pick(event):
            if hasattr(event.artist, 'change_data'):
                change = event.artist.change_data
                selected_change['value'] = change
                
                # Reset all lines to normal appearance
                for line in change_lines:
                    line.set_linewidth(line.original_linewidth)
                    line.set_alpha(line.original_alpha)
                
                # Highlight selected line
                event.artist.set_linewidth(3)
                event.artist.set_alpha(1.0)
                
                # Update title to show selection
                change_type = "increase" if change['magnitude'] > 0 else "decrease"
                ax.set_title(f'SELECTED: Frame {change["frame"]} at {change["time"]:.2f}s ' +
                           f'({change["magnitude"]:+.1f} brightness {change_type})\n' +
                           'Close window to confirm selection', fontsize=14)
                
                fig.canvas.draw()
                
                print(f"Selected: Frame {change['frame']} ({change['time']:.2f}s) - {change['magnitude']:+.1f} brightness {change_type}")
        
        # Connect the pick event
        fig.canvas.mpl_connect('pick_event', on_pick)
        
        plt.tight_layout()
        plt.show()
        
        # Process the selection
        if selected_change['value'] is not None:
            change = selected_change['value']
            selected_time = change['time']
            selected_frame = change['frame']
            
            print(f"\nUser selected:")
            print(f"  Frame: {selected_frame}")
            print(f"  Time: {selected_time:.3f}s")
            print(f"  Brightness change: {change['magnitude']:+.1f} ({change['type']})")
            
            # Confirm selection
            response = input(f"\nConfirm houselight onset at {selected_time:.2f}s? (y/n): ")
            
            if response.lower() == 'y':
                self.houselight_onset_time = selected_time
                print(f"Houselight onset set to {selected_time:.3f}s")
                return selected_time
            else:
                print("Selection cancelled")
                return None
        else:
            print("No change selected. Please run detection again and click on a line.")
            return None
    
    def debug_brightness_detection(self, search_frames=2000, show_frames=10):
        """
        Debug function to examine brightness changes in detail
        Shows actual frame images around brightness changes
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Cannot open video")
            return
        
        print(f"Debugging brightness detection...")
        
        brightness_values = []
        for frame_num in range(min(search_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            brightness_values.append(brightness)
        
        # Find top changes
        brightness_changes = np.diff(brightness_values)
        top_changes = np.argsort(brightness_changes)[-5:][::-1]
        
        print(f"Examining top {len(top_changes)} brightness changes:")
        
        for i, change_idx in enumerate(top_changes):
            frame_num = change_idx + 1
            change_magnitude = brightness_changes[change_idx]
            time_sec = self.frame_to_timestamp(frame_num)
            
            print(f"\n{i+1}. Frame {frame_num} ({time_sec:.2f}s): +{change_magnitude:.1f} brightness")
            
            # Show frames before and after the change
            for offset in [-2, -1, 0, 1, 2]:
                check_frame = max(0, frame_num + offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
                ret, frame = cap.read()
                if ret:
                    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    print(f"    Frame {check_frame}: brightness = {brightness:.1f}")
        
        cap.release()
    
    def get_body_keypoints(self):
        """
        Find and organize all available body keypoints for mouse skeleton
        Returns dictionary with keypoint coordinates
        """
        print("Identifying available body keypoints...")
        
        # Common keypoint names in SLEAP data
        keypoint_patterns = {
            'head_parts': ['snout', 'nose', 'tip', 'left_ear', 'leftear', 'ear_left', 'right_ear', 'rightear', 'ear_right'],
            'body_parts': ['body','neck', 'shoulder', 'chest', 'back', 'spine', 'center', 'mid', 'torso'],
            'tail_parts': ['tail', 'tail_base', 'tailbase', 'tail_tip', 'tailtip', 'tail_end'],
            'limb_parts': ['front_left', 'front_right', 'back_left', 'back_right', 'paw', 'leg']
        }
        
        found_keypoints = {}
        
        for category, patterns in keypoint_patterns.items():
            found_keypoints[category] = {}
            for pattern in patterns:
                x_col = f'{pattern}_x_pix'
                y_col = f'{pattern}_y_pix'
                if x_col in self.combined_data.columns and y_col in self.combined_data.columns:
                    found_keypoints[category][pattern] = {'x': x_col, 'y': y_col}
                    print(f"Found {category}: {pattern}")
        
        # Remove empty categories
        found_keypoints = {k: v for k, v in found_keypoints.items() if v}
        
        return found_keypoints
    
    def create_robust_head_position(self, keypoints):
        """
        Create a single robust head position by averaging available head keypoints
        """
        head_parts = keypoints.get('head_parts', {})
        
        if not head_parts:
            raise ValueError("No head keypoints found")
        
        print(f"Creating robust head position from: {list(head_parts.keys())}")
        
        # Create arrays for interpolated head coordinates
        head_x = np.full(len(self.combined_data), np.nan)
        head_y = np.full(len(self.combined_data), np.nan)
        head_quality = np.zeros(len(self.combined_data))
        
        for i in range(len(self.combined_data)):
            valid_x = []
            valid_y = []
            
            # Collect all valid head coordinates
            for part_name, cols in head_parts.items():
                x_val = self.combined_data.iloc[i][cols['x']]
                y_val = self.combined_data.iloc[i][cols['y']]
                
                if not (pd.isna(x_val) or pd.isna(y_val)):
                    valid_x.append(x_val)
                    valid_y.append(y_val)
            
            # Calculate average position if we have valid points
            if valid_x:
                head_x[i] = np.mean(valid_x)
                head_y[i] = np.mean(valid_y)
                head_quality[i] = len(valid_x)
        
        # Fill small gaps using interpolation
        self._interpolate_missing_segments(head_x, max_gap=10)
        self._interpolate_missing_segments(head_y, max_gap=10)
        
        # Add to combined data
        self.combined_data['head_x_robust'] = head_x
        self.combined_data['head_y_robust'] = head_y
        self.combined_data['head_quality'] = head_quality
        
        valid_points = np.sum(~np.isnan(head_x))
        total_points = len(head_x)
        
        print(f"Robust head position created:")
        print(f"  Valid positions: {valid_points}/{total_points} ({100*valid_points/total_points:.1f}%)")
        
        return head_x, head_y, head_quality
    
    def get_body_orientation(self, keypoints):
        """
        Calculate mouse body orientation using tail and body keypoints
        This gives us the overall direction the mouse is facing
        """
        body_parts = keypoints.get('body_parts', {})
        tail_parts = keypoints.get('tail_parts', {})
        
        if not body_parts and not tail_parts:
            print("Warning: No body or tail keypoints found for orientation calculation")
            return None, None
        
        print(f"Calculating body orientation from body parts: {list(body_parts.keys())}")
        print(f"And tail parts: {list(tail_parts.keys())}")
        
        # Create arrays for body orientation
        orientation_angle = np.full(len(self.combined_data), np.nan)
        orientation_quality = np.zeros(len(self.combined_data))
        
        for i in range(len(self.combined_data)):
            # Try to get a body center point
            body_x, body_y = None, None
            tail_x, tail_y = None, None
            
            # Get body center from available body parts
            body_valid_x, body_valid_y = [], []
            for part_name, cols in body_parts.items():
                x_val = self.combined_data.iloc[i][cols['x']]
                y_val = self.combined_data.iloc[i][cols['y']]
                if not (pd.isna(x_val) or pd.isna(y_val)):
                    body_valid_x.append(x_val)
                    body_valid_y.append(y_val)
            
            if body_valid_x:
                body_x = np.mean(body_valid_x)
                body_y = np.mean(body_valid_y)
            
            # Get tail position from available tail parts
            tail_valid_x, tail_valid_y = [], []
            for part_name, cols in tail_parts.items():
                x_val = self.combined_data.iloc[i][cols['x']]
                y_val = self.combined_data.iloc[i][cols['y']]
                if not (pd.isna(x_val) or pd.isna(y_val)):
                    tail_valid_x.append(x_val)
                    tail_valid_y.append(y_val)
            
            if tail_valid_x:
                tail_x = np.mean(tail_valid_x)
                tail_y = np.mean(tail_valid_y)
            
            # Calculate orientation if we have both body and tail
            if body_x is not None and tail_x is not None:
                # Vector from tail to body = direction mouse is facing
                direction_x = body_x - tail_x
                direction_y = body_y - tail_y
                
                # Calculate angle in degrees
                angle_rad = np.arctan2(direction_y, direction_x)
                angle_deg = np.degrees(angle_rad)
                
                # Normalize to 0-360 degrees
                if angle_deg < 0:
                    angle_deg += 360
                
                orientation_angle[i] = angle_deg
                orientation_quality[i] = len(body_valid_x) + len(tail_valid_x)
        
        # Fill small gaps using interpolation
        self._interpolate_missing_segments(orientation_angle, max_gap=5)
        
        # Add to combined data
        self.combined_data['body_orientation'] = orientation_angle
        self.combined_data['orientation_quality'] = orientation_quality
        
        valid_orientations = np.sum(~np.isnan(orientation_angle))
        total_points = len(orientation_angle)
        
        print(f"Body orientation calculated:")
        print(f"  Valid orientations: {valid_orientations}/{total_points} ({100*valid_orientations/total_points:.1f}%)")
        
        return orientation_angle, orientation_quality
    
    def _interpolate_missing_segments(self, coords, max_gap=10):
        """Fill small gaps in coordinate data using linear interpolation"""
        valid_indices = ~np.isnan(coords)
        
        if not np.any(valid_indices):
            return coords
        
        # Find gaps
        diff = np.diff(valid_indices.astype(int))
        gap_starts = np.where(diff == -1)[0] + 1
        gap_ends = np.where(diff == 1)[0]
        
        # Handle edge cases
        if valid_indices[0] == False:
            gap_starts = np.concatenate([[0], gap_starts])
        if valid_indices[-1] == False:
            gap_ends = np.concatenate([gap_ends, [len(coords)-1]])
        
        # Interpolate small gaps
        for start, end in zip(gap_starts, gap_ends):
            gap_length = end - start + 1
            
            if gap_length <= max_gap:
                # Find nearest valid points
                before_idx = start - 1 if start > 0 else None
                after_idx = end + 1 if end < len(coords) - 1 else None
                
                if before_idx is not None and after_idx is not None:
                    # Linear interpolation
                    before_val = coords[before_idx]
                    after_val = coords[after_idx]
                    
                    for i in range(start, end + 1):
                        t = (i - before_idx) / (after_idx - before_idx)
                        coords[i] = before_val + t * (after_val - before_val)
        
        return coords
    
    def determine_screen_from_orientation(self, head_pos, body_orientation):
        """
        Determine which screen the mouse is approaching based on body orientation
        
        Args:
            head_pos: (x, y) position of mouse head
            body_orientation: angle in degrees (0-360) that mouse body is facing
            
        Returns:
            ('left'/'right'/None, confidence_score)
        """
        if pd.isna(body_orientation):
            return None, 0.0
        
        head_x, head_y = head_pos
        
        # Get screen centers
        left_center_x = (self.touchscreen_lines['left']['start'][0] + 
                        self.touchscreen_lines['left']['end'][0]) / 2
        left_center_y = (self.touchscreen_lines['left']['start'][1] + 
                        self.touchscreen_lines['left']['end'][1]) / 2
        
        right_center_x = (self.touchscreen_lines['right']['start'][0] + 
                         self.touchscreen_lines['right']['end'][0]) / 2
        right_center_y = (self.touchscreen_lines['right']['start'][1] + 
                         self.touchscreen_lines['right']['end'][1]) / 2
        
        # Calculate angles from head to each screen
        left_angle = np.degrees(np.arctan2(left_center_y - head_y, left_center_x - head_x))
        right_angle = np.degrees(np.arctan2(right_center_y - head_y, right_center_x - head_x))
        
        # Normalize to 0-360
        if left_angle < 0:
            left_angle += 360
        if right_angle < 0:
            right_angle += 360
        
        # Calculate angular differences (handle wrap-around)
        def angular_diff(a1, a2):
            diff = abs(a1 - a2)
            return min(diff, 360 - diff)
        
        left_diff = angular_diff(body_orientation, left_angle)
        right_diff = angular_diff(body_orientation, right_angle)
        
        # Determine target screen (smaller angle difference = better match)
        if left_diff < right_diff:
            confidence = max(0, (90 - left_diff) / 90)  # Scale so 0° diff = 1.0 confidence
            return 'left', confidence
        else:
            confidence = max(0, (90 - right_diff) / 90)
            return 'right', confidence
    
    def distance_to_line(self, point, line_start, line_end):
        """Calculate distance from point to line"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Distance from point to line formula
        num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        den = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        return num / den if den > 0 else float('inf')
    
    def detect_approach_events(self, proximity_threshold=60, velocity_threshold=15, 
                             time_window_frames=30, min_quality=1, suppression_frames=10):
        """
        Detect approach events using robust head position and body orientation
        
        Args:
            proximity_threshold: Max distance to screen to consider "approach"
            velocity_threshold: Min velocity change to consider "rapid movement"
            time_window_frames: Frames to analyze for velocity change (~1 sec at 30fps)
            min_quality: Minimum head tracking quality required
            suppression_frames: Minimum frames between events (prevents duplicates)
        """
        if self.combined_data is None:
            raise ValueError("No combined data available")
        
        # Get body keypoints
        keypoints = self.get_body_keypoints()
        
        # Create robust head position
        head_x, head_y, head_quality = self.create_robust_head_position(keypoints)
        
        # Get body orientation
        body_orientation, orientation_quality = self.get_body_orientation(keypoints)
        
        events = []
        last_event_frame = -float('inf')  # Track last event frame for suppression
        
        print(f"Analyzing {len(self.combined_data)} frames for approach events...")
        print("Using body orientation for screen determination")
        print(f"Suppression window: {suppression_frames} frames ({suppression_frames/self.video_fps:.2f} seconds)")
        
        for i in range(time_window_frames, len(self.combined_data) - time_window_frames):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(self.combined_data)} frames")
            
            # Skip if we're too close to the last detected event (suppression)
            if i - last_event_frame < suppression_frames:
                continue
            
            # Skip if tracking quality is too low
            if head_quality[i] < min_quality:
                continue
            
            # Get current head position
            head_x_curr = head_x[i]
            head_y_curr = head_y[i]
            
            # Skip if coordinates are missing
            if pd.isna(head_x_curr) or pd.isna(head_y_curr):
                continue
            
            # Check if mouse is near either screen
            left_dist = self.distance_to_line(
                (head_x_curr, head_y_curr),
                self.touchscreen_lines['left']['start'],
                self.touchscreen_lines['left']['end']
            )
            
            right_dist = self.distance_to_line(
                (head_x_curr, head_y_curr),
                self.touchscreen_lines['right']['start'],
                self.touchscreen_lines['right']['end']
            )
            
            # Only proceed if mouse is near at least one screen
            near_screen = (left_dist < proximity_threshold) or (right_dist < proximity_threshold)
            
            if not near_screen:
                continue
            
            # Determine target screen using body orientation
            current_orientation = body_orientation[i] if body_orientation is not None else np.nan
            target_screen, screen_confidence = self.determine_screen_from_orientation(
                (head_x_curr, head_y_curr), current_orientation
            )
            
            if target_screen is None:
                continue
            
            # Check for rapid movement (approach event detection)
            future_positions = []
            quality_scores = []
            
            for j in range(i, min(i + time_window_frames, len(self.combined_data))):
                fx = head_x[j]
                fy = head_y[j]
                quality = head_quality[j]
                
                if not (pd.isna(fx) or pd.isna(fy)) and quality >= min_quality:
                    future_positions.append((fx, fy))
                    quality_scores.append(quality)
            
            if len(future_positions) >= time_window_frames // 2:
                # Calculate movement distance and velocity
                start_pos = (head_x_curr, head_y_curr)
                end_pos = future_positions[-1]
                movement_distance = euclidean(start_pos, end_pos)
                
                velocity = movement_distance / (len(future_positions) / self.video_fps)
                
                avg_quality = np.mean([head_quality[i]] + quality_scores)
                
                # If rapid movement detected, record event
                if velocity > velocity_threshold:
                    event_time = self.frame_to_timestamp(i)
                    
                    # Convert to hh:mm:ss format
                    hours = int(event_time // 3600)
                    minutes = int((event_time % 3600) // 60)
                    seconds = event_time % 60
                    time_hms = f"{hours:02d}h{minutes:02d}m{seconds:06.3f}s"
                    
                    event = {
                        'frame': i,
                        'time_sec': event_time,
                        'time_hms': time_hms,
                        'screen': target_screen,
                        'velocity': velocity,
                        'movement_distance': movement_distance,
                        'head_x': head_x_curr,
                        'head_y': head_y_curr,
                        'end_x': end_pos[0],
                        'end_y': end_pos[1],
                        'body_orientation': current_orientation,
                        'screen_confidence': screen_confidence,
                        'head_quality': avg_quality,
                        'detection_method': 'body_orientation',
                        'frames_since_last_event': i - last_event_frame if last_event_frame != -float('inf') else float('inf')
                    }
                    
                    events.append(event)
                    last_event_frame = i  # Update suppression tracker
                    
                    print(f"Event detected at frame {i} (suppressed next {suppression_frames} frames)")
        
        self.approach_events = events
        
        # Calculate suppression statistics
        if len(events) > 1:
            frame_gaps = [events[i]['frames_since_last_event'] for i in range(1, len(events))]
            frame_gaps = [gap for gap in frame_gaps if gap != float('inf')]
            if frame_gaps:
                min_gap = min(frame_gaps)
                avg_gap = np.mean(frame_gaps)
                print(f"Event spacing: min={min_gap} frames ({min_gap/self.video_fps:.2f}s), avg={avg_gap:.1f} frames ({avg_gap/self.video_fps:.2f}s)")
        
        print(f"Detected {len(events)} approach events using body orientation method with suppression")
        return events
    
    def load_touchscreen_data(self, touchscreen_file=None):
        """Load touchscreen data"""
        if touchscreen_file is None:
            # Try both possible locations
            possible_files = [
                os.path.join(self.base_path, "RRD414 05142024 ABET.csv"),
                os.path.join(os.path.dirname(self.base_path), "RRD414 05142024 ABET.csv")
            ]
            
            for file_path in possible_files:
                if os.path.exists(file_path):
                    touchscreen_file = file_path
                    break
        
        if touchscreen_file is None or not os.path.exists(touchscreen_file):
            print("Touchscreen data file not found")
            return None
        
        try:
            df = pd.read_csv(touchscreen_file)
            print(f"Loaded touchscreen data: {df.shape}")
            
            # Filter for touch events
            touch_events = df[df.iloc[:, 2].str.contains("Touch", case=False, na=False)]
            touch_times = touch_events.iloc[:, 0].values
            
            print(f"Found {len(touch_times)} touch events")
            return touch_times
            
        except Exception as e:
            print(f"Error loading touchscreen data: {e}")
            return None
    
    def classify_events(self, touch_times, touch_window=1):
        """
        Classify approach events as touch vs abort
        Correctly converts both timelines to the same reference point
        """
        if touch_times is None or self.houselight_onset_time is None:
            print("Cannot classify events without touch data and houselight onset")
            return
        
        print(f"Classifying events with correct timeline conversion...")
        print(f"Houselight onset: {self.houselight_onset_time:.3f}s")
        print(f"Touch window: ±{touch_window}s")
        
        # Convert approach events to same reference as ABET by subtracting houselight onset
        # This converts absolute video timestamps to relative timestamps
        approach_relative_times = []
        for event in self.approach_events:
            relative_time = event['time_sec'] - self.houselight_onset_time
            approach_relative_times.append(relative_time)
        
        print(f"Total touch events in ABET data: {len(touch_times)}")
        print(f"ABET touch times range: {touch_times.min():.3f}s to {touch_times.max():.3f}s")
        print(f"Approach events (converted) range: {min(approach_relative_times):.3f}s to {max(approach_relative_times):.3f}s")
        
        # Classify each approach event
        touch_events = 0
        abort_events = 0
        
        for i, event in enumerate(self.approach_events):
            # Convert approach event to same reference as ABET
            approach_relative_time = event['time_sec'] - self.houselight_onset_time
            
            # Look for touch within touch_window seconds of the approach event
            time_diffs = np.abs(touch_times - approach_relative_time)
            nearby_touches = time_diffs <= touch_window
            
            if np.any(nearby_touches):
                # Find the closest touch
                closest_touch_idx = np.argmin(time_diffs)
                closest_touch_time_abet = touch_times[closest_touch_idx]
                closest_touch_delay = closest_touch_time_abet - approach_relative_time
                
                event['classification'] = 'touch'
                event['touch_delay'] = closest_touch_delay
                event['closest_touch_time_abet'] = closest_touch_time_abet
                event['approach_relative_time'] = approach_relative_time
                touch_events += 1
                
                # Debug output for first few events
                if i < 5:
                    print(f"  Event {i+1}: {event['time_sec']:.3f}s (video) -> {approach_relative_time:.3f}s (relative) -> TOUCH at {closest_touch_time_abet:.3f}s (ABET), delay: {closest_touch_delay:+.3f}s")
            else:
                event['classification'] = 'abort'
                event['touch_delay'] = None
                event['closest_touch_time_abet'] = None
                event['approach_relative_time'] = approach_relative_time
                abort_events += 1
                
                # Debug output for first few events
                if i < 5:
                    closest_touch_time = touch_times[np.argmin(np.abs(touch_times - approach_relative_time))]
                    closest_diff = closest_touch_time - approach_relative_time
                    print(f"  Event {i+1}: {event['time_sec']:.3f}s (video) -> {approach_relative_time:.3f}s (relative) -> ABORT (closest touch: {closest_diff:+.3f}s away)")
        
        print(f"\nClassification results:")
        print(f"  Touch events: {touch_events}")
        print(f"  Abort events: {abort_events}")
        print(f"  Touch rate: {100*touch_events/(touch_events+abort_events):.1f}%")
        
        # Additional diagnostic: show overlap analysis
        if len(touch_times) > 0 and len(approach_relative_times) > 0:
            print(f"\nTiming analysis (both in relative timeline):")
            print(f"  ABET touch times range: {touch_times.min():.3f}s to {touch_times.max():.3f}s")
            print(f"  Approach events range: {min(approach_relative_times):.3f}s to {max(approach_relative_times):.3f}s")
            
            # Check overlap
            touch_span = (touch_times.min(), touch_times.max())
            event_span = (min(approach_relative_times), max(approach_relative_times))
            overlap_start = max(touch_span[0], event_span[0])
            overlap_end = min(touch_span[1], event_span[1])
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                total_span = max(touch_span[1], event_span[1]) - min(touch_span[0], event_span[0])
                print(f"  Timeline overlap: {overlap_duration:.1f}s ({100*overlap_duration/total_span:.1f}% of total timespan)")
            else:
                print(f"  WARNING: No timeline overlap detected!")
                print(f"    Approach events: {event_span[0]:.1f}s to {event_span[1]:.1f}s")
                print(f"    ABET touches: {touch_span[0]:.1f}s to {touch_span[1]:.1f}s")
                print(f"    Check houselight detection accuracy!")
            
            # Show some example matches
            print(f"\nExample matches within {touch_window}s:")
            match_count = 0
            for i, event in enumerate(self.approach_events[:10]):  # Check first 10 events
                approach_relative_time = event['time_sec'] - self.houselight_onset_time
                time_diffs = np.abs(touch_times - approach_relative_time)
                min_diff = np.min(time_diffs)
                if min_diff <= touch_window:
                    closest_touch_time = touch_times[np.argmin(time_diffs)]
                    print(f"    Event {i+1}: {approach_relative_time:.3f}s -> Touch at {closest_touch_time:.3f}s (diff: {closest_touch_time - approach_relative_time:+.3f}s)")
                    match_count += 1
            
            if match_count == 0:
                print(f"    No matches found in first 10 events within {touch_window}s window")
                print(f"    Consider increasing touch_window or checking houselight detection")
    
    def debug_event_timing(self, event_index=0):
        """
        Debug timing alignment for a specific event
        """
        if not self.approach_events or event_index >= len(self.approach_events):
            print("No events available or invalid event index")
            return
        
        event = self.approach_events[event_index]
        
        print(f"\n=== Debug Event {event_index + 1} Timing ===")
        print(f"Frame: {event['frame']}")
        print(f"Video time: {event['time_sec']:.3f}s")
        print(f"Video FPS: {self.video_fps}")
        print(f"Houselight onset: {self.houselight_onset_time:.3f}s")
        
        # Calculate what the behavioral timestamp would be
        behavioral_time = event['time_sec'] + self.houselight_onset_time
        print(f"Corresponding behavioral time: {behavioral_time:.3f}s")
        
        # Load touch data to check nearby touches
        touch_times = self.load_touchscreen_data()
        if touch_times is not None:
            # Find touches near this event
            time_diffs = np.abs(touch_times - behavioral_time)
            closest_touches = np.where(time_diffs <= 2.0)[0]  # Within 2 seconds
            
            print(f"\nNearby touches in behavioral timeline:")
            for touch_idx in closest_touches[:5]:  # Show up to 5 closest
                touch_time = touch_times[touch_idx]
                diff = touch_time - behavioral_time
                video_touch_time = touch_time - self.houselight_onset_time
                print(f"  Touch at {touch_time:.3f}s (video: {video_touch_time:.3f}s, diff: {diff:+.3f}s)")
        
        print(f"\nEvent classification: {event.get('classification', 'unknown')}")
        if event.get('touch_delay') is not None:
            print(f"Touch delay: {event['touch_delay']:+.3f}s")
        
        return behavioral_time
    
    def save_events(self, output_path=None):
        """Save approach events to CSV"""
        if not self.approach_events:
            print("No events to save")
            return
        
        if output_path is None:
            output_path = os.path.join(self.base_path, 'approach_events.csv')
        
        df = pd.DataFrame(self.approach_events)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(self.approach_events)} events to {output_path}")
        return output_path
    
    def create_event_screenshots(self, num_screenshots=20):
        """
        Create screenshots showing body orientation and screen determination
        """
        if not self.approach_events or not self.video_path:
            print("No events or video available for screenshots")
            return
        
        # Select events evenly distributed
        events_to_screenshot = []
        if len(self.approach_events) <= num_screenshots:
            events_to_screenshot = self.approach_events.copy()
        else:
            indices = np.linspace(0, len(self.approach_events)-1, num_screenshots, dtype=int)
            events_to_screenshot = [self.approach_events[i] for i in indices]
        
        output_folder = os.path.join(self.base_path, 'event_screenshots')
        os.makedirs(output_folder, exist_ok=True)
        
        # Open video once for efficiency
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Cannot open video for screenshots")
            return
        
        # Sort events by frame number for efficient seeking
        sorted_events = sorted(enumerate(events_to_screenshot), key=lambda x: x[1]['frame'])
        
        created = 0
        
        for original_index, event in sorted_events:
            frame_number = event['frame']
            
            # Get frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Get event data
            head_x = event.get('head_x', np.nan)
            head_y = event.get('head_y', np.nan)
            body_orientation = event.get('body_orientation', np.nan)
            
            if not (pd.isna(head_x) or pd.isna(head_y)):
                # Draw screen lines with clear labels
                left_line = self.touchscreen_lines['left']
                right_line = self.touchscreen_lines['right']
                
                cv2.line(frame, 
                        (int(left_line['start'][0]), int(left_line['start'][1])),
                        (int(left_line['end'][0]), int(left_line['end'][1])),
                        (0, 0, 255), 6)  # Red for left screen
                
                cv2.line(frame, 
                        (int(right_line['start'][0]), int(right_line['start'][1])),
                        (int(right_line['end'][0]), int(right_line['end'][1])),
                        (255, 0, 0), 6)  # Blue for right screen
                
                # Label screens
                left_center_x = int((left_line['start'][0] + left_line['end'][0]) / 2)
                left_center_y = int((left_line['start'][1] + left_line['end'][1]) / 2)
                right_center_x = int((right_line['start'][0] + right_line['end'][0]) / 2)
                right_center_y = int((right_line['start'][1] + right_line['end'][1]) / 2)
                
                cv2.putText(frame, "LEFT", (left_center_x-30, left_center_y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "RIGHT", (right_center_x-40, right_center_y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                
                # Draw head position
                cv2.circle(frame, (int(head_x), int(head_y)), 12, (0, 255, 0), -1)  # Green head
                cv2.circle(frame, (int(head_x), int(head_y)), 12, (0, 0, 0), 2)      # Black border
                
                # Draw body orientation arrow if available
                if not pd.isna(body_orientation):
                    arrow_length = 100
                    end_x = int(head_x + arrow_length * np.cos(np.radians(body_orientation)))
                    end_y = int(head_y + arrow_length * np.sin(np.radians(body_orientation)))
                    
                    # Draw thick orientation arrow
                    cv2.arrowedLine(frame, (int(head_x), int(head_y)), (end_x, end_y), 
                                   (255, 255, 0), 4, tipLength=0.1)  # Yellow arrow
                
                # Add comprehensive text annotations
                classification = event.get('classification', 'unknown')
                velocity = event.get('velocity', 0)
                screen = event.get('screen', 'unknown')
                confidence = event.get('screen_confidence', 0)
                
                text_lines = [
                    f"Event {original_index+1}: Frame {frame_number}",
                    f"Classification: {classification.upper()}",
                    f"Target Screen: {screen.upper()}",
                    f"Body Orientation: {body_orientation:.1f}°" if not pd.isna(body_orientation) else "Body Orientation: N/A",
                    f"Screen Confidence: {confidence:.2f}",
                    f"Velocity: {velocity:.1f} px/s"
                ]
                
                y_offset = 30
                for line in text_lines:
                    # Add background for readability
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (5, y_offset-25), (text_size[0]+15, y_offset+5), (0, 0, 0), -1)
                    
                    # Add text
                    cv2.putText(frame, line, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    y_offset += 35
            
            # Save screenshot
            filename = f"event_{original_index+1:03d}_frame_{frame_number}_{screen}_{classification}.png"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            created += 1
            
            if created % 10 == 0:
                print(f"Created {created}/{len(events_to_screenshot)} screenshots")
        
        cap.release()
        print(f"Created {created} event screenshots in {output_folder}")
        return output_folder
    
    def plot_orientation_analysis(self):
        """Plot body orientation and screen confidence over time"""
        if 'body_orientation' not in self.combined_data.columns:
            print("No body orientation data available")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot body orientation over time
        valid_mask = ~pd.isna(self.combined_data['body_orientation'])
        valid_times = self.combined_data.loc[valid_mask, 'idx_time']
        valid_orientations = self.combined_data.loc[valid_mask, 'body_orientation']
        
        ax1.plot(valid_times, valid_orientations, alpha=0.7, linewidth=1)
        ax1.set_title('Mouse Body Orientation Over Time')
        ax1.set_ylabel('Orientation (degrees)')
        ax1.set_ylim(0, 360)
        
        # Add horizontal lines for screen directions if we have events
        if self.approach_events:
            left_events = [e for e in self.approach_events if e.get('screen') == 'left']
            right_events = [e for e in self.approach_events if e.get('screen') == 'right']
            
            if left_events:
                left_orientations = [e.get('body_orientation', np.nan) for e in left_events]
                left_orientations = [o for o in left_orientations if not pd.isna(o)]
                if left_orientations:
                    ax1.axhline(y=np.mean(left_orientations), color='red', linestyle='--', 
                              label=f'Avg Left Screen: {np.mean(left_orientations):.1f}°')
            
            if right_events:
                right_orientations = [e.get('body_orientation', np.nan) for e in right_events]
                right_orientations = [o for o in right_orientations if not pd.isna(o)]
                if right_orientations:
                    ax1.axhline(y=np.mean(right_orientations), color='blue', linestyle='--',
                              label=f'Avg Right Screen: {np.mean(right_orientations):.1f}°')
            
            ax1.legend()
        
        # Plot head tracking quality
        ax2.plot(self.combined_data['idx_time'], self.combined_data.get('head_quality', 0))
        ax2.set_title('Head Tracking Quality Over Time')
        ax2.set_ylabel('Number of Head Parts Tracked')
        ax2.set_ylim(0, max(self.combined_data.get('head_quality', [1]).max(), 3))
        
        # Mark approach events
        if self.approach_events:
            event_frames = [e['frame'] for e in self.approach_events]
            event_qualities = [e.get('head_quality', 0) for e in self.approach_events]
            ax2.scatter(event_frames, event_qualities, color='red', s=30, alpha=0.7, 
                       label=f'Approach Events (n={len(self.approach_events)})')
            ax2.legend()
        
        # Plot screen confidence histogram
        if self.approach_events:
            confidences = [e.get('screen_confidence', 0) for e in self.approach_events]
            ax3.hist(confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.set_title('Screen Detection Confidence Distribution')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Number of Events')
            ax3.axvline(x=np.mean(confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidences):.2f}')
            ax3.legend()
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self, num_clips=20, proximity_threshold=60, velocity_threshold=15, suppression_frames=10):
        """Run complete approach-abort analysis using body orientation method"""
        print("=== Body Orientation Approach-Abort Analysis ===")
        print("Using mouse body skeleton for accurate screen determination")
        
        # Load existing data if available
        combined_loaded, events_loaded = self.load_existing_data()
        
        if not combined_loaded:
            print("\n1. Combining SLEAP data...")
            self.combine_csv_data()
        
        if not events_loaded:
            # Find video
            print("\n2. Finding video file...")
            self.find_video_file()
            
            # Annotate screens
            print("\n3. Annotating touchscreen positions...")
            self.annotate_touchscreens()
            
            # Detect approach events using body orientation with suppression
            print("\n4. Detecting approach events using body orientation...")
            self.detect_approach_events(
                proximity_threshold=proximity_threshold,
                velocity_threshold=velocity_threshold,
                suppression_frames=suppression_frames
            )
            
            # Show orientation analysis plots
            print("\n5. Analyzing body orientation patterns...")
            self.plot_orientation_analysis()
            
            # Detect houselight for timestamp alignment
            print("\n6. Detecting houselight onset...")
            houselight_time = self.detect_houselight_onset()
            
            if houselight_time is not None:
                # Load and classify with touchscreen data
                print("\n7. Loading touchscreen data...")
                touch_times = self.load_touchscreen_data()
                
                if touch_times is not None:
                    print("\n8. Classifying events...")
                    self.classify_events(touch_times)
            
            # Save events
            print("\n9. Saving events...")
            self.save_events()
        
        # Create screenshots
        if self.approach_events:
            print(f"\n10. Creating {num_clips} event screenshots...")
            self.create_event_screenshots(num_screenshots=num_clips)
            
            # Print comprehensive summary
            touch_count = sum(1 for e in self.approach_events if e.get('classification') == 'touch')
            abort_count = sum(1 for e in self.approach_events if e.get('classification') == 'abort')
            
            # Screen distribution
            left_events = sum(1 for e in self.approach_events if e.get('screen') == 'left')
            right_events = sum(1 for e in self.approach_events if e.get('screen') == 'right')
            
            # Quality statistics
            avg_head_quality = np.mean([e.get('head_quality', 0) for e in self.approach_events])
            avg_confidence = np.mean([e.get('screen_confidence', 0) for e in self.approach_events])
            
            # Orientation statistics
            orientations = [e.get('body_orientation', np.nan) for e in self.approach_events]
            valid_orientations = [o for o in orientations if not pd.isna(o)]
            
            # Event spacing analysis
            if len(self.approach_events) > 1:
                frame_gaps = []
                for i in range(1, len(self.approach_events)):
                    gap = self.approach_events[i]['frame'] - self.approach_events[i-1]['frame']
                    frame_gaps.append(gap)
                
                if frame_gaps:
                    min_gap_frames = min(frame_gaps)
                    avg_gap_frames = np.mean(frame_gaps)
                    min_gap_seconds = min_gap_frames / self.video_fps
                    avg_gap_seconds = avg_gap_frames / self.video_fps
            
            print(f"\n=== Body Orientation Analysis Summary ===")
            print(f"Total approach events: {len(self.approach_events)}")
            print(f"Touch events: {touch_count}")
            print(f"Abort events: {abort_count}")
            print(f"Screen distribution: Left={left_events}, Right={right_events}")
            print(f"Average head tracking quality: {avg_head_quality:.2f}")
            print(f"Average screen confidence: {avg_confidence:.2f}")
            print(f"Suppression window used: {suppression_frames} frames ({suppression_frames/self.video_fps:.2f} seconds)")
            
            if len(self.approach_events) > 1 and frame_gaps:
                print(f"Event spacing: min={min_gap_frames} frames ({min_gap_seconds:.2f}s), avg={avg_gap_frames:.1f} frames ({avg_gap_seconds:.2f}s)")
                close_events = sum(1 for gap in frame_gaps if gap < suppression_frames * 2)
                print(f"Events closer than {suppression_frames * 2} frames: {close_events} (suppression working properly)")
            
            if valid_orientations:
                print(f"Body orientation range: {min(valid_orientations):.1f}° - {max(valid_orientations):.1f}°")
                print(f"Valid orientations: {len(valid_orientations)}/{len(self.approach_events)} ({100*len(valid_orientations)/len(self.approach_events):.1f}%)")
            
            # Show velocity distribution
            velocities = [e['velocity'] for e in self.approach_events]
            print(f"Velocity range: {min(velocities):.1f} - {max(velocities):.1f} pixels/sec")
            
            print(f"\n*** Check screenshots in 'event_screenshots' folder to verify classifications! ***")


# Example usage with suppression
if __name__ == "__main__":
    base_path = r"C:\Users\Patrick\Desktop\RRD414\RDT D1 CNO\RDT D1 CNO"
    
    detector = ApproachAbortDetector(base_path)
    
    try:
        # Run with body orientation method and event suppression
        detector.run_analysis(
            num_clips=20,
            proximity_threshold=60,      # Distance threshold to screens
            velocity_threshold=15,       # Movement speed threshold
            suppression_frames=30        # Minimum frames between events (prevents duplicates)
        )
        
        print("\nAnalysis complete with event suppression!")
        print("Multiple events within 10 frames have been prevented.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()