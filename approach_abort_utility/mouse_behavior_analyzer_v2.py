import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
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
    
    def get_head_coordinates(self):
        """
        Create robust head position estimates using interpolation of snout and ears
        Returns arrays of interpolated head x,y coordinates
        """
        if self.combined_data is None:
            raise ValueError("No combined data available")
        
        # Find available head part coordinates
        head_parts = {}
        possible_names = {
            'snout': ['snout', 'nose', 'tip'],
            'left_ear': ['left_ear', 'leftear', 'ear_left', 'left-ear'],
            'right_ear': ['right_ear', 'rightear', 'ear_right', 'right-ear']
        }
        
        print("Looking for head part coordinates...")
        
        for part_type, possible_part_names in possible_names.items():
            for name in possible_part_names:
                x_col = f'{name}_x_pix'
                y_col = f'{name}_y_pix'
                if x_col in self.combined_data.columns and y_col in self.combined_data.columns:
                    head_parts[part_type] = {'x': x_col, 'y': y_col}
                    print(f"Found {part_type}: {name}")
                    break
        
        if not head_parts:
            raise ValueError("No head part coordinates found in data")
        
        # Create interpolated head position
        head_x = np.full(len(self.combined_data), np.nan)
        head_y = np.full(len(self.combined_data), np.nan)
        tracking_quality = np.zeros(len(self.combined_data))  # Track how many parts contributed
        
        for i in range(len(self.combined_data)):
            valid_x = []
            valid_y = []
            weights = []
            
            # Collect valid coordinates from available parts
            for part_type, cols in head_parts.items():
                x_val = self.combined_data.iloc[i][cols['x']]
                y_val = self.combined_data.iloc[i][cols['y']]
                
                if not (pd.isna(x_val) or pd.isna(y_val)):
                    valid_x.append(x_val)
                    valid_y.append(y_val)
                    
                    # Weight snout more heavily as it's typically most accurate for approach direction
                    if part_type == 'snout':
                        weights.append(2.0)
                    else:
                        weights.append(1.0)
            
            # Calculate weighted average if we have valid coordinates
            if valid_x:
                weights = np.array(weights)
                head_x[i] = np.average(valid_x, weights=weights)
                head_y[i] = np.average(valid_y, weights=weights)
                tracking_quality[i] = len(valid_x)
        
        # Fill gaps using interpolation for short missing segments
        self._interpolate_missing_segments(head_x, max_gap=10)
        self._interpolate_missing_segments(head_y, max_gap=10)
        
        # Add to combined data for reference
        self.combined_data['head_x_interpolated'] = head_x
        self.combined_data['head_y_interpolated'] = head_y
        self.combined_data['head_tracking_quality'] = tracking_quality
        
        valid_points = np.sum(~np.isnan(head_x))
        total_points = len(head_x)
        
        print(f"Head position interpolation complete:")
        print(f"  Valid head positions: {valid_points}/{total_points} ({100*valid_points/total_points:.1f}%)")
        print(f"  Used parts: {list(head_parts.keys())}")
        
        return head_x, head_y, tracking_quality
    
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
    
    def distance_to_line(self, point, line_start, line_end):
        """Calculate distance from point to line"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Distance from point to line formula
        num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        den = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        return num / den if den > 0 else float('inf')
    
    def calculate_heading_angle(self, snout_pos, left_ear_pos, right_ear_pos):
        """
        Calculate mouse heading angle using snout and ear positions
        Returns angle in degrees (0° = facing right, 90° = facing up, etc.)
        """
        snout_x, snout_y = snout_pos
        left_ear_x, left_ear_y = left_ear_pos
        right_ear_x, right_ear_y = right_ear_pos
        
        # Calculate midpoint between ears (back of head)
        ear_mid_x = (left_ear_x + right_ear_x) / 2
        ear_mid_y = (left_ear_y + right_ear_y) / 2
        
        # Vector from ear midpoint to snout (head direction)
        head_vector_x = snout_x - ear_mid_x
        head_vector_y = snout_y - ear_mid_y
        
        # Calculate angle in degrees
        angle_rad = np.arctan2(head_vector_y, head_vector_x)
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to 0-360 degrees
        if angle_deg < 0:
            angle_deg += 360
            
        return angle_deg
    
    def get_screen_angles(self):
        """
        Calculate the angle from center of arena to each screen
        This helps determine which screen the mouse is facing
        """
        # Get screen centers
        left_center_x = (self.touchscreen_lines['left']['start'][0] + 
                        self.touchscreen_lines['left']['end'][0]) / 2
        left_center_y = (self.touchscreen_lines['left']['start'][1] + 
                        self.touchscreen_lines['left']['end'][1]) / 2
        
        right_center_x = (self.touchscreen_lines['right']['start'][0] + 
                         self.touchscreen_lines['right']['end'][0]) / 2
        right_center_y = (self.touchscreen_lines['right']['start'][1] + 
                         self.touchscreen_lines['right']['end'][1]) / 2
        
        # Estimate arena center (midpoint between screens)
        arena_center_x = (left_center_x + right_center_x) / 2
        arena_center_y = (left_center_y + right_center_y) / 2
        
        # Calculate angles to each screen from arena center
        left_angle = np.degrees(np.arctan2(left_center_y - arena_center_y, 
                                          left_center_x - arena_center_x))
        right_angle = np.degrees(np.arctan2(right_center_y - arena_center_y, 
                                           right_center_x - arena_center_x))
        
        # Normalize to 0-360
        if left_angle < 0:
            left_angle += 360
        if right_angle < 0:
            right_angle += 360
            
        return left_angle, right_angle, (arena_center_x, arena_center_y)
    
    def debug_screen_setup(self):
        """
        Debug function to visualize screen setup and angles
        Call this after annotating screens to verify the setup
        """
        if not self.touchscreen_lines['left'] or not self.touchscreen_lines['right']:
            print("Screens not annotated yet")
            return
            
        # Get random frame for visualization
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame = random.randint(1000, min(2000, total_frames-1))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Could not load frame for debugging")
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Draw screen lines
        left_line = self.touchscreen_lines['left']
        right_line = self.touchscreen_lines['right']
        
        ax.plot([left_line['start'][0], left_line['end'][0]], 
               [left_line['start'][1], left_line['end'][1]], 
               'r-', linewidth=4, label='Left Screen')
        ax.plot([right_line['start'][0], right_line['end'][0]], 
               [right_line['start'][1], right_line['end'][1]], 
               'b-', linewidth=4, label='Right Screen')
        
        # Calculate and show centers and angles
        left_angle, right_angle, arena_center = self.get_screen_angles()
        
        left_center_x = (left_line['start'][0] + left_line['end'][0]) / 2
        left_center_y = (left_line['start'][1] + left_line['end'][1]) / 2
        right_center_x = (right_line['start'][0] + right_line['end'][0]) / 2
        right_center_y = (right_line['start'][1] + right_line['end'][1]) / 2
        
        # Mark centers
        ax.plot(left_center_x, left_center_y, 'ro', markersize=12, label=f'Left Center ({left_angle:.1f}°)')
        ax.plot(right_center_x, right_center_y, 'bo', markersize=12, label=f'Right Center ({right_angle:.1f}°)')
        ax.plot(arena_center[0], arena_center[1], 'go', markersize=15, label='Arena Center')
        
        # Draw direction arrows from arena center to screen centers
        arrow_length = 100
        left_arrow_end_x = arena_center[0] + arrow_length * np.cos(np.radians(left_angle))
        left_arrow_end_y = arena_center[1] + arrow_length * np.sin(np.radians(left_angle))
        right_arrow_end_x = arena_center[0] + arrow_length * np.cos(np.radians(right_angle))
        right_arrow_end_y = arena_center[1] + arrow_length * np.sin(np.radians(right_angle))
        
        ax.annotate('', xy=(left_arrow_end_x, left_arrow_end_y), 
                   xytext=(arena_center[0], arena_center[1]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=3))
        ax.annotate('', xy=(right_arrow_end_x, right_arrow_end_y), 
                   xytext=(arena_center[0], arena_center[1]),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=3))
        
        ax.legend()
        ax.set_title(f"Screen Setup Debug\nLeft Screen: {left_angle:.1f}° | Right Screen: {right_angle:.1f}°")
        
        plt.show()
        
        print(f"Screen Setup Summary:")
        print(f"  Left screen angle: {left_angle:.1f}°")
        print(f"  Right screen angle: {right_angle:.1f}°")
        print(f"  Arena center: ({arena_center[0]:.1f}, {arena_center[1]:.1f})")
        print(f"  Angular separation: {abs(left_angle - right_angle):.1f}°")
        
        return left_angle, right_angle, arena_center
    
    def debug_heading_calculation(self, frame_number=None):
        """
        Debug heading calculation for a specific frame
        """
        if self.combined_data is None:
            print("No combined data available")
            return
            
        if frame_number is None:
            frame_number = len(self.combined_data) // 2  # Middle frame
            
        # Find head part coordinates
        head_parts = {}
        possible_names = {
            'snout': ['snout', 'nose', 'tip'],
            'left_ear': ['left_ear', 'leftear', 'ear_left', 'left-ear'],
            'right_ear': ['right_ear', 'rightear', 'ear_right', 'right-ear']
        }
        
        for part_type, possible_part_names in possible_names.items():
            for name in possible_part_names:
                x_col = f'{name}_x_pix'
                y_col = f'{name}_y_pix'
                if x_col in self.combined_data.columns and y_col in self.combined_data.columns:
                    head_parts[part_type] = {'x': x_col, 'y': y_col}
                    break
        
        if len(head_parts) < 3:
            print(f"Not enough head parts for heading calculation. Found: {list(head_parts.keys())}")
            return
            
        # Get coordinates
        snout_x = self.combined_data.iloc[frame_number][head_parts['snout']['x']]
        snout_y = self.combined_data.iloc[frame_number][head_parts['snout']['y']]
        left_ear_x = self.combined_data.iloc[frame_number][head_parts['left_ear']['x']]
        left_ear_y = self.combined_data.iloc[frame_number][head_parts['left_ear']['y']]
        right_ear_x = self.combined_data.iloc[frame_number][head_parts['right_ear']['x']]
        right_ear_y = self.combined_data.iloc[frame_number][head_parts['right_ear']['y']]
        
        if any(pd.isna(x) for x in [snout_x, snout_y, left_ear_x, left_ear_y, right_ear_x, right_ear_y]):
            print(f"Missing coordinates at frame {frame_number}")
            return
            
        # Calculate heading
        heading_angle = self.calculate_heading_angle(
            (snout_x, snout_y), 
            (left_ear_x, left_ear_y), 
            (right_ear_x, right_ear_y)
        )
        
        # Get screen angles
        left_angle, right_angle, arena_center = self.get_screen_angles()
        
        # Determine target
        target_screen, angular_diff = self.determine_target_screen(
            heading_angle, (left_angle, right_angle), tolerance_deg=45
        )
        
        print(f"Frame {frame_number} Heading Analysis:")
        print(f"  Snout: ({snout_x:.1f}, {snout_y:.1f})")
        print(f"  Left ear: ({left_ear_x:.1f}, {left_ear_y:.1f})")
        print(f"  Right ear: ({right_ear_x:.1f}, {right_ear_y:.1f})")
        print(f"  Mouse heading: {heading_angle:.1f}°")
        print(f"  Left screen angle: {left_angle:.1f}°")
        print(f"  Right screen angle: {right_angle:.1f}°")
        print(f"  Angular diff to target: {angular_diff:.1f}°")
        print(f"  Target screen: {target_screen}")
        
        return heading_angle, target_screen
    
    def determine_target_screen(self, mouse_heading, screen_angles, tolerance_deg=45):
        """
        Determine which screen the mouse is facing based on heading angle
        
        Args:
            mouse_heading: Mouse head angle in degrees
            screen_angles: Tuple of (left_angle, right_angle) from get_screen_angles()
            tolerance_deg: Angular tolerance for screen targeting
        """
        left_angle, right_angle = screen_angles
        
        # Calculate angular difference to each screen
        def angular_diff(angle1, angle2):
            diff = abs(angle1 - angle2)
            return min(diff, 360 - diff)  # Handle wrap-around
        
        left_diff = angular_diff(mouse_heading, left_angle)
        right_diff = angular_diff(mouse_heading, right_angle)
        
        # Determine target screen
        if left_diff <= tolerance_deg and left_diff < right_diff:
            return 'left', left_diff
        elif right_diff <= tolerance_deg:
            return 'right', right_diff
        else:
            return None, min(left_diff, right_diff)  # Not facing either screen clearly
    
    def detect_approach_events(self, proximity_threshold=50, velocity_threshold=20, 
                             time_window_frames=30, min_tracking_quality=1, heading_tolerance=45):
        """
        Detect approach events using interpolated head position AND heading direction
        
        Args:
            proximity_threshold: Max distance to screen to consider "approach"
            velocity_threshold: Min velocity change to consider "rapid movement"
            time_window_frames: Frames to analyze for velocity change (~1 sec at 30fps)
            min_tracking_quality: Minimum number of head parts needed for reliable tracking
            heading_tolerance: Angular tolerance (degrees) for determining screen target
        """
        if self.combined_data is None:
            raise ValueError("No combined data available")
        
        # Get interpolated head coordinates
        print("Creating interpolated head position...")
        head_x, head_y, tracking_quality = self.get_head_coordinates()
        
        # Find individual head part coordinates for heading calculation
        head_parts = {}
        possible_names = {
            'snout': ['snout', 'nose', 'tip'],
            'left_ear': ['left_ear', 'leftear', 'ear_left', 'left-ear'],
            'right_ear': ['right_ear', 'rightear', 'ear_right', 'right-ear']
        }
        
        for part_type, possible_part_names in possible_names.items():
            for name in possible_part_names:
                x_col = f'{name}_x_pix'
                y_col = f'{name}_y_pix'
                if x_col in self.combined_data.columns and y_col in self.combined_data.columns:
                    head_parts[part_type] = {'x': x_col, 'y': y_col}
                    break
        
        # Check if we have enough parts for heading calculation
        has_snout = 'snout' in head_parts
        has_both_ears = 'left_ear' in head_parts and 'right_ear' in head_parts
        
        if not (has_snout and has_both_ears):
            print("Warning: Cannot calculate heading - need snout + both ears")
            print(f"Available parts: {list(head_parts.keys())}")
            print("Falling back to distance-based screen detection...")
            use_heading = False
        else:
            print("Using heading-based screen detection (snout + ears)")
            use_heading = True
            # Get screen angles for reference
            screen_angles = self.get_screen_angles()
            print(f"Screen angles: Left={screen_angles[0]:.1f}°, Right={screen_angles[1]:.1f}°")
        
        events = []
        
        print(f"Analyzing {len(self.combined_data)} frames for approach events...")
        
        for i in range(time_window_frames, len(self.combined_data) - time_window_frames):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(self.combined_data)} frames")
            
            # Skip if head position is not reliable enough
            if tracking_quality[i] < min_tracking_quality:
                continue
            
            # Get current head position
            head_x_curr = head_x[i]
            head_y_curr = head_y[i]
            
            # Skip if coordinates are still NaN after interpolation
            if pd.isna(head_x_curr) or pd.isna(head_y_curr):
                continue
            
            # Determine target screen using heading or distance
            target_screen = None
            screen_confidence = 0
            heading_angle = None
            
            if use_heading:
                # Get individual part positions for heading calculation
                snout_x = self.combined_data.iloc[i][head_parts['snout']['x']]
                snout_y = self.combined_data.iloc[i][head_parts['snout']['y']]
                left_ear_x = self.combined_data.iloc[i][head_parts['left_ear']['x']]
                left_ear_y = self.combined_data.iloc[i][head_parts['left_ear']['y']]
                right_ear_x = self.combined_data.iloc[i][head_parts['right_ear']['x']]
                right_ear_y = self.combined_data.iloc[i][head_parts['right_ear']['y']]
                
                # Check if all parts are available for heading calculation
                parts_available = not any(pd.isna(x) for x in [snout_x, snout_y, left_ear_x, 
                                                              left_ear_y, right_ear_x, right_ear_y])
                
                if parts_available:
                    # Calculate heading angle
                    heading_angle = self.calculate_heading_angle(
                        (snout_x, snout_y), 
                        (left_ear_x, left_ear_y), 
                        (right_ear_x, right_ear_y)
                    )
                    
                    # Determine target screen based on heading
                    target_screen, angular_diff = self.determine_target_screen(
                        heading_angle, screen_angles[:2], heading_tolerance
                    )
                    
                    if target_screen:
                        screen_confidence = max(0, (heading_tolerance - angular_diff) / heading_tolerance)
                    
            # Fall back to distance-based method if heading failed
            if target_screen is None:
                # Calculate distance to each screen (original method)
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
                
                # Check if close to either screen
                if left_dist < proximity_threshold and left_dist < right_dist:
                    target_screen = 'left'
                    screen_confidence = max(0, (proximity_threshold - left_dist) / proximity_threshold)
                elif right_dist < proximity_threshold:
                    target_screen = 'right'  
                    screen_confidence = max(0, (proximity_threshold - right_dist) / proximity_threshold)
            
            # If we have a target screen, check for rapid movement
            if target_screen is not None:
                # Calculate velocity over next time_window_frames
                future_positions = []
                quality_scores = []
                
                for j in range(i, min(i + time_window_frames, len(self.combined_data))):
                    fx = head_x[j]
                    fy = head_y[j]
                    quality = tracking_quality[j]
                    
                    if not (pd.isna(fx) or pd.isna(fy)) and quality >= min_tracking_quality:
                        future_positions.append((fx, fy))
                        quality_scores.append(quality)
                
                if len(future_positions) >= time_window_frames // 2:  # Need at least half the frames
                    # Calculate movement distance
                    start_pos = (head_x_curr, head_y_curr)
                    end_pos = future_positions[-1]
                    movement_distance = euclidean(start_pos, end_pos)
                    
                    # Calculate average velocity
                    velocity = movement_distance / (len(future_positions) / self.video_fps)
                    
                    # Calculate average tracking quality for this event
                    avg_quality = np.mean([tracking_quality[i]] + quality_scores)
                    
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
                            'tracking_quality': avg_quality,
                            'heading_angle': heading_angle,
                            'screen_confidence': screen_confidence,
                            'detection_method': 'heading' if use_heading and heading_angle is not None else 'distance',
                            'interpolated_head': True
                        }
                        
                        # Avoid duplicate events too close together
                        if not any(abs(e['frame'] - event['frame']) < time_window_frames for e in events):
                            events.append(event)
        
        self.approach_events = events
        method_counts = {}
        for event in events:
            method = event.get('detection_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"Detected {len(events)} approach events using heading-based screen detection")
        print(f"Detection methods used: {method_counts}")
        return events
    
    def detect_houselight_onset(self, search_frames=2000):
        """Detect houselight onset for timestamp alignment"""
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
        
        # Find largest brightness change
        brightness_changes = np.diff(brightness_values)
        max_change_idx = np.argmax(brightness_changes)
        max_change_frame = max_change_idx + 1
        houselight_time = self.frame_to_timestamp(max_change_frame)
        
        # Show plot
        plt.figure(figsize=(12, 4))
        plt.plot(range(len(brightness_values)), brightness_values)
        plt.axvline(x=max_change_frame, color='red', linestyle='--', 
                   label=f'Max change at frame {max_change_frame}')
        plt.xlabel('Frame Number')
        plt.ylabel('Brightness')
        plt.title('Video Brightness Over Time')
        plt.legend()
        plt.show()
        
        response = input(f"Houselight onset detected at {houselight_time:.2f}s. Confirm? (y/n): ")
        
        if response.lower() == 'y':
            self.houselight_onset_time = houselight_time
            return houselight_time
        else:
            return None
    
    def load_touchscreen_data(self, touchscreen_file=None):
        """Load touchscreen data"""
        if touchscreen_file is None:
            # Try both possible locations
            possible_files = [
                os.path.join(self.base_path, "D2-9 10232019.csv"),
                os.path.join(os.path.dirname(self.base_path), "D2-9 10232019.csv")
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
    
    def classify_events(self, touch_times, touch_window=0.5):
        """Classify approach events as touch vs abort"""
        if touch_times is None or self.houselight_onset_time is None:
            print("Cannot classify events without touch data and houselight onset")
            return
        
        # Align touch times with video timeline
        aligned_touch_times = touch_times - self.houselight_onset_time
        valid_touches = aligned_touch_times[aligned_touch_times >= 0]
        
        print(f"Aligned touch times: {len(valid_touches)} touches during video")
        
        # Classify each approach event
        for event in self.approach_events:
            event_time = event['time_sec']
            
            # Look for touch within touch_window seconds AFTER the approach
            time_diffs = valid_touches - event_time
            nearby_touches = time_diffs[(time_diffs >= 0) & (time_diffs <= touch_window)]
            
            if len(nearby_touches) > 0:
                event['classification'] = 'touch'
                event['touch_delay'] = nearby_touches[0]
            else:
                event['classification'] = 'abort'
                event['touch_delay'] = None
        
        # Print classification results
        touch_events = [e for e in self.approach_events if e.get('classification') == 'touch']
        abort_events = [e for e in self.approach_events if e.get('classification') == 'abort']
        
        print(f"Classification complete:")
        print(f"  Touch events: {len(touch_events)}")
        print(f"  Abort events: {len(abort_events)}")
        print(f"  Average tracking quality: {np.mean([e.get('tracking_quality', 0) for e in self.approach_events]):.2f}")
    
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
    
    def create_clips(self, num_clips=20, clip_type='all'):
        """Create video clips of approach events"""
        if not self.approach_events or not self.video_path:
            print("No events or video available for clips")
            return
        
        # Filter events based on type
        if clip_type == 'abort':
            events_to_clip = [e for e in self.approach_events if e.get('classification') == 'abort']
        elif clip_type == 'touch':
            events_to_clip = [e for e in self.approach_events if e.get('classification') == 'touch']
        else:
            events_to_clip = self.approach_events.copy()
        
        # Sort by screen confidence first, then velocity
        events_to_clip.sort(key=lambda x: (x.get('screen_confidence', 0), x.get('velocity', 0)), reverse=True)
        events_to_clip = events_to_clip[:num_clips]
        
        output_folder = os.path.join(self.base_path, 'approach_clips')
        os.makedirs(output_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Cannot open video")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        created = 0
        
        for i, event in enumerate(events_to_clip):
            frame = event['frame']
            classification = event.get('classification', 'unknown')
            screen = event['screen']
            quality = event.get('tracking_quality', 0)
            confidence = event.get('screen_confidence', 0)
            method = event.get('detection_method', 'unknown')
            heading = event.get('heading_angle', 0)
            
            start_frame = max(0, frame - 60)
            clip_frames = 120
            
            filename = f"approach_{i+1:02d}_{classification}_{screen}_{method}_conf{confidence:.2f}_frame_{frame}.avi"
            filepath = os.path.join(output_folder, filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            if not out.isOpened():
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for j in range(clip_frames):
                ret, frame_img = cap.read()
                if not ret:
                    break
                
                # Add detailed annotation
                line1 = f"{classification.upper()} - {screen} ({method})"
                line2 = f"v={event['velocity']:.1f} conf={confidence:.2f}"
                if heading is not None:
                    line3 = f"heading={heading:.1f}° q={quality:.1f}"
                else:
                    line3 = f"q={quality:.1f}"
                
                cv2.putText(frame_img, line1, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame_img, line2, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_img, line3, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                out.write(frame_img)
            
            out.release()
            created += 1
            print(f"Created clip {i+1}: {filename}")
        
        cap.release()
        print(f"Created {created} clips in {output_folder}")
    
    def plot_tracking_quality(self):
        """Plot tracking quality over time to assess interpolation effectiveness"""
        if 'head_tracking_quality' not in self.combined_data.columns:
            print("No tracking quality data available. Run detect_approach_events() first.")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Plot tracking quality over time
        plt.subplot(2, 1, 1)
        plt.plot(self.combined_data['idx_time'], self.combined_data['head_tracking_quality'])
        plt.title('Head Tracking Quality Over Time')
        plt.ylabel('Number of Parts Tracked')
        plt.ylim(0, 3.5)
        
        # Mark approach events
        if self.approach_events:
            event_frames = [e['frame'] for e in self.approach_events]
            event_qualities = [e.get('tracking_quality', 0) for e in self.approach_events]
            plt.scatter(event_frames, event_qualities, color='red', s=50, alpha=0.7, 
                       label=f'Approach Events (n={len(self.approach_events)})')
            plt.legend()
        
        # Plot head position coordinates
        plt.subplot(2, 1, 2)
        valid_mask = ~pd.isna(self.combined_data['head_x_interpolated'])
        valid_times = self.combined_data.loc[valid_mask, 'idx_time']
        valid_x = self.combined_data.loc[valid_mask, 'head_x_interpolated']
        valid_y = self.combined_data.loc[valid_mask, 'head_y_interpolated']
        
        plt.plot(valid_times, valid_x, alpha=0.7, label='Head X (interpolated)')
        plt.plot(valid_times, valid_y, alpha=0.7, label='Head Y (interpolated)')
        plt.title('Interpolated Head Position Over Time')
        plt.ylabel('Pixel Coordinates')
        plt.xlabel('Frame Number')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self, num_clips=20, proximity_threshold=50, velocity_threshold=20):
        """Run complete approach-abort analysis with enhanced head tracking"""
        print("=== Enhanced Approach-Abort Analysis ===")
        print("Using interpolated head position from snout + ears")
        
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
            
            # Debug screen setup
            print("\n4. Debugging screen setup...")
            self.debug_screen_setup()
            
            # Test heading calculation on a few frames
            print("\n5. Testing heading calculation...")
            if self.combined_data is not None:
                test_frames = [len(self.combined_data)//4, len(self.combined_data)//2, 3*len(self.combined_data)//4]
                for frame in test_frames:
                    self.debug_heading_calculation(frame)
                    print()
            
            # Ask user to confirm setup looks correct
            response = input("Does the screen setup look correct? (y/n): ")
            if response.lower() != 'y':
                print("Please re-run and re-annotate screens")
                return
            
            # Detect approach events with enhanced head tracking
            print("\n6. Detecting approach events with heading-based screen detection...")
            self.detect_approach_events(
                proximity_threshold=proximity_threshold,
                velocity_threshold=velocity_threshold
            )
            
            # Show tracking quality plot
            print("\n7. Analyzing tracking quality...")
            self.plot_tracking_quality()
            
            # Detect houselight for timestamp alignment
            print("\n8. Detecting houselight onset...")
            houselight_time = self.detect_houselight_onset()
            
            if houselight_time is not None:
                # Load and classify with touchscreen data
                print("\n9. Loading touchscreen data...")
                touch_times = self.load_touchscreen_data()
                
                if touch_times is not None:
                    print("\n10. Classifying events...")
                    self.classify_events(touch_times)
            
            # Save events
            print("\n11. Saving events...")
            self.save_events()
        
        # Create clips
        if self.approach_events:
            print(f"\n12. Creating {num_clips} video clips...")
            self.create_clips(num_clips)
            
            # Print detailed summary
            touch_count = sum(1 for e in self.approach_events if e.get('classification') == 'touch')
            abort_count = sum(1 for e in self.approach_events if e.get('classification') == 'abort')
            avg_quality = np.mean([e.get('tracking_quality', 0) for e in self.approach_events])
            avg_confidence = np.mean([e.get('screen_confidence', 0) for e in self.approach_events])
            
            # Screen distribution
            left_events = sum(1 for e in self.approach_events if e.get('screen') == 'left')
            right_events = sum(1 for e in self.approach_events if e.get('screen') == 'right')
            
            # Method distribution
            heading_events = sum(1 for e in self.approach_events if e.get('detection_method') == 'heading')
            distance_events = sum(1 for e in self.approach_events if e.get('detection_method') == 'distance')
            
            print(f"\n=== Enhanced Analysis Summary ===")
            print(f"Total approach events: {len(self.approach_events)}")
            print(f"Touch events: {touch_count}")
            print(f"Abort events: {abort_count}")
            print(f"Screen distribution: Left={left_events}, Right={right_events}")
            print(f"Detection methods: Heading={heading_events}, Distance={distance_events}")
            print(f"Average tracking quality: {avg_quality:.2f}/3.0")
            print(f"Average screen confidence: {avg_confidence:.2f}")
            
            # Show velocity and confidence distributions
            velocities = [e['velocity'] for e in self.approach_events]
            confidences = [e.get('screen_confidence', 0) for e in self.approach_events]
            print(f"Velocity range: {min(velocities):.1f} - {max(velocities):.1f} pixels/sec")
            print(f"Confidence range: {min(confidences):.2f} - {max(confidences):.2f}")
            
            # Show events with low confidence (might be misclassified)
            low_confidence_events = [e for e in self.approach_events if e.get('screen_confidence', 0) < 0.3]
            if low_confidence_events:
                print(f"\nWarning: {len(low_confidence_events)} events have low screen confidence (<0.3)")
                print("These might be misclassified and should be reviewed manually")


# Example usage with enhanced debugging
if __name__ == "__main__":
    base_path = r"C:\Users\dicbr\Desktop\D2-9\RDT D1"
    
    detector = ApproachAbortDetector(base_path)
    
    try:
        # Run with debugging enabled
        detector.run_analysis(
            num_clips=20,
            proximity_threshold=15,  # Slightly larger since head position is more accurate
            velocity_threshold=150    # Slightly lower since tracking is more stable
        )
        
        # Optional: Debug specific events if needed
        # detector.debug_heading_calculation(frame_number=12345)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()