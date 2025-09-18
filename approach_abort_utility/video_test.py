import cv2
import os
import numpy as np

def test_video_reading(video_path, start_frame=1000, num_frames=50):
    """Test basic video reading functionality"""
    print(f"Testing video reading from: {video_path}")
    
    # Test if file exists
    if not os.path.exists(video_path):
        print("ERROR: Video file does not exist!")
        return False
    
    print(f"File size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
    
    # Try to open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video file!")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Dimensions: {width}x{height}")
    
    # Test seeking to specific frame
    print(f"Testing frame seeking to frame {start_frame}...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print(f"  Requested: {start_frame}, Actual: {current_pos}")
    
    # Test reading frames
    print(f"Testing reading {num_frames} frames...")
    frames_read = 0
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  ERROR: Could not read frame {i}")
            break
        
        frames_read += 1
        if i % 10 == 0:
            print(f"  Read frame {i}: {frame.shape}")
    
    cap.release()
    print(f"Successfully read {frames_read}/{num_frames} frames")
    
    return frames_read > 0

def test_video_writing(output_path, width=640, height=480, num_frames=30, fps=30):
    """Test basic video writing functionality"""
    print(f"Testing video writing to: {output_path}")
    
    # Try different codecs
    codecs = [
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MP4V', cv2.VideoWriter_fourcc(*'MP4V')),
    ]
    
    for codec_name, fourcc in codecs:
        print(f"  Testing {codec_name} codec...")
        
        test_file = output_path.replace('.avi', f'_test_{codec_name.lower()}.avi')
        
        # Create video writer
        out = cv2.VideoWriter(test_file, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"    ERROR: Cannot create video writer with {codec_name}")
            continue
        
        # Write test frames
        frames_written = 0
        for i in range(num_frames):
            # Create a simple test frame
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add frame number text
            cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
            frames_written += 1
        
        out.release()
        
        # Check if file was created
        if os.path.exists(test_file) and os.path.getsize(test_file) > 1000:
            print(f"    SUCCESS: {codec_name} - wrote {frames_written} frames ({os.path.getsize(test_file)} bytes)")
            return codec_name, fourcc
        else:
            print(f"    FAILED: {codec_name} - no valid file created")
            if os.path.exists(test_file):
                os.remove(test_file)
    
    return None, None

def create_simple_clip(video_path, output_path, start_frame, num_frames=60):
    """Create a simple video clip without complex processing"""
    print(f"Creating simple clip: frames {start_frame} to {start_frame + num_frames}")
    
    # Test video reading first
    if not test_video_reading(video_path, start_frame, 5):
        return False
    
    # Test video writing
    working_codec, fourcc = test_video_writing(output_path)
    if not working_codec:
        print("ERROR: No working video codec found!")
        return False
    
    print(f"Using {working_codec} codec for clip creation...")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output writer
    final_output = output_path.replace('.avi', f'_final_{working_codec.lower()}.avi')
    out = cv2.VideoWriter(final_output, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("ERROR: Cannot create final output video!")
        cap.release()
        return False
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read and write frames
    frames_written = 0
    print("Writing frames...")
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Cannot read frame {i}")
            break
        
        # Simple annotation
        cv2.putText(frame, f"Frame {start_frame + i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
        frames_written += 1
        
        if i % 15 == 0:  # Progress every 15 frames
            print(f"  Progress: {i+1}/{num_frames}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Verify output
    if os.path.exists(final_output) and os.path.getsize(final_output) > 10000:
        print(f"SUCCESS: Created clip with {frames_written} frames")
        print(f"Output file: {final_output} ({os.path.getsize(final_output)} bytes)")
        return True
    else:
        print("ERROR: Failed to create valid output file")
        return False

if __name__ == "__main__":
    # Test with your video file
    video_path = input("Enter path to your video file: ").strip().strip('"')
    output_dir = os.path.dirname(video_path)
    
    # Test basic functionality
    print("=== Video Diagnostics ===")
    test_video_reading(video_path)
    
    print("\n=== Codec Testing ===")
    test_output = os.path.join(output_dir, "test_output.avi")
    test_video_writing(test_output)
    
    print("\n=== Simple Clip Creation ===")
    clip_output = os.path.join(output_dir, "test_clip.avi")
    create_simple_clip(video_path, clip_output, 1000, 60)  # 2-second clip starting at frame 1000