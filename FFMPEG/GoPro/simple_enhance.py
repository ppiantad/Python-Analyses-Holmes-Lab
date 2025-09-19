import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from pathlib import Path

# Install required packages:
# pip install torch torchvision opencv-python pillow numpy

class ZeroDCE(nn.Module):
    """Zero-DCE network for low-light enhancement"""
    def __init__(self):
        super(ZeroDCE, self).__init__()
        # Lightweight CNN for curve estimation
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        
        # Output layer for curve parameters (24 curves = 8 iterations * 3 channels)
        self.conv_out = nn.Conv2d(32, 24, 3, padding=1)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        
        # Output curve parameters
        curves = torch.tanh(self.conv_out(x4))
        return curves

def curve_enhancement(img, curves, iterations=8):
    """Apply curve enhancement based on Zero-DCE"""
    enhanced = img.clone()
    
    for i in range(iterations):
        # Extract curve parameters for this iteration
        curve_params = curves[:, i*3:(i+1)*3, :, :]
        
        # Apply curve transformation
        enhanced = enhanced + curve_params * (torch.pow(enhanced, 2) - enhanced)
        enhanced = torch.clamp(enhanced, 0, 1)
    
    return enhanced

class EnhancementPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize Zero-DCE model
        self.zero_dce = ZeroDCE().to(self.device)
        
        # Load pretrained weights if available
        self.load_pretrained_weights()
        
    def load_pretrained_weights(self):
        """Load pretrained Zero-DCE weights if available"""
        # Note: You would need to download pretrained weights or train the model
        # For demo purposes, we'll use random initialization
        print("Using randomly initialized weights. For best results, use pretrained Zero-DCE weights.")
        
    def preprocess_frame(self, frame):
        """Preprocess video frame for AI enhancement"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_norm = frame_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor.to(self.device)
    
    def postprocess_frame(self, tensor):
        """Convert tensor back to OpenCV format"""
        # Move to CPU and convert to numpy
        frame = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Convert back to [0, 255] and BGR
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
    def enhance_frame_zero_dce(self, frame):
        """Enhance frame using Zero-DCE approach"""
        frame_tensor = self.preprocess_frame(frame)
        
        with torch.no_grad():
            # Get curve parameters
            curves = self.zero_dce(frame_tensor)
            
            # Apply curve enhancement
            enhanced = curve_enhancement(frame_tensor, curves)
            
        return self.postprocess_frame(enhanced)
    
    def enhance_frame_adaptive_clahe(self, frame):
        """Enhanced CLAHE with adaptive parameters"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adaptive CLAHE parameters based on image statistics
        mean_brightness = np.mean(l)
        if mean_brightness < 50:  # Very dark
            clip_limit = 4.0
            tile_size = (4, 4)
        elif mean_brightness < 100:  # Moderately dark
            clip_limit = 3.0
            tile_size = (6, 6)
        else:  # Less dark
            clip_limit = 2.0
            tile_size = (8, 8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_enhanced = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def enhance_frame_gamma_correction(self, frame, gamma=None):
        """Adaptive gamma correction"""
        if gamma is None:
            # Adaptive gamma based on image brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            # Darker images need lower gamma (more brightening)
            gamma = max(0.3, min(2.0, 1.0 + (128 - mean_brightness) / 128))
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        
        # Apply gamma correction
        enhanced = cv2.LUT(frame, table)
        
        return enhanced
    
    def enhance_frame_hybrid(self, frame):
        """Hybrid approach combining multiple techniques"""
        # Step 1: Gamma correction for initial brightening
        gamma_enhanced = self.enhance_frame_gamma_correction(frame)
        
        # Step 2: Adaptive CLAHE for local contrast
        clahe_enhanced = self.enhance_frame_adaptive_clahe(gamma_enhanced)
        
        # Step 3: Noise reduction (important for low-light)
        denoised = cv2.bilateralFilter(clahe_enhanced, 9, 75, 75)
        
        return denoised
    
    def process_video(self, input_path, output_path, method='hybrid', max_frames=None):
        """Process entire video with selected enhancement method"""
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Apply selected enhancement method
            if method == 'zero_dce':
                enhanced = self.enhance_frame_zero_dce(frame)
            elif method == 'clahe':
                enhanced = self.enhance_frame_adaptive_clahe(frame)
            elif method == 'gamma':
                enhanced = self.enhance_frame_gamma_correction(frame)
            elif method == 'hybrid':
                enhanced = self.enhance_frame_hybrid(frame)
            else:
                enhanced = frame
            
            out.write(enhanced)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
            
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        out.release()
        print(f"Enhanced video saved to: {output_path}")
    
    def compare_methods(self, input_path, output_dir, sample_frame=100):
        """Compare different enhancement methods on a sample frame"""
        cap = cv2.VideoCapture(input_path)
        
        # Jump to sample frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Could not read sample frame")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Original
        cv2.imwrite(f"{output_dir}/original.jpg", frame)
        
        # Different methods
        methods = {
            'zero_dce': self.enhance_frame_zero_dce,
            'clahe': self.enhance_frame_adaptive_clahe,
            'gamma': self.enhance_frame_gamma_correction,
            'hybrid': self.enhance_frame_hybrid
        }
        
        for name, method in methods.items():
            try:
                enhanced = method(frame)
                cv2.imwrite(f"{output_dir}/{name}.jpg", enhanced)
                print(f"Saved {name} comparison")
            except Exception as e:
                print(f"Error with {name}: {e}")

# Usage example
def main():
    # Initialize enhancement pipeline
    enhancer = EnhancementPipeline()
    
    # Paths
    input_video = "dark_mouse_video.mp4"  # Replace with your video path
    output_dir = "enhanced_videos"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Compare different approaches on a single frame
    print("Comparing enhancement methods...")
    enhancer.compare_methods(input_video, "comparison_frames")
    
    # Method 2: Process full video with hybrid approach (recommended)
    print("Processing video with hybrid approach...")
    enhancer.process_video(
        input_video, 
        f"{output_dir}/enhanced_hybrid.mp4", 
        method='hybrid'
    )
    
    # Method 3: Process with adaptive CLAHE (fast, good results)
    print("Processing video with adaptive CLAHE...")
    enhancer.process_video(
        input_video, 
        f"{output_dir}/enhanced_clahe.mp4", 
        method='clahe'
    )

if __name__ == "__main__":
    main()

# Additional utility functions for fine-tuning
class VideoAnalyzer:
    """Analyze video properties to optimize enhancement parameters"""
    
    @staticmethod
    def analyze_brightness_distribution(video_path, sample_frames=50):
        """Analyze brightness distribution across video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        brightness_values = []
        frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_values.append(np.mean(gray))
        
        cap.release()
        
        return {
            'mean_brightness': np.mean(brightness_values),
            'std_brightness': np.std(brightness_values),
            'min_brightness': np.min(brightness_values),
            'max_brightness': np.max(brightness_values)
        }
    
    @staticmethod
    def recommend_parameters(analysis_results):
        """Recommend enhancement parameters based on analysis"""
        mean_brightness = analysis_results['mean_brightness']
        std_brightness = analysis_results['std_brightness']
        
        if mean_brightness < 30:
            return {
                'method': 'hybrid',
                'gamma': 0.4,
                'clahe_clip': 4.0,
                'clahe_tile': (4, 4)
            }
        elif mean_brightness < 60:
            return {
                'method': 'clahe',
                'gamma': 0.6,
                'clahe_clip': 3.0,
                'clahe_tile': (6, 6)
            }
        else:
            return {
                'method': 'gamma',
                'gamma': 0.8,
                'clahe_clip': 2.0,
                'clahe_tile': (8, 8)
            }