import cv2
import numpy as np

def remove_teleporting_watermark(input_video, output_video, sample_frames=30):
    """
    Remove watermark that randomly teleports to different positions.
    Perfect for TikTok-style moving watermarks.
    
    Args:
        input_video: Path to input video
        output_video: Path to output video  
        sample_frames: Number of frames to sample for building clean reference
    """
    cap = cv2.VideoCapture(input_video)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"\nStep 1/3: Sampling {sample_frames} frames to build watermark-free reference...")
    
    # Sample frames evenly throughout video
    sample_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    sampled_frames = []
    
    for i, idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
            if (i + 1) % 10 == 0:
                print(f"  Sampled {i + 1}/{sample_frames} frames")
    
    # Create clean reference by taking median
    # Since watermark teleports, each pixel is clean in most frames
    print("\nStep 2/3: Computing clean reference (this may take a moment)...")
    reference = np.median(sampled_frames, axis=0).astype(np.uint8)
    
    print("\nStep 3/3: Processing video and removing watermark...")
    
    # Process video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to LAB color space for better luminance detection
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        
        # Extract luminance channel (L)
        l_frame = lab_frame[:, :, 0]
        l_ref = lab_ref[:, :, 0]
        
        # Detect watermark (brighter areas in current frame)
        diff = cv2.subtract(l_frame, l_ref)
        
        # Threshold to find watermark
        # Adjust this value if needed: higher = less aggressive, lower = more aggressive
        _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask to remove noise
        kernel = np.ones((3, 3), np.uint8)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill gaps in watermark
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Slightly expand mask to ensure full watermark coverage
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Apply mask with smooth blending at edges
        # Convert mask to 3 channels
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Blend: use reference where watermark is detected
        result = (frame * (1 - mask_3ch) + reference * mask_3ch).astype(np.uint8)
        
        out.write(result)
        frame_count += 1
        
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    out.release()
    print(f"\n✓ Done! Watermark removed. Saved to: {output_file}")
    print(f"\nNote: If results aren't perfect, try adjusting:")
    print(f"  - sample_frames (current: {sample_frames}) - increase for better quality")
    print(f"  - threshold value (line 72) - increase if removing too much, decrease if missing watermark")


# Usage
if __name__ == "__main__":
    input_file = "video.mp4"  # Your TikTok video
    output_file = "no_watermark.mp4"  # Clean output
    
    # Process the video
    # Increase sample_frames (e.g., 50) for better quality on longer videos
    remove_teleporting_watermark(input_file, output_file, sample_frames=30)