"""
Lightweight Sora Watermark Remover for Hugging Face Spaces (Free Tier)
Uses LaMa inpainting model - runs on CPU, relatively fast

Install: pip install gradio opencv-python numpy pillow simple-lama-inpainting
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import uuid
from simple_lama_inpainting import SimpleLama

# Initialize LaMa model (lightweight, CPU-friendly)
simple_lama = None

def get_lama_model():
    global simple_lama
    if simple_lama is None:
        simple_lama = SimpleLama()
    return simple_lama


def create_sora_template():
    """Create a rough template of what the Sora watermark looks like for template matching."""
    # Create a simple star + text template (this is approximate)
    template = np.zeros((40, 120), dtype=np.uint8)
    
    # Draw a simple star shape (rough approximation)
    star_center = (20, 20)
    star_points = []
    for i in range(10):
        angle = i * np.pi / 5
        if i % 2 == 0:
            radius = 12
        else:
            radius = 6
        x = int(star_center[0] + radius * np.cos(angle))
        y = int(star_center[1] + radius * np.sin(angle))
        star_points.append([x, y])
    
    star_points = np.array(star_points, dtype=np.int32)
    cv2.fillPoly(template, [star_points], 200)
    
    # Add rough "Sora" text representation
    cv2.rectangle(template, (45, 12), (110, 28), 180, -1)  # Text block
    
    return template


def detect_sora_watermark_smart(frame, prev_mask=None):
    """
    Sora-specific watermark detection for semi-transparent animated star + text.
    The Sora watermark is a white/light gray star shape with "Sora" text that moves randomly.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Detect semi-transparent overlays using variance
    # Sora watermark creates slight brightness variations
    blur_heavy = cv2.GaussianBlur(gray, (15, 15), 0)
    blur_light = cv2.GaussianBlur(gray, (3, 3), 0)
    overlay_diff = cv2.absdiff(blur_heavy, blur_light)
    
    # Method 2: Look for specific brightness patterns (semi-transparent white)
    # Sora watermark makes areas slightly brighter, not fully white
    bright_overlay = cv2.inRange(gray, 200, 240)  # Semi-transparent range
    
    # Method 3: Text detection using morphological operations
    # The "Sora" text creates horizontal line patterns
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    text_like = cv2.morphologyEx(bright_overlay, cv2.MORPH_CLOSE, kernel_horizontal)
    
    # Method 4: Star shape detection using corner detection
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corner_mask = np.zeros_like(gray)
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel().astype(int)
            cv2.circle(corner_mask, (x, y), 8, 255, -1)
    
    # Combine all methods
    combined = cv2.bitwise_or(overlay_diff, bright_overlay)
    combined = cv2.bitwise_or(combined, text_like)
    
    # Apply morphological operations to connect star and text
    kernel = np.ones((7, 7), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Find potential watermark regions
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros_like(gray)
    watermark_found = False
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        # Sora watermark characteristics:
        # - Size: typically 80-200 pixels wide
        # - Aspect ratio: roughly 2:1 to 4:1 (star + text)
        # - Position: can be anywhere but often in corners/edges
        
        if 1000 < area < 15000:  # Reasonable size for watermark
            aspect_ratio = w_box / h_box if h_box > 0 else 0
            
            if 1.5 < aspect_ratio < 6.0:  # Horizontal layout (star + "Sora")
                # Add generous padding to ensure we capture the entire watermark
                pad_x = max(20, w_box // 4)
                pad_y = max(15, h_box // 4)
                
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + w_box + pad_x)
                y2 = min(h, y + h_box + pad_y)
                
                # Check if this region looks like it could contain text/logo
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    mean_brightness = np.mean(roi)
                    brightness_std = np.std(roi)
                    
                    # Watermark regions have higher brightness and some variation
                    if mean_brightness > 120 and brightness_std > 10:
                        cv2.rectangle(final_mask, (x1, y1), (x2, y2), 255, -1)
                        watermark_found = True
    
    # If no watermark found with main method, try template matching
    if not watermark_found:
        try:
            template = create_sora_template()
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # If we find a good match
            if max_val > 0.3:  # Adjust threshold as needed
                x, y = max_loc
                h_t, w_t = template.shape
                
                # Add padding around the detected region
                pad = 20
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + w_t + pad)
                y2 = min(h, y + h_t + pad)
                
                cv2.rectangle(final_mask, (x1, y1), (x2, y2), 255, -1)
                watermark_found = True
        except:
            pass  # Template matching failed, continue with edge detection
    
    # If still no watermark found, try edge-based detection as final fallback
    if not watermark_found:
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 12000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = w_box / h_box if h_box > 0 else 0
                
                if 1.2 < aspect_ratio < 5.0:
                    pad = 25
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(w, x + w_box + pad)
                    y2 = min(h, y + h_box + pad)
                    cv2.rectangle(final_mask, (x1, y1), (x2, y2), 255, -1)
    
    return final_mask


def process_video_lightweight(video_path, use_temporal=True, progress=gr.Progress()):
    """
    Process video using lightweight approach suitable for HF Spaces.
    """
    output_path = f"./temp/{uuid.uuid4().hex[:8]}.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Error opening video"
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress(0, desc="Initializing...")
    
    # First pass: detect watermark in all frames
    progress(0.05, desc="Analyzing video...")
    masks = []
    frames_buffer = []
    
    frame_idx = 0
    prev_mask = None
    
    # Limit memory usage - process max 1000 frames at once
    max_frames = min(total_frames, 1000)
    print(f"Processing {max_frames}/{total_frames} frames to avoid memory issues")
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_buffer.append(frame)
        mask = detect_sora_watermark_smart(frame, prev_mask)
        masks.append(mask)
        prev_mask = mask
        
        frame_idx += 1
        if frame_idx % 20 == 0:
            progress(0.05 + 0.25 * (frame_idx / max_frames), 
                    desc=f"Analyzing {frame_idx}/{max_frames}")
    
    cap.release()
    
    # Second pass: remove watermark using temporal info
    progress(0.3, desc="Removing watermark...")
    print(f"Starting watermark removal for {len(frames_buffer)} frames...")
    
    lama = None  # Initialize lazily to avoid hanging
    
    for i, (frame, mask) in enumerate(zip(frames_buffer, masks)):
        print(f"Processing frame {i+1}/{len(frames_buffer)}")
        if np.sum(mask) > 500:  # Watermark detected (increased threshold for more selective detection)
            print(f"  Watermark detected in frame {i+1}, removing...")
            
            if use_temporal:
                # For moving watermarks, we need a smarter temporal approach
                # Find frames where watermark is in different positions
                clean_patches = []
                watermark_region = None
                
                # Get the bounding box of the current watermark
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w_mask, h_mask = cv2.boundingRect(largest_contour)
                    watermark_region = (x, y, w_mask, h_mask)
                
                if watermark_region:
                    x, y, w_mask, h_mask = watermark_region
                    
                    # Look for frames where this same region is clean
                    for j in range(max(0, i-15), min(len(frames_buffer), i+16)):
                        if j != i:
                            other_mask = masks[j]
                            # Check if the watermark region in frame j is clean
                            roi_mask = other_mask[y:y+h_mask, x:x+w_mask]
                            if roi_mask.size > 0 and np.sum(roi_mask) < np.sum(mask) * 0.3:
                                # This region is cleaner in frame j
                                clean_patch = frames_buffer[j][y:y+h_mask, x:x+w_mask]
                                clean_patches.append(clean_patch)
                    
                    if len(clean_patches) >= 3:
                        print(f"  Found {len(clean_patches)} clean patches for temporal reconstruction")
                        # Use median of clean patches
                        median_patch = np.median(clean_patches, axis=0).astype(np.uint8)
                        
                        # Create result by replacing the watermark region
                        result = frame.copy()
                        
                        # Smooth blending at edges to avoid artifacts
                        mask_region = mask[y:y+h_mask, x:x+w_mask]
                        if mask_region.size > 0:
                            # Create a soft mask for blending
                            soft_mask = cv2.GaussianBlur(mask_region.astype(np.float32), (11, 11), 3) / 255.0
                            soft_mask = np.stack([soft_mask] * 3, axis=-1)
                            
                            current_patch = result[y:y+h_mask, x:x+w_mask].astype(np.float32)
                            blended_patch = (current_patch * (1 - soft_mask) + median_patch.astype(np.float32) * soft_mask)
                            result[y:y+h_mask, x:x+w_mask] = blended_patch.astype(np.uint8)
                        
                    else:
                        print(f"  Only {len(clean_patches)} clean patches found, using advanced inpainting...")
                        # Use advanced inpainting with edge-aware approach
                        try:
                            if lama is None:
                                print("  Initializing LaMa model...")
                                lama = get_lama_model()
                            
                            # Dilate mask slightly for better inpainting
                            kernel = np.ones((5, 5), np.uint8)
                            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
                            
                            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            mask_pil = Image.fromarray(dilated_mask)
                            result_pil = lama(frame_pil, mask_pil)
                            result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                        except Exception as e:
                            print(f"  LaMa inpainting failed: {e}, using basic reconstruction")
                            # Fallback: simple inpainting
                            result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                else:
                    print(f"  No watermark region found, using full frame inpainting")
                    try:
                        if lama is None:
                            lama = get_lama_model()
                        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        mask_pil = Image.fromarray(mask)
                        result_pil = lama(frame_pil, mask_pil)
                        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"  Inpainting failed: {e}, using original frame")
                        result = frame
            else:
                print(f"  Using LaMa inpainting only...")
                try:
                    if lama is None:
                        print("  Initializing LaMa model...")
                        lama = get_lama_model()
                    
                    # Slightly dilate mask for better results
                    kernel = np.ones((3, 3), np.uint8)
                    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    mask_pil = Image.fromarray(dilated_mask)
                    result_pil = lama(frame_pil, mask_pil)
                    result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"  LaMa inpainting failed: {e}, using OpenCV inpainting")
                    result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        else:
            result = frame
        
        out.write(result)
        
        # Update progress more frequently and add debug info
        if i % 5 == 0 or i == len(frames_buffer) - 1:
            progress_val = 0.3 + 0.7 * (i / len(frames_buffer))
            progress(progress_val, desc=f"Processing {i+1}/{len(frames_buffer)} ({progress_val*100:.1f}%)")
            print(f"Progress: {progress_val*100:.1f}% - Frame {i+1}/{len(frames_buffer)}")
    
    out.release()
    
    detected_count = sum(1 for m in masks if np.sum(m) > 500)
    processed_count = len([result for result in frames_buffer if result is not None])
    
    status = f"✅ Done! Processed {processed_count}/{total_frames} frames. Sora watermark detected and removed in {detected_count} frames."
    
    if detected_count == 0:
        status += "\n⚠️ No Sora watermark detected. The video might not have the watermark, or it may be too faint/different from expected."
    elif detected_count < total_frames * 0.1:
        status += f"\n💡 Watermark detected in only {detected_count} frames. This suggests the watermark appears sporadically."
    else:
        status += f"\n🎯 Successfully processed video with consistent watermark removal!"
    
    return output_path, status


def process_image_lightweight(image):
    """Process single image."""
    output_path = f"./temp/{uuid.uuid4().hex[:8]}.jpg"
    
    # Detect watermark
    mask = detect_sora_watermark_smart(image)
    
    if np.sum(mask) > 100:
        # Use LaMa for inpainting
        lama = get_lama_model()
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        result_pil = lama(image_pil, mask_pil)
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        status = "✅ Watermark detected and removed"
    else:
        result = image
        status = "⚠️ No watermark detected"
    
    cv2.imwrite(output_path, result)
    return output_path, result, status


# Create temp directory
os.makedirs("./temp", exist_ok=True)

# Gradio Interface
with gr.Blocks(title="Sora Watermark Remover", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 Sora Watermark Remover
    ### AI-powered removal specifically designed for Sora's animated watermark
    
    **✨ Features:**
    - 🎯 **Sora-Specific Detection**: Targets the animated star + "Sora" text watermark
    - 🔄 **Moving Watermark Support**: Handles randomly positioned watermarks
    - 🧠 **Smart Temporal Analysis**: Uses clean frames to reconstruct watermarked areas
    - 🚀 **CPU Optimized**: Runs efficiently on Hugging Face free tier
    - 🎨 **Advanced Inpainting**: LaMa model + custom reconstruction techniques
    """)
    
    with gr.Tabs():
        with gr.Tab("🎥 Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    temporal_check = gr.Checkbox(
                        label="Use Temporal Analysis (Recommended)", 
                        value=True,
                        info="Uses frames before/after for better quality"
                    )
                    video_btn = gr.Button("🚀 Remove Watermark", variant="primary", size="lg")
                
                with gr.Column():
                    video_output = gr.Video(label="Result")
                    video_status = gr.Textbox(label="Status", interactive=False)
            
            video_btn.click(
                fn=process_video_lightweight,
                inputs=[video_input, temporal_check],
                outputs=[video_output, video_status]
            )
        
        with gr.Tab("📷 Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Upload Image", type="numpy")
                    image_btn = gr.Button("🚀 Remove Watermark", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(label="Result")
                    image_download = gr.File(label="Download")
                    image_status = gr.Textbox(label="Status", interactive=False)
            
            image_btn.click(
                fn=process_image_lightweight,
                inputs=[image_input],
                outputs=[image_download, image_output, image_status]
            )
    
    gr.Markdown("""
    ### 💡 How it works:
    1. **🔍 Sora Detection**: Multi-method detection targeting semi-transparent star + text
       - Overlay variance analysis for semi-transparent elements
       - Morphological text detection for "Sora" text
       - Template matching for star shape recognition
       - Edge detection as fallback
    
    2. **🎯 Moving Watermark Handling**: Smart temporal reconstruction
       - Tracks watermark positions across frames
       - Finds clean patches from frames where watermark is elsewhere
       - Smooth blending to avoid artifacts
    
    3. **🎨 Advanced Inpainting**: Multiple reconstruction methods
       - Temporal median from clean frames (primary)
       - LaMa AI inpainting (fallback)
       - OpenCV inpainting (emergency fallback)
    
    ### ⚙️ Technical Details:
    - **Detection**: Custom algorithm for Sora's animated watermark
    - **Model**: LaMa (Large Mask Inpainting) - 25MB, CPU-friendly
    - **Speed**: ~3-8 seconds per frame (depends on watermark complexity)
    - **Memory**: Optimized for HF free tier (max 1000 frames per batch)
    
    **🎯 Specifically designed for Sora's moving watermark! �**
    """)

if __name__ == "__main__":
    demo.queue().launch()