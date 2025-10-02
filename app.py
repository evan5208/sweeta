import cv2
import numpy as np
import os
import uuid
import gradio as gr


def detect_sora_watermark(frame, brightness_threshold=200, min_area=800, max_area=15000):
    """
    Detect Sora watermark (white star + text) in frame.
    Returns binary mask of watermark location.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect bright/white areas
    _, bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask for watermark
    mask = np.zeros_like(gray)
    
    # Filter contours by size (to get watermark region)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            # Draw filled rectangle to cover entire watermark area
            cv2.rectangle(mask, (x-5, y-5), (x+w+5, y+h+5), 255, -1)
    
    # If multiple small regions detected close together, merge them
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Expand mask to ensure full coverage
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask


def remove_watermark_inpaint(frame, mask):
    """
    Remove watermark using inpainting.
    """
    # Inpaint the watermark area
    inpainted = cv2.inpaint(frame, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    
    # Apply slight blur for smoother result
    result = cv2.GaussianBlur(inpainted, (3, 3), 0)
    
    return result


def process_video(input_video_path, brightness_threshold=200, min_size=800, max_size=15000, progress=gr.Progress()):
    """
    Remove Sora watermark from video.
    """
    output_video_path = f"./temp/{uuid.uuid4().hex[:8]}.mp4"
    
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        return None, "❌ Error: Unable to open video file"
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    watermarks_detected = 0
    
    progress(0, desc="Starting video processing...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect Sora watermark
        mask = detect_sora_watermark(
            frame,
            brightness_threshold=brightness_threshold,
            min_area=min_size,
            max_area=max_size
        )
        
        # Remove if detected
        if np.any(mask > 0):
            cleaned_frame = remove_watermark_inpaint(frame, mask)
            watermarks_detected += 1
        else:
            cleaned_frame = frame
        
        out.write(cleaned_frame)
        frame_count += 1
        
        # Update progress
        if frame_count % 10 == 0:
            progress_val = frame_count / total_frames
            progress(progress_val, desc=f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    status_msg = f"✅ Done! Processed {frame_count} frames. Watermark detected in {watermarks_detected} frames."
    
    return output_video_path, status_msg


def process_image(input_image, brightness_threshold=200, min_size=800, max_size=15000):
    """
    Remove Sora watermark from single image.
    """
    output_path = f"./temp/{uuid.uuid4().hex[:8]}.jpg"
    
    # Detect watermark
    mask = detect_sora_watermark(
        input_image,
        brightness_threshold=brightness_threshold,
        min_area=min_size,
        max_area=max_size
    )
    
    # Remove if detected
    if np.any(mask > 0):
        result = remove_watermark_inpaint(input_image, mask)
        status = "✅ Watermark detected and removed"
    else:
        result = input_image
        status = "⚠️ No watermark detected - try adjusting the threshold"
    
    cv2.imwrite(output_path, result)
    
    return output_path, result, status


# Create temp directory
if not os.path.exists("./temp"):
    os.makedirs("./temp")


# Gradio interface for images
with gr.Blocks(title="Sora Watermark Remover") as demo:
    gr.Markdown("# 🎬 Sora Watermark Remover")
    gr.Markdown("Remove the animated Sora watermark (spinning star + text) from videos and images")
    
    with gr.Tabs():
        # Image tab
        with gr.Tab("📷 Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Upload Image", type="numpy")
                    
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        img_threshold = gr.Slider(
                            minimum=150, maximum=240, value=200, step=5,
                            label="Brightness Threshold (lower = catches dimmer watermarks)"
                        )
                        img_min_size = gr.Slider(
                            minimum=100, maximum=5000, value=800, step=100,
                            label="Minimum Watermark Size (pixels)"
                        )
                        img_max_size = gr.Slider(
                            minimum=5000, maximum=50000, value=15000, step=1000,
                            label="Maximum Watermark Size (pixels)"
                        )
                    
                    image_btn = gr.Button("🚀 Remove Watermark", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(label="Result")
                    image_download = gr.File(label="Download Image")
                    image_status = gr.Textbox(label="Status", interactive=False)
            
            image_btn.click(
                fn=process_image,
                inputs=[image_input, img_threshold, img_min_size, img_max_size],
                outputs=[image_download, image_output, image_status]
            )
        
        # Video tab
        with gr.Tab("🎥 Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        vid_threshold = gr.Slider(
                            minimum=150, maximum=240, value=200, step=5,
                            label="Brightness Threshold (lower = catches dimmer watermarks)"
                        )
                        vid_min_size = gr.Slider(
                            minimum=100, maximum=5000, value=800, step=100,
                            label="Minimum Watermark Size (pixels)"
                        )
                        vid_max_size = gr.Slider(
                            minimum=5000, maximum=50000, value=15000, step=1000,
                            label="Maximum Watermark Size (pixels)"
                        )
                    
                    video_btn = gr.Button("🚀 Remove Watermark", variant="primary")
                
                with gr.Column():
                    video_output = gr.Video(label="Result")
                    video_download = gr.File(label="Download Video")
                    video_status = gr.Textbox(label="Status", interactive=False)
            
            video_btn.click(
                fn=process_video,
                inputs=[video_input, vid_threshold, vid_min_size, vid_max_size],
                outputs=[video_download, video_status]
            )
    
    gr.Markdown("""
    ### 💡 Tips:
    - **Brightness Threshold**: If watermark isn't being removed, try lowering this (180-190)
    - **Min/Max Size**: Adjust based on your video resolution and watermark size
    - Works best when watermark is white/light colored and video background is darker
    - The spinning star animation doesn't affect removal since we detect frame-by-frame
    """)


if __name__ == "__main__":
    demo.queue().launch()