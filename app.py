import gradio as gr
import os
import tempfile
import shutil
from typing import List, Optional
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the SoraCleaner modules
try:
    from modules.erase import remove_watermark_from_frames
    from utils.logging_utils import update_status
    from utils.video_utils import (
        read_video_to_pil_images,
        write_pil_images_to_video,
        detect_video_orientation
    )
    from modules.config import load_config
    from download_model import ensure_model_weights
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Ensure model weights are available
model_path = ensure_model_weights()
if not model_path:
    print("❌ Model weights not found! Please ensure weights/sttn.pth exists.")
    sys.exit(1)

# Load configuration
try:
    CONFIG = load_config("config.yaml")
    print("✅ Configuration loaded successfully")
except Exception as e:
    print(f"❌ Failed to load configuration: {e}")
    sys.exit(1)

def process_video_gradio(
    input_video: str,
    progress: gr.Progress = gr.Progress()
) -> tuple[str, str]:
    """
    Process a video to remove watermarks using Gradio interface.
    
    Args:
        input_video: Path to the input video file
        progress: Gradio progress tracker
    
    Returns:
        tuple: (output_video_path, status_message)
    """
    if not input_video:
        return None, "❌ Please upload a video file."
    
    if not os.path.exists(input_video):
        return None, "❌ Uploaded video file not found."
    
    try:
        progress(0.1, desc="🎬 Reading video...")
        print(f"Processing video: {input_video}")
        
        # Read video frames
        frames, fps = read_video_to_pil_images(input_video)
        if not frames:
            return None, "❌ Failed to decode video. Please check the file format and ensure it's a valid video file."
        
        progress(0.3, desc=f"📊 Video loaded: {len(frames)} frames at {fps:.2f} FPS")
        print(f"Video loaded: {len(frames)} frames at {fps:.2f} FPS")
        
        # Detect orientation
        orientation = detect_video_orientation(input_video)
        progress(0.4, desc=f"🔍 Detected orientation: {orientation}")
        print(f"Detected orientation: {orientation}")
        
        # Check if we have watermark positions for this orientation
        positions_key = "positions_portrait" if orientation == "portrait" else "positions_landscape"
        positions = CONFIG["watermark"].get(positions_key, [])
        
        if not positions:
            return None, f"❌ No watermark positions configured for {orientation} videos. Please check config.yaml"
        
        progress(0.5, desc=f"🧹 Removing {len(positions)} watermarks...")
        print(f"Removing {len(positions)} watermarks using positions: {positions}")
        
        # Remove watermarks
        processed_frames = remove_watermark_from_frames(frames, input_video)
        
        if not processed_frames:
            return None, "❌ Failed to process frames. Watermark removal failed."
        
        progress(0.8, desc="🎞️ Encoding output video...")
        print("Encoding output video...")
        
        # Create output path in a temporary directory
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "cleaned_video.mp4")
        
        # Write processed video
        write_pil_images_to_video(output_path, processed_frames, fps, input_video)
        
        if not os.path.exists(output_path):
            return None, "❌ Failed to create output video file."
        
        progress(1.0, desc="✅ Processing complete!")
        print(f"Processing complete! Output saved to: {output_path}")
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        return output_path, f"✅ Successfully processed video!\n📹 Original: {len(frames)} frames at {fps:.2f} FPS\n🎯 Orientation: {orientation}\n🧹 Watermarks removed: {len(positions)} positions\n📁 Output size: {file_size:.1f} MB"
        
    except Exception as e:
        error_msg = f"❌ Error processing video: {str(e)}"
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        return None, error_msg

def get_watermark_info() -> str:
    """Get information about configured watermark positions."""
    config = CONFIG.get("watermark", {})
    
    landscape_positions = config.get("positions_landscape", [])
    portrait_positions = config.get("positions_portrait", [])
    
    info = "🎯 **Configured Watermark Positions:**\n\n"
    info += f"**Landscape Videos:** {len(landscape_positions)} positions\n"
    for i, pos in enumerate(landscape_positions, 1):
        info += f"  {i}. [{pos[0]}, {pos[1]}, {pos[2]}, {pos[3]}]\n"
    
    info += f"\n**Portrait Videos:** {len(portrait_positions)} positions\n"
    for i, pos in enumerate(portrait_positions, 1):
        info += f"  {i}. [{pos[0]}, {pos[1]}, {pos[2]}, {pos[3]}]\n"
    
    info += f"\n**Settings:**\n"
    info += f"- Mask Expansion: {config.get('mask_expand', 10)} pixels\n"
    info += f"- Neighbor Stride: {config.get('neighbor_stride', 10)}\n"
    
    return info

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="Sweeta - AI Video Watermark Remover",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🍭 Sweeta - AI Video Watermark Remover</h1>
            <p>Remove watermarks from videos using advanced AI inpainting technology</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("""
                <div class="feature-box">
                    <h3>✨ Features</h3>
                    <ul>
                        <li>🎯 <strong>Smart Detection:</strong> Auto-detects landscape/portrait orientation</li>
                        <li>🧠 <strong>AI-Powered:</strong> Uses STTN neural network for seamless inpainting</li>
                        <li>🎵 <strong>Audio Preserved:</strong> Maintains original audio track</li>
                        <li>⚡ <strong>Multiple Watermarks:</strong> Removes multiple watermarks simultaneously</li>
                    </ul>
                </div>
                """)
                
                input_video = gr.Video(
                    label="📹 Upload Video",
                    format="mp4"
                )
                
                process_btn = gr.Button(
                    "🧹 Remove Watermarks",
                    variant="primary",
                    size="lg"
                )
                
                status_text = gr.Textbox(
                    label="📊 Status",
                    lines=4,
                    interactive=False
                )
            
            with gr.Column(scale=2):
                output_video = gr.Video(
                    label="✨ Processed Video",
                    format="mp4"
                )
                
                gr.HTML("""
                <div class="feature-box">
                    <h3>ℹ️ How it works</h3>
                    <ol>
                        <li>Upload your video with watermarks</li>
                        <li>AI detects video orientation automatically</li>
                        <li>Configured watermark positions are masked</li>
                        <li>STTN model fills masked areas naturally</li>
                        <li>Download your clean video with original audio</li>
                    </ol>
                </div>
                """)
        
        with gr.Accordion("🎯 Watermark Configuration", open=False):
            watermark_info = gr.Markdown(get_watermark_info())
        
        with gr.Accordion("📚 About", open=False):
            gr.Markdown("""
            **Sweeta** is powered by SoraCleaner, an advanced AI tool for removing watermarks from videos.
            
            **Technology:**
            - **STTN (Spatio-Temporal Transformer Network)** for video inpainting
            - **PyAV** for efficient video processing
            - **PyTorch** for deep learning inference
            
            **Supported Formats:** MP4, AVI, MOV, MKV
            
            **Note:** Processing time depends on video length and resolution. Longer videos may take several minutes.
            """)
        
        # Event handlers
        process_btn.click(
            fn=process_video_gradio,
            inputs=[input_video],
            outputs=[output_video, status_text],
            show_progress=True
        )
        
        # Example videos (if available)
        example_videos = []
        if os.path.exists("STTN/examples"):
            example_dir = "STTN/examples"
            for file in os.listdir(example_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    example_videos.append(os.path.join(example_dir, file))
        
        if example_videos:
            gr.Examples(
                examples=[[video] for video in example_videos[:3]],  # Limit to 3 examples
                inputs=input_video,
                label="📁 Example Videos"
            )
    
    return demo

if __name__ == "__main__":
    # Ensure weights directory exists
    os.makedirs("weights", exist_ok=True)
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
