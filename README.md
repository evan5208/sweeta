---
title: Sweeta - AI Video Watermark Remover
emoji: 🍭
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🍭 Sweeta - AI Video Watermark Remover

A powerful AI-powered tool for removing watermarks from videos using advanced deep learning techniques. Built on top of SoraCleaner and powered by STTN (Spatio-Temporal Transformer Network).

## ✨ Features

- 🎯 **Smart Detection**: Automatically detects video orientation (landscape/portrait)
- 🧠 **AI-Powered Inpainting**: Uses STTN neural network for seamless watermark removal
- 🎵 **Audio Preservation**: Maintains original audio track in processed videos
- ⚡ **Multiple Watermarks**: Removes multiple watermarks simultaneously
- 🎬 **Batch Processing**: Support for various video formats (MP4, AVI, MOV, MKV)

## 🚀 How It Works

1. **Upload** your video with watermarks
2. **AI Detection** automatically identifies video orientation
3. **Watermark Masking** creates masks for configured watermark positions
4. **AI Inpainting** uses STTN model to fill masked areas naturally
5. **Download** your clean video with original audio preserved

## 🎯 Watermark Configuration

The tool comes pre-configured with watermark positions for:

### Landscape Videos
- Position 1: [35, 585, 176, 638]
- Position 2: [30, 68, 179, 118]  
- Position 3: [1112, 321, 1266, 367]

### Portrait Videos
- Position 1: [28, 1029, 175, 1091]
- Position 2: [538, 604, 685, 657]
- Position 3: [25, 79, 173, 136]

## 🔧 Technical Details

- **Model**: STTN (Spatio-Temporal Transformer Network)
- **Framework**: PyTorch
- **Video Processing**: PyAV
- **Interface**: Gradio
- **Supported Formats**: MP4, AVI, MOV, MKV

## 📝 Usage Notes

- Processing time depends on video length and resolution
- Longer videos may take several minutes to process
- GPU acceleration is used when available for faster processing
- Original audio is preserved in the output video

## 🏗️ Architecture

The tool consists of several key components:

- **Video Processing**: Efficient frame extraction and video reconstruction
- **Orientation Detection**: Automatic landscape/portrait detection
- **Watermark Masking**: Dynamic mask generation based on video orientation
- **AI Inpainting**: STTN-based neural network for seamless content filling
- **Audio Preservation**: Maintains original audio track throughout processing

## 🎬 Supported Video Formats

- MP4 (recommended)
- AVI
- MOV
- MKV

## ⚡ Performance

- **GPU Processing**: Utilizes CUDA when available for faster inference
- **Memory Efficient**: Optimized for processing various video sizes
- **Batch Capable**: Can handle multiple watermark positions simultaneously

## 🔬 Technology Stack

- **Deep Learning**: PyTorch, STTN
- **Video Processing**: PyAV, OpenCV
- **Image Processing**: PIL, NumPy
- **Web Interface**: Gradio
- **Configuration**: YAML

## 📊 Model Information

The STTN model used for inpainting is specifically trained for video content completion and provides:

- Temporal consistency across frames
- High-quality inpainting results
- Efficient processing for various video resolutions
- Robust performance on different content types

## 🤝 Acknowledgments

- [STTN](https://github.com/researchmm/STTN) - Spatio-Temporal Transformer Network for video inpainting
- [SoraCleaner](https://github.com/zstar1003/SoraCleaner) - Original watermark removal implementation
- [KLing-Video-WatermarkRemover-Enhancer](https://github.com/chenwr727/KLing-Video-WatermarkRemover-Enhancer) - Base implementation reference

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes. Please ensure you have the right to modify the videos you process and respect copyright laws.
