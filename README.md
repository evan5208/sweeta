<div align="center">

# Sweeta

### Remove Watermarks from SORA 2 Video Generations

[![License](https://img.shields.io/badge/License-Apache%202.0-pink.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-pink.svg)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14Z0_QynK6P9GSCEDDEJT2Tiv_WXUyAfK?usp=sharing)

</div>

SORA 2 is a State-of-the-art model by OpenAI and for the past few days, being on platforms like Instagram and Twitter, I've noticed how many non-technical people just assume the video is real despite the watermark.

Think what would happen if there was no watermark?
This is the reason that this project exists. It's not to abuse the great initiative by OpenAI to put logos onto every generation (though temporarily there's also an easy way to bypass that which I wouldn't cover), it's to hopefully encourage them to be harsher and more obvious with it in some form.

Sweeta is an AI-powered watermark removal tool specifically designed for SORA 2 video generations. It Uses advanced inpainting models (LaMA) and intelligent detection algorithms, it can seamlessly remove watermarks while (mostly) preserving the original image quality.

---
### Recommended Specs:
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **RAM**: 16GB or more
- **Storage**: 10GB free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM (or Apple Silicon for macOS)
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5/Apple M1 or better)

---

## Installation

### Prerequisites

1. **Python & Conda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or Anaconda

### Windows

#### Quick Install (Recommended)

1. Open Command Prompt or PowerShell as administrator
2. Navigate to the project folder
3. Run the installation script:
   ```cmd
   cd path\to\Sweeta
   windows\install_windows.bat
   ```
   
   Or for PowerShell:
   ```powershell
   powershell -ExecutionPolicy Bypass -File windows\install_windows.ps1
   ```

4. Follow the on-screen instructions

#### Manual Installation

```powershell
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate py312aiwatermark

# Install additional dependencies
pip install PyQt6 transformers iopaint opencv-python-headless

# Download the LaMA model
iopaint download --model lama
```

### Linux

```bash
# Navigate to the project directory
cd /path/to/Sweeta

# Run the setup script
bash linux/setup.sh

# Or manually:
conda env create -f environment.yml
conda activate py312aiwatermark
pip install PyQt6 transformers iopaint opencv-python-headless
iopaint download --model lama
```

### Colab

Access the Colab notebook from [here](https://colab.research.google.com/drive/14Z0_QynK6P9GSCEDDEJT2Tiv_WXUyAfK?usp=sharing) and follow the instructions.

---

## Usage

### Launching the Application

#### GUI Mode (Recommended)

1. Activate the conda environment:
   ```bash
   conda activate py312aiwatermark
   ```

2. Launch the GUI application:
   ```bash
   python remwmgui.py
   ```

(would be happy to prepare a hugging face port for Spaces too, which would technically be better but would require community GPU access)

#### Command Line Mode

```bash
conda activate py312aiwatermark
python remwm.py <input_path> <output_path> [options]
```

**Example:**
```bash
python remwm.py input_video.mp4 output_video.mp4 --max-bbox-percent 15 --force-format MP4 --transparent --overwrite
```

**Available options:**
- `--max-bbox-percent`: Detection sensitivity (default: 10.0)
- `--force-format`: Output format (PNG, WEBP, JPG, MP4, AVI)
- `--transparent`: Make watermark areas transparent
- `--overwrite`: Overwrite existing files

### Configuration Edit

Refer #ui.yml.example

**Configuration Options**
   - **Input Path**: Select your source file or folder
   - **Output Path**: Choose where to save processed files
   - **Overwrite Files**: Enable to replace existing output files
   - **Transparent Watermarks**: Make watermark areas transparent (PNG only)
   - **Max BBox Percent**: Adjust detection sensitivity (1-100%)
   - **Output Format**: Choose PNG, WEBP, JPG, or keep original format

### Common Issues

#### ImportError: cannot import name 'cached_download' from 'huggingface_hub'
**Solution**: This is a version compatibility issue. The installation scripts now automatically install the correct version. If you installed manually, run:
```bash
pip install "huggingface-hub<0.20"
pip install --upgrade iopaint
```

#### "Conda is not recognized as an internal or external command"
**Solution**: Ensure Conda is properly installed and added to your system PATH environment variable.

#### Dependency Installation Failures
**Solution**: Try installing dependencies individually:
```bash
pip install PyQt6
pip install transformers
pip install iopaint
pip install opencv-python-headless
```

#### Application Won't Start
**Solution**: Verify the environment is activated:
```bash
conda activate py312aiwatermark
python --version  # Should show Python 3.12.x
```

#### LaMA Model Download Issues
**Solution**: Ensure stable internet connection and retry:
```bash
iopaint download --model lama
```

#### CUDA/GPU Issues
**Solution**: Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If you run into any issues or have something to say, reach out! I'd be happy to talk :)
[Twitter](https://x.com/Kuberwastaken), [LinkedIn](https://www.linkedin.com/in/kubermehta/)

---

I (tired to) microblog the development process in #journal.md but uh, read it at your own risk lol

---

## 📄 License & Disclaimer

### License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
Thanks to D-Ogi for the WatermarkRemover-AI model which was heavily modified for this project.

### ⚠️ Important Disclaimer

**THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**
**Use this tool responsibly and ethically.**

---

<div align="center">

Made with <3 by [Kuber Mehta](https://kuber.studio/)

Star this repo if you found it cool

</div>
