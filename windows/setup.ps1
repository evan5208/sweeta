# PowerShell equivalent of setup.sh for Windows
$ENV_NAME = "py312aiwatermark"
$ENV_FILE = "environment.yml"

# Check if conda is installed
try {
    conda --version > $null
}
catch {
    Write-Host "Conda could not be found. Please install Conda or Miniconda and try again."
    exit 1
}

# Check if the environment already exists
$envExists = conda env list | Select-String "^$ENV_NAME\s"

if ($envExists) {
    Write-Host "Environment '$ENV_NAME' already exists. Activating it..."
    conda activate $ENV_NAME
}
else {
    # Create the Conda environment
    Write-Host "Creating Conda environment '$ENV_NAME' from '$ENV_FILE'..."
    conda env create -f $ENV_FILE
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create the Conda environment."
        exit 1
    }
    conda activate $ENV_NAME
}

# Ensure required dependencies are installed
Write-Host "Installing PyQt6, transformers, opencv-python-headless..."
pip install PyQt6 transformers opencv-python-headless

Write-Host "Installing compatible huggingface-hub version..."
pip install "huggingface-hub<0.20"

Write-Host "Installing iopaint..."
pip install iopaint

# Download the LaMA model
Write-Host "Downloading the LaMA model..."
iopaint download --model lama

# Launch the GUI
Write-Host "Launching the GUI application..."
python remwmgui.py 