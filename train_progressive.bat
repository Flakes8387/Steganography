@echo off
REM Quick Start Script for Progressive Training
REM Windows Batch Script

echo ================================================================================
echo   PROGRESSIVE TRAINING - QUICK START
echo   Target: 90%% Accuracy on All Attacks
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install requirements
echo Checking dependencies...
pip install -q torch torchvision pillow pyyaml tensorboard tqdm numpy matplotlib
echo Dependencies OK
echo.

REM Check for dataset
if not exist "data\DIV2K\train\" (
    echo.
    echo WARNING: DIV2K dataset not found at data\DIV2K\train\
    echo.
    echo Please download the dataset first:
    echo   python download_div2k.py
    echo.
    pause
    exit /b 1
)

echo ================================================================================
echo   STARTING PROGRESSIVE TRAINING
echo ================================================================================
echo.
echo Configuration:
echo   - Dataset: data\DIV2K\train (400 images)
echo   - Message Length: 16 bits
echo   - Image Size: 128x128
echo   - Batch Size: 8
echo   - Learning Rate: 1e-4 to 5e-3 (Cyclical)
echo   - Phases: 6 (Clean, JPEG, Blur, Resize, ColorJitter, Combined)
echo.
echo Estimated Training Time: 10-15 hours on GPU
echo.
echo Press Ctrl+C to stop training
echo.
pause

REM Run training
python train_progressive_90plus.py ^
    --train_dir data/DIV2K/train ^
    --max_images 400 ^
    --message_length 16 ^
    --image_size 128 ^
    --batch_size 8 ^
    --min_epochs_per_phase 20 ^
    --max_epochs_per_phase 100 ^
    --base_lr 1e-4 ^
    --max_lr 5e-3 ^
    --cyclic_step_size 200 ^
    --checkpoint_dir checkpoints ^
    --log_dir runs

echo.
echo ================================================================================
echo   TRAINING COMPLETE!
echo ================================================================================
echo.
echo Checkpoints saved in: checkpoints\
echo   - model_phase1_clean_94.pth
echo   - model_phase2_jpeg_90.pth
echo   - model_phase3_blur_90.pth
echo   - model_phase4_resize_90.pth
echo   - model_phase5_all_90.pth
echo   - model_phase6_combined_88.pth (FINAL MODEL)
echo.
echo To evaluate the models:
echo   python evaluate_progressive_models.py
echo.
echo To view training logs:
echo   tensorboard --logdir runs
echo.
pause
