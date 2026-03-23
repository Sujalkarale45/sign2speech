@echo off
echo ============================================
echo  SignVoice — Preprocessing Pipeline
echo ============================================

echo.
echo [1/3] Extracting audio clips from raw audio...
python src/preprocessing/extract_audio_clips.py
if %ERRORLEVEL% neq 0 ( echo ERROR in step 1 & pause & exit /b )

echo.
echo [2/3] Extracting MediaPipe keypoints from videos...
python src/preprocessing/extract_keypoints.py
if %ERRORLEVEL% neq 0 ( echo ERROR in step 2 & pause & exit /b )

echo.
echo [3/3] Extracting mel spectrograms from audio clips...
python src/preprocessing/extract_mels.py
if %ERRORLEVEL% neq 0 ( echo ERROR in step 3 & pause & exit /b )

echo.
echo ============================================
echo  All preprocessing complete!
echo  Next: python src/training/trainer.py
echo ============================================
pause
