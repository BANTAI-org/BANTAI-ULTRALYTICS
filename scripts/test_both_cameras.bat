@echo off
REM Test both pipeline streams simultaneously

start "Stream 1" python scripts/test_camera.py --source "rtsp://192.168.1.101:8554/stream" --save-dir "runs/camera_test/stream1"
start "Stream 2" python scripts/test_camera.py --source "rtsp://192.168.1.102:8554/stream" --save-dir "runs/camera_test/stream2"

echo Testing both pipeline streams...
echo Close the windows to stop testing
