"""
Test weapons detection model on live camera feed.
Press 'q' to quit, 's' to save screenshot of detections.
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import argparse

def test_camera(
    model_path="runs/detect/runs/weapons-merged-enhanced/weights/best.pt",
    source=0,  # 0 for webcam, or RTSP URL
    conf_threshold=0.25,
    show_fps=True,
    save_dir="runs/camera_test"
):
    """Test model on camera/video stream."""
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Open video source
    print(f"Opening camera/stream: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video: {width}x{height} @ {fps} fps")
    print(f"Confidence threshold: {conf_threshold}")
    print("\nPress 'q' to quit, 's' to save screenshot")
    print("-" * 60)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Stats
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    fps_display = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or read error")
                break
            
            frame_count += 1
            frame_start = time.time()
            
            # Run detection
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False,
                device=0  # Use GPU
            )
            
            # Get annotated frame
            annotated = results[0].plot()
            
            # Count detections
            boxes = results[0].boxes
            num_detections = len(boxes)
            if num_detections > 0:
                detection_count += 1
            
            # Calculate FPS
            if show_fps:
                frame_time = time.time() - frame_start
                fps_display = 1.0 / frame_time if frame_time > 0 else 0
                
                # Add FPS and detection info to frame
                info_text = f"FPS: {fps_display:.1f} | Detections: {num_detections}"
                cv2.putText(
                    annotated,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            
            # Display detailed detection info
            if num_detections > 0:
                y_offset = 60
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    
                    det_text = f"{cls_name}: {conf:.2f}"
                    cv2.putText(
                        annotated,
                        det_text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )
                    y_offset += 25
                    
                    print(f"[Frame {frame_count}] Detected: {cls_name} ({conf:.2%})")
            
            # Show frame
            cv2.imshow("Weapons Detection - Camera Test", annotated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_path = save_path / f"detection_{frame_count}.jpg"
                cv2.imwrite(str(screenshot_path), annotated)
                print(f"Screenshot saved: {screenshot_path}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Frames with detections: {detection_count}")
        print(f"Detection rate: {detection_count/frame_count*100:.1f}%")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Total time: {elapsed:.1f}s")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test weapons detection on camera")
    parser.add_argument(
        "--model",
        default="runs/detect/runs/weapons-merged-enhanced/weights/best.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: 0 for webcam, RTSP URL, or video file path"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0-1)"
    )
    parser.add_argument(
        "--save-dir",
        default="runs/camera_test",
        help="Directory to save screenshots"
    )
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit (webcam index)
    source = int(args.source) if args.source.isdigit() else args.source
    
    print("=" * 60)
    print("BANTAI Weapons Detection - Camera Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Source: {source}")
    print(f"Confidence: {args.conf}")
    print("=" * 60)
    
    test_camera(
        model_path=args.model,
        source=source,
        conf_threshold=args.conf,
        save_dir=args.save_dir
    )
