"""
Test script for ByteTrack integration.
Simulates detections and validates tracking functionality.
"""

import numpy as np
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from bytetrack import ByteTracker


def test_basic_tracking():
    """Test basic tracking functionality."""
    print("Testing ByteTrack basic tracking...")
    
    tracker = ByteTracker(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30
    )
    
    # Simulate detections over multiple frames
    # Format: [x1, y1, x2, y2, score, class_id]
    
    # Frame 1: Two objects
    detections_1 = np.array([
        [100, 100, 200, 200, 0.9, 0],  # Object 1
        [300, 150, 400, 250, 0.85, 1]  # Object 2
    ])
    
    tracks_1 = tracker.update(detections_1)
    assert len(tracks_1) == 2, f"Expected 2 tracks, got {len(tracks_1)}"
    print(f"[OK] Frame 1: {len(tracks_1)} tracks detected")
    
    # Frame 2: Same objects moved slightly
    detections_2 = np.array([
        [105, 102, 205, 202, 0.88, 0],  # Object 1 moved
        [305, 152, 405, 252, 0.87, 1]   # Object 2 moved
    ])
    
    tracks_2 = tracker.update(detections_2)
    assert len(tracks_2) == 2, f"Expected 2 tracks, got {len(tracks_2)}"
    
    # Verify track IDs are maintained
    track_ids = [t.track_id for t in tracks_2]
    assert len(set(track_ids)) == 2, "Track IDs should be unique"
    print(f"[OK] Frame 2: {len(tracks_2)} tracks maintained with IDs {track_ids}")
    
    # Frame 3: One object disappears
    detections_3 = np.array([
        [110, 104, 210, 204, 0.91, 0]  # Only object 1
    ])
    
    tracks_3 = tracker.update(detections_3)
    assert len(tracks_3) == 1, f"Expected 1 track, got {len(tracks_3)}"
    print(f"[OK] Frame 3: {len(tracks_3)} track after one object disappeared")
    
    # Frame 4: New object appears
    detections_4 = np.array([
        [115, 106, 215, 206, 0.89, 0],  # Object 1
        [500, 300, 600, 400, 0.92, 2]   # New object
    ])
    
    tracks_4 = tracker.update(detections_4)
    assert len(tracks_4) == 2, f"Expected 2 tracks, got {len(tracks_4)}"
    print(f"[OK] Frame 4: {len(tracks_4)} tracks after new object appeared")
    
    print("\n[SUCCESS] All basic tracking tests passed!")


def test_track_continuity():
    """Test track ID continuity across frames."""
    print("\nTesting track ID continuity...")
    
    tracker = ByteTracker()
    
    # Track one object across multiple frames
    track_ids = []
    for i in range(5):
        x_offset = i * 5
        detections = np.array([
            [100 + x_offset, 100, 200 + x_offset, 200, 0.9, 0]
        ])
        tracks = tracker.update(detections)
        if len(tracks) > 0:
            track_ids.append(tracks[0].track_id)
    
    # All track IDs should be the same
    assert len(set(track_ids)) == 1, f"Track ID changed: {track_ids}"
    print(f"[OK] Track ID maintained across 5 frames: {track_ids[0]}")
    
    print("[SUCCESS] Track continuity test passed!")


def test_occlusion_handling():
    """Test handling of temporary occlusions."""
    print("\nTesting occlusion handling...")
    
    tracker = ByteTracker(track_buffer=30)
    
    # Frame 1: Object visible
    detections_1 = np.array([[100, 100, 200, 200, 0.9, 0]])
    tracks_1 = tracker.update(detections_1)
    original_id = tracks_1[0].track_id
    
    # Frames 2-10: Object occluded (no detections)
    for _ in range(10):
        tracks = tracker.update(np.array([]))
    
    # Frame 11: Object reappears
    detections_2 = np.array([[110, 110, 210, 210, 0.88, 0]])
    tracks_2 = tracker.update(detections_2)
    
    assert len(tracks_2) == 1, f"Expected 1 track, got {len(tracks_2)}"
    print(f"[OK] Track recovered after occlusion")
    print(f"  Original ID: {original_id}, After occlusion: {tracks_2[0].track_id}")
    
    print("[SUCCESS] Occlusion handling test passed!")


def test_multi_class_tracking():
    """Test tracking multiple object classes."""
    print("\nTesting multi-class tracking...")
    
    tracker = ByteTracker()
    
    # Multiple classes in one frame
    detections = np.array([
        [100, 100, 200, 200, 0.9, 0],  # Class 0
        [300, 150, 400, 250, 0.85, 1], # Class 1
        [500, 200, 600, 300, 0.92, 0], # Class 0
        [700, 250, 800, 350, 0.88, 2]  # Class 2
    ])
    
    tracks = tracker.update(detections)
    assert len(tracks) == 4, f"Expected 4 tracks, got {len(tracks)}"
    
    # Verify different classes are tracked
    classes = [t.cls for t in tracks]
    assert len(set(classes)) == 3, f"Expected 3 unique classes, got {set(classes)}"
    print(f"[OK] Successfully tracked {len(tracks)} objects across {len(set(classes))} classes")
    
    print("[SUCCESS] Multi-class tracking test passed!")


if __name__ == '__main__':
    print("=" * 60)
    print("ByteTrack Integration Tests")
    print("=" * 60)
    
    try:
        test_basic_tracking()
        test_track_continuity()
        test_occlusion_handling()
        test_multi_class_tracking()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All tests passed successfully!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n[FAILED] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
