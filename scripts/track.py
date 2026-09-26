from bantai_ml.inference.track import track_objects


if __name__ == "__main__":
    track_objects("runs/train/weights/best.pt", "data/raw/video.mp4")
