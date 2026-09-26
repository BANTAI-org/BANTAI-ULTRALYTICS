from bantai_ml.inference.detect import detect_objects


if __name__ == "__main__":
    detect_objects("runs/train/weights/best.pt", "data/raw")
