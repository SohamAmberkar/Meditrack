from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 model (e.g., yolov8n.pt for a small, fast model)
    model = YOLO("yolov8n.pt")

    # Train the model
    print("Starting model training...")
    results = model.train(
        data="data.yaml",  # Path to your dataset configuration file
        epochs=100,        # Number of training epochs
        imgsz=640,         # Image size for training
        batch=16,          # Batch size
        name="yolo_counterfeit_detection" # Name for the 'runs' folder
    )
    print("Training complete.")
    print(f"Model saved to: {results.save_dir}")

if __name__ == "__main__":
    main()