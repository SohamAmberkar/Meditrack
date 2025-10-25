from ultralytics import YOLO
import cv2
import os

def main():
    # --- UPDATE THESE PATHS ---
    
    # 1. Path to your trained model weights.
    #    (Look inside your 'runs/detect/' folder)
    MODEL_PATH = "runs/detect/train9/weights/best.pt"

    # 2. Path to the single image you want to test.
    IMAGE_PATH = "test_image/1.jpeg"
    
    # --- End of paths ---


    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please find your 'best.pt' file and update the MODEL_PATH variable.")
        return

    # Check if the image file exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
        print("Please make sure your image exists and the IMAGE_PATH is correct.")
        return

    # Load your trained YOLO model
    model = YOLO(MODEL_PATH)

    # Read the image using OpenCV
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Error: Could not read image from {IMAGE_PATH}")
        return

    # Run inference on the single image
    # 'stream=True' is not needed for a single image
    results = model(frame)

    # Get the first result (since we only have one image)
    # The .plot() method automatically draws all bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display the image with detections
    cv2.imshow("Detection Result - Press 'q' to Quit", annotated_frame)

    # Wait until the 'q' key is pressed to close the window
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up and close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()