import cv2
from ultralytics import YOLO
import os

def main():
    # --- IMPORTANT ---
    # 1. Find your 'best.pt' file inside your 'runs' folder.
    # 2. Copy the path to it and paste it below.
    #    It will look something like: 'runs/detect/train9/weights/best.pt'
    MODEL_PATH = "runs/detect/train9/weights/best.pt" # <--- UPDATE THIS PATH

    # 2. Choose your detection source
    #    To use your webcam:
    SOURCE = 0
    #    To use an image from your test_image folder:
    # SOURCE = "test_image/some_image.jpg" # <--- UPDATE THIS PATH

    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please find your 'best.pt' file in the 'runs' folder and update MODEL_PATH in this script.")
        return

    # Load your trained model
    model = YOLO(MODEL_PATH)

    print("Model loaded. Starting detection...")
    print("Press 'q' to quit.")

    # Run detection
    # 'stream=True' is used for video feeds like webcams
    results = model(SOURCE, stream=True, show=True)

    # Loop through the results (this will run until you press 'q' or the video ends)
    for r in results:
        # 'plot()' draws the bounding boxes and labels on the frame
        annotated_frame = r.plot()
        
        # Display the frame (if you are using a source that isn't show=True)
        # cv2.imshow("Counterfeit Detection", annotated_frame)
        
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break

    # cv2.destroyAllWindows()
    print("Detection finished.")


if __name__ == "__main__":
    main()