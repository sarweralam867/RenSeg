import os
from config import DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR
from inference import load_model, set_paths, process_images_in_folder

def run_inference():
    weights_path = os.path.join(CHECKPOINT_DIR, "train", "weights", "best.pt")
    load_model(weights_path)
    set_paths(str(DATA_DIR / "test"), str(OUTPUT_DIR))
    process_images_in_folder(str(DATA_DIR / "test"))

if __name__ == "__main__":
    run_inference()
    print("Segmentation completed.")
