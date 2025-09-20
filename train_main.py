import os
from config import DATA_DIR, CHECKPOINT_DIR
from training import train_model

def run_training():
    data_yaml = os.path.join(DATA_DIR, "data.yaml")
    model = train_model(
        data_yaml=data_yaml,
        epochs=80,
        batch_size=16,
        imgsz=640,
        save_dir=str(CHECKPOINT_DIR)
    )
    return model

if __name__ == "__main__":
    run_training()
    print("Training completed.")
