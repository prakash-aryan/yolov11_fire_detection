import os
import torch
from ultralytics import YOLO
import time
from tqdm import tqdm
import json

def train_model(dataset_path, epochs=100, batch_size=64, imgsz=640):
    # Set CUDA device
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    # Load the YOLOv11 model
    model = YOLO('yolo11n.pt')

    # Initialize lists to store loss values
    train_losses = []
    val_losses = []

    # Configure training parameters
    try:
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=0,
            workers=8,
            amp=True,
            cache=False,  # Disable caching
            patience=0,  # Disable early stopping
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            momentum=0.937,
            warmup_epochs=3,
            close_mosaic=10,
            plots=True,
            save=True,
            save_period=-1,
            verbose=False,
            single_cls=False
        )
        
        # Extract and store loss values
        for epoch in range(epochs):
            if 'train/box_loss' in results.results:
                train_losses.append({
                    'box_loss': results.results['train/box_loss'][epoch],
                    'cls_loss': results.results['train/cls_loss'][epoch],
                    'dfl_loss': results.results['train/dfl_loss'][epoch]
                })
            if 'val/box_loss' in results.results:
                val_losses.append({
                    'box_loss': results.results['val/box_loss'][epoch]
                })

        # Save the trained model
        model.save('yolo11n_fire_detection_final.pt')
        print("Training completed and model saved.")

        # Save loss values
        losses = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        with open('training_losses.json', 'w') as f:
            json.dump(losses, f)

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return None, None

    return model, results

def estimate_training_time(model, dataset_path, epochs, batch_size, imgsz):
    # Run a single iteration to estimate time per epoch
    start_time = time.time()
    try:
        model.train(data=dataset_path, epochs=1, imgsz=imgsz, batch=batch_size, plots=False, verbose=False)
    except Exception as e:
        print(f"An error occurred during time estimation: {str(e)}")
        return 0
    end_time = time.time()
    
    time_per_epoch = end_time - start_time
    total_estimated_time = time_per_epoch * epochs
    
    return total_estimated_time

if __name__ == "__main__":
    dataset_path = os.path.join('datasets', 'combined_data.yaml')
    epochs = 100
    batch_size = 64
    imgsz = 640

    # Load the model for time estimation
    model = YOLO('yolo11n.pt')
    
    # Estimate total training time
    estimated_time = estimate_training_time(model, dataset_path, epochs, batch_size, imgsz)
    print(f"Estimated total training time: {estimated_time:.2f} seconds")

    # Start actual training with progress bar
    start_time = time.time()
    with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
        trained_model, results = train_model(dataset_path, epochs=epochs, batch_size=batch_size, imgsz=imgsz)
        
        for epoch in range(epochs):
            elapsed_time = time.time() - start_time
            estimated_remaining = max(0, estimated_time - elapsed_time)
            
            pbar.set_postfix({"Remaining Time": f"{estimated_remaining:.2f}s"})
            pbar.update(1)

    if trained_model is not None:
        print(f"Actual training time: {time.time() - start_time:.2f} seconds")
    else:
        print("Training was not completed successfully.")