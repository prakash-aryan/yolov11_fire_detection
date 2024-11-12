import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cv2
import pickle
import json

def plot_learning_curves(losses_file):
    with open(losses_file, 'r') as f:
        losses = json.load(f)

    plt.figure(figsize=(12, 8))
    plt.plot(losses['train_box_loss'], label='train_box_loss')
    plt.plot(losses['train_cls_loss'], label='train_cls_loss')
    plt.plot(losses['train_dfl_loss'], label='train_dfl_loss')
    plt.plot(losses['val_box_loss'], label='val_box_loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('learning_curves.png')
    plt.close()

def plot_confusion_matrix(conf_matrix_file, class_names_file):
    conf_matrix = np.load(conf_matrix_file)
    with open(class_names_file, 'rb') as f:
        class_names = pickle.load(f)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def visualize_predictions(model_path, test_dir, num_images=5):
    model = YOLO(model_path)
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    for i, img_file in enumerate(image_files[:num_images]):
        img_path = os.path.join(test_dir, img_file)
        results = model(img_path)
        
        im_array = results[0].plot()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Test Image {i+1}')
        plt.tight_layout()
        plt.savefig(f'prediction_{i+1}.png')
        plt.close()

if __name__ == "__main__":
    losses_file = 'training_losses.json'
    if os.path.exists(losses_file):
        plot_learning_curves(losses_file)
    else:
        print(f"Warning: {losses_file} not found. Skipping learning curve plot.")

    conf_matrix_file = 'confusion_matrix.npy'
    class_names_file = 'class_names.pkl'
    if os.path.exists(conf_matrix_file) and os.path.exists(class_names_file):
        plot_confusion_matrix(conf_matrix_file, class_names_file)
    else:
        print(f"Warning: {conf_matrix_file} or {class_names_file} not found. Skipping confusion matrix plot.")

    model_path = 'runs/detect/train/weights/best.pt'
    test_dir = 'datasets/test/images'
    if os.path.exists(model_path) and os.path.exists(test_dir):
        visualize_predictions(model_path, test_dir)
    else:
        print(f"Warning: {model_path} or {test_dir} not found. Skipping prediction visualization.")

print("Visualization process completed.")