import pandas as pd
import numpy as np
from ultralytics import YOLO
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, dataset_path):
    # Load the best model from training
    model = YOLO(model_path)

    # Perform final evaluation
    metrics = model.val(data=dataset_path)

    # Print final metrics
    print("Final Evaluation Results:")
    print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")
    print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics.results_dict])
    metrics_df.to_csv('final_metrics.csv', index=False)
    
    # Extract and save confusion matrix
    conf_matrix = metrics.confusion_matrix.matrix
    np.save('confusion_matrix.npy', conf_matrix)
    
    # Save class names
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(metrics.names, f)
    
    print("Final evaluation completed. Metrics and confusion matrix saved.")

    # Plot confusion matrix
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("Confusion matrix plot saved as confusion_matrix.png")
    except Exception as e:
        print(f"Error plotting confusion matrix: {str(e)}")

    # Plot training results if available
    results_csv = 'runs/detect/train2/results.csv'
    try:
        results = pd.read_csv(results_csv)
        metrics_to_plot = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 
                           'val/box_loss', 'metrics/precision(B)', 'metrics/recall(B)', 
                           'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        
        fig, axs = plt.subplots(4, 2, figsize=(20, 30))
        fig.suptitle('Training and Validation Metrics', fontsize=16)
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i // 2, i % 2]
            ax.plot(results['epoch'], results[metric])
            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        print("Training metrics plot saved as training_metrics.png")
    except FileNotFoundError:
        print(f"Training results file not found at {results_csv}")
    except Exception as e:
        print(f"Error plotting training metrics: {str(e)}")

    return metrics

if __name__ == "__main__":
    model_path = 'runs/detect/train2/weights/best.pt'  # Updated path to your newly trained model
    dataset_path = 'datasets/combined_data.yaml'
    final_metrics = evaluate_model(model_path, dataset_path)