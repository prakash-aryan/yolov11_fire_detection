import os
import yaml
from ultralytics import YOLO

def prepare_combined_dataset():
    combined_data = {
        'path': os.path.abspath('datasets'),
        'train': 'train',
        'val': 'valid',
        'test': 'test',
        'nc': 0,
        'names': []
    }

    for yaml_file in os.listdir('datasets'):
        if yaml_file.startswith('data_') and yaml_file.endswith('.yaml'):
            with open(os.path.join('datasets', yaml_file), 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    if isinstance(data['names'], dict):
                        combined_data['names'].extend(data['names'].values())
                    elif isinstance(data['names'], list):
                        combined_data['names'].extend(data['names'])

    combined_data['names'] = list(dict.fromkeys(combined_data['names']))
    combined_data['nc'] = len(combined_data['names'])

    combined_yaml_path = os.path.join('datasets', 'combined_data.yaml')
    with open(combined_yaml_path, 'w') as f:
        yaml.dump(combined_data, f)

    print(f"Combined dataset YAML created at {combined_yaml_path}")
    print("Class names:", combined_data['names'])
    print("Number of classes:", combined_data['nc'])

    return combined_yaml_path

def verify_dataset(dataset_path):
    if os.path.exists(dataset_path):
        print(f"Dataset file found: {dataset_path}")
        with open(dataset_path, 'r') as f:
            dataset_content = yaml.safe_load(f)
            print("Dataset content:", dataset_content)
    else:
        print(f"Warning: Dataset file not found at {dataset_path}")

if __name__ == "__main__":
    combined_yaml_path = prepare_combined_dataset()
    verify_dataset(combined_yaml_path)