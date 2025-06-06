import os
import csv

def generate_labels_csv(dataset_dir, output_csv):
    """
    Generate CSV file with image paths and class labels.
    """
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    rows = []

    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        for img in images:
            img_path = os.path.join(cls, img)
            rows.append([img_path, cls])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label'])
        writer.writerows(rows)

    print(f"Labels CSV saved to {output_csv}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate CSV labels file for dataset')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_csv', type=str, required=True, help='Path for output CSV file')

    args = parser.parse_args()
    generate_labels_csv(args.dataset_dir, args.output_csv)
