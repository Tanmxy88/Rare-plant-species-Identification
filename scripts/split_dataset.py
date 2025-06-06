import os
import shutil
import random
import argparse

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.01, "Ratios must sum to 1"

    classes = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    for cls in classes:
        class_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(class_path) 
                  if os.path.isfile(os.path.join(class_path, f))]
        
        random.shuffle(images)
        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        splits = {
            'train': images[:n_train],
            'validation': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }

        for split, files in splits.items():
            split_dir = os.path.join(dest_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            
            for file in files:
                src = os.path.join(class_path, file)
                dst = os.path.join(split_dir, file)
                shutil.copy2(src, dst)
    
    print(f"✅ Dataset split into {dest_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument('--source', required=True, help='Source dataset directory')
    parser.add_argument('--dest', required=True, help='Destination directory for splits')
    parser.add_argument('--train', type=float, default=0.7, help='Train ratio')
    parser.add_argument('--val', type=float, default=0.15, help='Validation ratio')
    parser.add_argument('--test', type=float, default=0.15, help='Test ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    split_dataset(
        args.source, 
        args.dest,
        args.train,
        args.val,
        args.test,
        args.seed
    )