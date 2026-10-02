import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['cc', 'cvc', 'polyester', 'tc']

def prepare_dataset(raw_data_dir='raw_data', output_dir='dataset', test_size=0.15, val_size=0.15):
    """
    Menyiapkan dataset dari folder raw data ke struktur train/val/test
    """
    
    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)
    
    # Buat direktori output
    for split in ['train', 'validation', 'test']:
        for class_name in CLASS_NAMES:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process setiap kelas
    for class_name in CLASS_NAMES:
        class_path = raw_path / class_name
        
        if not class_path.exists():
            print(f"⚠️ Warning: Directory {class_path} tidak ditemukan")
            continue
            
        # Dapatkan semua file gambar
        image_files = [f for f in class_path.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        
        print(f"📁 Processing {class_name}: {len(image_files)} images")
        
        if len(image_files) == 0:
            print(f"⚠️ No images found in {class_path}")
            continue
        
        # Split data: train -> 70%, validation -> 15%, test -> 15%
        train_files, test_files = train_test_split(
            image_files, test_size=test_size, random_state=42
        )
        
        train_files, val_files = train_test_split(
            train_files, test_size=val_size/(1-test_size), random_state=42
        )
        
        # Copy files ke direktori yang sesuai
        for file in train_files:
            shutil.copy2(file, output_path / 'train' / class_name / file.name)
        
        for file in val_files:
            shutil.copy2(file, output_path / 'validation' / class_name / file.name)
        
        for file in test_files:
            shutil.copy2(file, output_path / 'test' / class_name / file.name)
        
        print(f"✅ {class_name}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    print("🎉 Dataset preparation completed!")

def analyze_dataset(dataset_dir='dataset'):
    """
    Menganalisis distribusi dataset
    """
    dataset_path = Path(dataset_dir)
    
    print("📊 Dataset Analysis:")
    print("-" * 40)
    
    for split in ['train', 'validation', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            print(f"\n{split.upper()}:")
            for class_name in CLASS_NAMES:
                class_path = split_path / class_name
                if class_path.exists():
                    count = len(list(class_path.glob('*.*')))
                    print(f"  {class_name}: {count} images")

if __name__ == "__main__":
    # Siapkan dataset
    prepare_dataset(
        raw_data_dir='raw_data',
        output_dir='dataset',
        test_size=0.15,
        val_size=0.15
    )
    
    # Analisis dataset
    analyze_dataset('dataset')