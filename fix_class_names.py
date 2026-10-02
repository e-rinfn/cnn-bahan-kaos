# fix_class_names.py
import pickle
from datetime import datetime
from pathlib import Path

def fix_class_names():
    """Fix and update class names mapping for 4-class fabric classification"""

    # 4 kelas bahan kaos; urutan harus sama dengan folder dataset.
    class_names = [
        'cc',
        'cvc',
        'polyester',
        'tc',
    ]

    # Buat folder model jika belum ada
    Path("model").mkdir(exist_ok=True)

    # Mapping class index otomatis
    class_indices = {name: i for i, name in enumerate(class_names)}

    class_mapping = {
        'class_names': class_names,
        'class_indices': class_indices,
        'timestamp': datetime.now().isoformat(),
        'input_shape': (150, 150, 3),
        'classes_count': len(class_names),
        'note': 'Fixed mapping for 4-class Bahan Kaos model'
    }

    # Save mapping
    with open('model/class_names.pkl', 'wb') as f:
        pickle.dump(class_mapping, f)

    print("✅ Fixed class_names.pkl successfully!")
    print("📌 Class order:")
    for i, name in enumerate(class_names):
        print(f"   {i} → {name}")

if __name__ == "__main__":
    fix_class_names()
