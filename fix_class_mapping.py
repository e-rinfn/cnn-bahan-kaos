# fix_class_mapping.py
import pickle
import os
from datetime import datetime

CLASS_MAPPING_PATH = "model/class_names.pkl"


def create_directory_if_needed():
    """Ensure model/ directory exists"""
    model_dir = os.path.dirname(CLASS_MAPPING_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 Folder '{model_dir}' dibuat.")


def fix_class_mapping():
    """Replace class_names.pkl with correct mapping for 4-class kaos model"""

    create_directory_if_needed()

    # Mapping final yang benar
    class_mapping = {
        'class_names': ['cc', 'cvc', 'polyester', 'tc'],
        'class_indices': {
            'cc': 0,
            'cvc': 1,
            'polyester': 2,
            'tc': 3
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input_shape': (150, 150, 3),
        'classes_count': 4,
        'note': 'AUTO-FIX: Updated for 4-class Bahan Kaos model'
    }

    with open(CLASS_MAPPING_PATH, "wb") as f:
        pickle.dump(class_mapping, f)

    print("\n✅ CLASS MAPPING SUCCESSFULLY FIXED")
    print("   -------------------------------")
    for idx, name in enumerate(class_mapping['class_names']):
        print(f"   Index {idx}: {name}")
    print(f"\n   Total classes: {len(class_mapping['class_names'])}")
    print(f"   Saved to: {CLASS_MAPPING_PATH}")


def check_current_mapping():
    """Read and print existing class mapping"""

    if not os.path.exists(CLASS_MAPPING_PATH):
        print(f"❌ File mapping tidak ditemukan: {CLASS_MAPPING_PATH}")
        print("   Jalankan `fix_class_mapping()` untuk membuat yang baru.")
        return

    try:
        with open(CLASS_MAPPING_PATH, "rb") as f:
            current = pickle.load(f)

        print("\n🔍 CURRENT CLASS MAPPING")
        print("-------------------------")
        print(f"Class names: {current.get('class_names')}")
        print(f"Class indices: {current.get('class_indices')}")
        print(f"Total classes: {current.get('classes_count')}")
        print(f"Input shape: {current.get('input_shape')}")
        print(f"Timestamp: {current.get('timestamp')}")
        print(f"Note: {current.get('note')}")
    except Exception as e:
        print(f"❌ Error membaca mapping: {e}")


if __name__ == "__main__":
    # Jalankan cek dulu
    check_current_mapping()

    print("\n" + "=" * 60)

    # Baru perbaiki
    fix_class_mapping()
