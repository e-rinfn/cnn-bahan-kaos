import tensorflow as tf
import numpy as np
import pickle
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
from pathlib import Path
from tkinterdnd2 import DND_FILES, TkinterDnD

class ModelTester:
    """
    Class untuk testing model Kaos (4 kelas)
    """
    
    def __init__(self, model_path='model/model_cnn.h5', class_names_path='model/class_names.pkl'):
        self.model_path = Path(model_path)
        # EfficientNet uses a TensorFlow checkpoint because Keras 2.10 cannot
        # reliably restore its weights from legacy HDF5 files.
        if self.model_path.name in {'efficientnet_kaos.h5', 'best_efficientnet_model.h5'}:
            self.model = self.build_efficientnet()
            checkpoint_path = str(self.model_path.with_suffix('.ckpt'))
            self.model.load_weights(checkpoint_path)
        else:
            self.model = tf.keras.models.load_model(str(self.model_path))

        # Load class names
        with open(class_names_path, 'rb') as f:
            self.class_mapping = pickle.load(f)
        
        self.class_names = self.class_mapping['class_names']
        print(f"✅ Model loaded. Classes: {self.class_names}")

    @staticmethod
    def build_efficientnet():
        """Build the EfficientNet architecture used during training."""
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3)
        )
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)
    
    def predict_single_image(self, image_path, img_size=None):
        """
        Predict single image menggunakan ukuran input model yang aktif.
        """
        if img_size is None:
            input_shape = self.model.input_shape
            if len(input_shape) != 4 or input_shape[1] is None or input_shape[2] is None:
                raise ValueError(f"Ukuran input model tidak didukung: {input_shape}")
            img_size = (int(input_shape[2]), int(input_shape[1]))

        img = Image.open(image_path).convert("RGB")
        img = img.resize(img_size)
        img_array = np.asarray(img, dtype=np.float32)
        if self.model_path.name == 'model_cnn.h5':
            img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction (softmax)
        prediction = self.model.predict(img_array, verbose=0)[0]

        class_index = int(np.argmax(prediction))
        class_name = self.class_names[class_index]
        confidence = float(np.max(prediction)) * 100

        return {
            'class': class_name,
            'confidence': confidence,
            'all_probabilities': prediction.tolist()
        }

class SeratKaosGUI:
    """
    GUI untuk aplikasi deteksi serat kaos dengan Tkinter (Drag & Drop)
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Deteksi Serat Kaos - AI Powered")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        
        # Warna dan style
        self.bg_color = "#f0f0f0"
        self.primary_color = "#2c3e50"
        self.secondary_color = "#3498db"
        self.success_color = "#27ae60"
        self.warning_color = "#e74c3c"
        self.info_color = "#3498db"
        
        self.root.configure(bg=self.bg_color)
        
        self.model_dir = Path(__file__).resolve().parent / 'model'
        self.available_models = self.find_models()
        self.model_tester = None
        self.model_loaded = False
        self.selected_model = tk.StringVar()

        if self.available_models:
            self.selected_model.set(next(iter(self.available_models)))
            self.load_selected_model(show_error=False)
        
        # Variable untuk menyimpan path gambar
        self.current_image_path = None
        
        # Setup UI
        self.setup_ui()
        
        # Setup drag and drop
        self.setup_drag_drop()
        
        # Update style untuk ttk
        self.setup_styles()

    def find_models(self):
        """Temukan model Keras dan pasangan file class names di folder model."""
        models = {}
        for model_path in sorted(self.model_dir.glob('*.h5')):
            if model_path.name.endswith('.weights.h5'):
                continue

            candidates = [
                self.model_dir / f'{model_path.stem.replace("_kaos", "")}_class_names.pkl',
                self.model_dir / f'{model_path.stem.replace("best_", "").replace("_model", "")}_class_names.pkl',
                self.model_dir / 'class_names.pkl'
            ]
            class_names_path = next((path for path in candidates if path.exists()), None)
            if class_names_path and class_names_path.name != 'class_names.pkl':
                models[model_path.name] = (model_path, class_names_path)
        return models

    def load_selected_model(self, show_error=True):
        """Muat model yang dipilih dari dropdown."""
        model_info = self.available_models.get(self.selected_model.get())
        if not model_info:
            self.model_loaded = False
            if show_error:
                messagebox.showerror("Error", "Tidak ada model yang dapat digunakan di folder model.")
            return

        try:
            self.model_tester = ModelTester(*model_info)
            self.model_loaded = True
            if hasattr(self, 'status_bar'):
                self.update_status(f"✓ Model aktif: {self.selected_model.get()}")
            if hasattr(self, 'prob_container'):
                self.create_probability_bars()
                self.reset()
        except Exception as error:
            self.model_loaded = False
            self.model_tester = None
            if hasattr(self, 'status_bar'):
                self.update_status(f"Model gagal dimuat: {self.selected_model.get()}")
            if show_error:
                messagebox.showerror("Error", f"Gagal memuat model:\n{str(error)}")

    def create_probability_bars(self):
        """Buat progress bar sesuai kelas dari model aktif."""
        for child in self.prob_container.winfo_children():
            child.destroy()

        colors = [self.success_color, self.warning_color, self.info_color, '#8e44ad']
        self.prob_bars = {}
        for index, class_name in enumerate(self.model_tester.class_names):
            display_name = class_name.replace('_', ' ').title()
            color = colors[index % len(colors)]
            class_frame = tk.Frame(self.prob_container, bg='white')
            class_frame.pack(fill=tk.X, padx=15, pady=10)
            header_frame = tk.Frame(class_frame, bg='white')
            header_frame.pack(fill=tk.X, pady=(0, 5))
            tk.Label(header_frame, text=display_name, font=('Arial', 10, 'bold'),
                     bg='white', fg=self.primary_color).pack(side=tk.LEFT)
            value_label = tk.Label(header_frame, text="0%", font=('Arial', 10, 'bold'),
                                   bg='white', fg=color)
            value_label.pack(side=tk.RIGHT)
            progress = ttk.Progressbar(class_frame, length=400, mode='determinate')
            progress.pack(fill=tk.X)
            self.prob_bars[class_name] = {
                'progress': progress,
                'label': value_label,
                'color': color
            }
    
    def setup_styles(self):
        """Setup style untuk ttk widgets"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', font=('Arial', 20, 'bold'), foreground=self.primary_color)
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground=self.primary_color)
        style.configure('Result.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Confidence.TLabel', font=('Arial', 12))
        
        style.configure('Upload.TButton', font=('Arial', 11, 'bold'), padding=10)
        style.map('Upload.TButton',
                  background=[('active', self.secondary_color)],
                  foreground=[('active', 'white')])
    
    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        # Register drop target untuk preview container
        self.preview_container.drop_target_register(DND_FILES)
        self.preview_container.dnd_bind('<<Drop>>', self.on_drop)
    
    def on_drop(self, event):
        """Handler untuk event drop file"""
        # Hapus kurung kurawal jika ada (Windows)
        file_path = event.data.strip('{}')
        
        # Cek apakah file adalah gambar
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        if any(file_path.lower().endswith(ext) for ext in image_extensions):
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_image(file_path)
        else:
            messagebox.showwarning("Format Tidak Didukung", 
                                 "File yang didukung: JPG, JPEG, PNG, BMP, GIF, WEBP")
    
    def setup_ui(self):
        """Setup semua komponen UI dengan layout dua kolom"""
        
        # Frame utama
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="Deteksi Kondisi Serat Kaos", 
                                style='Title.TLabel')
        title_label.pack(pady=(0, 5))
        
        subtitle_label = tk.Label(main_frame, text="Upload atau drag & drop foto serat kaos untuk mendeteksi kondisinya",
                                  font=('Arial', 10), bg=self.bg_color, fg="#666")
        subtitle_label.pack(pady=(0, 20))

        model_frame = tk.Frame(main_frame, bg=self.bg_color)
        model_frame.pack(fill=tk.X, pady=(0, 15))
        tk.Label(model_frame, text="Model yang digunakan:", font=('Arial', 10, 'bold'),
             bg=self.bg_color, fg=self.primary_color).pack(side=tk.LEFT, padx=(0, 8))
        model_values = list(self.available_models.keys())
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.selected_model,
                        values=model_values, state='readonly', width=35)
        self.model_combo.pack(side=tk.LEFT)
        self.model_combo.bind('<<ComboboxSelected>>', lambda event: self.load_selected_model())
        
        # Container untuk dua kolom
        columns_container = tk.Frame(main_frame, bg=self.bg_color)
        columns_container.pack(fill=tk.BOTH, expand=True)
        
        # === KOLOM KIRI: Preview Gambar ===
        left_column = tk.Frame(columns_container, bg=self.bg_color)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Label kolom kiri
        left_title = tk.Label(left_column, text="Preview Gambar", 
                             font=('Arial', 12, 'bold'), bg=self.bg_color)
        left_title.pack(anchor='w', pady=(0, 10))
        
        # Container untuk preview dengan drag & drop
        self.preview_container = tk.Frame(left_column, bg='white', relief=tk.GROOVE, bd=2)
        self.preview_container.pack(fill=tk.BOTH, expand=True)
        
        # Label preview
        self.preview_label = tk.Label(self.preview_container, 
                                      text="📤 Drag & Drop gambar di sini\n\natau\n\nKlik tombol di bawah",
                                      bg='white', font=('Arial', 12), fg="#999")
        self.preview_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Tombol upload alternatif
        self.upload_btn = ttk.Button(left_column, text="📁 Upload Foto dari Komputer", 
                                     command=self.upload_image, style='Upload.TButton')
        self.upload_btn.pack(pady=10)
        
        # Informasi format file
        info_label = tk.Label(left_column, text="Format didukung: JPG, PNG, BMP, GIF, WEBP",
                             font=('Arial', 8), bg=self.bg_color, fg="#888")
        info_label.pack(pady=(5, 0))
        
        # === KOLOM KANAN: Hasil Prediksi ===
        right_column = tk.Frame(columns_container, bg=self.bg_color)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Label kolom kanan
        right_title = tk.Label(right_column, text="Hasil Analisis", 
                              font=('Arial', 12, 'bold'), bg=self.bg_color)
        right_title.pack(anchor='w', pady=(0, 10))
        
        # Card hasil utama
        result_card = tk.Frame(right_column, bg='white', relief=tk.RAISED, bd=1)
        result_card.pack(fill=tk.X, pady=(0, 15))
        
        self.result_label = tk.Label(result_card, text="Belum ada gambar\nyang dianalisis",
                                     font=('Arial', 14, 'bold'), bg='white', fg="#666",
                                     justify='center')
        self.result_label.pack(pady=30, padx=20)
        
        # Frame untuk detail probabilitas
        prob_frame = tk.Frame(right_column, bg=self.bg_color)
        prob_frame.pack(fill=tk.X, pady=10)
        
        prob_label = tk.Label(prob_frame, text="Detail Probabilitas per Kelas:",
                              font=('Arial', 11, 'bold'), bg=self.bg_color)
        prob_label.pack(anchor='w', pady=(0, 10))
        
        # Container untuk progress bars
        self.prob_container = tk.Frame(prob_frame, bg='white', relief=tk.GROOVE, bd=1)
        self.prob_container.pack(fill=tk.X, pady=5)
        
        if self.model_tester:
            self.create_probability_bars()
        
        # Tombol reset
        self.reset_btn = ttk.Button(right_column, text="Reset Semua", 
                                    command=self.reset, state='disabled')
        self.reset_btn.pack(pady=15)
        
        # Progress bar untuk loading
        self.progress_frame = tk.Frame(right_column, bg=self.bg_color)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate', length=300)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="✓ Ready - Drag & drop gambar untuk memulai", 
                                   bd=1, relief=tk.SUNKEN, anchor=tk.W, 
                                   font=('Arial', 9), bg="#f0f0f0")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def upload_image(self):
        """Upload gambar dari file"""
        if not self.model_loaded:
            messagebox.showerror("Error", "Model belum dimuat. Silakan restart aplikasi.")
            return
        
        file_path = filedialog.askopenfilename(
            title="Pilih Foto Jalan",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_image(file_path)
    
    def display_image(self, image_path):
        """Menampilkan gambar yang diupload/drag"""
        try:
            # Buka dan resize gambar
            img = Image.open(image_path)
            
            # Dapatkan ukuran preview container
            self.preview_container.update_idletasks()
            container_width = self.preview_container.winfo_width()
            container_height = self.preview_container.winfo_height()
            
            # Gunakan ukuran container atau default 400x400
            preview_width = max(container_width - 40, 300)
            preview_height = max(container_height - 40, 300)
            
            # Resize gambar dengan maintain aspect ratio
            img.thumbnail((preview_width, preview_height), Image.Resampling.LANCZOS)
            
            # Konversi ke PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo  # Keep reference
            
            self.update_status(f"✓ Gambar dimuat: {os.path.basename(image_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menampilkan gambar:\n{str(e)}")
            self.update_status("✗ Error menampilkan gambar")
    
    def predict_image(self, image_path):
        """Melakukan prediksi pada gambar"""
        try:
            # Tampilkan progress bar
            self.show_progress(True)
            self.update_status("⏳ Memproses gambar dengan AI...")
            
            # Prediksi
            result = self.model_tester.predict_single_image(image_path)
            
            # Sembunyikan progress bar
            self.show_progress(False)
            
            # Tampilkan hasil
            self.display_result(result)
            self.update_status(f"✓ Prediksi selesai: {result['class']} ({result['confidence']:.1f}% confidence)")
            
            # Aktifkan tombol reset
            self.reset_btn.config(state='normal')
            
        except Exception as e:
            self.show_progress(False)
            messagebox.showerror("Error", f"Gagal melakukan prediksi:\n{str(e)}")
            self.update_status("✗ Error dalam prediksi")
    
    def display_result(self, result):
        """Menampilkan hasil prediksi"""
        class_name_id = result['class'].replace('_', ' ').title()
        confidence = result['confidence']

        color = self.prob_bars.get(result['class'], {}).get('color', self.info_color)
        # Update label hasil
        result_text = f"{class_name_id}\nConfidence: {confidence:.2f}%"
        self.result_label.config(text=result_text, fg=color, font=('Arial', 16, 'bold'))

        # Update progress bars probabilitas
        probabilities = result['all_probabilities']
        for index, class_name in enumerate(self.model_tester.class_names):
            if index >= len(probabilities) or class_name not in self.prob_bars:
                continue
            prob = probabilities[index] * 100
            self.prob_bars[class_name]['progress']['value'] = prob
            self.prob_bars[class_name]['label'].config(text=f"{prob:.1f}%")
    
    def show_progress(self, show):
        """Menampilkan atau menyembunyikan progress bar"""
        if show:
            self.progress_frame.pack(fill=tk.X, pady=10)
            self.progress_bar.pack(fill=tk.X)
            self.progress_bar.start(10)
            # Disable upload button during prediction
            self.upload_btn.config(state='disabled')
        else:
            self.progress_bar.stop()
            self.progress_frame.pack_forget()
            self.upload_btn.config(state='normal')
    
    def reset(self):
        """Reset semua tampilan"""
        # Reset preview
        self.preview_label.config(image='', 
                                 text="📤 Drag & Drop gambar di sini\n\natau\n\nKlik tombol di bawah",
                                 font=('Arial', 12), fg="#999")
        self.preview_label.image = None
        
        # Reset result label
        self.result_label.config(text="Belum ada gambar\nyang dianalisis", 
                                fg="#666", font=('Arial', 14, 'bold'))
        
        # Reset progress bars
        for cls_name, bar_data in self.prob_bars.items():
            bar_data['progress']['value'] = 0
            bar_data['label'].config(text="0%")
        
        # Reset variables
        self.current_image_path = None
        
        # Disable reset button
        self.reset_btn.config(state='disabled')
        
        # Update status
        self.update_status("✓ Reset selesai. Siap untuk drag & drop atau upload gambar baru.")

# ======================================================================
# DUMMY MODEL UNTUK DEVELOPMENT (4 KELAS)
# ======================================================================

def create_dummy_model():
    """
    Membuat dummy 4-class model (softmax)
    """
    print("🔧 Creating dummy 4-class model...")
    
    # Buat direktori model jika belum ada
    Path("model").mkdir(exist_ok=True)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 classes Serat Kaos
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save('model/model_cnn.h5')
    
    print("✅ Dummy model saved to 'model/model_cnn.h5'")

    # Class mapping
    class_mapping = {
        'class_names': ['teteron_cotton', 'chief_value_cotton', 'polyester', 'cotton_combed'],
        'class_indices': {
            'teteron_cotton': 0,
            'chief_value_cotton': 1,
            'polyester': 2,
            'cotton_combed': 3
        },
        'input_shape': (150, 150, 3),
        'classes_count': 4,
        'note': 'Dummy softmax model for development'
    }

    with open('model/class_names.pkl', 'wb') as f:
        pickle.dump(class_mapping, f)
    
    print("✅ Class names saved to 'model/class_names.pkl'")

# ======================================================================
# MAIN EXECUTION
# ======================================================================

if __name__ == "__main__":
    # Cek apakah model ada, jika tidak buat dummy model
    if not Path("model/model_cnn.h5").exists():
        create_dummy_model()
    
    # Buat folder test_samples jika diperlukan untuk testing
    Path("test_samples").mkdir(exist_ok=True)
    
    # Jalankan GUI dengan dukungan drag and drop
    # Catatan: Perlu install tkinterdnd2
    # pip install tkinterdnd2
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        print("⚠️ tkinterdnd2 tidak terinstall. Drag & drop tidak akan berfungsi.")
        print("Install dengan: pip install tkinterdnd2")
        root = tk.Tk()
    
    app = SeratKaosGUI(root)
    root.mainloop()