---

## KLASIFIKASI DATA CITRA 
Perbandingan hasil prediksi menggunakan model CNN, MobileNetV2, dan ResNet101 pada Klasifikasi Gambar Pose Yoga.

---

## DESKRIPSI PROJECT 
Proyek ini bertujuan untuk melakukan klasifikasi data citra pose yoga menggunakan beberapa model deep learning untuk mengenali berbagai jenis pose yoga dari gambar. Dataset yang digunakan diambil dari Kaggle dan terdiri dari 5.991 gambar dengan 107 kelas pose yoga yang berbeda. Setiap kelas memiliki jumlah gambar yang bervariasi, sehingga distribusi data bersifat tidak seimbang.

Pada penelitian ini digunakan tiga model klasifikasi citra, yaitu:
1.	Non-pretrained (CNN), yaitu model CNN yang dibuat dan dilatih dari awal tanpa menggunakan bobot awal (pretrained).
2.	Pretrained 1 – MobileNetV2, yaitu model CNN yang sudah pernah dilatih sebelumnya pada dataset besar, sehingga mampu mengenali pola dasar pada gambar. Model ini bersifat ringan dan cepat, lalu disesuaikan kembali untuk mengenali pose yoga.
3.	Pretrained 2 – ResNet101, yaitu model CNN yang sudah dilatih sebelumnya dengan struktur jaringan yang lebih dalam. Model ini mampu menangkap fitur gambar yang lebih kompleks, sehingga diharapkan memberikan hasil klasifikasi yang lebih baik.

---

## 📊 Dataset
Dataset pose yoga diperoleh dari **Kaggle** dan memiliki karakteristik sebagai berikut:
- Total gambar: **5.991**
- Jumlah kelas: **107 pose yoga**
- Distribusi data antar kelas **tidak seimbang**
  
Dataset yang digunakan diambil dari [Kaggle](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset)

Pada dataset ini, terdapat perbedaan jumlah data antar kelas menunjukkan bahwa dataset memiliki distribusi yang tidak seimbang (imbalanced), yang berpotensi memengaruhi performa model dalam mengenali kelas dengan jumlah data yang lebih sedikit.
Berikut Contoh 10 kelas pertama pada dataset:

| No | Nama Pose Yoga             | Jumlah Gambar |
|----|----------------------------|---------------|
| 1  | adho mukha svanasana        | 69            |
| 2  | adho mukha vriksasana       | 59            |
| 3  | agnistambhasana             | 33            |
| 4  | ananda balasana             | 59            |
| 5  | anantasana                  | 43            |
| 6  | anjaneyasana                | 64            |
| 7  | ardha bhekasana             | 40            |
| 8  | ardha chandrasana           | 52            |
| 9  | ardha matsyendrasana        | 90            |
| 10 | ardha pincha mayurasana     | 47            |

Ketidakseimbangan data ini dapat memengaruhi performa model, khususnya pada kelas dengan jumlah data yang sedikit.

---

## ⚙️ Preprocessing Data
Sebelum digunakan untuk melatih model, data citra akan melakukan tahap pra-pemrosesan (preprocessing). Tahap ini dilakukan untuk menyamakan ukuran gambar, memperbaiki kualitas data, serta membantu model agar dapat mengenali pola pose yoga dengan lebih baik.
Tahapan preprocessing dilakukan untuk meningkatkan kualitas data dan performa model, meliputi:

1. **Resize Gambar**  
   Semua gambar diubah ke ukuran **224 × 224 piksel** agar sesuai dengan input model CNN dan model pretrained.

2. **Normalisasi Piksel**  
   Nilai piksel dinormalisasi untuk mempercepat proses pelatihan dan meningkatkan kestabilan model.

3. **Augmentasi Data**  
   Untuk mengatasi keterbatasan data dan ketidakseimbangan kelas, dilakukan augmentasi seperti:
   - Rotasi
   - Flip
   - Zoom

4. **Pembagian Dataset**  
   Dataset dibagi menjadi data **training**, **validation**, dan **testing**. Pembagian ini dilakukan untuk melatih model serta mengevaluasi performa model secara objektif.

---

## 🧠 Model yang Digunakan
Terdapat tiga model yang digunakan untuk melakukan klasifikasi citra pose yoga. Ketiga model tersebut terdiri dari satu model Non-Pretrained, yaitu CNN, serta dua model Pretrained, yaitu MobileNetV2 dan ResNet101. Setiap model dilatih dan dievaluasi dengan menampilkan hasil berupa Akurasi, Grafik, Confusion Matrix, dan Classification Report. Setelah proses pelatihan selesai, model disimpan dari Google Colab agar dapat digunakan kembali pada aplikasi Streamlit sebagai dashboard interaktif.
Tiga model deep learning digunakan dan dibandingkan dalam proyek ini:

### 1️⃣ CNN (Non-Pretrained)
Model CNN dibangun dan dilatih dari awal tanpa menggunakan bobot pretrained. Model ini belajar mengenali fitur citra pose yoga langsung dari dataset.

### 2️⃣ MobileNetV2 (Pretrained)
MobileNetV2 merupakan model pretrained yang telah dilatih pada dataset berskala besar. Model ini bersifat ringan dan efisien, kemudian dilakukan **fine-tuning** agar sesuai dengan klasifikasi pose yoga sebanyak 107 kelas.

### 3️⃣ ResNet101 (Pretrained)
ResNet101 memiliki arsitektur jaringan yang lebih dalam dan mampu mengekstraksi fitur citra yang lebih kompleks. Model ini juga dilakukan fine-tuning untuk klasifikasi pose yoga.

Setelah pelatihan selesai, masing-masing model disimpan agar dapat digunakan kembali pada aplikasi web tanpa training ulang.

---

## ALUR DASHBOARD WEB
Aplikasi dashboard web ini digunakan untuk melakukan proses prediksi pose yoga secara interaktif dan real-time. Tanpa perlu melakukan training ulang, model dapat langsung dipilih, gambar diunggah, serta hasil prediksi beserta gambar yang diupload dapat langsung dilihat.

Berikut adalah langkah-langkah dalam menggunakan dashboard web:
1.	Menampilkan model klasifikasi citra beserta implementasi masing-masing model yang tertera pada halaman beranda (home), yaitu CNN, MobileNetV2, dan ResNet101.
2.	Memilih model yang ingin digunakan untuk melakukan prediksi.
3.	Mengunggah gambar, kemudian gambar yang dipilih akan tampil pada halaman dashboard sebagai preview.
4.	Model melakukan prediksi berdasarkan gambar tersebut, serta menampilkan akurasi dan probabilitas top 5 untuk setiap gambar.
5.	Setelah prediksi selesai, dapat kembali ke halaman beranda dan memilih model lain untuk melakukan proses prediksi selanjutnya.

---
## 📈 Evaluasi Model
Evaluasi model dilakukan untuk mengetahui kemampuan dari masing-masing model dalam melakukan klasifikasi citra pose yoga. Proses evaluasi ini bertujuan untuk membandingkan performa model CNN, MobileNetV2, dan ResNet101 berdasarkan hasil prediksi yang dihasilkan. 

Metrik evaluasi yang digunakan meliputi:
- **Akurasi** : Mengukur tingkat ketepatan model dalam mengklasifikasikan gambar.
- **Grafik training dan validation** : Melihat proses pembelajaran model selama pelatihan.
- **Confusion Matrix** : Mengetahui kesalahan prediksi antar kelas.
- **Classification Report** : Melihat nilai precision, recall, dan f1-score pada setiap kelas.
Evaluasi model dilakukan untuk mengukur performa masing-masing model dalam mengklasifikasikan citra pose yoga. Hasil evaluasi dari ketiga model kemudian dibandingkan untuk mengetahui model mana yang memberikan performa terbaik dalam mengenali pose yoga.

### 🔹 CNN (Non-Pretrained)

<p align="center">
  <img src="assets/grafik-plot%20CNN.png" width="600"/>
  <br>
  <em>Gambar 1. Grafik training dan validation model CNN</em>
</p>

<p align="center">
  <img src="assets/matrix%20CNN.png" width="500"/>
  <br>
  <em>Gambar 2. Confusion matrix model CNN</em>
</p>

---

### 🔹 MobileNetV2 (Pretrained)

<p align="center">
  <img src="assets/grafik-plot%20MobileNet.png" width="600"/>
  <br>
  <em>Gambar 3. Grafik training dan validation model MobileNetV2</em>
</p>

<p align="center">
  <img src="assets/matrix%20MobileNet.png" width="500"/>
  <br>
  <em>Gambar 4. Confusion matrix model MobileNetV2</em>
</p>

---

### 🔹 ResNet101 (Pretrained)

<p align="center">
  <img src="assets/grafik-plot%20resnet.png" width="600"/>
  <br>
  <em>Gambar 5. Grafik training dan validation model ResNet101</em>
</p>

<p align="center">
  <img src="assets/matrix%20resnet.png" width="500"/>
  <br>
  <em>Gambar 6. Confusion matrix model ResNet101</em>
</p>

---

## 🔍 Perbandingan Performa Model

| Nama Model   | Akurasi | Hasil Analisis |
|--------------|---------|----------------|
| CNN          | 46%     | Akurasi rendah karena jumlah kelas yang banyak dan data yang tidak seimbang. Arsitektur sederhana sulit membedakan pose yoga yang mirip. |
| MobileNetV2  | 65%     | Mengalami peningkatan signifikan berkat penggunaan model pretrained dan fine-tuning. Memberikan performa paling stabil. |
| ResNet101    | 52%     | Arsitektur lebih kompleks, namun kurang optimal karena keterbatasan data per kelas dan kompleksitas model. |

---

## 🖼️ Hasil Prediksi Model
Setelah proses pelatihan dan evaluasi selesai, masing-masing model diuji menggunakan citra pose yoga yang diambil dari folder dataset. Pengujian ini bertujuan untuk melihat kemampuan model dalam memprediksi kelas pose yoga. Hasil prediksi ditampilkan melalui dashboard berbasis Streamlit untuk melihat apakah model dapat menebak pose yoga dengan benar. 
Berikut ini ditampilkan contoh hasil prediksi dari ketiga model yang digunakan:

### 🔹 CNN (Non-Pretrained)

<p align="center">
  <img src="assets/cnn-hasil%20prediksi.jpg" width="600"/>
  <br>
  <em>Gambar 7. Contoh hasil prediksi pose yoga menggunakan model CNN</em>
</p>

Model berhasil memprediksi pose dengan benar menggunakan model CNN, dengan akurasi masing-masing sebesar **71,99%** dan **83,28%**.

---

### 🔹 MobileNetV2 (Pretrained)

<p align="center">
  <img src="assets/mobilenet-hasil%20prediksi.jpg" width="600"/>
  <br>
  <em>Gambar 8. Contoh hasil prediksi pose yoga menggunakan model MobileNetV2</em>
</p>

Model berhasil memprediksi pose dengan benar menggunakan model MobileNetV2, dengan akurasi masing-masing sebesar **82,71%** dan **99,60%**.

---

### 🔹 ResNet101 (Pretrained)

<p align="center">
  <img src="assets/resnet-hasil%20prediksi.jpg" width="600"/>
  <br>
  <em>Gambar 9. Contoh hasil prediksi pose yoga menggunakan model ResNet101</em>
</p>

Model berhasil memprediksi pose dengan benar menggunakan model ResNet101, dengan akurasi masing-masing sebesar **92,52%** dan **71,94%**.

---

## 🧪 Pembahasan
Berdasarkan hasil evaluasi dan pengujian:
- **MobileNetV2** memberikan performa terbaik dan paling stabil.
- **CNN** memiliki keterbatasan karena harus mempelajari fitur citra dari awal.
- **ResNet101** tidak menunjukkan keunggulan signifikan akibat kompleksitas model dan keterbatasan data.

Hasil ini menunjukkan bahwa **model pretrained yang ringan lebih efektif** untuk klasifikasi pose yoga dengan jumlah kelas yang besar.

---

## 🌐 Panduan Menjalankan Website Secara Lokal

Ikuti langkah-langkah berikut secara berurutan:

### 1. Clone repository
```bash
git clone https://github.com/username/uap-yogas-pose.git
cd uap-yogas-pose
```

### 2. Install PDM (Python Dependency Manager)
```bash
pip install pdm
pdm --version
```

### 3. Inisialisasi environment dengan PDM
```bash
pdm install           # Install seluruh dependency proyek
pdm venv activate     # Aktifkan environment
```

### 4. Install requirements tambahan
```bash
pdm add -d -r requirements.txt
```

### 5. Menjalankan Notebook untuk Training Model
```bash
pdm run jupyter notebook
# Buka file notebook: notebooks/UAP_Yoga_Pose_Image_Classification.ipynb
# Jalankan seluruh cell (Run All)
```
Setelah notebook dijalankan, model hasil training otomatis tersimpan di folder models:
- models/yoga_cnn_model.h5
- models/model_mobilenetv2_yoga.h5
- models/resnet101_model.keras

### 6. Menjalankan Website (Streamlit)
```bash
pdm run streamlit run app.py
# Buka browser di: http://localhost:8501
```
Cara Menggunakan Website
- Pilih model klasifikasi (CNN / MobileNetV2 / ResNet101)
- Upload gambar pose yoga (.jpg, .png, .jpeg)
- Sistem menampilkan:
  - Gambar input
  - Hasil prediksi kelas pose yoga
  - Probabilitas Top-5 kelas

### 7. Menghentikan Program
```bash 
# Tekan CTRL + C pada terminal untuk menghentikan aplikasi Streamlit
````

Catatan Penting:
1. Notebook hanya dijalankan sekali untuk training.
2. File model disimpan dan digunakan kembali oleh website.
3. Website hanya melakukan inference (prediksi).
4. Pastikan struktur folder models tidak berubah.

---
