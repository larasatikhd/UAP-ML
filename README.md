
---

## 📊 Dataset
Dataset pose yoga diperoleh dari **Kaggle** dan memiliki karakteristik sebagai berikut:
- Total gambar: **5.991**
- Jumlah kelas: **107 pose yoga**
- Distribusi data antar kelas **tidak seimbang**

Contoh 10 kelas pertama pada dataset:

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
   Dataset dibagi menjadi data **training**, **validation**, dan **testing**.

---

## 🧠 Model yang Digunakan
Tiga model deep learning digunakan dan dibandingkan dalam proyek ini:

### 1️⃣ CNN (Non-Pretrained)
Model CNN dibangun dan dilatih dari awal tanpa menggunakan bobot pretrained. Model ini belajar mengenali fitur citra pose yoga langsung dari dataset.

### 2️⃣ MobileNetV2 (Pretrained)
MobileNetV2 merupakan model pretrained yang telah dilatih pada dataset berskala besar. Model ini bersifat ringan dan efisien, kemudian dilakukan **fine-tuning** agar sesuai dengan klasifikasi pose yoga sebanyak 107 kelas.

### 3️⃣ ResNet101 (Pretrained)
ResNet101 memiliki arsitektur jaringan yang lebih dalam dan mampu mengekstraksi fitur citra yang lebih kompleks. Model ini juga dilakukan fine-tuning untuk klasifikasi pose yoga.

Setelah pelatihan selesai, masing-masing model disimpan agar dapat digunakan kembali pada aplikasi web tanpa training ulang.

---

## 📈 Evaluasi Model
Evaluasi model dilakukan untuk mengetahui performa masing-masing model dalam mengklasifikasikan citra pose yoga. Metrik evaluasi yang digunakan meliputi:
- **Akurasi**
- **Grafik training dan validation**
- **Confusion Matrix**
- **Classification Report** (precision, recall, dan f1-score)
Evaluasi model dilakukan untuk mengukur performa masing-masing model dalam mengklasifikasikan citra pose yoga. Evaluasi mencakup grafik pelatihan (training & validation) serta confusion matrix.

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
Pengujian dilakukan melalui dashboard Streamlit menggunakan citra pose yoga untuk melihat kemampuan model dalam melakukan prediksi kelas pose yoga.

### 🔹 CNN (Non-Pretrained)

<p align="center">
  <img src="assets/cnn-hasil%20prediksi.jpg" width="600"/>
  <br>
  <em>Gambar 7. Contoh hasil prediksi pose yoga menggunakan model CNN</em>
</p>

Model menghasilkan tingkat kepercayaan prediksi sebesar **71,99%** dan **83,28%**.

---

### 🔹 MobileNetV2 (Pretrained)

<p align="center">
  <img src="assets/mobilenet-hasil%20prediksi.jpg" width="600"/>
  <br>
  <em>Gambar 8. Contoh hasil prediksi pose yoga menggunakan model MobileNetV2</em>
</p>

Model memberikan hasil prediksi paling konsisten dengan tingkat kepercayaan **82,71%** dan **99,60%**.

---

### 🔹 ResNet101 (Pretrained)

<p align="center">
  <img src="assets/resnet-hasil%20prediksi.jpg" width="600"/>
  <br>
  <em>Gambar 9. Contoh hasil prediksi pose yoga menggunakan model ResNet101</em>
</p>

Model menghasilkan tingkat kepercayaan prediksi sebesar **92,52%** dan **71,94%**.

---

## 🧪 Pembahasan
Berdasarkan hasil evaluasi dan pengujian:
- **MobileNetV2** memberikan performa terbaik dan paling stabil.
- **CNN** memiliki keterbatasan karena harus mempelajari fitur citra dari awal.
- **ResNet101** tidak menunjukkan keunggulan signifikan akibat kompleksitas model dan keterbatasan data.

Hasil ini menunjukkan bahwa **model pretrained yang ringan lebih efektif** untuk klasifikasi pose yoga dengan jumlah kelas yang besar.

---

## 🌐 Panduan Menjalankan Website Secara Lokal

1. Clone repository:
   ```bash
   git clone https://github.com/username/uap-yogas-pose.git
   cd uap-yogas-pose
