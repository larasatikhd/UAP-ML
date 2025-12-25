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

## DATASET
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

## PRE-PROCESSING DATA
Sebelum digunakan untuk melatih model, data citra akan melakukan tahap pra-pemrosesan (preprocessing). Tahap ini dilakukan untuk menyamakan ukuran gambar, memperbaiki kualitas data, serta membantu model agar dapat mengenali pola pose yoga dengan lebih baik.

Tahapan preprocessing yang dilakukan pada data citra pose yoga meliputi:

1. **Resize Gambar**  
   Gambar diubah ke ukuran yang sama, yaitu **224 × 224 piksel**, agar sesuai dengan kebutuhan input model CNN dan model pretrained yang digunakan.

2. **Normalisasi Piksel**  
   Nilai piksel pada gambar dinormalisasi ke rentang tertentu untuk mempercepat proses pelatihan dan meningkatkan kestabilan model.

3. **Augmentasi Data**  
   Mengatasi keterbatasan jumlah data dan ketidakseimbangan antar kelas, dilakukan augmentasi data seperti rotasi, flip, dan zoom. Augmentasi ini membantu meningkatkan variasi data tanpa menambah jumlah gambar secara langsung.

4. **Pembagian Dataset**  
   Dataset dibagi menjadi data latih (training), data validasi (validation), dan data uji (testing). Pembagian ini dilakukan untuk melatih model serta mengevaluasi performa model secara objektif.

Tahap preprocessing penting karena dapat meningkatkan kemampuan model dalam mengenali berbagai pose yoga meskipun terdapat variasi posisi, sudut pengambilan gambar, dan kondisi pencahayaan.

---

## MODEL YANG DIGUNAKAN
Terdapat tiga model yang digunakan untuk melakukan klasifikasi citra pose yoga. Ketiga model tersebut terdiri dari satu model Non-Pretrained, yaitu CNN, serta dua model Pretrained, yaitu MobileNetV2 dan ResNet101. Setiap model dilatih dan dievaluasi dengan menampilkan hasil berupa Akurasi, Grafik, Confusion Matrix, dan Classification Report. Setelah proses pelatihan selesai, model disimpan dari Google Colab agar dapat digunakan kembali pada aplikasi Streamlit sebagai dashboard interaktif. Sehingga, dengan adanya penyimpanan model ini, proses prediksi gambar dapat dilakukan tanpa perlu melakukan pelatihan ulang.

Tiga model deep learning digunakan dan dibandingkan dalam proyek ini:

### CNN (Non-Pretrained)
Model CNN yang dibuat dan dilatih dari awal tanpa adanya model lain. Model ini belajar mengenali pose yoga langsung dari gambar yang ada di dataset, mulai dari pola sederhana hingga pola yang lebih rumit.

### MobileNetV2 (Pretrained)
Model CNN yang sudah dilatih sebelumnya menggunakan dataset besar. Model ini dapat mengenali pola dasar pada gambar, sehingga proses pelatihannya menjadi lebih cepat. MobileNetV2 digunakan kembali dan disesuaikan untuk mengenali pose yoga sesuai dengan jumlah kelas yang digunakan.

### ResNet101 (Pretrained)
Model CNN dengan struktur yang lebih dalam. Model ini juga sudah dilatih sebelumnya dan mampu mengenali fitur gambar yang lebih kompleks. Dengan kemampuan tersebut, ResNet101 diharapkan dapat memberikan hasil klasifikasi pose yoga yang lebih baik.

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
## EVALUASI MODEL
Evaluasi model dilakukan untuk mengetahui kemampuan dari masing-masing model dalam melakukan klasifikasi citra pose yoga. Proses evaluasi ini bertujuan untuk membandingkan performa model CNN, MobileNetV2, dan ResNet101 berdasarkan hasil prediksi yang dihasilkan. 

Pada tahap evaluasi, beberapa metrik yang digunakan antara lain:
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

## PERBANDINGAN ANTAR MODEL 

Pada bagian ini akan menampilkan hasil yang diperoleh dari setiap model terkait performa masing-masing model dalam mengenali pose yoga.

| Nama Model   | Akurasi | Hasil Analisis |
|--------------|---------|----------------|
| CNN          | 46%     | Dikarenakan jumlah class yang sangat banyak (107 class) dengan rata-rata data per class yang relatif sedikit dan tidak seimbang, sehingga model kesulitan dalam membedakan pose yoga tersebut. Arsitektur CNN yang sederhana hanya mampu menangkap fitur dasar, sehingga kurang efektif untuk membedakan pose yoga yang kompleks dan mirip secara visual. |
| MobileNetV2  | 65%     | Model mengalami peningkatan yang cukup signifikan karena menggunakan model pretrained yang sudah mengenali fitur visual dasar. Proses fine-tuning membantu penyesuaian pada pose yoga, namun performa masih terbatas oleh banyaknya kelas dan kemiripan antar pose. |
| ResNet101    | 52%     | Memiliki akurasi lebih rendah dibandingkan MobileNetV2 meskipun arsitekturnya lebih dalam. Kompleksitas model dan jumlah parameter yang besar menyulitkan proses fine-tuning pada dataset dengan data per kelas yang terbatas. |

---

## HASIL PREDIKSI MODEL 
Setelah proses pelatihan dan evaluasi selesai, masing-masing model diuji menggunakan citra pose yoga yang diambil dari folder dataset. Pengujian ini bertujuan untuk melihat kemampuan model dalam memprediksi kelas pose yoga. Hasil prediksi ditampilkan melalui dashboard berbasis Streamlit untuk melihat apakah model dapat menebak pose yoga dengan benar. 
Berikut ini ditampilkan contoh hasil prediksi dari ketiga model yang digunakan:

### 🔹 CNN (Non-Pretrained)

<p align="center">
  <img src="assets/cnn-hasil.jpeg" width="600"/>
  <br>
  <em>Gambar 7. Contoh hasil prediksi pose yoga menggunakan model CNN</em>
</p>

Model berhasil memprediksi pose dengan benar menggunakan model CNN, dengan akurasi masing-masing sebesar **71,99%** dan **83,28%**.

---

### 🔹 MobileNetV2 (Pretrained)

<p align="center">
  <img src="assets/mobilenet-hasil.jpeg" width="600"/>
  <br>
  <em>Gambar 8. Contoh hasil prediksi pose yoga menggunakan model MobileNetV2</em>
</p>

Model berhasil memprediksi pose dengan benar menggunakan model MobileNetV2, dengan akurasi masing-masing sebesar **82,71%** dan **99,60%**.

---

### 🔹 ResNet101 (Pretrained)

<p align="center">
  <img src="assets/resnet-hasil.jpeg" width="600"/>
  <br>
  <em>Gambar 9. Contoh hasil prediksi pose yoga menggunakan model ResNet101</em>
</p>

Model berhasil memprediksi pose dengan benar menggunakan model ResNet101, dengan akurasi masing-masing sebesar **92,52%** dan **71,94%**.

---

## HASIL PEMBAHASAN MODEL
Berdasarkan hasil perbandingan akurasi dari ketiga model tersebut, MobileNetV2 menghasilkan akurasi tertinggi dibandingkan CNN dan ResNet101. Model CNN memiliki akurasi terendah karena harus mempelajari fitur citra dari awal pada dataset dengan jumlah kelas yang besar dan data yang tidak seimbang. Selanjutnya, MobileNetV2 mampu memberikan hasil yang lebih baik karena telah menggunakan model pretrained yang membantu dalam mengenali fitur visual penting, sementara ResNet101 yang memiliki arsitektur lebih dalam tetapi idak menunjukkan performa yang optimal akibat kompleksitas model dan keterbatasan data per kelas. Hasil ini menunjukkan bahwa penggunaan model pretrained yang ringan lebih efektif untuk klasifikasi pose yoga dengan jumlah kelas yang besar.

Selanjutnya, berdasarkan hasil prediksi yang ditampilkan melalui dashboard Streamlit, ketiga model diuji menggunakan dua citra pose yoga yang sama untuk melihat kemampuan masing-masing dalam mengenali pose tersebut. Ketiga model berhasil memprediksi pose yoga dengan benar, meskipun dengan tingkat kepercayaan yang berbeda-beda, sebagai berikut: 

- **CNN**: Akurasi prediksi sebesar 71,99% dan 83,28%, yang menunjukkan bahwa walaupun performa pelatihannya masih rendah, model ini tetap mampu mengenali beberapa pose dengan cukup baik.
- **MobileNetV2**: Hasil paling konsisten dengan akurasi 82,71% dan 99,60%, menandakan bahwa penggunaan model pretrained sangat membantu dalam mengenali fitur visual pose yoga secara lebih akurat. 
- **ResNet101**: Akurasi 92,52% dan 71,94%, yang menggambarkan kemampuan ekstraksi fitur yang baik, namun hasilnya masih dipengaruhi oleh kompleksitas pose dan kemiripan antar kelas. 

Sehingga, secara keseluruhan, hasil ini menegaskan bahwa **MobileNetV2** memberikan performa prediksi yang paling stabil dibandingkan kedua model lainnya.

---

## PANDUAN MENJALANKAN WEB SECARA LOKAL

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

### CATATAN:
1. Notebook hanya dijalankan sekali untuk training.
2. File model disimpan dan digunakan kembali oleh website.
3. Website hanya melakukan inference (prediksi).
4. Pastikan struktur folder models tidak berubah.

---
