# 🔐 Deep Learning Based Multi Modal Steganography

A secure and intelligent system that hides multiple types of secret data (text, image, audio, video) inside digital media using deep learning techniques.

---

## 📌 Project Overview

In the modern digital world, secure communication is essential. This project implements a **Deep Learning-based Multi-Modal Steganography System** that allows users to hide and retrieve confidential data inside cover media without raising suspicion.

Unlike traditional methods, this system supports **multiple data modalities** and uses **deep learning models** to ensure high security, imperceptibility, and robustness.

---

## 🚀 Key Features

- 🔒 Secure data hiding using password-based encryption
- 🧠 Deep learning-based embedding and extraction
- 🎯 Multi-modal support:
  - Text
  - Image
  - Audio
  - Video
- 📊 High PSNR & Low MSE (high-quality stego output)
- 🌐 Web-based interface using Flask
- 🗄️ MongoDB integration for storage and logs
- ⚡ Real-time communication using Socket.IO
- 🛡️ Resistant to noise, compression, and steganalysis attacks

---

## 🧠 Technologies Used

- Python
- Flask
- PyTorch
- OpenCV
- NumPy
- Pillow
- Librosa
- MongoDB
- HTML, CSS, JavaScript

---

## 🏗️ System Architecture

The system follows a **sender-receiver model**:

1. Sender selects:
   - Cover media
   - Secret data
   - Password

2. Deep learning encoder:
   - Extracts features
   - Encrypts and embeds data

3. Stego file is generated and transmitted

4. Receiver:
   - Inputs stego file + password
   - Decoder extracts hidden data

---

## ⚙️ Working Modules

### 🔹 Image Steganography
- Uses CNN-based encoding
- Embeds image/text into cover image

### 🔹 Audio Steganography
- Uses spectrogram-based deep learning
- Embeds audio into audio signals

### 🔹 Video Steganography
- Uses CNN + LSTM
- Frame-by-frame embedding

### 🔹 Text Steganography
- NLP-based encoding
- Maintains semantic meaning

---

## 📊 Performance Metrics

- High PSNR (Peak Signal-to-Noise Ratio)
- Low MSE (Mean Squared Error)
- High SSIM (Structural Similarity Index)

Ensures:
✔ Imperceptibility  
✔ Accuracy  
✔ Robustness  

---

## 🧪 Testing

- Unit Testing
- Integration Testing
- System Testing
- Performance Testing
- Security Testing

---

## 💻 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/vaish4596/Deep-Learning-Based-Multi-Modal-Steganography.git
cd Deep-Learning-Based-Multi-Modal-Steganography
````

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python app.py
```

### 5️⃣ Open in browser

```text
http://127.0.0.1:8000
```

---

## 📸 Results

* Text Steganography Interface
* Image Steganography Output
* Audio Steganography Results
* Video Embedding and Extraction

---

## 🎯 Applications

* Secure Communication
* Digital Watermarking
* Military & Defense Systems
* Data Protection
* Multimedia Copyright Protection

---

## 🔮 Future Scope

* Integration with GANs and Transformers
* Real-time Streaming Steganography
* Mobile & Cloud Deployment
* IoT-based Secure Communication
* Advanced Encryption Integration

---

## 👩‍💻 Author

**Vaishnavi Shetty**

```


