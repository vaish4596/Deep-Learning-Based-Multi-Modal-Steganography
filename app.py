from flask import Flask, request, render_template
from image_embed import embed_image
from image_extract import extract_image
from video_embed import embed_secret_video as embed_video
from video_extract import extract_secret_video as extract_video
from audio_embed import embed_audio
from audio_extract import extract_audio
from sender import hideFunc
from receiver import revealFunc

import os
import math
import tempfile
import numpy as np
from PIL import Image
import cv2
import datetime

# ---- DL imports
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# ---- MongoDB
from pymongo import MongoClient

# =========================
# Flask & storage
# =========================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# =========================
# MongoDB Setup
# =========================
MONGO_URI = "mongodb://localhost:27017/"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["stego_app"]
logs_col = db["logs"]

def log_action(action_type, filename=None, details=None):
    logs_col.insert_one({
        "timestamp": datetime.datetime.utcnow(),
        "action": action_type,
        "file": filename,
        "details": details
    })

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# =========================
# Model Builders
# =========================
class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def build_image_model(num_classes=2):
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class VideoRNN(nn.Module):
    """CNN + LSTM hybrid for video stego detection"""
    def __init__(self, hidden_size=256, num_classes=2):
        super(VideoRNN, self).__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = base.fc.in_features
        self.rnn = nn.LSTM(input_size=self.feature_dim,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x).view(B, T, -1)
        out, _ = self.rnn(feats)
        out = out[:, -1, :]
        return self.fc(out)


def build_video_model(num_classes=2):
    return VideoRNN(num_classes=num_classes)

# =========================
# Load checkpoint
# =========================
CKPT_PATH = "stego_multi_modal.pth"
image_model = None
video_model = None
audio_model = None

def infer_num_classes_from_state_dict(sd_key_state_dict, default_nc=2):
    for k, v in sd_key_state_dict.items():
        if k.endswith("fc.weight") and v.ndim == 2:
            return v.shape[0]
    return default_nc

def load_models():
    global image_model, video_model, audio_model

    if not os.path.exists(CKPT_PATH):
        print(f"⚠️ WARNING: Checkpoint '{CKPT_PATH}' not found. Using randomly initialized models.")
        image_model = build_image_model(num_classes=2).to(device).eval()
        video_model = build_video_model(num_classes=2).to(device).eval()
        audio_model = AudioCNN(num_classes=2).to(device).eval()
        return

    ckpt = torch.load(CKPT_PATH, map_location=device)

    img_nc = infer_num_classes_from_state_dict(ckpt.get("image_model", {}), default_nc=2)
    vid_nc = infer_num_classes_from_state_dict(ckpt.get("video_model", {}), default_nc=2)
    aud_nc = infer_num_classes_from_state_dict(ckpt.get("audio_model", {}), default_nc=2)

    image_model = build_image_model(num_classes=img_nc).to(device)
    video_model = build_video_model(num_classes=vid_nc).to(device)
    audio_model = AudioCNN(num_classes=aud_nc).to(device)

    if "image_model" in ckpt:
        image_model.load_state_dict(ckpt["image_model"], strict=False)
        print(f"✅ Loaded image_model ({img_nc} classes)")
    if "video_model" in ckpt:
        video_model.load_state_dict(ckpt["video_model"], strict=False)
        print(f"✅ Loaded video_model ({vid_nc} classes)")
    if "audio_model" in ckpt:
        audio_model.load_state_dict(ckpt["audio_model"], strict=False)
        print(f"✅ Loaded audio_model ({aud_nc} classes)")

    image_model.eval()
    video_model.eval()
    audio_model.eval()

load_models()

# =========================
# Common transforms
# =========================
img_transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def interpret_logits(logits):
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        if logits.shape[1] == 2:
            label = "Stego" if pred.item() == 1 else "Normal"
        else:
            label = f"Class {pred.item()}"
        return label, conf.item()

# =========================
# DL Detectors
# =========================
def detect_stego_image(image_path):
    img = Image.open(image_path).convert("RGB")
    x = img_transform_224(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = image_model(x)
    label, conf = interpret_logits(logits)
    return f"{label} Image (conf {conf:.2f})"

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def detect_stego_video(video_path, clip_len=16, sample_rate=2):
    frames = read_video_frames(video_path)
    need = clip_len * sample_rate
    if len(frames) < need:
        return "Video too short for detection."
    sampled = [frames[i * sample_rate] for i in range(clip_len)]
    arr = np.stack(sampled, axis=0)
    arr = [Image.fromarray(f).resize((224, 224)) for f in arr]
    arr = [img_transform_224(f).unsqueeze(0) for f in arr]
    x = torch.cat(arr, dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = video_model(x)
    label, conf = interpret_logits(logits)
    return f"{label} Video (conf {conf:.2f})"

def detect_stego_audio(audio_path):
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_t = torch.tensor(spec_db).unsqueeze(0).unsqueeze(0).float()
        spec_t = torch.nn.functional.interpolate(spec_t, size=(128, 128), mode="bilinear", align_corners=False)
        spec_t = spec_t.to(device)
        with torch.no_grad():
            logits = audio_model(spec_t)
        label, conf = interpret_logits(logits)
        return f"{label} Audio (conf {conf:.2f})"
    except Exception as e:
        return f"Audio detection unavailable: {e}"

# =========================
# Views
# =========================
@app.route("/")
def index():
    return render_template("homepage.html")

# ===== All routes below include MongoDB logging =====

# Image hide/reveal
@app.route("/hide_image", methods=["POST"])
def hide_image():
    secret = request.files.get('secret_img')
    cover = request.files.get('cover_img')
    password = request.form.get('password', '')
    if not secret or not cover:
        return render_template("homepage.html", image_error="Please upload both secret and cover images.")

    secret_path = os.path.join(app.config["UPLOAD_FOLDER"], 'temp_secret.png')
    cover_path = os.path.join(app.config["UPLOAD_FOLDER"], 'temp_cover.png')
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], 'stego_image.png')

    secret.save(secret_path)
    cover.save(cover_path)

    try:
        cover_img = Image.open(cover_path).convert("RGB")
        secret_img = Image.open(secret_path).convert("L")
        cover_capacity = (cover_img.width * cover_img.height * 3) // 8
        secret_pixels = secret_img.width * secret_img.height
        if secret_pixels > cover_capacity:
            scale = math.sqrt(cover_capacity / secret_pixels)
            new_size = (max(1, int(secret_img.width * scale)),
                        max(1, int(secret_img.height * scale)))
            secret_img = secret_img.resize(new_size)
            secret_img.save(secret_path)
        embed_image(secret_path, cover_path, output_path, password)
        prediction = detect_stego_image(output_path)

        log_action("hide_image", filename='stego_image.png', details={"prediction": prediction})

        return render_template("homepage.html", result_image='stego_image.png', dl_result=f"Prediction: {prediction}")
    except Exception as e:
        return render_template("homepage.html", image_error=f"❌ Error: {str(e)}")

@app.route("/reveal_image", methods=["POST"])
def reveal_image():
    steg = request.files.get('stego_img')
    password = request.form.get('password', '')
    if not steg:
        return render_template("homepage.html", image_error="Please upload a stego image.")
    steg_path = os.path.join(app.config["UPLOAD_FOLDER"], 'uploaded_stego.png')
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], 'revealed_secret.png')
    steg.save(steg_path)
    try:
        extract_image(steg_path, output_path, password)
        prediction = detect_stego_image(steg_path)

        log_action("reveal_image", filename='revealed_secret.png', details={"prediction": prediction})

        return render_template("homepage.html", revealed_image='revealed_secret.png', dl_result=f"Prediction: {prediction}")
    except Exception as e:
        return render_template("homepage.html", image_error=f"❌ Error: {str(e)}")

# Text hide/reveal
@app.route("/hide_text", methods=["POST"])
def hide_text():
    secret = request.form.get('secret_msg', '')
    cover = request.form.get('cover_msg', '')
    password = request.form.get('password', '')
    try:
        stego = hideFunc(secret, password, cover)
        log_action("hide_text", details={"stego": stego})
        return render_template("homepage.html", stego_msg=stego)
    except Exception as e:
        return render_template("homepage.html", text_error=f"❌ Error: {str(e)}")

@app.route("/reveal_text", methods=["POST"])
def reveal_text():
    stego = request.form.get('stego_msg', '')
    password = request.form.get('password', '')
    try:
        revealed = revealFunc(stego, password)
        log_action("reveal_text", details={"revealed": revealed})
        return render_template("homepage.html", revealed_msg=revealed, stego_msg=stego)
    except Exception as e:
        return render_template("homepage.html", text_error=f"❌ Error: {str(e)}")

# Video hide/reveal
@app.route("/hide_video", methods=["POST"])
def hide_video():
    cover = request.files.get('cover_video')
    secret = request.files.get('secret_video')
    if not cover or not secret:
        return render_template("homepage.html", video_error="Please upload both cover and secret videos.")
    cover_path = os.path.join(app.config["UPLOAD_FOLDER"], 'cover_video.mp4')
    secret_path = os.path.join(app.config["UPLOAD_FOLDER"], 'secret_video.mp4')
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], 'stego_video.mp4')
    cover.save(cover_path)
    secret.save(secret_path)
    try:
        embed_video(cover_path, secret_path, output_path)
        log_action("hide_video", filename='stego_video.mp4')
        return render_template("homepage.html", result_video='stego_video.mp4')
    except Exception as e:
        return render_template("homepage.html", video_error=f"❌ Error: {str(e)}")

@app.route("/reveal_video", methods=["POST"])
def reveal_video():
    steg = request.files.get('stego_video')
    if not steg:
        return render_template("homepage.html", video_error="Please upload a stego video.")
    steg_path = os.path.join(app.config["UPLOAD_FOLDER"], 'uploaded_stego_video.mp4')
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], 'revealed_secret_video.mp4')
    steg.save(steg_path)
    try:
        extract_video(steg_path, output_path)
        log_action("reveal_video", filename='revealed_secret_video.mp4')
        return render_template("homepage.html", revealed_video='revealed_secret_video.mp4')
    except Exception as e:
        return render_template("homepage.html", video_error=f"❌ Error: {str(e)}")

# Audio hide/reveal
@app.route("/hide_audio", methods=["POST"])
def hide_audio():
    cover = request.files.get('cover_audio')
    secret = request.files.get('secret_audio')
    if not cover or not secret:
        return render_template("homepage.html", audio_error="Please upload both cover and secret audio.")
    cover_path = os.path.join(app.config["UPLOAD_FOLDER"], 'cover_audio.wav')
    secret_path = os.path.join(app.config["UPLOAD_FOLDER"], 'secret_audio.wav')
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], 'stego_audio.wav')
    cover.save(cover_path)
    secret.save(secret_path)
    try:
        embed_audio(cover_path, secret_path, output_path)
        prediction = detect_stego_audio(output_path)
        log_action("hide_audio", filename='stego_audio.wav', details={"prediction": prediction})
        return render_template("homepage.html", result_audio='stego_audio.wav', dl_result=f"Prediction: {prediction}")
    except Exception as e:
        return render_template("homepage.html", audio_error=f"❌ Error: {str(e)}")

@app.route("/reveal_audio", methods=["POST"])
def reveal_audio():
    steg = request.files.get('stego_audio')
    if not steg:
        return render_template("homepage.html", audio_error="Please upload a stego audio.")
    steg_path = os.path.join(app.config["UPLOAD_FOLDER"], 'uploaded_stego_audio.wav')
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], 'revealed_secret_audio.wav')
    steg.save(steg_path)
    try:
        extract_audio(steg_path, output_path)
        prediction = detect_stego_audio(steg_path)
        log_action("reveal_audio", filename='revealed_secret_audio.wav', details={"prediction": prediction})
        return render_template("homepage.html", revealed_audio='revealed_secret_audio.wav', dl_result=f"Prediction: {prediction}")
    except Exception as e:
        return render_template("homepage.html", audio_error=f"❌ Error: {str(e)}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


