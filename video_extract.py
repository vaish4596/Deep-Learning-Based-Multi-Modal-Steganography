# video_extract.py
import cv2
import numpy as np
import os
import tempfile
import subprocess
from pathlib import Path

def _ensure_dir(d): Path(d).mkdir(parents=True, exist_ok=True)

def _frames_from_video_to_dir(video_path, out_dir):
    _ensure_dir(out_dir)
    # Use ffmpeg to extract frames (lossless) if possible
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            os.path.join(out_dir, "frame_%06d.png")
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        # fallback to OpenCV frame dump
        cap = cv2.VideoCapture(video_path)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:06d}.png"), frame)
            i += 1
        cap.release()
        return True

def extract_secret_video(stego_path, output_path, tmp_dir=None):
    """
    Extract the secret frames previously embedded by embed_secret_video.
    This reconstructs the secret frames by taking stego & 0b11 (LSB bits) and
    shifting them back to top bits (<<6) to approximate original appearance.
    Finally re-mux via ffmpeg to a lossless video file.
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="stego_extract_")
    frames_dir = os.path.join(tmp_dir, "frames")
    _ensure_dir(frames_dir)

    # extract frames as PNGs
    _frames_from_video_to_dir(stego_path, frames_dir)

    # iterate frames, reconstruct secret
    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if len(files) == 0:
        raise RuntimeError("No frames found to extract.")

    out_frames_dir = os.path.join(tmp_dir, "secret_frames")
    _ensure_dir(out_frames_dir)

    for idx, fname in enumerate(files):
        img = cv2.imread(os.path.join(frames_dir, fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # secret was stored in 2 LSBs; recover and shift to MSBs
        secret_bits = img & 0b00000011  # 0..3
        secret_recon = (secret_bits << 6).astype('uint8')  # back to high 2 bits
        # Write PNG (3-channel)
        cv2.imwrite(os.path.join(out_frames_dir, f"secret_{idx:06d}.png"), secret_recon)

    # mux secret frames into lossless video (ffmpeg)
    fps = 30
    try:
        # try to get fps using ffprobe (optional) or default 30
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(out_frames_dir, "secret_%06d.png"),
            "-c:v", "ffv1", output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Secret video extracted (lossless via ffmpeg).")
    except Exception as e:
        # fallback to writing via OpenCV (uncompressed)
        print("⚠️ ffmpeg not available for muxing; writing uncompressed AVI (fallback).", e)
        sample = cv2.imread(os.path.join(out_frames_dir, files[0].replace("frame_", "secret_")))
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'DIB ')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for idx in range(len(files)):
            im = cv2.imread(os.path.join(out_frames_dir, f"secret_{idx:06d}.png"))
            out.write(im)
        out.release()
        print("✅ Secret video extracted (fallback).")

    # optional: cleanup tmp_dir
