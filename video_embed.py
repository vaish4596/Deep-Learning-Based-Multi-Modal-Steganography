# video_embed.py
import cv2
import numpy as np
import os
import subprocess
from pathlib import Path

def _ensure_dir(d): Path(d).mkdir(parents=True, exist_ok=True)

def embed_secret_video(cover_path, secret_path, output_path, tmp_dir="tmp_frames"):
    """
    Lossless-friendly method:
      - Extract frames from cover and secret
      - Embed top 2 bits (MSBs) of secret into 2 LSBs of cover per channel
      - Save stego frames as PNG into tmp_dir
      - Use ffmpeg to encode into a lossless container (matroska with FFV1) if ffmpeg available;
        otherwise fallback to an uncompressed avi via OpenCV (big files).
    """
    _ensure_dir(tmp_dir)
    cap_c = cv2.VideoCapture(cover_path)
    cap_s = cv2.VideoCapture(secret_path)

    width = int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_c.get(cv2.CAP_PROP_FPS) or 30

    i = 0
    while True:
        ok_c, frame_c = cap_c.read()
        ok_s, frame_s = cap_s.read()

        if not ok_c:
            break

        if not ok_s:
            # pad secret with zeros if shorter
            frame_s = np.zeros_like(frame_c)
        else:
            frame_s = cv2.resize(frame_s, (width, height))

        # ensure uint8
        cover = frame_c.astype(np.uint8)
        secret = frame_s.astype(np.uint8)

        # take top 2 bits of secret (bits 7 and 6) -> shift right by 6 (0..3)
        secret_top2 = (secret >> 6) & 0x03  # values 0..3

        # clear cover's 2 LSBs, then OR with secret_top2
        stego = (cover & 0b11111100) | secret_top2

        # write PNG frame (lossless)
        frame_file = os.path.join(tmp_dir, f"frame_{i:06d}.png")
        cv2.imwrite(frame_file, stego)
        i += 1

    cap_c.release()
    cap_s.release()

    # Use ffmpeg to make a lossless video (MKV + FFV1) if available.
    # Final file will be output_path (any extension but recommend .mkv or .avi)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(int(fps)),
        "-i", os.path.join(tmp_dir, "frame_%06d.png"),
        "-c:v", "ffv1", output_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Stego video created (lossless via ffmpeg).")
    except Exception as e:
        # Fallback: write an uncompressed avi via OpenCV (huge file)
        print("⚠️ ffmpeg not available or failed; falling back to writing uncompressed AVI (very large).", e)
        fourcc = cv2.VideoWriter_fourcc(*'DIB ')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for j in range(i):
            img = cv2.imread(os.path.join(tmp_dir, f"frame_{j:06d}.png"))
            out.write(img)
        out.release()
        print("✅ Stego video created (fallback).")

    # Optionally keep tmp_dir for debugging or remove it
    # import shutil; shutil.rmtree(tmp_dir)
