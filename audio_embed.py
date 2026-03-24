# audio_embed.py
import numpy as np
import soundfile as sf
import librosa
import os

def embed_audio(cover_audio_path, secret_audio_path, output_path, bits_to_use=2):
    """
    Embed secret audio MSBs into cover audio LSBs.
    bits_to_use: 1..4 recommended (2 is a good balance).
    Works on PCM int16 domain.
    """
    assert 1 <= bits_to_use <= 8

    cover, sr = sf.read(cover_audio_path, always_2d=False)
    secret, sr2 = sf.read(secret_audio_path, always_2d=False)

    # If stereo, operate on first channel only or on each channel similarly;
    # here we convert to mono for simplicity.
    if cover.ndim > 1:
        cover = cover.mean(axis=1)
    if secret.ndim > 1:
        secret = secret.mean(axis=1)

    if sr2 != sr:
        # resample secret to cover sr using librosa
        secret = librosa.resample(secret.astype(float), orig_sr=sr2, target_sr=sr)

    min_len = min(len(cover), len(secret))
    cover = cover[:min_len]
    secret = secret[:min_len]

    # Convert from float [-1,1] to int16
    cover_i16 = np.round(cover * 32767).astype(np.int16)
    secret_i16 = np.round(secret * 32767).astype(np.int16)

    # Extract top bits_to_use MSBs from secret
    # For int16, we treat as unsigned magnitude by shifting after converting to uint16
    secret_u16 = secret_i16.astype(np.int16).astype(np.uint16)
    # MSBs: shift right by (16 - bits_to_use)
    secret_msbs = (secret_u16 >> (16 - bits_to_use)) & ((1 << bits_to_use) - 1)

    # Clear cover's lowest bits_to_use bits, then insert secret_msbs
    mask = ~((1 << bits_to_use) - 1)
    stego_i16 = (cover_i16 & mask) | secret_msbs.astype(np.int16)

    stego = stego_i16.astype(np.float32) / 32767.0
    sf.write(output_path, stego, sr)
    print("✅ Stego audio written to", output_path)
