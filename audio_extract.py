# audio_extract.py
import numpy as np
import soundfile as sf
import os

def extract_audio(stego_audio_path, output_path, bits_used=2):
    """
    Extract secret audio embedded using embed_audio.
    bits_used must match what was used when embedding.
    """

    stego, sr = sf.read(stego_audio_path, always_2d=False)
    if stego.ndim > 1:
        stego = stego.mean(axis=1)

    stego_i16 = np.round(stego * 32767).astype(np.int16)

    # Get the bits_used LSBs
    secret_bits = (stego_i16.astype(np.int16) & ((1 << bits_used) - 1)).astype(np.uint16)

    # Shift the recovered bits up to MSB positions to reconstruct amplitude
    secret_u16 = (secret_bits.astype(np.uint16) << (16 - bits_used))

    # Convert back to signed int16
    secret_i16 = secret_u16.astype(np.int16)

    secret = secret_i16.astype(np.float32) / 32767.0

    sf.write(output_path, secret, sr)
    print("✅ Secret audio extracted to", output_path)


