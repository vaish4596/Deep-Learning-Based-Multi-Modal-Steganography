from PIL import Image
import numpy as np

def str_to_bits(s):
    return np.unpackbits(np.frombuffer(s.encode(), dtype=np.uint8))

def xor_bits(data_bits, key_bits):
    key_repeated = np.resize(key_bits, data_bits.shape)
    return data_bits ^ key_repeated

def embed_image(secret_image_path, cover_image_path, output_image_path, password):
    cover_image = Image.open(cover_image_path).convert("RGB")
    secret_image = Image.open(secret_image_path).convert("L")

    secret_array = np.array(secret_image)
    secret_bits = np.unpackbits(secret_array.flatten())

    password_bits = str_to_bits(password)
    encrypted_bits = xor_bits(secret_bits, password_bits)

    max_capacity = cover_image.size[0] * cover_image.size[1] * 3
    if len(encrypted_bits) > max_capacity:
        raise ValueError("❌ Secret image too large for this cover image. Use a bigger cover.")

    cover_array = np.array(cover_image)
    flat_cover = cover_array.reshape(-1, 3)

    for i, bit in enumerate(encrypted_bits):
        flat_cover[i // 3, i % 3] = (flat_cover[i // 3, i % 3] & 0xFE) | bit

    stego_array = flat_cover.reshape(cover_array.shape)
    stego_image = Image.fromarray(stego_array.astype(np.uint8))
    stego_image.save(output_image_path)

    with open("secret_size.txt", "w") as f:
        f.write(f"{secret_image.size[0]},{secret_image.size[1]}")

    print(f"[+] Stego image saved to {output_image_path}")
    return output_image_path







