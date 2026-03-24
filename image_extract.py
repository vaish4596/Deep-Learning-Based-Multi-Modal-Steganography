from PIL import Image
import numpy as np

def str_to_bits(s):
    return np.unpackbits(np.frombuffer(s.encode(), dtype=np.uint8))

def xor_bits(data_bits, key_bits):
    key_repeated = np.resize(key_bits, data_bits.shape)
    return data_bits ^ key_repeated

def extract_image(stego_path, output_path, password):
    stego_img = Image.open(stego_path).convert("RGB")
    stego_array = np.array(stego_img)
    flat_stego = stego_array.reshape(-1, 3)

    with open("secret_size.txt", "r") as f:
        width, height = map(int, f.read().split(","))

    bits = []
    for pixel in flat_stego:
        for channel in pixel:
            bits.append(channel & 1)

    bits = np.array(bits[:width*height*8], dtype=np.uint8)

    password_bits = str_to_bits(password)
    decrypted_bits = xor_bits(bits, password_bits)

    bytes_list = []
    for i in range(0, len(decrypted_bits), 8):
        byte = 0
        for bit in decrypted_bits[i:i+8]:
            byte = (byte << 1) | bit
        bytes_list.append(byte)

    secret_array = np.array(bytes_list, dtype=np.uint8).reshape((height, width))

    secret_img = Image.fromarray(secret_array, "L")
    secret_img.save(output_path)









