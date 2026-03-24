import binascii
from Crypto.Cipher import AES
from Crypto import Random
from Crypto.Protocol.KDF import PBKDF2

def get_private_key(password, salt):
    return PBKDF2(password, salt, dkLen=32, count=200000)

def encrypt(password, message):
    salt = Random.get_random_bytes(16)
    key = get_private_key(password, salt)
    iv = Random.get_random_bytes(AES.block_size)
    cipher = AES.new(key, AES.MODE_CFB, iv)
    ciphertext = cipher.encrypt(message.encode("utf-8"))
    encrypted = salt + iv + ciphertext
    return binascii.hexlify(encrypted).decode("utf-8")

def decrypt(password, hex_message):
    encrypted = binascii.unhexlify(hex_message.encode("utf-8"))
    salt = encrypted[:16]
    iv = encrypted[16:32]
    ciphertext = encrypted[32:]
    key = get_private_key(password, salt)
    cipher = AES.new(key, AES.MODE_CFB, iv)
    plaintext = cipher.decrypt(ciphertext)
    return plaintext.decode("utf-8")

