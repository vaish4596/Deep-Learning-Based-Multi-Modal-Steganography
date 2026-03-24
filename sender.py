from embed import embedFunc
from AES import encrypt

def hideFunc(secret_msg, password, cover_msg):
    encrypted_msg = encrypt(password, secret_msg)
    stego_text = embedFunc(encrypted_msg, cover_msg, password)
    return stego_text



