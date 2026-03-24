from extract import extractFunc
from AES import decrypt

def revealFunc(stego_msg, password):
    encrypted_msg = extractFunc(stego_msg, password)

    if encrypted_msg:
        try:
            return decrypt(password, encrypted_msg)
        except Exception:
            return "❌ Failed to decrypt. Wrong password or corrupted data."
    else:
        return "❌ No hidden message found!"


