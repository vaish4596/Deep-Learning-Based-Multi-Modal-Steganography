ZWC_reverse = {
    u'\u200C': "00",
    u'\u202C': "01",
    u'\u202D': "10",
    u'\u200E': "11"
}

def xor(a, b, n):
    return "".join("0" if a[i] == b[i] else "1" for i in range(n))

def binaryToDecimal(n):
    return int(n, 2)

def derive_key_from_password(password):
    return str(sum(ord(c) for c in password) % 256)

def extractFunc(stego_message, password):
    hashed_SM_binary_extract = ""
    for letter in stego_message:
        if letter in ZWC_reverse:
            hashed_SM_binary_extract += ZWC_reverse[letter]

    if not hashed_SM_binary_extract:
        return ""

    MS_SK = derive_key_from_password(password)
    MR_SK_binary = f'{int(MS_SK):08b}'
    LSK = len(MR_SK_binary)
    P = 0 if len(hashed_SM_binary_extract) % LSK == 0 else 1
    NC = (len(hashed_SM_binary_extract) // LSK) + P
    hash_position_bits = MR_SK_binary * NC

    SM_binary_extract = xor(hashed_SM_binary_extract, hash_position_bits, len(hashed_SM_binary_extract))

    SM_extract = ""
    while len(SM_binary_extract) >= 12:
        alpha_beta = SM_binary_extract[:12]
        SM_binary_extract = SM_binary_extract[12:]
        alpha = binaryToDecimal(alpha_beta[:6])
        beta = binaryToDecimal(alpha_beta[6:])
        n_final = int((2 ** alpha) * (2 * beta + 1) - 1)
        SM_extract += chr(n_final)

    return SM_extract




