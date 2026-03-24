import math

ZWC = {
    "00": u'\u200C',
    "01": u'\u202C',
    "10": u'\u202D',
    "11": u'\u200E'
}

def isPowerOfTwo(n):
    return (math.ceil(math.log2(n)) == math.floor(math.log2(n))) if n > 0 else False

def xor(a, b, n):
    return "".join("0" if a[i] == b[i] else "1" for i in range(n))

def derive_key_from_password(password):
    return str(sum(ord(c) for c in password) % 256)

def embedFunc(secret_message, cover_message, password):
    SM_binary = ""
    MS_SK = derive_key_from_password(password)

    for letter in secret_message:
        n = ord(letter)
        factors = [i for i in range(1, n + 1) if (n + 1) % i == 0]
        odd_factors = [f for f in factors if f % 2 != 0]
        alpha = -1
        for odd_factor in odd_factors:
            if isPowerOfTwo((n + 1) // odd_factor):
                power = math.log2((n + 1) // odd_factor)
                if power > alpha:
                    alpha = int(power)
        if alpha == -1 and n % 2 == 0:
            alpha = 0

        alpha_binary = f'{alpha:06b}'
        beta = int((((n + 1) / (2 ** alpha)) - 1) / 2)
        beta_binary = f'{beta:06b}'

        SM_binary += alpha_binary + beta_binary

    MS_SK_binary = f'{int(MS_SK):08b}'
    LSK = len(MS_SK_binary)
    P = 0 if len(SM_binary) % LSK == 0 else 1
    NC = (len(SM_binary) // LSK) + P
    hash_position_bits = MS_SK_binary * NC
    hashed_SM_binary = xor(SM_binary, hash_position_bits, len(SM_binary))

    HM_ZWC = "".join(ZWC[hashed_SM_binary[i:i + 2]] for i in range(0, len(hashed_SM_binary), 2))
    
    if len(cover_message) > 1:
        stego_message = cover_message[:-1] + HM_ZWC + cover_message[-1]
    else:
        stego_message = cover_message + HM_ZWC

    return stego_message


