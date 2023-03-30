import numpy as np


def dE76(L_1: float, a_1: float, b_1: float, L_2: float, a_2: float, b_2: float) -> float:
    return np.sqrt(sum(map(lambda i: i * i, [L_1 - L_2, a_1 - a_2, b_1 - b_2])))


def dE94(L_1: float, a_1: float, b_1: float, L_2: float, a_2: float, b_2: float) -> float:
    KL = 1
    KC = 1
    KH = 1

    K1 = 0.045
    K2 = 0.015

    dL = L_1 - L_2
    da = a_1 - a_2
    db = b_1 - b_2

    C1 = np.sqrt(a_1**2 + b_1**2)
    C2 = np.sqrt(a_2**2 + b_2**2)
    dC = C1 - C2

    dH = np.sqrt(da**2 + db**2 - dC**2)

    SL = 1
    SC = 1 + K1 * C1
    SH = 1 + K2 * C1

    return np.sqrt(
        sum(map(lambda i: i * i, [dL / (KL * SL), dC / (KC * SC), dH / (KH * SH)]))
    )


def dE00(L_1: float, a_1: float, b_1: float, L_2: float, a_2: float, b_2: float) -> float:
    KL = 1
    KC = 1
    KH = 1

    L_ = (L_1 + L_2) / 2
    dL_ = L_2 - L_1

    C1 = np.sqrt(a_1**2 + b_1**2)
    C2 = np.sqrt(a_2**2 + b_2**2)
    C = (C1 + C2) / 2

    G = (1 / 2) * (1 - np.sqrt(C**7 / (C**7 + 25**7)))

    a_1_ = a_1 * (1 + G)
    a_2_ = a_2 * (1 + G)

    C1_ = np.sqrt(a_1_**2 + b_1**2)
    C2_ = np.sqrt(a_2_**2 + b_2**2)
    Cbar_ = (C1_ + C2_) / 2
    dC_ = C2_ - C1_

    h1_ = (
        np.arctan(b_1 / a_1_)
        if np.arctan(b_1 / a_1_) >= 0
        else np.arctan(b_1 / a_1_) + 360
    )
    h2_ = (
        np.arctan(b_2 / a_2_)
        if np.arctan(b_2 / a_2_) >= 0
        else np.arctan(b_2 / a_2_) + 360
    )

    if abs(h1_ - h2_) <= 180:
        dh_ = h2_ - h1_
    elif abs(h1_ - h2_) > 180 and h2_ <= h1_:
        dh_ = h2_ - h1_ + 360
    elif abs(h1_ - h2_) > 180 and h2_ > h1_:
        dh_ = h2_ - h1_ - 360

    dH_ = 2 * np.sqrt(C1_ * C2_) * np.sin(dh_ / 2)
    Hbar_ = (h1_ + h2_) / 2 if abs(h1_ - h2_) <= 180 else (h1_ + h2_ + 360) / 2

    T = (
        1
        - 0.17 * np.cos(Hbar_ - 30)
        + 0.24 * np.cos(2 * Hbar_)
        + 0.32 * np.cos(3 * Hbar_ + 6)
        - 0.20 * np.cos(4 * Hbar_ - 63)
    )

    SL = 1 + ((0.015 * (L_ - 50) ** 2) / np.sqrt(20 + (L_ - 50) ** 2))
    SC = 1 + 0.045 * Cbar_
    SH = 1 + 0.015 * Cbar_ * T

    dTheta = 30 * np.exp(-(((Hbar_ - 275) / 25) ** 2))

    RC = 2 * np.sqrt(Cbar_**7 / (Cbar_**7 + 25**7))

    RT = -1 * RC * np.sin(2 * dTheta)

    return np.sqrt(
        (dL_ / (KL * SL)) ** 2
        + (dC_ / (KC * SC)) ** 2
        + (dH_ / (KH * SH)) ** 2
        + RT * dC_ * dH_ / (KC * SC * KH * SH)
    )
