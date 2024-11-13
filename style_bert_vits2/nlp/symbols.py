# Punctuations
PUNCTUATIONS = ["!", "?", "…", ",", ".", "'", "-"]

# Punctuations and special tokens
PUNCTUATION_SYMBOLS = PUNCTUATIONS + ["SP", "UNK"]

# Padding
PAD = "_"

# Chinese symbols
ZH_SYMBOLS = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
NUM_ZH_TONES = 6

# Japanese
JP_SYMBOLS = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "w",
    "y",
    "z",
    "zy",
]
NUM_JP_TONES = 2

# English
EN_SYMBOLS = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
NUM_EN_TONES = 4

KO_SYMBOLS = [
    ### 한국어 표준 발음법을 참고하여 자음 19개, 모음 21개, 받침 7개
    # 자음
    'ᄀ',
    'ᄁ',
    'ᄂ',
    'ᄃ',
    'ᄄ',
    'ᄅ',
    'ᄆ',
    'ᄇ',
    'ᄈ',
    'ᄉ',
    'ᄊ',
    'ᄋ',
    'ᄌ',
    'ᄍ',
    'ᄎ',
    'ᄏ',
    'ᄐ',
    'ᄑ',
    'ᄒ',
    # 모음
    'ᅡ',
    'ᅢ',
    'ᅣ',
    'ᅤ',
    'ᅥ',
    'ᅦ',
    'ᅧ',
    'ᅨ',
    'ᅩ',
    'ᅪ',
    'ᅫ',
    'ᅬ',
    'ᅭ',
    'ᅮ',
    'ᅯ',
    'ᅰ',
    'ᅱ',
    'ᅲ',
    'ᅳ',
    'ᅴ',
    'ᅵ',
    # 받침
    'ᆨ',
    'ᆫ',
    'ᆮ',
    'ᆯ',
    'ᆷ',
    'ᆸ',
    'ᆼ',
]

NUM_KO_TONES = 2 # 정상 / punctuation (symbol에 이미 초중종성에 대한 정보가 내포되어 있기 때문에)

# Combine all symbols
NORMAL_SYMBOLS = sorted(set(ZH_SYMBOLS + JP_SYMBOLS + EN_SYMBOLS))
# SYMBOLS = [PAD] + NORMAL_SYMBOLS + PUNCTUATION_SYMBOLS (기존)
# 인덱스 유지를 위해 KO_SYMBOLS를 추가하고, PUNCTUATION_SYMBOLS를 뒤로 보냄
# 추후에 모델을 아예 새로 만든다면 `[PAD + PUNCTUATION_SYMBOLS + sorted(set(ZH_SYMBOLS + JP_SYMBOLS + EN_SYMBOLS + KO_SYMBOLS))]`로 변경
SYMBOLS = [PAD] + NORMAL_SYMBOLS + PUNCTUATION_SYMBOLS + KO_SYMBOLS 
SIL_PHONEMES_IDS = [SYMBOLS.index(i) for i in PUNCTUATION_SYMBOLS]

# Combine all tones
NUM_TONES = NUM_ZH_TONES + NUM_JP_TONES + NUM_EN_TONES + NUM_KO_TONES

# Language maps
LANGUAGE_ID_MAP = {"ZH": 0, "JP": 1, "EN": 2, "KO": 3}
NUM_LANGUAGES = len(LANGUAGE_ID_MAP.keys())

# Language tone start map
LANGUAGE_TONE_START_MAP = {
    "ZH": 0,
    "JP": NUM_ZH_TONES,
    "EN": NUM_ZH_TONES + NUM_JP_TONES,
    "KO": NUM_ZH_TONES + NUM_JP_TONES + NUM_EN_TONES,
}


if __name__ == "__main__":
    a = set(ZH_SYMBOLS)
    b = set(EN_SYMBOLS)
    print(sorted(a & b))
