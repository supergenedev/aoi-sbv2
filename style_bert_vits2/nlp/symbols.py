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
    "ㅏ",  # a
    "ㅐ",  # ae
    "ㅂ",  # b
    "ㅊ",  # ch
    "ㄷ",  # d
    "ㅔ",  # e
    "ㅓ",  # eo
    "ㅡ",  # eu
    "ㅍ",  # f
    "ㄱ",  # g
    "ㅎ",  # h
    "ㅣ",  # i
    "ㅈ",  # j
    "ㅋ",  # k
    "ㄹ",  # l (or r as per phonetic context)
    "ㅁ",  # m
    "ㄴ",  # n
    "ㅇ",  # ng (or placeholder for nasalized sounds)
    "ㅗ",  # o
    "ㅍ",  # p
    "ㄹ",  # r (when representing r sounds specifically)
    "ㅅ",  # s
    "ㅌ",  # t
    "ㅜ",  # u
    "ㅂ",  # v (often approximated with "ㅂ" in Korean phonetics)
    "ㅈ",  # z (often approximated with "ㅈ" in phonetic contexts)
]

NUM_KO_TONES = 4 # 초성, 중성, 종성, punctuation

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
