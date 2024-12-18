import re
import unicodedata

from num2words import num2words
from g2pk2 import G2p

from style_bert_vits2.logging import logger

# 사용할 문장 부호 목록
PUNCTUATIONS = [".", ",", "!", "?", "'", "-", "..."]

# 기호 치환 맵
__REPLACE_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    # 하이픈 및 대시를 표준 하이픈으로 변환
    "\u2010": "-",  # ‐
    "\u2013": "-",  # –
    "\u2014": "-",  # —
    "\u2015": "-",  # ―
    "\u2212": "-",  # −
    "\u2500": "-",  # ─
    "～": "-",      # ～, wave dash
    "〜": "-",      # 〜, fullwidth tilde
    "~": "-",       # ~
    "「": "'",
    "」": "'",
    "『": "'",
    "』": "'"
}

# 기호 치환 패턴 컴파일 (빈 문자열 키 제외)
__REPLACE_PATTERN = re.compile("|".join(re.escape(p) for p in __REPLACE_MAP if p))

# 허용되는 문자 패턴: 한글, 영문자, 공백, 특정 문장 부호
__VALID_CHAR_PATTERN = re.compile(
    r"[^\uAC00-\uD7A3"  # 한글 음절
    + r"\u0041-\u005A\u0061-\u007A"  # 영문자 A-Z a-z
    + r"\s"  # 공백 문자 추가
    + "".join(re.escape(p) for p in PUNCTUATIONS)
    + r"]+"
)

# 통화 기호 맵
__CURRENCY_MAP = {"$": "달러", "₩": "원", "£": "파운드", "€": "유로", "¥": "엔"}

# 통화 패턴
__CURRENCY_PATTERN = re.compile(r"([$₩£€¥])([0-9.,]*[0-9])")

# 숫자 패턴 (소수 포함)
__NUMBER_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)?")

# 천 단위 구분자 패턴
__NUMBER_WITH_SEPARATOR_PATTERN = re.compile(r"[0-9]{1,3}(,[0-9]{3})+")

# 서수 패턴
__ORDINAL_PATTERN = re.compile(r"([0-9]+)(번째|째)")

# 시간 패턴
__TIME_PATTERN = re.compile(r"([0-9]{1,2}):([0-9]{1,2})")

# 단위 명사 리스트
COUNTER_UNITS = ['개', '명', '권', '잔', '대', '장', '마리', '살', '번', '시간', '분', '병', '그루', '송이', '켤레', '채', '판', '곡', '줄', '쌍', '배']

# 단위 명사 패턴 생성
COUNTER_PATTERN = re.compile(r"([0-9]+)\s*(" + '|'.join(COUNTER_UNITS) + r")")

g2p = G2p()
ENGLISH_PATTERN = re.compile(r"[a-zA-Z]+")
__ALPHABET_TO_KOREAN_MAP = {
    "a": "에이",
    "b": "비",
    "c": "씨",
    "d": "디",
    "e": "이",
    "f": "에프",
    "g": "지",
    "h": "에이치",
    "i": "아이",
    "j": "제이",
    "k": "케이",
    "l": "엘",
    "m": "엠",
    "n": "엔",
    "o": "오",
    "p": "피",
    "q": "큐",
    "r": "알",
    "s": "에스",
    "t": "티",
    "u": "유",
    "v": "브이",
    "w": "더블유",
    "x": "엑스",
    "y": "와이",
    "z": "지",
}

def normalize_text(text: str) -> str:
    res = unicodedata.normalize("NFKC", text)
    res = __TIME_PATTERN.sub(lambda m: __expand_time(m), res)
    res = __ORDINAL_PATTERN.sub(lambda m: __expand_ordinal(m), res)  # 서수 변환을 숫자 변환보다 먼저 수행
    res = COUNTER_PATTERN.sub(lambda m: __expand_counter(m), res)  # 수량 표현 변환 추가
    res = __convert_numbers_to_words(res)
    if ENGLISH_PATTERN.search(res):
        logger.info(f"English words found in the text: {text}")
        res = __convert_english_to_korean(res)
    res = replace_punctuation(res)
    return res

def replace_punctuation(text: str) -> str:
    # 기호 치환
    replaced_text = __REPLACE_PATTERN.sub(lambda x: __REPLACE_MAP.get(x.group(), x.group()), text)
    # 허용되지 않는 문자 제거
    replaced_text = __VALID_CHAR_PATTERN.sub("", replaced_text)
    return replaced_text

def __convert_english_to_korean(text: str) -> str:
    # 영문자 변환
    def replace_english(match):
        word = match.group()
        # Step 1: Use g2p to convert English to Korean pronunciation
        converted = g2p(word)
        # Step 2: If English letters remain, convert them using the map
        if re.search(r'[a-zA-Z]', converted):
            letters = [__ALPHABET_TO_KOREAN_MAP.get(ch.lower(), ch) for ch in word]
            converted = ' '.join(letters)
        return converted

    # Replace English words in the text
    return ENGLISH_PATTERN.sub(replace_english, text)

def __convert_numbers_to_words(text: str) -> str:
    # 천 단위 구분자 제거
    res = __NUMBER_WITH_SEPARATOR_PATTERN.sub(lambda m: m.group().replace(",", ""), text)
    # 통화 기호 변환
    res = __CURRENCY_PATTERN.sub(lambda m: __expand_currency(m), res)
    # 숫자 변환
    res = __NUMBER_PATTERN.sub(lambda m: __expand_number(m.group()), res)
    return res

def __expand_currency(match):
    symbol = match.group(1)
    amount = match.group(2).replace(",", "")
    try:
        amount_in_words = num2words(float(amount), lang='ko')
    except ValueError:
        amount_in_words = amount  # 숫자가 아닌 경우 원래 문자열 유지
    currency_name = __CURRENCY_MAP.get(symbol, "")
    return f"{amount_in_words} {currency_name}"  # 금액과 통화 단위 사이에 공백 추가

def __expand_number(number_str):
    # 소수점 여부 확인
    if '.' in number_str:
        parts = number_str.split('.')
        integer_part = num2words(int(parts[0]), lang='ko')
        decimal_part = ''.join(num2words(int(digit), lang='ko') for digit in parts[1])
        return f"{integer_part} 쩜 {decimal_part}"
    else:
        return num2words(int(number_str), lang='ko')

def __expand_ordinal(match):
    number = int(match.group(1))
    suffix = match.group(2)
    # 서수 변환을 위한 매핑
    ordinal_numbers = {
        1: "첫",
        2: "두",
        3: "세",
        4: "네",
        5: "다섯",
        6: "여섯",
        7: "일곱",
        8: "여덟",
        9: "아홉",
        10: "열",
        20: "스무",
        30: "서른",
        40: "마흔",
        50: "쉰",
        60: "예순",
        70: "일흔",
        80: "여든",
        90: "아흔",
    }
    if number in ordinal_numbers:
        ordinal_word = ordinal_numbers[number]
    else:
        tens = (number // 10) * 10
        ones = number % 10
        tens_word = ordinal_numbers.get(tens, num2words(tens, lang='ko') if tens else '')
        ones_word = ordinal_numbers.get(ones, num2words(ones, lang='ko') if ones else '')
        ordinal_word = tens_word + ones_word
    return f"{ordinal_word}{suffix}"

def __expand_time(match):
    hours = int(match.group(1))
    minutes = int(match.group(2))
    hours_in_words = __convert_hour(hours)
    minutes_in_words = __convert_minute(minutes)
    return f"{hours_in_words}시 {minutes_in_words}분"

def __convert_hour(hour):
    hour_words = {
        1: "한",
        2: "두",
        3: "세",
        4: "네",
        5: "다섯",
        6: "여섯",
        7: "일곱",
        8: "여덟",
        9: "아홉",
        10: "열",
        11: "열한",
        12: "열두",
    }
    return hour_words.get(hour, num2words(hour, lang='ko'))

def __convert_minute(minute):
    if minute == 0:
        return ""
    else:
        return num2words(minute, lang='ko')

def __expand_counter(match):
    number = int(match.group(1))
    unit = match.group(2)
    # 고유어 수사 변환을 위한 매핑
    native_numbers = {
        1: '한',
        2: '두',
        3: '세',
        4: '네',
        5: '다섯',
        6: '여섯',
        7: '일곱',
        8: '여덟',
        9: '아홉',
        10: '열',
        20: '스물',
        30: '서른',
        40: '마흔',
        50: '쉰',
        60: '예순',
        70: '일흔',
        80: '여든',
        90: '아흔',
    }
    if number in native_numbers:
        number_word = native_numbers[number]
    else:
        tens = (number // 10) * 10
        ones = number % 10
        tens_word = native_numbers.get(tens, num2words(tens, lang='ko') if tens else '')
        ones_word = native_numbers.get(ones, num2words(ones, lang='ko') if ones else '')
        number_word = tens_word + ones_word
    return f"{number_word} {unit}"

# 테스트 코드
if __name__ == "__main__":
    sample_text = [
        "오늘은 2023년 7월 15일입니다.",
        "현재 시간은 3:45입니다.",
        "상품의 가격은 12,345원이며, 할인된 가격은 $10,000입니다.",
        "제 2번째 시도입니다.",
        "소수점 예시: 3.14는 원주율입니다.",
        "나는 사과를 3개 먹었습니다.",
        "나무를 5그루 심었습니다.",
        "나무를 25그루 심었습니다.",
        "나무를 125 그루 심었습니다.",
        "첫 번째, 두 번째, 세 번째로 발표했습니다.",
        "난 90퍼센트 확신해",
        "어쨌든 촬영을 무사히 끝내서 다행이야. 1등상을 타낼 자신이 있다고!",
        "…크흠! 이 얘긴 여기까지 하고——사람들을 피해 2층으로 올라온 걸 보니 아직 인기에 적응을 못했구나? 슈퍼스타였던 이 푸리나 님의 노하우를 들어볼래?",
        "그래. 그럼 모두가 보는 앞에서 그 1분 동안 일어났던 일을 낱낱이 설명해 줄 수 있겠네",
        "…엥? 잠깐, 베스트셀러 1위로 등극했던 그 미스터리 소설?",
        "나 포칼로스, 푸리나·드·폰타인은 7명의 집정관 중 하나이자 「모든 물과 백성을, 그리고 법을 다스리는 여왕」이라고… 신이 아닐 리가 없잖아",
        "이 자리를 500년이 넘는 세월 동안 지켜온 나야. 그만큼 오래 살 수 있었던 이유는 뭔데?",
        "「내」 이야기는 막을 내렸지만, 이젠 「우리」 이야기가 시작될 차례야… 생각해 보니까 그럼 출연료를 2배로 받을 수 있겠네? 신난다!",
        "하지만 30초쯤까지 셌을 때 쿵 하는 소리가 났는걸? 모두가 들었을 정도로 큰 소리였지",
        "「정의의 수호자」? 설마 20년 전에 밤마다 활약했다는 히어로 말이야?",
        "Hello world",
        "HP가 얼마 안 남았어...",
        "혹은 바로 교환하지 않고 모집을 통해 해당 학생을 얻고 나서 신비해방의 그 학생의 LF를 사용하는 방법도 있습니다.",
    ]
    for text in sample_text:
        normalized_text = normalize_text(text)
        print(normalized_text)