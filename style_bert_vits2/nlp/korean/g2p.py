import re
import logging
from g2pk2 import G2p
from jamo import h2j
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.symbols import PUNCTUATIONS, SYMBOLS
from style_bert_vits2.logging import logger

# Initialize the G2p converter for Korean
g2p_converter = G2p()

def g2p(text: str) -> tuple[list[str], list[int], list[int]]:
    phones = []
    tones = []
    phone_len = []
    words = __text_to_words(text)
    word2ph = []

    for word in words:
        temp_phones = []
        temp_tones = []

        # Reconstruct the word from tokens
        word_str = ''.join(word)

        # Check if the word is punctuation
        if word_str in PUNCTUATIONS:
            temp_phones.append(word_str)
            temp_tones.append(0)
            word2ph.append(1)
        else:
            for w in word:
                # Get pronunciation using g2p
                pronunciation = g2p_converter(w)
                # Convert pronunciation to jamo (phonemes)
                phones_list = list(h2j(pronunciation))
                # Remove '#' characters from phones
                phones_list = [p for p in phones_list if p != '#']
                temp_phones.extend(phones_list)
                # Assign tone 1 to regular characters
                temp_tones.extend([1]*len(phones_list))
                word2ph.append(len(phones_list))

        phones.extend(temp_phones)
        tones.extend(temp_tones)
        phone_len.append(len(temp_phones))

    # word2ph = []
    # for word, pl in zip(words, phone_len):
    #     word_len = len(word)
    #     word2ph += __distribute_phone(pl, word_len)

    # Add start and end symbols
    phones = ['_'] + phones + ['_']
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    # set_trace()
    assert len(phones) == len(tones), text
    assert len(phones) == sum(word2ph), text

    return phones, tones, word2ph

# def __distribute_phone(n_phone: int, n_word: int) -> list[int]:
#     phones_per_word = [0] * n_word
#     for _ in range(n_phone):
#         min_tasks = min(phones_per_word)
#         min_index = phones_per_word.index(min_tasks)
#         phones_per_word[min_index] += 1
#     return phones_per_word

def __text_to_words(text: str) -> list[list[str]]:
    tokenizer = bert_models.load_tokenizer(Languages.KO)
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding.tokens()
    offsets = encoding['offset_mapping']

    words = []
    current_word = []
    # current_offsets = []

    for token, (start, end) in zip(tokens, offsets):
        # Handle [UNK] tokens
        if token == '[UNK]':
            # If there's an ongoing word, append it to the words list
            if current_word:
                words.append(current_word)
                current_word = []
            # Append the original text corresponding to the [UNK] token as a list
            original_text = text[start:end]
            words.append([original_text])
        # Handle subword tokens
        elif token.startswith('##'):
            current_word.append(token[2:])
            # current_offsets.append((start, end))
        else:
            if current_word:
                words.append(current_word)
            current_word = [token]
            # current_offsets = [(start, end)]

    if current_word:
        words.append(current_word)

    return words

if __name__ == "__main__":
    from style_bert_vits2.nlp.korean.normalizer import normalize_text
    sentences = [
        "안녕하세요. 저는 인공지능입니다.",
        "이 문장은 테스트용입니다.",
        "한국어 음성 합성을 위한 g2p 함수를 구현합니다.",
        "이 프로젝트는 재미있습니다!",
        "오늘 날씨가 좋네요.",
        "이 Computer는 정말 Nice하군요",
        "응, 고마워. 그 호칭도, 커피 향도 새벽녘에 들려오는 새의 지저귐처럼 가슴에 와닿는구나",
        "ㅋㅋㅋㅋㅋ",
        "쟨 이해해도 걘 이해 못할 수도 있어...",
        "ㅋㅋㅋㅋㅋ아 진짜 개웃곀ㅋㅋㅋㅋㅋ",
        "푸히힛, 올리쨩 최고!"
    ]
    for sentence in sentences:
        sentence = normalize_text(sentence)
        print(sentence)
        words = __text_to_words(sentence)
        print(f'Tokenized Words: {words}')
        phones, tones, word2ph = g2p(sentence)
        print(f'\tPhones: {phones}')
        print(f'\tTones: {tones}')
        print(f'\tWord2Ph: {word2ph}')
        print()