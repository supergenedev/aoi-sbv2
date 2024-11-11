import pickle
from pathlib import Path
from glob import glob
from style_bert_vits2.constants import DEFAULT_USER_DICT_DIR

CMU_DICT_PATH = Path(__file__).parent / "cmudict.rep"
CACHE_PATH = Path(__file__).parent / "cmudict_cache.pickle"
CMU_DICT_UPDATES = glob(f'{DEFAULT_USER_DICT_DIR}/cmudict_update/*.rep')

def get_dict() -> dict[str, list[list[str]]]:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        # cache_dict(g2p_dict, CACHE_PATH)
        g2p_dict = update_dict(g2p_dict)

    return g2p_dict


def read_dict() -> dict[str, list[list[str]]]:
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH, "r", encoding="utf-8") as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict: dict[str, list[list[str]]], file_path: Path) -> None:
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def update_dict(g2p_dict: dict[str, list[list[str]]]) -> dict[str, list[list[str]]]:
    for file in CMU_DICT_UPDATES:
        lines = open(file, "r").read().splitlines()
        for line in lines:
            line = line.strip()
            word_split = line.split("  ")
            word = word_split[0]
            syllable_split = word_split[1].split(" - ")
            g2p_dict[word] = []
            for syllable in syllable_split:
                phone_split = syllable.split(" ")
                g2p_dict[word].append(phone_split)
    
    return g2p_dict