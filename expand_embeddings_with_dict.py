import torch
from safetensors.torch import load_file, save_file
from style_bert_vits2.nlp.symbols import SYMBOLS, KO_SYMBOLS, LANGUAGE_ID_MAP, NUM_LANGUAGES, LANGUAGE_TONE_START_MAP, NUM_JP_TONES, NUM_KO_TONES

def expand_embeddings(input_path: str, output_path: str) -> None:
    # Load the existing safetensors model
    tensors = load_file(input_path)

    # Extract the text_encoder related state_dict
    text_encoder_state_dict = {
        k.replace("enc_p.", ""): v for k, v in tensors.items() if k.startswith("enc_p.")
    }

    # Get the existing embeddings
    language_emb = text_encoder_state_dict["language_emb.weight"]
    tone_emb = text_encoder_state_dict["tone_emb.weight"]
    emb = text_encoder_state_dict["emb.weight"]

    # Print shapes for debugging
    print(f"language_emb.shape = {language_emb.shape}")
    print(f"tone_emb.shape = {tone_emb.shape}")
    print(f"emb.shape = {emb.shape}")

    embedding_dim = emb.shape[1]

    # 1. Update language embeddings
    new_language_emb_size = NUM_LANGUAGES  # Should be 4 (ZH, JP, EN, KO)
    new_language_emb = torch.zeros(new_language_emb_size, embedding_dim)
    new_language_emb[:language_emb.shape[0]] = language_emb

    # Copy the Japanese language embedding to Korean
    new_language_emb[LANGUAGE_ID_MAP['KO']] = language_emb[LANGUAGE_ID_MAP['JP']]

    # 2. Update tone embeddings
    num_existing_tones = tone_emb.shape[0]
    new_tone_emb_size = num_existing_tones + NUM_KO_TONES
    new_tone_emb = torch.zeros(new_tone_emb_size, embedding_dim)
    new_tone_emb[:tone_emb.shape[0]] = tone_emb

    # Initialize Korean tone embeddings
    ko_tone_start_index = num_existing_tones
    # KO tone 0: average of all existing tone embeddings
    new_tone_emb[ko_tone_start_index] = tone_emb.mean(dim=0)
    # KO tone 1: average of Japanese tone embeddings
    jp_tone_start = LANGUAGE_TONE_START_MAP['JP']
    jp_tone_end = jp_tone_start + NUM_JP_TONES
    jp_tone_indices = range(jp_tone_start, jp_tone_end)
    jp_tone_embs = tone_emb[jp_tone_indices]
    new_tone_emb[ko_tone_start_index + 1] = jp_tone_embs.mean(dim=0)

    # 3. Update symbol embeddings
    num_existing_symbols = emb.shape[0]
    new_emb_size = num_existing_symbols + len(KO_SYMBOLS)
    new_emb = torch.zeros(new_emb_size, embedding_dim)
    new_emb[:emb.shape[0]] = emb

    # Build symbol to index mapping for existing symbols
    symbol_to_index = {s: i for i, s in enumerate(SYMBOLS)}

    # Mapping from KO symbols to existing symbols (lists for averaging)
    KO_TO_EXISTING_SYMBOLS = {
        'ᄀ': ['k'],
        'ᄁ': ['k'],
        'ᄂ': ['n'],
        'ᄃ': ['d'],
        'ᄄ': ['d'],
        'ᄅ': ['r'],
        'ᄆ': ['m'],
        'ᄇ': ['b'],
        'ᄈ': ['b'],
        'ᄉ': ['s', "sh", ],
        'ᄊ': ['s'],
        'ᄋ': ['a', 'a:' 'i', 'i:' 'u', 'u:' 'e', 'e:' 'o', 'o:'],
        'ᄌ': ['j'],
        'ᄍ': ['j'],
        'ᄎ': ['ch'],
        'ᄏ': ['k'],
        'ᄐ': ['t'],
        'ᄑ': ['p'],
        'ᄒ': ['h'],
        'ᅡ': ['a', 'a:'],
        'ᅢ': ['e'],
        'ᅣ': ['y', 'a'],
        'ᅤ': ['y', 'e'],
        'ᅥ': ['o'],
        'ᅦ': ['e'],
        'ᅧ': ['y', 'o'],
        'ᅨ': ['y', 'e'],
        'ᅩ': ['o', 'o:'],
        'ᅪ': ['w', 'a'],   # 'w' + 'a'
        'ᅫ': ['w', 'e'],   # 'w' + 'e'
        'ᅬ': ['w', 'e'],   # 'w' + 'e'
        'ᅭ': ['y', 'o'],   # 'y' + 'o'
        'ᅮ': ['u', 'u:'],
        'ᅯ': ['w', 'o'],   # 'w' + 'o'
        'ᅰ': ['w', 'e'],   # 'w' + 'e'
        'ᅱ': ['w', 'i'],   # 'w' + 'i'
        'ᅲ': ['y', 'u'],   # 'y' + 'u'
        'ᅳ': ['u'],
        'ᅴ': ['i'],
        'ᅵ': ['i', 'i:'],
        'ᆨ': ['k'],
        'ᆫ': ['n', 'N'],
        'ᆮ': ['t'],
        'ᆯ': ['l'],
        'ᆷ': ['m'],
        'ᆸ': ['p'],
        'ᆼ': ['N'],
    }

    # For each KO symbol, map to the existing symbols and average their embeddings
    for i, ko_symbol in enumerate(KO_SYMBOLS):
        mapped_symbols = KO_TO_EXISTING_SYMBOLS.get(ko_symbol, [])
        embeddings = []
        for ms in mapped_symbols:
            ms_index = symbol_to_index.get(ms)
            if ms_index is not None and ms_index < num_existing_symbols:
                embeddings.append(emb[ms_index])
        if embeddings:
            # Average the embeddings of the mapped symbols
            mapped_embedding = torch.stack(embeddings).mean(dim=0)
        else:
            # Default to average embedding if no mapping is found
            mapped_embedding = emb[:num_existing_symbols].mean(dim=0)
        new_emb[num_existing_symbols + i] = mapped_embedding

    # 4. Update the state_dict with the new embeddings
    text_encoder_state_dict["language_emb.weight"] = new_language_emb
    text_encoder_state_dict["tone_emb.weight"] = new_tone_emb
    text_encoder_state_dict["emb.weight"] = new_emb

    # 5. Reflect the updated state_dict in the overall model's state_dict
    new_tensors = {**tensors, **{f"enc_p.{k}": v for k, v in text_encoder_state_dict.items()}}

    # 6. Save the updated model to a new safetensors file
    save_file(new_tensors, output_path)
    print(f"Updated model saved as {output_path}")

if __name__ == "__main__":
    # Set input and output paths
    input_model_path = "pretrained_jp_extra/G_0.safetensors"
    output_model_path = "pretrained_ko_extra/G_0.safetensors"

    # Call the function
    expand_embeddings(input_model_path, output_model_path)