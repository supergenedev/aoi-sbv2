import torch
from safetensors.torch import load_file, save_file


def expand_embeddings(input_path: str, output_path: str) -> None:
    # 1. 기존 safetensors 모델 불러오기
    tensors = load_file(input_path)

    # 2. text_encoder 관련 state_dict 생성
    text_encoder_state_dict = {
        k.replace("enc_p.", ""): v for k, v in tensors.items() if k.startswith("enc_p.")
    }

    # 3. 기존 임베딩 확장
    language_emb = text_encoder_state_dict["language_emb.weight"]
    tone_emb = text_encoder_state_dict["tone_emb.weight"]
    emb = text_encoder_state_dict["emb.weight"]

    # 새로운 임베딩 크기 설정
    new_language_emb = torch.zeros(4, 192)
    new_language_emb[:3] = language_emb
    new_language_emb[3] = language_emb.mean(dim=0)

    new_tone_emb = torch.zeros(14, 192)
    new_tone_emb[:12] = tone_emb
    new_tone_emb[12:] = tone_emb.mean(dim=0)

    new_emb = torch.zeros(159, 192)
    new_emb[:112] = emb
    new_emb[112:] = emb.mean(dim=0)

    # 4. 확장된 임베딩을 state_dict에 업데이트
    text_encoder_state_dict["language_emb.weight"] = new_language_emb
    text_encoder_state_dict["tone_emb.weight"] = new_tone_emb
    text_encoder_state_dict["emb.weight"] = new_emb

    # 5. 업데이트된 text_encoder_state_dict을 전체 모델의 state_dict에 반영
    new_tensors = {**tensors, **{f"enc_p.{k}": v for k, v in text_encoder_state_dict.items()}}

    # 6. 업데이트된 모델을 새로운 safetensors 파일에 저장
    save_file(new_tensors, output_path)
    print(f"Updated model saved as {output_path}")


if __name__ == "__main__":
    # 입력 경로와 출력 경로 설정
    input_model_path = "pretrained_jp_extra/G_0.safetensors"
    output_model_path = "pretrained_ko_extra/G_0.safetensors"

    # 함수 호출
    expand_embeddings(input_model_path, output_model_path)