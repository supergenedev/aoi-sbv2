from typing import Optional

import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    한국어 텍스트에서 BERT 특징을 추출합니다.

    Args:
        text (str): 한국어 텍스트
        word2ph (list[int]): 원본 텍스트의 각 문자에 할당된 음소의 개수를 나타내는 리스트
        device (str): 추론에 사용할 장치
        assist_text (Optional[str], optional): 보조 텍스트 (기본값: None)
        assist_text_weight (float, optional): 보조 텍스트의 가중치 (기본값: 0.7)

    Returns:
        torch.Tensor: BERT feature
    """

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = bert_models.load_model(Languages.KO).to(device)  # type: ignore

    style_res_mean = None
    with torch.no_grad():
        tokenizer = bert_models.load_tokenizer(Languages.KO)
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  # type: ignore
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        if assist_text:
            style_inputs = tokenizer(assist_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)  # type: ignore
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - assist_text_weight)
                + style_res_mean.repeat(word2phone[i], 1) * assist_text_weight
            )
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 첫 1024 차원만 가져오기 (xlarge의 경우)
    phone_level_feature = phone_level_feature[:, :1024]  # Now (tokens, 1024)


    return phone_level_feature.T
