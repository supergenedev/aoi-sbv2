# aoi-sbv2

Style-Bert-VITS2 한국어 확장 및 모델 축소 등의 실험을 위한 프로젝트

## 소개
aoi-sbv2는 Style-Bert-VITS2 모델을 기반으로 한국어 지원 확장과 모델 경량화를 위한 다양한 실험을 수행하는 프로젝트입니다.
Style-Bert-VITS2의 ver 2.6.1를 기반으로 프로젝트를 시작하였으며, 한국어로의 언어 확장, 모델의 파라미터 수를 줄이는 경량화 모델을 실험합니다.


## 주요 목표
- **한국어 지원 확장**: 기존 모델에 한국어 데이터를 추가하여 한국어 TTS 성능을 향상시킵니다.
- **모델 축소 및 경량화**: 모델의 파라미터 수를 줄여 효율적인 경량화 모델을 실험합니다.

## 시작하기

### 요구 사항

- Python 3.8 이상
- PyTorch, NVIDIA CUDA 등 상세한 라이브러리는 설치 방법 참고하세요.

### 설치

```bash
git clone https://github.com/supergenedev/aoi-sbv2.git
cd aoi-sbv2
python -m venv venv
source venv/bin/activate
pip install "torch<2.4" "torchaudio<2.4" --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python initialize.py  # 필요한 모델과 기본 TTS 모델을 다운로드합니다.

```

### 실행

#### CLI
- `docs/CLI.md`를 통해 

#### 앱 실행

    ```bash
    python app.py 
    python app.py --share # 외부 데모 시연 필요 시
    ```

(탭별 실행 방법은 앱 혹은 `docs/APP.md`를 참고하세요)

### ROADMAP

- [  ] 문서 및 주요 코드 한글화
- [  ] 한국어 텍스트 처리
- [  ] 한국어 g2p 구현
- [  ] BERT 경량화 pre-train
- [  ] VITS2 경량화 실험


## 참고 자료
- [Style-Bert-VITS2 원본 리포지토리](https://github.com/litagin02/Style-Bert-VITS2)
- [BERT-VITS2](https://github.com/fishaudio/Bert-VITS2)