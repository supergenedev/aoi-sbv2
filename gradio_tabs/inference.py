import datetime
import json
from typing import Optional

import gradio as gr

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    GRADIO_THEME,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models.infer import InvalidToneError
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.tts_model import TTSModelHolder


# pyopenjtalk_worker 시작
## pyopenjtalk_worker는 TCP 소켓 서버이므로 여기에서 시작합니다.
pyopenjtalk.initialize_worker()

# Web UI에서 학습 시 불필요한 GPU VRAM 소비를 피하기 위해 여기서는 BERT 모델의 사전 로드를 하지 않습니다.
# 데이터셋의 BERT 특징량은 bert_gen.py에 의해 사전에 추출되어 있으므로, 학습 시 BERT 모델을 로드할 필요는 없습니다.
# BERT 모델의 사전 로드는 "로드" 버튼을 누를 때 실행되는 TTSModelHolder.get_model_for_gradio() 내에서 수행됩니다.
# Web UI에서 학습할 때, 음성 합성 탭의 "로드" 버튼을 누르지 않으면 BERT 모델이 VRAM에 로드되지 않은 상태에서 학습을 시작할 수 있습니다.

languages = [lang.value for lang in Languages]

initial_text = "こんにちは、初めまして。あなたの名前はなんていうの？"

examples = [
    [initial_text, "JP"],
    [
        """あなたがそんなこと言うなんて、私はとっても嬉しい。
あなたがそんなこと言うなんて、私はとっても怒ってる。
あなたがそんなこと言うなんて、私はとっても驚いてる。
あなたがそんなこと言うなんて、私はとっても辛い。""",
        "JP",
    ],
    [  # ChatGPTに考えてもらった告白セリフ
        """私、ずっと前からあなたのことを見てきました。あなたの笑顔、優しさ、強さに、心惹かれていたんです。
友達として過ごす中で、あなたのことがだんだんと特別な存在になっていくのがわかりました。
えっと、私、あなたのことが好きです！もしよければ、私と付き合ってくれませんか？""",
        "JP",
    ],
    [  # 夏目漱石『吾輩は猫である』
        """吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。なんでも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで初めて人間というものを見た。しかもあとで聞くと、それは書生という、人間中で一番獰悪な種族であったそうだ。
この書生というのは時々我々を捕まえて煮て食うという話である。""",
        "JP",
    ],
    [  # 梶井基次郎『桜の樹の下には』
        """桜の樹の下には屍体が埋まっている！これは信じていいことなんだよ。
何故って、桜の花があんなにも見事に咲くなんて信じられないことじゃないか。俺はあの美しさが信じられないので、このにさんにち不安だった。
しかしいま、やっとわかるときが来た。桜の樹の下には屍体が埋まっている。これは信じていいことだ。""",
        "JP",
    ],
    [  # ChatGPTと考えた、感情を表すセリフ
        """やったー！テストで満点取れた！私とっても嬉しいな！
どうして私の意見を無視するの？許せない！ムカつく！あんたなんか死ねばいいのに。
あはははっ！この漫画めっちゃ笑える、見てよこれ、ふふふ、あはは。
あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しい。""",
        "JP",
    ],
    [  # 上の丁寧語バージョン
        """やりました！テストで満点取れましたよ！私とっても嬉しいです！
どうして私の意見を無視するんですか？許せません！ムカつきます！あんたなんか死んでください。
あはははっ！この漫画めっちゃ笑えます、見てくださいこれ、ふふふ、あはは。
あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しいです。""",
        "JP",
    ],
    [  # ChatGPTに考えてもらった音声合成の説明文章
        """音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。
この分野の最新の研究成果を使うと、より自然で表現豊かな音声の生成が可能である。深層学習の応用により、感情やアクセントを含む声質の微妙な変化も再現することが出来る。""",
        "JP",
    ],
    [
        "Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and can be implemented in software or hardware products.",
        "EN",
    ],
    [
        "语音合成是人工制造人类语音。用于此目的的计算机系统称为语音合成器，可以通过软件或硬件产品实现。",
        "ZH",
    ],
]

initial_md = """
- Ver 2.5에 추가된 기본 [`koharune-ami`(코하루네 아미) 모델](https://huggingface.co/litagin/sbv2_koharune_ami)과 [`amitaro`(아미타로) 모델](https://huggingface.co/litagin/sbv2_amitaro)은 [아미타로의 목소리 자료 공방](https://amitaro.net/)에서 공개된 코퍼스 음원 및 라이브 방송 음성을 사용하여 사전 허가를 얻어 학습한 모델입니다. **아래의 이용 규칙을 반드시 읽고** 사용하시기 바랍니다.

- Ver 2.5 업데이트 이후 위 모델을 다운로드하려면, `Initialize.bat`를 더블 클릭하거나 수동으로 다운로드하여 `model_assets` 디렉토리에 배치하세요.

- Ver 2.3에서 추가된 **에디터 버전**이 실제로 읽어내기에는 더 편리할 수 있습니다. `Editor.bat`를 실행하거나 `python server_editor.py --inbrowser`로 시작할 수 있습니다.
"""

terms_of_use_md = """
## 안내 및 기본 모델의 라이선스

최신 안내 및 이용 규칙은 [여기](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/TERMS_OF_USE.md)에서 확인하세요. 항상 최신 버전이 적용됩니다.

Style-Bert-VITS2를 사용할 때는 아래의 안내를 지켜주시기 바랍니다. 단, 모델의 이용 규칙 이전 부분은 어디까지나 "안내"이며 강제력은 없고, Style-Bert-VITS2의 이용 규칙이 아닙니다. 따라서 [저장소의 라이선스](https://github.com/litagin02/Style-Bert-VITS2#license)와 충돌하지 않으며, 저장소 이용에 있어서는 항상 저장소의 라이선스만이 구속력을 가집니다.

### 하지 말아야 할 일

Style-Bert-VITS2를 다음의 목적으로 사용하지 않기를 바랍니다.

- 법률을 위반하는 목적
- 정치적 목적(본래 Bert-VITS2에서 금지됨)
- 타인에게 해를 입히려는 목적
- 사칭 및 딥페이크 생성 목적

### 지켜야 할 일

- Style-Bert-VITS2를 사용할 때는 사용하려는 모델의 이용 규칙 및 라이선스를 반드시 확인하고, 존재할 경우 그에 따라야 합니다.
- 또한 소스 코드를 사용할 때는 [저장소의 라이선스](https://github.com/litagin02/Style-Bert-VITS2#license)에 따라야 합니다.

다음은 기본적으로 포함된 모델의 라이선스입니다.

### JVNV 코퍼스 (jvnv-F1-jp, jvnv-F2-jp, jvnv-M1-jp, jvnv-M2-jp)

- [JVNV 코퍼스](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus)의 라이선스는 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ko)이며, 이를 따릅니다.

### 코하루네 아미(koharune-ami) / 아미타로(amitaro)

[아미타로의 목소리 자료 공방 규칙](https://amitaro.net/voice/voice_rule/) 및 [아미타로의 라이브 방송 음성 이용 규칙](https://amitaro.net/voice/livevoice/#index_id6)을 모두 준수해야 합니다. 특히, 다음 사항을 준수하세요 (규칙을 준수하면 상업용/비상업용 관계없이 사용할 수 있습니다).

#### 금지 사항

- 연령 제한이 있는 작품/용도로 사용
- 신흥 종교/정치/다단계 관련 작품/용도로 사용
- 특정 단체, 개인 또는 국가를 비방하는 작품/용도로 사용
- 생성된 음성을 아미타로 본인의 목소리로 다루는 것
- 생성된 음성을 아미타로 이외의 사람의 목소리로 다루는 것

#### 크레딧 표기

생성된 음성을 공개할 때는 (매체 무관) 반드시 `아미타로의 목소리 자료 공방 (https://amitaro.net/)`의 목소리를 기반으로 한 음성 모델을 사용하고 있음을 알 수 있도록 눈에 띄는 곳에 크레딧 표기를 기재하세요.

크레딧 표기 예:
- `Style-BertVITS2 모델: 코하루네 아미, 아미타로의 목소리 자료 공방 (https://amitaro.net/)`
- `Style-BertVITS2 모델: 아미타로, 아미타로의 목소리 자료 공방 (https://amitaro.net/)`

#### 모델 병합

모델 병합에 대해서는 [아미타로의 목소리 자료 공방의 FAQ](https://amitaro.net/voice/faq/#index_id17)를 따르세요:
- 본 모델을 다른 모델과 병합할 수 있는 것은 그 다른 모델을 생성할 때 학습에 사용된 목소리의 권리자가 허가한 경우에 한함
- 아미타로의 목소리 특징이 남아 있는 경우 (병합 비율이 25% 이상인 경우) 해당 사용은 [아미타로의 목소리 자료 공방 규칙](https://amitaro.net/voice/voice_rule/)의 범위 내로 제한되며, 그 모델에도 이 규칙이 적용됨
"""

how_to_md = """
아래와 같이 `model_assets` 디렉토리 내에 모델 파일들을 배치하세요.
```
model_assets
├── your_model
│   ├── config.json
│   ├── your_model_file1.safetensors
│   ├── your_model_file2.safetensors
│   ├── ...
│   └── style_vectors.npy
└── another_model
    ├── ...
```
각 모델에 필요한 파일들:
- `config.json`: 학습 시 설정 파일
- `*.safetensors`: 학습된 모델 파일 (1개 이상 필요, 복수 가능)
- `style_vectors.npy`: 스타일 벡터 파일

위 두 가지 파일은 `Train.bat`를 통한 학습 시 자동으로 올바른 위치에 저장됩니다. `style_vectors.npy`는 `Style.bat`를 실행하여 지시에 따라 생성하세요.
"""

style_md = f"""
- 프리셋 또는 음성 파일을 통해 읽기 음색, 감정, 스타일 등을 조절할 수 있습니다.
- 기본 {DEFAULT_STYLE}로도 충분히 읽어내는 문장에 따라 감정적으로 풍부하게 읽어낼 수 있습니다. 이 스타일 조절은 이를 가중치로 덮어쓰는 느낌입니다.
- 강도를 너무 높이면 발음이 이상해지거나 음성이 무너지기도 합니다.
- 적절한 강도는 모델이나 스타일에 따라 다릅니다.
- 음성 파일을 입력할 경우 학습 데이터와 유사한 음색의 화자(특히 같은 성별)여야 좋은 효과가 날 수 있습니다.
"""


def make_interactive():
    return gr.update(interactive=True, value="음성 합성")


def make_non_interactive():
    return gr.update(interactive=False, value="음성 합성 (모델을 로드하십시오)")


def gr_util(item):
    if item == "프리셋에서 선택":
        return (gr.update(visible=True), gr.Audio(visible=False, value=None))
    else:
        return (gr.update(visible=False), gr.update(visible=True))


def create_inference_app(model_holder: TTSModelHolder) -> gr.Blocks:
    def tts_fn(
        model_name,
        model_path,
        text,
        language,
        reference_audio_path,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        line_split,
        split_interval,
        assist_text,
        assist_text_weight,
        use_assist_text,
        style,
        style_weight,
        kata_tone_json_str,
        use_tone,
        speaker,
        pitch_scale,
        intonation_scale,
    ):
        model_holder.get_model(model_name, model_path)
        assert model_holder.current_model is not None

        wrong_tone_message = ""
        kata_tone: Optional[list[tuple[str, int]]] = None
        if use_tone and kata_tone_json_str != "":
            if language != "JP":
                logger.warning("Only Japanese is supported for tone generation.")
                wrong_tone_message = "악센트 지정은 현재 일본어만 지원하고 있습니다."
            if line_split:
                logger.warning("Tone generation is not supported for line split.")
                wrong_tone_message = (
                    "악센트 지정은 줄 바꿈으로 구분 생성을 사용하지 않는 경우에만 지원합니다."
                )
            try:
                kata_tone = []
                json_data = json.loads(kata_tone_json_str)
                # tupleを使うように変換
                for kana, tone in json_data:
                    assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                    kata_tone.append((kana, tone))
            except Exception as e:
                logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
                wrong_tone_message = f"악센트 지정이 잘못되었습니다: {e}"
                kata_tone = None

        # toneは実際に音声合成に代入される際のみnot Noneになる
        tone: Optional[list[int]] = None
        if kata_tone is not None:
            phone_tone = kata_tone2phone_tone(kata_tone)
            tone = [t for _, t in phone_tone]

        speaker_id = model_holder.current_model.spk2id[speaker]

        start_time = datetime.datetime.now()

        try:
            sr, audio = model_holder.current_model.infer(
                text=text,
                language=language,
                reference_audio_path=reference_audio_path,
                sdp_ratio=sdp_ratio,
                noise=noise_scale,
                noise_w=noise_scale_w,
                length=length_scale,
                line_split=line_split,
                split_interval=split_interval,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                use_assist_text=use_assist_text,
                style=style,
                style_weight=style_weight,
                given_tone=tone,
                speaker_id=speaker_id,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
            )
        except InvalidToneError as e:
            logger.error(f"Tone error: {e}")
            return f"Error: 악센트 지정이 잘못되었습니다:\n{e}", None, kata_tone_json_str
        except ValueError as e:
            logger.error(f"Value error: {e}")
            return f"Error: {e}", None, kata_tone_json_str

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        if tone is None and language == "JP":
            # アクセント指定に使えるようにアクセント情報を返す
            # (번역) 악센트 지정에 사용할 수 있도록 악센트 정보를 반환합니다.
            norm_text = normalize_text(text)
            kata_tone = g2kata_tone(norm_text)
            kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
        elif tone is None:
            kata_tone_json_str = ""
        message = f"Success, time: {duration} seconds."
        if wrong_tone_message != "":
            message = wrong_tone_message + "\n" + message
        return message, (sr, audio), kata_tone_json_str

    model_names = model_holder.model_names
    if len(model_names) == 0:
        logger.error(
            f"모델을 찾을 수 없습니다.{model_holder.root_dir}에 모델을 올려주세요."
        )
        with gr.Blocks() as app:
            gr.Markdown(
                f"Error: モデルが見つかりませんでした。{model_holder.root_dir}에 모델을 올려주세요."
            )
        return app
    initial_id = 0
    initial_pth_files = [
        str(f) for f in model_holder.model_files_dict[model_names[initial_id]]
    ]

    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(initial_md)
        gr.Markdown(terms_of_use_md)
        with gr.Accordion(label="使い方", open=False):
            gr.Markdown(how_to_md)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=3):
                        model_name = gr.Dropdown(
                            label="モデル一覧",
                            choices=model_names,
                            value=model_names[initial_id],
                        )
                        model_path = gr.Dropdown(
                            label="モデルファイル",
                            choices=initial_pth_files,
                            value=initial_pth_files[0],
                        )
                    refresh_button = gr.Button("업데이트", scale=1, visible=True)
                    load_button = gr.Button("로드", scale=1, variant="primary")
                text_input = gr.TextArea(label="텍스트", value=initial_text)
                pitch_scale = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    value=1,
                    step=0.05,
                    label="Pitch(1 이외에는 음질 저하)",
                )
                intonation_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=1,
                    step=0.1,
                    label="억양(1 이외에는 음질 저하)",
                )

                line_split = gr.Checkbox(
                    label="줄바꿈으로 구분하여 생성 (구분하는 것이 감정이입이 잘 됨)",
                    value=DEFAULT_LINE_SPLIT,
                )
                split_interval = gr.Slider(
                    minimum=0.0,
                    maximum=2,
                    value=DEFAULT_SPLIT_INTERVAL,
                    step=0.1,
                    label="개행으로 삽입되는 무음 구간의 길이(초)",
                )
                line_split.change(
                    lambda x: (gr.Slider(visible=x)),
                    inputs=[line_split],
                    outputs=[split_interval],
                )
                tone = gr.Textbox(
                    label="악센트 조정(0=낮음 또는 1=높음만 가능)",
                    info="줄 바꿈으로 구분할 수 있습니다.",
                )
                use_tone = gr.Checkbox(label="악센트 조정 사용", value=False)
                use_tone.change(
                    lambda x: (gr.Checkbox(value=False) if x else gr.Checkbox()),
                    inputs=[use_tone],
                    outputs=[line_split],
                )
                language = gr.Dropdown(choices=languages, value="JP", label="Language")
                speaker = gr.Dropdown(label="화자")
                with gr.Accordion(label="세부 설정", open=False):
                    sdp_ratio = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_SDP_RATIO,
                        step=0.1,
                        label="SDP Ratio",
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISE,
                        step=0.1,
                        label="Noise",
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISEW,
                        step=0.1,
                        label="Noise_W",
                    )
                    length_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_LENGTH,
                        step=0.1,
                        label="Length",
                    )
                    use_assist_text = gr.Checkbox(
                        label="Assist text 사용", value=False
                    )
                    assist_text = gr.Textbox(
                        label="Assist text",
                        placeholder="どうして私の意見を無視するの？許せない、ムカつく！死ねばいいのに。",
                        info="이 텍스트 낭독과 비슷한 음색과 감정이 나오기 쉽습니다. 다만 억양이나 템포 등이 희생되는 경향이 있습니다.",
                        visible=False,
                    )
                    assist_text_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_ASSIST_TEXT_WEIGHT,
                        step=0.1,
                        label="Assist textの強さ",
                        visible=False,
                    )
                    use_assist_text.change(
                        lambda x: (gr.Textbox(visible=x), gr.Slider(visible=x)),
                        inputs=[use_assist_text],
                        outputs=[assist_text, assist_text_weight],
                    )
            with gr.Column():
                with gr.Accordion("스타일에 대해 자세히 알아보기", open=False):
                    gr.Markdown(style_md)
                style_mode = gr.Radio(
                    ["프리셋에서 선택", "음성 파일 적용"],
                    label="스타일 지정 방법",
                    value="프리셋에서 선택",
                )
                style = gr.Dropdown(
                    label=f"스타일({DEFAULT_STYLE}이 평균 스타일)",
                    choices=["모델 불러오기"],
                    value="모델 불러오기",
                )
                style_weight = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=DEFAULT_STYLE_WEIGHT,
                    step=0.1,
                    label="스타일 강도 (목소리가 무너지면 작게 하십시오)",
                )
                ref_audio_path = gr.Audio(
                    label="참조 음성", type="filepath", visible=False
                )
                tts_button = gr.Button(
                    "음성 합성 (모델을 로드하십시오)",
                    variant="primary",
                    interactive=False,
                )
                text_output = gr.Textbox(label="정보")
                audio_output = gr.Audio(label="합성 결과물")
                with gr.Accordion("텍스트 예시", open=False):
                    gr.Examples(examples, inputs=[text_input, language])

        tts_button.click(
            tts_fn,
            inputs=[
                model_name,
                model_path,
                text_input,
                language,
                ref_audio_path,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                line_split,
                split_interval,
                assist_text,
                assist_text_weight,
                use_assist_text,
                style,
                style_weight,
                tone,
                use_tone,
                speaker,
                pitch_scale,
                intonation_scale,
            ],
            outputs=[text_output, audio_output, tone],
        )

        model_name.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name],
            outputs=[model_path],
        )

        model_path.change(make_non_interactive, outputs=[tts_button])

        refresh_button.click(
            model_holder.update_model_names_for_gradio,
            outputs=[model_name, model_path, tts_button],
        )

        load_button.click(
            model_holder.get_model_for_gradio,
            inputs=[model_name, model_path],
            outputs=[style, tts_button, speaker],
        )

        style_mode.change(
            gr_util,
            inputs=[style_mode],
            outputs=[style, ref_audio_path],
        )

    return app


if __name__ == "__main__":
    from config import get_path_config
    import torch

    path_config = get_path_config()
    assets_root = path_config.assets_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_holder = TTSModelHolder(assets_root, device)
    app = create_inference_app(model_holder)
    app.launch(inbrowser=True)
