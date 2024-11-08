import json
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import gradio as gr
import yaml

from config import get_path_config
from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
from style_bert_vits2.utils.subprocess import run_script_with_log, second_elem_of


logger_handler = None
tensorboard_executed = False

path_config = get_path_config()
dataset_root = path_config.dataset_root


@dataclass
class PathsForPreprocess:
    dataset_path: Path
    esd_path: Path
    train_path: Path
    val_path: Path
    config_path: Path


def get_path(model_name: str) -> PathsForPreprocess:
    assert model_name != "", "모델 이름은 비어 있을 수 없습니다."
    dataset_path = dataset_root / model_name
    esd_path = dataset_path / "esd.list"
    train_path = dataset_path / "train.list"
    val_path = dataset_path / "val.list"
    config_path = dataset_path / "config.json"
    return PathsForPreprocess(dataset_path, esd_path, train_path, val_path, config_path)


def initialize(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    freeze_EN_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    log_interval: int,
):
    global logger_handler
    paths = get_path(model_name)

    # 전처리 로그를 파일에 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"preprocess_{timestamp}.log"
    if logger_handler is not None:
        logger.remove(logger_handler)
    logger_handler = logger.add(paths.dataset_path / file_name)

    logger.info(
        f"Step 1: 초기화 시작...\n모델 이름: {model_name}, 배치 크기: {batch_size}, 에포크: {epochs}, 저장 간격 (스텝): {save_every_steps}, freeze_ZH_bert: {freeze_ZH_bert}, freeze_JP_bert: {freeze_JP_bert}, freeze_EN_bert: {freeze_EN_bert}, freeze_style: {freeze_style}, freeze_decoder: {freeze_decoder}, use_jp_extra: {use_jp_extra}"
    )

    default_config_path = (
        "configs/config.json" if not use_jp_extra else "configs/config_jp_extra.json"
    )

    with open(default_config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["model_name"] = model_name
    config["data"]["training_files"] = str(paths.train_path)
    config["data"]["validation_files"] = str(paths.val_path)
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs
    config["train"]["eval_interval"] = save_every_steps
    config["train"]["log_interval"] = log_interval

    config["train"]["freeze_EN_bert"] = freeze_EN_bert
    config["train"]["freeze_JP_bert"] = freeze_JP_bert
    config["train"]["freeze_ZH_bert"] = freeze_ZH_bert
    config["train"]["freeze_style"] = freeze_style
    config["train"]["freeze_decoder"] = freeze_decoder

    config["train"]["bf16_run"] = False  # 기본적으로 False임

    # 현재는 기본 설정이지만, 이전에는 JP-Extra 버전이 아니어서 오류 원인이 될 수 있음
    config["data"]["use_jp_extra"] = use_jp_extra

    model_path = paths.dataset_path / "models"
    if model_path.exists():
        logger.warning(
            f"Step 1: {model_path}이 이미 존재하므로 백업을 위해 {model_path}_backup으로 복사합니다."
        )
        shutil.copytree(
            src=model_path,
            dst=paths.dataset_path / "models_backup",
            dirs_exist_ok=True,
        )
        shutil.rmtree(model_path)
    pretrained_dir = Path("pretrained" if not use_jp_extra else "pretrained_jp_extra")
    try:
        shutil.copytree(
            src=pretrained_dir,
            dst=model_path,
        )
    except FileNotFoundError:
        logger.error(f"Step 1: {pretrained_dir} 폴더를 찾을 수 없습니다.")
        return False, f"Step 1, 오류: {pretrained_dir} 폴더가 없습니다."

    with open(paths.config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if not Path("config.yml").exists():
        shutil.copy(src="default_config.yml", dst="config.yml")
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = str(paths.dataset_path)
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)
    logger.success("Step 1: 초기화 완료.")
    return True, "Step 1, 성공: 초기 설정이 완료되었습니다."


def resample(model_name: str, normalize: bool, trim: bool, num_processes: int):
    logger.info("Step 2: 리샘플링 시작...")
    dataset_path = get_path(model_name).dataset_path
    input_dir = dataset_path / "raw"
    output_dir = dataset_path / "wavs"
    cmd = [
        "resample.py",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "--num_processes",
        str(num_processes),
        "--sr",
        "44100",
    ]
    if normalize:
        cmd.append("--normalize")
    if trim:
        cmd.append("--trim")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error("Step 2: 리샘플링 실패.")
        return False, f"Step 2, 오류: 음성 파일 전처리에 실패했습니다:\n{message}"
    elif message:
        logger.warning("Step 2: 리샘플링 완료 (stderr 있음).")
        return True, f"Step 2, 성공: 음성 파일 전처리가 완료되었습니다:\n{message}"
    logger.success("Step 2: 리샘플링 완료.")
    return True, "Step 2, 성공: 음성 파일 전처리가 완료되었습니다."


def preprocess_text(
    model_name: str, use_jp_extra: bool, val_per_lang: int, yomi_error: str
):
    logger.info("Step 3: 텍스트 전처리 시작...")
    paths = get_path(model_name)
    if not paths.esd_path.exists():
        logger.error(f"Step 3: {paths.esd_path}가 없습니다.")
        return (
            False,
            f"Step 3, 오류: 텍스트 파일 {paths.esd_path}을 찾을 수 없습니다.",
        )

    cmd = [
        "preprocess_text.py",
        "--config-path",
        str(paths.config_path),
        "--transcription-path",
        str(paths.esd_path),
        "--train-path",
        str(paths.train_path),
        "--val-path",
        str(paths.val_path),
        "--val-per-lang",
        str(val_per_lang),
        "--yomi_error",
        yomi_error,
        "--correct_path",  # 음성 파일 경로 수정
    ]
    if use_jp_extra:
        cmd.append("--use_jp_extra")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error("Step 3: 텍스트 전처리 실패.")
        return (
            False,
            f"Step 3, 오류: 텍스트 전처리에 실패했습니다:\n{message}",
        )
    elif message:
        logger.warning("Step 3: 텍스트 전처리 완료 (stderr 있음).")
        return (
            True,
            f"Step 3, 성공: 텍스트 전처리가 완료되었습니다:\n{message}",
        )
    logger.success("Step 3: 텍스트 전처리 완료.")
    return True, "Step 3, 성공: 텍스트 전처리가 완료되었습니다."


def bert_gen(model_name: str):
    logger.info("Step 4: BERT 생성 시작...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        ["bert_gen.py", "--config", str(config_path)]
    )
    if not success:
        logger.error("Step 4: BERT 생성 실패.")
        return False, f"Step 4, 오류: BERT 특성 파일 생성 실패:\n{message}"
    elif message:
        logger.warning("Step 4: BERT 생성 완료 (stderr 있음).")
        return (
            True,
            f"Step 4, 성공: BERT 특성 파일 생성 완료:\n{message}",
        )
    logger.success("Step 4: BERT 생성 완료.")
    return True, "Step 4, 성공: BERT 특성 파일 생성 완료."


def style_gen(model_name: str, num_processes: int):
    logger.info("Step 5: 스타일 생성 시작...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        [
            "style_gen.py",
            "--config",
            str(config_path),
            "--num_processes",
            str(num_processes),
        ]
    )
    if not success:
        logger.error("Step 5: 스타일 생성 실패.")
        return (
            False,
            f"Step 5, 오류: 스타일 특성 파일 생성 실패:\n{message}",
        )
    elif message:
        logger.warning("Step 5: 스타일 생성 완료 (stderr 있음).")
        return (
            True,
            f"Step 5, 성공: 스타일 특성 파일 생성 완료:\n{message}",
        )
    logger.success("Step 5: 스타일 생성 완료.")
    return True, "Step 5, 성공: 스타일 특성 파일 생성 완료."


def preprocess_all(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    num_processes: int,
    normalize: bool,
    trim: bool,
    freeze_EN_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    val_per_lang: int,
    log_interval: int,
    yomi_error: str,
):
    if model_name == "":
        return False, "오류: 모델 이름을 입력해 주세요."
    success, message = initialize(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        save_every_steps=save_every_steps,
        freeze_EN_bert=freeze_EN_bert,
        freeze_JP_bert=freeze_JP_bert,
        freeze_ZH_bert=freeze_ZH_bert,
        freeze_style=freeze_style,
        freeze_decoder=freeze_decoder,
        use_jp_extra=use_jp_extra,
        log_interval=log_interval,
    )
    if not success:
        return False, message
    success, message = resample(
        model_name=model_name,
        normalize=normalize,
        trim=trim,
        num_processes=num_processes,
    )
    if not success:
        return False, message

    success, message = preprocess_text(
        model_name=model_name,
        use_jp_extra=use_jp_extra,
        val_per_lang=val_per_lang,
        yomi_error=yomi_error,
    )
    if not success:
        return False, message
    success, message = bert_gen(model_name=model_name)  # bert_gen은 무거우므로 프로세스 수를 조정하지 않음
    if not success:
        return False, message
    success, message = style_gen(model_name=model_name, num_processes=num_processes)
    if not success:
        return False, message
    logger.success("성공: 모든 전처리가 완료되었습니다!")
    return (
        True,
        "성공: 모든 전처리가 완료되었습니다. 터미널에서 이상 여부를 확인해 보시기 바랍니다.",
    )


def train(
    model_name: str,
    skip_style: bool = False,
    use_jp_extra: bool = True,
    speedup: bool = False,
    not_use_custom_batch_sampler: bool = False,
):
    paths = get_path(model_name)
    # 학습 재개 시 config.yml의 이름 등 업데이트 필요
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = str(paths.dataset_path)
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)

    train_py = "train_ms.py" if not use_jp_extra else "train_ms_jp_extra.py"
    cmd = [
        train_py,
        "--config",
        str(paths.config_path),
        "--model",
        str(paths.dataset_path),
    ]
    if skip_style:
        cmd.append("--skip_default_style")
    if speedup:
        cmd.append("--speedup")
    if not_use_custom_batch_sampler:
        cmd.append("--not_use_custom_batch_sampler")
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        logger.error("학습 실패.")
        return False, f"오류: 학습에 실패했습니다:\n{message}"
    elif message:
        logger.warning("학습 완료 (stderr 있음).")
        return True, f"성공: 학습이 완료되었습니다:\n{message}"
    logger.success("학습 완료.")
    return True, "성공: 학습이 완료되었습니다."


def wait_for_tensorboard(port: int = 6006, timeout: float = 10):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True  # 포트가 열려 있을 경우
        except OSError:
            pass  # 포트가 아직 열리지 않은 경우

        if time.time() - start_time > timeout:
            return False  # 타임아웃

        time.sleep(0.1)


def run_tensorboard(model_name: str):
    global tensorboard_executed
    if not tensorboard_executed:
        python = sys.executable
        tensorboard_cmd = [
            python,
            "-m",
            "tensorboard.main",
            "--logdir",
            f"Data/{model_name}/models",
        ]
        subprocess.Popen(
            tensorboard_cmd,
            stdout=SAFE_STDOUT,  # 타입 무시
            stderr=SAFE_STDOUT,  # 타입 무시
        )
        yield gr.Button("실행 중...")
        if wait_for_tensorboard():
            tensorboard_executed = True
        else:
            logger.error("Tensorboard가 예상 시간 내에 시작되지 않았습니다.")
    webbrowser.open("http://localhost:6006")
    yield gr.Button("Tensorboard 열기")


change_log_md = """
**Ver 2.5 이후 변경사항**

- `raw/` 폴더 내에서 음성을 하위 디렉토리로 분류하면, 자동으로 스타일이 생성됩니다. 자세한 내용은 아래 "사용 방법/데이터 전처리 준비"를 참조하세요.
- 이전에는 파일당 약 14초 이상의 음성 파일은 학습에 사용되지 않았지만, Ver 2.5 이후 "커스텀 배치 샘플러 비활성화"에 체크하면 이 제한 없이 학습이 가능합니다 (기본값은 비활성화). 단:
    - 음성 파일이 길 경우 학습 효율이 떨어질 수 있으며, 해당 동작은 확인되지 않았습니다.
    - 체크할 경우 VRAM 요구량이 크게 증가할 수 있으므로, 학습 실패 또는 VRAM 부족이 발생하면 배치 크기를 줄이거나 체크를 해제하세요.
"""

how_to_md = """
## 사용 방법

- 데이터를 준비하고, 모델 이름을 입력한 후 필요에 따라 설정을 조정한 다음, "자동 전처리 실행" 버튼을 누르세요. 진행 상황은 터미널에 표시됩니다.

- 각 단계별로 실행하려면 "수동 전처리"를 사용하세요 (기본적으로 자동을 권장).

- 전처리가 완료되면 "학습 시작" 버튼을 눌러 학습을 시작할 수 있습니다.

- 중간에 학습을 재개하려면 모델 이름을 입력한 후 "학습 시작"을 누르기만 하면 됩니다.

## JP-Extra 버전에 대해

기본 모델 구조로 [Bert-VITS2 Japanese-Extra](https://github.com/fishaudio/Bert-VITS2/releases/tag/JP-Exta)를 사용할 수 있습니다.
일본어 억양 및 자연스러움이 향상되지만, 영어와 중국어는 사용할 수 없게 됩니다.
"""

prepare_md = """
먼저 음성 데이터와 텍스트 전사 파일을 준비하세요.

이를 다음과 같이 배치합니다:
```
├── Data/
│   ├── {모델 이름}
│   │   ├── esd.list
│   │   ├── raw/
│   │   │   ├── foo.wav
│   │   │   ├── bar.mp3
│   │   │   ├── style1/
│   │   │   │   ├── baz.wav
│   │   │   │   ├── qux.wav
│   │   │   ├── style2/
│   │   │   │   ├── corge.wav
│   │   │   │   ├── grault.wav
...
```

### 배치 방법
- 위와 같이 배치하면, `style1/`과 `style2/` 폴더 내부의 음성 파일로부터 자동으로 기본 스타일 외에 `style1` 및 `style2`라는 스타일이 생성됩니다.
- 별도로 스타일을 생성할 필요가 없거나, 스타일 분류 기능 등을 사용하여 스타일을 생성하려는 경우, `raw/` 폴더 바로 아래에 모든 파일을 배치하세요. `raw/`의 하위 디렉토리가 0개 또는 1개일 경우 기본 스타일만 생성됩니다.
- 음성 파일 형식은 wav 이외에도 mp3 등 다양한 음성 파일을 지원합니다.

### 텍스트 전사 파일 `esd.list`

`Data/{모델 이름}/esd.list` 파일은 각 음성 파일의 정보를 다음 형식으로 기재해야 합니다:

```
path/to/audio.wav|{화자 이름}|{언어 ID, ZH 또는 JP 또는 EN}|{전사 텍스트}
```

- 첫 번째 `path/to/audio.wav`는 `raw/` 폴더에서의 상대 경로입니다. 예를 들어 `raw/foo.wav`인 경우 `foo.wav`, `raw/style1/bar.wav`인 경우 `style1/bar.wav`로 기재합니다.
- 확장자가 wav가 아닌 경우에도 `esd.list`에서는 `wav`로 표기해 주세요. 예를 들어 `raw/bar.mp3`인 경우 `bar.wav`로 작성합니다.

예시:
```
foo.wav|hanako|

JP|안녕하세요, 잘 지내고 계신가요?
bar.wav|taro|JP|네, 잘 들립니다... 무슨 일이 있으신가요?
style1/baz.wav|hanako|JP|오늘 날씨가 좋네요.
style1/qux.wav|taro|JP|네, 그렇네요.
...
english_teacher.wav|Mary|EN|How are you? I'm fine, thank you, and you?
...
```
물론 일본어 화자의 단일 화자 데이터셋도 가능합니다.
"""


def create_train_app():
    with gr.Blocks(theme=GRADIO_THEME).queue() as app:
        gr.Markdown(change_log_md)
        with gr.Accordion("사용 방법", open=False):
            gr.Markdown(how_to_md)
            with gr.Accordion(label="데이터 전처리 준비", open=False):
                gr.Markdown(prepare_md)

        model_name = gr.Textbox(label="모델 이름")
        gr.Markdown("### 자동 전처리")
        with gr.Row(variant="panel"):
            with gr.Column():
                use_jp_extra = gr.Checkbox(
                    label="JP-Extra 버전 사용 (일본어 성능이 향상되지만 영어와 중국어는 사용할 수 없게 됩니다)",
                    value=True,
                )
                batch_size = gr.Slider(
                    label="배치 크기",
                    info="학습 속도가 느릴 경우 작게 설정해 보세요. VRAM 여유가 있으면 크게 설정하세요. JP-Extra 버전의 VRAM 사용량 예상: 1: 6GB, 2: 8GB, 3: 10GB, 4: 12GB",
                    value=2,
                    minimum=1,
                    maximum=64,
                    step=1,
                )
                epochs = gr.Slider(
                    label="에포크 수",
                    info="100이면 충분할 수 있지만 더 많이 돌리면 품질이 향상될 수 있습니다.",
                    value=100,
                    minimum=10,
                    maximum=1000,
                    step=10,
                )
                save_every_steps = gr.Slider(
                    label="몇 스텝마다 결과를 저장할지",
                    info="에포크 수와는 다른 개념입니다.",
                    value=1000,
                    minimum=100,
                    maximum=10000,
                    step=100,
                )
                normalize = gr.Checkbox(
                    label="음성의 음량을 정규화하기 (음량이 일관되지 않은 경우 등)",
                    value=False,
                )
                trim = gr.Checkbox(
                    label="음성의 처음과 끝의 무음을 제거하기",
                    value=False,
                )
                yomi_error = gr.Radio(
                    label="읽기 불가능한 전사 파일 처리 방식",
                    choices=[
                        ("에러가 발생하면 중단", "raise"),
                        ("읽기 불가능한 파일을 무시하고 계속 진행", "skip"),
                        ("읽기 불가능한 파일도 강제로 읽어서 학습에 사용", "use"),
                    ],
                    value="skip",
                )
                with gr.Accordion("세부 설정", open=False):
                    num_processes = gr.Slider(
                        label="프로세스 수",
                        info="전처리 시 동시 처리 프로세스 수, 전처리 중 멈춤이 발생하면 낮춰 보세요.",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                    val_per_lang = gr.Slider(
                        label="검증 데이터 수",
                        info="학습에 사용되지 않으며, TensorBoard에서 원본 음성과 합성 음성을 비교하기 위한 데이터 수입니다.",
                        value=0,
                        minimum=0,
                        maximum=100,
                        step=1,
                    )
                    log_interval = gr.Slider(
                        label="TensorBoard 로그 출력 간격",
                        info="TensorBoard에서 자세히 보고 싶다면 작게 설정하세요.",
                        value=200,
                        minimum=10,
                        maximum=1000,
                        step=10,
                    )
                    gr.Markdown("학습 시 특정 부분을 고정할지 설정")
                    freeze_EN_bert = gr.Checkbox(
                        label="영어 BERT 부분 고정",
                        value=False,
                    )
                    freeze_JP_bert = gr.Checkbox(
                        label="일본어 BERT 부분 고정",
                        value=False,
                    )
                    freeze_ZH_bert = gr.Checkbox(
                        label="중국어 BERT 부분 고정",
                        value=False,
                    )
                    freeze_style = gr.Checkbox(
                        label="스타일 부분 고정",
                        value=False,
                    )
                    freeze_decoder = gr.Checkbox(
                        label="디코더 부분 고정",
                        value=False,
                    )

            with gr.Column():
                preprocess_button = gr.Button(
                    value="자동 전처리 실행", variant="primary"
                )
                info_all = gr.Textbox(label="상태")

        with gr.Accordion(open=False, label="수동 전처리"):
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 1: 설정 파일 생성")
                    use_jp_extra_manual = gr.Checkbox(
                        label="JP-Extra 버전 사용",
                        value=True,
                    )
                    batch_size_manual = gr.Slider(
                        label="배치 크기",
                        value=2,
                        minimum=1,
                        maximum=64,
                        step=1,
                    )
                    epochs_manual = gr.Slider(
                        label="에포크 수",
                        value=100,
                        minimum=1,
                        maximum=1000,
                        step=1,
                    )
                    save_every_steps_manual = gr.Slider(
                        label="몇 스텝마다 결과를 저장할지",
                        value=1000,
                        minimum=100,
                        maximum=10000,
                        step=100,
                    )
                    log_interval_manual = gr.Slider(
                        label="TensorBoard 로그 출력 간격",
                        value=200,
                        minimum=10,
                        maximum=1000,
                        step=10,
                    )
                    freeze_EN_bert_manual = gr.Checkbox(
                        label="영어 BERT 부분 고정",
                        value=False,
                    )
                    freeze_JP_bert_manual = gr.Checkbox(
                        label="일본어 BERT 부분 고정",
                        value=False,
                    )
                    freeze_ZH_bert_manual = gr.Checkbox(
                        label="중국어 BERT 부분 고정",
                        value=False,
                    )
                    freeze_style_manual = gr.Checkbox(
                        label="스타일 부분 고정",
                        value=False,
                    )
                    freeze_decoder_manual = gr.Checkbox(
                        label="디코더 부분 고정",
                        value=False,
                    )
                with gr.Column():
                    generate_config_btn = gr.Button(value="실행", variant="primary")
                    info_init = gr.Textbox(label="상태")

            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 2: 음성 파일 전처리")
                    num_processes_resample = gr.Slider(
                        label="프로세스 수",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                    normalize_resample = gr.Checkbox(
                        label="음성의 음량을 정규화하기",
                        value=False,
                    )
                    trim_resample = gr.Checkbox(
                        label="음성의 처음과 끝의 무음을 제거하기",
                        value=False,
                    )
                with gr.Column():
                    resample_btn = gr.Button(value="실행", variant="primary")
                    info_resample = gr.Textbox(label="상태")

            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 3: 텍스트 파일 전처리")
                    val_per_lang_manual = gr.Slider(
                        label="검증 데이터 수",
                        value=0,
                        minimum=0,
                        maximum=100,
                        step=1,
                    )
                    yomi_error_manual = gr.Radio(
                        label="읽기 불가능한 파일 처리 방식",
                        choices=[
                            ("에러 발생 시 텍스트 전처리 중단", "raise"),
                            ("읽기 불가능한 파일 무시하고 계속", "skip"),
                            ("읽기 불가능한 파일 강제로 읽어서 학습에 사용", "use"),
                        ],
                        value="raise",
                    )
                with gr.Column():
                    preprocess_text_btn = gr.Button(value="실행", variant="primary")
                    info_preprocess_text = gr.Textbox(label="상태")

            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 4: BERT 특성 파일 생성")
                with gr.Column():
                    bert_gen_btn = gr.Button(value="실행", variant="primary")
                    info_bert = gr.Textbox(label="상태")

            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 5: 스타일 특성 파일 생성")
                    num_processes_style = gr.Slider(
                        label="프로세스 수",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                with gr.Column():
                    style_gen_btn = gr.Button(value="실행", variant="primary")
                    info_style = gr.Textbox(label="상태")

        gr.Markdown("## 학습")
        with gr.Row():
            skip_style = gr.Checkbox(
                label="스타일 파일 생성을 생략",
                info="학습 재개 시에는 체크하세요.",
                value=False,
            )
            use_jp_extra_train = gr.Checkbox(
                label="JP-Extra 버전 사용",
                value=True,
            )
            not_use_custom_batch_sampler = gr.Checkbox(
                label="커스텀 배치 샘플러 비활성화",
                info="VRAM 여유가 있는 경우 체크 시 긴 음성 파일도 학습에 사용",
                value=False,
            )
            speedup = gr.Checkbox(
                label="로그 생략하여 학습 속도 향상",
                value=False,
                visible=False,  # 실험적 옵션
            )
            train_btn = gr.Button(value="학습 시작", variant="primary")
            tensorboard_btn = gr.Button(value="Tensorboard 열기")

        gr.Markdown(
            "진행 상황은 터미널에서 확인하세요. 결과는 지정된 스텝마다 저장되며, 학습을 중단했다가 재개할 수 있습니다. 학습을 종료하려면 터미널을 닫으면 됩니다."
        )
        info_train = gr.Textbox(label="상태")

        preprocess_button.click(
            second_elem_of(preprocess_all),
            inputs=[
                model_name,
                batch_size,
                epochs,
                save_every_steps,
                num_processes,
                normalize,
                trim,
                freeze_EN_bert,
                freeze_JP_bert,
                freeze_ZH_bert,
                freeze_style,
                freeze_decoder,
                use_jp_extra,
                val_per_lang,
                log_interval,
                yomi_error,
            ],
            outputs=[info_all],
        )

        # 수동 전처리 단계
        generate_config_btn.click(
            second_elem_of(initialize),
            inputs=[
                model_name,
                batch_size_manual,
                epochs_manual,
                save_every_steps_manual,
                freeze_EN_bert_manual,
                freeze_JP_bert_manual,
                freeze_ZH_bert_manual,
                freeze_style_manual,
                freeze_decoder_manual,
                use_jp_extra_manual,
                log_interval_manual,
            ],
            outputs=[info_init],
        )
        resample_btn.click(
            second_elem_of(resample),
            inputs=[
                model_name,
                normalize_resample,
                trim_resample,
                num_processes_resample,
            ],
            outputs=[info_resample],
        )
        preprocess_text_btn.click(
            second_elem_of(preprocess_text),
            inputs=[
                model_name,
                use_jp_extra_manual,
                val_per_lang_manual,
                yomi_error_manual,
            ],
            outputs=[info_preprocess_text],
        )
        bert_gen_btn.click(
            second_elem_of(bert_gen),
            inputs=[model_name],
            outputs=[info_bert],
        )
        style_gen_btn.click(
            second_elem_of(style_gen),
            inputs=[model_name, num_processes_style],
            outputs=[info_style],
        )

        # 학습
        train_btn.click(
            second_elem_of(train),
            inputs=[
                model_name,
                skip_style,
                use_jp_extra_train,
                speedup,
                not_use_custom_batch_sampler,
            ],
            outputs=[info_train],
        )
        tensorboard_btn.click(
            run_tensorboard, inputs=[model_name], outputs=[tensorboard_btn]
        )

        use_jp_extra.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra],
            outputs=[use_jp_extra_train],
        )
        use_jp_extra_manual.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra_manual],
            outputs=[use_jp_extra_train],
        )

    return app

if __name__ == "__main__":
    app = create_train_app()
    app.launch(inbrowser=True)
