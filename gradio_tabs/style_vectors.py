"""
TODO:
importが重いので、WebUI全般が重くなっている。どうにかしたい。
"""

import json
import shutil
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from umap import UMAP

from config import get_path_config
from default_style import save_styles_by_dirs
from style_bert_vits2.constants import DEFAULT_STYLE, GRADIO_THEME
from style_bert_vits2.logging import logger


path_config = get_path_config()
dataset_root = path_config.dataset_root
assets_root = path_config.assets_root

MAX_CLUSTER_NUM = 10
MAX_AUDIO_NUM = 10

tsne = TSNE(n_components=2, random_state=42, metric="cosine")
umap = UMAP(n_components=2, random_state=42, metric="cosine", n_jobs=1, min_dist=0.0)

wav_files: list[Path] = []
x = np.array([])
x_reduced = None
y_pred = np.array([])
mean = np.array([])
centroids = []


def load(model_name: str, reduction_method: str):
    global wav_files, x, x_reduced, mean
    wavs_dir = dataset_root / model_name / "wavs"
    style_vector_files = [f for f in wavs_dir.rglob("*.npy") if f.is_file()]
    # foo.wav.npy -> foo.wav
    wav_files = [f.with_suffix("") for f in style_vector_files]
    logger.info(f"Found {len(style_vector_files)} style vectors in {wavs_dir}")
    style_vectors = [np.load(f) for f in style_vector_files]
    x = np.array(style_vectors)
    mean = np.mean(x, axis=0)
    if reduction_method == "t-SNE":
        x_reduced = tsne.fit_transform(x)
    elif reduction_method == "UMAP":
        x_reduced = umap.fit_transform(x)
    else:
        raise ValueError("Invalid reduction method")
    x_reduced = np.asarray(x_reduced)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1])
    return plt


def do_clustering(n_clusters=4, method="KMeans"):
    global centroids, x_reduced, y_pred
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x)
    elif method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x)
    elif method == "KMeans after reduction":
        assert x_reduced is not None
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x_reduced)
    elif method == "Agglomerative after reduction":
        assert x_reduced is not None
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x_reduced)
    else:
        raise ValueError("Invalid method")

    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))

    return y_pred, centroids


def do_dbscan(eps=2.5, min_samples=15):
    global centroids, x_reduced, y_pred
    model = DBSCAN(eps=eps, min_samples=min_samples)
    assert x_reduced is not None
    y_pred = model.fit_predict(x_reduced)
    n_clusters = max(y_pred) + 1
    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))
    return y_pred, centroids


def representative_wav_files(cluster_id, num_files=1):
    # y_pred에서 cluster_id에 해당하는 메도이드를 찾음
    cluster_indices = np.where(y_pred == cluster_id)[0]
    cluster_vectors = x[cluster_indices]
    # 클러스터 내 모든 벡터 간 거리를 계산
    distances = pdist(cluster_vectors)
    distance_matrix = squareform(distances)

    # 각 벡터와 다른 모든 벡터와의 평균 거리를 계산
    mean_distances = distance_matrix.mean(axis=1)

    # 평균 거리가 가장 작은 순서대로 num_files개의 인덱스를 가져옴
    closest_indices = np.argsort(mean_distances)[:num_files]

    return cluster_indices[closest_indices]


def do_dbscan_gradio(eps=2.5, min_samples=15):
    global x_reduced, centroids

    y_pred, centroids = do_dbscan(eps, min_samples)

    assert x_reduced is not None

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(max(y_pred) + 1):
        plt.scatter(
            x_reduced[y_pred == i, 0],
            x_reduced[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i + 1}",
        )
    # Noise cluster (-1) is black
    plt.scatter(
        x_reduced[y_pred == -1, 0],
        x_reduced[y_pred == -1, 1],
        color="black",
        label="Noise",
    )
    plt.legend()

    n_clusters = int(max(y_pred) + 1)

    if n_clusters > MAX_CLUSTER_NUM:
        # raise ValueError(f"The number of clusters is too large: {n_clusters}")
        return [
            plt,
            gr.Slider(maximum=MAX_CLUSTER_NUM),
            f"クラスタ数が多すぎます、パラメータを変えてみてください。: {n_clusters}",
        ] + [gr.Audio(visible=False)] * MAX_AUDIO_NUM

    elif n_clusters == 0:
        return [
            plt,
            gr.Slider(maximum=MAX_CLUSTER_NUM),
            "クラスタが数が0です。パラメータを変えてみてください。",
        ] + [gr.Audio(visible=False)] * MAX_AUDIO_NUM

    return [plt, gr.Slider(maximum=n_clusters, value=1), n_clusters] + [
        gr.Audio(visible=False)
    ] * MAX_AUDIO_NUM


def representative_wav_files_gradio(cluster_id, num_files=1):
    cluster_id = cluster_id - 1  # UI는 1부터 시작하므로 0부터 시작한다.
    closest_indices = representative_wav_files(cluster_id, num_files)
    actual_num_files = len(closest_indices)  # 파일 수가 적을 때
    return [
        gr.Audio(wav_files[i], visible=True, label=str(wav_files[i]))
        for i in closest_indices
    ] + [gr.update(visible=False)] * (MAX_AUDIO_NUM - actual_num_files)


def do_clustering_gradio(n_clusters=4, method="KMeans"):
    global x_reduced, centroids
    y_pred, centroids = do_clustering(n_clusters, method)

    assert x_reduced is not None
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(n_clusters):
        plt.scatter(
            x_reduced[y_pred == i, 0],
            x_reduced[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i + 1}",
        )
    plt.legend()

    return [plt, gr.Slider(maximum=n_clusters, value=1)] + [
        gr.Audio(visible=False)
    ] * MAX_AUDIO_NUM


def save_style_vectors_from_clustering(model_name: str, style_names_str: str):
    """center와 centroids를 저장합니다"""
    result_dir = assets_root / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    style_vectors = np.stack([mean] + centroids)
    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"{style_vector_path}를 {style_vector_path}.bak에 백업합니다.")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    np.save(style_vector_path, style_vectors)
    logger.success(f"{style_vector_path}에 스타일 벡터를 저장했습니다.")

    # config.json 업데이트
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path}가 존재하지 않습니다."
    style_names = [name.strip() for name in style_names_str.split(",")]
    style_name_list = [DEFAULT_STYLE] + style_names
    if len(style_name_list) != len(centroids) + 1:
        return f"스타일의 수가 맞지 않습니다. ','로 정확히 {len(centroids)}개로 구분되어 있는지 확인하세요: {style_names_str}"
    if len(set(style_names)) != len(style_names):
        return "스타일 이름이 중복됩니다."

    logger.info(f"{config_path}를 {config_path}.bak에 백업합니다.")
    shutil.copy(config_path, f"{config_path}.bak")
    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.success(f"{config_path}를 업데이트했습니다.")
    return f"성공!\n{style_vector_path}에 저장하고 {config_path}를 업데이트했습니다."


def save_style_vectors_from_files(
    model_name: str, audio_files_str: str, style_names_str: str
):
    """오디오 파일에서 스타일 벡터를 생성하여 저장합니다"""
    global mean
    if len(x) == 0:
        return "Error: 스타일 벡터를 로드해 주세요."
    mean = np.mean(x, axis=0)

    result_dir = assets_root / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    audio_files = [name.strip() for name in audio_files_str.split(",")]
    style_names = [name.strip() for name in style_names_str.split(",")]
    if len(audio_files) != len(style_names):
        return f"오디오 파일과 스타일 이름의 수가 맞지 않습니다. ','로 정확히 {len(style_names)}개로 구분되어 있는지 확인하세요: {audio_files_str}와 {style_names_str}"
    style_name_list = [DEFAULT_STYLE] + style_names
    if len(set(style_names)) != len(style_names):
        return "스타일 이름이 중복됩니다."
    style_vectors = [mean]

    wavs_dir = dataset_root / model_name / "wavs"
    for audio_file in audio_files:
        path = wavs_dir / audio_file
        if not path.exists():
            return f"{path}가 존재하지 않습니다."
        style_vectors.append(np.load(f"{path}.npy"))
    style_vectors = np.stack(style_vectors)
    assert len(style_name_list) == len(style_vectors)
    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"{style_vector_path}를 {style_vector_path}.bak에 백업합니다.")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    np.save(style_vector_path, style_vectors)

    # config.json 업데이트
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path}가 존재하지 않습니다."
    logger.info(f"{config_path}를 {config_path}.bak에 백업합니다.")
    shutil.copy(config_path, f"{config_path}.bak")

    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    return f"성공!\n{style_vector_path}에 저장하고 {config_path}를 업데이트했습니다."


def save_style_vectors_by_dirs(model_name: str, audio_dir_str: str):
    if model_name == "":
        return "모델 이름을 입력해 주세요."
    if audio_dir_str == "":
        return "오디오 파일이 있는 디렉터리를 입력해 주세요."

    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import cpu_count

    from tqdm import tqdm

    from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
    from style_gen import save_style_vector

    # 먼저 각 오디오 파일에 대해 스타일 벡터를 생성합니다.

    audio_dir = Path(audio_dir_str)
    audio_suffixes = [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"]
    audio_files = [f for f in audio_dir.rglob("*") if f.suffix in audio_suffixes]

    def process(file: Path):
        # f: `test.wav` -> `test.wav.npy`가 있는지 확인
        if (file.with_name(file.name + ".npy")).exists():
            return file, None
        try:
            save_style_vector(str(file))
        except Exception as e:
            return file, e
        return file, None

    with ThreadPoolExecutor(max_workers=cpu_count() // 2) as executor:
        _ = list(
            tqdm(
                executor.map(
                    process,
                    audio_files,
                ),
                total=len(audio_files),
                file=SAFE_STDOUT,
                desc="스타일 벡터 생성 중",
            )
        )

    result_dir = assets_root / model_name
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path}가 존재하지 않습니다."
    logger.info(f"{config_path}를 {config_path}.bak에 백업합니다.")
    shutil.copy(config_path, f"{config_path}.bak")

    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"{style_vector_path}를 {style_vector_path}.bak에 백업합니다.")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    save_styles_by_dirs(
        wav_dir=audio_dir,
        output_dir=result_dir,
        config_path=config_path,
        config_output_path=config_path,
    )
    return f"성공!\n{result_dir}에 스타일 벡터를 저장했습니다."


how_to_md = f"""
Style-Bert-VITS2로 세부적인 스타일을 지정하여 음성을 합성하려면, 모델마다 스타일 벡터 파일 `style_vectors.npy`를 생성해야 합니다.

단, 학습 과정에서는 자동으로 평균 스타일 「{DEFAULT_STYLE}」과, (**Ver 2.5.0 이상에서는**) 음성이 하위 폴더로 나뉘어 있는 경우 해당 하위 폴더마다 스타일이 저장됩니다.

## 방법

- 방법 0: 음성을 스타일별로 하위 폴더에 나누고, 각 폴더별로 스타일 벡터를 생성
- 방법 1: 음성 파일을 자동으로 스타일별로 나누고, 각 스타일의 평균을 계산하여 저장
- 방법 2: 대표 스타일의 음성 파일을 수동으로 선택하고, 해당 음성의 스타일 벡터를 저장
- 방법 3: 직접 더 노력하여 스타일 벡터를 만듭니다 (예: JVNV 코퍼스 등 원래 스타일 레이블이 있는 경우 권장)
"""

method0 = """
음성을 스타일별로 하위 폴더를 만들어 그 안에 음성 파일을 넣어 주세요.

**주의사항**

- Ver 2.5.0 이상에서는 `inputs/` 폴더나 `raw/` 폴더의 하위 디렉토리에 음성 파일을 넣으면 스타일 벡터가 자동으로 생성되므로, 이 과정이 필요하지 않습니다.
- 이전 버전에서 학습한 모델에 새로운 스타일 벡터를 추가하거나 학습에 사용한 것과 다른 음성으로 스타일 벡터를 생성하려는 경우에만 사용하세요.
- 학습과의 일관성을 위해, **현재 학습 중이거나 앞으로 학습할 예정이 있는 경우** 음성 파일은 `Data/{모델명}/wavs` 폴더가 아닌 **새로운 별도의 디렉토리에 저장하세요**.

예시:

```bash
audio_dir
├── style1
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── style2
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```
"""

method1 = f"""
학습 시 생성한 스타일 벡터를 불러와서, 시각화를 통해 스타일을 구분해 나갑니다.

절차:
1. 그림을 살펴봅니다.
2. 스타일 수를 결정합니다 (평균 스타일 제외).
3. 스타일 구분을 진행하고 결과를 확인합니다.
4. 스타일 이름을 결정하여 저장합니다.

상세 설명: 스타일 벡터(256차원)들을 적절한 알고리즘으로 클러스터링하여, 각 클러스터 중심 벡터와 전체 평균 벡터를 저장합니다.

평균 스타일({DEFAULT_STYLE})은 자동으로 저장됩니다.
"""

dbscan_md = """
DBSCAN이라는 방법으로 스타일을 구분합니다.
이 방법은 방법 1보다 뚜렷한 특징이 있는 스타일만을 추출하여 더 좋은 스타일 벡터를 만들 수 있을 수도 있습니다.
단, 스타일 수는 사전에 지정할 수 없습니다.

파라미터:
- eps: 이 값보다 가까운 점들을 같은 스타일로 분류합니다. 작을수록 스타일 수가 증가하고, 클수록 스타일 수가 줄어드는 경향이 있습니다.
- min_samples: 특정 점을 스타일의 중심으로 간주하기 위해 필요한 주변 점의 수입니다. 작을수록 스타일 수가 증가하고, 클수록 스타일 수가 줄어드는 경향이 있습니다.

UMAP의 경우 eps 값은 0.3 정도, t-SNE의 경우 2.5 정도가 적절할 수 있습니다. min_samples 값은 데이터 수에 따라 달라지므로 여러 값을 시도해 보세요.

자세한 내용:
https://ko.wikipedia.org/wiki/DBSCAN
"""


def create_style_vectors_app():
    with gr.Blocks(theme=GRADIO_THEME) as app:
        with gr.Accordion("사용 방법", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(placeholder="your_model_name", label="모델 이름")
        with gr.Tab("방법 0: 하위 폴더별로 스타일 벡터 생성"):
            gr.Markdown(method0)
            audio_dir = gr.Textbox(
                placeholder="path/to/audio_dir",
                label="음성이 들어 있는 폴더",
                info="음성 파일을 스타일별로 하위 폴더에 나눠서 저장해 주세요.",
            )
            method0_btn = gr.Button("스타일 벡터 생성", variant="primary")
            method0_info = gr.Textbox(label="결과")
            method0_btn.click(
                save_style_vectors_by_dirs,
                inputs=[model_name, audio_dir],
                outputs=[method0_info],
            )
        with gr.Tab("기타 방법"):
            with gr.Row():
                reduction_method = gr.Radio(
                    choices=["UMAP", "t-SNE"],
                    label="차원 축소 방법",
                    info="버전 1.3 이전에는 t-SNE를 사용했지만, UMAP이 더 나을 수 있습니다.",
                    value="UMAP",
                )
                load_button = gr.Button("스타일 벡터 로드", variant="primary")
            output = gr.Plot(label="음성 스타일 시각화")
            load_button.click(
                load, inputs=[model_name, reduction_method], outputs=[output]
            )
            with gr.Tab("방법 1: 스타일 자동 분류"):
                with gr.Tab("스타일 분류 1"):
                    n_clusters = gr.Slider(
                        minimum=2,
                        maximum=10,
                        step=1,
                        value=4,
                        label="생성할 스타일 수 (평균 스타일 제외)",
                        info="위의 그림을 참고하며 스타일 수를 조정해 보세요.",
                    )
                    c_method = gr.Radio(
                        choices=[
                            "Agglomerative after reduction",
                            "KMeans after reduction",
                            "Agglomerative",
                            "KMeans",
                        ],
                        label="알고리즘",
                        info="분류할 (클러스터링) 알고리즘을 선택하세요. 다양한 방법을 시도해 보세요.",
                        value="Agglomerative after reduction",
                    )
                    c_button = gr.Button("스타일 분류 실행")
                with gr.Tab("스타일 분류 2: DBSCAN"):
                    gr.Markdown(dbscan_md)
                    eps = gr.Slider(
                        minimum=0.1,
                        maximum=10,
                        step=0.01,
                        value=0.3,
                        label="eps",
                    )
                    min_samples = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=15,
                        label="최소 샘플 수",
                    )
                    with gr.Row():
                        dbscan_button = gr.Button("스타일 분류 실행")
                        num_styles_result = gr.Textbox(label="스타일 수")
                gr.Markdown("스타일 분류 결과")
                gr.Markdown(
                    "주의: 원래 256차원 데이터를 2차원으로 축소한 것이므로, 정확한 벡터 간 위치 관계는 아닙니다."
                )
                with gr.Row():
                    gr_plot = gr.Plot()
                    with gr.Column():
                        with gr.Row():
                            cluster_index = gr.Slider(
                                minimum=1,
                                maximum=MAX_CLUSTER_NUM,
                                step=1,
                                value=1,
                                label="스타일 번호",
                                info="선택한 스타일의 대표 음성을 표시합니다.",
                            )
                            num_files = gr.Slider(
                                minimum=1,
                                maximum=MAX_AUDIO_NUM,
                                step=1,
                                value=5,
                                label="대표 음성 수 표시",
                            )
                            get_audios_button = gr.Button("대표 음성 가져오기")
                        with gr.Row():
                            audio_list = []
                            for i in range(MAX_AUDIO_NUM):
                                audio_list.append(
                                    gr.Audio(visible=False, show_label=True)
                                )
                    c_button.click(
                        do_clustering_gradio,
                        inputs=[n_clusters, c_method],
                        outputs=[gr_plot, cluster_index] + audio_list,
                    )
                    dbscan_button.click(
                        do_dbscan_gradio,
                        inputs=[eps, min_samples],
                        outputs=[gr_plot, cluster_index, num_styles_result]
                        + audio_list,
                    )
                    get_audios_button.click(
                        representative_wav_files_gradio,
                        inputs=[cluster_index, num_files],
                        outputs=audio_list,
                    )
                gr.Markdown("결과가 만족스럽다면, 이를 저장하세요.")
                style_names = gr.Textbox(
                    "Angry, Sad, Happy",
                    label="스타일 이름",
                    info=f"스타일 이름을 ','로 구분하여 입력하세요 (한국어 가능). 예: `Angry, Sad, Happy` 또는 `화남, 슬픔, 기쁨` 등. 평균 음성은 {DEFAULT_STYLE}로 자동 저장됩니다.",
                )
                with gr.Row():
                    save_button1 = gr.Button(
                        "스타일 벡터 저장", variant="primary"
                    )
                    info2 = gr.Textbox(label="저장 결과")

                save_button1.click(
                    save_style_vectors_from_clustering,
                    inputs=[model_name, style_names],
                    outputs=[info2],
                )
            with gr.Tab("방법 2: 수동으로 스타일 선택"):
                gr.Markdown(
                    "아래 텍스트 칸에 각 스타일의 대표 음성 파일 이름을 ','로 구분하여, 그 옆에 해당 스타일 이름을 ','로 구분하여 입력하세요."
                )
                gr.Markdown("예: `angry.wav, sad.wav, happy.wav`와 `Angry, Sad, Happy`")
                gr.Markdown(
                    f"주의: {DEFAULT_STYLE} 스타일은 자동으로 저장되므로, 수동으로 {DEFAULT_STYLE}라는 이름의 스타일을 지정하지 마세요."
                )
                with gr.Row():
                    audio_files_text = gr.Textbox(
                        label="음성 파일 이름",
                        placeholder="angry.wav, sad.wav, happy.wav",
                    )
                    style_names_text = gr.Textbox(
                        label="스타일 이름", placeholder="Angry, Sad, Happy"
                    )
                with gr.Row():
                    save_button2 = gr.Button(
                        "스타일 벡터 저장", variant="primary"
                    )
                    info2 = gr.Textbox(label="저장 결과")
                    save_button2.click(
                        save_style_vectors_from_files,
                        inputs=[model_name, audio_files_text, style_names_text],
                        outputs=[info2],
                    )

    return app


if __name__ == "__main__":
    app = create_style_vectors_app()
    app.launch(inbrowser=True)
