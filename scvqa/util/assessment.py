import torch
from torch import Tensor
from torchvision import transforms

import numpy as np

from PIL import Image
import skvideo
import skvideo.io

import pandas as pd

from scipy.optimize import curve_fit

from .model import ResNet50, Transformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_BATCH_SIZE = 32
MAX_FRAME_SIZE = 300
FEATURE_SIZE = 4096
VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280

SCVD_PATH = "scvqa/model/Transformer/SCVD/ResNet50/20240327_164442"
CSCVQ_PATH = "scvqa/model/Transformer/CSCVQ/ResNet50/20240321_172018"

SCVD_MOS_MIN, SCVD_MOS_MAX = 20.1179, 74.0773
CSCVQ_MOS_MIN, CSCVQ_MOS_MAX = 20.53108285, 72.75596394

cnn_model = ResNet50().to(device=DEVICE)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def objective(x, b1, b2, b3, b4, b5):
    return (b1 * ((1 / 2) - (1 / (1 + np.exp(b2 * (x - b3)))))) + (x * b4) + (b5)


# prediction_df = pd.read_csv(f"{SCVD_PATH}/prediction.csv")
# popt, _ = curve_fit(
#     objective, prediction_df["y_norm"], prediction_df["y_pred_norm"], maxfev=5000
# )
# b1, b2, b3, b4, b5 = popt
# 0.14336393353069424, -15.673693191562277, 0.8467178950373697, 0.8180427321935722, -0.0026796865306528738
def nonlinear_map_SCVD(x):
    b1, b2, b3, b4, b5 = (
        0.14336393353069424,
        -15.673693191562277,
        0.8467178950373697,
        0.8180427321935722,
        -0.0026796865306528738,
    )
    return objective(x, b1, b2, b3, b4, b5)


# prediction_df = pd.read_csv(f"{CSCVQ_PATH}/prediction.csv")
# popt, _ = curve_fit(
#     objective, prediction_df["y_norm"], prediction_df["y_pred_norm"], maxfev=9000
# )
# b1, b2, b3, b4, b5 = popt
# 303.6521282057421, 0.5023242558125017, 0.5164839843431867, -36.94749181046035, 19.58965406940685
def nonlinear_map_CSCVQ(x):
    b1, b2, b3, b4, b5 = (
        303.6521282057421,
        0.5023242558125017,
        0.5164839843431867,
        -36.94749181046035,
        19.58965406940685,
    )
    return objective(x, b1, b2, b3, b4, b5)


def up_scale(score_predicted, mos_max, mos_min):
    return score_predicted * (mos_max - mos_min) + mos_min


def load_video(video_path) -> Tensor:

    video = skvideo.io.vread(
        video_path, VIDEO_HEIGHT, VIDEO_WIDTH, inputdict={"-pix_fmt": "yuvj420p"}
    )

    # video = skvideo.io.vread(video_path, VIDEO_HEIGHT, VIDEO_WIDTH)

    frame_size, height, width, channel = video.shape

    if frame_size > MAX_FRAME_SIZE:
        frame_size = MAX_FRAME_SIZE

    transformed_video = torch.zeros([frame_size, channel, height, width])

    for i in range(frame_size):
        frame = video[i]
        frame = Image.fromarray(frame)
        frame = transform(frame)
        transformed_video[i] = frame
    return transformed_video


def feature_extraction(video_path):
    video = load_video(video_path)
    cnn_feature = None
    cnn_model.eval()
    with torch.inference_mode():

        current = 0
        end_frame = len(video)

        video = video.to(device=DEVICE)

        feature_mean = torch.Tensor().to(device=DEVICE)
        feature_std = torch.Tensor().to(device=DEVICE)
        cnn_feature = torch.Tensor().to(device=DEVICE)

        while current < end_frame:
            head = current
            tail = (
                (head + FRAME_BATCH_SIZE)
                if (head + FRAME_BATCH_SIZE < end_frame)
                else end_frame
            )
            print(f"Extracting video: frames[{head}, {tail-1}]")
            batch_frames = video[head:tail]

            mean, std = cnn_model(batch_frames)
            feature_mean = torch.cat((feature_mean, mean), 0)
            feature_std = torch.cat((feature_std, std), 0)

            current += FRAME_BATCH_SIZE

        cnn_feature = (
            torch.cat((feature_mean, feature_std), 1).squeeze().numpy(force=True)
            if torch.cuda.is_available()
            else torch.cat((feature_mean, feature_std), 1).squeeze().numpy()
        )
    cnn_feature = torch.from_numpy(cnn_feature)
    return cnn_feature


def test(frames):

    qualityScoreSCVD, qualityScoreCSCVQ = None, None

    model_tfm_SCVD = Transformer(
        device=DEVICE,
        feature_size=FEATURE_SIZE,
    ).to(device=DEVICE)

    model_tfm_CSCVQ = Transformer(
        device=DEVICE,
        feature_size=FEATURE_SIZE,
    ).to(device=DEVICE)

    model_tfm_SCVD.load_state_dict(
        torch.load(
            f=f"{SCVD_PATH}/model.pt",
            map_location=torch.device(DEVICE),
        )
    )

    model_tfm_CSCVQ.load_state_dict(
        torch.load(
            f=f"{CSCVQ_PATH}/model.pt",
            map_location=torch.device(DEVICE),
        )
    )

    model_tfm_SCVD.eval()
    with torch.inference_mode():

        input = frames.to(device=DEVICE)
        input = input.unsqueeze(dim=0)

        qualityScoreSCVD = model_tfm_SCVD(input).item()

    model_tfm_CSCVQ.eval()
    with torch.inference_mode():

        input = frames.to(device=DEVICE)
        input = input.unsqueeze(dim=0)

        qualityScoreCSCVQ = model_tfm_CSCVQ(input).item()

    qualityScoreSCVD = up_scale(
        nonlinear_map_SCVD(qualityScoreSCVD), SCVD_MOS_MIN, SCVD_MOS_MAX
    )
    qualityScoreCSCVQ = up_scale(
        nonlinear_map_CSCVQ(qualityScoreCSCVQ), CSCVQ_MOS_MIN, CSCVQ_MOS_MAX
    )
    return {
        "qualityScoreSCVD": round(qualityScoreSCVD, 2),
        "qualityScoreCSCVQ": round(qualityScoreCSCVQ, 2),
    }


def VQA(video_path):
    frames = feature_extraction(video_path)
    return test(frames)


def FQA(feature_path):
    feature = np.load(feature_path)
    frames = torch.from_numpy(feature)
    return test(frames)
