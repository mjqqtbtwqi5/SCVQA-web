from django.shortcuts import render
from django.http import JsonResponse
from .util.assessment import VQA, FQA
from pathlib import Path
from website.settings import BASE_DIR
import os
import numpy as np
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def videoAssessment(request):
    if request.method == "POST" and request.FILES["videoFile"]:
        video_file = request.FILES["videoFile"]
        video_path = video_file.temporary_file_path()
        qualityScore = VQA(video_path)
        return JsonResponse(qualityScore)
    return render(request, "home.html")


@csrf_exempt
def featureAssessment(request):
    if request.method == "POST":
        videoName = request.POST.get("videoName", "")
        feature_path = (
            BASE_DIR / "static" / "feature" / "CSCVQ" / videoName / "feature.npy"
        )
        qualityScore = FQA(feature_path)
        return JsonResponse(qualityScore)

    return render(request, "home.html")


def home(request):
    scan_dir = BASE_DIR / "static" / "feature" / "CSCVQ"

    context = {}
    video_list = list()

    for video_name in os.listdir(scan_dir):
        snapshot_file = os.path.join(scan_dir, video_name, "snapshot.jpg")
        if os.path.isfile(snapshot_file):
            mos_file = os.path.join(scan_dir, video_name, "mos.npy")
            mos = np.load(mos_file)
            mos = np.float32(mos.item())

            video_list.append(
                {
                    "video_name": video_name,
                    "snapshot_file": os.path.join(
                        "feature", "CSCVQ", video_name, "snapshot.jpg"
                    ),
                    "mos": round(mos, 2),
                }
            )
    context["video_list"] = video_list
    return render(request, "home.html", context)
