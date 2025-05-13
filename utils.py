# Borrowed from https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/cli/hub.py

import os
import requests
import sys
from pathlib import Path
import tarfile
import zipfile
from urllib.request import urlretrieve
from typing import List

import srt
import tqdm

# Model URL
MODEL_TYPE = "voxblink2_samresnet100_ft.zip"

def download(url: str, dest: str, only_child=True):
    """download from url to dest
    Borrowed from https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/cli/hub.py
    """
    assert os.path.exists(dest)
    print("Downloading {} to {}".format(url, dest))

    def progress_hook(t):
        last_b = [0]

        def update_to(b=1, bsize=1, tsize=None):
            if tsize not in (None, -1):
                t.total = tsize
            displayed = t.update((b - last_b[0]) * bsize)
            last_b[0] = b
            return displayed

        return update_to

    # *.tar.gz
    name = url.split("?")[0].split("/")[-1]
    file_path = os.path.join(dest, name)
    with tqdm.tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=(name)
    ) as t:
        urlretrieve(
            url, filename=file_path, reporthook=progress_hook(t), data=None
        )
        t.total = t.n

    if name.endswith((".tar.gz", ".tar")):
        with tarfile.open(file_path) as f:
            if not only_child:
                f.extractall(dest)
            else:
                for tarinfo in f:
                    if "/" not in tarinfo.name:
                        continue
                    name = os.path.basename(tarinfo.name)
                    fileobj = f.extractfile(tarinfo)
                    with open(os.path.join(dest, name), "wb") as writer:
                        writer.write(fileobj.read())

    elif name.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            if not only_child:
                zip_ref.extractall(dest)
            else:
                for member in zip_ref.namelist():
                    member_path = os.path.relpath(
                        member, start=os.path.commonpath(zip_ref.namelist())
                    )
                    if "/" not in member_path:
                        continue
                    name = os.path.basename(member_path)
                    with zip_ref.open(member_path) as source, open(
                        os.path.join(dest, name), "wb"
                    ) as target:
                        target.write(source.read())


def get_wespeaker_model() -> str:
    """
    Download the WeSpeaker model.
    Modified from https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/cli/hub.py
    """
    model_dir = os.path.join(Path.home(), ".wespeaker", "simamresnet100_ft")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if set(["avg_model.pt", "config.yaml"]).issubset(
        set(os.listdir(model_dir))
    ):
        return model_dir
    else:
        response = requests.get(
            "https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree"  # noqa
        )
        model_info = next(
            data
            for data in response.json()["Data"]
            if data["Key"] == MODEL_TYPE
        )
        model_url = model_info["Url"]
        download(model_url, model_dir)
        return model_dir


def write_subtitle(segments: List, out_file: str) -> None:
    """Write segments to an SRT subtitle file."""
    subs = []
    for idx, seg in enumerate(segments, 1):
        subs.append(
            srt.Subtitle(
                index=idx,
                start=srt.timedelta(seconds=seg[0]),
                end=srt.timedelta(seconds=seg[1]),
                content=f"{seg[3]} : {seg[2]}"
            )
        )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
