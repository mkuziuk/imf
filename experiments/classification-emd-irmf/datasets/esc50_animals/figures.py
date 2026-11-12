"""Diagnostic figures for the prepared esc50_animals frames (same plots as birdclef2023).

    .venv/bin/python -I experiments/classification-emd-irmf/datasets/esc50_animals/figures.py
"""

import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np

NAME = "esc50_animals"
ARCHIVE = "ESC-50-master.zip"


def _figures_module():
    path = Path(__file__).resolve().parents[1] / "birdclef2023" / "figures.py"
    spec = importlib.util.spec_from_file_location("birdclef2023_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def esc50_context(raw_dir, fs, length):
    """Loader giving a clip at fs in stored start_sample coordinates (leading zeros trimmed)."""
    import librosa
    import soundfile as sf

    archive = zipfile.ZipFile(raw_dir / ARCHIVE)

    def load(row):
        payload = archive.read("ESC-50-master/audio/" + row["source_file"])
        audio, sr = sf.read(io.BytesIO(payload), dtype="float64", always_2d=True)
        audio = audio.mean(axis=1)
        nonzero = np.flatnonzero(audio != 0.0)
        audio = audio[nonzero[0]: nonzero[-1] + 1]
        audio = librosa.resample(audio, orig_sr=sr, target_sr=fs, res_type="soxr_hq")
        return audio - audio.mean()

    return load


if __name__ == "__main__":
    _figures_module().render(NAME, esc50_context, "y_animal", "y_animal")
