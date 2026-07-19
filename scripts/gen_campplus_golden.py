"""Regenerate the committed CAM++ golden embedding fixture.

Produces ``tests/fixtures/speech_sp0307_campplus_golden.npy`` from
``tests/fixtures/speech_sp0307.wav`` using WeSpeaker's *reference* featurization
(``torchaudio.compliance.kaldi.fbank``, exactly as ``wespeaker/bin/infer_onnx.py``)
through the CAM++ onnx model — so the golden test genuinely checks that larry's
``kaldi-native-fbank`` pipeline reproduces the reference.

Dev-only. Requires the model cached (``uv run larry fetch-models``) and torchaudio:

    uv run --with torchaudio python scripts/gen_campplus_golden.py
"""

import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
MODEL = ROOT / "data" / "models" / "voxceleb_CAM++_LM.onnx"


def main() -> None:
    with wave.open(str(FIX / "speech_sp0307.wav"), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    waveform = torch.from_numpy(wav).unsqueeze(0) * (1 << 15)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
        window_type="hamming",
        use_energy=False,
    )
    feats = (mat - torch.mean(mat, dim=0)).numpy()[np.newaxis, :, :].astype(np.float32)
    sess = ort.InferenceSession(str(MODEL))
    emb = np.asarray(sess.run(["embs"], {"feats": feats})[0])[0].astype(np.float32)
    emb = emb / np.linalg.norm(emb)

    out = FIX / "speech_sp0307_campplus_golden.npy"
    np.save(out, emb)
    print(f"wrote {out} (shape {emb.shape}, norm {np.linalg.norm(emb):.6f})")


if __name__ == "__main__":
    main()
