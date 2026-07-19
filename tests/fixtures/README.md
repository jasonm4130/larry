# Test fixtures

## `speech_sp0307.wav`

A ~3.4s clip of a single speaker (VOiCES id `sp0307`), 16 kHz mono PCM16.

- **Source:** the VOiCES corpus sample `Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042`,
  as mirrored in the torchaudio tutorial assets
  (`https://download.pytorch.org/torchaudio/tutorial-assets/`).
- **License:** VOiCES is released under **CC BY 4.0**.

Used by the CAM++ embedder tests as the golden-gate input and (split into halves,
plus seeded noise) as the discrimination smoke test.

## `speech_sp0307_campplus_golden.npy`

The 512-d, L2-normalized CAM++ embedding of `speech_sp0307.wav`, produced from
**WeSpeaker's reference featurization** — `torchaudio.compliance.kaldi.fbank`
(80-dim, Hamming window, dither 0, CMN) exactly as `wespeaker/bin/infer_onnx.py`
does — through `voxceleb_CAM++_LM.onnx`. It is deliberately NOT generated from
larry's own `kaldi-native-fbank` code, so the golden test is a genuine check that
our featurization reproduces the reference, not a tautology.

### Regenerating

Requires the CAM++ model cached at `data/models/` (`uv run larry fetch-models`)
and torchaudio:

```
uv run --with torchaudio python scripts/gen_campplus_golden.py
```
