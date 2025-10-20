# AR–EN Transformer (PyTorch)
Implementation of a Transformer-based neural machine translation (NMT) model inspired by **“Attention Is All You Need” (Vaswani et al., 2017)**, trained on **custom parallel data** for **Arabic ↔ English**. The project focuses on **low‑resource Arabic** scenarios and demonstrates practical techniques for data preparation, training, and evaluation in PyTorch.

> Notebook: `Text-Translation-Transformer.ipynb`

---

## ✨ What this repo contains
- A clean, from‑scratch **Transformer** in **PyTorch** for sequence‑to‑sequence translation.
- **Custom-data** training pipeline for **Arabic ↔ English (AR–EN)**.
- Low‑resource strategies (shared BPE, label smoothing, subword regularization, early stopping).
- Reproducible evaluation (SacreBLEU) and simple inference utilities.

---

## 🧠 Background
This implementation follows the core ideas from the Transformer architecture introduced in *Attention Is All You Need*. No RNNs or CNNs — just multi‑head self‑attention, positional encodings, and feed‑forward layers in an encoder–decoder arrangement.

Key components:
- Token + positional embeddings
- Multi‑Head Self‑Attention (encoder) and Cross‑Attention (decoder)
- Position‑wise feed‑forward networks
- Residual connections & LayerNorm
- Teacher forcing during training and autoregressive decoding at inference

---

## 📂 Project structure
```
.
├── Text-Translation-Transformer.ipynb   # End-to-end training & evaluation
├── data/
│   ├── train.ar  train.en               # Your parallel training files
│   ├── valid.ar  valid.en               # Validation files
│   └── test.ar   test.en                # Test files
├── artifacts/
│   ├── spm.model  spm.vocab             # SentencePiece model (shared)
│   ├── vocab.json                       # Token → id mapping (optional)
│   └── checkpoints/                     # Saved PyTorch checkpoints
└── README.md
```
> Adjust paths if your dataset layout differs.

---

## 🛠️ Setup
Tested with Python 3.10+.

```bash
# 1) Create environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu
pip install sentencepiece sacrebleu torchmetrics tqdm pyyaml
```

**Optional (nice to have):**
```bash
pip install rich matplotlib
```

---

## 🗂️ Data preparation
1. **Normalize & clean Arabic/English text** (Unicode NFKC, remove control chars, strip punctuation where appropriate).
2. **Train shared subword model** with SentencePiece (BPE or Unigram):
   ```bash
   spm_train      --input=data/train.ar,data/train.en      --model_prefix=artifacts/spm      --vocab_size=16000      --character_coverage=1.0      --model_type=bpe
   ```
3. **Encode splits**:
   ```bash
   spm_encode --model=artifacts/spm.model --output_format=id      < data/train.ar > data/train.ar.ids
   spm_encode --model=artifacts/spm.model --output_format=id      < data/train.en > data/train.en.ids
   # repeat for valid/test
   ```

**Low‑resource tips (Arabic):**
- Use **shared BPE** across AR/EN to increase lexical sharing.
- Enable **subword regularization** (`--sample_piece`) during training for robustness.
- Apply **label smoothing** (e.g., ε = 0.1) and **dropout** (e.g., 0.1–0.3).
- **Early stopping** on SacreBLEU or validation loss.
- If you have monolingual data, consider **back‑translation** to augment pairs.

---

## 🚀 Training (from the notebook)
Open `Text-Translation-Transformer.ipynb` and run the cells in order. The notebook includes:
- Dataset loading & tokenization (SentencePiece)
- Model definition (Encoder, Decoder, Generator)
- Training loop with Adam/AdamW, warmup schedule (Noam), mixed precision
- Checkpointing & early stopping
- Evaluation with SacreBLEU

**Typical hyperparameters (good starting point):**
- `d_model=512`, `n_heads=8`, `num_layers=6`
- `ffn_dim=2048`, `dropout=0.1`
- `bpe_vocab_size=16k` (tune 8k–32k)
- `batch_size=4096` tokens (dynamic batching by length is recommended)
- `lr=5e-4` with **warmup** (e.g., 4000 steps), **label_smoothing=0.1`

---

## 🔎 Evaluation
Use **SacreBLEU** for standardized BLEU:
```python
from sacrebleu import corpus_bleu
refs = [["the reference translation one", "another reference"]]
hyps = ["the system translation one"]
print(corpus_bleu(hyps, refs).format())
```
For Arabic, also inspect **chrF** and **COMET** if available. Manual spot‑checks are valuable due to morphology and diacritics.

---

## 💬 Inference
Greedy or beam search decoding (beam=4–8) with length penalty often helps.

```python
from pathlib import Path
import sentencepiece as spm
import torch

# Load
sp = spm.SentencePieceProcessor(model_file="artifacts/spm.model")
model = torch.load("artifacts/checkpoints/best.pt", map_location="cpu")
model.eval()

def translate_ar_to_en(text: str, max_len=128, beam=5):
    ids = sp.encode(text, out_type=int)
    # ... pass through encoder; run beam search in decoder ...
    # return detokenized string
```

> The notebook shows a complete example of encoding, decoding, and detokenization.

---

## 📈 Reproducibility
- Set `torch.manual_seed(42)` and deterministic flags where feasible.
- Log config & metrics to a JSON/YAML file per run.
- Save the **SentencePiece model**, **checkpoint**, and **commit hash** for each experiment.

---

## 🧪 Known limitations
- Truecasing/diacritics in Arabic are not modeled explicitly.
- Domain mismatch can degrade performance; consider domain‑adaptive fine‑tuning.
- For very low resources, back‑translation and multilingual transfer can be decisive.

---

## 🧯 Ethical use
Machine translation can produce errors that alter meaning. Do not rely on this model for high‑stakes contexts (legal, medical, safety‑critical) without human review. Be mindful of biases in training data.

---

## 📚 Reference
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. *Attention Is All You Need.* NeurIPS 2017.

---

## 📝 Citation
If you use this code/notebook in academic work, please cite the original paper above and this repository/notebook as appropriate.

---

## 📄 License
Specify your license (e.g., MIT) here.
