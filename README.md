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

```
   ```

**Low‑resource tips (Arabic):**
- Use **shared BPE** across AR/EN to increase lexical sharing.
- Enable **subword regularization** (`--sample_piece`) during training for robustness.
- Apply **label smoothing** (e.g., ε = 0.1) and **dropout** (e.g., 0.1–0.3).
- **Early stopping** on SacreBLEU or validation loss.
- If you have monolingual data, consider **back‑translation** to augment pairs.

---





## 📚 Reference
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. *Attention Is All You Need.* NeurIPS 2017.

---


---

## 📄 License
Specify your license (e.g., MIT) here.
