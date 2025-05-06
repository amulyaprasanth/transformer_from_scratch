# Transformer from Scratch 🚀

This repository contains a clean and modular implementation of the original Transformer model (as proposed in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)) using PyTorch.

It includes:
- A custom **Transformer** class with Encoder and Decoder modules
- Training pipeline using **tqdm** progress bars
- Tokenization with Hugging Face `datasets` and `tokenizers`
- Teacher forcing using target shifting (as done in standard implementations)
- Padding masks and look-ahead masks to correctly handle variable-length sequences

---

## 📚 Dataset

We use Hugging Face's translation datasets (e.g., English → Czech) for training. 
Tokenization is handled using a pretrained tokenizer with max sequence length support.

---

## 🚀 Training

The model can be trained using the provided `train_model` function.  
It shows a real-time progress bar and calculates average loss per epoch.

Example:
```python
trained_model = train_model(model, train_loader, num_epochs=10, lr=3e-4, device='cuda')
````

---

## ✅ Requirements

* Python 3.8+
* PyTorch
* Hugging Face `datasets`
* `tqdm`

Install requirements:

```bash
pip install torch datasets tqdm
```

---

## 🤖 Special Note

> During this project, I used **ChatGPT** (GPT-4) as a mentor to help correct bugs, clarify ambiguous points from the paper, and guide through complex parts of the implementation.
> It acted as an instant code reviewer and a paper simplifier 💪.

---

## 📝 Reference

* Vaswani et al., 2017: [Attention is All You Need](https://arxiv.org/abs/1706.03762)

---


