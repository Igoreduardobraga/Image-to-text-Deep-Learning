# Image Captioning with CNN+RNN and BLIP

**Department of Computer Science – ICEx – UFMG**  
**Course:** Introduction to Deep Learning  
**Professor:** Flávio Figueiredo  

## 📌 Final Project — Image-to-Text (Image Captioning)

Authors:
- Igor Eduardo Martins Braga — igoreduardobraga@ufmg.br 
- Etelvina Costa Santos Sá Oliveira — etelvina.oliveira2003@gmail.com  
- Filipe Pirola Santos — filipepirolasantos@hotmail.com  

---

## 🧠 Project Overview

This project compares two distinct approaches for the task of **automatically generating image captions** (Image Captioning):

- A custom **CNN + RNN (LSTM)** model implemented from scratch using **PyTorch**.
- A **pre-trained BLIP model** (`Salesforce/blip-image-captioning-base`) fine-tuned using Hugging Face.

Image captioning is a challenging task that combines **Computer Vision** and **Natural Language Processing (NLP)**, with applications in accessibility, multimedia indexing, and robotics.

---

## 🏗️ Model Architectures

### 🔷 CNN + RNN (Custom Implementation)
- **Encoder:** Pre-trained ResNet-50 (frozen), projecting images into embedding space.
- **Decoder:** LSTM with word embeddings and linear projection to vocabulary.
- **Training:** Adam optimizer, CrossEntropyLoss (ignoring `<PAD>`), and evaluation using BLEU and WER scores.

### 🔷 BLIP (Pre-trained)
- Based on the `Salesforce/blip-image-captioning-base` model from Hugging Face.
- Fine-tuned on flower images with human-written captions.
- Evaluation using `transformers.Trainer` with BLEU and WER metrics.

---

## 🖼️ Dataset

- **Oxford 102 Flowers**: 8,189 images across 102 flower categories.
- We randomly selected **5,000 images**, using **1 caption per image**.
- Preprocessing steps included:
  - Resizing to 480x480
  - Normalization aligned with pre-trained networks
  - Vocabulary built with words appearing at least 3 times

---

## ⚙️ Training Setup

| Hyperparameter     | Value                  |
|--------------------|------------------------|
| Optimizer          | Adam                   |
| Learning Rate      | 3e-4 to 5e-3           |
| Epochs             | 5 and 10               |
| Embedding Size     | 256                    |
| LSTM Hidden Size   | 512                    |
| Loss Function      | CrossEntropyLoss       |

BLEU and WER metrics were monitored throughout training to assess the quality of the generated captions.

---

## 📊 Results

### ✅ BLIP Model (Pre-trained)

| Metric         | Test Value   |
|----------------|--------------|
| Loss           | 0.2058       |
| BLEU Score     | **0.3336**   |
| WER            | **0.4546**   |

### 🔧 CNN + RNN Model

| Metric         | Test Value   |
|----------------|--------------|
| Loss           | 2.0252       |
| BLEU Score     | 0.1306       |
| WER            | 0.8168       |

### 📌 Conclusions

- The **BLIP model** outperformed the custom CNN+RNN model across all metrics.
- The custom model suffered from **overfitting starting from epoch 4**.
- The pre-trained model demonstrated better generalization and overall caption quality.

---

## 💡 Future Improvements

- Incorporate **attention mechanisms** (e.g., Attention layers, Transformers).
- Apply **beam search** during caption generation.
- Use techniques such as **dropout**, **early stopping**, and **data augmentation**.
- Integrate into real-world applications for accessibility and multimodal systems.

---

## 📚 References

- [Oxford 102 Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Hugging Face: Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [BLEU Score](https://en.wikipedia.org/wiki/BLEU) | [WER Metric](https://huggingface.co/spaces/evaluate-metric/wer)