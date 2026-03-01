# 🧠 Brain Tumor MRI Classifier

A CNN-based image classifier trained on 17-class brain tumor MRI data, built with PyTorch. Includes training, evaluation, and (coming soon) Grad-CAM visualization for interpretability.

---

## 📁 Dataset

[Brain Tumor MRI Images – 17 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes) via Kaggle.

Downloaded automatically using `opendatasets`:
```bash
pip install opendatasets
```
You'll need a Kaggle API key (`kaggle.json`) in your working directory.

---

## 🏗️ Model Architecture

A straightforward but effective CNN stacked with `nn.Sequential`:

```
Conv2d(3→32) → ReLU → MaxPool
Conv2d(32→64) → ReLU → MaxPool
Conv2d(64→128) → ReLU → MaxPool
Flatten
Linear(128×16×16 → 256) → ReLU → Dropout(0.5)
Linear(256 → 17)
```

- Input size: `128×128` RGB
- Output: 17 tumor classes
- Optimizer: `AdamW` (lr=1e-4)
- Loss: `CrossEntropyLoss`
- Epochs: 20
- Train/Test split: 80/20

---

## 🚀 Usage

**1. Clone the repo**
```bash
git clone https://github.com/TheOrganic-code/Brain-Tumor-Classifier
cd Brain-Tumor-Classifier
```

**2. Install dependencies**
```bash
pip install torch torchvision opendatasets pandas
```

**3. Run**
```bash
python classifier.py
```
The script will automatically download the dataset on first run (Kaggle credentials required).

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Train epochs | 20 |
| Batch size | 32 |
| Test accuracy | ~88.98% |
| Test loss | ~0.3974 |

> Update the table above after your run — accuracy varies by hardware/seed.

---

## 🔍 Interpretability (Coming Soon)

Planning to add **Grad-CAM** visualizations to highlight which regions of the MRI the model attends to — making predictions more trustworthy for diagnostic use cases.

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- opendatasets
- pandas

---

## 📌 Notes

- Originally developed on Google Colab (hence the `opendatasets` download)
- Runs on GPU automatically if CUDA is available, falls back to CPU
- `pin_memory=True` and `num_workers=2` set for faster data loading on GPU machines

---

## 👤 Author

**Ayush Pandey**  
B.Tech Mathematics & Computing, RGIPT  
[github.com/TheOrganic-code](https://github.com/TheOrganic-code)
