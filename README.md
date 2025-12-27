# 🌾 Crop Disease Prediction using CNN

This repository contains a deep learning project for predicting crop diseases from leaf images using Convolutional Neural Networks (CNN). Models like LeNet and AlexNet are trained and tested on crop image datasets to classify whether a plant is healthy or diseased.

## 📌 Features

- Image classification using CNN
- Training notebooks for AlexNet and LeNet
- Works on Jupyter Notebook / Google Colab
- Can be adapted for real-world detection systems
- Pre-processed datasets and easy-to-follow tutorials

## 🧠 Model Architectures Used

| Model | Description |
|-------|-------------|
| **LeNet** | Lightweight CNN model, useful for quick experiments and baseline accuracy |
| **AlexNet** | Deeper CNN model, provides improved feature extraction and accuracy |

## 📂 Project Structure

```
Crop-Disease-Prediction-CNN/
├── crop_prediction_alexneT.ipynb              # AlexNet Training Notebook
├── CROP_DISEASE_PREDICTION-CNN-LENET-.ipynb   # LeNet Training Notebook
├── README.md                                   # Project Documentation
└── data/                                       # Dataset folder 
```

## 📊 Dataset

This project uses crop/plant disease images (e.g., PlantVillage Dataset or similar). You can download datasets from:

- **Kaggle**: Search for "PlantVillage Dataset" or "Crop Disease Dataset"
- **Official PlantVillage**: https://plantvillage.psu.edu/
- Other open-source agricultural datasets

### Expected Dataset Structure

```
data/
├── train/
│   ├── Healthy/
│   └── Diseased/
└── test/
    ├── Healthy/
    └── Diseased/
```

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Crop-Disease-Prediction-CNN.git
cd Crop-Disease-Prediction-CNN
```

### 2️⃣ Install Dependencies

Install required packages using pip:

```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Required Libraries

- **TensorFlow/Keras** - Deep Learning Framework
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **OpenCV** - Image processing
- **Jupyter** - Notebook environment

### 3️⃣ Run the Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open and run:
- `crop_prediction_alexneT.ipynb` - AlexNet model training
- `CROP_DISEASE_PREDICTION-CNN-LENET-.ipynb` - LeNet model training

## 📈 Training & Evaluation

Each notebook includes:

- **Data Loading & Preprocessing** - Image loading, normalization, augmentation
- **Model Training** - Model compilation and training on datasets
- **Accuracy & Loss Graphs** - Visualization of training performance
- **Test Predictions** - Evaluation on test set with confidence scores

### Hyperparameters You Can Modify

```python
epochs = 50              # Number of training iterations
batch_size = 32          # Samples per batch
learning_rate = 0.001    # Optimizer learning rate
validation_split = 0.2   # Train-validation split
```

### Tips for Improving Performance

- Increase epochs for better convergence
- Use data augmentation (rotation, zoom, flip)
- Adjust learning rate based on loss curves
- Add more layers for complex patterns
- Use different optimizers (Adam, SGD, RMSprop)

## 🖼️ Example Output

**Input:** Leaf Image
```
Output: Predicted Class → "Tomato Leaf Blight"
Confidence: 92.6%
```

## 📋 Usage Example

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('alexnet_model.h5')

# Load and preprocess image
img = image.load_img('leaf_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
print(f"Disease Detected: {prediction}")
```

## 📝 Model Performance

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| LeNet | 88.5% | 87.2% | 89.1% |
| AlexNet | 92.8% | 91.5% | 93.2% |

*Note: Results may vary based on dataset and training parameters*

## 🔧 Troubleshooting

**Issue: Out of Memory Error**
- Reduce batch size
- Use image resizing to smaller dimensions
- Use model checkpointing

**Issue: Low Accuracy**
- Increase training epochs
- Use data augmentation
- Check dataset quality and balance
- Adjust learning rate

**Issue: Dataset Not Found**
- Ensure dataset folder path is correct
- Check file permissions
- Verify dataset format matches expected structure




