# 🐱🐶 Cats vs Dogs Image Classification using TensorFlow

A Convolutional Neural Network (CNN) built with **TensorFlow/Keras** to classify images of cats and dogs. The project automatically extracts the dataset, removes corrupted images, performs data augmentation, trains a deep learning model, evaluates its performance, and saves the trained model for future predictions.

---

## 📌 Features

* Automatic dataset extraction from ZIP file
* Detects and removes corrupted images
* Splits dataset into **80% training** and **20% validation**
* Applies real-time data augmentation
* CNN architecture built using TensorFlow/Keras
* Binary classification using Sigmoid activation
* Model evaluation on validation data
* Saves the trained model in `.keras` format

---

## 📂 Project Structure

```
project/
│
├── data/
│   ├── archive (9).zip
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
│
├── cats_vs_dogs_model.keras
├── main.py
├──.gitignore
└── README.md
```

---

## 🛠 Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pillow (PIL)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/cats-vs-dogs-classifier.git

cd cats-vs-dogs-classifier
```

Install the required libraries:

```bash
pip install tensorflow pillow numpy
```

---

## ▶️ Running the Project

Place the dataset ZIP file inside the `data` folder:

```
data/archive (9).zip
```

Run:

```bash
python main.py
```

The program will automatically:

1. Extract the dataset
2. Remove corrupted images
3. Create training and validation datasets
4. Train the CNN model
5. Evaluate model performance
6. Save the trained model

---

## 🧠 CNN Architecture

The model consists of:

* Input Layer (180 × 180 RGB images)
* Data Augmentation

  * Random Horizontal Flip
  * Random Rotation
  * Random Zoom
* Rescaling Layer
* Conv2D (32 filters)
* MaxPooling2D
* Conv2D (64 filters)
* MaxPooling2D
* Conv2D (128 filters)
* MaxPooling2D
* Flatten Layer
* Dense Layer (128 neurons)
* Dropout (0.5)
* Output Layer (1 neuron, Sigmoid)

---

## ⚙️ Model Configuration

| Parameter     | Value               |
| ------------- | ------------------- |
| Optimizer     | Adam                |
| Loss Function | Binary Crossentropy |
| Metric        | Accuracy            |
| Batch Size    | 32                  |
| Image Size    | 180 × 180           |
| Epochs        | 10                  |

---

## 📈 Data Preprocessing

Before training, the project performs several preprocessing steps:

* Extracts images from the ZIP archive.
* Verifies every image using Pillow.
* Removes corrupted or unreadable images.
* Resizes all images to **180 × 180** pixels.
* Normalizes pixel values to the range **0–1**.
* Randomly shuffles training data for better learning.

---

## 🔄 Data Augmentation

To improve model generalization and reduce overfitting, the following augmentations are applied during training:

* Horizontal Flip
* Random Rotation (15%)
* Random Zoom (15%)

Since these transformations are random, the model rarely sees the exact same image twice.

---

## 💾 Model Saving

After training, the model is saved as:

```
cats_vs_dogs_model.keras
```

It can later be loaded using:

```python
import tensorflow as tf

model = tf.keras.models.load_model("cats_vs_dogs_model.keras")
```

---

## 📊 Output

After training, the program displays:

* Training accuracy
* Validation accuracy
* Validation loss
* Final model accuracy

Example:

```
Final accuracy: 91.84%
Model saved successfully.
```

---

## 🚀 Future Improvements

* Add prediction for custom images.
* Implement Early Stopping.
* Save the best model using ModelCheckpoint.
* Plot training and validation accuracy/loss graphs.
* Use Transfer Learning (MobileNetV2, EfficientNet, ResNet50) for higher accuracy.
* Build a simple web interface using Flask or Streamlit.

---

## 📚 Learning Outcomes

This project demonstrates:

* Image preprocessing
* Data augmentation
* Convolutional Neural Networks (CNNs)
* Binary image classification
* TensorFlow/Keras workflow
* Model training and evaluation
* Saving and loading trained models

---

## 👨‍💻 Author

**Lokesh Kottur**

Deep Learning • Machine Learning • Python • TensorFlow

---

## 📄 License

This project is open-source and available for educational and learning purposes.
