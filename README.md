# Human Activity Recognition using CNN

This project implements a **vision-based Human Activity Recognition (HAR)** model using **Convolutional Neural Networks (CNN)** and **Transfer Learning with MobileNetV2**.  
The model classifies **15 types of human activities** (e.g., sleeping, calling, running, dancing, eating, using_laptop, etc.) from image data.

---

## Project Overview

Human Activity Recognition (HAR) aims to identify human actions from image or sensor data.  
This project focuses on **vision-based HAR** — using deep learning techniques to recognize human activities from static images.

### Objectives
- Build an image classification model to recognize human activities.
- Use **MobileNetV2** as the pre-trained feature extractor.
- Apply advanced **image preprocessing** and **data augmentation** techniques.
- Evaluate model performance and generate predictions for unseen test data.

---

## Dataset Overview

The dataset consists of labeled images representing 15 different human activities.

| Dataset Split | Description | Count |
|----------------|--------------|--------|
| Training Set | Images across 15 activity categories | 12,600 |
| Test Set | Unlabeled images for prediction | 5,400 |
| Labels | Provided via CSV files | 15 |

Each image belongs to one of the following classes:
> calling, clapping, cycling, dancing, drinking, eating, fighting, hugging, laughing, running, sitting, sleeping, texting, using_laptop, listening_to_music.

---

## Dataset

The dataset used in this project contains over **12,000 images** categorized into 15 human activity classes.  
Due to GitHub file size limits, the dataset is **not included in this repository**.

You can use your own dataset structured as:
```
train/
├── image_1.jpg
├── image_2.jpg
├── image_3.jpg
└── ...
test/
├── image_1.jpg
├── image_2.jpg
├── image_3.jpg
└── ...
```
---

## Tech Stack

- **Language:** Python  
- **Frameworks:** TensorFlow, Keras  
- **Libraries:** NumPy, Pandas, OpenCV, Matplotlib, Scikit-learn, Seaborn  
- **Model Architecture:** MobileNetV2 (Transfer Learning)

---

## Project Workflow

### **1. Data Exploration & Visualization**
- Verified dataset structure and label distribution  
- Visualized image samples and color histograms  
- Analyzed dataset balance across 15 classes
- Color Histogram (HSV)

### **2. Preprocessing**
- Label Encoding using `LabelEncoder`  
- Image Resizing (224×224)  
- Normalization (scaling pixels to [0,1])  
- Data Augmentation (rotation, zoom, flip, shift, etc.)  

### **3. Feature Engineering**
- Histogram Equalization  
- Color Space Transformation (HSV)  
- Histogram of Oriented Gradients (HOG)  
- Principal Component Analysis (PCA)  

### **4. Model Building**
- Used **MobileNetV2** as base model (ImageNet weights)  
- Added custom dense layers with ReLU + Dropout  
- Compiled using Adam optimizer and categorical crossentropy  
- Implemented callbacks:
  - `ModelCheckpoint` – Save best model  
  - `EarlyStopping` – Prevent overfitting  

### **5. Training**
- 80–20 train-validation split  
- Augmented data batches using `ImageDataGenerator`  
- Trained model for 10 epochs  

### **6. Evaluation & Prediction**
- Tested on unseen images  
- Generated predictions and saved as `predictions.csv`

---

## Model Architecture

```
MobileNetV2 (frozen base)
↓
Global Average Pooling
↓
Dense(1024, ReLU)
↓
Dropout(0.5)
↓
Dense(15, Softmax)
```

---

## Results

- **Model Accuracy:** ~85–90% (depending on dataset splits)
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Saved Model:** `action_recognition_model.keras`  

---

## File Structure

```
human-activity-recognition-cnn/
│
├── train/                              # Training images
├── test/                               # Testing images
├── Training_set.csv                    # Training filenames and labels
├── Testing_set.csv                     # Testing filenames
│
├── human_activity_recognition.ipynb    # Main Jupyter Notebook
├── predictions.csv # Model predictions
├── model/
│   ├── action_recognition_model.keras  # Trained model
│   └── action_recognition.weights.h5   # Saved weights
│
├── requirements.txt                    # Dependencies
└── README.md                           # Project documentation
```


## Setup

### 1. Clone this repo
```bash
git clone https://github.com/SoeRatch/human-activity-recognition-cnn.git
cd human-activity-recognition-cnn
```

### 2. Create virtual environment & install deps
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Activate the virtual environment
```bash
# On macOS/Linux:
. venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the notebook
```bash
human_activity_recognition.ipynb
```

---

## Industry Applications

- **Healthcare:** Physical therapy tracking

- **Sports Analytics:** Player movement analysis

- **Security & Surveillance:** Automatic activity monitoring

- **Smart Homes:** Gesture or posture recognition


---