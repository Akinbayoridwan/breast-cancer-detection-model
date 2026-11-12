# Breast Cancer Detection Model

This project is a notebook-based machine learning workflow for detecting breast cancer from histopathology microscope images. It uses the BreakHis dataset, copies and organizes the relevant image files, then builds a compact latent representation with an autoencoder before classifying images as benign or malignant using a Random Forest classifier.

The notebook is an end-to-end experimental pipeline designed to explore how image feature extraction and classification can support medical image diagnosis. It combines image preprocessing, data visualization, unsupervised representation learning, supervised classification, and model explainability.

## Project Overview

The goal of the project is to distinguish between benign and malignant breast tissue samples using microscopic pathology images. In a clinical setting, pathologists review slides manually to identify abnormal tissue patterns. This project attempts to automate that decision support process by using image data and machine learning.

The workflow in the notebook is:

1. Load the dataset metadata from `Folds.csv`.
2. Organize image paths and labels.
3. Copy images into a simplified local folder structure.
4. Inspect class balance and image data.
5. Build a convolutional autoencoder to compress images into a lower-dimensional latent space.
6. Flatten those encoded features.
7. Train a Random Forest classifier on the encoded representations.
8. Evaluate the classifier with precision, recall, F1-score, confusion matrix, and accuracy.
9. Use SHAP values to explain which latent features contribute most to classification.

## Why This Problem Matters

Breast cancer is one of the most common cancers worldwide. Early and accurate diagnosis can significantly improve treatment outcomes. Histopathology image analysis is a critical part of cancer diagnosis, but manual review is time-consuming and can be affected by observer variability.

This project demonstrates a practical approach to assist pathologists by combining:

- image preprocessing
- feature extraction via deep learning
- classical machine learning classification
- model interpretation for explainability

It is not presented as a clinical diagnosis system, but as an educational and research-oriented prototype.

## Dataset

The notebook uses the BreakHis dataset, which is a widely used breast cancer histopathology dataset. It contains microscopic images of breast tissue with labels indicating whether the tissue is benign or malignant.

The metadata file used in the notebook is:

- `../input/breakhis/Folds.csv`

This file contains the paths to each image and includes information such as:

- `fold` : dataset split/fold identifier
- `mag` : magnification level
- `grp` : train/test grouping
- `path` : path to image within the dataset

The image directory used in the notebook is:

- `../input/breakhis/BreaKHis_v1/`

The notebook assumes that the BreakHis folder exists next to the project, or that the dataset is available in the same environment at that relative path.

## Repository Structure

The project currently contains:

- `Breast Cancer Model.ipynb` — the main notebook containing the full analysis and model pipeline
- `README.md` — project documentation
- `pyproject.toml` — Python project metadata and dependency configuration
- `src/breast_cancer_detection_model/__init__.py` — package placeholder

## Environment and Dependencies

The project depends on a scientific Python stack, including:

- Python 3.13+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV (`cv2`)
- scikit-learn
- TensorFlow / Keras
- SHAP

These are reflected in the `pyproject.toml` for the core libraries, although the notebook itself also uses TensorFlow and OpenCV directly during execution.

## Notebook Workflow in Detail

### 1. Imports and Setup

The notebook begins by importing the libraries needed for:

- numerical processing (`numpy`, `pandas`)
- plotting (`matplotlib`, `seaborn`)
- image handling (`cv2`)
- model training (`tensorflow`, `sklearn`)
- data augmentation and callbacks (`ImageDataGenerator`, `EarlyStopping`, `ModelCheckpoint`)

This stage establishes the environment for the full workflow.

### 2. Loading the Metadata

The notebook reads:

```python
df = pd.read_csv("../input/breakhis/Folds.csv")
```

This creates a DataFrame containing all file records. Since the file uses a column named `filename` but the notebook expects a `path` column, it renames the column:

```python
df = df.rename(columns={"filename": "path"})
```

This is a useful preprocessing step because later code uses the image path to reconstruct labels and file locations.

### 3. Creating Image Labels and File Names

The notebook extracts information embedded in the path structure:

```python
df['filename'] = df['path'].apply(lambda x: x.split("/")[-1])
df["label"] = df['path'].apply(lambda x: x.split("/")[3])
```

This logic does two things:

- extracts the actual image filename from the path
- extracts the tissue label from the folder hierarchy, which is either `benign` or `malignant`

This is a crucial step because the machine learning model ultimately needs a binary target label.

### 4. Copying Images into a Working Folder

The notebook creates a local output folder:

```python
os.makedirs("../Cancer/")
```

Then it loops through every image path and copies each file into a simplified folder structure:

```python
for p in df['path']:
    src = "../input/breakhis/BreaKHis_v1/" + p
    dest = "../Cancer/"
    dest = os.path.join(dest, src.split("/")[7] + "_" + src.split("/")[-1])
    shutil.copyfile(src, dest)
```

This step gives the project a uniform folder containing files named like:

- `benign_<image_name>.png`
- `malignant_<image_name>.png`

This makes it easier to load images by filename and assign labels consistently.

### 5. Dataset Exploration and Visualization

The notebook visualizes image distribution and class balance.

#### Class count plot

```python
plt.figure(figsize=(9,6))
sns.countplot(data=df, x='label', palette='Set2')
```

This chart shows the number of benign and malignant cases. It helps confirm dataset balance and identify major class distribution differences.

#### Pie chart

```python
tumor_counts = df['label'].value_counts()
plt.pie(tumor_counts, labels=tumor_counts.index, autopct='%1.1f%%')
```

This provides a percentage-based summary of the class distribution.

#### Patient-wise breakdown

The notebook adds a patient identifier and groups records by patient and label:

```python
df['patient_id'] = df['path'].apply(lambda x: x.split("/")[2])
grouped = df.groupby(['patient_id', 'label']).size().unstack(fill_value=0)
```

This helps assess whether images are distributed across patients fairly and whether multiple samples per patient are being used in the dataset.

### 6. Sample Image Inspection

The notebook extracts image dimensions and reads test images with OpenCV:

```python
def get_image_dimensions(filepath):
    try:
        img = cv2.imread(filepath)
        h, w = img.shape[:2]
        return w, h, w * h
    except:
        return 0, 0, 0
```

Then it adds:

- width
- height
- resolution

These metadata fields are useful for verifying the consistency of image sizes and understanding their visual scale.

The notebook then displays a grid of sample images from the dataset using `matplotlib` and OpenCV conversion from BGR to RGB:

```python
img = cv2.imread(row['full_path'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

This step helps confirm that the image files are correctly loaded and that the labels align with the actual tissue morphology.

### 7. Image Collection and Label Assignment

After organizing images into `../Cancer/`, the notebook rebuilds a DataFrame from the folder contents:

```python
df = pd.DataFrame(os.listdir("../Cancer/"))
df = df.rename(columns = {0:'file_loc'})
df['label'] = df['file_loc'].apply(lambda x:x.split("_")[0])
df['class'] = df['label'].apply(lambda x: 0 if x =='benign' else 1)
df.set_index("file_loc", inplace=True)
```

This creates a clean dataset for training where:

- `label` is the string category (`benign` or `malignant`)
- `class` is a numeric target (`0` for benign, `1` for malignant)
- the index is the filename, which is convenient for loading images

### 8. Image Loading Pipeline

The notebook defines a custom image loader:

```python
IMG_SIZE = 128

def load_images(df, base_path="../Cancer/"):
    images = []
    labels = []
    for filename in df.index:
        img = cv2.imread(base_path + filename)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        images.append(img)
        labels.append(df.loc[filename, 'class'])
    return np.array(images), np.array(labels)

X, y = load_images(df)
```

This is an important preprocessing step because:

- all images are resized to a consistent dimension
- pixel values are normalized to `0..1`
- inputs are converted to a NumPy array suitable for TensorFlow and machine learning models

The constant `IMG_SIZE = 128` means each image is reduced to 128x128x3 before model training.

## Autoencoder Model

The core of the notebook is an autoencoder. An autoencoder is an unsupervised neural network designed to learn a compressed representation of the input data by attempting to reconstruct the original image from a reduced latent encoding.

### Autoencoder architecture

The input image is shaped as:

```python
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
```

Then a convolutional encoder is built:

```python
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
```

This reduces the image spatial dimensions while learning useful feature maps.

The decoder reconstructs the image:

```python
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
```

The output is a reconstructed 3-channel image similar to the input.

### Why use an autoencoder?

Instead of feeding raw high-dimensional pixels directly into a classifier, the model first learns a compressed representation. This helps:

- reduce feature dimensionality
- remove noise and redundancy
- capture the most informative structure in the images

The notebook then takes the encoded representation and reshapes it into a 1D feature vector:

```python
encoder = Model(inputs=autoencoder.input, outputs=encoded)
X_encoded = encoder.predict(X)
X_flat = X_encoded.reshape((X_encoded.shape[0], -1))
```

This generates a compact set of latent features used for classification.

### Training the autoencoder

The autoencoder is trained with:

```python
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=20, batch_size=32, shuffle=True, validation_split=0.2)
```

This means it tries to reconstruct the original images from the compressed representation. The training objective is minimizing mean squared error (MSE), which measures pixel-wise reconstruction error.

This stage is unsupervised because the labels are not used to train the autoencoder; the network learns the structural patterns in the images directly.

## Random Forest Classifier

After generating the latent features, the notebook splits the data:

```python
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
```

This creates a training set and a holdout test set.

Then the model is trained:

```python
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
```

Random Forest is a classic ensemble method that builds many decision trees and aggregates their predictions. It works well with tabular features such as latent embeddings and is often interpretable compared with a deep network.

## Model Evaluation

The notebook predicts the test labels:

```python
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

The classification report contains:

- precision
- recall
- F1-score
- support

for each class, giving a breakdown of how well the model distinguishes benign from malignant samples.

A confusion matrix is also created:

```python
cm = confusion_matrix(y_test, y_pred)
```

Then a heatmap is drawn:

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
```

This heatmap visualizes:

- true negatives
- false positives
- false negatives
- true positives

The notebook also computes:

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")
```

This gives an overall model performance measure.

## Explainability with SHAP

The notebook uses SHAP (SHapley Additive exPlanations) to interpret the model. SHAP values estimate how much each feature contributes to the prediction for a given observation.

### Resampling for explanation

The notebook creates a smaller subset of the test data:

```python
X_test_sample, y_test_sample = resample(
    X_test, y_test,
    n_samples=int(0.2 * len(X_test)),
    random_state=42,
    stratify=y_test
)
```

This improves performance and keeps the explanation step manageable.

### Tree Explainer

```python
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_sample, check_additivity=False)
```

Tree Explainer works efficiently with decision tree-based models like Random Forest.

### Feature importance

The code calculates the absolute mean SHAP value per feature:

```python
feature_importance = np.abs(shap_values[1]).mean(axis=0)
```

Then it ranks the features by importance to identify the latent features that most strongly influence malignant predictions.

### SHAP summary plot

```python
shap.summary_plot(shap_values[1], X_test_sample, max_display=10)
```

This plot shows which latent features contribute most to the model’s decisions and in what direction they push predictions. It helps answer: “What patterns in the learned image embeddings are driving the classification?”

The final section of the notebook builds a DataFrame of feature importance and prints the top latent features contributing to predictions.

## Key Takeaways

This project demonstrates a hybrid machine learning pipeline for medical image classification:

- images are loaded and normalized
- labels are derived from folder structure
- image features are compressed using an autoencoder
- latent image embeddings are used to train a Random Forest classifier
- model performance is evaluated using standard metrics
- SHAP is used to explain which features influence results

This is a strong example of a practical workflow combining deep learning and classical machine learning for image classification tasks.

## How to Run the Project

### Prerequisites

Make sure the BreakHis dataset is available locally in a location matching the notebook paths, for example:

```text
../input/breakhis/BreaKHis_v1/
```

If you are in a project folder, ensure the dataset is placed relative to the notebook or adjust the paths accordingly.

### Install dependencies

Using `pip`:

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow shap
```

Using `uv`:

```bash
uv sync
```

### Open the notebook

Open `Breast Cancer Model.ipynb` in Jupyter Notebook, VS Code notebooks, or Jupyter Lab and run the cells in order.

## Limitations and Notes

This project is a useful prototype, but it has some limitations:

- It depends on a specific dataset path layout
- The dataset is not fully packaged with the repository
- The notebook is exploratory rather than a production-ready training pipeline
- The autoencoder and classifier are trained on a relatively simple preprocessing workflow
- No external validation or deployment pipeline is included

## Conclusion

The Breast Cancer Detection Model notebook shows a complete image classification pipeline for distinguishing benign and malignant breast cancer histology samples. It combines data preparation, exploratory analysis, autoencoder-based feature learning, Random Forest classification, and SHAP-based explainability in one cohesive workflow.

This makes it a strong educational project for understanding how medical image classification can be tackled with a practical machine learning pipeline.
