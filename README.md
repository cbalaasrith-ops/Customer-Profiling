## Customer Profiling Project

This repository contains four Jupyter notebooks that train and use convolutional models to predict **age**, **gender**, and **emotion** from face images in the **UTKFace** dataset.

All logic lives inside the notebooks; there is no separate Python package, API, or UI.

---

## Notebooks

- `2.1_train_age_model.ipynb` – trains an age prediction model and saves it to disk.  
- `2.2_train_gender_model.ipynb` – trains a gender classifier (male/female) and saves it to disk.  
- `2.3_train_emotion_model.ipynb` – trains an emotion classifier and saves it to disk.  
- `3.1_Pred_Final (1).ipynb` – loads the three saved models and runs them on new images.

The prediction notebook assumes that all three trained model files are available at the paths referenced inside it.

---

## Data

The notebooks use the **UTKFace** dataset, where each image filename encodes labels:

- `[age]_[gender]_[race]_[time].jpg`
  - `age`: integer age
  - `gender`: `0` = male, `1` = female

Expected layout (can be changed if you update paths in the notebooks):

- Images under `./input/UTKFace/`  
- Each image:
  - Read with OpenCV  
  - Converted to grayscale  
  - Resized to `100 × 100` pixels  
  - Stored in NumPy arrays along with its label(s)

The age and emotion notebooks follow the same loading and preprocessing pattern but map labels to their own targets.

When run in Google Colab, the notebooks mount Google Drive with `drive.mount('/content/drive')` and read the dataset from Drive paths; if you move the data to a different Drive folder or run locally, update the dataset paths in the first few cells of each notebook.

---

## Models

Each training notebook builds a small CNN in Keras/TensorFlow with:

- Input shape `(100, 100, 1)`  
- Several `Conv2D + Dropout + MaxPooling2D` blocks  
- A fully connected head for the final prediction (age / gender / emotion)

For gender:

- Output layer: 2 units with `sigmoid` activation  
- Loss: `sparse_categorical_crossentropy`  
- Optimizer: `adam`

The age and emotion models use the same building blocks but adjust the final layer and loss to match their prediction tasks.

---

## How to run

1. **Set up environment**  
   Use either a local Python environment with Jupyter or a hosted notebook service (e.g. Colab). Install the following Python packages (exact versions are not fixed):
   - `tensorflow`
   - `keras`
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `opencv-python`
   - `scikit-learn`

2. **Prepare data**  
   - Download UTKFace.  
   - Place all images under `./input/UTKFace/` (or adjust the paths in the notebooks).  
   - Verify that filenames follow `[age]_[gender]_[race]_[time].jpg`. If not, update the filename parsing cells in the notebooks.

3. **Train models**  
   - Open `2.1_train_age_model.ipynb` and run all cells. The notebook reads images, builds the age model, trains it, and saves the trained model to disk (e.g. `./output/age_model.h5`).  
   - Open `2.2_train_gender_model.ipynb` and run all cells to train the gender model and save `gender_model.h5` (or a similarly named file).  
   - Open `2.3_train_emotion_model.ipynb` and run all cells to train the emotion model and save its `.h5` file.

4. **Run predictions**  
   - Open `3.1_Pred_Final (1).ipynb`.  
   - Check the paths to the three saved model files and adjust them if your `.h5` files are stored elsewhere.  
   - Point the notebook to a directory containing one or more new face images.  
   - Run all cells. For each image, the notebook will print or display the predicted age, gender, and emotion.

---

## Notes

- All functionality is notebook-based; there is no command-line interface or app.  
- The code assumes UTKFace-style filenames for label extraction; if filenames or folder structure change, the parsing logic must be updated.  
- For consistent results, use a similar environment (TensorFlow/Keras versions, directory layout) to the one expected in the notebooks.

You can further adapt these notebooks by changing the image size, network depth, or loss functions, or by swapping UTKFace for another labeled face dataset and updating the loading code accordingly.


# Customer-Profiling-
