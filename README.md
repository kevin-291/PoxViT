# Multi class Disease Classification using Vision Transformers

This project is a multi-class disease classification using Vision Transformers. The project aims to develop a model that can classify images pertaining to diseases such as Chickenpox, Cowpox, Measles, Monkeypox and HFMD (Hand, Foot, Mouth Disease). It can also classify healthy individuals also. Initially, a [Pre Trained Vision Transformer](https://huggingface.co/google/vit-base-patch16-224-in21k) is taken and fine tuned on the dataset. The model is then used to classify the images. The model is then deployed as an application using Streamlit. The model achieves an accuracy of 99% on the test set.

## Dataset

The dataset used for this project is taken from the [Mpox Skin Lesion Dataset (MSLD v2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20). The dataset has been preprocessed to get the images in a single directory pertaining to their classes.

## Prerequisites

Run the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

## Structure

The repository contains the following files:

- `config.py` contains the code to obtain the dataset in the required directory structure.
- `model.ipynb` contains the code to train the model.
- `app.py` contains the code to run the model as an application using Streamlit.

## Usage

- Clone the repository using the following command

```bash
git clone https://github.com/kevin-291/disease-classification.git
```

- Download the dataset.
- Add the source and destination paths in `config.py`.
- Run the code using the following command:

```bash
python config.py
```

- Add the required paths in `model.ipynb` and run the notebook to train the model.
- Add the path of the trained model in `app.py`.
- Run the application using the following command:

```bash
streamlit run app.py
```



