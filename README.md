# Mental Health Depression Prediction

This project predicts whether a person may be at risk of depression based on mental health and lifestyle related factors such as academic pressure, work pressure, sleep duration, dietary habits, financial stress, and family history of mental illness.

## Project Files

- `app.py` - polished Streamlit web app for prediction
- `train_and_save_model.py` - trains the model and saves the deployment bundle
- `depression_model_bundle.pkl` - saved model bundle used by the app
- `requirements.txt` - project dependencies
- `Sam_wording_output.ipynb` - notebook version of the project

## Features

- Predicts **Depressed** or **Not Depressed**
- Uses **word-based inputs** instead of encoded number values for category fields
- Cleaner app layout with helpful labels and descriptions
- Ready for GitHub and Streamlit Community Cloud deployment

## How to Run Locally

### 1. Install libraries

```bash
pip install -r requirements.txt
```

### 2. Keep the dataset in the project folder

Place your dataset file in the same folder with this exact name:

```text
Student_Depression_Dataset.csv
```

### 3. Train and save the model bundle

```bash
python train_and_save_model.py
```

This will create:

```text
depression_model_bundle.pkl
```

### 4. Run the app

```bash
streamlit run app.py
```

## Files to Upload to GitHub

Upload these files to your repository:

- `app.py`
- `train_and_save_model.py`
- `requirements.txt`
- `README.md`
- `Sam_wording_output.ipynb`
- `depression_model_bundle.pkl` after running the training script

## How to Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Sign in with GitHub.
4. Click **New app**.
5. Select your repository.
6. Choose `app.py` as the main file.
7. Click **Deploy**.

## Important Note

This project is for educational purposes only. It should not be used as a real medical or psychiatric diagnosis tool.
