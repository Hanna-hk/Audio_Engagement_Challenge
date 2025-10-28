# Audio_Engagement_Challenge
Predict how long a listener stays engaged with a specific audio track based on the episode's characteristics – such as its length, genre, or the popularity of its creators.
---

## Technologies
- Python 3.9+
- pandas, numpy (data handling)
- scikit-learn, scipy (machine learning)
- matplotlib, seaborn (for data analysis)
- dill (for saving)
- kaggle (for dataset download)

---

## Features
- Data Ingestion - load dataset
- Data Transformation - preprocessing and feature engineering
- Model Training - train and evaluate ML model

---

## Installation and Running

1. Clone the repository:

git clone https://github.com/Hanna-hk/Audio_Engagement_Challenge.git
cd Audio_Engagement_Challenge

2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

3. Install dependencies:
```
pip install -r requirements.txt
```
4. For making predictions in the testing dataset:
```
python -m src.components.data_ingestion
```
