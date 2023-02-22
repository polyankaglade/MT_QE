import pandas as pd
import numpy as np
import pickle
import streamlit as st

st.title('Quality Evaluation: predicting TER')

class Ensemble:
    def __init__(self):
        self.estimators = {}

        for i in range(1, 8):
            est = pickle.load(open(f'{i}_model.pickle', 'rb'))
            self.estimators[i] = est


    def predict(self, X):
        preds = []

        for i, clf in self.estimators.items():
            pred = clf.predict(X)
            preds.append(pred)

        avg = np.array(preds).mean(axis=0)
        return avg

model_load_state = st.text('Loading models...')
single_model = pickle.load(open('best_model.pickle', 'rb'))
ensemble_model = Ensemble()
model_load_state.text("Models loaded")

st.text('Training model scores')
scores = st.text('Training model scores')
def get_model(name):
    if name == 'Single':
        scores.text('MAE: 10.8374\nRMSE: 17.3682\nCorr: 0.7816')
        return single_model
    else:
        scores.text('MAE: 13.1663\nRMSE: 17.7422\nCorr: 0.7861')
        return ensemble_model



model_name = st.sidebar.selectbox(
    'Select model',
    ('Single', 'Ensemble')
)
model = get_model(model_name)

COLS = [f'F{i}' for i in range(1, 18)]
feature_pairs = [(1, 2),
                 (4, 5),
                 (16, 17)]


st.text('Here you can upload your file, containing 17 features, named F1 to F17.')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    data_raw = pd.read_csv(uploaded_file, encoding='cp1252')
    data = data_raw[COLS].copy()

    for i, p in enumerate(feature_pairs):
        data[f'F{i + 18}'] = data[f'F{p[0]}'] - data[f'F{p[1]}']

    st.text("We have extracted the features from the file and added 3 new ones.")
    st.text("Here is what's gonna be used for prediction")
    st.write(data)


    if st.button(label='Predict'):
        pred = model.predict(data)
        pred_df = pd.DataFrame({'pred': pred})
        st.write(pred_df)

        st.download_button(label='Download Prediction', data=pred_df.to_csv(index=False), file_name=model_name+"_pred.csv")