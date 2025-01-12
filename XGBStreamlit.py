# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:40:31 2025

@author: ostac
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from urllib.parse import urlparse
import re
import os

# Verificarea existenței modelului
if not os.path.exists("Fine_Tuned_XGBoost.joblib"):
    st.error("Fișierul modelului nu a fost găsit în directorul curent.")
    st.stop()

try:
    import joblib
except ModuleNotFoundError:
    st.error("Biblioteca 'joblib' nu este instalată. Verificați fișierul requirements.txt.")
    st.stop()


# Path to the saved model
xgb_model_path = "Fine_Tuned_XGBoost.joblib"

# Load the fine-tuned XGBoost model
fine_tuned_xgb_model = joblib.load(xgb_model_path)

# Feature extraction function
def extract_features(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname if parsed_url.hostname else ''
    path = parsed_url.path if parsed_url.path else ''
    query = parsed_url.query if parsed_url.query else ''
    
    # Compute entropy for a string
    def calculate_entropy(s):
        probabilities = [float(s.count(c)) / len(s) for c in set(s)]
        return -sum([p * np.log2(p) for p in probabilities])

    features = {
        'length_url': len(url),
        'length_hostname': len(hostname),
        'length_path': len(path),
        'nb_dots': url.count('.'),
        'nb_hyphens': url.count('-'),
        'nb_slashes': url.count('/'),
        'nb_digits': len(re.findall(r'\d', url)),
        'contains_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', hostname) else 0,
        'check_www': 1 if 'www' in hostname else 0,
        'check_com': 1 if '.com' in hostname else 0,
        'count_subdomains': hostname.count('.') - 1,
        'shortening_service': 1 if re.search(r'bit\.ly|goo\.gl|tinyurl|short\.to|ow\.ly', url) else 0,
        'abnormal_subdomain': 1 if re.search(r'(http[s]?://(w[w]?|\d))([w]?(\d|-))', url) else 0,
        'count_special_chars': sum(1 for c in url if c in ['@', '!', '$', '%', '^', '&', '*', '(', ')']),
        'path_extension': 1 if re.search(r'\.(exe|zip|pdf|js|html|php|asp)$', path) else 0,
        'avg_word_length': np.mean([len(word) for word in re.findall(r'\w+', hostname)]) if hostname else 0,
        'total_words': len(re.findall(r'\w+', hostname)),
        'ratio_digits_url': len(re.findall(r'\d', url)) / (len(url) + 0.001),
        'ratio_digits_host': len(re.findall(r'\d', hostname)) / (len(hostname) + 0.001),
    }
    return pd.DataFrame([features])

# Streamlit App
st.title("Phishing URL Detection (XGBoost)")
st.write("""
Această aplicație vă permite să preziceți dacă un URL este de tip phishing sau nu, utilizând modelul optimizat:
- **XGBoost**

Aplicația a fost creată de Ostache Andrei Tudor în cadrul proiectului de Securitate Cibernetică.
""")

# Input URL
url_input = st.text_input("Introduceți un URL suspect:", placeholder="https://example.com")

if st.button("Prezice tipul URL-ului"):
    if url_input:
        # Extract features from the input URL
        features_df = extract_features(url_input)

        # XGBoost prediction
        xgb_prediction_proba = fine_tuned_xgb_model.predict_proba(features_df)[0]
        xgb_prediction = "Phishing" if xgb_prediction_proba[1] > 0.5 else "Legitim"

        # Display results
        st.subheader("Rezultatele Predicției")
        
        st.write(f"**Predicție XGBoost:** {xgb_prediction}")
        st.write(f"**Probabilitate Phishing:** {xgb_prediction_proba[1]:.2%}")
        st.write(f"**Probabilitate Legitim:** {xgb_prediction_proba[0]:.2%}")
    else:
        st.error("Vă rugăm să introduceți un URL valid!")