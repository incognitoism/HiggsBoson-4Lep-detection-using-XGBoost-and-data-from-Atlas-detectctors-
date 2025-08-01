import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from tensorflow import keras
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="HB 4lep detection using XGBoost",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    mc_df = pd.read_parquet("mc_df.parquet")
    data_df = pd.read_parquet("data_df.parquet")
    return mc_df, data_df

@st.cache_resource
def load_models():
    bst = joblib.load("xgb_model.joblib")
    model_dnn = keras.models.load_model("dnn_model.keras")
    scaler = joblib.load("scaler.joblib")
    return bst, model_dnn, scaler

try:
    mc_df, data_df = load_data()
    bst, model_dnn, scaler = load_models()
except FileNotFoundError:
    st.error("Error: Model or data files not found. Please run `train_and_save.py` first to generate the necessary assets.")
    st.stop()

features = [f'lep_pt_{i}' for i in range(4)] + [f'lep_eta_{i}' for i in range(4)] + \
           [f'lep_phi_{i}' for i in range(4)] + [f'lep_E_{i}' for i in range(4)] + ['M_4l']

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Introduction", "Exploratory Data Analysis", "AI Model Performance", "The Discovery Plot" , "Documentation"])

if page == "Project Introduction":
    st.title("A prediction mechanism that can successfully predict and isolate the Higgs particle ie  4 Lepton ( 4 Electron / 4 muon / 2electron + 2 muon ) signatures from excessive background events. ")
    st.markdown("""
    The project trains and evaluates two independent models (XGBoost and DNN) to see which one performs better. They are not used successively (one after the other). The app compares their performance on the "AI Model Performance" page and then uses the best one (XGBoost, in this case) for the final "Discovery Plot".

    ### The Physics Context
    - **The Standard Model:** Our current best theory of fundamental particles and forces.
    - **The Higgs Boson:** The particle responsible for giving other particles mass, discovered at CERN in 2012.
    - **The Large Hadron Collider (LHC):** The world's most powerful particle accelerator, used to recreate the conditions of the early universe.

    ### The Challenge: Signal vs. Background
    The central challenge in experimental particle physics is the statistical separation of a rare signal process from copious background processes. The production cross-section for the Higgs boson is orders of magnitude smaller than that of many Standard Model backgrounds. We search for the Higgs via its decay products, focusing on the $H \\rightarrow ZZ^* \\rightarrow 4\\ell$ channel (where $\ell$ is an electron or muon). This is known as the "golden channel" due to its clean experimental signature and the high precision with which the four-lepton invariant mass, $m_{4\ell}$, can be reconstructed.

    The primary "irreducible" background is the direct production of a $ZZ$ pair, which can decay to the same four-lepton final state. While the final state particles are identical, the kinematics of the event—the momenta and angular distributions of the leptons—differ subtly due to the different spin and production mechanisms of the parent particle.

    This analysis leverages machine learning to exploit these kinematic differences. We train advanced classifiers (XGBoost and a Deep Neural Network) on high-fidelity Monte Carlo simulations to learn the multi-dimensional decision boundary that optimally separates signal from background. The output of these models is a continuous discriminant score, which allows us to define signal-enriched regions of phase space, thereby maximizing the statistical significance of the final measurement.
    """)
    st.image("higgs_boson.png", caption="A simulated particle collision event at the LHC.")

elif page == "Exploratory Data Analysis":
    st.title("1.Exploratory Data Analysis (EDA)")
    st.markdown("We created a usable simulated datatet (mc_df) to show inherent differences between signal and background , there are three implemented histograms -> 1.Four-Lepton Invariant Mass , 2.Lepton Transverse Momentum ,  3.Invariant Mass after AI Cut:")
    
    features_to_plot = {
        "Four-Lepton Invariant Mass ($M_{4\ell}$)": "M_4l",
        "Leading Lepton Transverse Momentum ($p_T$)": "lep_pt_0",
        "Subleading Lepton Transverse Momentum ($p_T$)": "lep_pt_1"
    }
    selected_feature_name = st.selectbox("Select a Kinematic Variable to Plot:", list(features_to_plot.keys()))
    feature_to_plot = features_to_plot[selected_feature_name]

    st.write(f"### Distribution for `{selected_feature_name}`")
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = 50
    if 'M_4l' in feature_to_plot:
        bins = np.linspace(80, 180, 50)
    else:
        bins = np.linspace(0, 200, 50)

    signal_data = mc_df[mc_df['is_signal'] == 1]
    background_data = mc_df[mc_df['is_signal'] == 0]

    ax.hist(background_data[feature_to_plot], bins=bins, weights=background_data['total_weight'],
             label='Background', color='dodgerblue', alpha=0.7, density=True)
    ax.hist(signal_data[feature_to_plot], bins=bins, weights=signal_data['total_weight'],
             label='Signal', color='red', histtype='step', linewidth=2, density=True)

    ax.set_xlabel(f"{selected_feature_name} [GeV]", fontsize=12)
    ax.set_ylabel("Normalized Events", fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    st.pyplot(fig)

elif page == "AI Model Performance":
    st.title("2.AI Model Performance Comparison")
    st.markdown("AUC Comparison for XG Boost , DNN vs Rndom chance ( 0.5 ) where a AUC od 1.0 is a perfect classifier ")

    X = mc_df[features]
    y = mc_df['is_signal']
    _, X_test, _, y_test, _, _ = train_test_split(X, y, mc_df['total_weight'], test_size=0.3, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)

    y_pred_proba_xgb = bst.predict(xgb.DMatrix(X_test))
    y_pred_proba_dnn = model_dnn.predict(X_test_scaled).ravel()

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    fpr_dnn, tpr_dnn, _ = roc_curve(y_test, y_pred_proba_dnn)
    auc_dnn = auc(fpr_dnn, tpr_dnn)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'XGBoost ROC (AUC = {auc_xgb:.4f})')
    ax.plot(fpr_dnn, tpr_dnn, color='green', lw=2, label=f'DNN ROC (AUC = {auc_dnn:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Model ROC Curve Comparison', fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)

elif page == "The Discovery Plot":
    st.title("3. The Discovery Plot: Using XGBoost for determination of Montecarlo signals based on the Testdata provided in the Model ")
    st.markdown("""
    Here i am using the best model ( XGBoost ) for deterministic verification of Montecarlo signals based on the users choice of cut-off value , a noisy sample is produced at a value of 0.50 while a cut off of 0.99 gives a cleaner histogram with a few spikes and only one spike at 125GeV which is signature for the target boson.

    **Use the slider below to change the classification threshold.** A higher threshold is stricter, keeping fewer events but creating a purer sample. Observe how a potential Higgs peak around 125 GeV might emerge from the background as you increase the cut value.
    """)
   
    X_real = data_df[features].astype('float32')
    
    predictions = bst.predict(xgb.DMatrix(X_real))
    data_df['score'] = predictions

    classifier_threshold = st.slider("Classifier Probability Threshold (Cut)", 0.50, 0.99, 0.90, 0.01)

    filtered_data = data_df[data_df['score'] > classifier_threshold]
    
    st.write(f"Number of events passing the cut: **{len(filtered_data)}**")

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(80, 180, 40)

    ax.hist(filtered_data['M_4l'], bins=bins, color='mediumseagreen', label='Real Data passing AI cut')
    
    ax.set_xlabel("Four-Lepton Invariant Mass, $m_{4\ell}$ [GeV]", fontsize=12)
    ax.set_ylabel("Number of Events", fontsize=12)
    ax.set_title(f"Invariant Mass of Real Data after Classifier Cut > {classifier_threshold:.2f}", fontsize=14)
    ax.legend()
    st.pyplot(fig)

elif page == "Documentation":
    st.title("4.Project Documentation")
    st.markdown("""
    <div align="center">
    
    *This project was undertaken as part of **Project Polaris**, an initiative dedicated to developing high-quality, impactful projects to foster advanced skills in science and technology.*
    
    </div>
    """, unsafe_allow_html=True)

    st.header("Abstract")
    st.markdown("""
    This project presents a comprehensive, end-to-end data analysis pipeline demonstrating the application of modern machine learning techniques to a fundamental challenge in experimental particle physics: the discovery of the Higgs boson. Using public 13 TeV proton-proton collision data from the ATLAS experiment at CERN, this analysis focuses on the $H \\rightarrow ZZ^* \\rightarrow 4\\ell$ "golden channel." The central challenge is the statistical separation of this rare signal process from copious Standard Model background processes. We leverage high-fidelity Monte Carlo simulations to train and compare two advanced classifiers—a Gradient Boosted Decision Tree (XGBoost) and a Deep Neural Network (DNN)—to learn the subtle kinematic differences between signal and background events. The output of these models is a continuous discriminant score, which allows us to define signal-enriched regions of phase space, thereby maximizing the statistical significance of the final measurement and demonstrating a powerful method for particle discovery in large-scale physics experiments.
    """)

    st.header("1. Project Introduction & Motivation")
    st.subheader("1.1. The Standard Model and the Higgs Boson")
    st.markdown("""
    The Standard Model of particle physics is our most successful theory describing the fundamental constituents of matter and their interactions. A cornerstone of this model is the Higgs boson, the particle associated with the Higgs field, which is responsible for giving fundamental particles like W and Z bosons their mass. Its discovery at the Large Hadron Collider (LHC) in 2012 was a landmark achievement, confirming the final missing piece of the theory.
    """)
    st.subheader("1.2. The Analytical Challenge: Signal vs. Background")
    st.markdown("""
    The central challenge in experimental particle physics is the statistical separation of a rare signal process from copious background processes. The production cross-section for the Higgs boson is orders of magnitude smaller than that of many Standard Model backgrounds. We search for the Higgs via its decay products, focusing on the $H \\rightarrow ZZ^* \\rightarrow 4\\ell$ channel (where $\\ell$ is an electron or muon). This is known as the "golden channel" due to its clean experimental signature and the high precision with which the four-lepton invariant mass, $m_{4\ell}$, can be reconstructed.

    The primary "irreducible" background is the direct production of a $ZZ$ pair, which can decay to the same four-lepton final state. While the final state particles are identical, the kinematics of the event—the momenta and angular distributions of the leptons—differ subtly due to the different spin and production mechanisms of the parent particle.

    This analysis leverages machine learning to exploit these kinematic differences. We train advanced classifiers (XGBoost and a Deep Neural Network) on high-fidelity Monte Carlo simulations to learn the multi-dimensional decision boundary that optimally separates signal from background. The output of these models is a continuous discriminant score, which allows us to define signal-enriched regions of phase space, thereby maximizing the statistical significance of the final measurement.
    """)

    st.header("2. Methodology")
    st.markdown("""
    This project follows a systematic workflow common in high-energy physics analyses:

    1.  **Data Sourcing & Preprocessing:**
        * Utilized public 13 TeV collision data from the ATLAS Open Data portal.
        * Developed a Python-based pipeline using `uproot` and `awkward` to read complex ROOT files.
        * Cleaned the data by applying physics-based selections, such as requiring exactly four leptons with a net charge of zero.
        * Engineered new features, most notably the four-lepton invariant mass ($M_{4l}$), from the raw lepton four-momenta using the `vector` library.

    2.  **Exploratory Data Analysis (EDA):**
        * Visualized the kinematic distributions of key variables for simulated signal and background events.
        * Confirmed expected physical differences, such as the 125 GeV mass peak for the signal and harder momentum spectra for background processes.

    3.  **Model Training & Comparison:**
        * Trained two independent, state-of-the-art classifiers:
            * **XGBoost:** A powerful gradient-boosted decision tree algorithm.
            * **Deep Neural Network (DNN):** A multi-layered network built with Keras/TensorFlow.
        * The models were trained on the processed Monte Carlo data, using event weights to correct for simulation inaccuracies.

    4.  **Quantitative Evaluation:**
        * Assessed model performance on a held-out test set using the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) metric.
        * This provides an unbiased measure of the models' ability to distinguish signal from background.

    5.  **Qualitative Evaluation & Discovery:**
        * Applied the best-performing model (XGBoost) to the real, unlabeled collision data.
        * Used the model's output score to filter the real data, creating a signal-enriched sample.
        * Plotted the invariant mass of this purified sample to search for a statistically significant excess of events around 125 GeV—the signature of the Higgs boson.
    """)

    st.header("3. The Interactive Application")
    st.markdown("""
    To effectively communicate the results of this complex analysis, this web application was developed using Streamlit. The application serves as an interactive "showroom" for the project, allowing users to:
    * Learn about the physics context of the Higgs boson search.
    * Explore the kinematic data distributions interactively.
    * Compare the performance of the trained AI models.
    * Engage with the final "Discovery Plot," using a slider to apply the AI model's cut to the real data and see the Higgs signal emerge from the background noise.

    This interactive component transforms the project from a static analysis into a dynamic and engaging scientific narrative.
    """)

    st.markdown("---")
    st.image("polaris_main.png", caption="Parthiv Dasgupta aka ENIGMA")
    st.markdown("""
    <div align="center">
    
    </div>
    """, unsafe_allow_html=True)
