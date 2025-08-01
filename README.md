<div align="center">

# Project Polaris: AI-Driven Search for the Higgs Boson

**Author: Parthiv Dasgupta**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)

</div>

---

### **1. Project Overview**

This project presents a complete, end-to-end data analysis pipeline demonstrating the application of modern machine learning techniques to a fundamental challenge in experimental particle physics: the discovery of the Higgs boson. Using public 13 TeV proton-proton collision data from the ATLAS experiment at CERN, this analysis focuses on the $H \rightarrow ZZ^* \rightarrow 4\ell$ "golden channel."

The core challenge is to distinguish the incredibly rare Higgs boson **signal** from a vast sea of more common **background** processes that produce an identical experimental signature. To solve this, we leverage high-fidelity Monte Carlo simulations to train and compare two state-of-the-art classifiers—a Gradient Boosted Decision Tree (XGBoost) and a Deep Neural Network (DNN). These models learn the subtle kinematic differences between signal and background events, acting as a highly sophisticated filter.

The entire analysis is presented in an interactive web application built with Streamlit, which allows users to explore the data, compare model performance, and engage with the final "discovery plot" to see how the AI helps the Higgs signal emerge from the noise.

### **2. The Scientific Challenge**

The production of a Higgs boson is a rare event, and it decays almost instantly. We can only identify it by its decay products. The "golden channel" ($H \rightarrow 4\ell$) is exceptionally clean but is mimicked by a primary background from direct $ZZ$ pair production. While the final particles are the same, the underlying physics of their production imparts subtle differences in their energy and angular distributions. This project's goal is to train an AI to learn this multi-dimensional boundary between signal and background, allowing us to enhance the statistical significance of a potential discovery.

### **3. The Interactive Application**

The project culminates in a multi-page Streamlit web application that serves as an interactive portal to the analysis.

* **Project Introduction & Documentation:** Provides the scientific context and a detailed summary of the project's methodology and goals.
* **Exploratory Data Analysis (EDA):** An interactive page to visualize the kinematic distributions of the particles, showing the inherent differences between simulated signal and background events.
* **AI Model Performance:** A side-by-side comparison of the XGBoost and DNN models, featuring their ROC curves and AUC scores to quantitatively assess their performance.
* **The Discovery Plot:** The final, interactive result. This page applies the trained XGBoost model to the real collision data. A user can move a slider to apply a stricter cut on the model's output score, filtering out background events to see the Higgs boson peak emerge around 125 GeV.

### **4. Technical Stack**

* **Language:** Python
* **Data Processing:** `uproot`, `awkward`, `pandas`, `numpy`, `vector`
* **Machine Learning:** `scikit-learn`, `xgboost`, `tensorflow (Keras)`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Web Application:** `streamlit`
* **Model & Data Persistence:** `joblib`, `pickle`

### **5. Project Structure**

The project is organized into two main scripts:

1.  **`train_and_save.py` (The Workshop):** This script is the offline data processing and training engine. It reads the raw `.root` files, performs feature engineering, trains both the XGBoost and DNN models, and saves all the final assets (`.pkl`, `.joblib`, `.keras` files) to disk. **This script is run only once.**

2.  **`app.py` (The Showroom):** This is the Streamlit web application. It is a lightweight script that loads the pre-processed data and pre-trained models to create the interactive user experience. It does not perform any training itself.

### **6. How to Run This Project Locally**

To run this application on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Download the Data:**
    * Download the 13 TeV ATLAS Open Data 4-lepton dataset from the [CERN Open Data Portal](http://opendata.cern.ch/record/15005).
    * Unzip the files and place them in a folder structure like: `your-repo-name/dataset/4lep 2/MC/` and `your-repo-name/dataset/4lep 2/Data/`.

3.  **Install Dependencies:**
    * It is highly recommended to create a virtual environment.
    * Install all required libraries using the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Run the Training Script:**
    * Execute the script to process the data and train the models. This will create the necessary asset files.
        ```bash
        python train_and_save.py
        ```

5.  **Launch the Streamlit App:**
    * Once the training script is complete, run the Streamlit application:
        ```bash
        streamlit run app.py
        ```
    * Your web browser will automatically open with the application running.

### **7. Results & Key Learnings**

* **Model Performance:** Both models demonstrated high performance, with the XGBoost classifier achieving a slightly superior **AUC of ~0.96** and the DNN achieving an **AUC of ~0.94**.
* **Feature Importance:** The four-lepton invariant mass ($M_{4l}$) was confirmed to be the most powerful discriminating variable, but other kinematic features related to lepton momentum and energy also contributed significantly to the models' performance.
* **Discovery Potential:** The interactive "Discovery Plot" successfully demonstrates the core principle of a physics analysis. By applying a stringent cut based on the AI model's output, the signal-to-background ratio is dramatically improved, revealing a clear excess of events in the real data around the 125 GeV mass region, consistent with the Higgs boson.

This project was a deep dive into the practical application of data science in a fundamental scientific domain, highlighting the critical interplay between physics knowledge, data processing, and advanced machine learning.

---
<div align="center">

*This project was undertaken as part of **Project Polaris***

</div>
