# app.py
# (Keep all the previous imports and functions:
# COL_NAMES, CATEGORICAL_COLS, load_data, load_model_and_preprocessor,
# create_binary_labels, get_preprocessor, preprocess_data_with_preprocessor)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- Configuration & Constants ---
MODEL_PATH = "nsl_kdd_rf_model.joblib"
PREPROCESSOR_PATH = "nsl_kdd_preprocessor.joblib"
TRAIN_DATA_PATH = "KDDTrain+.TXT" # Define path for training data
TEST_DATA_PATH = "KDDTest+.TXT"   # Define path for default test data

COL_NAMES = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
             "num_failed_logins", "logged_in", "num_compromised", "root_shell",
             "su_attempted", "num_root", "num_file_creations", "num_shells",
             "num_access_files", "num_outbound_cmds", "is_host_login",
             "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
             "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
             "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
             "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
             "dst_host_srv_serror_rate", "dst_host_rerror_rate",
             "dst_host_srv_rerror_rate", "attack_type", "difficulty_score"]

# Identify categorical and numerical feature names (excluding target and score)
FEATURE_COL_NAMES = [col for col in COL_NAMES if col not in ['attack_type', 'difficulty_score']]
CATEGORICAL_FEATURE_COLS = ['protocol_type', 'service', 'flag']
NUMERICAL_FEATURE_COLS = [col for col in FEATURE_COL_NAMES if col not in CATEGORICAL_FEATURE_COLS]

# --- Caching Functions for Performance ---
@st.cache_data
def load_data(file_path, col_names_list):
    try:
        df = pd.read_csv(file_path, header=None, names=col_names_list)
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Please make sure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        return None

@st.cache_resource
def load_model_and_preprocessor():
    model = None
    preprocessor = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            model = None # Ensure model is None if loading fails
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
        except Exception as e:
            st.error(f"Error loading preprocessor: {e}")
            preprocessor = None # Ensure preprocessor is None
    return model, preprocessor

@st.cache_data
def get_categorical_options(train_data_path, categorical_cols_list):
    """Loads training data to extract unique values for categorical features."""
    options = {}
    train_df = load_data(train_data_path, COL_NAMES)
    if train_df is not None:
        for col in categorical_cols_list:
            if col in train_df.columns:
                options[col] = sorted(train_df[col].unique().tolist())
            else:
                options[col] = [] # Default to empty list if column not found
    else: # Fallback if training data can't be loaded
        st.warning("Could not load training data to get categorical options. Using common defaults.")
        options['protocol_type'] = ['tcp', 'udp', 'icmp']
        options['service'] = ['http', 'ftp_data', 'ecr_i', 'smtp', 'private', 'domain_u', 'other'] # Common ones
        options['flag'] = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'S2', 'S3', 'OTH']
    return options


# --- Core ML Functions (create_binary_labels, get_preprocessor, preprocess_data_with_preprocessor) ---
# These functions remain largely the same as before.
# Ensure get_preprocessor uses CATEGORICAL_FEATURE_COLS and NUMERICAL_FEATURE_COLS

def create_binary_labels(df):
    df_copy = df.copy()
    if 'attack_type' not in df_copy.columns:
        st.error("Column 'attack_type' not found in DataFrame for creating labels.")
        # Potentially return df_copy or handle error, e.g., by adding a dummy label column if appropriate for the flow
        return df_copy # Or df_copy with a new dummy 'label' column
    df_copy['label'] = df_copy['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
    df_copy = df_copy.drop(columns=['attack_type', 'difficulty_score'], errors='ignore')
    return df_copy

def get_preprocessor(df_for_fitting_preprocessor):
    preprocessor_obj = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURE_COLS),
            ('scaler', StandardScaler(), NUMERICAL_FEATURE_COLS)
        ],
        remainder='passthrough' # should be 'drop' or specify all columns
    )
    # Ensure only feature columns are passed for fitting
    feature_df = df_for_fitting_preprocessor[FEATURE_COL_NAMES]
    preprocessor_obj.fit(feature_df)
    return preprocessor_obj

def preprocess_data_with_preprocessor(df_features_only, preprocessor_to_use):
    processed_data = preprocessor_to_use.transform(df_features_only)
    try:
        feature_names_out = preprocessor_to_use.get_feature_names_out()
        processed_df = pd.DataFrame(processed_data, columns=feature_names_out)
    except Exception:
        processed_df = pd.DataFrame(processed_data)
        st.warning("Could not retrieve feature names from preprocessor. Columns will be numbered.")
    return processed_df


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="NSL-KDD IDS Demo")
st.title("🛡️ Intrusion Detection System using NSL-KDD Dataset")
st.markdown("""
This application demonstrates a Machine Learning-based Intrusion Detection System (IDS).
- **Train/Load Model:** Use the sidebar to train a new model or load a pre-trained one.
- **Test on File:** Evaluate the model on a test dataset (`KDDTest+.TXT` or uploaded file).
- **Live Prediction:** Input individual network connection features to get a live classification.
""")

# --- Sidebar (same as before) ---
st.sidebar.header("⚙️ Controls")
model_action = st.sidebar.radio("Model Action:", ("Load Pre-trained Model", "Train New Model"))

model = None
preprocessor = None
categorical_options = get_categorical_options(TRAIN_DATA_PATH, CATEGORICAL_FEATURE_COLS)


if 'model_ready' not in st.session_state: # Use a single flag
    st.session_state.model_ready = False

if model_action == "Load Pre-trained Model":
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        model, preprocessor = load_model_and_preprocessor()
        if model is not None and preprocessor is not None:
            st.sidebar.success("Pre-trained model and preprocessor loaded!")
            st.session_state.model_ready = True
        else:
            st.sidebar.error("Failed to load model or preprocessor. Files might be corrupted or not found.")
            st.session_state.model_ready = False # Explicitly set to false
    else:
        st.sidebar.warning("No pre-trained model/preprocessor found. Please train a new model.")
        st.session_state.model_ready = False

elif model_action == "Train New Model":
    if st.sidebar.button("Start Training Model"):
        with st.spinner("Training new model... This might take a few minutes."):
            st.subheader("Model Training")
            train_df_raw = load_data(TRAIN_DATA_PATH, COL_NAMES)
            if train_df_raw is not None:
                st.write("Loaded `KDDTrain+.TXT` successfully.")
                train_df_labeled = create_binary_labels(train_df_raw)
                y_train = train_df_labeled['label']
                X_train_raw_features = train_df_labeled.drop(columns=['label'])

                st.write("Fitting preprocessor on training data...")
                preprocessor = get_preprocessor(X_train_raw_features) # Pass only features
                joblib.dump(preprocessor, PREPROCESSOR_PATH)
                st.write("Preprocessor fitted and saved.")

                X_train_processed = preprocess_data_with_preprocessor(X_train_raw_features, preprocessor)
                st.write(f"Processed training data shape: {X_train_processed.shape}")

                st.write("Training Random Forest model...")
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
                model.fit(X_train_processed, y_train)
                joblib.dump(model, MODEL_PATH)
                st.session_state.model_ready = True
                st.success("Model trained and saved successfully!")
                st.balloons()
            else:
                st.error("Could not load training data. Aborting training.")
                st.session_state.model_ready = False


# --- Main Area Tabs for Organization ---
tab1, tab2 = st.tabs(["📊 Test on File", "🔮 Live Prediction"])

with tab1:
    st.header("🧪 Test Model and Evaluate Performance on a File")
    if not st.session_state.get('model_ready', False) or model is None or preprocessor is None:
        st.warning("Model is not ready. Please load or train a model first using the sidebar controls.")
    else:
        test_data_option = st.radio("Select Test Data Source:",
                                    ("Use KDDTest+.TXT", "Upload Custom Test File (.txt or .csv)"),
                                    key="file_test_source") # Unique key for radio

        test_df_raw = None
        if test_data_option == "Use KDDTest+.TXT":
            test_df_raw = load_data(TEST_DATA_PATH, COL_NAMES)
            if test_df_raw is not None: st.write("Loaded `KDDTest+.TXT` for testing.")
        else:
            uploaded_file = st.file_uploader("Upload your test file (NSL-KDD format, 43 cols)", type=["txt", "csv"], key="file_uploader")
            if uploaded_file is not None:
                test_df_raw = load_data(uploaded_file, COL_NAMES)
                if test_df_raw is not None: st.write("Custom test file uploaded and loaded.")

        if test_df_raw is not None:
            st.subheader("Test Data Overview (First 5 Rows)")
            st.dataframe(test_df_raw.head())

            with st.spinner("Preprocessing test data and making predictions..."):
                test_df_labeled = create_binary_labels(test_df_raw)
                if 'label' not in test_df_labeled.columns:
                    st.error("Label column missing after creating binary labels. Check input data.")
                else:
                    y_test = test_df_labeled['label']
                    X_test_raw_features = test_df_labeled.drop(columns=['label'])
                    X_test_processed = preprocess_data_with_preprocessor(X_test_raw_features, preprocessor)

                    y_pred = model.predict(X_test_processed)
                    st.subheader("📊 Evaluation Metrics")
                    # ... (rest of evaluation metrics display from previous version) ...
                    col1, col2, col3, col4 = st.columns(4)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    col1.metric("Accuracy", f"{accuracy:.4f}")
                    col2.metric("Precision (Attack)", f"{precision:.4f}")
                    col3.metric("Recall (Attack)", f"{recall:.4f}")
                    col4.metric("F1-Score (Attack)", f"{f1:.4f}")

                    st.subheader("Classification Report")
                    report_str = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)'], zero_division=0, output_dict=False)
                    st.text_area("Report", report_str, height=150)

                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(6,4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                xticklabels=['Predicted Normal', 'Predicted Attack'],
                                yticklabels=['Actual Normal', 'Actual Attack'])
                    ax_cm.set_ylabel('Actual Label')
                    ax_cm.set_xlabel('Predicted Label')
                    st.pyplot(fig_cm)
                    plt.close(fig_cm) # Close the figure to free memory

                    tn, fp, fn, tp = cm.ravel()
                    st.markdown(f"""
                    - **True Negatives (Normal):** {tn}
                    - **False Positives (Normal predicted as Attack):** {fp} *(False Alarms)*
                    - **False Negatives (Attack predicted as Normal):** {fn} *(Missed Detections!)*
                    - **True Positives (Attack):**  {tp}
                    """)

        elif test_data_option == "Upload Custom Test File (.txt or .csv)" and uploaded_file is None:
            st.info("Upload a test file to see predictions and evaluation.")
        elif test_data_option == "Use KDDTest+.TXT" and test_df_raw is None:
            st.error(f"Could not load {TEST_DATA_PATH}.")


with tab2:
    st.header("🔮 Live Prediction for a Single Connection")

    if not st.session_state.get('model_ready', False) or model is None or preprocessor is None:
        st.warning("Model is not ready. Please load or train a model first using the sidebar controls.")
    else:
        st.markdown("Enter the 41 features of a network connection below:")

        # Create input fields for each feature
        input_data = {}
        cols = st.columns(3) # Display inputs in 3 columns for better layout

        # FEATURE_COL_NAMES is defined at the top: all 41 feature names
        for i, feature_name in enumerate(FEATURE_COL_NAMES):
            current_col = cols[i % 3]
            if feature_name in CATEGORICAL_FEATURE_COLS:
                # Use options derived from training data if available
                options_list = categorical_options.get(feature_name, [])
                if not options_list: # Fallback if options somehow empty
                    options_list = ["N/A"]
                input_data[feature_name] = current_col.selectbox(
                    f"{feature_name}",
                    options=options_list,
                    key=f"live_{feature_name}" # Unique key for each widget
                )
            elif feature_name in NUMERICAL_FEATURE_COLS:
                # For numerical, try to get min/max/mean from training data if we want defaults
                # For simplicity, using 0.0 as default now.
                input_data[feature_name] = current_col.number_input(
                    f"{feature_name}",
                    value=0.0,  # Default value
                    step=0.01 if "." in str(0.0) else 1.0, # Auto step based on type
                    format="%.2f" if "." in str(0.0) else "%d",
                    key=f"live_{feature_name}"
                )
            else: # Should not happen if lists are correct
                current_col.text(f"Unknown feature type: {feature_name}")


        if st.button("🚀 Predict Connection Type", key="live_predict_button"):
            # Create a DataFrame from the input data
            single_instance_df = pd.DataFrame([input_data], columns=FEATURE_COL_NAMES)

            st.subheader("Input Features:")
            st.dataframe(single_instance_df)

            # Preprocess the single instance using the FITTED preprocessor
            try:
                single_instance_processed = preprocess_data_with_preprocessor(single_instance_df, preprocessor)

                # Make prediction
                prediction = model.predict(single_instance_processed)
                prediction_proba = model.predict_proba(single_instance_processed)

                st.subheader("📈 Prediction Result:")
                if prediction[0] == 1:
                    st.error("🚨 Prediction: ATTACK 🚨")
                else:
                    st.success("✅ Prediction: NORMAL ✅")

                st.write("Prediction Probabilities:")
                # Assuming binary classification: class 0 (normal), class 1 (attack)
                # The order of classes in `model.classes_` determines this
                proba_df = pd.DataFrame(prediction_proba, columns=[f"Prob(Normal)", f"Prob(Attack)"])
                st.dataframe(proba_df)

            except Exception as e:
                st.error(f"Error during live prediction: {e}")
                st.error("Ensure the preprocessor was fitted correctly and all feature types are handled.")

st.sidebar.markdown("---")
st.sidebar.info("IDS Demo by [Your Name/Project Name]")