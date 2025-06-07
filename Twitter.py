import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍Sentiment Analysis Tool")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
mode = st.sidebar.selectbox("Choose Mode", ["Train New Model", "Predict Sentiment"])

def preprocess_text(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def train_model():
    st.header("📊 Train New Model")
    
    # Model selection
    st.subheader("Model Configuration")
    model_type = st.selectbox(
        "Choose Model Type:", 
        ["Logistic Regression", "Random Forest", "Support Vector Machine"]
    )
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file for training", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Display data info
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dataset Shape:**", df.shape)
                st.write("**Columns:**", list(df.columns))
            
            with col2:
                st.write("**Data Preview:**")
                st.dataframe(df.head())
            
            # Column selection
            st.subheader("Select Columns")
            text_column = st.selectbox("Select text column:", df.columns)
            label_column = st.selectbox("Select label/category column:", df.columns)
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                max_features = st.slider("Max TF-IDF Features", 1000, 10000, 5000)
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
                random_state = st.number_input("Random State", value=42)
            
            if st.button("Start Training"):
                with st.spinner("Training model..."):
                    # Data preprocessing
                    X = df[text_column].apply(preprocess_text)
                    y = df[label_column]
                    
                    # Remove empty texts
                    mask = X.str.len() > 0
                    X = X[mask]
                    y = y[mask]
                    
                    # Encode target variable
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)
                    
                    # TF-IDF Vectorization
                    vectorizer = TfidfVectorizer(
                        max_features=max_features, 
                        stop_words='english',
                        ngram_range=(1, 2)
                    )
                    X_vectorized = vectorizer.fit_transform(X)
                    
                    # Train-Test Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_vectorized, y_encoded, 
                        test_size=test_size, 
                        random_state=random_state,
                        stratify=y_encoded
                    )
                    
                    # Model Selection and Training
                    if model_type == "Logistic Regression":
                        model = LogisticRegression(random_state=random_state, max_iter=1000)
                    elif model_type == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=100, 
                            random_state=random_state,
                            n_jobs=-1
                        )
                    else:  # SVM
                        model = SVC(
                            kernel='linear', 
                            random_state=random_state,
                            probability=True
                        )
                    
                    # Training with progress bar
                    progress_bar = st.progress(0)
                    model.fit(X_train, y_train)
                    progress_bar.progress(100)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Display results
                    st.success("Training completed!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test Accuracy", f"{accuracy:.3f}")
                    with col2:
                        st.metric("Model Type", model_type)
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    class_names = label_encoder.classes_
                    report = classification_report(
                        y_test, y_pred, 
                        target_names=class_names,
                        output_dict=True
                    )
                    
                    # Display metrics table
                    metrics_df = pd.DataFrame(report).transpose()
                    st.dataframe(metrics_df.round(3))
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names,
                        ax=ax
                    )
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    
                    # Save model and components
                    joblib.dump(model, "sentiment_model.pkl")
                    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
                    joblib.dump(label_encoder, "label_encoder.pkl")
                    
                    # Save additional info
                    model_info = {
                        'model_type': model_type,
                        'accuracy': accuracy,
                        'class_names': class_names.tolist(),
                        'feature_count': max_features
                    }
                    with open("model_info.pkl", "wb") as f:
                        pickle.dump(model_info, f)
                    
                    st.success("Model and components saved successfully!")
                    
                    # Display class distribution
                    st.subheader("Class Distribution")
                    class_counts = pd.Series(y).value_counts()
                    st.bar_chart(class_counts)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV file to start training.")

def predict_sentiment():
    st.header("🎯 Predict Sentiment")
    
    # Check if model files exist
    model_files = ["sentiment_model.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl"]
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"Missing model files: {', '.join(missing_files)}")
        st.info("Please train a model first using the 'Train New Model' mode.")
        return
    
    try:
        # Load model and components
        with st.spinner("Loading model..."):
            model = joblib.load("sentiment_model.pkl")
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            label_encoder = joblib.load("label_encoder.pkl")
            
            # Load model info if available
            try:
                with open("model_info.pkl", "rb") as f:
                    model_info = pickle.load(f)
            except:
                model_info = {
                    'model_type': 'Unknown',
                    'accuracy': 'Unknown',
                    'class_names': label_encoder.classes_.tolist()
                }
        
        st.success("Model loaded successfully!")
        
        # Display model info
        with st.expander("Model Information"):
            st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
            st.write(f"**Accuracy:** {model_info.get('accuracy', 'Unknown')}")
            st.write(f"**Number of classes:** {len(model_info['class_names'])}")
            st.write(f"**Classes:** {', '.join(model_info['class_names'])}")
            st.write(f"**Vocabulary size:** {len(vectorizer.vocabulary_)}")
        
        # Prediction interface
        st.subheader("Enter Text for Prediction")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Single Text", "Batch Text"])
        
        if input_method == "Single Text":
            user_input = st.text_area("Enter your text:", height=150)
            
            if st.button("Predict Sentiment"):
                if user_input.strip():
                    # Preprocess and predict
                    processed_text = preprocess_text(user_input)
                    text_vectorized = vectorizer.transform([processed_text])
                    
                    # Get prediction
                    prediction = model.predict(text_vectorized)[0]
                    probabilities = model.predict_proba(text_vectorized)[0]
                    
                    predicted_class = label_encoder.inverse_transform([prediction])[0]
                    confidence = np.max(probabilities)
                    
                    # Display results
                    st.subheader("Prediction Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Sentiment", predicted_class)
                    with col2:
                        st.metric("Confidence", f"{confidence:.3f}")
                    
                    # Show all class probabilities
                    st.subheader("All Class Probabilities")
                    class_names = model_info['class_names']
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    st.dataframe(prob_df, use_container_width=True)
                    
                    # Visualization
                    st.bar_chart(prob_df.set_index('Class')['Probability'])
                    
                else:
                    st.warning("Please enter some text to predict.")
        
        else:  # Batch Text
            st.write("Upload a CSV file with text data for batch prediction:")
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("**Data Preview:**")
                    st.dataframe(df.head())
                    
                    text_column = st.selectbox("Select text column:", df.columns)
                    
                    if st.button("Predict Batch"):
                        with st.spinner("Making predictions..."):
                            # Preprocess texts
                            texts = df[text_column].apply(preprocess_text)
                            
                            # Vectorize
                            texts_vectorized = vectorizer.transform(texts)
                            
                            # Predict
                            predictions = model.predict(texts_vectorized)
                            probabilities = model.predict_proba(texts_vectorized)
                            
                            predicted_classes = label_encoder.inverse_transform(predictions)
                            confidences = np.max(probabilities, axis=1)
                            
                            # Create results dataframe
                            results_df = df.copy()
                            results_df['Predicted_Sentiment'] = predicted_classes
                            results_df['Confidence'] = confidences
                            
                            st.success("Batch prediction completed!")
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="sentiment_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Show summary
                            st.subheader("Prediction Summary")
                            summary = pd.Series(predicted_classes).value_counts()
                            st.bar_chart(summary)
                            
                            # Show confidence distribution
                            st.subheader("Confidence Distribution")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.hist(confidences, bins=20, alpha=0.7, color='skyblue')
                            ax.set_xlabel('Confidence Score')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Prediction Confidence')
                            st.pyplot(fig)
                            
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

# Main app logic
if mode == "Train New Model":
    train_model()
else:
    predict_sentiment()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Suraj Patil")
