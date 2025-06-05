import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import pickle

# Set page config
st.set_page_config(
    page_title="NLP Sentiment Analysis",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 NLP Sentiment Analysis Tool")
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
                    y_categorical = to_categorical(y_encoded)
                    
                    # TF-IDF Vectorization
                    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                    X_vectorized = vectorizer.fit_transform(X).toarray()
                    
                    # Train-Test Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_vectorized, y_categorical, test_size=0.2, random_state=42
                    )
                    
                    # Model Definition
                    model = Sequential([
                        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                        Dropout(0.5),
                        Dense(128, activation='relu'),
                        Dropout(0.3),
                        Dense(64, activation='relu'),
                        Dropout(0.2),
                        Dense(y_categorical.shape[1], activation='softmax')
                    ])
                    
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Training with progress bar
                    progress_bar = st.progress(0)
                    epochs = 10
                    
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.1,
                        verbose=0
                    )
                    
                    progress_bar.progress(100)
                    
                    # Evaluate
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    
                    # Display results
                    st.success("Training completed!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test Accuracy", f"{accuracy:.3f}")
                    with col2:
                        st.metric("Test Loss", f"{loss:.3f}")
                    
                    # Save model and components
                    model.save("text_classifier_model.h5")
                    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
                    joblib.dump(label_encoder, "label_encoder.pkl")
                    
                    # Save class names for reference
                    class_names = label_encoder.classes_
                    with open("class_names.pkl", "wb") as f:
                        pickle.dump(class_names, f)
                    
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
    model_files = ["text_classifier_model.h5", "tfidf_vectorizer.pkl", "label_encoder.pkl"]
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"Missing model files: {', '.join(missing_files)}")
        st.info("Please train a model first using the 'Train New Model' mode.")
        return
    
    try:
        # Load model and components
        with st.spinner("Loading model..."):
            model = load_model("text_classifier_model.h5")
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            label_encoder = joblib.load("label_encoder.pkl")
            
            # Load class names if available
            try:
                with open("class_names.pkl", "rb") as f:
                    class_names = pickle.load(f)
            except:
                class_names = label_encoder.classes_
        
        st.success("Model loaded successfully!")
        
        # Display model info
        with st.expander("Model Information"):
            st.write(f"**Number of classes:** {len(class_names)}")
            st.write(f"**Classes:** {', '.join(class_names)}")
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
                    text_vectorized = vectorizer.transform([processed_text]).toarray()
                    
                    # Get prediction
                    prediction = model.predict(text_vectorized, verbose=0)
                    predicted_class_idx = np.argmax(prediction[0])
                    predicted_class = class_names[predicted_class_idx]
                    confidence = prediction[0][predicted_class_idx]
                    
                    # Display results
                    st.subheader("Prediction Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Sentiment", predicted_class)
                    with col2:
                        st.metric("Confidence", f"{confidence:.3f}")
                    
                    # Show all class probabilities
                    st.subheader("All Class Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': prediction[0]
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
                            texts_vectorized = vectorizer.transform(texts).toarray()
                            
                            # Predict
                            predictions = model.predict(texts_vectorized, verbose=0)
                            predicted_classes = [class_names[np.argmax(pred)] for pred in predictions]
                            confidences = [np.max(pred) for pred in predictions]
                            
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
st.markdown("Built with ❤️ using Streamlit and TensorFlow")
