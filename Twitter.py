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

# Deep Learning imports (with error handling)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, GRU, Embedding, Dropout, GlobalMaxPooling1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    st.warning("TensorFlow is not available. Deep Learning models will be disabled.")
    TENSORFLOW_AVAILABLE = False
except Exception as e:
    st.error(f"TensorFlow import error: {str(e)}")
    st.warning("Deep Learning models will be disabled due to TensorFlow compatibility issues.")
    TENSORFLOW_AVAILABLE = False

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

def create_ann_model(vocab_size, max_length, num_classes):
    """Create Artificial Neural Network model"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_lstm_model(vocab_size, max_length, num_classes):
    """Create LSTM model"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_gru_model(vocab_size, max_length, num_classes):
    """Create GRU model"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        GRU(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    st.header("📊 Train New Model")
    
    # Model selection
    st.subheader("Model Configuration")
    
    # Available models based on TensorFlow availability
    if TENSORFLOW_AVAILABLE:
        available_models = ["Logistic Regression", "Random Forest", "Support Vector Machine", 
                          "Artificial Neural Network (ANN)", "LSTM", "GRU"]
    else:
        available_models = ["Logistic Regression", "Random Forest", "Support Vector Machine"]
        st.info("📌 Deep Learning models (ANN, LSTM, GRU) are not available due to TensorFlow compatibility issues. Traditional ML models are still available.")
    
    model_type = st.selectbox("Choose Model Type:", available_models)
    
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
                if TENSORFLOW_AVAILABLE and model_type in ["Artificial Neural Network (ANN)", "LSTM", "GRU"]:
                    max_features = st.slider("Max Vocabulary Size", 1000, 20000, 10000)
                    max_length = st.slider("Max Sequence Length", 50, 500, 200)
                    epochs = st.slider("Training Epochs", 5, 50, 20)
                    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
                else:
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
                    num_classes = len(label_encoder.classes_)
                    
                    if TENSORFLOW_AVAILABLE and model_type in ["Artificial Neural Network (ANN)", "LSTM", "GRU"]:
                        # Deep Learning Models
                        st.info("Training Deep Learning Model...")
                        
                        # Tokenization for deep learning
                        tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
                        tokenizer.fit_on_texts(X)
                        X_sequences = tokenizer.texts_to_sequences(X)
                        X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')
                        
                        # Convert labels to categorical
                        y_categorical = to_categorical(y_encoded, num_classes=num_classes)
                        
                        # Train-Test Split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_padded, y_categorical, 
                            test_size=test_size, 
                            random_state=random_state,
                            stratify=y_encoded
                        )
                        
                        # Create model based on selection
                        vocab_size = len(tokenizer.word_index) + 1
                        
                        if model_type == "Artificial Neural Network (ANN)":
                            model = create_ann_model(vocab_size, max_length, num_classes)
                        elif model_type == "LSTM":
                            model = create_lstm_model(vocab_size, max_length, num_classes)
                        else:  # GRU
                            model = create_gru_model(vocab_size, max_length, num_classes)
                        
                        # Early stopping callback
                        early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            patience=5, 
                            restore_best_weights=True
                        )
                        
                        # Training with progress updates
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        history = model.fit(
                            X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("Training completed!")
                        
                        # Evaluate
                        y_pred_proba = model.predict(X_test)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        y_test_labels = np.argmax(y_test, axis=1)
                        accuracy = accuracy_score(y_test_labels, y_pred)
                        
                        # Save deep learning components
                        model.save("sentiment_dl_model.h5")
                        joblib.dump(tokenizer, "tokenizer.pkl")
                        joblib.dump(label_encoder, "label_encoder.pkl")
                        
                        # Plot training history
                        st.subheader("Training History")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.plot(history.history['accuracy'], label='Training Accuracy')
                            ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
                            ax.set_title('Model Accuracy')
                            ax.set_xlabel('Epoch')
                            ax.set_ylabel('Accuracy')
                            ax.legend()
                            st.pyplot(fig)
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.plot(history.history['loss'], label='Training Loss')
                            ax.plot(history.history['val_loss'], label='Validation Loss')
                            ax.set_title('Model Loss')
                            ax.set_xlabel('Epoch')
                            ax.set_ylabel('Loss')
                            ax.legend()
                            st.pyplot(fig)
                        
                        # Save additional info for deep learning
                        model_info = {
                            'model_type': model_type,
                            'accuracy': accuracy,
                            'class_names': label_encoder.classes_.tolist(),
                            'vocab_size': vocab_size,
                            'max_length': max_length,
                            'is_deep_learning': True
                        }
                        with open("model_info.pkl", "wb") as f:
                            pickle.dump(model_info, f)
                        
                    else:
                        # Traditional ML Models
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
                        
                        # Save traditional ML components
                        joblib.dump(model, "sentiment_model.pkl")
                        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
                        joblib.dump(label_encoder, "label_encoder.pkl")
                        
                        # Save additional info for traditional ML
                        model_info = {
                            'model_type': model_type,
                            'accuracy': accuracy,
                            'class_names': label_encoder.classes_.tolist(),
                            'feature_count': max_features,
                            'is_deep_learning': False
                        }
                        with open("model_info.pkl", "wb") as f:
                            pickle.dump(model_info, f)
                    
                    # Display results (common for all models)
                    st.success("Training completed!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test Accuracy", f"{accuracy:.3f}")
                    with col2:
                        st.metric("Model Type", model_type)
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    class_names = label_encoder.classes_
                    
                    if model_type in ["Artificial Neural Network (ANN)", "LSTM", "GRU"]:
                        report = classification_report(
                            y_test_labels, y_pred, 
                            target_names=class_names,
                            output_dict=True
                        )
                        cm = confusion_matrix(y_test_labels, y_pred)
                    else:
                        report = classification_report(
                            y_test, y_pred, 
                            target_names=class_names,
                            output_dict=True
                        )
                        cm = confusion_matrix(y_test, y_pred)
                    
                    # Display metrics table
                    metrics_df = pd.DataFrame(report).transpose()
                    st.dataframe(metrics_df.round(3))
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
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
    
    try:
        # Load model info first to determine model type
        with open("model_info.pkl", "rb") as f:
            model_info = pickle.load(f)
        
        is_deep_learning = model_info.get('is_deep_learning', False)
        
        if is_deep_learning:
            # Check for deep learning model files
            required_files = ["sentiment_dl_model.h5", "tokenizer.pkl", "label_encoder.pkl"]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                st.error(f"Missing deep learning model files: {', '.join(missing_files)}")
                st.info("Please train a deep learning model first.")
                return
            
            # Load deep learning components
            with st.spinner("Loading deep learning model..."):
                model = tf.keras.models.load_model("sentiment_dl_model.h5")
                tokenizer = joblib.load("tokenizer.pkl")
                label_encoder = joblib.load("label_encoder.pkl")
                max_length = model_info.get('max_length', 200)
        else:
            # Check for traditional ML model files
            required_files = ["sentiment_model.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl"]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                st.error(f"Missing traditional ML model files: {', '.join(missing_files)}")
                st.info("Please train a traditional ML model first.")
                return
            
            # Load traditional ML components
            with st.spinner("Loading traditional ML model..."):
                model = joblib.load("sentiment_model.pkl")
                vectorizer = joblib.load("tfidf_vectorizer.pkl")
                label_encoder = joblib.load("label_encoder.pkl")
        
        st.success("Model loaded successfully!")
        
        # Display model info
        with st.expander("Model Information"):
            st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
            st.write(f"**Accuracy:** {model_info.get('accuracy', 'Unknown'):.3f}")
            st.write(f"**Number of classes:** {len(model_info['class_names'])}")
            st.write(f"**Classes:** {', '.join(model_info['class_names'])}")
            if is_deep_learning:
                st.write(f"**Vocabulary size:** {model_info.get('vocab_size', 'Unknown')}")
                st.write(f"**Max sequence length:** {model_info.get('max_length', 'Unknown')}")
            else:
                st.write(f"**Vocabulary size:** {len(vectorizer.vocabulary_)}")
        
        # Prediction interface
        st.subheader("Enter Text for Prediction")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Single Text", "Batch Text"])
        
        if input_method == "Single Text":
            user_input = st.text_area("Enter your text:", height=150)
            
            if st.button("Predict Sentiment"):
                if user_input.strip():
                    # Preprocess text
                    processed_text = preprocess_text(user_input)
                    
                    if is_deep_learning:
                        # Deep learning prediction
                        sequence = tokenizer.texts_to_sequences([processed_text])
                        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
                        
                        probabilities = model.predict(padded_sequence)[0]
                        prediction = np.argmax(probabilities)
                    else:
                        # Traditional ML prediction
                        text_vectorized = vectorizer.transform([processed_text])
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
                            
                            if is_deep_learning:
                                # Deep learning batch prediction
                                sequences = tokenizer.texts_to_sequences(texts)
                                padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
                                
                                probabilities = model.predict(padded_sequences)
                                predictions = np.argmax(probabilities, axis=1)
                                confidences = np.max(probabilities, axis=1)
                            else:
                                # Traditional ML batch prediction
                                texts_vectorized = vectorizer.transform(texts)
                                predictions = model.predict(texts_vectorized)
                                probabilities = model.predict_proba(texts_vectorized)
                                confidences = np.max(probabilities, axis=1)
                            
                            predicted_classes = label_encoder.inverse_transform(predictions)
                            
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
    
    except FileNotFoundError:
        st.error("No trained model found. Please train a model first.")
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
