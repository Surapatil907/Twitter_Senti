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

# Deep Learning imports with error handling
TENSORFLOW_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Embedding, Bidirectional
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow not available. Deep learning models (ANN, LSTM, GRU) will be disabled.")
    st.info("To enable deep learning models, please install TensorFlow: `pip install tensorflow`")

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

def create_ann_model(input_dim, num_classes, hidden_units=128):
    """Create ANN model"""
    model = Sequential([
        Dense(hidden_units, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(hidden_units // 2, activation='relu'),
        Dropout(0.3),
        Dense(hidden_units // 4, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_lstm_model(vocab_size, embedding_dim, max_length, num_classes, lstm_units=64):
    """Create LSTM model"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_gru_model(vocab_size, embedding_dim, max_length, num_classes, gru_units=64):
    """Create GRU model"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(GRU(gru_units, dropout=0.5, recurrent_dropout=0.5)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    st.header("📊 Train New Model")
    
    # Model selection
    st.subheader("Model Configuration")
    
    # Available models based on TensorFlow availability
    available_models = ["Logistic Regression", "Random Forest", "Support Vector Machine"]
    if TENSORFLOW_AVAILABLE:
        available_models.extend(["Artificial Neural Network (ANN)", "LSTM Network", "GRU Network"])
    
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
            
            # Check if deep learning model is selected
            is_deep_learning = TENSORFLOW_AVAILABLE and model_type in ["Artificial Neural Network (ANN)", "LSTM Network", "GRU Network"]
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                if not is_deep_learning:
                    max_features = st.slider("Max TF-IDF Features", 1000, 10000, 5000)
                else:
                    vocab_size = st.slider("Vocabulary Size", 5000, 20000, 10000)
                    max_length = st.slider("Max Sequence Length", 50, 500, 100)
                    embedding_dim = st.slider("Embedding Dimension", 50, 300, 128)
                    if model_type == "Artificial Neural Network (ANN)":
                        hidden_units = st.slider("Hidden Units", 64, 512, 128)
                    else:
                        rnn_units = st.slider("RNN Units", 32, 128, 64)
                    epochs = st.slider("Training Epochs", 5, 50, 10)
                    batch_size = st.slider("Batch Size", 16, 128, 32)
                
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
                    
                    if not is_deep_learning:
                        # Traditional ML models
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
                        
                        # Save model and components
                        joblib.dump(model, "sentiment_model.pkl")
                        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
                        joblib.dump(label_encoder, "label_encoder.pkl")
                        
                    else:
                        # Deep Learning models
                        # Tokenization
                        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
                        tokenizer.fit_on_texts(X)
                        
                        # Convert texts to sequences
                        X_sequences = tokenizer.texts_to_sequences(X)
                        X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')
                        
                        # One-hot encode labels for multi-class
                        if num_classes > 2:
                            y_categorical = to_categorical(y_encoded, num_classes=num_classes)
                        else:
                            y_categorical = y_encoded
                        
                        # Train-Test Split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_padded, y_categorical, 
                            test_size=test_size, 
                            random_state=random_state,
                            stratify=y_encoded
                        )
                        
                        # Create model based on selection
                        if model_type == "Artificial Neural Network (ANN)":
                            # For ANN, we need to flatten the sequences or use different preprocessing
                            # Let's use TF-IDF for ANN but with dense layers
                            vectorizer = TfidfVectorizer(
                                max_features=vocab_size, 
                                stop_words='english',
                                ngram_range=(1, 2)
                            )
                            X_tfidf = vectorizer.fit_transform(X).toarray()
                            
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_tfidf, y_categorical, 
                                test_size=test_size, 
                                random_state=random_state,
                                stratify=y_encoded
                            )
                            
                            model = create_ann_model(X_tfidf.shape[1], num_classes, hidden_units)
                        elif model_type == "LSTM Network":
                            model = create_lstm_model(vocab_size, embedding_dim, max_length, num_classes, rnn_units)
                        else:  # GRU Network
                            model = create_gru_model(vocab_size, embedding_dim, max_length, num_classes, rnn_units)
                        
                        # Callbacks
                        callbacks = [
                            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
                        ]
                        
                        # Training with progress tracking
                        progress_bar = st.progress(0)
                        history = model.fit(
                            X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks,
                            verbose=0
                        )
                        progress_bar.progress(100)
                        
                        # Evaluate
                        if num_classes > 2:
                            y_pred_prob = model.predict(X_test)
                            y_pred = np.argmax(y_pred_prob, axis=1)
                            y_test_labels = np.argmax(y_test, axis=1)
                        else:
                            y_pred_prob = model.predict(X_test)
                            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                            y_test_labels = y_test
                        
                        accuracy = accuracy_score(y_test_labels, y_pred)
                        
                        # Save model and components
                        model.save("sentiment_model.h5")
                        if model_type == "Artificial Neural Network (ANN)":
                            joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
                        else:
                            with open("tokenizer.pkl", "wb") as f:
                                pickle.dump(tokenizer, f)
                        joblib.dump(label_encoder, "label_encoder.pkl")
                        
                        # Plot training history
                        st.subheader("Training History")
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        ax1.plot(history.history['accuracy'], label='Training Accuracy')
                        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
                        ax1.set_title('Model Accuracy')
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Accuracy')
                        ax1.legend()
                        
                        ax2.plot(history.history['loss'], label='Training Loss')
                        ax2.plot(history.history['val_loss'], label='Validation Loss')
                        ax2.set_title('Model Loss')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Loss')
                        ax2.legend()
                        
                        st.pyplot(fig)
                    
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
                    if is_deep_learning and TENSORFLOW_AVAILABLE:
                        report = classification_report(
                            y_test_labels, y_pred, 
                            target_names=class_names,
                            output_dict=True
                        )
                    else:
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
                    if is_deep_learning:
                        cm = confusion_matrix(y_test_labels, y_pred)
                    else:
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
                    
                    # Save additional info
                    model_info = {
                        'model_type': model_type,
                        'accuracy': accuracy,
                        'class_names': class_names.tolist(),
                        'is_deep_learning': is_deep_learning
                    }
                    
                    if is_deep_learning:
                        if model_type == "Artificial Neural Network (ANN)":
                            model_info['feature_count'] = vocab_size
                        else:
                            model_info.update({
                                'vocab_size': vocab_size,
                                'max_length': max_length,
                                'embedding_dim': embedding_dim
                            })
                    else:
                        model_info['feature_count'] = max_features
                    
                    with open("model_info.pkl", "wb") as f:
                        pickle.dump(model_info, f)
                    
                    st.success("Model and components saved successfully!")
                    
                    # Display class distribution
                    st.subheader("Class Distribution")
                    class_counts = pd.Series(y).value_counts()
                    st.bar_chart(class_counts)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    else:
        st.info("Please upload a CSV file to start training.")

def predict_sentiment():
    st.header("🎯 Predict Sentiment")
    
    # Check if model files exist
    if os.path.exists("sentiment_model.h5"):
        model_file = "sentiment_model.h5"
        is_deep_learning = True
    elif os.path.exists("sentiment_model.pkl"):
        model_file = "sentiment_model.pkl"
        is_deep_learning = False
    else:
        st.error("No trained model found!")
        st.info("Please train a model first using the 'Train New Model' mode.")
        return
    
    try:
        # Load model info
        with open("model_info.pkl", "rb") as f:
            model_info = pickle.load(f)
        
        # Load model and components
        with st.spinner("Loading model..."):
            if is_deep_learning and TENSORFLOW_AVAILABLE:
                model = tf.keras.models.load_model(model_file)
                if model_info['model_type'] == "Artificial Neural Network (ANN)":
                    vectorizer = joblib.load("tfidf_vectorizer.pkl")
                    tokenizer = None
                else:
                    with open("tokenizer.pkl", "rb") as f:
                        tokenizer = pickle.load(f)
                    vectorizer = None
            else:
                model = joblib.load(model_file)
                vectorizer = joblib.load("tfidf_vectorizer.pkl")
                tokenizer = None
            
            label_encoder = joblib.load("label_encoder.pkl")
        
        st.success("Model loaded successfully!")
        
        # Display model info
        with st.expander("Model Information"):
            st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
            st.write(f"**Accuracy:** {model_info.get('accuracy', 'Unknown')}")
            st.write(f"**Number of classes:** {len(model_info['class_names'])}")
            st.write(f"**Classes:** {', '.join(model_info['class_names'])}")
            if is_deep_learning and model_info['model_type'] != "Artificial Neural Network (ANN)":
                st.write(f"**Vocabulary size:** {model_info.get('vocab_size', 'Unknown')}")
                st.write(f"**Max sequence length:** {model_info.get('max_length', 'Unknown')}")
            elif vectorizer:
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
                        if model_info['model_type'] == "Artificial Neural Network (ANN)":
                            # Use TF-IDF for ANN
                            text_vectorized = vectorizer.transform([processed_text]).toarray()
                            probabilities = model.predict(text_vectorized)[0]
                        else:
                            # Use tokenizer for LSTM/GRU
                            text_seq = tokenizer.texts_to_sequences([processed_text])
                            text_padded = pad_sequences(text_seq, maxlen=model_info['max_length'], padding='post')
                            probabilities = model.predict(text_padded)[0]
                        
                        if len(model_info['class_names']) > 2:
                            prediction = np.argmax(probabilities)
                        else:
                            prediction = (probabilities > 0.5).astype(int)[0] if len(probabilities.shape) > 0 else int(probabilities > 0.5)
                            probabilities = np.array([1-probabilities, probabilities]) if probabilities.ndim == 0 else probabilities
                    else:
                        # Traditional ML
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
                            
                            if is_deep_learning and TENSORFLOW_AVAILABLE:
                                if model_info['model_type'] == "Artificial Neural Network (ANN)":
                                    # Use TF-IDF for ANN
                                    texts_vectorized = vectorizer.transform(texts).toarray()
                                    probabilities = model.predict(texts_vectorized)
                                else:
                                    # Use tokenizer for LSTM/GRU
                                    texts_seq = tokenizer.texts_to_sequences(texts)
                                    texts_padded = pad_sequences(texts_seq, maxlen=model_info['max_length'], padding='post')
                                    probabilities = model.predict(texts_padded)
                                
                                if len(model_info['class_names']) > 2:
                                    predictions = np.argmax(probabilities, axis=1)
                                else:
                                    predictions = (probabilities > 0.5).astype(int).flatten()
                                    if probabilities.shape[1] == 1:
                                        probabilities = np.column_stack([1-probabilities.flatten(), probabilities.flatten()])
                            else:
                                # Traditional ML
                                texts_vectorized = vectorizer.transform(texts)
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
                    st.exception(e)
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.exception(e)

# Main app logic
if mode == "Train New Model":
    train_model()
else:
    predict_sentiment()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Suraj Patil")
