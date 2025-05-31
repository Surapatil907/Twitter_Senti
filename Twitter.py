import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LSTM Model Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class LSTMPredictor:
    def __init__(self):
        self.model = None
        self.model_info = {}
        
    def load_model(self, model_path):
        """Load the LSTM model from file"""
        try:
            if not os.path.exists(model_path):
                st.error(f"Model file not found at: {model_path}")
                return False
                
            with open(model_path, 'rb') as file:
                model_data = pickle.load(file)
                
            # Handle different pickle formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.model_info = model_data.get('info', {})
            else:
                self.model = model_data
                
            if self.model is None:
                st.error("Model could not be loaded from the pickle file")
                return False
                
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            logger.error(f"Model loading error: {e}")
            return False
    
    def validate_input(self, input_data, expected_shape=None):
        """Validate input data"""
        if input_data is None or len(input_data) == 0:
            return False, "Input data is empty"
            
        if expected_shape and input_data.shape[1] != expected_shape[1]:
            return False, f"Expected {expected_shape[1]} features, got {input_data.shape[1]}"
            
        return True, "Input is valid"
    
    def predict(self, input_data):
        """Make prediction using the loaded model"""
        try:
            if self.model is None:
                return None, "Model not loaded"
                
            # Validate input
            is_valid, message = self.validate_input(input_data)
            if not is_valid:
                return None, message
                
            # Make prediction
            prediction = self.model.predict(input_data)
            return prediction, "Success"
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 LSTM Model Prediction App</h1>', unsafe_allow_html=True)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = LSTMPredictor()
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Model path input with different options
        st.subheader("Model Location")
        
        # Option to choose between different environments
        env_option = st.radio(
            "Select Environment:",
            ["Local File", "Google Drive (Colab)", "Upload File"]
        )
        
        model_loaded = False
        
        if env_option == "Google Drive (Colab)":
            # Google Colab Drive mounting
            if st.button("Mount Google Drive"):
                try:
                    from google.colab import drive
                    drive.mount('/content/drive')
                    st.success("Google Drive mounted successfully!")
                except ImportError:
                    st.error("Google Colab not detected. Please use local file option.")
                except Exception as e:
                    st.error(f"Error mounting drive: {e}")
            
            model_path = st.text_input(
                "Model Path in Drive:",
                value="/content/drive/MyDrive/models/tuned_lstm_model.pkl",
                help="Full path to your model file in Google Drive"
            )
            
        elif env_option == "Local File":
            model_path = st.text_input(
                "Local Model Path:",
                value="./models/tuned_lstm_model.pkl",
                help="Path to your local model file"
            )
            
        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Upload Model File",
                type=['pkl', 'pickle'],
                help="Upload your pickled LSTM model"
            )
            if uploaded_file:
                # Save uploaded file temporarily
                model_path = f"temp_model_{uploaded_file.name}"
                with open(model_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        # Load model button
        if st.button("🔄 Load Model", type="primary"):
            if env_option != "Upload File" or (env_option == "Upload File" and 'uploaded_file' in locals() and uploaded_file):
                model_loaded = st.session_state.predictor.load_model(model_path)
                if model_loaded:
                    st.success("✅ Model loaded successfully!")
                    # Display model info if available
                    if st.session_state.predictor.model_info:
                        st.json(st.session_state.predictor.model_info)
            else:
                st.error("Please upload a model file first")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Input Data")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Upload CSV", "Random Sample"]
        )
        
        input_data = None
        
        if input_method == "Manual Input":
            # Manual input with better validation
            st.subheader("Enter Values")
            
            # Option for different input formats
            input_format = st.selectbox(
                "Input Format:",
                ["Comma-separated values", "Space-separated values", "One value per line"]
            )
            
            if input_format == "One value per line":
                input_text = st.text_area(
                    "Enter values (one per line):",
                    height=150,
                    placeholder="1.5\n2.3\n-0.8\n4.2"
                )
                if input_text:
                    try:
                        values = [float(line.strip()) for line in input_text.split('\n') if line.strip()]
                        input_data = np.array(values).reshape(1, -1)
                    except ValueError as e:
                        st.error(f"Invalid input format: {e}")
            else:
                separator = ',' if input_format == "Comma-separated values" else ' '
                input_text = st.text_input(
                    f"Enter values ({input_format}):",
                    placeholder=f"1.5{separator}2.3{separator}-0.8{separator}4.2"
                )
                if input_text:
                    try:
                        values = [float(x.strip()) for x in input_text.split(separator) if x.strip()]
                        input_data = np.array(values).reshape(1, -1)
                    except ValueError as e:
                        st.error(f"Invalid input format: {e}")
        
        elif input_method == "Upload CSV":
            uploaded_csv = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_csv:
                try:
                    df = pd.read_csv(uploaded_csv)
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    # Let user select which row/column to use
                    if st.checkbox("Use specific row"):
                        row_idx = st.number_input("Row index:", min_value=0, max_value=len(df)-1, value=0)
                        input_data = df.iloc[row_idx].values.reshape(1, -1)
                    else:
                        # Use all numeric columns from first row
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            input_data = df[numeric_cols].iloc[0].values.reshape(1, -1)
                        else:
                            st.error("No numeric columns found in CSV")
                            
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
        
        else:  # Random Sample
            st.subheader("Generate Random Sample")
            num_features = st.number_input("Number of features:", min_value=1, max_value=100, value=10)
            random_range = st.slider("Value range:", -10.0, 10.0, (-5.0, 5.0))
            
            if st.button("🎲 Generate Random Data"):
                input_data = np.random.uniform(
                    random_range[0], random_range[1], size=(1, num_features)
                )
                st.success(f"Generated random data with {num_features} features")
        
        # Display input data
        if input_data is not None:
            st.subheader("Input Data Preview")
            
            # Show as dataframe for better visualization
            input_df = pd.DataFrame(
                input_data, 
                columns=[f"Feature_{i+1}" for i in range(input_data.shape[1])]
            )
            st.dataframe(input_df)
            
            # Show basic statistics
            st.write("**Input Statistics:**")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Features", input_data.shape[1])
            with col_b:
                st.metric("Mean", f"{np.mean(input_data):.3f}")
            with col_c:
                st.metric("Std", f"{np.std(input_data):.3f}")
            with col_d:
                st.metric("Range", f"{np.ptp(input_data):.3f}")
    
    with col2:
        st.header("🎯 Prediction")
        
        # Prediction button
        if st.button("🚀 Make Prediction", type="primary", disabled=(input_data is None)):
            if st.session_state.predictor.model is None:
                st.error("Please load a model first!")
            elif input_data is None:
                st.error("Please provide input data!")
            else:
                with st.spinner("Making prediction..."):
                    prediction, message = st.session_state.predictor.predict(input_data)
                    
                    if prediction is not None:
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.success("✅ Prediction Complete!")
                        
                        # Display prediction results
                        if prediction.ndim == 1:
                            if len(prediction) == 1:
                                st.metric("Predicted Value", f"{prediction[0]:.6f}")
                            else:
                                st.write("**Prediction Results:**")
                                pred_df = pd.DataFrame(
                                    prediction.reshape(1, -1),
                                    columns=[f"Output_{i+1}" for i in range(len(prediction))]
                                )
                                st.dataframe(pred_df)
                        else:
                            st.write("**Prediction Shape:**", prediction.shape)
                            st.write("**Prediction Values:**")
                            st.write(prediction)
                        
                        # Visualization if prediction is single value or small array
                        if prediction.size <= 10:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=[f"Output_{i+1}" for i in range(prediction.size)],
                                y=prediction.flatten(),
                                marker_color='lightblue'
                            ))
                            fig.update_layout(
                                title="Prediction Visualization",
                                xaxis_title="Output",
                                yaxis_title="Value",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Option to download results
                        if st.button("💾 Download Results"):
                            results_df = pd.DataFrame({
                                'Input_Features': [input_data.flatten().tolist()],
                                'Prediction': [prediction.flatten().tolist()],
                                'Timestamp': [pd.Timestamp.now()]
                            })
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"lstm_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    else:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error(f"❌ Prediction failed: {message}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with additional information
    st.markdown("---")
    with st.expander("ℹ️ Help & Information"):
        st.markdown("""
        ### How to use this app:
        1. **Load Model**: Use the sidebar to load your LSTM model
        2. **Input Data**: Choose your preferred input method and provide data
        3. **Predict**: Click the prediction button to get results
        
        ### Supported Input Formats:
        - Manual entry (comma/space separated or line-by-line)
        - CSV file upload
        - Random data generation
        
        ### Model Requirements:
        - Model should be saved as a pickle (.pkl) file
        - Model should have a `predict()` method
        - Input data should match the model's expected input shape
        
        ### Troubleshooting:
        - Ensure your model file path is correct
        - Check that input data format matches model requirements
        - Verify that all dependencies are installed
        """)

if __name__ == "__main__":
    main()