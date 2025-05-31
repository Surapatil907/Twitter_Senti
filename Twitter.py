import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gdown
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def download_model(model_url, model_path):
    """Download model from Google Drive with caching"""
    try:
        gdown.download(model_url, model_path, quiet=False)
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

@st.cache_resource
def load_model(model_path):
    """Load model using joblib only"""
    try:
        import joblib
        model = joblib.load(model_path)
        logger.info("Model loaded successfully with joblib")
        return model, None
    except Exception as e:
        logger.error(f"Failed to load model with joblib: {e}")
        return None, str(e)

def validate_input(input_string):
    """Validate and parse user input"""
    if not input_string.strip():
        return None, "Input cannot be empty"
    
    try:
        # Handle different separators
        input_string = input_string.replace(';', ',').replace('|', ',').replace(' ', ',')
        # Remove multiple commas
        input_string = ','.join([x.strip() for x in input_string.split(',') if x.strip()])
        
        values = [float(x.strip()) for x in input_string.split(',')]
        
        if len(values) == 0:
            return None, "No valid numbers found in input"
        
        return np.array(values), None
    except ValueError as e:
        return None, f"Invalid input format. Please enter only numbers separated by commas. Error: {str(e)}"

def create_input_visualization(input_array):
    """Create visualization of input data"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Line plot
    ax1.plot(input_array, marker='o', linewidth=2, markersize=6)
    ax1.set_title('Input Sequence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(input_array, bins=min(10, len(input_array)), alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Value Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 LSTM Model Prediction App</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model file configuration
    model_url = st.sidebar.text_input(
        "Google Drive Model URL", 
        value='https://drive.google.com/uc?id=1zbxtWOf3FMSnniQ-EsWC6lrdN0PBZyn2',
        help="Enter the full Google Drive download URL for your model"
    )
    
    model_path = st.sidebar.text_input(
        "Model File Name", 
        value='tuned_lstm_model.pkl',
        help="Local filename for the downloaded model (should end with .pkl)"
    )
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        reshape_option = st.selectbox(
            "Input Reshape Method",
            ["Auto (1, -1)", "Custom"],
            help="How to reshape input for prediction"
        )
        
        if reshape_option == "Custom":
            custom_shape = st.text_input(
                "Custom Shape (e.g., 1,10,1)",
                help="Enter dimensions separated by commas"
            )
        
        show_visualizations = st.checkbox("Show Input Visualizations", value=True)
        show_statistics = st.checkbox("Show Input Statistics", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Loading")
        
        # Model download and loading
        if not os.path.exists(model_path):
            st.warning("Model file not found locally. Downloading from Google Drive...")
            
            with st.spinner("Downloading model..."):
                download_success = download_model(model_url, model_path)
            
            if download_success:
                st.success("✅ Model downloaded successfully!")
            else:
                st.error("❌ Failed to download model. Please check the URL and try again.")
                st.stop()
        else:
            st.info(f"📁 Model file found: {model_path}")
        
        # Load model using TensorFlow/Keras only
        with st.spinner("Loading model..."):
            model, error = load_model(model_path)
        
        if model is None:
            st.markdown(f'<div class="error-box">❌ Failed to load model: {error}</div>', unsafe_allow_html=True)
            
            # Show specific troubleshooting for PKL models with joblib
            with st.expander("🔧 PKL Model Troubleshooting"):
                st.write("""
                **For .pkl model files using joblib:**
                
                1. **Ensure joblib is installed**: `pip install joblib`
                2. **Check model file integrity**: Re-download if corrupted
                3. **Version compatibility**: Model saved with different joblib/sklearn version
                4. **Model type**: Ensure model is compatible with joblib loading
                
                **Manual loading test:**
                """)
                
                st.code("""
import joblib

# Load PKL model with joblib
try:
    model = joblib.load('your_model.pkl')
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
    if hasattr(model, 'predict'):
        print("Model has predict method")
except Exception as e:
    print(f"Error: {e}")
                """)
            
            st.stop()
        else:
            st.markdown('<div class="info-box">✅ Model loaded successfully with joblib!</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Model Info")
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            st.metric("File Size", f"{file_size:.2f} MB")
            st.metric("Last Modified", mod_time.strftime("%Y-%m-%d %H:%M"))
            
            # Model type info
            if hasattr(model, '__class__'):
                st.info(f"Model Type: {model.__class__.__name__}")
    
    st.divider()
    
    # Input section
    st.subheader("🔢 Input Data")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "File Upload", "Random Sample"],
        horizontal=True
    )
    
    input_array = None
    
    if input_method == "Manual Entry":
        user_input = st.text_area(
            "Enter your data:",
            placeholder="1.0, 2.5, 3.6, 4.2, 5.1\n(You can use commas, semicolons, or spaces as separators)",
            height=100
        )
        
        if user_input:
            input_array, error = validate_input(user_input)
            if error:
                st.error(f"❌ {error}")
            else:
                st.success(f"✅ Parsed {len(input_array)} values successfully")
    
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload a CSV file", 
            type=['csv', 'txt'],
            help="Upload a file with comma-separated values"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                input_array = df.iloc[0].values.astype(float)
                st.success(f"✅ Loaded {len(input_array)} values from file")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    elif input_method == "Random Sample":
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_samples = st.number_input("Number of samples", min_value=1, max_value=1000, value=10)
        with col_b:
            min_val = st.number_input("Min value", value=0.0)
        with col_c:
            max_val = st.number_input("Max value", value=10.0)
        
        if st.button("Generate Random Data"):
            np.random.seed(42)  # For reproducibility
            input_array = np.random.uniform(min_val, max_val, n_samples)
            st.success(f"✅ Generated {len(input_array)} random values")
    
    # Display input analysis
    if input_array is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if show_visualizations:
                st.subheader("📊 Input Visualization")
                fig = create_input_visualization(input_array)
                st.pyplot(fig)
        
        with col2:
            if show_statistics:
                st.subheader("📈 Statistics")
                st.metric("Count", len(input_array))
                st.metric("Mean", f"{np.mean(input_array):.3f}")
                st.metric("Std Dev", f"{np.std(input_array):.3f}")
                st.metric("Min", f"{np.min(input_array):.3f}")
                st.metric("Max", f"{np.max(input_array):.3f}")
    
    st.divider()
    
    # Prediction section
    st.subheader("🎯 Prediction")
    
    if input_array is not None:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            predict_button = st.button("🚀 Make Prediction", type="primary", use_container_width=True)
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()
        
        if predict_button:
            try:
                with st.spinner("Making prediction..."):
                    # Reshape input
                    if reshape_option == "Auto (1, -1)":
                        reshaped_input = input_array.reshape(1, -1)
                    else:
                        try:
                            shape_dims = [int(x.strip()) for x in custom_shape.split(',')]
                            reshaped_input = input_array.reshape(shape_dims)
                        except Exception as e:
                            st.error(f"❌ Invalid custom shape: {e}")
                            st.stop()
                    
                    # Make prediction
                    prediction = model.predict(reshaped_input)
                    
                    # Display results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.success("🎉 Prediction completed successfully!")
                    
                    # Format prediction output
                    if isinstance(prediction, np.ndarray):
                        if prediction.size == 1:
                            st.metric("Prediction", f"{prediction.item():.6f}")
                        else:
                            st.write("**Prediction Array:**")
                            st.write(prediction)
                            
                            # Show prediction statistics if it's an array
                            if len(prediction.flatten()) > 1:
                                pred_flat = prediction.flatten()
                                col_p1, col_p2, col_p3 = st.columns(3)
                                with col_p1:
                                    st.metric("Mean", f"{np.mean(pred_flat):.6f}")
                                with col_p2:
                                    st.metric("Min", f"{np.min(pred_flat):.6f}")
                                with col_p3:
                                    st.metric("Max", f"{np.max(pred_flat):.6f}")
                    else:
                        st.metric("Prediction", f"{prediction}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional info
                    with st.expander("Technical Details"):
                        st.write(f"**Input Shape:** {reshaped_input.shape}")
                        st.write(f"**Prediction Shape:** {prediction.shape if hasattr(prediction, 'shape') else 'N/A'}")
                        st.write(f"**Prediction Type:** {type(prediction)}")
                        st.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Show raw input and prediction
                        st.write("**Raw Input:**")
                        st.code(str(input_array))
                        st.write("**Raw Prediction:**")
                        st.code(str(prediction))
                    
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error during prediction: {str(e)}</div>', unsafe_allow_html=True)
                logger.error(f"Prediction error: {e}")
                
                # Show debugging info
                with st.expander("Debug Information"):
                    st.write(f"**Input array shape:** {input_array.shape}")
                    st.write(f"**Input array type:** {type(input_array)}")
                    st.write(f"**Model type:** {type(model)}")
                    st.write(f"**Error details:** {str(e)}")
    else:
        st.info("👆 Please provide input data to make predictions")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-center; color: #666; font-size: 0.8em; margin-top: 2rem;">
        LSTM Model Prediction App | Enhanced Version with Advanced Features
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
