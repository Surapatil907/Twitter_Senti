import streamlit as st
import pickle
import numpy as np
import requests
import tempfile
import os

# Function to download file from Google Drive
def download_from_gdrive(file_id, destination):
    """Download file from Google Drive using file ID"""
    try:
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        # Create session for handling redirects
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Handle the download confirmation for large files
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value, 'export': 'download'}
                response = session.get(url, params=params, stream=True)
                break
        
        # Save the file
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        
        return True
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return False

# Function to load model
@st.cache_resource
def load_model():
    """Load the LSTM model from Google Drive"""
    file_id = "1zbxtWOf3FMSnniQ-EsWC6lrdN0PBZyn2"  # Extracted from your URL
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        temp_path = tmp_file.name
    
    # Download the model file
    with st.spinner("Downloading model from Google Drive..."):
        if download_from_gdrive(file_id, temp_path):
            try:
                # Load the model from the temporary file
                with open(temp_path, 'rb') as file:
                    model = pickle.load(file)
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                st.success("✅ Model loaded successfully!")
                return model
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None
        else:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return None

# Main Streamlit app
def main():
    st.title("🧠 LSTM Model Prediction App")
    st.write("This app loads an LSTM model from Google Drive and makes predictions.")
    
    # Load the model
    model = load_model()
    
    if model is not None:
        st.success("Model is ready for predictions!")
        
        # Input section
        st.header("📊 Input Data")
        st.write("Enter your input values below:")
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Comma-separated values", "Space-separated values", "Text area (one per line)"]
        )
        
        input_data = None
        
        if input_method == "Comma-separated values":
            input_values = st.text_input(
                "Enter values separated by commas:",
                placeholder="1.5, 2.3, -0.8, 4.2, 0.1"
            )
            if input_values:
                try:
                    values = [float(x.strip()) for x in input_values.split(',') if x.strip()]
                    input_data = np.array(values).reshape(1, -1)
                except ValueError:
                    st.error("Please enter valid numbers separated by commas")
        
        elif input_method == "Space-separated values":
            input_values = st.text_input(
                "Enter values separated by spaces:",
                placeholder="1.5 2.3 -0.8 4.2 0.1"
            )
            if input_values:
                try:
                    values = [float(x.strip()) for x in input_values.split() if x.strip()]
                    input_data = np.array(values).reshape(1, -1)
                except ValueError:
                    st.error("Please enter valid numbers separated by spaces")
        
        else:  # Text area
            input_values = st.text_area(
                "Enter values (one per line):",
                placeholder="1.5\n2.3\n-0.8\n4.2\n0.1",
                height=150
            )
            if input_values:
                try:
                    values = [float(line.strip()) for line in input_values.split('\n') if line.strip()]
                    input_data = np.array(values).reshape(1, -1)
                except ValueError:
                    st.error("Please enter valid numbers, one per line")
        
        # Display input preview
        if input_data is not None:
            st.subheader("Input Preview")
            st.write(f"**Shape:** {input_data.shape}")
            st.write(f"**Values:** {input_data.flatten()}")
            
            # Show basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Count", len(input_data.flatten()))
            with col2:
                st.metric("Mean", f"{np.mean(input_data):.3f}")
            with col3:
                st.metric("Min", f"{np.min(input_data):.3f}")
            with col4:
                st.metric("Max", f"{np.max(input_data):.3f}")
        
        # Prediction section
        st.header("🎯 Prediction")
        
        if st.button("🚀 Make Prediction", type="primary", disabled=(input_data is None)):
            if input_data is not None:
                try:
                    with st.spinner("Making prediction..."):
                        # Make prediction
                        prediction = model.predict(input_data)
                        
                        # Display results
                        st.success("✅ Prediction completed!")
                        
                        # Show prediction results
                        st.subheader("Prediction Results")
                        
                        if prediction.ndim == 1 and len(prediction) == 1:
                            # Single value prediction
                            st.metric("Predicted Value", f"{prediction[0]:.6f}")
                        else:
                            # Multiple values or array
                            st.write(f"**Prediction Shape:** {prediction.shape}")
                            st.write(f"**Prediction Values:**")
                            
                            if len(prediction.flatten()) <= 10:
                                # Show as metrics if small array
                                cols = st.columns(min(len(prediction.flatten()), 5))
                                for i, val in enumerate(prediction.flatten()):
                                    with cols[i % 5]:
                                        st.metric(f"Output {i+1}", f"{val:.6f}")
                            else:
                                # Show as array for larger predictions
                                st.write(prediction)
                        
                        # Option to download results
                        if st.button("💾 Download Results"):
                            import pandas as pd
                            
                            results_df = pd.DataFrame({
                                'Input': [input_data.flatten().tolist()],
                                'Prediction': [prediction.flatten().tolist()],
                                'Timestamp': [pd.Timestamp.now()]
                            })
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download as CSV",
                                data=csv,
                                file_name=f"lstm_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.write("Please check your input format and try again.")
            else:
                st.warning("Please enter input data first!")
    
    else:
        st.error("❌ Failed to load model. Please check your internet connection and try again.")
        if st.button("🔄 Retry Loading Model"):
            st.cache_resource.clear()
            st.experimental_rerun()

if __name__ == "__main__":
    main()
