import streamlit as st
import pickle
import requests
import tempfile
import os

# --- Google Drive download function ---
def download_from_gdrive(file_id, destination):
    """Download file from Google Drive using file ID."""
    try:
        URL = f"https://drive.google.com/uc?id={1zbxtWOf3FMSnniQ-EsWC6lrdN0PBZyn2
}&export=download"
        session = requests.Session()
        response = session.get(URL, stream=True)

        # Handle confirmation for large files
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                response = session.get(URL, params={"id": file_id, "confirm": value}, stream=True)
                break

        # Write to destination file
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

        return True
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

# --- Load model from Google Drive ---
@st.cache_resource
def load_model_from_drive(file_id):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        path = tmp.name

    if download_from_gdrive(file_id, path):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            os.unlink(path)
            return model
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            os.unlink(path)
            return None
    else:
        os.unlink(path)
        return None

# --- Streamlit UI ---
def main():
    st.title("🎯 Load LSTM Model from Google Drive")

    # Provide your actual file ID here
    file_id = "1zbxtWOf3FMSnniQ-EsWC6lrdN0PBZyn2"

    with st.spinner("Loading model from Google Drive..."):
        model = load_model_from_drive(file_id)

    if model:
        st.success("✅ Model loaded successfully!")
        st.write("Model type:", type(model))
    else:
        st.error("❌ Failed to load model from Google Drive.")

if __name__ == "__main__":
    main()
