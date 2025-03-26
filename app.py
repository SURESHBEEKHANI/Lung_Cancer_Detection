import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import base64

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="logo/logo.png",
    layout="centered"
)

# Cache model loaders for each detection type
@st.cache_resource()
def load_lung_model():
    return YOLO("weights/Lung Cancer Detection.pt")  # Path for lung cancer detection model

# Load lung model only
lung_model = load_lung_model()

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Update prediction function to accept a model parameter
def predict_tumor(image: Image.Image, model):
    try:
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        results = model.predict(image_tensor)
        output_image = results[0].plot()  # Overlay segmentation mask
        return Image.fromarray(output_image)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# Function to encode image to base64 for embedding
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Display logo
image_base64 = get_base64_image("logo/logo.png")
st.markdown(
    f'<div style="text-align: center;"><img src="data:image/png;base64,{image_base64}" width="100"></div>',
    unsafe_allow_html=True
)

# --- UI Customization ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1E1E2F; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: white; }
        h1 { text-align: center; font-size: 36px; font-weight: bold; color: #2C3E50; }
        div.stButton > button { background-color: #3498DB; color: white; font-weight: bold; }
        div.stButton > button:hover { background-color: #2980B9; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("📤 Upload a CT Image")
uploaded_file = st.sidebar.file_uploader("Drag and drop or browse", type=['jpg', 'png', 'jpeg'])

# Updated: remove detection option since only lung cancer is supported now
detection_option = "Lung Cancer"

# --- Main Page ---
st.title("Lung Cancer Detection")
st.markdown("<p style='text-align: center;'>Detect and segment lung cancer from CT scans.</p>", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
    
    if st.sidebar.button("🔍 Predict " + detection_option):
        segmented_image = predict_tumor(image, lung_model)
        if segmented_image:
            with col2:
                st.image(segmented_image, caption="🎯 Segmented Lung Cancer", use_container_width=True)
        else:
            st.error("Segmentation failed. Please try again.")

st.markdown("---")
st.info("This app uses **YOLO-Seg** for real-time lung cancer detection. Upload a CT image to get started.")