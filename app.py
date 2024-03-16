import streamlit as st
from PIL import Image

from utils import *
from constants import *

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱🍂", layout="wide")



add_background(background_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your model and class names
# model_save_path = model_state_dict_path
class_names = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)healthy', 'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy']
num_classes = len(class_names)

model, _ = create_vit_model(num_classes=num_classes)

# Load the saved model state dictionary
state_dict = torch.load(model_state_dict_path,map_location=torch.device('cpu'))

# Create a new dictionary and copy the state dictionary's keys and values to the new dictionary
modified_state_dict = {}
for key, value in state_dict.items():
    modified_key = key.replace("heads.weight", "heads.0.weight").replace("heads.bias", "heads.0.bias")
    modified_state_dict[modified_key] = value

# Load the modified state dictionary into the model
model.load_state_dict(modified_state_dict, strict=False)


# Ensure the model is in evaluation mode
model.eval()

st.title("Plant Disease Detection 🌱🍂")
#background_image = 'istockphoto-503646746-612x612.jpg'  
#st.image(background_image, use_column_width=True)
st.sidebar.title("Options")

option = st.sidebar.radio("Select Input Method", ["Upload File", "Enter URL"])

if option == "Upload File":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
#             st.write("Classifying...")
#             pred_and_plot_image(model, class_names, img)

elif option == "Enter URL":
    url = st.text_input("Enter Image URL:")
    if url:
        img = download_image_from_url(url)
        if img is not None:
            st.image(img, caption='Image from URL.', use_column_width=True)
            st.write("")


if st.button("Predict"):
        st.write("Classifying...")
        pred_and_plot_image(model, class_names, img)