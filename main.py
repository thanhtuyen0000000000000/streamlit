# File: main.py
import streamlit as st
import os
import clip
import torch
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
from ut import setup_qdrant, load_data_and_upsert, load_clip_model
from qdrant_client.http import models  # Add this line

# Khởi tạo Streamlit
st.title("SEARCH")

# Kiểm tra và thiết lập client và dữ liệu nếu chưa có trong session state
if 'client' not in st.session_state:
    st.session_state.client = setup_qdrant()
    load_data_and_upsert(st.session_state.client)
    st.session_state.model, st.session_state.preprocess = load_clip_model()

# Load mô hình CLIP từ session state
model = st.session_state.model
preprocess = st.session_state.preprocess
client = st.session_state.client

def display_image_with_text(image_path, name_vid, image_name):
    if os.path.exists(image_path):
        # Load the image and prepare text
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Define font size and load font
        font_size = max(15, image.width // 20)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()  # Use default font if custom font is not available

        # Define text and its bounding box
        text = f"Video: {name_vid}, Frame: {image_name}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_position = (image.width // 2 - text_width // 2, image.height - text_height - 10)

        # Create an image for text
        text_image = Image.new("RGB", (image.width, text_height + 30), (0, 0, 0))
        draw_text = ImageDraw.Draw(text_image)
        draw_text.text((image.width // 2 - text_width // 2, 10), text, fill=(255, 255, 255), font=font)

        # Combine the original image with text image
        combined = Image.new("RGB", (image.width, image.height + text_image.height))
        combined.paste(image, (0, 0))
        combined.paste(text_image, (0, image.height))

        st.image(combined, use_column_width=True)
    else:
        st.write(f"Không tìm thấy ảnh: {image_path}")


# Initialize the translator
translator = Translator()

def translate_text(text):
    """
    Translates Vietnamese text to English using googletrans.
    """
    try:
        translation = translator.translate(text, src='vi', dest='en')
        return translation.text
    except Exception as e:
        st.write(f"Error in translation: {e}")
        return text  # Return original text if translation fails
    
# Nhập văn bản tìm kiếm
user_input = st.text_input("Nhập nội dung để tìm kiếm:", "a couple")


translated_input = translate_text(user_input)
# Display the translated text
st.write(f"Nội dung đã dịch: {translated_input}")
# # Tìm kiếm khi người dùng nhập văn bản
# if translated_input:
#     text_input = clip.tokenize([translated_input]).to("cpu")  # Chuyển đổi sang thiết bị phù hợp
#     with torch.no_grad():
#         text_features = model.encode_text(text_input)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#     text_features_flat = text_features.squeeze().tolist()
# Modify the translate_text function to handle long inputs by splitting into chunks
def split_text(text, max_tokens=50):
    """
    Splits a long text into smaller chunks that fit within the model's context length.
    """
    words = text.split()
    return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def encode_text_chunks(model, text_chunks):
    """
    Encodes each text chunk using CLIP and combines the features.
    """
    all_features = []
    for chunk in text_chunks:
        text_input = clip.tokenize([chunk]).to("cpu")  # Convert to tensor and move to CPU
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize the features
        all_features.append(text_features.squeeze())

    # Combine all features by averaging
    combined_features = torch.mean(torch.stack(all_features), dim=0)
    return combined_features.tolist()

text_chunks = split_text(translated_input)

# Display input chunk information
st.write(f"Split input into {len(text_chunks)} chunks for processing.")

# Encode the text chunks
text_features_flat = encode_text_chunks(model, text_chunks)
    # Tìm kiếm trên Qdrant
image_hits = client.search(
        collection_name="mycollection",
        query_vector=models.NamedVector(
            name="image",
            vector=text_features_flat,
        ),
        limit=100
    )

    # Hiển thị kết quả tìm kiếm
st.write("Kết quả tìm kiếm:")
columns = st.columns(3)  # Tạo 6 cột để hiển thị 5-6 ảnh mỗi hàng

for idx, result in enumerate(image_hits):
        image_path = result.payload["image"]
        name_vid = result.payload["Name_Vid"]
        image_name = result.payload["Image Name"]

        col = columns[idx % 3]  # Chọn cột theo chỉ số
        with col:
            display_image_with_text(image_path, name_vid, image_name)
