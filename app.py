import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import base64
import time
import pandas as pd
import openai
from dotenv import load_dotenv
import logging

# -- Configure Logging --
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting PDF Image Analyzer application")

# --- API Key Handling ---
def get_openai_api_key():
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        logger.info("API key retrieved from environment variable")
        return env_key
    logger.info("API key retrieved from session state input")
    return st.session_state.api_key_input_value

# --- Session State Initialization ---
for key in [
    'uploaded_pdf_name', 'extracted_images', 'analysis_results',
    'api_key_input_value', 'custom_prompt', 'current_page', 'batch_size']:
    if key not in st.session_state:
        if 'results' in key or 'images' in key:
            st.session_state[key] = []
        elif key == 'batch_size':
            st.session_state[key] = 10
        elif key == 'current_page':
            st.session_state[key] = 0
        else:
            st.session_state[key] = ""

# --- Default Prompt ---
DEFAULT_PROMPT = (
    """
    Analyze the image thoroughly and describe it in a story format. Include details about objects, subjects, actions, and any visible text. Interpret what the image communicates and its significance. Avoid any mention of colors.
    Also, provide a sharp, concise summary under 300 characters that captures the core message of the image.
    """
)

# --- Extract Images from PDF ---
@st.cache_data

def extract_images_for_analysis(pdf_bytes, start_page=0, end_page=None):
    extracted_data = []
    try:
        pdf_file = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(pdf_file)
        if end_page is None:
            end_page = total_pages
    except Exception as e:
        st.error(f"Error opening PDF: {e}")
        return []

    for page_number in range(start_page, min(end_page, total_pages)):
        page = pdf_file[page_number]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = (base_image.get("ext") or "").lower()
                mime_type = f"image/{image_ext}" if image_ext else "image/png"
                if mime_type == "image/jpg":
                    mime_type = "image/jpeg"

                extracted_data.append({
                    'page_number': page_number + 1,
                    'image_index': img_index + 1,
                    'image_bytes': image_bytes,
                    'mime_type': mime_type
                })
            except Exception as e:
                st.error(f"Error processing image {img_index + 1} on page {page_number + 1}: {e}")

    pdf_file.close()
    return extracted_data

# --- OpenAI Vision Analysis ---
def get_image_description_from_openai(image_bytes, prompt_text, api_key, identifier=""):
    if not api_key:
        return "ERROR: OpenAI API Key not provided.", ""
    if not prompt_text:
        return "ERROR: Analysis prompt cannot be empty.", ""

    openai.api_key = api_key
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"

        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            temperature=0.4,
            max_tokens=400
        )

        full_content = response.choices[0].message.content.strip()
        summary_marker = "Summary:"
        if summary_marker in full_content:
            parts = full_content.split(summary_marker, 1)
            long_text = parts[0].strip()
            summary_text = f"\n{parts[1].strip()}"
        else:
            long_text = full_content
            summary_text = (full_content[:300] + "...") if len(full_content) > 300 else full_content

        return long_text, summary_text

    except Exception as e:
        return f"OpenAI error for {identifier}: {e}", ""

# --- Streamlit UI Setup ---
st.set_page_config(page_title="PDF Image Analyzer", page_icon="📄", layout="wide")
st.title("📄 PDF Image Analyzer (Batch Mode)")
st.markdown("Upload a PDF and analyze its images using OpenAI's vision model, 10 pages at a time.")

# --- Sidebar Config ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key_input = st.text_input("OpenAI API Key", type="password", value=st.session_state.api_key_input_value)
    st.session_state.api_key_input_value = api_key_input

# --- Prompt Input ---
st.subheader("📝 Analysis Prompt")
st.session_state.custom_prompt = st.text_area("Enter your prompt for image analysis:",
    value=st.session_state.custom_prompt or DEFAULT_PROMPT, height=150)

# --- File Upload ---
uploaded_file = st.file_uploader("📎 Upload a PDF", type="pdf")

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    total_pages = fitz.open(stream=pdf_bytes, filetype="pdf").page_count
    current_page = st.session_state.current_page
    batch_size = st.session_state.batch_size
    next_page = min(current_page + batch_size, total_pages)

    st.markdown(f"**Processing pages {current_page + 1} to {next_page} of {total_pages}**")

    if st.button("▶️ Process This Batch"):
        with st.spinner("Extracting and analyzing images..."):
            extracted = extract_images_for_analysis(pdf_bytes, start_page=current_page, end_page=next_page)

            if extracted:
                api_key = get_openai_api_key()
                prompt = st.session_state.custom_prompt
                my_bar = st.progress(0, text="Analyzing images...")

                for i, item in enumerate(extracted):
                    identifier = f"Page {item['page_number']} Image {item['image_index']}"
                    long, short = get_image_description_from_openai(
                        item['image_bytes'], prompt, api_key, identifier
                    )
                    st.session_state.analysis_results.append({
                        'Page Number': item['page_number'],
                        'Alt Short Text': short,
                        'Alt Long Text': long
                    })

                    my_bar.progress((i + 1) / len(extracted), text=f"Analyzing {identifier}...")

                my_bar.empty()
                st.session_state.current_page = next_page
                st.success(f"✅ Batch {current_page + 1}–{next_page} processed.")

# --- Final Download Option ---
if st.session_state.analysis_results and st.session_state.current_page >= total_pages:
    st.subheader("📥 Download Complete Results")
    df = pd.DataFrame(st.session_state.analysis_results)
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)

    st.download_button(
        label="Download Full Analysis (Excel)",
        data=excel_buffer,
        file_name="full_image_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Reset Option ---
if st.button("🔁 Reset Analysis"):
    st.session_state.current_page = 0
    st.session_state.analysis_results = []
    st.session_state.extracted_images = []
    st.success("Session reset. You can reprocess the PDF now.")
