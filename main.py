
import streamlit as st
import pandas as pd
import numpy as np
import random
import cv2
from PIL import Image
import scipy.spatial.distance as distance
import qrcode
from io import BytesIO

# ----------------------------------------------------
# 1. SETUP & SIMULATED DATABASE GENERATION
# ----------------------------------------------------
st.set_page_config(page_title="AI Emergency Identity Recovery", page_icon="🏥", layout="wide")

@st.cache_data
def load_synthetic_database():
    random.seed(42)
    np.random.seed(42)
    
    first_names = ["Kelechi", "Tunde", "Chidi", "Abubakar", "Olumide", "Chioma", "Funmi", "Amina", "Amaka", "Blessing"]
    last_names = ["Okonkwo", "Balogun", "Bello", "Adeleke", "Nwachukwu", "Danladi", "Onyema", "Sani", "Eze", "Awolowo"]
    blood_groups = ["O+", "A+", "B+", "AB+", "O-", "A-"]
    allergies_pool = ["None", "Penicillin", "Sulfa drugs", "Aspirin", "Nuts"]
    states = ["Lagos", "Ogun", "Oyo", "Kano", "Anambra", "Rivers", "FCT Abuja"]
    
    consent_pool = [
        "Fully Consented (Accept All Products)", 
        "Fully Consented (Accept All Products)", 
        "Fully Consented (Accept All Products)", 
        "Refused (No Blood Products or Fractions)", 
        "Consented with Restrictions (Autologous Only)",
        "Consented with Restrictions (Blood Components and Minor Fractions)",
        "Consented with Restrictions (Minor Blood Fractions Only)"
    ]
    
    records = []
    for i in range(50):
        f_name = random.choice(first_names)
        l_name = random.choice(last_names)
        prefix = random.choice(["0803", "0805", "0810", "0902", "0703"])
        
        mock_embedding = np.random.normal(0, 1, 128)
        mock_embedding /= np.linalg.norm(mock_embedding)
        
        records.append({
            "citizen_id": f"NGA-{100500 + i}",
            "full_name": f"{f_name} {l_name}",
            "state": random.choice(states),
            "blood_group": random.choice(blood_groups),
            "allergies": random.choice(allergies_pool),
            "blood_transfusion_consent": random.choice(consent_pool),
            "next_of_kin_name": f"{random.choice(first_names)} {l_name}",
            "next_of_kin_phone": f"+234 {prefix} {random.randint(100, 999)} {random.randint(1000, 9999)}",
            "facial_embedding": mock_embedding.tolist()
        })
    return pd.DataFrame(records)

if 'db' not in st.session_state:
    st.session_state.db = load_synthetic_database()

# ----------------------------------------------------
# 2. IMAGE MATH & QR DETECTOR ENGINE
# ----------------------------------------------------
def extract_opencv_features(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_resized = cv2.resize(img, (64, 64))
    flattened = img_resized.flatten()
    signature = flattened[::32].astype(float) 
    if np.linalg.norm(signature) != 0:
        signature /= np.linalg.norm(signature)
    return signature

def process_and_match(image_bytes, registry_df, threshold=0.75):
    try:
        victim_vector = extract_opencv_features(image_bytes)
        if victim_vector is None:
            return None, "Invalid image format received."
    except Exception as e:
        return None, f"Matrix Processing Error: {str(e)}"
    
    best_match_idx = None
    min_distance = float("inf")
    
    for idx, row in registry_df.iterrows():
        db_vector = np.array(row["facial_embedding"])
        if len(db_vector) != len(victim_vector):
            continue
        cos_dist = distance.cosine(victim_vector, db_vector)
        if cos_dist < min_distance:
            min_distance = cos_dist
            best_match_idx = idx
            
    if best_match_idx is not None and min_distance <= threshold:
        confidence = (1 - min_distance) * 100
        return registry_df.loc[best_match_idx].to_dict(), confidence
    return None, (1 - min_distance) * 100 if best_match_idx is not None else 0

def decode_qr_code(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not resolve scanner input matrix."
    detector = cv2.QRCodeDetector()
    data, bbox, straight_qrcode = detector.detectAndDecode(img)
    if data:
        return data, None
    return None, "No active QR structure detected in the image area."

def generate_qr_image(text_data):
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(text_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ----------------------------------------------------
# 3. STREAMLIT DUAL-MODE UI DESIGN
# ----------------------------------------------------
st.title("🏥 AI-Assisted Patient Identity Recovery System")
st.subheader("Nigerian Hospital Emergency Triage MVP (Face + QR Wristband Hybrid)")
st.markdown("---")

col_control, col_display = st.columns([1.2, 1.3])

with col_control:
    st.header("📸 Emergency Scanning & Registration Intake")
    
    tab_face, tab_qr, tab_register, tab_sandbox = st.tabs([
        "👤 Biometric Face Scan", 
        "🎫 Bracelet / Card Scan", 
        "📝 Citizen Registration",
        "🛠️ Sandbox Database"
    ])
    
    active_profile = None
    status_msg = None
    confidence_metric = None
    
    with tab_face:
        uploaded_face = st.file_uploader("Upload or Capture Patient Face Image...", type=["jpg", "jpeg", "png"], key="face_up")
        if uploaded_face:
            st.image(uploaded_face, caption="ER Input Photo", use_container_width=True)
            with st.spinner("Analyzing image grid signatures via OpenCV..."):
                uploaded_face.seek(0)
                matched_dict, confidence_metric = process_and_match(uploaded_face, st.session_state.db)
                if matched_dict is not None:
                    active_profile = matched_dict
                else:
                    status_msg = "No biometric match found in active registry records."

    with tab_qr:
        uploaded_qr = st.file_uploader("Scan Emergency Bracelet or Medical ID Card QR...", type=["jpg", "jpeg", "png"], key="qr_up")
        if uploaded_qr:
            st.image(uploaded_qr, caption="Scanned Token Area", use_container_width=True)
            with st.spinner("Decoding token structures via OpenCV QR Engine..."):
                uploaded_qr.seek(0)
                decoded_id, qr_err = decode_qr_code(uploaded_qr)
                if decoded_id:
                    match_rows = st.session_state.db[st.session_state.db["citizen_id"] == decoded_id]
                    if not match_rows.empty:
                        # Extract row safely as a flat dictionary object
                        active_profile = match_rows.iloc[0].to_dict()
                        confidence_metric = 100.0
                    else:
                        status_msg = f"Token Decoded as `{decoded_id}`, but no citizen matches this ID."
                else:
                    status_msg = qr_err
                    
    with tab_register:
        st.markdown("### 📋 National Health Registry Onboarding Form")
        
        with st.form("registration_form", clear_on_submit=True):
            reg_name = st.text_input("Full Name (First Last)", placeholder="e.g. Chukwuma Awolowo")
            
            c1, c2 = st.columns(2)
            with c1:
                reg_blood = st.selectbox("Blood Group", ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"])
                reg_state = st.text_input("State of Residence", value="Lagos")
            with c2:
                reg_consent = st.selectbox("Transfusion Consent Directive", [
                    "Fully Consented (Accept All Products)",
                    "Refused (No Blood Products or Fractions)",
                    "Consented with Restrictions (Autologous Only)",
                    "Consented with Restrictions (Blood Components and Minor Fractions)",
                    "Consented with Restrictions (Minor Blood Fractions Only)"
                ])
                reg_allergies = st.text_input("Known Medical Allergies", value="None")
                
            st.markdown("---")
            st.write("**Emergency Contact (Next of Kin)**")
            reg_nok_name = st.text_input("Next of Kin Full Name")
            reg_nok_phone = st.text_input("Next of Kin Phone Number", placeholder="e.g. +234 0803 123 4567")
            
            st.markdown("---")
            reg_photo = st.file_uploader("Upload Profile Image to Extract Biometric Blueprint Signature", type=["jpg", "jpeg", "png"], key="reg_pic")
            
            submit_btn = st.form_submit_button("Register Citizen & Generate Identity Card Tokens")
            
            if submit_btn:
                if not reg_name or not reg_nok_name or not reg_nok_phone:
                    st.error("Missing Profile Criteria: Name, Next of Kin, and Phone numbers are mandatory.")
                else:
                    new_id = f"NGA-{random.randint(110000, 999999)}"
                    if reg_photo:
                        reg_photo.seek(0)
                        extracted_sig = extract_opencv_features(reg_photo)
                        vector_list = extracted_sig.tolist() if extracted_sig is not None else np.random.normal(0, 1, 128).tolist()
                    else:
                        vector_list = np.random.normal(0, 1, 128).tolist()
                        
                    new_row = {
                        "citizen_id": new_id,
                        "full_name": reg_name,
                        "state": reg_state,
                        "blood_group": reg_blood,
                        "allergies": reg_allergies,
                        "blood_transfusion_consent": reg_consent,
                        "next_of_kin_name": reg_nok_name,
                        "next_of_kin_phone": reg_nok_phone,
                        "facial_embedding": vector_list
                    }

