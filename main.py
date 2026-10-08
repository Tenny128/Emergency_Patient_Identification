## Import dependencies

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

    # Pool to simulate realistic consent choices (incorporating personal/religious preferences)
    consent_pool = [
        "Fully Consented (Accept All Products)",
        "Fully Consented (Accept All Products)",
        "Fully Consented (Accept All Products)",
        "Refused (No Blood Products or Fractions)",
        "Consented with Restrictions (Autologous Only)"
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
            "blood_transfusion_consent": random.choice(consent_pool),  # New Policy Directives Column
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

    best_match = None
    min_distance = float("inf")

    for idx, row in registry_df.iterrows():
        db_vector = np.array(row["facial_embedding"])
        if len(db_vector) != len(victim_vector):
            continue
        cos_dist = distance.cosine(victim_vector, db_vector)
        if cos_dist < min_distance:
            min_distance = cos_dist
            best_match = row

    confidence = (1 - min_distance) * 100
    if min_distance <= threshold:
        return best_match, confidence
    return None, confidence


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

col_control, col_display = st.columns([1.1, 1.4])

with col_control:
    st.header("📸 Emergency Scanning Intake")
    tab_face, tab_qr, tab_sandbox = st.tabs(["👤 Biometric Face Scan", "🎫 Bracelet / Card Scan", "🛠️ Sandbox Tools"])

    active_profile = None
    status_msg = None
    confidence_metric = None

    with tab_face:
        uploaded_face = st.file_uploader("Upload or Capture Patient Face Image...", type=["jpg", "jpeg", "png"],
                                         key="face_up")
        if uploaded_face:
            st.image(uploaded_face, caption="ER Input Photo", use_container_width=True)
            with st.spinner("Analyzing image grid signatures via OpenCV..."):
                uploaded_file_bytes = uploaded_face
                uploaded_file_bytes.seek(0)
                active_profile, confidence_metric = process_and_match(uploaded_file_bytes, st.session_state.db)

    with tab_qr:
        uploaded_qr = st.file_uploader("Scan Emergency Bracelet or Medical ID Card QR...", type=["jpg", "jpeg", "png"],
                                       key="qr_up")
        if uploaded_qr:
            st.image(uploaded_qr, caption="Scanned Token Area", use_container_width=True)
            with st.spinner("Decoding token structures via OpenCV QR Engine..."):
                uploaded_qr.seek(0)
                decoded_id, qr_err = decode_qr_code(uploaded_qr)
                if decoded_id:
                    match_row = st.session_state.db[st.session_state.db["citizen_id"] == decoded_id]
                    if not match_row.empty:
                        active_profile = match_row.iloc[0]
                        confidence_metric = 100.0
                    else:
                        status_msg = f"Token Decoded as `{decoded_id}`, but no citizen matches this ID."
                else:
                    status_msg = qr_err

    with tab_sandbox:
        st.markdown("### 🧬 Demo Code Seed Tools")
        demo_target = st.session_state.db.iloc[0]
        st.info(f"Target Citizen for testing: **{demo_target['full_name']}** (`{demo_target['citizen_id']}`)")

        qr_bytes = generate_qr_image(demo_target["citizen_id"])
        st.download_button(label="📥 Download Test QR Code Wristband Image", data=qr_bytes,
                           file_name="mock_emergency_bracelet.png", mime="image/png")

        st.markdown("---")
        if uploaded_face and st.button("Link Face Image to Target Slot"):
            try:
                uploaded_face.seek(0)
                sig = extract_opencv_features(uploaded_face)
                if sig is not None:
                    st.session_state.db.at[0, "facial_embedding"] = sig.tolist()
                    st.success("Face signature linked to Target Slot successfully!")
            except Exception as e:
                st.error(f"Mapping failed: {e}")

with col_display:
    st.header("📋 Identity & Life-Saving Metadata")

    if isinstance(active_profile, pd.Series):
        if confidence_metric == 100.0:
            st.success("✅ Match Confirmed via Token Scan! (Accuracy: 100%)")
        else:
            st.success(f"✅ Match Confirmed via Biometric Features! (Confidence Score: {confidence_metric:.2f}%)")

        # Display Critical Medical Info First
        st.markdown("### 🩸 Critical Medical Profile")
        med_c1, med_c2 = st.columns(2)
        with med_c1:
            st.metric(label="Blood Group", value=active_profile["blood_group"])
        with med_c2:
            st.metric(label="Known Allergies", value=active_profile["allergies"])

        # Clear, actionable notice box for transfusion directives
        st.markdown("#### 📢 Transfusion & Blood Components Directive")
        consent_status = active_profile["blood_transfusion_consent"]

        if "Fully Consented" in consent_status:
            st.success(f"✔️ {consent_status}")
        elif "Refused" in consent_status:
            st.error(f"⚠️ LEGAL NOTICE: {consent_status}")
        else:
            st.warning(f"⚡ RESTRICTED HANDLING: {consent_status}")

        st.markdown("### 👤 Personal Details")
        st.write(f"**Full Name:** {active_profile['full_name']}")
        st.write(f"**National Health ID:** {active_profile['citizen_id']}")
        st.write(f"**State of Residence:** {active_profile['state']}")

        st.markdown("---")
        st.markdown("### 📞 Next of Kin (Emergency Notification Contact)")
        st.warning(f"**Primary Contact:** {active_profile['next_of_kin_name']}")
        st.subheader(f"📱 Phone: `{active_profile['next_of_kin_phone']}`")

    elif status_msg:
        st.error(f"❌ Verification Error: {status_msg}")

    else:
        st.info("Awaiting input data. Use the panels on the left to complete a face match or scan a token.")
