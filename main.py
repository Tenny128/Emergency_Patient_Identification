import streamlit as st
import pandas as pd
import numpy as np
import random
import cv2
from PIL import Image
import scipy.spatial.distance as distance
import qrcode
from io import BytesIO
import sqlite3
import json

DB_FILE = "hospital_registry.db"


# ----------------------------------------------------
# 1. PERMANENT SQL DATABASE SETUP & SEEDING
# ----------------------------------------------------
def init_sql_database():
    """Initializes the SQLite database schema and seeds 50 baseline profiles if completely empty."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create the relational schema table layout with strict data types
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS citizens
                   (
                       citizen_id
                       TEXT
                       PRIMARY
                       KEY,
                       full_name
                       TEXT
                       NOT
                       NULL,
                       state
                       TEXT
                       NOT
                       NULL,
                       blood_group
                       TEXT
                       NOT
                       NULL,
                       allergies
                       TEXT
                       NOT
                       NULL,
                       blood_transfusion_consent
                       TEXT
                       NOT
                       NULL,
                       next_of_kin_name
                       TEXT
                       NOT
                       NULL,
                       next_of_kin_phone
                       TEXT
                       NOT
                       NULL,
                       facial_embedding
                       TEXT
                       NOT
                       NULL
                   )
                   """)
    conn.commit()

    # Check if table data density equals 0
    cursor.execute("SELECT COUNT(*) FROM citizens")
    if cursor.fetchone()[0] == 0:
        random.seed(42)
        np.random.seed(42)

        first_names = ["Kelechi", "Tunde", "Chidi", "Abubakar", "Olumide", "Chioma", "Funmi", "Amina", "Amaka",
                       "Blessing"]
        last_names = ["Okonkwo", "Balogun", "Bello", "Adeleke", "Nwachukwu", "Danladi", "Onyema", "Sani", "Eze",
                      "Awolowo"]
        blood_groups = ["O+", "A+", "B+", "AB+", "O-", "A-"]
        allergies_pool = ["None", "Penicillin", "Sulfa drugs", "Aspirin", "Nuts"]
        states = ["Lagos", "Ogun", "Oyo", "Kano", "Anambra", "Rivers", "FCT Abuja"]

        consent_pool = [
            "Fully Consented (Accept All Products)",
            "Fully Consented (Accept All Products)",
            "Refused (No Blood Products or Fractions)",
            "Consented with Restrictions (Autologous Only)",
            "Consented with Restrictions (Blood Components and Minor Fractions)",
            "Consented with Restrictions (Minor Blood Fractions Only)"
        ]

        insert_query = "INSERT INTO citizens VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"

        for i in range(50):
            f_name = random.choice(first_names)
            l_name = random.choice(last_names)
            prefix = random.choice(["0803", "0805", "0810", "0902", "0703"])

            mock_embedding = np.random.normal(0, 1, 128)
            mock_embedding /= np.linalg.norm(mock_embedding)
            vector_json = json.dumps(mock_embedding.tolist())

            cursor.execute(insert_query, (
                f"NGA-{100500 + i}",
                f"{f_name} {l_name}",
                random.choice(states),
                random.choice(blood_groups),
                random.choice(allergies_pool),
                random.choice(consent_pool),
                f"{random.choice(first_names)} {l_name}",
                f"+234 {prefix} {random.randint(100, 999)} {random.randint(1000, 9999)}",
                vector_json
            ))
        conn.commit()
    conn.close()


# Initialize the structural data backbone file immediately upon compilation
init_sql_database()


def fetch_active_registry_dataframe():
    """Queries live rows from the SQLite database file into a standard dataframe matrix."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM citizens", conn)
    conn.close()

    # Parse the stringified JSON array text back into standard float vectors
    df["facial_embedding"] = df["facial_embedding"].apply(json.loads)
    return df


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


def process_and_match(image_bytes, threshold=0.75):
    registry_df = fetch_active_registry_dataframe()
    if registry_df.empty:
        return None, 0.0

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
    data, _, _ = detector.detectAndDecode(img)
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
st.subheader("Nigerian Hospital Emergency Triage MVP (Persistent SQLite Core)")
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
        uploaded_face = st.file_uploader("Upload or Capture Patient Face Image...", type=["jpg", "jpeg", "png"],
                                         key="face_up")
        if uploaded_face:
            st.image(uploaded_face, caption="ER Input Photo", use_container_width=True)
            with st.spinner("Analyzing image grid signatures via SQL Matrix Cache..."):
                uploaded_face.seek(0)
                active_profile, confidence_metric = process_and_match(uploaded_face)
                if active_profile is None:
                    status_msg = "No biometric match identified within live database files."

    with tab_qr:
        uploaded_qr = st.file_uploader("Scan Emergency Bracelet or Medical ID Card QR...", type=["jpg", "jpeg", "png"],
                                       key="qr_up")
        if uploaded_qr:
            st.image(uploaded_qr, caption="Scanned Token Area", width='stretch')
            with st.spinner("Decoding token structures via local SQLite engine..."):
                uploaded_qr.seek(0)
                decoded_id, qr_err = decode_qr_code(uploaded_qr)
                if decoded_id:
                    # Clear out any hidden formatting spaces from the scanner
                    cleaned_id = str(decoded_id).strip()

                    # Query SQLite directly using a parameterized WHERE clause
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM citizens WHERE TRIM(citizen_id) = ?", (cleaned_id,))
                    row_data = cursor.fetchone()

                    if row_data:
                        # Rebuild your Pandas Series array with the matching columns
                        cursor.execute("PRAGMA table_info(citizens)")
                        cols = [col[1] for col in cursor.fetchall()]
                        conn.close()

                        active_profile = pd.Series(row_data, index=cols)
                        confidence_metric = 100.0
                    else:
                        conn.close()
                        status_msg = f"Token Decoded as `{cleaned_id}`, but no matching row exists in the SQL file."
                else:
                    status_msg = qr_err

    with tab_register:
        st.markdown("### 📋 National Health Registry Onboarding Form")
        st.write("Input patient demographics to provision a physical identity token and map biometric signatures.")

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
            reg_photo = st.file_uploader("Upload Profile Image to Extract Biometric Blueprint Signature",
                                         type=["jpg", "jpeg", "png"], key="reg_pic")
            submit_btn = st.form_submit_button("Commit Permanent Citizen to SQL Database Registry")

            if submit_btn:
                if not reg_name or not reg_nok_name or not reg_nok_phone:
                    st.error(
                        "Missing Profile Criteria: Name, Next of Kin, and Phone numbers are strictly mandatory fields.")
                else:
                    new_id = f"NGA-{random.randint(110000, 999999)}"

                    if reg_photo:
                        reg_photo.seek(0)
                        extracted_sig = extract_opencv_features(reg_photo)
                        vector_list = extracted_sig.tolist() if extracted_sig is not None else np.random.normal(0, 1,
                                                                                                                128).tolist()
                    else:
                        vector_list = np.random.normal(0, 1, 128).tolist()

                        # Execute direct SQLite database write cursor statement
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO citizens (citizen_id, full_name, state, blood_group, allergies,
                                                blood_transfusion_consent, next_of_kin_name, next_of_kin_phone,
                                                facial_embedding)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,(new_id, reg_name, reg_state, reg_blood, reg_allergies, reg_consent, reg_nok_name,
                                    reg_nok_phone, json.dumps(vector_list)))
                        conn.commit()
                        conn.close()
                        st.success(f"🎉 Success! Permanently wrote {reg_name} to SQLite rows with ID {new_id}.")
                    card_qr_bytes = generate_qr_image(new_id)
                    st.image(card_qr_bytes, caption=f"Printable QR Badge for {reg_name}", width=200)
                    st.download_button(label="📥 Download Newly Issued Wristband Code", data=card_qr_bytes,
                                       file_name=f"bracelet_{new_id}.png", mime="image/png")

with tab_sandbox:
    st.markdown("### 🔐 Restricted Access Area")
    # Simple state authorization check
    access_password = st.text_input("Enter Hospital Administrator Security Key", type="password")

    if access_password == "NGA_ADMIN_2026":
        st.markdown("### 🧬 Current Active SQLite Database Rows View")
        current_db_df = fetch_active_registry_dataframe()
        st.dataframe(current_db_df[['citizen_id', 'full_name', 'blood_group', 'blood_transfusion_consent']],
                    use_container_width=True)
        st.markdown("---")
        if not current_db_df.empty:
            # Added [0] index to safely pull the first row slice as a flat series array
            demo_target = current_db_df.iloc[0]
            st.info(f"Target Sandbox Selector for testing: **{demo_target['full_name']}** (`{demo_target['citizen_id']}`)")
            qr_bytes = generate_qr_image(demo_target["citizen_id"])
            st.download_button(label="📥 Download Selected Registry Code Wristband", data=qr_bytes,
                            file_name="mock_emergency_bracelet.png", mime="image/png")

            st.markdown("---")
            if uploaded_face and st.button("Link Face Image to Target Slot"):
                try:
                    uploaded_face.seek(0)
                    sig = extract_opencv_features(uploaded_face)
                    if sig is not None:
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute("UPDATE citizens SET facial_embedding = ? WHERE citizen_id = ?",
                                    (json.dumps(sig.tolist()), demo_target['citizen_id']))
                        conn.commit()
                        conn.close()
                        st.success(
                            "Face signature permanently linked to Target SQL slot successfully! Refresh page to execute trace match.")
                except Exception as e:
                    st.error(f"Mapping failed: {e}")
    else:
        st.warning(
            "Access Denied: This tab contains protected health information (PHI) restricted to authorized clinical staff.")

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



