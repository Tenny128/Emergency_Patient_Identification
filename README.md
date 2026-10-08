# 🏥 AI-Assisted Patient Identity Recovery System(YEMPI)
### Hybrid Emergency Triage Dashboard for Nigerian Hospitals

🚀 **[Launch App](https://emergencypatientidentification-ppxdf2q3hrgqmhoji5qfcr.streamlit.app/)
[![License: MIT](https://shields.io)](https://opensource.org)

An intelligent, multi-modal emergency patient identification and identity recovery prototype designed to bridge critical administrative bottlenecks in Nigerian emergency rooms, by reducing the number of unidentified accident victims in emergency rooms and gave hospitals quick access to emergency information of patients brought in during an emergency who cannot speak for th emselves.

---

## 🛑 The Problem
In Nigeria, road traffic accidents and acute trauma incidents frequently involve unidentified or unconscious patients. When victims are brought to emergency departments, their smartphones, wallets, and identity cards are often lost or stolen at the scene of the crash. 

Because Nigeria lacks an instantly accessible, universally integrated national health registry, ER doctors and nurses face critical delays trying to:
1. **Identify the patient** to pull baseline medical records.
2. **Determine legal consent directives** (such as blood transfusions).
3. **Locate and contact the Next of Kin (NoK)** for emergency notification.

These delays directly cost lives during the "golden hour" of trauma care.

## 💡 The Solution
This project implements a lightweight, fault-tolerant **Hybrid Face Recognition + Physical Token Verification Engine** built specifically for localized resource-constrained environments. 

By prioritizing speed, low processing overhead, and full offline stability, the system uses two distinct verification pipelines:

1. **👤 Biometric Face Scan:** A lightweight matrix-signature projection system that maps facial geometry layout configurations into a 128-dimensional vector space. It executes fast similarity matches against a cached regional citizen database without needing heavy cloud-dependent deep learning clusters.
2. **🎫 Physical Token Scan (QR Wristbands):** A fallback vector lane utilizing an offline-first OpenCV `QRCodeDetector`. If a patient has sustained severe facial trauma, facial swelling, or bandages, clinicians can scan a durable, thief-resilient silicone medical ID wristband or wallet card to pull an identical 100% accurate database match in milliseconds.

---

## 🩸 Critical Features Included

* **🚨 Transfusion & Blood Component Directives:** Features an automated policy notification box that screens for legal consent directives (e.g., patient-authorized refusal of blood products due to personal, religious, or pre-existing clinical directives). It flashes highly prominent color-coded alerts (`🟢 Fully Consented`, `🔴 Refused`, `🟡 Restricted Handling`) to the triage staff before treatment begins.
* **👻 Automatic Ghost Profile Generation:** If both facial biometrics and physical tokens fail to trigger a database match, the dashboard initializes a structured temporary tracking identifier (e.g., `GHOST-4821`). This allows doctors to instantly begin charting vitals, blood grouping tests, and medication tracking under a unified ID while investigations continue.
* **🌐 Cloud and Edge Optimized:** Bypasses heavy framework runtime dependencies (TensorFlow/Keras compilation locks) by running calculations natively on pre-compiled matrix pipelines. This enables the script to run smoothly on low-end cloud environments or local offline servers in clinics with unreliable grid access.

---

## 🛠️ Technical Stack

* **Frontend Dashboard UI:** Streamlit
* **Computer Vision Engine:** OpenCV (`cv2`)
* **Data Engineering & Math Utilities:** NumPy, Pandas, SciPy
* **Token Matrix Synthesizers:** `qrcode`

---

## 🚀 How to Review Locally

If you wish to test this project locally inside your own isolated environment:

1. Clone this repository:
   ```bash
   git clone https://github.com
   cd Emergency_Patient_Identification
   ```

2. Create a clean virtual environment and activate it:
   ```bash
   python -m venv er_mvp
   # On Windows:
   er_mvp\Scripts\activate
   # On Mac/Linux:
   source er_mvp/bin/activate
   ```

3. Install the pre-compiled application layers:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the local web server console:
   ```bash
   streamlit run main.py
   ```

---

## 📜 Legal & Licensing
This software is shared under the **MIT License**. It includes a full legal disclaimer stating that the software is provided "as is", without warranty of any kind, protecting the creator from operational liability in clinical testing spaces.
