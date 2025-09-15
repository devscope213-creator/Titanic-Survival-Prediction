import streamlit as st
import joblib
import numpy as np
import base64

# ---------------- Page Config ----------------
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")

# ---------------- Background Base64 ----------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Segoe UI', sans-serif;
        }}
        .box {{
            background: rgba(255,255,255,0.9);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            margin-bottom: 20px;
        }}
        .creator-card {{
            display: flex;
            align-items: center;
            gap: 25px;
            margin-bottom: 20px;
        }}
        .creator-card img {{
            border-radius: 50%;
            border: 3px solid #ddd;
        }}
        .contact-links a {{
            text-decoration: none;
            font-weight: bold;
            display: inline-flex;
            align-items: center;
            margin: 8px 15px 8px 0;
            padding: 8px 12px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }}
        .contact-links img {{
            width: 22px;
            vertical-align: middle;
            margin-right: 8px;
        }}
        .whatsapp {{
            background: #25D366;
            color: white;
        }}
        .whatsapp:hover {{
            background: #1DA851;
        }}
        .instagram {{
            background: #E1306C;
            color: white;
        }}
        .instagram:hover {{
            background: #C13584;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("back.jpg")

# ---------------- About Page ----------------
def about_page():
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.title("ℹ️ About Titanic Survival Prediction")
    st.write("""
    The **Titanic Survival Prediction App** uses a trained machine learning model  
    to estimate the likelihood of a passenger surviving the Titanic disaster,  
    based on their class, gender, age, and family connections.
    """)
    st.subheader("👨‍💻 Creator")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("me.jpg", width=150, caption="Mostafa Arafat")

    with col2:
        st.write("**Mostafa Arafat**")
        st.write("Visionary student and AI enthusiast shaping predictive technology.")
        st.write("📧 s78270375@gmail.com")
        st.write("📞 +20 155 384 8286")

        st.markdown("---")
        st.subheader("📱 Connect with me")
        st.markdown(
            """
            <div class="contact-links">
                <a href="https://wa.me/201553848286" target="_blank" class="whatsapp">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733585.png"> WhatsApp
                </a>
                <a href="https://www.instagram.com/sasa.7_9?igsh=bnplMHpxMjN5anNv" target="_blank" class="instagram">
                    <img src="https://cdn-icons-png.flaticon.com/512/174/174855.png"> Instagram
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Main Page ----------------
def main():
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.title("🚢 Titanic Survival Prediction")
    st.write("Fill in the passenger details below to check survival chances:")

    with st.form("titanic_form"):
        pclass = st.selectbox(
            "Passenger Class (Pclass)",
            [1, 2, 3],
            help="1 = First Class (luxury), 2 = Second Class, 3 = Third Class (economy)"
        )
        sex = st.selectbox(
            "Sex",
            ["Male", "Female"],
            help="Females had a significantly higher survival rate."
        )
        age = st.slider(
            "Age",
            0, 80, 25,
            help="Younger passengers (especially children) had better chances."
        )
        fare = st.number_input(
            "Fare",
            min_value=0.0, value=32.0, step=0.1,
            help="Ticket cost in pounds. Higher fares often meant First Class."
        )
        sibsp = st.number_input(
            "Siblings/Spouses (SibSp)",
            min_value=0, value=0, step=1,
            help="Number of siblings or spouses aboard the ship."
        )
        parch = st.number_input(
            "Parents/Children (Parch)",
            min_value=0, value=0, step=1,
            help="Number of parents or children aboard the ship."
        )
        embarked = st.selectbox(
            "Embarked",
            ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"],
            help="Port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton"
        )

        submit_button = st.form_submit_button("Predict")

    st.markdown("</div>", unsafe_allow_html=True)

    try:
        model = joblib.load("titanic_model.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model file 'titanic_model.pkl' not found.")
        return

    if submit_button:
        sex_val = 1 if sex == "Male" else 0
        embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
        embarked_val = embarked_map[embarked]

        features = np.array([[pclass, sex_val, age, fare, sibsp, parch, embarked_val]])

        try:
            prediction = model.predict(features)[0]
            if prediction == 1:
                st.success("✅ The passenger is predicted to survive.")
            else:
                st.error("❌ The passenger is predicted not to survive.")
        except Exception as e:
            st.error(f"⚠️ Prediction error: {str(e)}")

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Prediction", "About"])

# ---------------- Run App ----------------
if page == "Prediction":
    main()
else:
    about_page()
