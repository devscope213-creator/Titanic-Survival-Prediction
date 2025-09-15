import streamlit as st
import joblib
import numpy as np
import base64

# ---------------- Page Config ----------------
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 25px;
    }
    
    .title {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        margin-bottom: 25px;
        font-size: 2.5rem;
    }
    
    .subtitle {
        color: #3498db;
        font-weight: 600;
        margin-bottom: 15px;
        font-size: 1.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    
    .input-label {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    
    .help-icon {
        margin-left: 8px;
        color: #3498db;
        cursor: pointer;
        position: relative;
        display: inline-block;
    }
    
    .help-icon .tooltip {
        visibility: hidden;
        width: 200px;
        background-color: #2c3e50;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        font-weight: normal;
    }
    
    .help-icon:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }
    
    .success-box {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .danger-box {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .creator-card {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 25px;
    }
    
    .creator-img {
        border-radius: 50%;
        border: 4px solid #3498db;
        width: 120px;
        height: 120px;
        object-fit: cover;
    }
    
    .contact-links a {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        font-weight: 600;
        margin: 10px 15px 10px 0;
        padding: 10px 20px;
        border-radius: 50px;
        transition: all 0.3s ease;
        color: white;
    }
    
    .contact-links img {
        width: 20px;
        margin-right: 8px;
    }
    
    .whatsapp {
        background: #25D366;
        box-shadow: 0 4px 10px rgba(37, 211, 102, 0.3);
    }
    
    .whatsapp:hover {
        background: #1DA851;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(37, 211, 102, 0.4);
    }
    
    .instagram {
        background: linear-gradient(45deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);
        box-shadow: 0 4px 10px rgba(220, 39, 67, 0.3);
    }
    
    .instagram:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(220, 39, 67, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #1a2530 100%);
        color: white;
    }
    
    .sidebar-title {
        color: white;
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- About Page ----------------
def about_page():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>ℹ️ About Titanic Survival Prediction</h1>", unsafe_allow_html=True)
    st.write("""
    The **Titanic Survival Prediction App** uses a trained machine learning model  
    to estimate the likelihood of a passenger surviving the Titanic disaster,  
    based on their class, gender, age, and family connections.
    """)
    
    st.markdown("<div class='subtitle'>👨‍💻 Creator</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("me.jpg", width=150, caption="Mostafa Arafat")

    with col2:
        st.write("**Mostafa Arafat**")
        st.write("Visionary student and AI enthusiast shaping predictive technology.")
        st.write("📧 s78270375@gmail.com")
        st.write("📞 +20 155 384 8286")

        st.markdown("---")
        st.markdown("<div class='subtitle'>📱 Connect with me</div>", unsafe_allow_html=True)
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

# ---------------- Helper Function for Input Labels ----------------
def input_label_with_help(label, help_text):
    return f"""
    <div class="input-label">
        {label}
        <div class="help-icon">ℹ️
            <span class="tooltip">{help_text}</span>
        </div>
    </div>
    """

# ---------------- Main Page ----------------
def main():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>🚢 Titanic Survival Prediction</h1>", unsafe_allow_html=True)
    st.write("Fill in the passenger details below to check survival chances:")

    with st.form("titanic_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(input_with_help("Passenger Class (Pclass)", 
                                       "1 = First Class (luxury), 2 = Second Class, 3 = Third Class (economy)"), 
                       unsafe_allow_html=True)
            pclass = st.selectbox(
                "Pclass",
                [1, 2, 3],
                label_visibility="collapsed"
            )
            
            st.markdown(input_with_help("Sex", 
                                       "Females had a significantly higher survival rate"), 
                       unsafe_allow_html=True)
            sex = st.selectbox(
                "Sex",
                ["Male", "Female"],
                label_visibility="collapsed"
            )
            
            st.markdown(input_with_help("Age", 
                                       "Younger passengers (especially children) had better chances"), 
                       unsafe_allow_html=True)
            age = st.slider(
                "Age",
                0, 80, 25,
                label_visibility="collapsed"
            )
            
        with col2:
            st.markdown(input_with_help("Fare", 
                                       "Ticket cost in pounds. Higher fares often meant First Class"), 
                       unsafe_allow_html=True)
            fare = st.number_input(
                "Fare",
                min_value=0.0, value=32.0, step=0.1,
                label_visibility="collapsed"
            )
            
            st.markdown(input_with_help("Siblings/Spouses (SibSp)", 
                                       "Number of siblings or spouses aboard the ship"), 
                       unsafe_allow_html=True)
            sibsp = st.number_input(
                "Siblings/Spouses (SibSp)",
                min_value=0, value=0, step=1,
                label_visibility="collapsed"
            )
            
            st.markdown(input_with_help("Parents/Children (Parch)", 
                                       "Number of parents or children aboard the ship"), 
                       unsafe_allow_html=True)
            parch = st.number_input(
                "Parents/Children (Parch)",
                min_value=0, value=0, step=1,
                label_visibility="collapsed"
            )
        
        st.markdown(input_with_help("Embarked", 
                                   "Port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton"), 
                   unsafe_allow_html=True)
        embarked = st.selectbox(
            "Embarked",
            ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"],
            label_visibility="collapsed"
        )

        submit_button = st.form_submit_button("Predict Survival")

    st.markdown("</div>", unsafe_allow_html=True)

    try:
        model = joblib.load("titanic_model.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model file 'titanic_model.pkl' not found.")
        st.info("Please make sure the model file is in the same directory as this app.")
        return

    if submit_button:
        sex_val = 1 if sex == "Male" else 0
        embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
        embarked_val = embarked_map[embarked]

        features = np.array([[pclass, sex_val, age, fare, sibsp, parch, embarked_val]])

        try:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            if prediction == 1:
                survival_chance = probability[1] * 100
                st.markdown(f"""
                <div class='success-box'>
                    <h2>✅ The passenger is predicted to survive!</h2>
                    <h3>Survival chance: {survival_chance:.2f}%</h3>
                    <p>Based on the details provided, this passenger had a high likelihood of survival.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                survival_chance = probability[1] * 100
                st.markdown(f"""
                <div class='danger-box'>
                    <h2>❌ The passenger is predicted not to survive</h2>
                    <h3>Survival chance: {survival_chance:.2f}%</h3>
                    <p>Based on the details provided, this passenger had a low likelihood of survival.</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Show some facts based on input
            if sex == "Female" and pclass == 1:
                st.info("💡 Historical fact: First-class women had a survival rate of about 97% on the Titanic.")
            elif age < 16:
                st.info("💡 Historical fact: Children had priority during the evacuation, resulting in higher survival rates.")
            elif pclass == 3:
                st.info("💡 Historical fact: Third-class passengers had much lower survival rates due to their location on the ship and limited access to lifeboats.")
                
        except Exception as e:
            st.error(f"⚠️ Prediction error: {str(e)}")

# ---------------- Helper Function for Input Labels ----------------
def input_with_help(label, help_text):
    return f"""
    <div class="input-label">
        {label}
        <div class="help-icon">ℹ️
            <span class="tooltip">{help_text}</span>
        </div>
    </div>
    """

# ---------------- Sidebar ----------------
st.sidebar.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Prediction", "About"])

# Add some info in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Titanic Facts")
st.sidebar.info("- Overall, only 31.8% of passengers survived\n- 74% of women survived vs 18.9% of men\n- First-class passengers: 62.9% survival rate")

# ---------------- Run App ----------------
if page == "Prediction":
    main()
else:
    about_page()
