import streamlit as st
import joblib
import numpy as np
import base64

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
   
    /* Global Styles for Consistent Text Sizing */
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
   
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Consistent Text Sizing Across Devices */
    html {
        font-size: 16px;
    }
    
    body {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Responsive Font Sizes */
    h1 {
        font-size: clamp(2rem, 4vw, 3rem) !important;
    }
    
    h2 {
        font-size: clamp(1.5rem, 3vw, 2rem) !important;
    }
    
    h3 {
        font-size: clamp(1.25rem, 2.5vw, 1.5rem) !important;
    }
    
    p, div, span {
        font-size: clamp(0.9rem, 1.5vw, 1rem) !important;
    }
    
    /* Streamlit Specific Elements */
    .stMarkdown {
        font-size: clamp(0.9rem, 1.5vw, 1rem) !important;
    }
    
    .stSelectbox, .stSlider, .stNumberInput {
        font-size: clamp(0.9rem, 1.5vw, 1rem) !important;
    }
    
    .stButton > button {
        font-size: clamp(1rem, 1.8vw, 1.1rem) !important;
    }
    
    /* Card and Container Responsive Width */
    .card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: clamp(20px, 3vw, 35px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin-bottom: 30px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        max-width: 100%;
        width: 100%;
        box-sizing: border-box;
    }
   
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
   
    .title {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: clamp(20px, 3vw, 30px);
        font-size: clamp(2rem, 4vw, 3rem);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        line-height: 1.2;
    }
   
    .subtitle {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: clamp(15px, 2vw, 20px);
        font-size: clamp(1.2rem, 2.5vw, 1.6rem);
        border-left: 5px solid #3498db;
        padding-left: clamp(10px, 2vw, 15px);
        line-height: 1.3;
    }
   
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: clamp(12px, 2vw, 15px) clamp(20px, 3vw, 30px);
        font-weight: 700;
        width: 100%;
        transition: all 0.4s ease;
        font-size: clamp(1rem, 1.8vw, 1.1rem);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
   
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    }
   
    .input-label {
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        font-size: clamp(1rem, 1.8vw, 1.1rem);
        line-height: 1.4;
    }
   
    .help-icon {
        margin-left: 10px;
        color: #3498db;
        cursor: pointer;
        position: relative;
        display: inline-block;
        font-size: clamp(0.8rem, 1.5vw, 0.9rem);
    }
   
    .help-icon .tooltip {
        visibility: hidden;
        width: min(250px, 90vw);
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        text-align: center;
        border-radius: 10px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 150%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: all 0.3s ease;
        font-size: clamp(0.8rem, 1.5vw, 0.85rem);
        font-weight: 400;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        line-height: 1.4;
        word-wrap: break-word;
    }
   
    .help-icon:hover .tooltip {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(-5px);
    }
   
    .success-box {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: clamp(20px, 3vw, 25px);
        border-radius: 20px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 10px 25px rgba(46, 204, 113, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
        animation: pulse 2s infinite;
    }
   
    .danger-box {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: clamp(20px, 3vw, 25px);
        border-radius: 20px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 10px 25px rgba(231, 76, 60, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }
   
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
   
    .creator-card {
        display: flex;
        align-items: center;
        gap: clamp(15px, 3vw, 25px);
        margin-bottom: 30px;
        padding: 20px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        flex-wrap: wrap;
    }
   
    .creator-img {
        border-radius: 50%;
        border: 5px solid #3498db;
        width: clamp(100px, 20vw, 140px);
        height: clamp(100px, 20vw, 140px);
        object-fit: cover;
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
    }
   
    .contact-links {
        display: flex;
        flex-wrap: wrap;
        gap: clamp(10px, 2vw, 15px);
        margin-top: 20px;
    }
   
    .contact-links a {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        font-weight: 700;
        padding: clamp(10px, 2vw, 12px) clamp(20px, 3vw, 25px);
        border-radius: 50px;
        transition: all 0.4s ease;
        color: white;
        min-width: min(160px, 40vw);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        font-size: clamp(0.9rem, 1.8vw, 1rem);
    }
   
    .contact-links img {
        width: clamp(18px, 4vw, 22px);
        margin-right: 8px;
        filter: brightness(0) invert(1);
    }
   
    .whatsapp {
        background: linear-gradient(135deg, #25D366 0%, #1DA851 100%);
    }
   
    .whatsapp:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 25px rgba(37, 211, 102, 0.4);
    }
   
    .instagram {
        background: linear-gradient(135deg, #405DE6 0%, #E1306C 50%, #FD1D1D 100%);
    }
   
    .instagram:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 25px rgba(225, 48, 108, 0.4);
    }
   
    .email {
        background: linear-gradient(135deg, #EA4335 0%, #4285F4 100%);
    }
   
    .email:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 25px rgba(234, 67, 53, 0.4);
    }
   
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #1a2530 100%);
        color: white;
        box-shadow: 5px 0 25px rgba(0,0,0,0.3);
    }
   
    .sidebar-title {
        color: white;
        font-weight: 800;
        text-align: center;
        margin-bottom: 35px;
        font-size: clamp(1.5rem, 3vw, 2rem);
        background: linear-gradient(135deg, #3498db 0%, #2ecc71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
   
    .fact-box {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #3498db;
        backdrop-filter: blur(5px);
        font-size: clamp(0.8rem, 1.5vw, 0.9rem);
    }
   
    .feature-icon {
        font-size: clamp(1.5rem, 3vw, 2rem);
        margin-bottom: 15px;
        text-align: center;
    }
   
    .stats-container {
        display: flex;
        justify-content: space-around;
        text-align: center;
        margin: 25px 0;
        flex-wrap: wrap;
        gap: 15px;
    }
   
    .stat-item {
        padding: 15px;
        flex: 1;
        min-width: 120px;
    }
   
    .stat-number {
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: 800;
        color: #2c3e50;
        line-height: 1.2;
    }
   
    .stat-label {
        font-size: clamp(0.8rem, 1.5vw, 0.9rem);
        color: #7f8c8d;
        font-weight: 600;
        line-height: 1.3;
    }
   
    .watermark {
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: clamp(0.7rem, 1.5vw, 0.8rem);
        margin-top: 30px;
    }
    
    /* Mobile Optimizations */
    @media (max-width: 768px) {
        .card {
            margin: 10px;
            padding: 15px;
        }
        
        .creator-card {
            flex-direction: column;
            text-align: center;
        }
        
        .stats-container {
            flex-direction: column;
        }
        
        .contact-links {
            justify-content: center;
        }
        
        .contact-links a {
            min-width: 140px;
        }
    }
    
    /* Dark Theme Support */
    @media (prefers-color-scheme: dark) {
        .card {
            background-color: rgba(30, 30, 30, 0.95);
            color: #ffffff;
        }
        
        .input-label {
            color: #ffffff;
        }
        
        .subtitle {
            color: #ffffff;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- About Page ----------------
def about_page():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>About Titanic Survival Prediction</h1>", unsafe_allow_html=True)
   
    st.write("""
    ### Welcome to the Ultimate Titanic Survival Predictor!
   
    This advanced machine learning application predicts whether a passenger would have survived
    the tragic Titanic disaster based on their personal details and circumstances.
    Our model analyzes multiple factors to provide accurate survival predictions with detailed probability analysis.
    """)
   
    # Statistics Section
    st.markdown("<div class='stats-container'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.markdown("<div class='stat-item'>", unsafe_allow_html=True)
        st.markdown("<div class='stat-number'>31.8%</div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>Overall Survival Rate</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
   
    with col2:
        st.markdown("<div class='stat-item'>", unsafe_allow_html=True)
        st.markdown("<div class='stat-number'>74%</div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>Women Survived</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
   
    with col3:
        st.markdown("<div class='stat-item'>", unsafe_allow_html=True)
        st.markdown("<div class='stat-number'>18.9%</div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>Men Survived</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
   
    with col4:
        st.markdown("<div class='stat-item'>", unsafe_allow_html=True)
        st.markdown("<div class='stat-number'>62.9%</div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>First Class Survival</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
   
    st.markdown("---")
   
    st.markdown("<div class='subtitle'>About the Creator</div>", unsafe_allow_html=True)
   
    col1, col2 = st.columns([1, 2])
    with col1:
        # Using a placeholder image - replace with actual image path
        st.image("me.jpg",
                width=200, caption="Mostafa Arafat")
   
    with col2:
        st.markdown("""
        ### **Mostafa Arafat**
        *AI Visionary & Tech Innovator*
       
        Passionate student and AI enthusiast dedicated to shaping the future of predictive technology.
        Founder & Team Lead of **[Dev Scope](https://dev-scope-tech-site.netlify.app/)** -
        where innovation meets execution.
       
        **Mission:** Making advanced AI accessible and understandable for everyone.
        """)
       
        st.markdown("""
        <div style='padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px;'>
            <p style="margin: 8px 0; font-weight: 600;"> <a href="mailto:s78270375@gmail.com" style="color: #3498db; text-decoration: none;">s78270375@gmail.com</a></p>
            <p style="margin: 8px 0; font-weight: 600;"> <a href="tel:+201553848286" style="color: #3498db; text-decoration: none;">+20 155 384 8286</a></p>
        </div>
        """, unsafe_allow_html=True)
   
    st.markdown("---")
    st.markdown("<div class='subtitle'>Let's Connect & Collaborate</div>", unsafe_allow_html=True)
   
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
def input_with_help(label, help_text):
    return f"""
    <div class="input-label">
        {label}
        <div class="help-icon">ℹ️
            <span class="tooltip">{help_text}</span>
        </div>
    </div>
    """

# ---------------- Main Prediction Page ----------------
def main():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Titanic Survival Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: clamp(1rem, 2vw, 1.2rem); color: #7f8c8d;'>Fill in the passenger details below to discover their survival chances</p>", unsafe_allow_html=True)
   
    with st.form("titanic_form"):
        col1, col2 = st.columns(2)
       
        with col1:
            st.markdown(input_with_help("Passenger Class (Pclass)",
                                       "1st Class = Luxury, 2nd Class = Middle, 3rd Class = Economy"),
                       unsafe_allow_html=True)
            pclass = st.selectbox(
                "Pclass",
                [1, 2, 3],
                label_visibility="collapsed"
            )
           
            st.markdown(input_with_help("Gender",
                                       "Women and children had priority during evacuation"),
                       unsafe_allow_html=True)
            sex = st.selectbox(
                "Sex",
                ["Male", "Female"],
                label_visibility="collapsed"
            )
           
            st.markdown(input_with_help("Age",
                                       "Children under 16 had significantly higher survival rates"),
                       unsafe_allow_html=True)
            age = st.slider(
                "Age",
                0, 80, 25,
                label_visibility="collapsed"
            )
           
        with col2:
            st.markdown(input_with_help("Fare Price",
                                       "Ticket cost in pounds - higher fares usually meant better locations"),
                       unsafe_allow_html=True)
            fare = st.number_input(
                "Fare",
                min_value=0.0, max_value=512.0, value=32.0, step=0.1,
                label_visibility="collapsed"
            )
           
            st.markdown(input_with_help("Siblings/Spouses",
                                       "Number of siblings or spouses traveling together"),
                       unsafe_allow_html=True)
            sibsp = st.slider(
                "Siblings/Spouses (SibSp)",
                0, 8, 0,
                label_visibility="collapsed"
            )
           
            st.markdown(input_with_help("Parents/Children",
                                       "Number of parents or children traveling together"),
                       unsafe_allow_html=True)
            parch = st.slider(
                "Parents/Children (Parch)",
                0, 6, 0,
                label_visibility="collapsed"
            )
       
        st.markdown(input_with_help("Embarkation Port",
                                   "C = Cherbourg, Q = Queenstown, S = Southampton"),
                   unsafe_allow_html=True)
        embarked = st.selectbox(
            "Embarked",
            ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"],
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button("Predict Survival Chance")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Load Model and Make Prediction
    try:
        model = joblib.load("titanic_model.pkl")
    except FileNotFoundError:
        st.error("""
        **Model file 'titanic_model.pkl' not found.**
       
        Please make sure:
        - The model file is in the same directory as this app
        - The file name is exactly 'titanic_model.pkl'
        - It's a trained scikit-learn model
        """)
        st.info("**Tip:** If you don't have a model yet, you can train one using the Titanic dataset from Kaggle.")
        return
    
    if submit_button:
        # Process inputs
        sex_val = 1 if sex == "Male" else 0
        embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
        embarked_val = embarked_map[embarked]
        features = np.array([[pclass, sex_val, age, fare, sibsp, parch, embarked_val]])
        
        try:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            survival_chance = probability[1] * 100
           
            if prediction == 1:
                st.markdown(f"""
                <div class='success-box'>
                    <h2>Congratulations! This Passenger Would Survive</h2>
                    <h3>Survival Probability: {survival_chance:.1f}%</h3>
                    <p>Based on historical patterns and machine learning analysis, this passenger had excellent survival chances.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='danger-box'>
                    <h2>Unfortunately, This Passenger Would Not Survive</h2>
                    <h3>Survival Probability: {survival_chance:.1f}%</h3>
                    <p>The analysis indicates this passenger faced significant challenges for survival.</p>
                </div>
                """, unsafe_allow_html=True)
           
            # Show contextual historical facts
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='subtitle'>Historical Context</div>", unsafe_allow_html=True)
           
            if sex == "Female" and pclass == 1:
                st.success("**First Class Women:** Had the highest survival rate (97%) - 'Women and children first' was strongly enforced in first class.")
            elif sex == "Female" and pclass == 3:
                st.info("**Third Class Women:** Faced more challenges reaching lifeboats but still had better chances than men (50% survival rate).")
            elif age < 16:
                st.success("**Children:** Were prioritized during evacuation, resulting in 52% survival rate compared to 31% overall.")
            elif pclass == 3:
                st.warning("**Third Class:** Passengers faced barriers reaching lifeboat decks and had only 24% survival rate.")
            elif pclass == 1:
                st.success("**First Class:** Enjoyed prime location near lifeboats and had 62% survival rate.")
           
            # Additional insights based on specific combinations
            if sex == "Male" and pclass == 3 and age > 16:
                st.error("**Third Class Men:** Had the lowest survival rate (13%) due to location and evacuation protocols.")
           
            st.markdown("</div>", unsafe_allow_html=True)
               
        except Exception as e:
            st.error(f"**Prediction Error:** {str(e)}")
            st.info("Please check that your model is compatible with the input features.")

# ---------------- Sidebar Navigation ----------------
st.sidebar.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Prediction", "About"], label_visibility="collapsed")

# Sidebar Facts Section
st.sidebar.markdown("---")
st.sidebar.markdown("### Titanic Statistics")
st.sidebar.markdown("""
<div class="fact-box">
    <strong>Overall Survival:</strong> 31.8%
</div>
<div class="fact-box">
    <strong>Women Survived:</strong> 74% vs Men: 19%
</div>
<div class="fact-box">
    <strong>Class Survival:</strong> 1st: 63%, 2nd: 43%, 3rd: 24%
</div>
<div class="fact-box">
    <strong>Children Under 16:</strong> 52% survival rate
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Features")
st.sidebar.markdown("""
- Passenger Class
- Gender
- Age
- Fare Price
- Family Size
- Embarkation Port
""")

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="watermark">Built by Mostafa Arafat</div>', unsafe_allow_html=True)

# ---------------- Run App ----------------
if page == "Prediction":
    main()
else:
    about_page()



