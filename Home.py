import streamlit as st

headline = 'NutriViz: Interactive Nutritional Insights'

st.set_page_config(
    page_title=headline,
    page_icon='🥗',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Background with a dark overlay and nutrition-themed image
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                          url('https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&cs=tinysrgb&w=1600');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        color: #f9e79f;
    }
    h1, h2, h3, h4, h5, h6, p, strong {
        color: #f9e79f;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(headline)
st.markdown(""" """, unsafe_allow_html=False)

# 🥦 Project Introduction
st.markdown("""
**Nutrition plays a vital role in maintaining health, preventing disease, and supporting overall well-being. Yet, understanding complex food nutrient labels and databases can be overwhelming.**

**NutriViz is a Streamlit-based visual analytics dashboard that simplifies this complexity by enabling users to explore over 8,800 foods through interactive plots.** The dashboard empowers users to:
- Discover macronutrient and micronutrient compositions,
- Visualize fat types and cholesterol relationships,
- Benchmark food categories,
- Receive personalized food recommendations for health conditions like anemia, heart disease, or diabetes.

**Key features include:**
-  Interactive visualizations (bar plots, radar charts, treemaps, scatter plots, t-SNE, PCA)
-  Clustering-based food suggestions using GMM
-  Nutrient correlation analysis to uncover hidden trade-offs and synergies
-  Tailored exploration by nutrient goals, deficiencies, or dietary patterns

**Dive in to explore your food from a data-driven lens and make smarter nutrition choices with NutriViz!**
""", unsafe_allow_html=False)
