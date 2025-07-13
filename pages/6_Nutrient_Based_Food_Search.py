import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# -----------------------------
# %DV Reference Values
# -----------------------------
DAILY_VALUES = {
    "calories": 2000,
    "total_fat": 78,
    "saturated_fat": 20,
    "cholesterol": 300,
    "sodium": 2300,
    "carbohydrate": 275,
    "fiber": 28,
    "sugars": 50,
    "protein": 50,
    "caffeine": 400,
    "vitamin_a": 900,
    "vitamin_c": 90,
    "vitamin_d": 20,
    "vitamin_e": 15,
    "vitamin_k": 120,
    "thiamin": 1.2,
    "riboflavin": 1.3,
    "niacin": 16,
    "vitamin_b6": 1.7,
    "folate": 400,
    "vitamin_b12": 2.4,
    "pantothenic_acid": 5,
    "choline": 550,
    "calcium": 1300,
    "iron": 18,
    "zinc": 11,
    "potassium": 4700,
    "copper": 0.9,
    "magnesium": 420,
    "manganese": 2.3,
    "phosphorus": 1250,
    "selenium": 55
}

health_criteria = {
    'anemia_friendly': {
        'maximize': ['iron', 'vitamin_b6', 'folate', 'vitamin_b12', 'vitamin_c'],  # Added vitamin_c
        'minimize': [],
        'name': 'Anemia-Friendly',
        'description': 'Foods rich in Iron, Vitamin B6, Folate (B9), B12, and Vitamin C (enhances iron absorption)'
    },
    'low_bp_support': {
        'maximize': ['sodium', 'vitamin_b12', 'folate', 'iron'],
        'minimize': [],
        'name': 'Low BP Support',
        'description': 'Moderate Sodium and Rich in B12, Folate, and Iron to Support Blood Volume and Pressure'
    },
    'heart_healthy': {
        'maximize': ['fiber', 'potassium'],
        'minimize': ['sodium', 'saturated_fat', 'cholesterol', 'trans_fat'],  # Added trans_fat
        'name': 'Heart-Healthy',
        'description': 'Low in Sodium, Saturated Fat, Trans Fat, and Cholesterol; High in Fiber and Potassium'
    },
    'diabetes_aware': {
        'maximize': ['fiber', 'magnesium'],  # Added magnesium
        'minimize': ['sugars'],
        'name': 'Diabetes-Aware',
        'description': 'Low in Sugar, High in Fiber and Magnesium'
    },
    'high_bp_support': {
        'maximize': ['potassium', 'magnesium'],
        'minimize': ['sodium'],
        'name': 'High BP Support',
        'description': 'Low in Sodium, High in Potassium and Magnesium'
    },
    'bone_strengthening': {   
        'maximize': ['calcium', 'vitamin_d', 'vitamin_k', 'phosphorus'],
        'minimize': [],
        'name': 'Bone Health',   
        'description': 'Rich in Calcium, Vitamin D, Vitamin K, and Phosphorus (strengthens bones and prevents osteoporosis)'  # Updated description
    },
    'immune_boosting': {
        'maximize': ['vitamin_c', 'vitamin_e', 'vitamin_a', 'zinc', 'selenium'],
        'minimize': [],
        'name': 'Immune-Boosting',
        'description': 'Rich in Vitamin C, E, A, Zinc, and Selenium'
    },
    'anti_inflammatory': {
        'maximize': ['fiber', 'vitamin_c', 'vitamin_e', 'selenium'],
        'minimize': [],
        'name': 'Anti-Inflammatory',
        'description': 'Rich in Fiber, Vitamin C, E, and Selenium (antioxidants)'
    },
    'weight_management': {
        'maximize': ['protein', 'fiber'],
        'minimize': ['calories'],
        'name': 'Weight Management',
        'description': 'Low in Calories, High in Protein and Fiber'
    },
    'cholesterol_lowering': {
        'maximize': ['fiber'],
        'minimize': [],
        'name': 'Cholesterol-Lowering',
        'description': 'High in Fiber'
    },
    'scurvy_prevention': {
        'maximize': ['vitamin_c'],
        'minimize': [],
        'name': 'Scurvy Prevention',
        'description': 'Rich in Vitamin C'
    },
    'eye_health': {  
        'maximize': ['vitamin_a'],
        'minimize': [],
        'name': 'Eye Health',  # Updated name
        'description': 'Rich in Vitamin A (supports vision and night blindness prevention)'  # Updated description
    },
    'thyroid_support': {
        'maximize': ['selenium', 'zinc', 'iodine'],  # Lowercased iodine
        'minimize': [],
        'name': 'Thyroid Support',
        'description': 'Rich in Selenium, Zinc, and Iodine'  # Updated description
    },
    'mental_wellness': {
        'maximize': ['vitamin_b6', 'folate', 'vitamin_b12', 'magnesium'],
        'minimize': [],
        'name': 'Mental Wellness',
        'description': 'Rich in Vitamin B6, Folate (B9), B12, and Magnesium'
    },
    'fatigue_relief': {
        'maximize': ['iron', 'vitamin_b12', 'folate', 'vitamin_c'],
        'minimize': [],
        'name': 'Fatigue Relief',
        'description': 'Rich in Iron, Vitamin B12, Folate (B9), and Vitamin C'
    },
    'skin_health': {
        'maximize': ['vitamin_e', 'vitamin_c', 'vitamin_a', 'zinc', 'riboflavin'],
        'minimize': [],
        'name': 'Skin Health',
        'description': 'Rich in Vitamin E, C, A, Zinc, and Riboflavin (B2)'
    },
    'hair_growth_support': {
        'maximize': ['zinc', 'protein', 'iron', 'biotin'],  # Added biotin
        'minimize': [],
        'name': 'Hair Growth Support',
        'description': 'Rich in Zinc, Protein, Iron, and Biotin (B7)'
    },
    'pellagra_prevention': {
        'maximize': ['niacin'],
        'minimize': [],
        'name': 'Pellagra Prevention',
        'description': 'Rich in Niacin (B3)'
    },
    'beriberi_prevention': {
        'maximize': ['thiamin'],
        'minimize': [],
        'name': 'Beriberi Prevention',
        'description': 'Rich in Thiamin (B1)'
    },
    'ariboflavinosis_prevention': {
        'maximize': ['riboflavin'],
        'minimize': [],
        'name': 'Ariboflavinosis Prevention',
        'description': 'Rich in Riboflavin (B2)'
    },
    'paresthesia_relief': {
        'maximize': ['pantothenic_acid'],
        'minimize': [],
        'name': 'Paresthesia Relief',
        'description': 'Rich in Pantothenic Acid (B5)'
    },
    'clotting_support': {  # Renamed key, name, and description
        'maximize': ['vitamin_k'],
        'minimize': [],
        'name': 'Blood Clotting Support',  # Updated name
        'description': 'Rich in Vitamin K (essential for blood clotting)'  # Updated description
    },
    'fertility_boosting': {
        'maximize': ['vitamin_e', 'zinc', 'folate'],
        'minimize': [],
        'name': 'Fertility Boosting',
        'description': 'Rich in Vitamin E, Zinc, and Folate (B9)'
    },
    'anti_anxiety': {
        'maximize': ['magnesium', 'vitamin_b6'],
        'minimize': [],
        'name': 'Anti-Anxiety',
        'description': 'Rich in Magnesium and Vitamin B6'
    },
    'muscle_recovery': {
        'maximize': ['protein', 'potassium', 'magnesium'],
        'minimize': [],
        'name': 'Muscle Recovery',
        'description': 'Rich in Protein, Potassium, and Magnesium'
    },
    'liver_detox': {
        'maximize': ['choline', 'thiamin', 'riboflavin', 'niacin', 'vitamin_b6', 'folate', 'vitamin_b12', 'pantothenic_acid'],
        'minimize': [],
        'name': 'Liver Detox',
        'description': 'Rich in Choline and B Vitamins'
    },
    'cognitive_support': {
        'maximize': ['choline', 'vitamin_b12', 'iron'],
        'minimize': [],
        'name': 'Cognitive Support',
        'description': 'Rich in Choline, Vitamin B12, and Iron'
    },
    'constipation_relief': {
        'maximize': ['fiber', 'magnesium', 'water'],  # Added water
        'minimize': [],
        'name': 'Constipation Relief',
        'description': 'Rich in Fiber, Magnesium, and Hydration'
    },
    'kidney_friendly': {
        'maximize': [],
        'minimize': ['potassium', 'phosphorus', 'sodium', 'protein'],  # Added protein
        'name': 'Kidney-Friendly',
        'description': 'Low in Potassium, Phosphorus, Sodium, and Protein (for CKD)'
    }
}

# -----------------------------
# Load and Clean the Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel('data/nutrition_fullcleaned.xlsx')
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    nutrient_cols = df.columns[2:]
    for col in nutrient_cols:
        df[col] = df[col].apply(lambda x: float(re.findall(r'\d+\.?\d*', str(x))[0]) if re.findall(r'\d+\.?\d*', str(x)) else 0.0)
    return df, nutrient_cols

df, nutrient_cols = load_data()

# -----------------------------
# Apply GMM Clustering
# -----------------------------
def apply_gmm(df, nutrient_cols, n_components=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[nutrient_cols])
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    df['gmm_cluster'] = gmm.fit_predict(X_scaled)
    return df, X_scaled, gmm

df, X_scaled, gmm_model = apply_gmm(df, nutrient_cols)

# -----------------------------
# Suggest Similar Foods
# -----------------------------
def suggest_similar_foods(df, food_name, X_scaled, top_n=None):
    if food_name not in df['name'].values:
        return pd.DataFrame([{"error": f"❌ Food '{food_name}' not found."}])
    
    food_row = df[df['name'] == food_name].iloc[0]
    cluster_id = food_row['gmm_cluster']
    cluster_df = df[(df['gmm_cluster'] == cluster_id) & (df['name'] != food_name)].copy()
    
    input_index = df[df['name'] == food_name].index[0]
    input_vector = X_scaled[input_index]
    cluster_vectors = X_scaled[cluster_df.index]
    
    distances = np.linalg.norm(cluster_vectors - input_vector, axis=1)
    cluster_df['similarity'] = distances
    
    sorted_df = cluster_df.sort_values(by='similarity')
    return sorted_df.head(top_n) if top_n else sorted_df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🥗 Nutrient-Based Food Search")

col1, col2 = st.columns(2)
with col1:
    nutrient_choices = list(DAILY_VALUES.keys())
    selected_nutrient = st.selectbox("Choose Nutrient", nutrient_choices)
with col2:
    level = st.selectbox("Choose Requirement Level", ["High", "Moderate", "Low"])

# -----------------------------
# Filter by %DV
# -----------------------------
def compute_percent_dv(df, nutrient):
    if nutrient not in DAILY_VALUES or nutrient not in df.columns:
        return pd.Series([np.nan] * len(df))
    return (df[nutrient] / DAILY_VALUES[nutrient]) * 100

def filter_by_dv(df, nutrient, level):
    df = df.copy()
    df['percent_dv'] = compute_percent_dv(df, nutrient)
    if level == "Low":
        return df[df["percent_dv"] <= 5]
    elif level == "High":
        return df[df["percent_dv"] >= 20]
    else:
        return df[(df["percent_dv"] > 5) & (df["percent_dv"] < 20)]

result_df = filter_by_dv(df, selected_nutrient, level)

# -----------------------------
# Display Results
# -----------------------------
st.subheader(f"📟 Products with {level} {selected_nutrient.replace('_', ' ').title()}")
if result_df.empty:
    st.warning("No products found for this selection.")
else:
    st.dataframe(
        result_df[['name', 'serving_size', selected_nutrient, 'percent_dv']]
        .sort_values(by='percent_dv', ascending=(level == "Low"))
        .reset_index(drop=True), height=250
    )

# -----------------------------
# Show GMM-Based Alternatives
# -----------------------------
st.subheader("🔁 GMM-Based Similar Food Suggestions")
selected_food = st.selectbox("Pick a Food Item to Find Similar Alternatives", df['name'].unique())
alt_df = suggest_similar_foods(df, selected_food, X_scaled, top_n=None)
# Show similar foods in scrollable full-height table
if "error" in alt_df.columns:
    st.error(alt_df.iloc[0]['error'])
else:
    st.dataframe(
        alt_df[['name', 'serving_size', 'gmm_cluster', 'similarity']].reset_index(drop=True),
        height=250  # Or adjust as needed
    )


# ----------------------------------------
# Disease-Friendly Search
# ----------------------------------------
st.subheader("🩺 Search by Health Condition")

disease_options = {v["name"]: k for k, v in health_criteria.items()}
selected_disease = st.selectbox("Choose a Condition", list(disease_options.keys()))

if selected_disease:
    condition_key = disease_options[selected_disease]
    condition = health_criteria[condition_key]

    st.markdown(f"**🔍 Criteria:** {condition['description']}")

    df_copy = df.copy()

    # Normalize nutrient columns for fair comparison
    for nutrient in condition['maximize'] + condition['minimize']:
        if nutrient in df_copy.columns:
            max_val = df_copy[nutrient].max()
            if max_val > 0:
                df_copy[nutrient] = df_copy[nutrient] / max_val

    # Scoring: foods higher in 'maximize' nutrients and lower in 'minimize'
    df_copy["disease_score"] = 0
    for nutrient in condition["maximize"]:
        if nutrient in df_copy.columns:
            df_copy["disease_score"] += df_copy[nutrient]
    for nutrient in condition["minimize"]:
        if nutrient in df_copy.columns:
            df_copy["disease_score"] -= df_copy[nutrient]

    # Show results
    disease_results = df_copy.sort_values(by="disease_score", ascending=False).head(20)

    st.subheader(f"🍽️ Top Foods for: {selected_disease}")
    # Only include columns that are actually in the DataFrame
    display_cols = ['name', 'serving_size', 'disease_score']
    for col in condition['maximize'] + condition['minimize']:
        if col in df_copy.columns:
            display_cols.append(col)

    st.dataframe(disease_results[display_cols].reset_index(drop=True), height=250)

