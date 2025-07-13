import streamlit as st
import pandas as pd
import plotly.express as px
import re

# Load and cache the dataset
@st.cache_data
def load_data():
    df = pd.read_excel('data/nutrition_fullcleaned.xlsx')

    # Clean column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Convert nutrient columns to numeric
    nutrient_cols = df.columns[2:]  # Exclude name and serving_size
    for col in nutrient_cols:
        # Extract numeric values from strings
        df[col] = df[col].apply(lambda x: float(re.findall(r'\d+\.?\d*', str(x))[0]) if re.findall(r'\d+\.?\d*', str(x)) else 0.0)
    
    
    return df

# Load data
df = load_data()

st.header("Food Nutrition Overview")
st.write("Explore detailed nutritional profiles of food products")

# Search bar
selected_food = st.selectbox("Search for a food product", df['name'], index=4)
food_data = df[df['name'] == selected_food].iloc[0]

# Nutrient Batch Labels (unchanged)
st.subheader("Nutritional Characteristics")
labels = []

# Define thresholds (using typical FDA guidelines)
# Convert values to float before comparison
if float(food_data.get('sugars', 0)) > 15:
    labels.append("🍬 High Sugar")
if float(food_data.get('sugars', 0)) < 5:
    labels.append("🍭 low Sugar")
if float(food_data.get('fat', 0)) < 3:
    labels.append("🥑 Low Fat")
if float(food_data.get('fiber', 0)) > 5:
    labels.append("🌾 High Fiber")
if float(food_data.get('sodium', 0)) < 0.140:
    labels.append("🧂 Low Sodium")
if float(food_data.get('sodium', 0)) > 0.570:
    labels.append("🥵 high Sodium")
if float(food_data.get('protein', 0)) > 10:
    labels.append("💪 High Protein")
if float(food_data.get('calories', 0)) < 100:
    labels.append("⚡ Low Calorie")
if float(food_data.get('fat', 0)) > 17.5:
    labels.append("🍟 High Fat")

# if float(food_data.get('saturated_fat', 0)) < 1:
#     labels.append("🫒 Low Saturated Fat")

# Display labels
if labels:
    cols = st.columns(len(labels))
    for i, label in enumerate(labels):
        cols[i].info(label)
else:
    st.info("No significant nutritional characteristics identified")


# Display top 7 nutrients (unchanged)
st.subheader(f"Top 7 Nutrients in {selected_food} (Horizontal Bar Graph)")
nutrient_cols = df.columns[3:]  # Exclude name and serving_size

# Convert to float and get top nutrients (exclude calories)
nutrient_values = food_data[nutrient_cols].drop('calories').astype(float)
top_nutrients = nutrient_values.nlargest(7).sort_values(ascending=True) 
fig1 = px.bar(
    x=top_nutrients.values,
    y=top_nutrients.index,
    orientation='h',
    labels={'x': 'Amount (per 100g)', 'y': 'Nutrient'},
    color=top_nutrients.values,
    color_continuous_scale='Bluered'
)
fig1.update_layout(showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

# Create 3 columns for pie charts
col1, col2, col3 = st.columns(3)

# Macronutrient Distribution - Column 1
with col1:
    st.subheader("Macronutrient Distribution")
    macronutrients = ['carbohydrate', 'protein', 'fat']
    macronutrient_values = [float(food_data.get(nut, 0)) for nut in macronutrients]
    fig2 = px.pie(
        names=macronutrients,
        values=macronutrient_values,
        hole=0.4
    )
    st.plotly_chart(fig2, use_container_width=True)

# Vitamin Distribution - Column 2
with col2:
    st.subheader("Vitamin Distribution")
    vitamins = [col for col in df.columns if 'vitamin' in col and col != 'vitamin_a_rae']
    vitamin_values = food_data[vitamins].astype(float).nlargest(7)
    fig3 = px.pie(
        names=vitamin_values.index,
        values=vitamin_values.values,
        hole=0.4
    )
    st.plotly_chart(fig3, use_container_width=True)

# Mineral Distribution - Column 3
with col3:
    st.subheader("Mineral Distribution(Top 7)")
    minerals = ['calcium', 'copper', 'iron', 'magnesium', 'manganese', 
                'phosphorous', 'potassium', 'selenium', 'zinc', 'sodium']
    mineral_values = food_data[minerals].astype(float).nlargest(7)
    fig4 = px.pie(
        names=mineral_values.index,
        values=mineral_values.values,
        hole=0.4
    )
    st.plotly_chart(fig4, use_container_width=True)


# Add spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Key Insights section
st.subheader("Key Insights")

st.write("""
**Top Nutrients Bar Chart**:  Highlights the most abundant nutrients in this food, helping identify its primary nutritional components at a glance.
         
**Macronutrient Distribution**:  Shows the balance between carbs, protein, and fat – essential for understanding the food’s energy sources and role.
         
**Vitamin Distribution**:  Reveals which vitamins are most prominent, indicating potential benefits related to immune function and metabolism.
         
**Mineral Distribution**:  Identifies key minerals present, crucial for bone health, electrolyte balance, and enzymatic functions in the body.     
                       
**Nutritional Labels**:  Flags significant characteristics like low/high sugar/fat/protein based on FDA guidelines for quick dietary assessment.
""")