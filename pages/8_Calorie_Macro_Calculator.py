import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px # Import plotly.express for bar charts
import numpy as np
from math import floor
import re
import random # For food recommendations

st.set_page_config(layout="wide", page_title="Calorie & Macro Calculator")

st.header("💪 Calorie & Macro Calculator")
st.write("Estimate your daily calorie and macronutrient needs based on your goals.")

# --- Calorie Constants ---
CAL_PER_G_CARB = 4
CAL_PER_G_PROTEIN = 4
CAL_PER_G_FAT = 9

st.markdown("""
<div style="background-color:white;color:black; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h4>Understanding Macronutrient Calories:</h4>
    <ul>
        <li>A gram of <b>Carbohydrate</b> provides <b>4 calories</b>.</li>
        <li>A gram of <b>Protein</b> provides <b>4 calories</b>.</li>
        <li>A gram of <b>Fat</b> provides <b>9 calories</b>.</li>
    </ul>
    <p>Fats are the most calorically dense, meaning they provide more energy per gram. Understanding these values is key to informed dietary choices for weight management and overall health.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Load Data Function ---
@st.cache_data
def load_data(file_path="data/nutrition_fullcleaned.xlsx"):
    df = pd.read_excel(file_path)
    
    # Clean nutrient columns (remove units and convert to numeric)
    nutrient_cols = df.columns.drop(['name', 'serving_size'])
    for col in nutrient_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

    # Fix column name typos
    df.rename(columns={
        'irom': 'iron',
        'zink': 'zinc',
        'phosphorous': 'phosphorus'    
    }, inplace=True)
    
    # Define categories in priority order (most specific first)
    categories_priority = [
        ("Cheese", [r'\bcheese\b', r'cheddar', r'mozzarella', r'parmesan', r'gouda', r'brie', r'feta']),
        ("Seafood", [r'\bfish\b', r'seafood', r'salmon', r'tuna', r'shrimp', r'prawn', r'crab', r'lobster', r'cod', r'tilapia']),
        ("Meats, Poultry, and Fish", [r'\bmeat\b', r'beef', r'chicken', r'pork', r'turkey', r'bacon', r'sausage', r'steak', r'ham', r'lamb']),
        ("Baked Goods", [r'\bbread\b', r'\bcake\b', r'\bcookie\b', r'\bbiscuit\b', r'\bmuffin\b', r'croissant', r'bagel', r'pastry', r'pie\b']),
        ("Cereals and Breakfast Foods", [r'\bcereal\b', r'oatmeal', r'granola', r'\bporridge\b', r'corn flakes', r'muesli']),
        ("Fruits and Fruit Juices", [r'\bfruit\b', r'apple', r'banana', r'orange', 'berry', r'grape', r'mango', r'juice\b', r'melon']),
        ("Vegetables and Vegetable Products", [r'\bvegetable\b', r'potato', r'tomato', r'carrot', r'broccoli', r'spinach', r'cabbage', r'lettuce', r'onion']),
        ("Dairy and Egg Products", [r'\bdairy\b', r'\bmilk\b', r'yogurt', r'cream', r'egg\b', r'curd', r'butter\b', r'custard', r'ice cream']),
        ("Legumes and Legume Products", [r'\blegume\b', r'bean\b', r'soy', r'tofu', r'hummus', r'lentil', r'chickpea']),
        ("Nuts and Seeds", [r'\bnut\b', r'almond', r'walnut', r'peanut', r'seed\b', r'pistachio', r'cashew', r'sunflower seed']),
        ("Grains and Pasta", [r'\bgrain\b', r'rice\b', r'pasta\b', r'spaghetti', r'macaroni', r'quinoa', 'barley', r'oat\b']),
        ("Sweets and Candies", [r'\bcandy\b', r'chocolate', r'sweet\b', r'caramel', r'fudge', r'lollipop', r'gummy']),
        ("Infant Foods", [r'\binfant\b', r'baby food', r'formula\b', r'infant cereal']),
        ("Prepared Meals and Fast Food", [r'meal\b', r'fast food', r'burger', r'pizza', r'lasagna', r'sandwich\b', r'hot dog']),
        ("Sauces, Dips, and Gravies", [r'\bsauce\b', r'\bdip\b', r'gravy\b', r'ketchup', 'mayonnaise', r'salsa', r'ranch']),
        ("Soups and Stews", [r'\bsoup\b', r'\bstew\b', r'broth', r'chowder', r'bisque']),
        ("Snacks", [r'\bsnack\b', r'chips', r'crackers', r'popcorn', r'pretzel', r'trail mix']),
        ("Spices and Herbs", [r'\bspice\b', r'\bherb\b', r'pepper\b', r'cinnamon', r'basil', r'oregano', r'cumin']),
        ("Baking Ingredients", [r'baking', r'flour\b', r'yeast', r'baking powder', r'vanilla extract', r'cocoa powder']),
        ("Fats and Oils", [r'\bfat\b', r'\boil\b', r'olive oil', r'vegetable oil', r'margarine', r'shortening']),
        ("Beverages", [r'\bbeverage\b', r'drink\b', r'soda\b', r'coffee\b', r'tea\b', r'water\b', r'energy drink']),
        ("Other", [])    
    ]
    
    # Pre-compile regex patterns
    compiled_categories = []
    for category, keywords in categories_priority:
        patterns = [re.compile(kw, re.IGNORECASE) for kw in keywords]
        compiled_categories.append((category, patterns))
    
    # Categorization function
    def assign_category(name):
        for category, patterns in compiled_categories:
            for pattern in patterns:
                if pattern.search(name):
                    return category
        return "Other"
    
    df['category'] = df['name'].apply(assign_category)
    return df

# Load the data right after defining the function
try:
    # Corrected file_path to go up one directory from 'pages' then into 'data'
    df_food_data = load_data(file_path="data/nutrition_fullcleaned.xlsx") 
    # Filter out rows with NaN in macronutrients to ensure clean data for visualization
    df_food_data_clean_macros = df_food_data.dropna(subset=['protein', 'fat', 'carbohydrate', 'calories'])
    st.sidebar.success("Food data loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'nutrition_fullcleaned.xlsx' not found. Please ensure it's in the 'data' folder, relative to your project's root, or check the path.")
    df_food_data_clean_macros = pd.DataFrame() # Create an empty DataFrame to avoid errors

# --- Food Recommendation Logic ---
def recommend_food_combination(
    df_food, 
    target_calories, 
    target_protein, 
    target_fat, 
    target_carb, 
    num_recommendations=1, # Changed to 1 as we're recommending one daily combination
    max_items_per_meal=4
):
    recommendations = []
    if df_food.empty:
        return recommendations

    # Filter out foods with zero calories or macros to avoid division by zero or nonsensical recommendations
    df_filtered = df_food[
        (df_food['calories'] > 0) &
        (df_food['protein'] >= 0) & 
        (df_food['fat'] >= 0) & 
        (df_food['carbohydrate'] >= 0)
    ].copy()

    # Pre-calculate calories per gram for each macro
    df_filtered['protein_cal_per_g'] = df_filtered['protein'] * CAL_PER_G_PROTEIN
    df_filtered['fat_cal_per_g'] = df_filtered['fat'] * CAL_PER_G_FAT
    df_filtered['carb_cal_per_g'] = df_filtered['carbohydrate'] * CAL_PER_G_CARB

    for _ in range(num_recommendations):
        current_calories = 0
        current_protein = 0
        current_fat = 0
        current_carb = 0
        
        meal_items = []
        
        # Aim for 3-4 "meals" or major components that contribute significantly
        # Adjusting the range to get a more balanced "daily" meal
        num_main_components = random.randint(4, 7) # More items for a full day
        
        for _ in range(num_main_components): 
            
            # Prioritize based on remaining target, always ensuring a sensible food choice
            remaining_protein_ratio = (target_protein - current_protein) / target_protein if target_protein > 0 else 0
            remaining_fat_ratio = (target_fat - current_fat) / target_fat if target_fat > 0 else 0
            remaining_carb_ratio = (target_carb - current_carb) / target_carb if target_carb > 0 else 0

            food_type_priority = 'protein'
            if remaining_fat_ratio > remaining_protein_ratio and remaining_fat_ratio > remaining_carb_ratio:
                food_type_priority = 'fat'
            elif remaining_carb_ratio > remaining_protein_ratio and remaining_carb_ratio > remaining_fat_ratio:
                food_type_priority = 'carb'

            available_foods = pd.DataFrame() # Initialize empty DataFrame

            if food_type_priority == 'protein':
                available_foods = df_filtered[
                    (df_filtered['protein'] > 5) & 
                    (df_filtered['calories'] > 20) &
                    (df_filtered['name'].str.contains('chicken|beef|fish|egg|tofu|lentil|bean|yogurt|cheese|pork|turkey|shrimp', flags=re.IGNORECASE, regex=True))
                ]
            elif food_type_priority == 'fat':
                 available_foods = df_filtered[
                    (df_filtered['fat'] > 5) & 
                    (df_filtered['calories'] > 20) &
                    (df_filtered['name'].str.contains('avocado|nut|oil|butter|cheese|bacon|salmon|seeds', flags=re.IGNORECASE, regex=True))
                 ]
            elif food_type_priority == 'carb':
                 available_foods = df_filtered[
                    (df_filtered['carbohydrate'] > 5) & 
                    (df_filtered['calories'] > 20) &
                    (df_filtered['name'].str.contains('rice|bread|potato|pasta|oats|fruit|vegetable|quinoa|cereal', flags=re.IGNORECASE, regex=True))
                 ]
            
            if available_foods.empty: # Fallback to general foods if specific category is empty or unsuitable
                available_foods = df_filtered[(df_filtered['calories'] > 10)]

            if available_foods.empty:
                continue # Skip if no foods are available at all

            food_added = False
            for _ in range(50): # Max attempts to find a suitable food for this "meal"
                chosen_food = available_foods.sample(1).iloc[0]

                # Adjust portion size more carefully
                # Aim for portions that contribute to the remaining target without massive overshoot
                portion_factor = 1.0 # default
                if chosen_food['calories'] > 0:
                    remaining_cal_needed = max(0, target_calories - current_calories)
                    if remaining_cal_needed > 0:
                        potential_portion = remaining_cal_needed / chosen_food['calories']
                        # Don't take more than 2x a serving, don't take less than 0.2x a serving
                        portion_factor = min(2.0, max(0.2, potential_portion * random.uniform(0.5, 1.5))) 
                        portion_factor = min(portion_factor, 1.0) if chosen_food['calories'] > target_calories * 0.3 else portion_factor # Limit large portions of very high-calorie foods
                    else: # If calories are met, try to add small amounts of diverse foods
                        portion_factor = random.uniform(0.2, 0.5)
                
                # Check if adding this food (even a small portion) is beneficial
                temp_calories = chosen_food['calories'] * portion_factor
                temp_protein = chosen_food['protein'] * portion_factor
                temp_fat = chosen_food['fat'] * portion_factor
                temp_carb = chosen_food['carbohydrate'] * portion_factor

                # Only add if it helps reach targets and doesn't cause excessive overshoot
                if (current_calories + temp_calories <= target_calories * 1.25 or 
                    (temp_protein > 0 and current_protein < target_protein * 1.1) or
                    (temp_fat > 0 and current_fat < target_fat * 1.1) or
                    (temp_carb > 0 and current_carb < target_carb * 1.1)):
                    
                    current_calories += temp_calories
                    current_protein += temp_protein
                    current_fat += temp_fat
                    current_carb += temp_carb
                    
                    meal_items.append({
                        'name': chosen_food['name'],
                        'calories': temp_calories,
                        'protein': temp_protein,
                        'fat': temp_fat,
                        'carbohydrate': temp_carb,
                        'serving_info': f"approx. {portion_factor:.1f}x serving ({chosen_food.get('serving_size', 'N/A')})"
                    })
                    food_added = True
                    break 
            
        if meal_items:
            recommendations.append({
                'foods': meal_items,
                'total_calories': current_calories,
                'total_protein': current_protein,
                'total_fat': current_fat,
                'total_carb': current_carb
            })
    return recommendations


# --- User Inputs ---
st.subheader("📊 Your Stats & Goals")

input_container = st.container()

with input_container:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Personal Information")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, key="age_input")
        
        weight_unit = st.radio("Weight Unit", ("kg", "lbs"), horizontal=True, key="weight_unit_radio")
        weight = st.number_input(f"Weight ({weight_unit})", min_value=30.0, max_value=300.0, value=70.0, step=0.5, key="weight_input")
        
        height_unit = st.radio("Height Unit", ("cm", "inches"), horizontal=True, key="height_unit_radio")
        height = st.number_input(f"Height ({height_unit})", min_value=100.0, max_value=250.0, value=170.0, step=0.5, key="height_input")
        
        gender = st.radio("Gender", ("Male", "Female"), horizontal=True, key="gender_radio")

    with col2:
        st.markdown("##### Diet Type & Activity")
        diet_type = st.selectbox(
            "Choose Calculator Type:",
            ("Standard", "Leangains", "Keto"),
            key="diet_type_select",
            help="Select 'Standard' for general use, 'Leangains' if following that method, or 'Keto' for a ketogenic diet."
        )

        activity_level_options = {
            "Sedentary: Little or no exercise, office job (1.2x BMR)": 1.2,
            "Lightly Active: Light daily activity & exercise 1-3 days/week (1.375x BMR)": 1.375,
            "Moderately Active: Moderate daily activity & exercise 3-5 days/week (1.55x BMR)": 1.55,
            "Very Active: Physically demanding lifestyle & exercise 6-7 days/week (1.725x BMR)": 1.725,
            "Extremely Active: Hard daily exercise/sports & physical job (1.9x BMR)": 1.9
        }
        activity_level_selection = st.selectbox(
            "Activity Level",
            list(activity_level_options.keys()),
            index=2, # Moderately Active by default
            key="activity_level_select",
            help="Be honest! Most people overestimate their activity. This is primarily based on what you do outside the gym."
        )
        activity_multiplier = activity_level_options[activity_level_selection]

        goal_options = {
            "Lose Weight (-20% deficit)": -0.20,
            "Slowly Lose Weight (-10% deficit)": -0.10,
            "Maintain Weight (0% deviation)": 0.0,
            "Slowly Gain Weight (+10% surplus)": 0.10,
            "Gain Weight (+20% surplus)": 0.20
        }
        goal_selection = st.selectbox(
            "Weight Goal",
            list(goal_options.keys()),
            index=2, # Maintain Weight by default
            key="goal_select",
            help="This sets your caloric deficit/surplus relative to your TDEE."
        )
        goal_multiplier = 1 + goal_options[goal_selection]

# --- Conditional Inputs based on Diet Type ---
st.markdown("---")
st.subheader("Diet Specific Adjustments")

if diet_type == "Leangains":
    st.markdown("##### Leangains Specific Modifiers")
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        body_fat_leangains = st.number_input("Body Fat % (estimated)", min_value=5.0, max_value=50.0, value=15.0, step=0.5, key="body_fat_leangains_input")
    with col_l2:
        muscle_mass = st.selectbox("Muscle Mass", ("Standard", "Muscular", "Very Muscular (Male only)"), index=0, key="muscle_mass_select")
    with col_l3:
        steps_per_day = st.number_input("Average Steps per Day", min_value=0, max_value=20000, value=8000, step=500, key="steps_per_day_input")
    
    leangains_goal_calories = 0
    if goal_selection.startswith("Lose"):
        leangains_goal_calories = -500 if gender == "Male" else -350
    elif goal_selection.startswith("Gain"):
        leangains_goal_calories = 500 if gender == "Male" else 350

elif diet_type == "Keto":
    st.markdown("##### Keto Specific Inputs")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        keto_carb_limit = st.number_input("Daily Carb Limit (g)", min_value=0, max_value=100, value=20, step=1, key="keto_carb_limit_input")
    with col_k2:
        keto_protein_per_lb = st.selectbox(
            "Protein per Pound of Bodyweight",
            ("0.8g/lb", "1.0g/lb", "1.2g/lb"),
            index=1,
            key="keto_protein_select",
            help="Higher protein helps maintain muscle on Keto."
        )
        keto_protein_multiplier = float(keto_protein_per_lb.split('g')[0])
        
if diet_type != "Keto":
    st.markdown("---")
    st.subheader("⚖ Macronutrient Split")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        protein_g_per_lb = st.selectbox(
            "How much protein?",
            ("1g per pound (recommended)", "0.82g per pound (lower bar)", "1.5g per pound (high satiety/muscle retention)"),
            index=0,
            key="protein_g_per_lb_select",
            help="Higher protein helps with satiety and muscle retention, especially during a cut."
        )
        protein_multiplier = float(protein_g_per_lb.split('g')[0])
    with col_m2:
        fat_carb_split = st.slider(
            "Fat % of remaining calories (after protein)",
            min_value=20, max_value=80, value=40, step=5,
            key="fat_carb_split_slider",
            help="Adjust the ratio between fat and carbohydrates from your remaining calories. Generally, consume no less than 0.25g fat per pound of bodyweight."
        )
        carb_percent_remaining = 100 - fat_carb_split
else:
    protein_multiplier = 0 
    fat_carb_split = 0 
    carb_percent_remaining = 0 

st.markdown("---")

# --- Calculations ---
if st.button("Calculate My Macros!", key="calculate_button"):
    if df_food_data_clean_macros.empty:
        st.warning("Cannot perform full visualizations or food recommendations as food data was not loaded.")
    
    with st.spinner("Crunching the numbers..."):
        weight_kg = weight if weight_unit == "kg" else weight * 0.453592
        height_cm = height if height_unit == "cm" else height * 2.54

        bmr = 0
        tdee = 0
        daily_calories = 0
        protein_grams = 0
        fat_grams = 0
        carb_grams = 0
        estimated_weight_change = 0

        # --- BMR Calculation ---
        if diet_type == "Standard" or diet_type == "Keto":
            if gender == "Male":
                bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
            else: # Female
                bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
        
        elif diet_type == "Leangains":
            base_value = 28 if gender == "Male" else 26

            if age < 25: base_value += 0.5
            elif age > 45: base_value -= 0.5
            
            if gender == "Male":
                if height_cm < 167: base_value -= 1
                elif height_cm > 185: base_value += 1
            else:
                if height_cm < 153: base_value -= 1
                elif height_cm > 170: base_value += 1

            if gender == "Male":
                if body_fat_leangains < 10: base_value += 0.5
                elif 20 <= body_fat_leangains < 25: base_value -= 0.5
                elif 25 <= body_fat_leangains < 30: base_value -= 1.5
                elif body_fat_leangains >= 30: base_value -= 2.5
            else:
                if body_fat_leangains < 18: base_value += 0.5
                elif 28 <= body_fat_leangains < 33: base_value -= 0.5
                elif 33 <= body_fat_leangains < 38: base_value -= 1.5
                elif body_fat_leangains >= 38: base_value -= 2.5
            
            if muscle_mass == "Muscular": base_value += 0.5
            elif muscle_mass == "Very Muscular (Male only)": base_value += 1

            if steps_per_day >= 6000 and steps_per_day < 7500: base_value += 0.5
            elif steps_per_day >= 7500:
                base_value += 0.5 + (floor((steps_per_day - 7500) / 1250) * 0.5)

            bmr = base_value * weight_kg

        # --- TDEE Calculation ---
        if diet_type == "Leangains":
            tdee = bmr 
        else:
            tdee = bmr * activity_multiplier

        # --- Daily Calories ---
        if diet_type == "Leangains":
            daily_calories = tdee + leangains_goal_calories
        else:
            daily_calories = tdee * goal_multiplier
        
        daily_calories = max(1000, daily_calories) # Sensible minimum

        # --- Macronutrient Calculation ---
        weight_lbs = weight if weight_unit == "lbs" else weight_kg / 0.453592

        if diet_type == "Keto":
            carb_grams = keto_carb_limit
            protein_grams = weight_lbs * keto_protein_multiplier

            calories_from_carbs = carb_grams * CAL_PER_G_CARB
            calories_from_protein = protein_grams * CAL_PER_G_PROTEIN
            
            remaining_calories = max(0, daily_calories - calories_from_carbs - calories_from_protein)
            fat_grams = remaining_calories / CAL_PER_G_FAT

        else:
            protein_grams = weight_lbs * protein_multiplier
            calories_from_protein = protein_grams * CAL_PER_G_PROTEIN

            remaining_calories_for_fat_carb = max(0, daily_calories - calories_from_protein)

            fat_calories = remaining_calories_for_fat_carb * (fat_carb_split / 100)
            carb_calories = remaining_calories_for_fat_carb * (carb_percent_remaining / 100)

            fat_grams = fat_calories / CAL_PER_G_FAT
            carb_grams = carb_calories / CAL_PER_G_CARB

        # --- Estimated Weight Change ---
        daily_calorie_difference = daily_calories - tdee
        calories_per_kg_fat = 7700 
        
        estimated_weight_change = (daily_calorie_difference * 7) / calories_per_kg_fat

    st.success("Calculations Complete!")

    st.markdown("---")
    st.subheader("Results Overview")

    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("Basal Metabolic Rate (BMR)", f"{int(bmr)} Calories", help="Energy needed to maintain basic bodily functions at rest.")
    with col_res2:
        st.metric("Total Daily Energy Expenditure (TDEE)", f"{int(tdee)} Calories", help="Calories needed to maintain current weight, factoring in activity.")
    with col_res3:
        st.metric("Target Daily Calories", f"🔥 {int(daily_calories)} Calories")

    st.markdown("---")

    st.subheader("🎯 Daily Macronutrient Targets")
    col_macros1, col_macros2, col_macros3 = st.columns(3)

    total_calculated_calories = (protein_grams * CAL_PER_G_PROTEIN) + (fat_grams * CAL_PER_G_FAT) + (carb_grams * CAL_PER_G_CARB)
    
    protein_perc = (protein_grams * CAL_PER_G_PROTEIN / total_calculated_calories * 100) if total_calculated_calories > 0 else 0
    fat_perc = (fat_grams * CAL_PER_G_FAT / total_calculated_calories * 100) if total_calculated_calories > 0 else 0
    carb_perc = (carb_grams * CAL_PER_G_CARB / total_calculated_calories * 100) if total_calculated_calories > 0 else 0

    with col_macros1:
        st.metric("Protein", f"🥩 {int(protein_grams)} g", f"{int(protein_perc)}% of calories")
    with col_macros2:
        st.metric("Fat", f"🥑 {int(fat_grams)} g", f"{int(fat_perc)}% of calories")
    with col_macros3:
        st.metric("Carbohydrates", f"🍚 {int(carb_grams)} g", f"{int(carb_perc)}% of calories")

    st.markdown("---")

    st.subheader("⚖ Estimated Weight Change")
    weight_unit_display = "kg"
    if estimated_weight_change > 0:
        st.info(f"You are estimated to *gain {abs(estimated_weight_change):.2f} {weight_unit_display}* per week based on your calorie target. (Approx. {abs(estimated_weight_change)*2.20462:.2f} lbs)")
    elif estimated_weight_change < 0:
        st.info(f"You are estimated to *lose {abs(estimated_weight_change):.2f} {weight_unit_display}* per week based on your calorie target. (Approx. {abs(estimated_weight_change)*2.20462:.2f} lbs)")
    else:
        st.info("You are estimated to *maintain your weight*.")

    st.markdown("---")

    # --- Visualization of Macro Distribution (Ternary Plot) ---
    st.subheader("📊 Macronutrient Distribution Visualization (Ternary Plot)")
    st.write("This plot shows the relative proportions of your target protein, fat, and carbohydrates, allowing for easy comparison with common dietary splits.")

    current_macros_for_plot = np.array([protein_perc, fat_perc, carb_perc])
    current_macros_for_plot = current_macros_for_plot / np.sum(current_macros_for_plot) if np.sum(current_macros_for_plot) > 0 else np.array([0.33, 0.33, 0.34])


    fig_ternary = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': [current_macros_for_plot[0]], # Protein
        'b': [current_macros_for_plot[1]], # Fat
        'c': [current_macros_for_plot[2]], # Carbohydrates
        'marker': {
            'symbol': 'circle',
            'size': 18,
            'color': 'red',
            'line': {'width': 3, 'color': 'darkred'}
        },
        'hoverinfo': 'text',
        'text': [f"Your Target: P: {current_macros_for_plot[0]:.1%} | F: {current_macros_for_plot[1]:.1%} | C: {current_macros_for_plot[2]:.1%}"],
        'name': 'Your Target Macros'
    }))

    example_diets = {
        "Standard Balanced (25/30/45)": {"Protein": 0.25, "Fat": 0.30, "Carbohydrates": 0.45},
        "Low Carb / Keto-like (25/65/10)": {"Protein": 0.25, "Fat": 0.65, "Carbohydrates": 0.10},
        "High Protein (35/25/40)": {"Protein": 0.35, "Fat": 0.25, "Carbohydrates": 0.40},
        "Lower Fat (25/15/60)": {"Protein": 0.25, "Fat": 0.15, "Carbohydrates": 0.60},
        "Athlete (30/20/50)": {"Protein": 0.30, "Fat": 0.20, "Carbohydrates": 0.50}
    }

    colors = ['blue', 'green', 'purple', 'orange', 'grey']
    for i, (name, macros) in enumerate(example_diets.items()):
        fig_ternary.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': [macros['Protein']],
            'b': [macros['Fat']],
            'c': [macros['Carbohydrates']],
            'marker': {
                'symbol': 'square',
                'size': 12,
                'color': colors[i % len(colors)],
                'line': {'width': 1, 'color': 'black'}
            },
            'hoverinfo': 'text',
            'text': [f"{name}<br>P: {macros['Protein']:.1%} | F: {macros['Fat']:.1%} | C: {macros['Carbohydrates']:.1%}"],
            'name': name
        }))

    fig_ternary.update_layout(
        title="Your Macronutrient Target vs. Common Diet Types",
        ternary={
            'sum': 1,
            'aaxis': {'title': 'Protein (%)', 'min': 0, 'linewidth': 2, 'ticks': 'outside', 'title_font_size': 14},
            'baxis': {'title': 'Fat (%)', 'min': 0, 'linewidth': 2, 'ticks': 'outside', 'title_font_size': 14},
            'caxis': {'title': 'Carbohydrates (%)', 'min': 0, 'linewidth': 2, 'ticks': 'outside', 'title_font_size': 14}
        },
        height=600,
        width=800,
        showlegend=True,
        margin={'l': 0, 'r': 0, 'b': 0, 't': 40},
        legend=dict(x=1.05, y=0.5, xanchor='left', yanchor='middle')
    )
    st.plotly_chart(fig_ternary, use_container_width=True)

    st.write("""
    <div style="background-color:white;color:black; padding: 10px; border-radius: 5px; margin-top: 15px;">
    <p><b>Ternary Plot Explanation:</b> This specialized plot visualizes the proportions of three components (Protein, Fat, Carbohydrates) that sum to 100%. Your calculated macro target is shown as a large <span style="color:red; font-weight:bold;">red circle</span>, while common dietary splits are <span style="color:blue; font-weight:bold;">smaller squares</span>. It helps you quickly see how your desired diet balances these key macronutrients relative to others.</p>
    </div>
    """, unsafe_allow_html=True)


    # --- Food Recommendation Section ---
    if not df_food_data_clean_macros.empty:
        st.markdown("---")
        st.subheader("🍽 Daily Food Combination Recommendation")
        st.write("Here's a sample daily food combination from your dataset designed to help you hit your macronutrient targets. This is an illustrative example, and actual meal planning requires more detailed consideration.")

        # Call the recommendation function
        recommended_meals = recommend_food_combination(
            df_food_data_clean_macros,
            target_calories=daily_calories,
            target_protein=protein_grams,
            target_fat=fat_grams,
            target_carb=carb_grams,
            num_recommendations=1 # We want one main daily combination
        )

        if recommended_meals:
            # Display the first (and only) recommendation
            meal_summary = recommended_meals[0]
            
            with st.expander(f"Suggested Food Combination for {int(meal_summary['total_calories'])} Calories", expanded=True):
                st.write("*Overall Macronutrient Balance for this Combination:*")
                
                # Create a DataFrame for target vs. actual macros for the bar chart
                macro_comparison_df = pd.DataFrame({
                    'Macronutrient': ['Protein', 'Fat', 'Carbohydrates'],
                    'Target (g)': [protein_grams, fat_grams, carb_grams],
                    'Recommended (g)': [meal_summary['total_protein'], meal_summary['total_fat'], meal_summary['total_carb']]
                })

                fig_comparison = go.Figure()
                fig_comparison.add_trace(go.Bar(
                    x=macro_comparison_df['Macronutrient'],
                    y=macro_comparison_df['Target (g)'],
                    name='Target (g)',
                    marker_color='lightblue'
                ))
                fig_comparison.add_trace(go.Bar(
                    x=macro_comparison_df['Macronutrient'],
                    y=macro_comparison_df['Recommended (g)'],
                    name='Recommended (g)',
                    marker_color='lightcoral'
                ))

                fig_comparison.update_layout(
                    barmode='group',
                    title='Target vs. Recommended Macronutrients (Grams)',
                    xaxis_title='Macronutrient',
                    yaxis_title='Grams (g)',
                    height=400,
                    legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

                st.markdown("---")
                st.write("*Detailed Breakdown of Recommended Food Items:*")
                
                # Create a DataFrame for individual food item macros for the stacked bar chart
                rec_df_list = []
                for item in meal_summary['foods']:
                    rec_df_list.append({
                        'Food Item': item['name'],
                        'Calories': item['calories'],
                        'Protein (g)': item['protein'],
                        'Fat (g)': item['fat'],
                        'Carbohydrates (g)': item['carbohydrate'],
                        'Serving Info': item['serving_info']
                    })
                rec_df = pd.DataFrame(rec_df_list)
                
                # Sort by calories or protein to make the chart more readable
                rec_df = rec_df.sort_values(by='Calories', ascending=False).reset_index(drop=True)

                # Display the data in a table first
                st.dataframe(rec_df[['Food Item', 'Serving Info', 'Calories', 'Protein (g)', 'Fat (g)', 'Carbohydrates (g)']].round(1).set_index('Food Item'), use_container_width=True)


                # Melt the DataFrame for stacked bar chart visualization
                rec_df_melted = rec_df.melt(id_vars=['Food Item', 'Calories', 'Serving Info'], var_name='Macronutrient', value_name='Grams', 
                                            value_vars=['Protein (g)', 'Fat (g)', 'Carbohydrates (g)'])
                
                fig_rec_bar = px.bar(
                    rec_df_melted,
                    x='Food Item',
                    y='Grams',
                    color='Macronutrient',
                    title='Macronutrient Contribution Per Recommended Food Item',
                    labels={'Grams': 'Grams (g)'},
                    height=550,
                    hover_data={'Serving Info': True, 'Calories':':.0f'}
                )
                fig_rec_bar.update_layout(xaxis={'categoryorder':'total descending'}) # Order by total grams

                st.plotly_chart(fig_rec_bar, use_container_width=True)
                
                st.info(
                    "*Note:* This is a programmatic suggestion and may not represent a perfectly balanced or palatable 'meal'. "
                    "It's designed to show how different food items can contribute to your macro goals. "
                    "Consider it a starting point for building your daily menu! Adjust servings as needed."
                )
        else:
            st.warning("Could not generate a suitable food combination with the available data.")
    else:
        st.warning("Food combination recommendation skipped because 'nutrition_fullcleaned.xlsx' was not found or was empty.")

# --- Key Concepts and Detailed Explanations ---
st.subheader("How It Works")

# Diet Types
st.markdown("**🔹 Diet Types:**")
st.markdown("""
- **Standard**: Uses the [Mifflin-St. Jeor equation](https://en.wikipedia.org/wiki/Basal_metabolic_rate#Mifflin-St._Jeor_equation), ideal for general weight goals when body fat % is unknown.
- **Leangains**: Based on [The Leangains Method](https://leangains.com/the-leangains-method-new-book/), incorporates muscle mass and activity to optimize lean bulking or cutting.
- **Keto**: Tailored for ketogenic diets; lets you fix carbs and protein, and fills the rest with fat.
""")

# Key Terms
st.markdown("**🔹 Key Terms:**")
st.markdown("""
- **BMR (Basal Metabolic Rate)**: Calories your body burns at complete rest (basic functions).
- **TDEE (Total Daily Energy Expenditure)**: Maintenance calories based on total daily activity.
- **Bulking**: Calorie surplus to gain weight, preferably muscle.
- **Cutting**: Calorie deficit to reduce fat while preserving muscle.
- **Maintenance**: Eating at TDEE to keep your current weight stable.
""")

# Activity Level
st.markdown("**🔹 Activity Level Considerations:**")
st.write("Overestimation is common. Non-gym daily movement defines your level more than workouts.")

# Protein Intake
st.markdown("**🔹 Protein Intake:**")
st.write("Essential for muscle building and fat loss. High intake improves satiety and thermic effect.")

# Fat/Carb Split
st.markdown("**🔹 Fat/Carb Split:**")
st.write("Adjust based on personal preference. Ensure at least 0.25g fat/lb of body weight.")

# Weight Change
st.markdown("**🔹 Estimated Weight Change:**")
st.write("Based on ~7700 kcal/kg fat (≈3500 kcal/lb). Early changes may include water weight.")

# Disclaimer
st.markdown("**⚠️ Disclaimer:**")
st.info("This tool is for educational use only. Consult a professional for personalized guidance.")