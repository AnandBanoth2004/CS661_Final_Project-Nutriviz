import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import random
from sklearn.metrics.pairwise import euclidean_distances


st.set_page_config(page_title="NutriViz - Nutrient Explorer", layout="wide")
st.title("Nutrient Correlation Explorer")

@st.cache_data
def load_data():
    df = pd.read_excel("data/nutrition_fullcleaned.xlsx")

    # Drop unwanted columns
    nutrient_cols = df.columns[3:]

    # Extract units from first row
    def extract_unit(value):
        if isinstance(value, str):
            match = re.search(r'[a-zA-Z]+', value)
            return match.group(0) if match else ''
        return ''

    unit_map = {
        col: extract_unit(df.iloc[0][col])
        for col in nutrient_cols
    }

    # Strip units from the values to make them numeric
    def to_numeric(val):
        if isinstance(val, str):
            num = re.findall(r"[-+]?\d*\.\d+|\d+", val)
            return float(num[0]) if num else 0.0
        return val

    for col in nutrient_cols:
        df[col] = df[col].apply(to_numeric)

    categories_priority = [
        ("Meats, Poultry, and Fish", [r'\bmeat\b', r'chicken', r'beef', r'pork', r'turkey', r'bacon', r'sausage', r'steak', r'ham', r'lamb']),
        ("Beverages", [r'\bbeverage\b', r'drink\b', r'soda\b', r'coffee\b', r'tea\b', r'water\b', r'energy drink']),
        ("Sweets and Candies", [r'\bcandy\b', r'chocolate', r'sweet\b', r'caramel', r'fudge', r'lollipop', r'gummy']),
        ("Fats and Oils", [r'\bfat\b', r'\boil\b', r'olive oil', r'vegetable oil', r'margarine', r'shortening']),
        ("Dairy and Egg Products", [r'\bdairy\b', r'\bmilk\b', r'yogurt', r'cream', r'egg\b', r'curd', r'butter\b', r'custard', r'ice cream']),
        ("Prepared Meals and Fast Food", [r'meal\b', r'fast food', r'burger', r'pizza', r'lasagna', r'sandwich\b', r'hot dog']),
        ("Baked Goods", [r'\bbread\b', r'\bcake\b', r'\bcookie\b', r'\bbiscuit\b', r'\bmuffin\b', r'croissant', r'bagel', r'pastry', r'pie\b']),
        ("Infant Foods", [r'\binfant\b', r'baby food', r'formula\b', r'infant cereal']),
        ("Fruits and Fruit Juices", [r'\bfruit\b', r'apple', r'banana', r'orange', r'berry', r'grape', r'mango', r'juice\b', r'melon']),
        ("Vegetables and Vegetable Products", [r'\bvegetable\b', r'potato', r'tomato', r'carrot', r'broccoli', r'spinach', r'cabbage', r'lettuce', r'onion']),
        ("Legumes and Legume Products", [r'\blegume\b', r'bean\b', r'soy', r'tofu', r'hummus', r'lentil', r'chickpea']),
        ("Nuts and Seeds", [r'\bnut\b', r'almond', r'walnut', r'peanut', r'seed\b', r'pistachio', r'cashew', r'sunflower seed']),
        ("Grains and Pasta", [r'\bgrain\b', r'rice\b', r'pasta\b', r'spaghetti', r'macaroni', r'quinoa', r'barley', r'oat\b']),
        ("Sauces, Dips, and Gravies", [r'\bsauce\b', r'\bdip\b', r'gravy\b', r'ketchup', r'mayonnaise', r'salsa', r'ranch']),
        ("Soups and Stews", [r'\bsoup\b', r'\bstew\b', r'broth', r'chowder', r'bisque']),
        ("Snacks", [r'\bsnack\b', r'chips', r'crackers', r'popcorn', r'pretzel', r'trail mix']),
        ("Spices and Herbs", [r'\bspice\b', r'\bherb\b', r'pepper\b', r'cinnamon', r'basil', r'oregano', r'cumin']),
        ("Cereals and Breakfast Foods", [r'\bcereal\b', r'oatmeal', r'granola', r'\bporridge\b', r'corn flakes', r'muesli']),
        ("Cheese", [r'\bcheese\b', r'cheddar', r'mozzarella', r'parmesan', r'gouda', r'brie', r'feta']),
        ("Seafood", [r'\bfish\b', r'seafood', r'salmon', r'tuna', r'shrimp', r'prawn', r'crab', r'lobster', r'cod', r'tilapia']),
        ("Baking Ingredients", [r'baking', r'flour\b', r'yeast', r'baking powder', r'vanilla extract', r'cocoa powder']),
        ("Other", [])
    ]

    def assign_category(name):
        for category, patterns in categories_priority:
            for pattern in patterns:
                if re.search(pattern, str(name), re.IGNORECASE):
                    return category
        return "Other"

    df['category'] = df['name'].apply(assign_category)
    df = df.dropna()
    return df, nutrient_cols.to_list(), unit_map

# Load data
df, nutrient_cols, unit_map = load_data()
categories = sorted(df['category'].unique())

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Correlation Matrix", "Scatter Plot", "Summary Statistics", "Nutrition Similarity Explorer", "Risky Nutrient Pair Flags"])

# Sidebar filters
st.sidebar.header("Filters")
selected_category = st.sidebar.selectbox("Filter by Category", ["All"] + categories)
st.sidebar.markdown("### Scatter Plot Settings")

# Quick Templates FIRST (before dropdowns)
template_pairs = {
    "Fiber vs Sugar": ("fiber", "sugars"),
    "Calcium vs Vitamin D": ("calcium", "vitamin_d"),
    "Protein vs Fat": ("protein", "total_fat"),
    "Sodium vs Potassium": ("sodium", "potassium"),
}

template = st.sidebar.radio("Quick Templates", ["None"] + list(template_pairs.keys()), key="template")

# Initialize on first load
if "xaxis" not in st.session_state or "yaxis" not in st.session_state:
    st.session_state.xaxis, st.session_state.yaxis = random.sample(nutrient_cols, 2)

# Template override
if template != "None" and template in template_pairs:
    st.session_state.xaxis, st.session_state.yaxis = template_pairs[template]

# Filter + valid nutrients
data = df if selected_category == "All" else df[df['category'] == selected_category]
valid_nutrients = [col for col in nutrient_cols if data[col].nunique(dropna=True) > 1]

# Sidebar dropdowns
tab_x, tab_y = st.sidebar.columns(2)
x_axis = tab_x.selectbox("X-axis", sorted(valid_nutrients), index=sorted(valid_nutrients).index(st.session_state.xaxis), help="Search nutrient")
y_axis = tab_y.selectbox("Y-axis", sorted(valid_nutrients), index=sorted(valid_nutrients).index(st.session_state.yaxis), help="Search nutrient")

# Update session state
st.session_state.xaxis = x_axis
st.session_state.yaxis = y_axis

# Button style
st.sidebar.markdown("""
    <style>
        div.stButton > button {
            background-color: #0e1117;
            color: white;
            width: 80%;
            margin-left: 10%;
        }
    </style>
""", unsafe_allow_html=True)

if st.sidebar.button("Randomize Axes"):
    st.session_state.xaxis, st.session_state.yaxis = random.sample(valid_nutrients, 2)
    st.rerun()

# Quadrant setting last
st.sidebar.markdown("#### Advanced Options")
stat_split = st.sidebar.radio("Quadrant split by:", ["Mean", "Median"])

# Tab 1: Correlation Matrix
with tab1:
    st.subheader("Nutrient Correlation Matrix")

    numeric_data = data[nutrient_cols]
    valid_cols = numeric_data.loc[:, numeric_data.nunique(dropna=True) > 1]

    dropped_cols = set(nutrient_cols) - set(valid_cols.columns)
    # if dropped_cols:
    #     st.warning(f"⚠️ The following nutrients were excluded due to no variation: {', '.join(sorted(dropped_cols))}")

    if valid_cols.shape[1] < 2:
        st.error("❌ Not enough nutrients with variation to show correlation matrix.")
    else:
        corr = valid_cols.corr()
        fig_corr = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)

# Tab 2: Scatter Plot with Quadrants
with tab2:
    st.subheader(f"{x_axis.title()} vs {y_axis.title()} Scatter Plot")
    x_data = data[x_axis]
    y_data = data[y_axis]

    x_split = x_data.mean() if stat_split == "Mean" else x_data.median()
    y_split = y_data.mean() if stat_split == "Mean" else y_data.median()
    
    # Format axis labels with units
    x_unit = unit_map.get(x_axis, '')
    y_unit = unit_map.get(y_axis, '')

    x_label = f"{x_axis} ({x_unit})" if x_unit else x_axis
    y_label = f"{y_axis} ({y_unit})" if y_unit else y_axis

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        mode='markers',
        marker=dict(color='teal', opacity=0.6, size=6),
        text=data['name'],
        hovertemplate=f"<b>%{{text}}</b><br>{x_axis.title()}: %{{x}}{x_unit}<br>{y_axis.title()}: %{{y}}{y_unit}<extra></extra>"
    ))

    # Red dotted lines for quadrant split
    fig.add_shape(type="line", x0=x_split, x1=x_split, y0=y_data.min(), y1=y_data.max(), line=dict(color="red", dash="dot"))
    fig.add_shape(type="line", y0=y_split, y1=y_split, x0=x_data.min(), x1=x_data.max(), line=dict(color="red", dash="dot"))

    # Quadrant Annotations
    dx = 0.2 * (x_data.max() - x_data.min())
    dy = 0.2 * (y_data.max() - y_data.min())
    font = dict(size=16, color='white')

    fig.add_annotation(
        text=f"↑ High {y_axis.title()}<br>→ High {x_axis.title()}",
        x=x_split + dx, y=y_split + dy,
        showarrow=False, font=font
    )
    fig.add_annotation(
        text=f"↑ High {y_axis.title()}<br>← Low {x_axis.title()}",
        x=x_split - dx, y=y_split + dy,
        showarrow=False, font=font
    )
    fig.add_annotation(
        text=f"↓ Low {y_axis.title()}<br>← Low {x_axis.title()}",
        x=x_split - dx, y=y_split - dy,
        showarrow=False, font=font
    )
    fig.add_annotation(
        text=f"↓ Low {y_axis.title()}<br>→ High {x_axis.title()}",
        x=x_split + dx, y=y_split - dy,
        showarrow=False, font=font
    )


    fig.update_layout(
        title=f"{x_axis.title()} vs {y_axis.title()}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("_All nutrient values are per 100g of food._")

    st.markdown(f"**Pearson Correlation:** `{corr.loc[x_axis, y_axis]:.2f}`")

# Tab 3: Summary Statistics
with tab3:
    st.header("Summary Statistics by Category")
    selected_stat = st.selectbox("Choose a Nutrient", nutrient_cols)
    summary_df = df.groupby("category")[selected_stat].agg(["mean", "median", "min", "max"]).reset_index()
    st.dataframe(summary_df.style.background_gradient(axis=0, cmap='YlGnBu'), use_container_width=True)

    fig_bar = px.bar(summary_df, x="category", y="mean", title=f"Average {selected_stat.title()} by Category")
    fig_bar.update_layout(
        xaxis_tickangle=-45,  # Rotate to start from bottom-left visually
        xaxis_tickfont=dict(size=11),
        xaxis_ticklabelposition="outside bottom",  # Valid option
        margin=dict(b=120)  # Extra space for angled labels if needed
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# -------------------- Tab 4 --------------------
with tab4:
    st.header("Nutrition Similarity Explorer")
    st.write("Find foods with similar nutrient profiles to a selected item.")

    available_foods = data['name'].tolist()
    selected_food = st.selectbox("Select a Food Item", available_foods)

    similarity_nutrients = st.multiselect(
        "Nutrients for Similarity Calculation",
        nutrient_cols,
        default=['protein', 'total_fat', 'fiber', 'sugars', 'calcium', 'vitamin_d'],
        key="sim_nutrients"
    )

    if selected_food and similarity_nutrients:
        similarity_df = data[similarity_nutrients + ['name']].set_index('name').dropna()

        if selected_food not in similarity_df.index:
            st.warning(f"Nutrient data for '{selected_food}' is incomplete for selected similarity nutrients.")
        elif not similarity_df.empty:
            normalized_df = (similarity_df - similarity_df.mean()) / (similarity_df.std() + 1e-6)
            food_vector = normalized_df.loc[[selected_food]]
            other_foods_vectors = normalized_df.drop(index=selected_food)

            distances = euclidean_distances(food_vector, other_foods_vectors)
            distances_series = pd.Series(distances[0], index=other_foods_vectors.index)
            similar_foods = distances_series.nsmallest(5)

            st.markdown(f"### Foods Similar to *{selected_food}*:")
            similar_foods_list = []
            for food_name, distance in similar_foods.items():
                food_row = data[data['name'] == food_name].iloc[0]
                base_row = data[data['name'] == selected_food].iloc[0]
                tags = []
                if 'sugars' in food_row and food_row['sugars'] < base_row['sugars']:
                    tags.append("Less Sugar")
                if 'saturated_fat' in food_row and food_row['saturated_fat'] < base_row['saturated_fat']:
                    tags.append("Less Saturated Fat")
                tag_str = f" ({', '.join(tags)})" if tags else ""
                similar_foods_list.append(f"- *{food_name}* (Distance: {distance:.2f}){tag_str}")

            st.markdown("\n".join(similar_foods_list))
        else:
            st.info("Not enough data for similarity calculation.")
    else:
        st.info("Please select a food and at least one nutrient.")

# -------------------- Tab 5 --------------------
with tab5:
    st.header("Risky Nutrient Pair Flags")
    st.write("Set custom thresholds to identify foods with 'risky' nutrient profiles.")

    col1, col2 = st.columns(2)
    thresholds = {}

    with col1:
        st.subheader("Maximum Values (Risk if above)")
        for nutrient in nutrient_cols:
            max_val = st.number_input(f"Max {nutrient.replace('_', ' ').title()} ({unit_map.get(nutrient, '')})",
                                      value=float(data[nutrient].max() * 0.75),
                                      format="%.2f", key=f"max_{nutrient}")
            thresholds[f"max_{nutrient}"] = max_val

    with col2:
        st.subheader("Minimum Values (Risk if below)")
        for nutrient in nutrient_cols:
            min_val = st.number_input(f"Min {nutrient.replace('_', ' ').title()} ({unit_map.get(nutrient, '')})",
                                      value=float(data[nutrient].min() * 1.25),
                                      format="%.2f", key=f"min_{nutrient}")
            thresholds[f"min_{nutrient}"] = min_val

    risky_foods = []
    for idx, row in data.iterrows():
        warnings = []
        for nutrient in nutrient_cols:
            if row[nutrient] > thresholds[f"max_{nutrient}"]:
                warnings.append(f"High {nutrient.replace('_', ' ').title()} ({row[nutrient]:.2f}{unit_map.get(nutrient, '')})")
            if row[nutrient] < thresholds[f"min_{nutrient}"]:
                warnings.append(f"Low {nutrient.replace('_', ' ').title()} ({row[nutrient]:.2f}{unit_map.get(nutrient, '')})")
        if warnings:
            risky_foods.append({"Food Name": row['name'], "Warnings": "; ".join(warnings)})

    if risky_foods:
        st.dataframe(pd.DataFrame(risky_foods), use_container_width=True)

        st.markdown("### Visualize Risky Foods on Scatter Plot")
        plot_risky_x = st.selectbox("X-axis for Risky Plot", nutrient_cols, key="risky_x")
        plot_risky_y = st.selectbox("Y-axis for Risky Plot", nutrient_cols, key="risky_y")

        plot_data = data.copy()
        plot_data['is_risky'] = False
        plot_data['warnings'] = ""

        for idx, row in plot_data.iterrows():
            current_warnings = []
            for nutrient in nutrient_cols:
                if row[nutrient] > thresholds[f"max_{nutrient}"]:
                    current_warnings.append(f"High {nutrient.replace('_', ' ').title()}")
                if row[nutrient] < thresholds[f"min_{nutrient}"]:
                    current_warnings.append(f"Low {nutrient.replace('_', ' ').title()}")
            if current_warnings:
                plot_data.at[idx, 'is_risky'] = True
                plot_data.at[idx, 'warnings'] = "; ".join(current_warnings)

        fig_risky = px.scatter(
            plot_data, x=plot_risky_x, y=plot_risky_y, color='is_risky',
            hover_name='name', custom_data=['warnings'],
            symbol='is_risky', color_discrete_map={True: 'red', False: 'blue'},
            title=f"Foods Flagged as Risky: {plot_risky_x.title()} vs {plot_risky_y.title()}"
        )
        fig_risky.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                          f'{plot_risky_x.title()}: %{{x}}{unit_map.get(plot_risky_x, "")}<br>' +
                          f'{plot_risky_y.title()}: %{{y}}{unit_map.get(plot_risky_y, "")}<br>' +
                          'Warnings: %{customdata}<extra></extra>'
        )
        fig_risky.update_layout(
            xaxis_title=f"{plot_risky_x.title()} ({unit_map.get(plot_risky_x, '')})",
            yaxis_title=f"{plot_risky_y.title()} ({unit_map.get(plot_risky_y, '')})"
        )
        st.plotly_chart(fig_risky, use_container_width=True)
    else:
        st.info("No foods meet the risky criteria based on thresholds.")

st.markdown("---")
st.caption("Team NutriViz | Nutrient Correlation Explorer")
