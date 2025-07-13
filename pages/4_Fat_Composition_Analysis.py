import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re  # Needed for regex pattern matching in categorization
import plotly.colors  # For color generation

# Page configuration
st.set_page_config(
    page_title="Fat Composition Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and preprocess data with enhanced categorization
@st.cache_data
def load_data():
    df = pd.read_excel('data/nutrition_fullcleaned.xlsx')
    
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
        ("Fruits and Fruit Juices", [r'\bfruit\b', r'apple', r'banana', r'orange', r'berry', r'grape', r'mango', r'juice\b', r'melon']),
        ("Vegetables and Vegetable Products", [r'\bvegetable\b', r'potato', r'tomato', r'carrot', r'broccoli', r'spinach', r'cabbage', r'lettuce', r'onion']),
        ("Dairy and Egg Products", [r'\bdairy\b', r'\bmilk\b', r'yogurt', r'cream', r'egg\b', r'curd', r'butter\b', r'custard', r'ice cream']),
        ("Legumes and Legume Products", [r'\blegume\b', r'bean\b', r'soy', r'tofu', r'hummus', r'lentil', r'chickpea']),
        ("Nuts and Seeds", [r'\bnut\b', r'almond', r'walnut', r'peanut', r'seed\b', r'pistachio', r'cashew', r'sunflower seed']),
        ("Grains and Pasta", [r'\bgrain\b', r'rice\b', r'pasta\b', r'spaghetti', r'macaroni', r'quinoa', r'barley', r'oat\b']),
        ("Sweets and Candies", [r'\bcandy\b', r'chocolate', r'sweet\b', r'caramel', r'fudge', r'lollipop', r'gummy']),
        ("Infant Foods", [r'\binfant\b', r'baby food', r'formula\b', r'infant cereal']),
        ("Prepared Meals and Fast Food", [r'meal\b', r'fast food', r'burger', r'pizza', r'lasagna', r'sandwich\b', r'hot dog']),
        ("Sauces, Dips, and Gravies", [r'\bsauce\b', r'\bdip\b', r'gravy\b', r'ketchup', r'mayonnaise', r'salsa', r'ranch']),
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

df = load_data()

# Page title
st.title("Fat Composition Analysis")
st.markdown("""
This section analyzes the breakdown of different fat types across food categories and their relationship with cholesterol levels.
""")

# Section 1: Fat Composition by Category
st.header("1. Fat Composition Breakdown by Food Category")

if 'category' in df.columns and all(col in df.columns for col in ['saturated_fat', 'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids']):
    # Aggregate data by category
    fat_composition = df.groupby('category').agg({
        'saturated_fat': 'mean',
        'monounsaturated_fatty_acids': 'mean',
        'polyunsaturated_fatty_acids': 'mean'
    }).reset_index()
    
    # Melt for stacked bar chart
    melted_fat = fat_composition.melt(
        id_vars='category', 
        value_vars=['saturated_fat', 'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids'],
        var_name='fat_type', 
        value_name='amount'
    )
    
    # Create visualization
    fig = px.bar(
        melted_fat,
        x='category',
        y='amount',
        color='fat_type',
        labels={'amount': 'Average Amount (g per 100g)', 'category': 'Food Category'},
        title='Average Fat Composition by Food Category (Stacked Bar Chart)',
        barmode='stack',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Required columns for fat composition analysis are missing")

# Section 2: Fat-Cholesterol Correlation
st.header("2. Fat Types vs Cholesterol Levels")

if 'cholesterol' in df.columns and all(col in df.columns for col in ['saturated_fat', 'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids']):
    # Create subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        'Saturated Fat vs Cholesterol', 
        'Monounsaturated Fat vs Cholesterol',
        'Polyunsaturated Fat vs Cholesterol'
    ))
    
    # Add scatter plots
    fig.add_trace(
        go.Scatter(
            x=df['saturated_fat'],
            y=df['cholesterol'],
            mode='markers',
            name='Saturated',
            marker=dict(color='#FF4B4B', opacity=0.5)
        ), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['monounsaturated_fatty_acids'],
            y=df['cholesterol'],
            mode='markers',
            name='Monounsaturated',
            marker=dict(color='#42D778', opacity=0.5)
        ), row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['polyunsaturated_fatty_acids'],
            y=df['cholesterol'],
            mode='markers',
            name='Polyunsaturated',
            marker=dict(color='#1A73E8', opacity=0.5)
        ), row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="Relationship Between Fat Types and Cholesterol (Scatter Plots)"
    )
    
    # Add axis labels
    fig.update_xaxes(title_text="Saturated Fat (g)", row=1, col=1)
    fig.update_xaxes(title_text="Monounsaturated Fat (g)", row=1, col=2)
    fig.update_xaxes(title_text="Polyunsaturated Fat (g)", row=1, col=3)
    fig.update_yaxes(title_text="Cholesterol (mg)", row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display correlations
    sat_corr = df['saturated_fat'].corr(df['cholesterol'])
    mono_corr = df['monounsaturated_fatty_acids'].corr(df['cholesterol'])
    poly_corr = df['polyunsaturated_fatty_acids'].corr(df['cholesterol'])
    
    st.markdown(f"""
    **Correlation Coefficients:**
    - Saturated Fat vs Cholesterol: `{sat_corr:.2f}`
    - Monounsaturated Fat vs Cholesterol: `{mono_corr:.2f}`
    - Polyunsaturated Fat vs Cholesterol: `{poly_corr:.2f}`

    **Interpretation Guide:**
    - Values range from `-1.0` to `+1.0`
    - **Positive values** = cholesterol increases as fat increases
    - **Negative values** = cholesterol decreases as fat increases
    - **Values near 0** = no linear relationship
    """)
else:
    st.warning("Cholesterol or fat composition columns missing in dataset")

# Section 3: Sankey Diagram with Enhanced Clarity
st.header("3. Fat Composition Flow (Sankey Diagram)")

if 'category' in df.columns and all(col in df.columns for col in ['saturated_fat', 'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids']):
    # Aggregate and sort data by total fat (descending)
    category_fats = df.groupby('category').agg({
        'saturated_fat': 'sum',
        'monounsaturated_fatty_acids': 'sum',
        'polyunsaturated_fatty_acids': 'sum'
    }).reset_index()
    
    # Calculate total fat and sort categories
    category_fats['total_fat'] = category_fats[['saturated_fat', 'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids']].sum(axis=1)
    category_fats = category_fats.sort_values('total_fat', ascending=False)
    
    # Prepare Sankey data
    categories = category_fats['category'].tolist()
    fat_types = ['Saturated', 'Monounsaturated', 'Polyunsaturated']
    all_nodes = categories + fat_types
    
    # Generate distinct colors for categories
    n_categories = len(categories)
    category_colors = plotly.colors.qualitative.Plotly
    if n_categories > len(category_colors):
        category_colors = plotly.colors.qualitative.Alphabet * (
            n_categories // len(plotly.colors.qualitative.Alphabet) + 1
        )
    category_colors = category_colors[:n_categories]
    
    # Create color mapping
    category_color_map = dict(zip(categories, category_colors))
    
    # Assign fixed colors to fat types
    fat_type_colors = {
        'Saturated': '#FF4B4B',
        'Monounsaturated': '#42D778',
        'Polyunsaturated': '#1A73E8'
    }
    
    # Prepare node colors
    node_colors = [category_color_map[cat] for cat in categories] + [fat_type_colors[ft] for ft in fat_types]
    
    # Create links
    sources = []
    targets = []
    values = []
    link_colors = []
    
    # Helper function to convert hex to rgba
    def hex_to_rgba(hex_color, alpha=0.4):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    
    for _, row in category_fats.iterrows():
        category = row['category']
        cat_idx = categories.index(category)
        
        # Category to Saturated
        sources.append(cat_idx)
        targets.append(len(categories) + 0)
        values.append(row['saturated_fat'])
        link_colors.append(hex_to_rgba(category_color_map[category]))
        
        # Category to Monounsaturated
        sources.append(cat_idx)
        targets.append(len(categories) + 1)
        values.append(row['monounsaturated_fatty_acids'])
        link_colors.append(hex_to_rgba(category_color_map[category]))
        
        # Category to Polyunsaturated
        sources.append(cat_idx)
        targets.append(len(categories) + 2)
        values.append(row['polyunsaturated_fatty_acids'])
        link_colors.append(hex_to_rgba(category_color_map[category]))
    
    # Create Sankey diagram with enhanced clarity
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25,  # Increased padding for better spacing
            thickness=25,  # Slightly thicker nodes
            line=dict(color="rgba(0,0,0,0.5)", width=0.8),  # Softer border
            label=all_nodes,
            color=node_colors,
            # Removed font property from here
            hovertemplate='%{label}<br>Total Fat: %{value:.1f}g<extra></extra>',
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='From %{source.label}<br>To %{target.label}<br>Amount: %{value:.1f}g<extra></extra>'
        )
    )])
    
    # CORRECTED FONT SETTINGS: Applied to layout instead of node
    fig.update_layout(
        title_text="Fat Composition Flow from Food Categories",
        height=700,  # Increased height for better proportions
        font=dict(  # Set font properties for all text
            family='Arial, sans-serif',
            size=14,
            color='#333333'
        ),
        title_font=dict(
            size=18,
            family='Arial, sans-serif'
        ),
        margin=dict(t=80, b=20, l=20, r=20)  # Better margin utilization
    )
    
    # Display with high quality rendering
    st.plotly_chart(fig, use_container_width=True, config={
        'toImageButtonOptions': {
            'format': 'png',  # Download format
            'filename': 'sankey_diagram',
            'scale': 3  # High-resolution export (3x quality)
        }
    })
    
    # Add explanation
    st.markdown("""
    **Sankey Diagram Interpretation:**
    - **Nodes on the left** represent food categories (sorted by total fat content)
    - **Nodes on the right** represent fat types:
      - <span style="color:#FF4B4B">**Saturated**</span>
      - <span style="color:#42D778">**Monounsaturated**</span>
      - <span style="color:#1A73E8">**Polyunsaturated**</span>
    - **Flow width** shows the total amount of each fat type coming from each food category
    - **Hover** over elements to see detailed values
    - **Export** using the camera icon for high-resolution versions
    """, unsafe_allow_html=True)
else:
    st.warning("Required columns for Sankey diagram are missing")

# Section 4: Nutrient Correlation Matrix
st.header("4. Nutrient Correlation Exploration")

# Define nutrient columns we want to analyze
nutrient_cols = [
    'saturated_fat', 'monounsaturated_fatty_acids', 
    'polyunsaturated_fatty_acids', 'cholesterol',
    'total_fat', 'sodium', 'fiber', 'sugars'
]

# Filter to available nutrient columns
available_nutrients = [col for col in nutrient_cols if col in df.columns]

if len(available_nutrients) > 1:
    try:
        # Calculate correlations
        corr_matrix = df[available_nutrients].corr().round(2)
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Nutrient", y="Nutrient", color="Correlation"),
            x=available_nutrients,
            y=available_nutrients,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            range_color=[-1, 1],
            title="Nutrient Correlation Matrix"
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error calculating correlations: {str(e)}")
        st.warning("Could not generate correlation matrix due to computational limitations")
else:
    st.warning("Insufficient nutrient columns available for correlation matrix")


# Key Insights section
st.header("Key Insights")
st.write("""
**Fat Composition Breakdown**: Reveals which food categories have the highest proportions of saturated vs unsaturated fats.  
**Fat-Cholesterol Correlation**: Shows the strength of relationship between different fat types and cholesterol levels.  
**Sankey Diagram**: Visualizes the flow of different fat types from food categories to fat composition.  
**Nutrient Correlation**: Identifies positive/negative relationships between fats and other nutrients.  
""")