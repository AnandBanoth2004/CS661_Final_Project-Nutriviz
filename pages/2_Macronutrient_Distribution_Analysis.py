import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import re
import plotly.io as pio
pio.renderers.default = "notebook" 

# Load and preprocess data
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

# Load data
df = load_data()

st.set_page_config(layout="wide")
st.header("Macronutrient Distribution Analysis")

# Sidebar for category selection
st.sidebar.header("Filters")
all_categories = sorted(df['category'].unique())
selected_categories = st.sidebar.multiselect(
    "Select Food Categories", 
    all_categories,
    default=all_categories[:5]  # First 5 categories by default
)

# Filter data based on selected categories
filtered_df = df[df['category'].isin(selected_categories)]

# Section 1: Box and Violin Plots
st.subheader("Macronutrient Distribution Across Categories")
st.write("Compare protein, fat, and carbohydrate distributions using box and violin plots")

# Create tabs for different macronutrients
tab1, tab2, tab3 = st.tabs(["Protein", "Fat", "Carbohydrates"])

with tab1:
    fig1 = px.box(filtered_df, x='category', y='protein', 
                 title="Protein Distribution by Category(Box plot)")
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.violin(filtered_df, x='category', y='protein', 
                    title="Protein Distribution by Category(Violin Plot)")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    fig3 = px.box(filtered_df, x='category', y='fat', 
                 title="Fat Distribution by Category(Box plot)")
    st.plotly_chart(fig3, use_container_width=True)
    
    fig4 = px.violin(filtered_df, x='category', y='fat', 
                    title="Fat Distribution by Category(Violin Plot)")
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    fig5 = px.box(filtered_df, x='category', y='carbohydrate', 
                 title="Carbohydrate Distribution by Category(Box plot)")
    st.plotly_chart(fig5, use_container_width=True)
    
    fig6 = px.violin(filtered_df, x='category', y='carbohydrate', 
                    title="Carbohydrate Distribution by Category(Violin Plot)")
    st.plotly_chart(fig6, use_container_width=True)

# Section 2: Radar Charts
st.subheader("Macronutrient Composition by Category")
st.write("Showing average macronutrient profiles for each food category (Radar chart)")

# Calculate average macronutrients per category
avg_macros = filtered_df.groupby('category')[['protein', 'fat', 'carbohydrate']].mean().reset_index()

# Create radar chart
categories = avg_macros['category'].unique()
fig = go.Figure()

for category in categories:
    cat_data = avg_macros[avg_macros['category'] == category]
    fig.add_trace(go.Scatterpolar(
        r=cat_data[['protein', 'fat', 'carbohydrate']].values[0],
        theta=['Protein', 'Fat', 'Carbohydrates'],
        fill='toself',
        name=category
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, avg_macros[['protein', 'fat', 'carbohydrate']].values.max() * 1.1])),
    showlegend=True,
    height=600,
    title="Average Macronutrient Profile by Food Category"
)
st.plotly_chart(fig, use_container_width=True)

# Section 3: Amino Acid Treemap
st.subheader("Amino Acid Profile Visualization")
st.write("Showing amino acid composition for selected foods (Treemap)")

# Amino acid columns
amino_acids = ['alanine', 'arginine', 'aspartic_acid', 'cystine', 
               'glutamic_acid', 'glycine', 'histidine', 'isoleucine', 
               'leucine', 'lysine', 'methionine', 'phenylalanine', 
               'proline', 'serine', 'threonine', 'tryptophan', 
               'tyrosine', 'valine']

# Select food items
selected_foods = st.multiselect(
    "Select Foods for Amino Acid Profile", 
    filtered_df['name'], 
    default=filtered_df['name'][:3].tolist(),
    key="amino_acid_select"
)

if selected_foods:
    amino_df = filtered_df[filtered_df['name'].isin(selected_foods)][['name'] + amino_acids]
    amino_melted = amino_df.melt(id_vars='name', var_name='amino_acid', value_name='value')
    
    fig = px.treemap(
        amino_melted,
        path=['name', 'amino_acid'],
        values='value',
        color='amino_acid',
        height=700,
        title="Amino Acid Composition"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please select at least one food item")

# Section 4: Macronutrients vs. Energy Content (3D Visualization)
st.subheader("3D Macronutrients vs. Energy Content")
#st.write("Explore relationships between macronutrients and calories")

# Create scaled values
scatter_df = filtered_df.copy()
scatter_df['size'] = scatter_df['calories'].apply(lambda x: min(30, max(5, x/50)))  # Size scaling

# Create 3D scatter plot with plasma color scale
fig_3d = px.scatter_3d(
    scatter_df,
    x='protein',
    y='fat',
    z='carbohydrate',
    color='calories',
    size='size',
    hover_name='name',
    hover_data=['category', 'calories'],
    title="3D Macronutrient Relationship: Protein vs Fat vs Carbs (3D Scatter Plot)",
    labels={
        'protein': 'Protein (g)',
        'fat': 'Fat (g)',
        'carbohydrate': 'Carbs (g)',
        'calories': 'Calories'
    },
    height=700,
    color_continuous_scale='plasma'  # <--- KEY CHANGE HERE
)

# Enhance 3D plot appearance
fig_3d.update_layout(
    scene=dict(
        xaxis_title='<b>Protein</b> (g)',
        yaxis_title='<b>Fat</b> (g)',
        zaxis_title='<b>Carbohydrates</b> (g)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # Better default view
    ),
    coloraxis_colorbar=dict(title='Calories'),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Display 3D plot
st.plotly_chart(fig_3d, use_container_width=True)

# Section 5: t-SNE Visualization
st.subheader("t-SNE: Macronutrient Similarity Visualization with Clustering")
st.write("2D representation of food similarity based on macronutrient profiles, now with identified clusters.")

# Add sliders for t-SNE perplexity and number of clusters
perplexity_value = st.sidebar.slider(
    "Select t-SNE Perplexity",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
    help="Perplexity relates to the number of nearest neighbors t-SNE considers. Adjusting it can reveal different aspects of the data structure. A typical range is 5 to 50, but it can be adjusted based on dataset size."
)

n_clusters = st.sidebar.slider(
    "Number of Clusters (K-Means)",
    min_value=2,
    max_value=10,
    value=5,
    step=1,
    help="The number of clusters K-Means will attempt to find in the data. Adjust this to see different granularities of grouping."
)


# Prepare data
macronutrients = ['protein', 'fat', 'carbohydrate']
tsne_sample = filtered_df.dropna(subset=macronutrients).sample(min(500, len(filtered_df)), random_state=42)

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(tsne_sample[macronutrients])

# Perform t-SNE with user-selected perplexity
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
tsne_results = tsne.fit_transform(X)

# Create DataFrame for plotting
tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
tsne_df['name'] = tsne_sample['name'].values
tsne_df['category'] = tsne_sample['category'].values
tsne_df['calories'] = tsne_sample['calories'].values
tsne_df['protein'] = tsne_sample['protein'].values
tsne_df['fat'] = tsne_sample['fat'].values
tsne_df['carbohydrate'] = tsne_sample['carbohydrate'].values

# --- Cluster Identification (New Part) ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
tsne_df['cluster'] = kmeans.fit_predict(X) # Cluster on original scaled data

# --- Cluster Naming (New Part) ---
# Analyze each cluster to give it a meaningful name
cluster_names = {}
for i in range(n_clusters):
    cluster_data = tsne_df[tsne_df['cluster'] == i]
    
    # Get dominant original category within the cluster
    dominant_category = cluster_data['category'].mode()[0] if not cluster_data['category'].empty else "N/A"
    
    # Calculate average macronutrient profile for the cluster
    avg_protein = cluster_data['protein'].mean()
    avg_fat = cluster_data['fat'].mean()
    avg_carb = cluster_data['carbohydrate'].mean()

    # Simple rule-based naming based on dominant macronutrient
    macro_label = ""
    if avg_protein > avg_fat and avg_protein > avg_carb:
        macro_label = "High Protein"
    elif avg_fat > avg_protein and avg_fat > avg_carb:
        macro_label = "High Fat"
    elif avg_carb > avg_protein and avg_carb > avg_fat:
        macro_label = "High Carb"
    else:
        # If no single macro is clearly dominant, check for balanced or specific combinations
        if abs(avg_protein - avg_fat) < 5 and abs(avg_protein - avg_carb) < 5: # Arbitrary threshold
             macro_label = "Balanced Macro"
        elif avg_protein > avg_fat and avg_carb > avg_fat:
            macro_label = "Protein-Carb Rich"
        elif avg_fat > avg_protein and avg_carb > avg_protein:
            macro_label = "Fat-Carb Rich"
        elif avg_protein > avg_carb and avg_fat > avg_carb:
            macro_label = "Protein-Fat Rich"
        else:
            macro_label = "Mixed Macro" # Fallback for less clear cases

    # Combine approaches for a more descriptive label
    # You might need to refine these names based on actual cluster content
    cluster_names[i] = f"Cluster {i+1}: {macro_label} ({dominant_category} Dominant)"

tsne_df['cluster_label'] = tsne_df['cluster'].map(cluster_names)

# Visualization with Cluster Labels
fig = px.scatter(
    tsne_df,
    x='TSNE1',
    y='TSNE2',
    color='cluster_label', # Color by the new cluster labels
    size='calories',
    hover_name='name',
    hover_data=['calories', 'protein', 'fat', 'carbohydrate', 'category', 'cluster_label'], # Add cluster_label to hover
    title="t-SNE Clustering of Foods by Macronutrient Similarity with K-Means Clusters",
    height=600,
    labels={
        "cluster_label": "Identified Food Cluster", # Label for the new cluster legend
        "TSNE1": "t-SNE Dimension 1",
        "TSNE2": "t-SNE Dimension 2"
    },
    opacity=0.7
)

fig.update_layout(
    legend_title="Food Cluster",
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    ),
)

st.plotly_chart(fig, use_container_width=True)

# Add spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Key Insights section
st.subheader("Key Insights")




st.write("""
**Box/Violin Plots**: Reveals distribution patterns and outliers for macronutrients across food categories.  
**Radar Chart**: Compares average macronutrient balance between food groups at a glance.  
**Amino Acid Treemap**: Shows dominant amino acids and protein completeness in selected foods.  
**3D Scatter Plot**: Visualizes calorie density relationships between protein, fat and carbs.  
**t-SNE Clustering**: Groups nutritionally similar foods regardless of traditional categories.  
""")