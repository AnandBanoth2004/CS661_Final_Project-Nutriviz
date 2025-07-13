import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import re  

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
    
    # Define categories in priority order (most specific first) - REPLACED CATEGORIZATION
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

# Page configuration
st.set_page_config(layout="wide")
st.title("Vitamin & Mineral Density Analysis")

# Define micronutrients (note: keeping original mineral names with typos for compatibility)
minerals = ['calcium', 'copper', 'iron', 'magnesium', 'manganese', 
            'phosphorus', 'potassium', 'selenium', 'zinc']  
vitamins = ['vitamin_a', 'vitamin_b6', 'vitamin_b12', 'vitamin_c', 
            'vitamin_d', 'vitamin_e', 'vitamin_k', 'folate', 'niacin']
micronutrients = vitamins + minerals

# Section 1: Nutrient Density by Category
st.header("1. Nutrient Density by Food Category")
selected_nutrient = st.selectbox("Select Nutrient", micronutrients, key='nutrient_select')

# Define units for nutrients (adjusted based on your given max values)
nutrient_units = {
    "vitamin_a": "µg",  # micrograms
    "vitamin_b6": "mg",
    "vitamin_b12": "µg",
    "vitamin_c": "mg",
    "vitamin_d": "µg",
    "vitamin_e": "mg",
    "vitamin_k": "µg",
    "folate": "µg",
    "niacin": "mg",
    "calcium": "g",
    "copper": "µg",
    "iron": "mg",
    "magnesium": "g",
    "manganese": "mg",
    "phosphorus": "g",
    "potassium": "g",
    "selenium": "µg",
    "zinc": "µg"
}

# Fallback unit if not found
unit = nutrient_units.get(selected_nutrient, "per 100g")

# Calculate average nutrient by category
category_avg = df.groupby('category')[selected_nutrient].mean().reset_index()
category_avg = category_avg.sort_values(selected_nutrient, ascending=False)

# Visualization
fig1 = px.bar(
    category_avg,
    x='category',
    y=selected_nutrient,
    color=selected_nutrient,
    title=f"Average {selected_nutrient.replace('_', ' ').capitalize()} Content by Food Category (Bar Graph)",
    labels={
        "category": "Food Category",
        selected_nutrient: f"{selected_nutrient.replace('_', ' ').capitalize()} ({unit} per 100g)"
    },
    height=500
)

fig1.update_layout(
    xaxis_tickangle=-35
)

st.plotly_chart(fig1, use_container_width=True)


# Section 2: Treemap of High-Density Foods
st.header("2. High-Density Foods in Categories")
treemap_nutrient = st.selectbox("Select Nutrient for Treemap", micronutrients, key='treemap_select')
treemap_category = st.selectbox("Select Category", df['category'].unique(), key='category_select')

# Filter and get top foods
filtered_df = df[df['category'] == treemap_category]
top_foods = filtered_df.nlargest(20, treemap_nutrient)[['name', treemap_nutrient]].copy()

# Format nutrient values for display
top_foods['nutrient_display'] = top_foods[treemap_nutrient].apply(
    lambda x: f"{x:.3f} mg" if x > 1 else f"{x:.3f} mcg"
)

# Create custom text for hover information
top_foods['hover_text'] = top_foods['name'] + '<br>' + treemap_nutrient + ': ' + top_foods['nutrient_display']

# Visualization
fig2 = px.treemap(
    top_foods,
    path=['name'],
    values=treemap_nutrient,
    color=treemap_nutrient,
    color_continuous_scale='greens',
    title=f"Top {treemap_nutrient}-Dense Foods in {treemap_category}",
    hover_name='hover_text',  # Custom hover text
    hover_data={treemap_nutrient: False, 'hover_text': False, 'name': False},  # Hide raw data
    height=600
)

# Improve hover appearance
fig2.update_traces(
    hovertemplate='%{hovertext}<extra></extra>',
    marker_line_width=0.5,
    marker_line_color='white'
)

# Adjust layout
fig2.update_layout(
    margin=dict(t=50, l=25, r=25, b=25),
    title_font_size=20
)

st.plotly_chart(fig2, use_container_width=True)

# ===== SECTION 3: PARALLEL COORDINATES (DYNAMIC) =====
st.header("3. Parallel Coordinates for Micronutrient Profiles")
st.write("Compare average micronutrient profiles between food categories")

# Select micronutrients and categories - UPDATED DEFAULT CATEGORIES
default_micronutrients = ['vitamin_c', 'calcium', 'iron', 'potassium', 'zinc']  
selected_micronutrients = st.multiselect(
    "Select Micronutrients", 
    micronutrients, 
    default=default_micronutrients,
    key='parallel_select'
)

# Updated default categories to match new categorization
default_categories = [
    "Meats, Poultry, and Fish",
    "Beverages",
    "Sweets and Candies",
    "Fats and Oils",
    "Dairy and Egg Products",
    "Prepared Meals and Fast Food",
    "Baked Goods",
    "Infant Foods",
    "Fruits and Fruit Juices",
    "Vegetables and Vegetable Products",
    "Legumes and Legume Products",
    "Nuts and Seeds",
    "Grains and Pasta",
    "Sauces, Dips, and Gravies",
    "Soups and Stews",
    "Snacks",
    "Spices and Herbs",
    "Cereals and Breakfast Foods",
    "Cheese",
    "Seafood",
    "Baking Ingredients",
    "Other"
]
selected_categories = st.multiselect(
    "Select Categories to Compare",
    df['category'].unique(),
    default=default_categories,
    key='category_compare'
)

# Only proceed if selections are valid
if selected_categories and selected_micronutrients:
    # Prepare data - filter and aggregate
    cols_to_keep = ['category'] + selected_micronutrients
    df_filtered = df[cols_to_keep]
    df_filtered = df_filtered[df_filtered['category'].isin(selected_categories)]
    
    # Compute mean by category
    df_mean = df_filtered.groupby('category', as_index=False).mean()
    
    # Create category codes for coloring
    df_mean['CategoryCode'] = df_mean.index
    categories = df_mean['category'].tolist()
    tickvals = df_mean['CategoryCode'].tolist()
    ticktext = categories

    # Prepare colorscale
    palette = px.colors.qualitative.Plotly
    colorscale = [[i/(len(tickvals)-1), palette[i % len(palette)]] 
                for i in range(len(tickvals))]

    # Build dimensions dynamically
    dimensions = [
        dict(range=[min(tickvals), max(tickvals)],
        label='Category', 
        values=df_mean['CategoryCode'],
        tickvals=tickvals, 
        ticktext=ticktext
    )]
    
    # Add micronutrient dimensions
    for nutrient in selected_micronutrients:
        # Handle case where all values are same
        n_min = df_mean[nutrient].min()
        n_max = df_mean[nutrient].max()
        if n_min == n_max:
            n_min -= 0.1
            n_max += 0.1
            
        dimensions.append(
            dict(range=[n_min, n_max],
                 label=nutrient.replace('_', ' ').title(),
                 values=df_mean[nutrient]
            )
        )

    # Create plot
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=df_mean['CategoryCode'], 
                colorscale=colorscale,
                showscale=False),
        dimensions=dimensions
    ))

    # Improve layout
    fig.update_layout(
        plot_bgcolor='white', 
        paper_bgcolor='white',
        margin=dict(l=150, r=50, t=50, b=50),
        font=dict(family='Arial', size=12, color='black')
    )

    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.warning("Please select at least 1 category and 1 micronutrient")

# Section 4: Food Clustering by Micronutrient Similarity
st.header("4. Food Clustering by Micronutrient Similarity")
st.write("Visual clustering of foods based on micronutrient profiles")

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
    value=10,
    step=1,
    help="The number of clusters K-Means will attempt to find in the data. Adjust this to see different granularities of grouping."
)

# Select micronutrients for clustering
tsne_nutrients = st.multiselect(
    "Select Micronutrients for Clustering",
    micronutrients,
    default=['calcium', 'vitamin_c', 'iron', 'potassium', 'vitamin_b12'],
    key='tsne_select'
)

# Filter and sample data
if tsne_nutrients:
    tsne_sample = df.dropna(subset=tsne_nutrients).sample(n=500, random_state=42)
    
    # Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(tsne_sample[tsne_nutrients])
    
    # Perform K-Means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    tsne_sample['Cluster'] = clusters + 1  # Start clusters from 1
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    tsne_results = tsne.fit_transform(X)
    
    # Create DataFrame for plotting - INCLUDE THE NUTRIENT COLUMNS
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['name'] = tsne_sample['name'].values
    tsne_df['category'] = tsne_sample['category'].values
    tsne_df['Cluster'] = tsne_sample['Cluster'].values
    
    # ADD THE NUTRIENT VALUES TO THE DATAFRAME
    for nutrient in tsne_nutrients:
        tsne_df[nutrient] = tsne_sample[nutrient].values
    
    # --- Cluster Naming ---
    # Analyze each cluster to give it a meaningful name based on micronutrients
    cluster_names = {}
    for cluster_id in range(1, n_clusters + 1):
        cluster_data = tsne_df[tsne_df['Cluster'] == cluster_id]
        
        # Get dominant food category within the cluster
        if not cluster_data.empty:
            dominant_category = cluster_data['category'].mode()[0] 
        else:
            dominant_category = "N/A"
        
        # Calculate average micronutrient values for the cluster
        avg_nutrients = {}
        for nutrient in tsne_nutrients:
            avg_nutrients[nutrient] = cluster_data[nutrient].mean()
        
        # Find the micronutrient with the highest average value
        dominant_nutrient = max(avg_nutrients, key=avg_nutrients.get)
        
        # Create cluster name based on dominant micronutrient and category
        cluster_names[cluster_id] = (
            f"Cluster {cluster_id}: "
            f"High {dominant_nutrient.replace('_', ' ').title()} "
            f"({dominant_category})"
        )
    
    # Add cluster labels to DataFrame
    tsne_df['cluster_label'] = tsne_df['Cluster'].map(cluster_names)
    
    # Create plasma colors for clusters
    plasma_colors = px.colors.sequential.Plasma[:n_clusters]
    
    # Visualization 1: Clusters with meaningful labels
    st.subheader("K-Means Clusters Visualization")
    fig_clusters = px.scatter(
        tsne_df,
        x='TSNE1',
        y='TSNE2',
        color='cluster_label',  # Use the descriptive labels
        color_discrete_sequence=plasma_colors,
        hover_name='name',
        hover_data=['category'] + tsne_nutrients,
        title=f"t-SNE Clustering of Foods by Micronutrient Similarity (K={n_clusters})",
        height=600
    )
    fig_clusters.update_layout(
        legend_title_text='Cluster Description',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
    ))
    st.plotly_chart(fig_clusters, use_container_width=True)
    

    # Cluster analysis
    st.subheader("Cluster Analysis")
    
    # Calculate mean nutrients per cluster
    cluster_means = tsne_sample.groupby('Cluster')[tsne_nutrients].mean()
    
    # Add cluster names to the analysis table
    cluster_means['Cluster Name'] = cluster_means.index.map(cluster_names)
    cluster_means = cluster_means.reset_index().set_index('Cluster Name')
    
    st.write("Average nutrient values by cluster:")
    st.dataframe(cluster_means.style.background_gradient(cmap='plasma'))
    

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Key Insights section
st.subheader("Key Insights")

st.write("""
**Nutrient Density by Category**: Identifies food categories with highest concentrations of specific vitamins/minerals.  
**High-Density Foods Treemap**: Shows top foods within a category for selected nutrient density.  
**Parallel Coordinates**: Compares micronutrient profiles across categories to reveal nutritional strengths.  
**t-SNE Clustering**: Groups foods with similar micronutrient profiles regardless of category.  
""")