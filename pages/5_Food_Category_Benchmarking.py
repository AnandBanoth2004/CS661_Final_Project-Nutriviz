import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import re

# Page configuration
st.set_page_config(
    page_title="Food Category Benchmarking",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .header { color: #2c3e50; }
    .feature-card { 
        border-radius: 10px; 
        padding: 20px; 
        background-color: white; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .plot-container { margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

st.title("Food Category Benchmarking")
st.markdown("""
Compare nutritional profiles across food categories using statistical benchmarks, 
dimensionality reduction techniques, and clustering algorithms.
""", unsafe_allow_html=True)

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

if df.empty:
    st.warning("Data not loaded. Check file path or data format.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filter Options")

# Get categories with at least 5 samples
category_counts = df['category'].value_counts()
valid_categories = category_counts[category_counts >= 5].index.tolist()

# Category filter
# Select default categories: choose any 4 from valid_categories
default_categories = [
    "Cereals and Breakfast Foods",
    "Baked Goods",
    "Vegetables and Vegetable Products",
    "Seafood"
]

# Ensure default categories are actually in valid_categories
available_defaults = [cat for cat in default_categories if cat in valid_categories]
if not available_defaults:
    available_defaults = valid_categories[:min(4, len(valid_categories))]

selected_categories = st.sidebar.multiselect(
    "Select Food Categories",
    options=valid_categories,
    default=available_defaults
)

# Nutrient filter
all_nutrients = ['calories', 'protein', 'sugars', 'total_fat', 'fiber', 'sodium', 
                 'calcium', 'iron', 'potassium', 'vitamin_c', 'cholesterol', 
                 'saturated_fat', 'carbohydrates']

selected_nutrients = st.sidebar.multiselect(
    "Select Nutrients to Compare",
    options=all_nutrients,
    default=['calories', 'protein', 'sugars']
)

# Filter data based on selections
if not selected_categories:
    st.warning("Please select at least one category.")
    st.stop()

filtered_df = df[df['category'].isin(selected_categories)]

# Check if we have enough data after filtering
if filtered_df.empty:
    st.warning("No data available for the selected categories.")
    st.stop()

# Main content
st.header("1. Nutrient Benchmarking by Category")
st.markdown("Compare average nutrient values across food categories (Grouped Bar Chart)")

if selected_nutrients:
    # Calculate averages
    avg_nutrients = filtered_df.groupby('category')[selected_nutrients].mean().reset_index()
    melted_df = avg_nutrients.melt(id_vars='category', 
                                  var_name='nutrient', 
                                  value_name='average_value')
    
    # Create bar chart
    fig = px.bar(
        melted_df,
        x='category',
        y='average_value',
        color='nutrient',
        barmode='group',
        labels={'average_value': 'Average Value (in g)', 'category': 'Food Category'},
        height=500,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title_text='Nutrient'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Select at least one nutrient for comparison")
# PCA Analysis
st.header("2. PCA Biplot of Food Categories")
st.markdown("Principal Component Analysis showing nutritional profiles and category relationships")

if len(selected_nutrients) >= 2:
    # Prepare data for PCA
    pca_df = filtered_df.copy()
    pca_df = pca_df.dropna(subset=selected_nutrients)

    if len(pca_df) > 0:
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_df[selected_nutrients])
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(scaled_data)
        pca_df['PC1'] = pca_components[:, 0]
        pca_df['PC2'] = pca_components[:, 1]
        
        # Create biplot with distinct colors for each category
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get unique categories and create color map
        categories = pca_df['category'].unique()
        num_categories = len(categories)
        
        # Use a qualitative colormap with distinct colors
        cmap = plt.get_cmap('tab20')  # Good for up to 20 distinct colors
        colors = [cmap(i % cmap.N) for i in range(num_categories)]
        
        # Create a mapping from category to color
        category_colors = dict(zip(categories, colors))
        
        # Scatter plot for categories with distinct colors
        for category in categories:
            subset = pca_df[pca_df['category'] == category]
            ax.scatter(subset['PC1'], subset['PC2'], 
                       label=category, 
                       alpha=0.7, 
                       s=50,
                       color=category_colors[category])
        
        # Add loadings (nutrient vectors)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # Define colors for nutrient circles (bright, distinct colors)
        circle_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Calculate influence zones for each nutrient
        for i, nutrient in enumerate(selected_nutrients):
            # Get circle color
            circle_color = circle_colors[i % len(circle_colors)]
            
            # Get the loading vector for this nutrient
            x_load, y_load = loadings[i, 0], loadings[i, 1]
            
            # Calculate the magnitude of the loading vector
            loading_magnitude = np.sqrt(x_load**2 + y_load**2)
            
            # Create ellipse parameters based on the loading vector
            # The ellipse will be oriented along the loading vector direction
            angle = np.degrees(np.arctan2(y_load, x_load))
            
            # Calculate ellipse dimensions based on loading magnitude and data spread
            # Get the data points projected onto this nutrient's direction
            nutrient_values = pca_df[nutrient].values
            
            # Scale ellipse size based on the variance in the nutrient direction
            width = loading_magnitude * 3.5  # Major axis along loading direction
            height = loading_magnitude * 2.8  # Minor axis perpendicular to loading
            
            # Position the ellipse at the tip of the loading vector
            center_x = x_load
            center_y = y_load
            
            # Create ellipse
            ellipse = matplotlib.patches.Ellipse(
                (center_x, center_y), 
                width=width, 
                height=height,
                angle=angle,
                color=circle_color, 
                fill=False, 
                linewidth=2.5, 
                linestyle='-',
                alpha=0.7
            )
            ax.add_patch(ellipse)
            
            # Add filled ellipse with transparency for better visualization
            ellipse_filled = matplotlib.patches.Ellipse(
                (center_x, center_y), 
                width=width, 
                height=height,
                angle=angle,
                color=circle_color, 
                fill=True, 
                alpha=0.15
            )
            ax.add_patch(ellipse_filled)
            
            # Add nutrient vector arrow
            ax.arrow(0, 0, x_load, y_load, 
                    head_width=0.15, head_length=0.15, 
                    fc=circle_color, ec=circle_color, 
                    linewidth=3, alpha=0.9)
            
            # Calculate label position (slightly beyond the ellipse)
            label_offset = 0.3
            label_x = x_load + label_offset * (x_load / loading_magnitude)
            label_y = y_load + label_offset * (y_load / loading_magnitude)
            
            # Add nutrient label
            ax.text(label_x, label_y, 
                    nutrient, fontsize=11, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor=circle_color, 
                             alpha=0.8,
                             edgecolor='white',
                             linewidth=1),
                    fontweight='bold',
                    color='white')
        
        # Format plot
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
        ax.set_title("PCA Biplot of Food Categories with Nutrient Influence Zones", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set equal aspect ratio to make ellipses appear correctly
        ax.set_aspect('equal', adjustable='box')
        
        # Add axes lines through origin
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Place legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                   title='Food Categories', 
                   prop={'size': 9})
        
        # Add explanation text
        plt.figtext(0.02, 0.02, 
                   "Note: Colored ellipses represent nutrient influence zones centered at loading vectors.\nFood items within these zones have higher content of the corresponding nutrient.",
                   fontsize=10, style='italic', wrap=True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display additional information
        st.subheader("Nutrient Loadings")
        loadings_df = pd.DataFrame(loadings, 
                                  columns=['PC1', 'PC2'], 
                                  index=selected_nutrients)
        loadings_df['Magnitude'] = np.sqrt(loadings_df['PC1']**2 + loadings_df['PC2']**2)
        loadings_df['Angle (degrees)'] = np.degrees(np.arctan2(loadings_df['PC2'], loadings_df['PC1']))
        st.dataframe(loadings_df.round(3))
        
        # Display explained variance
        st.subheader("Explained Variance")
        variance_df = pd.DataFrame({
            'Component': ['PC1', 'PC2'],
            'Explained Variance Ratio': pca.explained_variance_ratio_,
            'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        st.dataframe(variance_df.round(3))
        
    else:
        st.warning("Insufficient data for PCA analysis after filtering")
else:
    st.warning("Select at least 2 nutrients for PCA analysis")

# GMM Clustering with t-SNE
st.header("3. Nutrient-Based Clustering with GMM")
st.markdown("""
Gaussian Mixture Model clustering visualized with t-SNE dimensionality reduction.
""")

# Get cluster count from sidebar slider
n_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=5,
    step=1,
    help="The number of clusters will attempt to find in the data. Adjust this to see different granularities of grouping."
)

if len(selected_nutrients) >= 1:
    # Prepare data for clustering
    cluster_df = filtered_df.copy()
    cluster_df = cluster_df.dropna(subset=selected_nutrients)

    if len(cluster_df) > 10:
        # Standardize data
        scaler = StandardScaler()
        cluster_data = scaler.fit_transform(cluster_df[selected_nutrients])
        
        # Validate cluster count doesn't exceed sample size
        n_clusters_used = min(n_clusters, len(cluster_df) - 1)
        if n_clusters_used != n_clusters:
            st.warning(f"Reduced cluster count to {n_clusters_used} to match sample size")
        
        # Fit GMM with user-selected cluster count
        gmm = GaussianMixture(n_components=n_clusters_used, random_state=42)
        clusters = gmm.fit_predict(cluster_data)
        
        # Shift clusters to start from 1 instead of 0
        cluster_df['Cluster'] = clusters + 1
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_results = tsne.fit_transform(cluster_data)
        cluster_df['t-SNE1'] = tsne_results[:, 0]
        cluster_df['t-SNE2'] = tsne_results[:, 1]
        
        # Generate plasma color sequence for clusters
        plasma_colors = px.colors.sequential.Plasma[:n_clusters_used]
        
        # Create cluster visualization with plasma colors
        fig = px.scatter(
            cluster_df,
            x='t-SNE1',
            y='t-SNE2',
            color='Cluster',
            hover_name='name',
            hover_data=['category'] + selected_nutrients,
            title=f't-SNE Visualization of Nutrient Clusters (k={n_clusters_used})',
            color_discrete_sequence=plasma_colors,
            height=600
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            legend_title='GMM Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cluster characteristics
        st.subheader("Cluster Characteristics")
        st.write("Average nutrient values by cluster:")
        
        # Calculate mean nutrients per cluster
        cluster_means = cluster_df.groupby('Cluster')[selected_nutrients].mean()
        st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))
    
        
    else:
        st.warning("Insufficient data for clustering analysis")
else:
    st.warning("Select at least one nutrient for clustering analysis")

st.header("Key Insights")
st.markdown("""
- **Category Benchmarks**: Compare average nutritional values across food categories
- **Nutritional Relationships**: PCA reveals how nutrients correlate and how categories cluster
- **Dietary Patterns**: GMM identifies natural groupings of foods based on nutrient profiles
""")