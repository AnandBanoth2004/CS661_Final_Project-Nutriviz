# 🥗 NutriViz: Interactive Nutritional Insights

## 📌 Overview

**NutriViz** is a Streamlit-based visual analytics dashboard built as a project for the course **Big Data Visual Analytics**. It aims to simplify the complexity of nutritional data, enabling users to explore over **8,800 foods** interactively.

Through a data-driven interface, NutriViz allows users to:
- Understand macronutrient and micronutrient compositions
- Visualize fat and cholesterol relationships
- Benchmark food categories
- Get personalized food recommendations for conditions like anemia, heart disease, or diabetes

---

## 📊 Data Source

The nutritional dataset used in this project was obtained from Kaggle:  
🔗 [Nutrition Dataset – Kaggle](https://www.kaggle.com/datasets/gokulprasantht/nutrition-dataset
)

---

## 🚀 Features

- **Interactive Charts**: Radar plots, bar charts, PCA, t-SNE, scatterplots, treemaps, and more
- **Search & Filter**: Search by food name, filter by category or health needs
- **Clustering-based Recommendations**: GMM-based smart suggestions
- **Nutrient Trade-off Explorer**: Identify outliers and balance nutritional goals

---

## 🛠️ Tech Stack

| Purpose                 | Libraries & Tools                             |
|-------------------------|-----------------------------------------------|
| **Data Processing**     | `Pandas`, `NumPy`, `SciPy`, `t-SNE`, `GMM`    |
| **Visualization**       | `Plotly`, `Matplotlib`, `Seaborn`            |
| **Dashboard Deployment**| `Streamlit`                                   |
| **Version Control**     | `Git`, `GitHub`                               |

---

## 💡 How to Run Locally

Follow these steps to set up and run NutriViz on your system:

1. **Download the Dataset**
   - Visit [this Kaggle link](https://www.kaggle.com/datasets/gokulprasantht/nutritiondataset)
   - Download the dataset ZIP and extract the files
   - Ensure the Excel file (e.g., `nutrition.xlsx`) is placed in a folder named `data/`

2. **Create Project Folder**
   - Open VS Code
   - Create a new directory named `CS661_Final_Project`
   - Inside it, create the following structure:
     ```
     CS661_Final_Project/
     ├── data/
     │   └── nutrition.xlsx
     ├── pages/
     │   ├── Overview.py
     │   ├── Macronutrient_Distribution_Analysis.py
     │   ├── Vitamin_and_Mineral_Density.py
     │   ├── Fat_Composition_Analysis.py
     │   ├── Food_Category_Benchmarking.py
     │   ├── Nutrition_Based_Food_Search.py
     │   ├── Nutrient_Correlation_Exploration.py
     │   └── Calorie_Macro_Calculator.py
     └── Home.py
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run the Streamlit App**
    ```bash
    streamlit run Home.py

- This will launch the Streamlit app in your browser. The homepage will guide you through the available visualizations and tools.

📊 **Pages Description**

Each Python file in the pages/ directory represents a dedicated feature module:

| Page                                     | Description                                           |
| ---------------------------------------- | ----------------------------------------------------- |
| `Overview.py`                            | Dashboard overview and usage guide                    |
| `Macronutrient_Distribution_Analysis.py` | Explore carb, protein, and fat distributions          |
| `Vitamin_and_Mineral_Density.py`         | Visualize micronutrient-rich foods                    |
| `Fat_Composition_Analysis.py`            | Dive into types of fats and their health implications |
| `Food_Category_Benchmarking.py`          | Compare nutrient profiles across food categories      |
| `Nutrition_Based_Food_Search.py`         | Search for foods using health-driven filters          |
| `Nutrient_Correlation_Exploration.py`    | Analyze interdependencies between nutrients           |
| `Calorie_Macro_Calculator.py`              | Estimate calorie intake from macronutrient breakdown  |


📈 **Example Use Cases**

- A diabetic patient exploring low-sugar, high-fiber foods

- A nutritionist identifying iron-rich foods for anemic clients

- General users benchmarking healthy snacks vs. “sugar bombs”

🙌 **Acknowledgments**

This project was developed as part of CS661: Big Data Visual Analytics at IIT Kanpur.
