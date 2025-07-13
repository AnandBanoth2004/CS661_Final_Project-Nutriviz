# CS661_Final_Project-Nutriviz

# 🥗 NutriViz: Interactive Nutritional Insights

## 📌 Overview

**NutriViz** is a Streamlit-based visual analytics dashboard built as a project for the course **Big Data Visual Analytics**. It aims to simplify the complexity of nutritional data, enabling users to explore over **8,800 foods** interactively.

Through a data-driven interface, NutriViz allows users to:
- Understand macronutrient and micronutrient compositions
- Visualize fat and cholesterol relationships
- Benchmark food categories
- Get personalized food recommendations for conditions like anemia, heart disease, or diabetes

---

## 📁 Project Structure

CS661_Final_Project/
├── data/
│ ├──nutrition.xlsx
| ├──nutrition_fullcleaned.xlsx
├── pages/
│ ├── Overview.py
│ ├── Macronutrient_Distribution_Analysis.py
│ ├── Vitamin_and_Mineral_Density.py
│ ├── Fat_Composition_Analysis.py
│ ├── Food_Category_Benchmarking.py
│ ├── Nutrition-Based_Food_Search.py
│ ├── Nutrient_Correlation_Exploration.py
│ └── CaloriemacroCalculator.py
└── Home.py


---

## 🚀 Features

- **Interactive Charts**: Radar plots, bar charts, PCA, t-SNE, scatterplots, treemaps, and more
- **Search & Filter**: Search by food name, filter by category or health needs
- **Clustering-based Recommendations**: GMM-based smart suggestions
- **Nutrient Trade-off Explorer**: Identify outliers and balance nutritional goals

---

## 🛠️ Tech Stack

| Purpose                | Libraries & Tools                            |
|------------------------|----------------------------------------------|
| **Data Processing**    | `Pandas`, `NumPy`, `SciPy`, `t-SNE`, `GMM`   |
| **Visualization**      | `Plotly`, `Matplotlib`, `Seaborn`           |
| **Dashboard Deployment** | `Streamlit`                              |
| **Version Control**    | `Git`, `GitHub`                              |

---

## 💡 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/AnandBanoth2004/CS661_Final_Project-Nutriviz
   cd CS661_Final_Project

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
3. **Run the Streamlit App**
    ```bash
    streamlit run Home.py

📊 **Pages Description**

Each Python file in the pages/ directory represents a dedicated feature module:

| Page                                     | Description                                           |
| ---------------------------------------- | ----------------------------------------------------- |
| `Overview.py`                            | Dashboard overview and usage guide                    |
| `Macronutrient_Distribution_Analysis.py` | Explore carb, protein, and fat distributions          |
| `Vitamin_and_Mineral_Density.py`         | Visualize micronutrient-rich foods                    |
| `Fat_Composition_Analysis.py`            | Dive into types of fats and their health implications |
| `Food_Category_Benchmarking.py`          | Compare nutrient profiles across food categories      |
| `Nutrition-Based_Food_Search.py`         | Search for foods using health-driven filters          |
| `Nutrient_Correlation_Exploration.py`    | Analyze interdependencies between nutrients           |
| `CaloriemacroCalculator.py`              | Estimate calorie intake from macronutrient breakdown  |


📈 **Example Use Cases**

- A diabetic patient exploring low-sugar, high-fiber foods

- A nutritionist identifying iron-rich foods for anemic clients

- General users benchmarking healthy snacks vs. “sugar bombs”

🙌 **Acknowledgments**

This project was developed as part of CS661: Big Data Visual Analytics at IIT Kanpur.
