<div align="center">
  <img src="https://img.icons8.com/external-flaticons-flat-flat-icons/64/000000/external-insights-marketing-agency-flaticons-flat-flat-icons.png" width="60" />
  <h1>🔍 DataLens — EDA & Preprocessing Studio</h1>
  <p><em>The Next-Generation, No-Code Data Analytics & Cleaning Dashboard.</em></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](#)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](#)
  [![Plotly](https://img.shields.io/badge/Plotly-3F4F75.svg?style=for-the-badge&logo=Plotly&logoColor=white)](#)
  [![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](#)
  
  <br>
  <!-- REPLACE THE LINK BELOW WITH YOUR ACTUAL STREAMLIT LINK IF IT DIFFERS -->
  <a href="https://share.streamlit.io/lekshmiparu23-ai/EDA_WITH_SQL/main/app.py">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" height="45">
  </a>
  <p><b>Live Dashboard:</b> <a href="https://share.streamlit.io/lekshmiparu23-ai/EDA_WITH_SQL/main/app.py">https://share.streamlit.io/lekshmiparu23-ai/EDA_WITH_SQL/main/app.py</a></p>
</div>

<hr>

## 🚀 Why DataLens?
Tired of writing hundreds of lines of Pandas code just to clean a dataset and see a few histograms? **DataLens** is an advanced, fully interactive web application that does all of that for you. 

Simply upload a CSV or connect directly to your production SQL database, and DataLens instantly provides deep insights, automated dataset health scoring, an entire suite of data cleaning tools, and 9 beautifully styled, interactive statistical charts.

## 📸 Sneak Peek

### 🏠 1. The Landing Page
*Upload your local dataset or connect seamlessly via MySQL / PostgreSQL with full table browsing capabilities.*
![Landing Page](screenshots/landing_page.png)

### 📊 2. Univariate Distributions & Counting
*Instantly check distributions and categorical frequencies with self-interpreting insights!*
![Univariate Analysis](screenshots/univariate_analysis.png)

### 🔮 3. Dense Multivariate Patterns
*Spot correlations, anomalies, skewness, and heavy-clustering via our intelligent Heatmaps and Pairplots.*
![Multivariate Analysis](screenshots/multivariate_analysis.png)

<hr>

## ✨ Core Features

### 1. Flexible Data Connectivity 🗄️
- **File Upload:** Instantly drag and drop any `.csv` dataset up to 200MB.
- **Direct SQL Native:** Provide your Host, Port, and Credentials to query directly from **MySQL** or **PostgreSQL** servers without exporting data manually.

### 2. Auto-Profiling Engine 🤖
- **Dataset Health Score:** Instantly evaluating your data on a scale of 0-100% based on nulls, duplicates, and statistical outliers.
- **Smart Type Detection:** Automatically categorizing variables into `Numerical`, `Categorical`, `Datetime` and `Text`.
- **Missing Value Heatmap:** Visualizing the exact rows where your data has gaps.

### 3. Smart Preprocessing Controls 🧹
- **Missing Value Handling:** Clean nulls by Dropping, or Imputing with Mean/Median/Mode/Custom values.
- **De-Duplication:** Strip out matching rows with a single button click.
- **Categorical Encoders:** Rapidly parse categorical labels via `Label Encoding` or `One-Hot Encoding`.
- **Feature Scaling:** Normalize numerical curves via `MinMax` or `Standard Scaler`.
- **Direct Download:** Hit "Download Processed CSV" in the sidebar at any point to save your work!

### 4. Interactive Plotly Dashboards 📉
No more overlapping labels and tiny axes. DataLens provides 9 robust, dynamically scaling, dark-themed charts covering:
- Distribution Overlays with KDE lines
- Red-Diamond Outlier Boxplots
- Frequency Distribution Bar Charts
- Correlational Annotated Heatmaps
- High-performance Scatter Matrices (Pairplots)
- Skewness vs Kurtosis horizontal indexing plots

<hr>

## ⚙️ How to Setup and Run Locally

It takes less than a minute to spin up DataLens! Make sure you have Python 3.9 or higher.

**1. Clone this repository & install requirements:**
```bash
git clone https://github.com/your-username/DataLens.git
cd DataLens
pip install -r requirements.txt
```

**2. Generate dummy data to test out (optional):**
We've included a script to generate a realistic 300-row sample dataset (including artificial nulls and outliers) so you can test the cleaning modules.
```bash
python sample_data/generate.py
```

**3. Launch the App! 🚀**
```bash
streamlit run app.py
```
*The app will open automatically in your default browser at `http://localhost:8501`.*

<hr>

## 🧠 Customization

The app features deep global CSS manipulation. You can effortlessly customize the UI aesthetics (like the `#0a0f1e` backgrounds, cyan `#00d4ff` markers, and purple `#7c3aed` gradient borders) by modifying the `st.markdown("<style>...</style>")` hooks inside `app.py` and `eda_charts.py`.
