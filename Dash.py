# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="Cost of Living vs Quality of Life — India")

@st.cache_data
def load_data():
    df = pd.read_csv("india_cost_quality_dataset.csv"))
    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    # Ensure expected columns exist; attempt to infer if slightly different names present
    needed = {
        "city": None,
        "rent": None,
        "food": None,
        "internet": None,
        "healthcare": None,
        "safety": None,
        "happiness": None
    }
    col_lower = {c.lower(): c for c in df.columns}
    # map heuristics
    for key in list(needed.keys()):
        if key == "city":
            for cand in ["city", "cities", "place", "location", "town"]:
                if cand in col_lower:
                    needed[key] = col_lower[cand]; break
        if key == "rent":
            for cand in ["avg_rent_inr/month", "avg_rent_inr", "avg_rent", "average rent (inr/month)", "average rent", "rent"]:
                if cand in col_lower:
                    needed[key] = col_lower[cand]; break
        if key == "food":
            for cand in ["food_cost_inr/month", "food cost (inr/month)", "food cost", "food", "avg_food_cost_usd", "avg_food_cost_inr"]:
                if cand in col_lower:
                    needed[key] = col_lower[cand]; break
        if key == "internet":
            for cand in ["internet speed (mbps)", "internet_speed_mbps", "internet speed", "internet"]:
                if cand in col_lower:
                    needed[key] = col_lower[cand]; break
        if key == "healthcare":
            for cand in ["healthcare rating", "healthcare_rating", "healthcare", "health"]:
                if cand in col_lower:
                    needed[key] = col_lower[cand]; break
        if key == "safety":
            for cand in ["safety score", "safety_score", "safety", "safety_index"]:
                if cand in col_lower:
                    needed[key] = col_lower[cand]; break
        if key == "happiness":
            for cand in ["happiness index", "happiness_index", "happiness", "happiness_score"]:
                if cand in col_lower:
                    needed[key] = col_lower[cand]; break

    missing = [k for k,v in needed.items() if v is None]
    if missing:
        st.warning(f"Couldn't automatically find columns for: {missing}. Please ensure your CSV has columns for city, rent, food, internet, healthcare, safety, happiness. Found columns: {df.columns.tolist()}")
    # Create working DF with standard column names (if mapping found)
    work = pd.DataFrame()
    work['City'] = df[needed['city']] if needed['city'] else df.iloc[:,0]
    # numeric columns - try to coerce
    def get_col(name, fallback_idx=None):
        if needed[name]:
            return pd.to_numeric(df[needed[name]], errors='coerce')
        elif fallback_idx is not None and fallback_idx < df.shape[1]:
            return pd.to_numeric(df.iloc[:, fallback_idx], errors='coerce')
        else:
            return pd.Series([np.nan]*len(df))
    work['Avg_Rent_INR'] = get_col('rent')
    work['Food_Cost_INR'] = get_col('food')
    work['Internet_Speed_Mbps'] = get_col('internet')
    work['Healthcare_Rating'] = get_col('healthcare')
    work['Safety_Score'] = get_col('safety')
    work['Happiness_Index'] = get_col('happiness')
    # Derived fields
    work['Cost_Total_INR'] = work['Avg_Rent_INR'] + work['Food_Cost_INR']
    # Fill small missing numeric values sensibly by median to keep visual coherent (user can re-run with real data)
    for col in ['Avg_Rent_INR','Food_Cost_INR','Internet_Speed_Mbps','Healthcare_Rating','Safety_Score','Happiness_Index','Cost_Total_INR']:
        if work[col].isna().any():
            work[col] = work[col].fillna(work[col].median())
    return work

df = load_data()

st.title("Cost of Living vs Quality of Life — India")
st.markdown("Interactive dashboards to explore where India’s happiest & most livable cities are relative to cost, safety, healthcare, and internet.")

# Sidebar controls
st.sidebar.header("Controls")
min_rent = int(df['Avg_Rent_INR'].min())
max_rent = int(df['Avg_Rent_INR'].max())
rent_range = st.sidebar.slider("Rent range (INR/month)", min_value=min_rent, max_value=max_rent, value=(min_rent, max_rent), step=500)
internet_min = float(df['Internet_Speed_Mbps'].min())
internet_max = float(df['Internet_Speed_Mbps'].max())
internet_thresh = st.sidebar.slider("Min Internet Speed (Mbps) for filtering", min_value=float(round(internet_min)), max_value=float(round(internet_max)), value=float(round(internet_min)), step=1.0)
top_n = st.sidebar.number_input("Top N cities to display in lists", min_value=5, max_value=50, value=10, step=1)

filtered = df[
    (df['Avg_Rent_INR'] >= rent_range[0]) &
    (df['Avg_Rent_INR'] <= rent_range[1]) &
    (df['Internet_Speed_Mbps'] >= internet_thresh)
].copy()

# Layout: 2 columns top
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("1) Cost vs Happiness — Best Bang for Buck")
    st.markdown("Scatter plot of total cost (rent + food) vs happiness. Bubble size = internet speed, color = safety.")
    fig1 = px.scatter(
        filtered,
        x='Cost_Total_INR',
        y='Happiness_Index',
        size='Internet_Speed_Mbps',
        color='Safety_Score',
        hover_name='City',
        labels={'Cost_Total_INR':'Total Monthly Cost (INR)','Happiness_Index':'Happiness Index (0-10)'},
        trendline="ols",
        height=600
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Insight idea:** Cities above the trendline give higher happiness than expected for their cost — great to highlight as `budget-happy` cities.")

with col2:
    st.subheader("Top 'Happiness per Rupee' Cities")
    # happiness per cost
    filtered['Happiness_per_INR'] = filtered['Happiness_Index'] / filtered['Cost_Total_INR']
    top_hp = filtered.sort_values('Happiness_per_INR', ascending=False).head(top_n)
    st.dataframe(top_hp[['City','Avg_Rent_INR','Food_Cost_INR','Cost_Total_INR','Happiness_Index','Happiness_per_INR']].reset_index(drop=True))

st.markdown("---")

# 2) Internet speed champions
st.subheader("2) Internet Speed Champions — Mbps per ₹ of Rent")
st.markdown("Which cities give the most internet bandwidth relative to rent cost (useful for remote workers).")
filtered['Mbps_per_1000INR_rent'] = filtered['Internet_Speed_Mbps'] / (filtered['Avg_Rent_INR'] / 1000.0)
fig2 = px.bar(
    filtered.sort_values('Mbps_per_1000INR_rent', ascending=False).head(30),
    x='City', y='Mbps_per_1000INR_rent',
    hover_data=['Internet_Speed_Mbps','Avg_Rent_INR'],
    labels={'Mbps_per_1000INR_rent':'Mbps per 1000 INR rent'},
    height=450
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# 3) Safety vs Happiness correlation
st.subheader("3) Safety vs Happiness — Correlation")
st.markdown("Evaluate whether safer cities tend to be happier.")
corr_val = df[['Safety_Score','Happiness_Index']].corr().iloc[0,1]
st.write(f"Pearson correlation (Safety vs Happiness): **{corr_val:.3f}**")
fig3 = px.scatter(df, x='Safety_Score', y='Happiness_Index', trendline='ols', hover_name='City', height=450)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# 4) Top 10 Most Balanced Cities
st.subheader("4) Top Balanced Cities (Good across Cost, Safety, Healthcare, Internet, Happiness)")
st.markdown("Balanced = above-average on Safety, Healthcare, Happiness, Internet & below-average Cost.")
avg = df.mean(numeric_only=True)
criteria = (
    (df['Safety_Score'] >= avg['Safety_Score']) &
    (df['Healthcare_Rating'] >= avg['Healthcare_Rating']) &
    (df['Happiness_Index'] >= avg['Happiness_Index']) &
    (df['Internet_Speed_Mbps'] >= avg['Internet_Speed_Mbps']) &
    (df['Cost_Total_INR'] <= avg['Cost_Total_INR'])
)
balanced = df[criteria].copy()
balanced['Balance_Score'] = ( (balanced['Safety_Score']/avg['Safety_Score']) +
                             (balanced['Healthcare_Rating']/avg['Healthcare_Rating']) +
                             (balanced['Happiness_Index']/avg['Happiness_Index']) +
                             (balanced['Internet_Speed_Mbps']/avg['Internet_Speed_Mbps']) +
                             (avg['Cost_Total_INR']/balanced['Cost_Total_INR']) )  # higher better
balanced = balanced.sort_values('Balance_Score', ascending=False).head(top_n)
if balanced.empty:
    st.info("No cities meet all 'balanced' criteria with current filters. Try expanding rent range or internet threshold.")
else:
    st.dataframe(balanced[['City','Avg_Rent_INR','Cost_Total_INR','Safety_Score','Healthcare_Rating','Internet_Speed_Mbps','Happiness_Index','Balance_Score']])

st.markdown("---")

# 5) Healthcare vs Rent vs Happiness bubble chart
st.subheader("5) Healthcare vs Rent vs Happiness")
st.markdown("Bubble chart: x = healthcare rating, y = happiness, bubble size = (1/cost) to emphasize good healthcare at low cost.")
df['Cost_inverse'] = 1 / (df['Cost_Total_INR'] + 1)  # avoid divide by zero
fig5 = px.scatter(df, x='Healthcare_Rating', y='Happiness_Index', size='Cost_inverse', color='Avg_Rent_INR',
                  hover_name='City', labels={'Healthcare_Rating':'Healthcare (0-10)', 'Happiness_Index':'Happiness (0-10)'},
                  height=600)
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# 6) Happiness Outliers using regression residuals
st.subheader("6) Happiness Outliers (Regression Residuals)")
st.markdown("Fit a regression predicting Happiness from Cost, Safety, Healthcare, Internet. Large positive residuals = happier than predicted.")
features = ['Cost_Total_INR','Safety_Score','Healthcare_Rating','Internet_Speed_Mbps']
X = df[features].values
y = df['Happiness_Index'].values
# scale features (important because Cost is large)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
model = LinearRegression()
model.fit(Xs, y)
preds = model.predict(Xs)
df['Happiness_Pred'] = preds
df['Happiness_Resid'] = df['Happiness_Index'] - df['Happiness_Pred']
outliers = df.sort_values('Happiness_Resid', ascending=False).head(top_n)
st.markdown("Top positive residuals (cities happier than predicted):")
st.dataframe(outliers[['City','Happiness_Index','Happiness_Pred','Happiness_Resid','Cost_Total_INR','Safety_Score','Healthcare_Rating','Internet_Speed_Mbps']])
fig6 = px.bar(df.sort_values('Happiness_Resid', ascending=False).head(30), x='City', y='Happiness_Resid', labels={'Happiness_Resid':'Residual (Actual - Predicted)'},
              title="Top 30 Positive Happiness Residuals", height=450)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# 7) Digital Nomad Hotspots
st.subheader("7) Digital Nomad Hotspots")
st.markdown("Filter for cities that are friendly to remote work: threshold sliders on the sidebar.")
dn_internet = st.sidebar.slider("DN: Min internet (Mbps)", int(df['Internet_Speed_Mbps'].min()), int(df['Internet_Speed_Mbps'].max()), 25)
dn_safety = st.sidebar.slider("DN: Min safety (0-10)", int(df['Safety_Score'].min()), int(df['Safety_Score'].max()), int(avg['Safety_Score']))
dn_rent_max = st.sidebar.slider("DN: Max rent (INR)", int(df['Avg_Rent_INR'].min()), int(df['Avg_Rent_INR'].max()), int(avg['Avg_Rent_INR']))
dn_candidates = df[
    (df['Internet_Speed_Mbps'] >= dn_internet) &
    (df['Safety_Score'] >= dn_safety) &
    (df['Avg_Rent_INR'] <= dn_rent_max)
].sort_values(['Internet_Speed_Mbps','Safety_Score','Happiness_Index'], ascending=[False,False,False])
st.markdown(f"Found **{len(dn_candidates)}** digital-nomad candidate cities (Top {top_n} shown).")
st.dataframe(dn_candidates.head(top_n)[['City','Avg_Rent_INR','Internet_Speed_Mbps','Safety_Score','Happiness_Index']])

# Optional: map if lat/lon are available or if user chooses to geocode (commented)
st.markdown("**Note:** The dataset doesn't require geocodes. If you have `lat`/`lon` columns, the app can plot maps. Geocoding inside the app is possible but may be slow and requires internet access; uncomment geocode block in code to enable.")

# Footer / export
st.markdown("---")
st.write("Export filtered dataset for further analysis:")
st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode('utf-8'), file_name="filtered_india_cost_quality_dataset.csv", mime='text/csv')

st.write("App created by Synthoria helper. Modify thresholds and re-run to generate more insights.")
