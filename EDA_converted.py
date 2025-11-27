#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import subprocess

def pip_install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", pkg], stdout=subprocess.DEVNULL)

required = ["pandas", "numpy", "plotly", "scikit-learn", "openpyxl", "seaborn"]
for pkg in required:
    try:
        __import__(pkg)
    except Exception:
        pip_install(pkg)


# In[3]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)


# In[4]:


# 1) Load data (change path if needed)
# -----------------------
path = "final_cleaned_zomato_data.xlsx"
try:
    data = pd.read_excel(path, engine="openpyxl")
    print("Loaded:", path)
except Exception as e:
    raise FileNotFoundError(f"Cannot open {path}: {e}")

# quick preview
print("\nData shape:", data.shape)
print(data.head())


# In[5]:


# 2) Standard cleaning helpers for columns we will use
#    - rate: keep left-side of "x/5", keep decimals; invalid -> NaN or blank
#    - approx_cost_for_two_people: convert numeric, remove commas, blanks -> NaN
#    - votes: numeric, invalid -> NaN
# -----------------------
def clean_rate_col(df, col="rate"):
    if col not in df.columns:
        return df
    s = df[col].astype(str).str.strip()
    # Extract left-side number (handles "3.5/5", "4/5", "4.2", "No Rating")
    s_num = s.str.extract(r'^([\d]+(?:\.\d+)?)')[0]
    df[col] = pd.to_numeric(s_num, errors="coerce")
    return df

def clean_cost_col(df, col="approx_cost_for_two_people"):
    if col not in df.columns:
        return df
    s = df[col].astype(str).str.replace(",","").str.strip()
    # remove non-digit except decimal
    s = s.str.replace(r'[^\d.]','', regex=True)
    df[col] = pd.to_numeric(s, errors="coerce")
    return df

def clean_votes_col(df, col="votes"):
    if col not in df.columns:
        return df
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

clean_rate_col(data, "rate")
clean_cost_col(data, "approx_cost_for_two_people")
clean_votes_col(data, "votes")

# show cleaned datatypes
print("\nColumn dtypes after cleaning:")
print(data[['rate','approx_cost_for_two_people','votes']].dtypes)


# In[6]:


# 3) Useful derived columns
# -----------------------
# Shorten/normalize cuisine: take first cuisine if multiple
if "cuisines" in data.columns:
    data["primary_cuisine"] = data["cuisines"].astype(str).str.split(",", n=1).str[0].str.strip().replace({"": np.nan})

# Standardize online_order and book_table if present
for col in ["online_order", "book_table"]:
    if col in data.columns:
        data[col] = data[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

# short name for location
if "location" in data.columns:
    data["location_short"] = data["location"].astype(str).str.split(",", n=1).str[0].str.strip().replace({"": np.nan})


# In[7]:


data.head()


# In[8]:


# 4) Descriptive stats & quick prints
# -----------------------
print("\n--- Basic descriptive stats ---")
print(data.describe(include='all').T)

# Unique counts for key categorical columns (top 10)
for col in ["primary_cuisine", "location_short", "online_order"]:
    if col in data.columns:
        print(f"\nTop values for {col}:")
        print(data[col].value_counts().head(10))


# ###Interactive Charts (Plotly)
# Each chart addresses parts of your goals:
#   - distribution of ratings & costs
#   - top cuisines & their average rating/cost
#   - rating vs cost scatter (bubble by votes)
#   - neighborhood density chart
#   - cluster analysis to identify 'restaurant models'

# In[9]:


# helper to show figures inline nicely
def show(fig):
    fig.update_layout(template="plotly_white", margin=dict(l=20,r=20,t=40,b=20))
    fig.show()


# In[10]:


# 5.1 Rating distribution
if "rate" in data.columns:
    fig = px.histogram(data, x="rate", nbins=20, title="Rating distribution", hover_data=["name"])
    show(fig)


# **5.1 Rating Distribution — Insights**
# 
# The rating distribution is right-skewed, meaning most restaurants have ratings **between 3.0 and 4.2**.
# 
# Very few restaurants achieve ratings above 4.5, indicating that customers are generally strict and high ratings require consistent quality.
# 
# A small number of restaurants have extremely low ratings (< 2.5), suggesting either poor service or limited customer engagement.
# 
# **Implication:**
# The average market rating level is moderate, so new restaurants must focus heavily on service quality & consistency to stand out.

# In[11]:


# 5.2 Cost distribution (approx cost)
if "approx_cost_for_two_people" in data.columns:
    fig = px.histogram(data, x="approx_cost_for_two_people", nbins=40,
                       title="Distribution of approx_cost_for_two_people", labels={"approx_cost_for_two_people":"Cost (approx for two)"})
    show(fig)


# **5.2 Cost Distribution — Insights**
# 
# The cost for two is heavily concentrated in the lower to mid-range, with most restaurants priced **between ₹200–₹600**.
# 
# High-end restaurants **(₹1000+)** form a small minority.
# 
# A wide variance indicates the market caters to both budget and premium consumers.
# 
# **Implication:**
# Budget-friendly pricing dominates the market. An affordable pricing strategy will attract more volume unless the brand positions itself as premium.

# In[12]:


# 5.3 Top cuisines (count) - interactive bar + treemap
if "primary_cuisine" in data.columns:
    top_cuis = data["primary_cuisine"].value_counts().nlargest(20)
    fig = px.bar(x=top_cuis.index, y=top_cuis.values, title="Top 20 cuisines by restaurant count",
                 labels={"x":"Cuisine","y":"Count"})
    fig.update_layout(xaxis_tickangle=-45)
    show(fig)

    fig2 = px.treemap(data.dropna(subset=["primary_cuisine"]), path=["primary_cuisine"], values=None,
                      title="Cuisine distribution (treemap, interactive)")
    show(fig2)


# **5.3 Top Cuisines — Insights**
# 
# A few cuisines (like **North Indian, Chinese, Fast Food)** dominate in restaurant count.
# 
# Less common cuisines (like **Japanese, Thai, Continental**) may indicate niche opportunities with less competition.
# 
# Cuisine variety suggests a diverse customer taste pattern, but the top cuisines remain largely **Indian and Indo-Chinese**.
# 
# **Implication:**
# For expansion, focusing on high-demand cuisines is safer, but niche cuisines can help differentiate in competitive neighborhoods.

# In[13]:


# 5.4 Rating vs Cost (bubble) colored by cuisine, bubble size = votes
cols_for_scatter = [c for c in ("rate","approx_cost_for_two_people","votes") if c in data.columns]
if set(["rate","approx_cost_for_two_people"]).issubset(data.columns):
    df_sc = data.dropna(subset=["rate","approx_cost_for_two_people"]).copy()
    # cap votes for marker sizes
    if "votes" in df_sc.columns:
        df_sc["votes_cap"] = df_sc["votes"].fillna(0).clip(upper=df_sc["votes"].quantile(0.95))
    else:
        df_sc["votes_cap"] = 1
    fig = px.scatter(df_sc, x="approx_cost_for_two_people", y="rate",
                     size="votes_cap", color="primary_cuisine" if "primary_cuisine" in df_sc.columns else None,
                     hover_name="name", title="Rating vs Cost (bubble size = votes)",
                     labels={"approx_cost_for_two_people":"Cost for two", "rate":"Rating"})
    show(fig)


# **5.4 Rating vs Cost (Bubble Chart) — Insights**
# 
# There is no strong linear relationship between cost and rating.
# 
# Some low-cost restaurants still achieve high ratings, proving affordability does not compromise perceived quality.
# 
# Restaurants with extremely high votes tend to have mid-range pricing and good ratings, showing that customers prefer value-for-money establishments.
# 
# **Implication:**
# Premium pricing doesn’t guarantee higher ratings. Focus on quality, hygiene, and customer experience to increase both ratings and votes.

# In[14]:


# 5.5 Neighborhood-based restaurant density (top locations)
if "location_short" in data.columns:
    loc_counts = data["location_short"].value_counts().nlargest(25).reset_index()
    loc_counts.columns = ["location_short","count"]
    fig = px.bar(loc_counts, x="count", y="location_short", orientation="h",
                 title="Top 25 neighborhoods by restaurant count", labels={"count":"Restaurants","location_short":"Neighborhood"})
    show(fig)


# **5.5 Neighborhood Restaurant Density — Insights**
# 
# A few neighborhoods have a very high concentration of restaurants, indicating strong competition.
# 
# High-density areas may attract more customers but make it harder to differentiate.
# 
# Some neighborhoods with low density could be potential expansion spots, especially if coupled with high average ratings.
# 
# **Implication:**
# Restaurants entering saturated zones require unique positioning. Under-served locations could provide better ROI.

# In[15]:


# 5.6 Votes vs Rating heatmap / correlation
num_cols = [c for c in ["rate","approx_cost_for_two_people","votes"] if c in data.columns]
if len(num_cols) >= 2:
    corr = data[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation matrix (numeric features)")
    show(fig)


# **1. Ratings and Votes (0.447) — Moderate Positive Relationship**
# 
# Restaurants with higher ratings tend to receive more votes.
# 
# This suggests that customer engagement increases as food quality, service, and experience improve.
# 
# However, the correlation is not extremely strong, meaning votes also depend on other factors such as popularity, location visibility, and marketing.
# 
# **Implication:**
# Improving rating can increase engagement, but reputation-building and publicity still matter.
# 
# **2. Ratings and Cost (0.39) — Weak to Moderate Relationship**
# 
# Higher-cost restaurants tend to receive slightly higher ratings, but the link is weak.
# 
# This proves that expensive food does NOT guarantee high satisfaction.
# 
# Many mid-range restaurants also maintain strong ratings.
# 
# **Implication:**
# Value-for-money restaurants are performing well. Price increases must be supported with high quality and experience.
# 
# **3. Cost and Votes (0.435) — Weak to Moderate Relationship**
# 
# Slight trend that costlier restaurants receive more votes, possibly due to visibility, brand awareness, or footfall.
# 
# Still, not strong enough to conclude that higher price attracts more customers.
# 
# **Implication:**
# Popularity depends more on marketing, customer loyalty, and location rather than just price.
# 
# 
# 
# **Overall Interpretation:**
# 
# No pair shows strong correlation (>0.7) — meaning the dataset does not have multicollinearity.
# 
# Customer behavior appears to be shaped by multiple factors, not dominated by price alone.
# 
# Ratings influence votes the most, showing the importance of consistent quality.
# 
# Cost has limited effect on ratings and votes, proving customers focus more on experience than pricing.

# In[16]:


# 5.7 Find 'successful' restaurants: high rating, high votes, reasonable cost
# define success score (customizable)
if set(["rate","votes"]).issubset(data.columns):
    df_success = data.copy()
    # normalize fields - handle missing
    df_success["rate_norm"] = (df_success["rate"] - df_success["rate"].min()) / (df_success["rate"].max() - df_success["rate"].min())
    df_success["votes_norm"] = (df_success["votes"].fillna(0) - df_success["votes"].min()) / max(1, (df_success["votes"].max() - df_success["votes"].min()))
    # lower cost is better => invert normalized cost if available
    if "approx_cost_for_two_people" in df_success.columns:
        df_success["cost_norm"] = (df_success["approx_cost_for_two_people"].fillna(df_success["approx_cost_for_two_people"].median()) - df_success["approx_cost_for_two_people"].min()) / max(1, (df_success["approx_cost_for_two_people"].max() - df_success["approx_cost_for_two_people"].min()))
        df_success["cost_score"] = 1 - df_success["cost_norm"]
    else:
        df_success["cost_score"] = 0.5
    # success score
    df_success["success_score"] = 0.6 * df_success["rate_norm"] + 0.4 * df_success["votes_norm"] + 0.1 * df_success["cost_score"]
    top_success = df_success.sort_values("success_score", ascending=False).head(30)
    print("\nTop 30 restaurants by success_score (custom):")
    print(top_success[["name","primary_cuisine","location_short","rate","votes","approx_cost_for_two_people","success_score"]].head(30))
    # interactive table (plotly)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(["name","primary_cuisine","location","rate","votes","cost","success_score"]),
                    fill_color='paleturquoise', align='left'),
        cells=dict(values=[top_success["name"], top_success["primary_cuisine"], top_success["location_short"],
                           top_success["rate"], top_success["votes"], top_success["approx_cost_for_two_people"], np.round(top_success["success_score"],3)],
                   fill_color='lavender', align='left'))
    ])
    fig.update_layout(title="Top successful restaurants (interactive table)")
    show(fig)


# **5.7 Top “Successful” Restaurants (Success Score) — Insights**
# 
# The most successful restaurants balance good ratings, high engagement (votes), and moderate pricing.
# 
# These restaurants typically offer mainstream cuisines, suggesting mainstream tastes drive volume.
# 
# Some mid-cost restaurants outperform premium ones due to better consistency and customer trust.

# In[17]:


# 5.8 Clustering to identify restaurant business models
# features: rate, votes, cost (if available)
cluster_cols = [c for c in ["rate","votes","approx_cost_for_two_people"] if c in data.columns]
if len(cluster_cols) >= 2:
    df_cluster = data[cluster_cols].copy().dropna()
    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df_cluster)
    k = 4
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df_cluster = df_cluster.assign(cluster=labels)
    # attach cluster label to original rows (only where full features exist)
    data.loc[df_cluster.index, "cluster"] = df_cluster["cluster"]
    # plot clusters on first two numeric cols
    xcol, ycol = cluster_cols[0], cluster_cols[1]
    fig = px.scatter(df_cluster.reset_index(), x=xcol, y=ycol, color="cluster", size=(df_cluster[cluster_cols[1]] / (df_cluster[cluster_cols[1]].max()+1)),
                     title=f"Cluster plot (k={k}) by {xcol} vs {ycol}")
    show(fig)
    print("\nCluster centers (scaled):")
    print(pd.DataFrame(km.cluster_centers_, columns=cluster_cols))


# **5.8 Clustering — Insights**
# 
# Cluster analysis reveals 4 distinct restaurant models, such as:
# 
# Low-cost, high-rating, high-vote restaurants **(mass-market leaders)**
# 
# Mid-cost restaurants with medium ratings **(average performers)**
# 
# High-cost, high-rating restaurants **(premium niche segment)**
# 
# Low-rating, low-engagement restaurants **(underperformers)**
# 
# The best-performing clusters combine reasonable pricing with consistently high ratings.
# 
# **Implication:**
# Clustering helps identify which business model a restaurant belongs to and which segment offers the best potential for scaling.

# In[18]:


# 6) Recommendations automatically generated (simple heuristics)
# -----------------------
print("\n--- Automated recommendations (basic heuristics) ---\n")

# Recommendation 1: neighborhoods with high density but low average rating → expansion risk
if "location_short" in data.columns and "rate" in data.columns:
    loc_stats = data.groupby("location_short").agg(count=("name","count"), avg_rate=("rate","mean")).dropna().sort_values("count", ascending=False)
    crowded = loc_stats[loc_stats["count"] >= max(5, loc_stats["count"].quantile(0.75))]  # busy neighborhoods
    risky = crowded[crowded["avg_rate"] < data["rate"].median()]
    if not risky.empty:
        print("Neighborhoods with many restaurants but below-median rating — caution when expanding there:")
        print(risky.reset_index().head(10))
    else:
        print("No high-density neighborhoods with below median rating detected.")


# **Insight (Recommendation 1)**
# 
# These neighborhoods have many restaurants but below-average ratings, meaning they are high-competition but low-performance markets. Places **like BTM, HSR, Whitefield, and Indiranagar** show **strong demand but inconsistent quality**.
# 
# This suggests:
# 
# Markets are crowded, but customer satisfaction is weak.
# 
# A new restaurant with better quality, service consistency, or niche cuisine can easily stand out.
# 
# Expansion is possible, but should be done strategically and cautiously.

# In[19]:


# Recommendation 2: cuisines with high average rating and moderate cost → good candidates for expansion
if "primary_cuisine" in data.columns and set(["rate","approx_cost_for_two_people"]).issubset(data.columns):
    cuisine_stats = data.groupby("primary_cuisine").agg(count=("name","count"), avg_rate=("rate","mean"), avg_cost=("approx_cost_for_two_people","mean")).dropna()
    good = cuisine_stats[(cuisine_stats["avg_rate"] >= cuisine_stats["avg_rate"].quantile(0.75)) & (cuisine_stats["avg_cost"] <= cuisine_stats["avg_cost"].quantile(0.6))]
    if not good.empty:
        print("\nCuisines with high rating and moderate cost (good to consider for expansion):")
        print(good.sort_values(["avg_rate","count"], ascending=[False,False]).head(10))
    else:
        print("\nNo cuisines strongly match 'high rating + moderate cost' filter in this dataset.")


# **Insight (Recommendation 2 – Insights (Cuisines with High Rating & Moderate Cost)**
# 
# The analysis highlights Bihari cuisine as the only category that meets both criteria:
# 
# **High average rating (4.2)**
# 
# **Moderate average cost (₹300)**
# 
# **Low competition (only 6 restaurants) **
# 
# Bihari cuisine shows strong customer approval and remains relatively untapped in the market. This represents a high-potential expansion opportunity for restaurants seeking differentiation with comparatively low entry competition.

# In[20]:


# Recommendation 3: pricing strategy suggestion
if "approx_cost_for_two_people" in data.columns and "rate" in data.columns:
    corr = data[["rate","approx_cost_for_two_people"]].corr().iloc[0,1]
    print(f"\nCorrelation between rate and average cost: {corr:.3f}")
    if corr > 0.25:
        print("Higher prices correlate with higher ratings — premium pricing may work for top restaurants.")
    elif corr < -0.1:
        print("Higher prices correlate with lower ratings — review pricing strategy; cheaper options may attract higher ratings.")
    else:
        print("Weak correlation between rating and price — pricing alone may not drive ratings in this market.")


# **Insight (Recommendation 3: Pricing Strategy)**
# 
# **The correlation between restaurant rating and price (approx_cost_for_two_people) is 0.39, which is a moderate positive relationship**.
# 
# This means higher-priced restaurants tend to receive slightly better ratings.
# 
# So, premium pricing may work well for high-quality or top-rated restaurants, but price alone doesn’t fully explain rating differences.
# 
# Restaurants should focus on value + experience + service, not just increasing prices, because the correlation isn’t strong enough for pricing alone to drive ratings.

# In[21]:


print("\nEDA completed. Use the interactive charts above to explore relationships visually.")
print("If you want, I can export selected charts to interactive HTML files, or create a Plotly Dash / Streamlit app for a clickable dashboard.")


# In[ ]:




