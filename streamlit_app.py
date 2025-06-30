
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load clustered + tagged data
@st.cache_data
def load_data():
    return pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")

df = load_data()

st.title("🔍 NYC Community Event Recommender")
st.markdown("Find events based on what you care about — effort level, theme, vibe, and more.")

# --- Filter UI ---
cluster = st.selectbox("🎯 Choose a cluster", sorted(df["Cluster"].dropna().unique()))
theme = st.multiselect("🎨 Topical Theme", sorted(df["Topical Theme"].dropna().unique()))
effort = st.multiselect("💪 Effort Level", sorted(df["Effort Estimate"].dropna().unique()))
religious = st.multiselect("🕊️ Religious Subtag", sorted(df["Religious Subtag"].dropna().unique()))

filtered_df = df[df["Cluster"] == cluster]

if theme:
    filtered_df = filtered_df[filtered_df["Topical Theme"].isin(theme)]
if effort:
    filtered_df = filtered_df[filtered_df["Effort Estimate"].isin(effort)]
if religious:
    filtered_df = filtered_df[filtered_df["Religious Subtag"].isin(religious)]

st.markdown(f"### 📋 {len(filtered_df)} matching events:")
st.dataframe(filtered_df[["description", "Topical Theme", "Effort Estimate", "Weather Badge"]])

# Optional: recommend similar events
st.markdown("---")
st.markdown("### 🤝 Content-Based Recommendation")
row_num = st.number_input("Pick an event row number for suggestions:", min_value=0, max_value=len(df)-1, value=0)

if st.button("Suggest Similar Events"):
    # TF-IDF setup
    df["combined_text"] = (
        df["Mood/Intent"].fillna('') + " " +
        df["Topical Theme"].fillna('') + " " +
        df["Activity Type"].fillna('') + " " +
        df["Effort Estimate"].fillna('') + " " +
        df["Social Setup"].fillna('') + " " +
        df["Ideal Participant Profile"].fillna('') + " " +
        df["Religious Subtag"].fillna('') + " " +
        df["description"].fillna('')
    )

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_text"])

    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[row_num], tfidf_matrix).flatten()))
    sim_scores = sorted(sim_scores[1:], key=lambda x: x[1], reverse=True)[:5]

    st.markdown("#### Events similar to your selection:")
    for idx, score in sim_scores:
        st.markdown(f"**{df.iloc[idx]['description']}** — {df.iloc[idx]['Topical Theme']} / {df.iloc[idx]['Effort Estimate']} — Similarity: {score:.2f}")

