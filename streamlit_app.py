# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import hashlib
from datetime import datetime
from uuid import uuid4
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# === Setup ===
FEEDBACK_CSV = "feedback_light.csv"
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# === Synonyms ===
SYNONYM_MAP = {
    "kids": ["youth", "children", "students", "tutoring"],
    "plant tree": ["planting", "gardening", "trees", "green", "environment"],
    "homelessness": ["shelter", "housing", "unsheltered", "support"],
    "elderly": ["seniors", "older adults", "companionship"],
    "animals": ["pets", "rescue", "dogs", "cats", "shelters"]
}

WEATHER_OPTIONS = ["Sunny", "Rainy", "Indoors", "Flexible"]

@st.cache_data
def load_data():
    enriched = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
    raw = pd.read_csv("NYC_Service__Volunteer_Opportunities__Historical__20250626.csv")

    enriched.columns = enriched.columns.str.strip()
    raw.columns = raw.columns.str.strip()

    enriched["description"] = enriched["description"].fillna("")
    enriched["short_description"] = enriched["description"].str.slice(0, 140) + "..."

    enriched["title_clean"] = enriched["title"].str.strip().str.lower()
    raw["title_clean"] = raw["title"].str.strip().str.lower()

    merged = pd.merge(enriched, raw, on="title_clean", how="left", suffixes=("", "_y"))

    def infer_mood(description):
        desc = str(description).lower()
        if any(word in desc for word in ["meditate", "journal", "quiet"]):
            return "Reflect"
        if any(word in desc for word in ["party", "social", "connect"]):
            return "Connect"
        if any(word in desc for word in ["support", "uplift", "inspire"]):
            return "Uplift"
        return ""

    merged["Mood/Intent"] = merged.apply(
        lambda row: row["Mood/Intent"] if pd.notna(row.get("Mood/Intent")) and row["Mood/Intent"].strip() != ""
        else infer_mood(row.get("description", "")), axis=1)

    location_cols = ["primary_loc", "primary_loc_y", "locality", "Borough", "City", "Postcode"]
    existing_cols = [col for col in location_cols if col in merged.columns]
    merged["primary_loc"] = merged[existing_cols].bfill(axis=1).iloc[:, 0].fillna("Unknown")

    merged["search_blob"] = (
        merged["title"].fillna("").astype(str) + " " +
        merged["description"].fillna("").astype(str) + " " +
        merged["Topical Theme"].fillna("").astype(str) + " " +
        merged["Activity Type"].fillna("").astype(str)
    ).str.lower()

    merged["event_id"] = merged.apply(lambda row: hashlib.md5((row["title"] + row["description"]).encode()).hexdigest(), axis=1)

    return merged

final_df = load_data()

# === TF-IDF + Embeddings ===
@st.cache_data
def embed_events():
    return EMBEDDING_MODEL.encode(final_df["search_blob"].tolist(), convert_to_tensor=True)

search_embeddings = embed_events()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(final_df["search_blob"])

# === Feedback ===
def store_feedback(event_id, rating):
    ts = datetime.utcnow().isoformat()
    session = st.session_state.get("session_id", "anon")
    if not os.path.exists(FEEDBACK_CSV):
        pd.DataFrame(columns=["session", "event_id", "rating", "timestamp"]).to_csv(FEEDBACK_CSV, index=False)
    df = pd.read_csv(FEEDBACK_CSV)
    df = df[df.session != session]  # one per session
    df = pd.concat([df, pd.DataFrame([[session, event_id, rating, ts]], columns=df.columns)], ignore_index=True)
    df.to_csv(FEEDBACK_CSV, index=False)

def get_feedback():
    if os.path.exists(FEEDBACK_CSV):
        return pd.read_csv(FEEDBACK_CSV)
    return pd.DataFrame(columns=["session", "event_id", "rating", "timestamp"])

# === Recommend ===
def get_top_matches(query, weather_filter=None, mood_filter=None):
    terms = [query.lower()]
    for key, synonyms in SYNONYM_MAP.items():
        if key in query.lower():
            terms += synonyms
    expanded = " ".join(terms)

    query_embed = EMBEDDING_MODEL.encode(expanded, convert_to_tensor=True)
    sim_scores = util.cos_sim(query_embed, search_embeddings).cpu().numpy().flatten()

    top_indices = sim_scores.argsort()[-100:][::-1]
    results = final_df.iloc[top_indices].copy()
    results["relevance"] = sim_scores[top_indices]

    if mood_filter:
        results = results[results["Mood/Intent"].str.contains(mood_filter, case=False, na=False)]

    if weather_filter:
        results = results[results["Weather Badge"].fillna("").apply(lambda x: any(w in x for w in weather_filter))]

    return results.head(20)

# === App ===
st.set_page_config("🌱 Event Recommender")
st.title("🌱 Community Event Recommender")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

query = st.text_input("🙋 What do you want to help with?", placeholder="e.g. tutoring, gardening, seniors")
mood = st.selectbox("💫 Mood", ["(no preference)", "Uplift", "Unwind", "Connect", "Reflect"])
weather_picks = st.multiselect("🌤️ Preferred Weather Setting", WEATHER_OPTIONS)

if st.button("Find Events"):
    results = get_top_matches(query, weather_filter=weather_picks, mood_filter=None if mood == "(no preference)" else mood)

    if results.empty:
        st.warning("No matching events found.")
    else:
        for _, row in results.iterrows():
            st.markdown(f"### {row['title']}")
            st.markdown(f"📍 {row['primary_loc']}")
            st.markdown(f"📝 {row['short_description']}")
            st.markdown(f"🏷️ {row.get('Topical Theme', '')} | {row.get('Mood/Intent', '')} | {row.get('Weather Badge', '')}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"👍 {row['event_id']}", key=f"up_{row['event_id']}"):
                    store_feedback(row['event_id'], 5)
                    st.success("Thanks for the feedback!")
            with col2:
                if st.button(f"👎 {row['event_id']}", key=f"down_{row['event_id']}"):
                    store_feedback(row['event_id'], 1)
                    st.success("Feedback noted.")
















