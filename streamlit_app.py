import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

FEEDBACK_CSV = "feedback_backup.csv"

st.set_page_config(page_title="🌱 NYC Community Event Agent")
st.title("🌱 NYC Community Event Agent")
st.markdown("Choose how you'd like to help and find meaningful events near you.")

# === Synonym Expansion ===
SYNONYM_MAP = {
    "kids": ["youth", "children", "students", "tutoring"],
    "plant tree": ["planting", "gardening", "trees", "green", "environment"],
    "homelessness": ["shelter", "housing", "unsheltered", "support"],
    "elderly": ["seniors", "older adults", "companionship"],
    "animals": ["pets", "rescue", "dogs", "cats", "shelters"],
    "dogs": ["dogs", "pets", "canines", "puppies", "animal care"]
}

# === Load Data ===
@st.cache_data
def load_data():
    enriched = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
    enriched.columns = enriched.columns.str.strip()
    enriched["description"] = enriched["description"].fillna("")
    enriched["short_description"] = enriched["description"].str.slice(0, 140) + "..."
    enriched["title_clean"] = enriched["title"].str.strip().str.lower()
    enriched["search_blob"] = (
        enriched["title"].fillna("") + " " +
        enriched["description"].fillna("") + " " +
        enriched["Topical Theme"].fillna("") + " " +
        enriched["Activity Type"].fillna("") + " " +
        enriched["primary_loc"].fillna("")
    ).str.lower()
    enriched["event_id"] = enriched.apply(lambda row: hashlib.md5((str(row["title"]) + str(row["description"])).encode()).hexdigest(), axis=1)
    return enriched

final_df = load_data()

# === Embeddings Setup ===
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
corpus_embeddings = embedder.encode(final_df["search_blob"].tolist(), show_progress_bar=False)

# === Feedback Logic ===
def ensure_feedback_csv():
    if not os.path.exists(FEEDBACK_CSV):
        pd.DataFrame(columns=["event_id", "rating", "comment", "timestamp"]).to_csv(FEEDBACK_CSV, index=False)

ensure_feedback_csv()

def load_feedback():
    if os.path.exists(FEEDBACK_CSV):
        return pd.read_csv(FEEDBACK_CSV)
    return pd.DataFrame(columns=["event_id", "rating", "comment", "timestamp"])

def save_feedback(df):
    df.to_csv(FEEDBACK_CSV, index=False)

def store_feedback(event_id, rating, comment):
    df = load_feedback()
    timestamp = datetime.utcnow().isoformat()
    idx = df[df.event_id == event_id].index
    if len(idx):
        df.loc[idx, ["rating", "comment", "timestamp"]] = [rating, comment, timestamp]
    else:
        df = pd.concat([df, pd.DataFrame([{"event_id": event_id, "rating": rating, "comment": comment, "timestamp": timestamp}])], ignore_index=True)
    save_feedback(df)

def get_event_rating(event_id):
    df = load_feedback()
    ratings = df[df.event_id == event_id]["rating"]
    return round(ratings.mean(), 2) if not ratings.empty else None

def filter_by_weather(df, tag):
    return df[df["Weather Badge"].fillna('').str.contains(tag, case=False)] if tag else df

# === Widget Key Helper ===
def make_widget_key(prefix, idx):
    return f"{prefix}_{idx}"

# === UI ===
query = st.text_input("👋️ How can I help?", placeholder="e.g. dogs, clean park, teach kids")
mood_input = st.selectbox("🌫️ Optional — Set an Intention", ["(no preference)"] + sorted(final_df["Mood/Intent"].dropna().unique()))
zipcode_input = st.text_input("📍 Optional — ZIP Code", placeholder="e.g. 10027")
weather_filter = st.selectbox("☀️ Filter by Weather Option", ["", "Indoors", "Outdoors", "Flexible"])

if st.button("Explore") and query:
    expanded_terms = [query.lower()]
    for key, synonyms in SYNONYM_MAP.items():
        if key in query.lower():
            expanded_terms.extend(synonyms)
    expanded_query = " ".join(expanded_terms)

    results_df = final_df.copy()

    if mood_input != "(no preference)":
        results_df = results_df[results_df["Mood/Intent"].str.contains(mood_input, na=False, case=False)]

    if zipcode_input:
        results_df = results_df[results_df["Postcode"].astype(str).str.startswith(zipcode_input)]

    results_df = filter_by_weather(results_df, weather_filter)
    results_df = results_df[~results_df["start_date"].fillna("").str.contains("2011|2012|2013|2014|2015")]

    query_vec = embedder.encode([expanded_query], show_progress_bar=False)
    similarities = cosine_similarity(query_vec, corpus_embeddings)[0]
    results_df["similarity"] = similarities

    results_df["score"] = results_df["similarity"]
    if mood_input != "(no preference)":
        results_df.loc[results_df["Mood/Intent"].str.contains(mood_input, na=False, case=False), "score"] += 0.1
    if zipcode_input:
        results_df.loc[results_df["Postcode"].astype(str).str.startswith(zipcode_input), "score"] += 0.1

    top_results = results_df.sort_values(by="score", ascending=False).head(30)

    st.subheader(f"🔍 Found {len(top_results)} matching events")

    for i, (_, row) in enumerate(top_results.iterrows()):
        with st.container():
            st.markdown(f"### {row.get('title', 'Untitled Event')}")
            st.markdown(f"**Org:** {row.get('org_title', 'Unknown')} | **Date:** {row.get('start_date', 'N/A')}")
            st.markdown(f"📍 {row.get('primary_loc', 'Unknown')}  ")
            st.markdown(f"🏷️ `{row.get('Topical Theme', '')}` `{row.get('Effort Estimate', '')}` `{row.get('Mood/Intent', '')}` `{row.get('Weather Badge', '')}`")
            st.markdown(f"{row.get('short_description', '')}")

            event_id = row.event_id
            avg_rating = get_event_rating(event_id)
            if avg_rating:
                st.markdown(f"⭐ Community Rating: {avg_rating}/5")

            rate_key = make_widget_key("rate", i)
            comm_key = make_widget_key("comm", i)
            submit_key = make_widget_key("submit", i)

            rating = st.slider("Rate this event:", 1, 5, key=rate_key)
            comment = st.text_input("Leave feedback:", key=comm_key)
            if st.button("Submit Feedback", key=submit_key):
                store_feedback(event_id, rating, comment)
                st.success("✅ Thanks for the feedback!")
else:
    st.info("Enter a topic like \"food\", \"kids\", \"Inwood\", etc. to explore events.")



















