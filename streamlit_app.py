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

# === Load Data - BULLETPROOF VERSION ===
@st.cache_data
def load_data():
    # Load CSV
    enriched = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
    enriched.columns = enriched.columns.str.strip()
    
    # Convert ALL columns to string to avoid any Series/object issues
    for col in enriched.columns:
        enriched[col] = enriched[col].astype(str).replace('nan', '').replace('None', '')
    
    # Now create derived columns safely
    enriched["description"] = enriched["description"]
    enriched["short_description"] = enriched["description"].str.slice(0, 140) + "..."
    enriched["title_clean"] = enriched["title"].str.strip().str.lower()
    
    # Create search blob safely
    enriched["search_blob"] = (
        enriched["title"] + " " +
        enriched["description"] + " " +
        enriched.get("Topical Theme", "") + " " +
        enriched.get("Activity Type", "") + " " +
        enriched.get("primary_loc", "")
    ).str.lower()
    
    # Create event IDs safely
    enriched["event_id"] = enriched.index.astype(str) + "_" + enriched["title"].str[:10]
    
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
    if "Weather Badge" in df.columns:
        return df[df["Weather Badge"].str.contains(tag, case=False, na=False)] if tag else df
    return df

# === Widget Key Helper ===
def make_unique_key(prefix, event_id, loop_idx):
    return f"{prefix}_{str(event_id).replace(' ', '_')}_{loop_idx}"

# === UI ===
query = st.text_input("👋️ How can I help?", placeholder="e.g. dogs, clean park, teach kids")

# Safe mood selector
mood_options = ["(no preference)"]
if "Mood/Intent" in final_df.columns:
    unique_moods = final_df["Mood/Intent"].unique()
    mood_options.extend([m for m in unique_moods if m and m != ""])
mood_input = st.selectbox("🌫️ Optional — Set an Intention", mood_options)

zipcode_input = st.text_input("📍 Optional — ZIP Code", placeholder="e.g. 10027")
weather_filter = st.selectbox("☀️ Filter by Weather Option", ["", "Indoors", "Outdoors", "Flexible"])

if st.button("Explore") and query:
    expanded_terms = [query.lower()]
    for key, synonyms in SYNONYM_MAP.items():
        if key in query.lower():
            expanded_terms.extend(synonyms)
    expanded_query = " ".join(expanded_terms)

    results_df = final_df.copy()

    # Safe filtering
    if mood_input != "(no preference)" and "Mood/Intent" in results_df.columns:
        results_df = results_df[results_df["Mood/Intent"].str.contains(mood_input, na=False, case=False)]

    if zipcode_input and "Postcode" in results_df.columns:
        results_df = results_df[results_df["Postcode"].str.startswith(zipcode_input, na=False)]

    results_df = filter_by_weather(results_df, weather_filter)
    
    # Safe date filtering
    date_col = "start_date_date" if "start_date_date" in results_df.columns else "start_date"
    if date_col in results_df.columns:
        results_df = results_df[~results_df[date_col].str.contains("2011|2012|2013|2014|2015", na=False)]

    query_vec = embedder.encode([expanded_query], show_progress_bar=False)
    similarities = cosine_similarity(query_vec, corpus_embeddings)[0]
    results_df["similarity"] = similarities
    results_df["score"] = similarities

    # Safe scoring
    if mood_input != "(no preference)" and "Mood/Intent" in results_df.columns:
        mask = results_df["Mood/Intent"].str.contains(mood_input, na=False, case=False)
        results_df.loc[mask, "score"] += 0.1
    
    if zipcode_input and "Postcode" in results_df.columns:
        mask = results_df["Postcode"].str.startswith(zipcode_input, na=False)
        results_df.loc[mask, "score"] += 0.1

    top_results = results_df.sort_values(by="score", ascending=False).head(30)

    st.subheader(f"🔍 Found {len(top_results)} matching events")

    for loop_idx, (_, row) in enumerate(top_results.iterrows()):
        event_id = row.event_id
        with st.container():
            st.markdown(f"### {row.get('title', 'Untitled Event')}")
            
            # Safe display
            org = row.get('org_title', 'Unknown')
            date = row.get('start_date_date', row.get('start_date', 'N/A'))
            loc = row.get('primary_loc', 'Unknown')
            
            st.markdown(f"**Org:** {org} | **Date:** {date}")
            st.markdown(f"📍 {loc}")
            
            # Safe tags
            tags = []
            for tag_col in ['Topical Theme', 'Effort Estimate', 'Mood/Intent', 'Weather Badge']:
                if tag_col in row and row[tag_col] and row[tag_col] != "":
                    tags.append(f"`{row[tag_col]}`")
            if tags:
                st.markdown(f"🏷️ {' '.join(tags)}")
            
            st.markdown(f"{row.get('short_description', '')}")

            avg_rating = get_event_rating(event_id)
            if avg_rating:
                st.markdown(f"⭐ Community Rating: {avg_rating}/5")

            # Simple feedback without forms
            col1, col2, col3 = st.columns([2, 3, 1])
            
            with col1:
                rating = st.slider("Rate:", 1, 5, 3, key=make_unique_key("rate", event_id, loop_idx))
            
            with col2:
                comment = st.text_input("Comment:", key=make_unique_key("comm", event_id, loop_idx), placeholder="Leave feedback...")
            
            with col3:
                if st.button("Submit", key=make_unique_key("submit", event_id, loop_idx)):
                    store_feedback(event_id, rating, comment)
                    st.success("✅ Thanks!")
            
            st.markdown("---")

else:
    st.info("Enter a topic like \"food\", \"kids\", \"Inwood\", etc. to explore events.")















