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
    try:
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
        enriched["event_id"] = enriched.apply(
            lambda row: hashlib.md5((str(row["title"]) + str(row["description"])).encode()).hexdigest(), 
            axis=1
        )
        return enriched
    except FileNotFoundError:
        st.error("❌ Required CSV file 'Merged_Enriched_Events_CLUSTERED.csv' not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

final_df = load_data()

# === Embeddings Setup ===
@st.cache_resource
def load_embedder():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"❌ Error loading sentence transformer: {str(e)}")
        st.stop()

embedder = load_embedder()

# Initialize embeddings
@st.cache_data
def compute_embeddings():
    try:
        return embedder.encode(final_df["search_blob"].tolist(), show_progress_bar=False)
    except Exception as e:
        st.error(f"❌ Error computing embeddings: {str(e)}")
        st.stop()

corpus_embeddings = compute_embeddings()

# === Feedback Logic ===
def ensure_feedback_csv():
    if not os.path.exists(FEEDBACK_CSV):
        pd.DataFrame(columns=["event_id", "rating", "comment", "timestamp"]).to_csv(FEEDBACK_CSV, index=False)

ensure_feedback_csv()

def load_feedback():
    try:
        if os.path.exists(FEEDBACK_CSV):
            return pd.read_csv(FEEDBACK_CSV)
        return pd.DataFrame(columns=["event_id", "rating", "comment", "timestamp"])
    except Exception as e:
        st.warning(f"⚠️ Error loading feedback: {str(e)}")
        return pd.DataFrame(columns=["event_id", "rating", "comment", "timestamp"])

def save_feedback(df):
    try:
        df.to_csv(FEEDBACK_CSV, index=False)
    except Exception as e:
        st.error(f"❌ Error saving feedback: {str(e)}")

def store_feedback(event_id, rating, comment):
    try:
        df = load_feedback()
        timestamp = datetime.utcnow().isoformat()
        idx = df[df.event_id == event_id].index
        if len(idx):
            df.loc[idx, ["rating", "comment", "timestamp"]] = [rating, comment, timestamp]
        else:
            new_row = pd.DataFrame([{
                "event_id": event_id, 
                "rating": rating, 
                "comment": comment, 
                "timestamp": timestamp
            }])
            df = pd.concat([df, new_row], ignore_index=True)
        save_feedback(df)
        return True
    except Exception as e:
        st.error(f"❌ Error storing feedback: {str(e)}")
        return False

def get_event_rating(event_id):
    try:
        df = load_feedback()
        ratings = df[df.event_id == event_id]["rating"]
        return round(ratings.mean(), 2) if not ratings.empty else None
    except Exception as e:
        st.warning(f"⚠️ Error getting rating: {str(e)}")
        return None

def filter_by_weather(df, tag):
    if not tag:
        return df
    try:
        return df[df["Weather Badge"].fillna('').str.contains(tag, case=False)]
    except Exception as e:
        st.warning(f"⚠️ Error filtering by weather: {str(e)}")
        return df

# === Fixed Widget Key Helper ===
def make_unique_key(prefix, event_id, loop_idx):
    # Use full event_id and loop_idx to ensure uniqueness
    return f"{prefix}_{event_id}_{loop_idx}"

# === UI ===
query = st.text_input("👋️ How can I help?", placeholder="e.g. dogs, clean park, teach kids")
mood_input = st.selectbox("🌫️ Optional — Set an Intention", ["(no preference)"] + sorted(final_df["Mood/Intent"].dropna().unique()))
zipcode_input = st.text_input("📍 Optional — ZIP Code", placeholder="e.g. 10027")
weather_filter = st.selectbox("☀️ Filter by Weather Option", ["", "Indoors", "Outdoors", "Flexible"])

if st.button("Explore") and query:
    try:
        # Expand query with synonyms
        expanded_terms = [query.lower()]
        for key, synonyms in SYNONYM_MAP.items():
            if key in query.lower():
                expanded_terms.extend(synonyms)
        expanded_query = " ".join(expanded_terms)

        results_df = final_df.copy()

        # Apply filters
        if mood_input != "(no preference)":
            results_df = results_df[results_df["Mood/Intent"].str.contains(mood_input, na=False, case=False)]

        if zipcode_input:
            results_df = results_df[results_df["Postcode"].astype(str).str.startswith(zipcode_input)]

        results_df = filter_by_weather(results_df, weather_filter)
        
        # Filter out old events
        results_df = results_df[~results_df["start_date"].fillna("").str.contains("2011|2012|2013|2014|2015")]

        # Compute similarities
        query_vec = embedder.encode([expanded_query], show_progress_bar=False)
        similarities = cosine_similarity(query_vec, corpus_embeddings)[0]
        
        # Get indices of filtered results
        filtered_indices = results_df.index.tolist()
        results_df = results_df.copy()
        results_df["similarity"] = [similarities[i] for i in filtered_indices]

        # Calculate scores
        results_df["score"] = results_df["similarity"]
        if mood_input != "(no preference)":
            mood_mask = results_df["Mood/Intent"].str.contains(mood_input, na=False, case=False)
            results_df.loc[mood_mask, "score"] += 0.1
        if zipcode_input:
            zip_mask = results_df["Postcode"].astype(str).str.startswith(zipcode_input)
            results_df.loc[zip_mask, "score"] += 0.1

        top_results = results_df.sort_values(by="score", ascending=False).head(30)

        st.subheader(f"🔍 Found {len(top_results)} matching events")

        # Display results with unique keys
        for loop_idx, (_, row) in enumerate(top_results.iterrows()):
            event_id = row.event_id
            
            with st.container():
                st.markdown(f"### {row.get('title', 'Untitled Event')}")
                st.markdown(f"**Org:** {row.get('org_title', 'Unknown')} | **Date:** {row.get('start_date', 'N/A')}")
                st.markdown(f"📍 {row.get('primary_loc', 'Unknown')}")
                
                # Display tags
                tags = []
                for tag_col in ['Topical Theme', 'Effort Estimate', 'Mood/Intent', 'Weather Badge']:
                    if row.get(tag_col):
                        tags.append(f"`{row.get(tag_col)}`")
                if tags:
                    st.markdown(f"🏷️ {' '.join(tags)}")
                
                st.markdown(f"{row.get('short_description', '')}")

                # Display average rating
                avg_rating = get_event_rating(event_id)
                if avg_rating:
                    st.markdown(f"⭐ Community Rating: {avg_rating}/5")

                # Feedback form with unique key
                with st.form(key=make_unique_key("form", event_id, loop_idx)):
                    rating = st.slider(
                        "Rate this event:", 
                        1, 5, 
                        key=make_unique_key("rate", event_id, loop_idx)
                    )
                    comment = st.text_input(
                        "Leave feedback:", 
                        key=make_unique_key("comm", event_id, loop_idx)
                    )
                    if st.form_submit_button("Submit Feedback"):
                        if store_feedback(event_id, rating, comment):
                            st.success("✅ Thanks for the feedback!")
                        else:
                            st.error("❌ Failed to save feedback. Please try again.")
                
                st.markdown("---")  # Separator between events
                
    except Exception as e:
        st.error(f"❌ An error occurred while searching: {str(e)}")
        st.info("Please try a different search term or check your data files.")

elif st.button("Explore") and not query:
    st.warning("⚠️ Please enter a search term to explore events.")

else:
    st.info("Enter a topic like \"food\", \"kids\", \"Inwood\", etc. to explore events.")

# === Footer ===
st.markdown("---")
st.markdown("💡 **Tips:** Try searching for specific activities, locations, or causes you care about!")





















