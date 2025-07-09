import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import os
from datetime import datetime
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FEEDBACK_CSV = "feedback_backup.csv"

# === Synonym Expansion Map ===
SYNONYM_MAP = {
    "kids": ["youth", "children", "students", "tutoring"],
    "plant tree": ["planting", "gardening", "trees", "green", "environment"],
    "homelessness": ["shelter", "housing", "unsheltered", "support"],
    "elderly": ["seniors", "older adults", "companionship"],
    "animals": ["pets", "rescue", "dogs", "cats", "shelters"]
}

# === Load Event Data ===
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

    merged = pd.merge(
        enriched,
        raw,
        on="title_clean",
        how="left",
        suffixes=("", "_y")
    )

    def infer_mood(description):
        desc = str(description).lower()
        if any(word in desc for word in ["meditate", "journal", "quiet", "contemplation", "healing"]):
            return "Reflect"
        if any(word in desc for word in ["party", "social", "connect", "meet", "talk"]):
            return "Connect"
        if any(word in desc for word in ["support", "uplift", "inspire", "empower"]):
            return "Uplift"
        return ""

    merged["Mood/Intent"] = merged.apply(
        lambda row: row["Mood/Intent"] if pd.notna(row.get("Mood/Intent")) and row["Mood/Intent"].strip() != ""
        else infer_mood(row.get("description", "")),
        axis=1
    )

    location_cols = [
        "primary_loc", "primary_loc_y", "locality", "Borough", "City", "Postcode",
        "Location Name", "Street Address", "Address 1", "Address 2"
    ]
    existing_cols = [col for col in location_cols if col in merged.columns]
    merged["primary_loc"] = merged[existing_cols].bfill(axis=1).iloc[:, 0].fillna("Unknown")

    title_col = "title" if "title" in merged.columns else "title_clean"

    for col in ["description", "Topical Theme", "Activity Type", "primary_loc", "Postcode", "City"]:
        if col not in merged.columns:
            merged[col] = ""

    merged["search_blob"] = (
        merged[title_col].fillna("").astype(str) + " " +
        merged["description"].fillna("").astype(str) + " " +
        merged["Topical Theme"].fillna("").astype(str) + " " +
        merged["Activity Type"].fillna("").astype(str) + " " +
        merged["primary_loc"].fillna("").astype(str) + " " +
        merged["Postcode"].fillna("").astype(str) + " " +
        merged["City"].fillna("").astype(str)
    ).str.lower()

    return merged

final_df = load_data()

# === TF-IDF Setup ===
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(final_df["search_blob"])

# === Ensure Feedback CSV Exists ===
if not os.path.exists(FEEDBACK_CSV):
    pd.DataFrame(columns=["user", "event_id", "rating", "comment", "timestamp"]).to_csv(FEEDBACK_CSV, index=False)

# === Feedback Storage Functions ===
def load_feedback():
    return pd.read_csv(FEEDBACK_CSV)

def save_feedback(feedback_df):
    feedback_df.to_csv(FEEDBACK_CSV, index=False)

def store_user_feedback(user, event_id, rating, comment):
    feedback_df = load_feedback()
    timestamp = datetime.utcnow().isoformat()

    existing = feedback_df[(feedback_df["user"] == user) & (feedback_df["event_id"] == event_id)]
    if not existing.empty:
        feedback_df.loc[existing.index, ["rating", "comment", "timestamp"]] = [rating, comment, timestamp]
    else:
        feedback_df = pd.concat([feedback_df, pd.DataFrame([{
            "user": user,
            "event_id": event_id,
            "rating": rating,
            "comment": comment,
            "timestamp": timestamp
        }])], ignore_index=True)

    save_feedback(feedback_df)

def get_user_feedback(user, event_id):
    feedback_df = load_feedback()
    match = feedback_df[(feedback_df["user"] == user) & (feedback_df["event_id"] == event_id)]
    if not match.empty:
        return match.iloc[0]["rating"], match.iloc[0]["comment"]
    return None, ""

def get_event_average_rating(event_id):
    feedback_df = load_feedback()
    ratings = feedback_df[feedback_df["event_id"] == event_id]["rating"]
    return round(ratings.mean(), 2) if not ratings.empty else None

# === Utility Functions for Recommendation (placeholder for now) ===
def compute_similarity(df, target_blob):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["search_blob"])
    target_vector = tfidf.transform([target_blob])
    cosine_sim = cosine_similarity(target_vector, tfidf_matrix).flatten()
    return cosine_sim

# === Fuzzy Match ===
def get_top_matches(query, top_n=50):
    expanded_terms = [query.lower()]
    for key, synonyms in SYNONYM_MAP.items():
        if key in query.lower():
            expanded_terms += synonyms
    expanded_query = " ".join(expanded_terms)

    query_vec = vectorizer.transform([expanded_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    results = final_df.iloc[top_indices].copy()
    results["relevance"] = similarity_scores[top_indices]
    results["relevance"] += results.get("title", results.get("title_clean", "")).astype(str).str.contains(query, case=False, na=False).astype(int) * 0.2
    results["relevance"] += results.get("Topical Theme", pd.Series("", index=results.index)).astype(str).str.contains(query, case=False, na=False).astype(int) * 0.2
    return results










