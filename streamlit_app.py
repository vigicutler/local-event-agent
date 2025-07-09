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

# === Streamlit UI ===
if "user" not in st.session_state:
    st.session_state.user = "Guest"

with st.sidebar:
    st.subheader("🔐 Login")
    username = st.text_input("Username", value=st.session_state.user)
    if st.button("Login"):
        st.session_state.user = username
    st.write(f"You are logged in as: `{st.session_state.user}`")
    if st.button("Logout"):
        st.session_state.user = "Guest"

st.set_page_config(page_title="Local Event Agent", layout="centered")
st.title("🌱 NYC Community Event Agent")
st.markdown("Choose how you'd like to help and find meaningful events near you.")

intent_input = st.text_input("🙋‍♀️ How can I help?", placeholder="e.g. help with homelessness, teach kids, plant trees")
mood_input = st.selectbox("💫 Optional — Set an Intention", ["(no preference)", "Uplift", "Unwind", "Connect", "Empower", "Reflect"])
zipcode_input = st.text_input("📍 Optional — ZIP Code", placeholder="e.g. 10027")

if st.button("Explore"):
    query = intent_input.strip()
    if query:
        filtered = get_top_matches(query)

        if mood_input != "(no preference)":
            def mood_match(row):
                mood_tag = str(row.get("Mood/Intent", "")).lower()
                desc = str(row.get("description", "")).lower()
                return (
                    fuzz.partial_ratio(mood_tag, mood_input.lower()) > 60 or
                    mood_input.lower() in desc
                )
            filtered = filtered[filtered.apply(mood_match, axis=1)]

        if zipcode_input.strip():
            filtered = filtered[filtered.get("Postcode", pd.Series("", index=filtered.index)).astype(str).str.startswith(zipcode_input.strip())]

        filtered = filtered.sort_values(by="relevance", ascending=False)
        st.subheader(f"🔍 Found {len(filtered)} matching events")

        if len(filtered) == 0:
            st.info("No matching events found. Try another keyword like 'clean', 'educate', or 'connect'.")

        for _, row in filtered.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row.get('title', 'Untitled Event')}")
                st.markdown(f"**Organization:** {row.get('org_title_y', 'Unknown')}")
                st.markdown(f"📍 **Location:** {row.get('primary_loc', 'Unknown')}")
                st.markdown(f"📅 **Date:** {row.get('start_date_date_y', 'N/A')}")
                tags = [row.get('Topical Theme', ''), row.get('Effort Estimate', ''), row.get('Mood/Intent', '')]
                tag_str = " ".join([f"`{t.strip()}`" for t in tags if t])
                st.markdown(f"🏷️ {tag_str}")
                st.markdown(f"📝 {row.get('short_description', '')}")

                event_id = hashlib.md5((row.get("title", "") + row.get("description", "")).encode()).hexdigest()
                avg_rating = get_event_average_rating(event_id)
                if avg_rating is not None:
                    st.markdown(f"⭐ **Community Rating:** {avg_rating} / 5")

                user_rating, user_comment = get_user_feedback(st.session_state.user, event_id)
                initial_rating = int(user_rating) if user_rating is not None else 3

                with st.form(key=f"form_{event_id}"):
                    rating = st.slider("Rate this event:", 1, 5, value=initial_rating, key=f"rating_{event_id}")
                    comment = st.text_input("Leave feedback:", value=user_comment, key=f"comment_{event_id}")
                    if st.form_submit_button("Submit Feedback"):
                        store_user_feedback(st.session_state.user, event_id, rating, comment)
                        st.success("Feedback submitted!")
    else:
        st.warning("Please enter something you'd like to help with.")
else:
    st.info("Enter your interest and click **Explore** to find matching events.")











