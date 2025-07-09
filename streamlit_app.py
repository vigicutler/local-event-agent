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
    "animals": ["pets", "rescue", "dogs", "cats", "shelters"],
    "dogs": ["dogs", "pets", "canines", "puppies", "animal care"]
}

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

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(final_df["search_blob"])

# === Feedback Storage ===
def ensure_feedback_csv():
    try:
        if not os.path.exists(FEEDBACK_CSV):
            pd.DataFrame(columns=["user", "event_id", "rating", "comment", "timestamp"]).to_csv(FEEDBACK_CSV, index=False)
        with open(FEEDBACK_CSV, 'a'): pass
    except Exception:
        st.session_state.feedback_memory = pd.DataFrame([
            {"user": "vigi",  "event_id": "evt_park",   "rating": 5, "comment": "Love park events!", "timestamp": datetime.utcnow()},
            {"user": "juan",  "event_id": "evt_nature", "rating": 4, "comment": "Educational and relaxing.", "timestamp": datetime.utcnow()},
            {"user": "bruce", "event_id": "evt_animals", "rating": 5, "comment": "Great for animal lovers!", "timestamp": datetime.utcnow()},
            {"user": "ana",   "event_id": "evt_art_kids", "rating": 4, "comment": "Loved the creative activities.", "timestamp": datetime.utcnow()},
            {"user": "andy",  "event_id": "evt_tech_music", "rating": 5, "comment": "Really cool tech + music combo.", "timestamp": datetime.utcnow()}
        ])
        st.warning("⚠️ Feedback is temporarily stored in memory. Changes won't persist between sessions.")

ensure_feedback_csv()

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

@st.cache_data
def fuzzy_search(query):
    expanded_terms = [query.lower()]
    for key, synonyms in SYNONYM_MAP.items():
        if key in query.lower():
            expanded_terms += synonyms
    expanded_query = " ".join(expanded_terms)
    query_vec = vectorizer.transform([expanded_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-50:][::-1]
    results = final_df.iloc[top_indices].copy()
    results["relevance"] = similarity_scores[top_indices]
    return results

def get_event_id(title, desc):
    return hashlib.md5((title + desc).encode()).hexdigest()

def load_feedback():
    if os.path.exists(FEEDBACK_CSV):
        return pd.read_csv(FEEDBACK_CSV)
    return st.session_state.get("feedback_memory", pd.DataFrame(columns=["user", "event_id", "rating", "comment", "timestamp"]))

def save_feedback(user, event_id, rating, comment):
    df = load_feedback()
    df = df[~((df.user == user) & (df.event_id == event_id))]
    new_row = pd.DataFrame([[user, event_id, rating, comment, datetime.utcnow()]], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
    try:
        df.to_csv(FEEDBACK_CSV, index=False)
    except Exception:
        st.session_state.feedback_memory = df

if st.button("Explore"):
    query = intent_input.strip()
    if query:
        results = fuzzy_search(query)

        if mood_input != "(no preference)":
            results = results[results["Mood/Intent"].str.contains(mood_input, case=False, na=False)]

        if zipcode_input.strip():
            results = results[results["Postcode"].astype(str).str.startswith(zipcode_input.strip())]

        st.subheader(f"🔍 Found {len(results)} matching events")

        feedback_df = load_feedback()

        for _, row in results.iterrows():
            title = row.get("title", "Untitled")
            desc = row.get("description", "")
            event_id = get_event_id(title, desc)
            st.markdown(f"### {title}")
            st.markdown(f"📍 **Location:** {row.get('primary_loc', 'Unknown')}  ")
            st.markdown(f"📅 **Date:** {row.get('start_date_date_y', 'N/A')}")
            st.markdown(f"📝 {row.get('short_description', '')}")

            event_feedback = feedback_df[feedback_df.event_id == event_id]
            avg_rating = event_feedback["rating"].mean() if not event_feedback.empty else None
            if avg_rating:
                st.markdown(f"⭐ **Community Rating:** {round(avg_rating, 2)} / 5")

            user_row = feedback_df[(feedback_df.user == st.session_state.user) & (feedback_df.event_id == event_id)]
            initial_rating = int(user_row["rating"].iloc[0]) if not user_row.empty else 3
            initial_comment = user_row["comment"].iloc[0] if not user_row.empty else ""

            with st.form(key=f"form_{event_id}"):
                rating = st.slider("Rate this event:", 1, 5, value=initial_rating)
                comment = st.text_input("Leave feedback:", value=initial_comment)
                if st.form_submit_button("Submit Feedback"):
                    save_feedback(st.session_state.user, event_id, rating, comment)
                    st.success("✅ Feedback submitted!")
    else:
        st.warning("Please enter something you'd like to help with.")
else:
    st.info("Enter your interest and click **Explore** to find matching events.")
















