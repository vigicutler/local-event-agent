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

# === Feedback Storage ===
def ensure_feedback_csv():
    try:
        if not os.path.exists(FEEDBACK_CSV):
            pd.DataFrame(columns=["user", "event_id", "rating", "comment", "timestamp"]).to_csv(FEEDBACK_CSV, index=False)
        with open(FEEDBACK_CSV, 'a'): pass
    except Exception as e:
        st.session_state.feedback_memory = pd.DataFrame([
            {"user": "vigi",  "event_id": "evt_park",   "rating": 5, "comment": "Love park events!", "timestamp": datetime.utcnow()},
            {"user": "juan",  "event_id": "evt_nature", "rating": 4, "comment": "Educational and relaxing.", "timestamp": datetime.utcnow()},
            {"user": "bruce", "event_id": "evt_animals", "rating": 5, "comment": "Great for animal lovers!", "timestamp": datetime.utcnow()},
            {"user": "ana",   "event_id": "evt_art_kids", "rating": 4, "comment": "Loved the creative activities.", "timestamp": datetime.utcnow()},
            {"user": "andy",  "event_id": "evt_tech_music", "rating": 5, "comment": "Really cool tech + music combo.", "timestamp": datetime.utcnow()}
        ])
        st.warning("⚠️ Feedback is temporarily stored in memory. Changes won't persist between sessions.")

ensure_feedback_csv()

# === Feedback Helpers ===
def load_feedback():
    try:
        return pd.read_csv(FEEDBACK_CSV)
    except:
        return st.session_state.get("feedback_memory", pd.DataFrame(columns=["user", "event_id", "rating", "comment", "timestamp"]))

def save_feedback(df):
    try:
        df.to_csv(FEEDBACK_CSV, index=False)
    except:
        st.session_state.feedback_memory = df

def store_user_feedback(user, event_id, rating, comment):
    df = load_feedback()
    timestamp = datetime.utcnow().isoformat()
    idx = df[(df.user == user) & (df.event_id == event_id)].index
    if len(idx):
        df.loc[idx, ["rating", "comment", "timestamp"]] = [rating, comment, timestamp]
    else:
        df = pd.concat([df, pd.DataFrame([{"user": user, "event_id": event_id, "rating": rating, "comment": comment, "timestamp": timestamp}])], ignore_index=True)
    save_feedback(df)

def get_user_feedback(user, event_id):
    df = load_feedback()
    row = df[(df.user == user) & (df.event_id == event_id)]
    if not row.empty:
        return int(row.iloc[0].rating), row.iloc[0].comment
    return None, ""

def get_event_average_rating(event_id):
    df = load_feedback()
    ratings = df[df.event_id == event_id]["rating"]
    return round(ratings.mean(), 2) if not ratings.empty else None

def get_event_rating_count(event_id):
    df = load_feedback()
    return df[df.event_id == event_id].shape[0]

def get_user_history(user):
    return load_feedback()[load_feedback().user == user]

def recommend_similar_events(user, top_n=5):
    history = get_user_history(user)
    if history.empty:
        return pd.DataFrame()
    rated_events = final_df.copy()
    rated_events["event_id"] = rated_events.apply(lambda row: hashlib.md5((str(row.get("title", "")) + str(row.get("description", ""))).encode()).hexdigest(), axis=1)
    joined = pd.merge(history, rated_events, on="event_id")
    if joined.empty:
        return pd.DataFrame()
    liked = joined[joined.rating >= 4]
    if liked.empty:
        return pd.DataFrame()
    liked_vec = vectorizer.transform(liked["search_blob"])
    sim_scores = cosine_similarity(liked_vec, tfidf_matrix).mean(axis=0)
    indices = sim_scores.argsort()[-top_n:][::-1]
    return final_df.iloc[indices]

# === Match (Improved) ===
def keyword_filter(df, keyword):
    keyword = keyword.lower()
    return df[
        df["Topical Theme"].str.lower().str.contains(keyword) |
        df["Activity Type"].str.lower().str.contains(keyword) |
        df["description"].str.lower().str.contains(keyword)
    ]

def get_top_matches(query, top_n=50):
    expanded_terms = [query.lower()]
    for key, synonyms in SYNONYM_MAP.items():
        if key in query.lower():
            expanded_terms += synonyms
    expanded_query = " ".join(expanded_terms)

    keyword_filtered = keyword_filter(final_df, query)
    query_vec = vectorizer.transform([expanded_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    tfidf_filtered = final_df.iloc[top_indices].copy()
    tfidf_filtered["relevance"] = similarity_scores[top_indices]

    combined = pd.concat([keyword_filtered, tfidf_filtered]).drop_duplicates().head(top_n)
    return combined

# === Streamlit UI ===
st.set_page_config(page_title="🌱 NYC Community Event Agent")
st.title("🌱 NYC Community Event Agent")
st.markdown("Choose how you'd like to help and find meaningful events near you.")

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

intent_input = st.text_input("🙋‍♀️ How can I help?", placeholder="e.g. help with dogs, teach kids")
mood_input = st.selectbox("💫 Optional — Set an Intention", ["(no preference)", "Uplift", "Unwind", "Connect", "Empower", "Reflect"])
zipcode_input = st.text_input("📍 Optional — ZIP Code", placeholder="e.g. 10027")

if st.button("Explore"):
    query = intent_input.strip()
    if query:
        results = get_top_matches(query)
        if mood_input != "(no preference)":
            results = results[results["Mood/Intent"].str.contains(mood_input, case=False, na=False)]
        if zipcode_input:
            results = results[results["Postcode"].astype(str).str.startswith(zipcode_input.strip())]

        st.subheader(f"🔍 Found {len(results)} matching events")

        for _, row in results.iterrows():
            with st.container():
                st.markdown(f"### {row.get('title', 'Untitled Event')}")
                st.markdown(f"**Organization:** {row.get('org_title_y', 'Unknown')}")
                st.markdown(f"📍 **Location:** {row.get('primary_loc', 'Unknown')}")
                st.markdown(f"📅 **Date:** {row.get('start_date_date_y', 'N/A')}")
                st.markdown(f"🏷️ `{row.get('Topical Theme', '')}` `{row.get('Effort Estimate', '')}` `{row.get('Mood/Intent', '')}`")
                st.markdown(f"📝 {row.get('short_description', '')}")

                event_id = hashlib.md5((row.get("title", "") + row.get("description", "")).encode()).hexdigest()
                avg_rating = get_event_average_rating(event_id)
                count = get_event_rating_count(event_id)
                if avg_rating is not None:
                    st.markdown(f"⭐ **Community Rating:** {avg_rating} / 5 ({count} ratings)")

                user_rating, user_comment = get_user_feedback(st.session_state.user, event_id)
                with st.form(key=f"form_{event_id}_{st.session_state.user}"):
                    rating = st.slider("Rate this event:", 1, 5, value=user_rating or 3)
                    comment = st.text_input("Leave feedback:", value=user_comment)
                    if st.form_submit_button("Submit Feedback"):
                        store_user_feedback(st.session_state.user, event_id, rating, comment)
                        st.success("✅ Feedback submitted and saved.")

        recs = recommend_similar_events(st.session_state.user)
        if not recs.empty:
            st.markdown("---")
            st.subheader("🎯 Recommended Events Based on Your Ratings")
            for _, row in recs.iterrows():
                st.markdown(f"- **{row.get('title')}** ({row.get('primary_loc', 'Unknown')})")

        history = get_user_history(st.session_state.user)
        if not history.empty:
            st.markdown("---")
            st.subheader("📝 Your Feedback History")
            st.dataframe(history.sort_values("timestamp", ascending=False).reset_index(drop=True))
    else:
        st.warning("Please enter a topic you'd like to help with.")















