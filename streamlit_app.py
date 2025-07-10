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

# === Feedback Storage ===
def ensure_feedback_csv():
    try:
        if not os.path.exists(FEEDBACK_CSV):
            pd.DataFrame(columns=["user", "event_id", "rating", "comment", "timestamp"]).to_csv(FEEDBACK_CSV, index=False)
        with open(FEEDBACK_CSV, 'a'): pass  # Ensure it's writable
    except Exception as e:
        st.session_state.feedback_memory = pd.DataFrame(columns=["user", "event_id", "rating", "comment", "timestamp"])
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

# === Match ===
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
    return results

# === UI ===
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
            filtered = filtered[filtered["Mood/Intent"].str.contains(mood_input, case=False, na=False)]

        if zipcode_input.strip():
            filtered = filtered[filtered["Postcode"].astype(str).str.startswith(zipcode_input.strip())]

        filtered = filtered.sort_values(by="relevance", ascending=False)
        st.subheader(f"🔍 Found {len(filtered)} matching events")

        if len(filtered) == 0:
            st.info("No matching events found. Try another keyword like 'clean', 'educate', or 'connect'.")

        used_forms = set()

        for i, row in filtered.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row.get('title', 'Untitled Event')}")
                st.markdown(f"**Organization:** {row.get('org_title_y', 'Unknown')}")
                st.markdown(f"📍 **Location:** {row.get('primary_loc', 'Unknown')}")
                st.markdown(f"📅 **Date:** {row.get('start_date_date_y', 'N/A')}")
                tags = [row.get('Topical Theme', ''), row.get('Effort Estimate', ''), row.get('Mood/Intent', '')]
                tag_str = " ".join([f"`{t.strip()}`" for t in tags if t])
                st.markdown(f"🏷️ {tag_str}")
                st.markdown(f"📝 {row.get('short_description', '')}")

                event_unique_str = row.get("title", "") + row.get("start_date_date_y", "") + row.get("org_title_y", "")
                event_id = hashlib.md5(event_unique_str.encode()).hexdigest()

                avg_rating = get_event_average_rating(event_id)
                rating_count = get_event_rating_count(event_id)
                if avg_rating is not None and rating_count > 0:
                    st.markdown(f"⭐ **Community Rating:** {avg_rating} / 5 from {rating_count} rating(s)")

                user_rating, user_comment = get_user_feedback(st.session_state.user, event_id)
                initial_rating = user_rating if user_rating is not None else 3

                if user_rating is not None:
                    st.markdown(f"🧠 _You’ve rated this {user_rating} stars._")

                form_key = f"form_{event_id}_{i}"
                if form_key not in used_forms:
                    with st.form(key=form_key):
                        rating = st.slider("Rate this event:", 1, 5, value=initial_rating, key=f"rating_{event_id}")
                        comment = st.text_input("Leave feedback:", value=user_comment, key=f"comment_{event_id}")
                        if st.form_submit_button("Submit Feedback"):
                            store_user_feedback(st.session_state.user, event_id, rating, comment)
                            st.success("Feedback submitted!")
                            st.experimental_rerun()
                    used_forms.add(form_key)
    else:
        st.warning("Please enter something you'd like to help with.")
else:
    st.info("Enter your interest and click **Explore** to find matching events.")

















