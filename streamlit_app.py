import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        lambda row: row["Mood/Intent"] if pd.notna(row["Mood/Intent"]) and row["Mood/Intent"].strip() != ""
        else infer_mood(row["description"]),
        axis=1
    )

    location_cols = ["primary_loc", "primary_loc_y", "locality", "Borough", "City", "Postcode", "Location Name", "Street Address"]
    merged["primary_loc"] = merged[location_cols].bfill(axis=1).iloc[:, 0].fillna("Unknown")

    merged["search_blob"] = (
        merged["title"].fillna("") + " " +
        merged["description"].fillna("") + " " +
        merged["Topical Theme"].fillna("") + " " +
        merged["Activity Type"].fillna("") + " " +
        merged["primary_loc"].fillna("") + " " +
        merged["Postcode"].fillna("") + " " +
        merged["City"].fillna("")
    ).str.lower()

    return merged

final_df = load_data()

# === TF-IDF Setup ===
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(final_df["search_blob"])

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
    results["relevance"] += results["title"].str.contains(query, case=False, na=False).astype(int) * 0.2
    results["relevance"] += results["Topical Theme"].str.contains(query, case=False, na=False).astype(int) * 0.2
    return results

# === SQLite Setup ===
conn = sqlite3.connect("feedback.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        user TEXT,
        event_id TEXT,
        event_name TEXT,
        rating INTEGER,
        feedback TEXT,
        timestamp TEXT
    )
''')
conn.commit()

# === UI ===
st.set_page_config(page_title="Local Event Agent", layout="centered")
st.title("\U0001F331 NYC Community Event Agent")

# --- Persistent login in sidebar ---
with st.sidebar:
    st.header("\U0001F512 User Login")
    if 'user' not in st.session_state or not st.session_state.user:
        username = st.text_input("Enter your name:", key="login_name")
        if username:
            st.session_state.user = username
            st.success(f"Welcome, {username}!")
    else:
        st.markdown(f"\U0001F44B Hello, **{st.session_state.user}**")
        if st.button("Logout"):
            del st.session_state.user

st.markdown("Choose how you'd like to help and find meaningful events near you.")

intent_input = st.text_input("\U0001F64B‍♀️ How can I help?", placeholder="e.g. help with homelessness, teach kids, plant trees")
mood_input = st.selectbox("\U0001F4AB Optional — Set an Intention", ["(no preference)", "Uplift", "Unwind", "Connect", "Empower", "Reflect"])
zipcode_input = st.text_input("\U0001F4CD Optional — ZIP Code", placeholder="e.g. 10027")

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
            filtered = filtered[filtered["Postcode"].astype(str).str.startswith(zipcode_input.strip())]

        feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)
        filtered["event_hash"] = filtered.apply(lambda row: hashlib.md5((row["title"] + row["description"]).encode()).hexdigest(), axis=1)
        filtered["community_rating"] = filtered["event_hash"].map(
            lambda eid: feedback_df[feedback_df["event_id"] == eid]["rating"].mean()
            if eid in feedback_df["event_id"].values else 0
        )

        filtered = filtered.sort_values(by="community_rating", ascending=False)

        st.subheader(f"\U0001F50D Found {len(filtered)} matching events")

        for _, row in filtered.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row.get('title', 'Untitled Event')}")
                st.markdown(f"**Organization:** {row.get('org_title_y', 'Unknown')}")
                location = row.get('primary_loc') or "Unknown"
                st.markdown(f"\U0001F4CD **Location:** {location}")
                st.markdown(f"\U0001F4C5 **Date:** {row.get('start_date_date_y', 'N/A')}")
                tags = [row.get('Topical Theme', ''), row.get('Effort Estimate', ''), row.get('Mood/Intent', '')]
                tag_str = " ".join([f"`{t.strip()}`" for t in tags if t])
                st.markdown(f"🏷️ {tag_str}")
                st.markdown(f"📝 {row.get('short_description', '')}")
                st.markdown(f"⭐ **Community Rating:** {row['community_rating']:.1f}/5" if row['community_rating'] > 0 else "⭐ No ratings yet")

                if st.session_state.get("user"):
                    event_id = row["event_hash"]
                    with st.form(key=f"feedback_form_{event_id}"):
                        rating = st.slider("Rate this event:", min_value=1, max_value=5, key=f"rating_{event_id}")
                        feedback = st.text_area("Tell us what you thought:", key=f"feedback_{event_id}")
                        submit = st.form_submit_button("Submit Feedback")
                        if submit:
                            timestamp = datetime.now().isoformat()
                            c.execute('''
                                INSERT INTO feedback (user, event_id, event_name, rating, feedback, timestamp)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (
                                st.session_state.user,
                                event_id,
                                row.get("title", "Untitled Event"),
                                rating,
                                feedback,
                                timestamp
                            ))
                            conn.commit()
                            st.success("✅ Feedback submitted!")
                else:
                    st.info("\U0001F512 Please log in from the sidebar to leave feedback.")

        if st.session_state.get("user"):
            user_feedback = pd.read_sql_query(
                "SELECT * FROM feedback WHERE user = ?", conn, params=(st.session_state.user,)
            )
            with st.sidebar:
                st.markdown("📝 **Your Previous Ratings**")
                if user_feedback.empty:
                    st.write("No feedback submitted yet.")
                else:
                    for _, fb_row in user_feedback.iterrows():
                        st.markdown(f"- **{fb_row['event_name']}**: {fb_row['rating']}⭐ — {fb_row['feedback']}")
else:
    st.info("Enter your interest and click **Explore** to find matching events.")






