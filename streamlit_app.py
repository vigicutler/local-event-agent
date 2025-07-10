import streamlit as st
import pandas as pd
import hashlib
import os
from datetime import datetime
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FEEDBACK_CSV = "feedback_backup.csv"

SYNONYM_MAP = {
    "kids": ["youth", "children", "students", "tutoring"],
    "plant tree": ["planting", "gardening", "trees", "green", "environment"],
    "homelessness": ["shelter", "housing", "unsheltered", "support"],
    "elderly": ["seniors", "older adults", "companionship"],
    "animals": ["pets", "rescue", "dogs", "cats", "shelters"],
    "dogs": ["dogs", "pets", "canines", "puppies", "animal care"]
}

PRELOADED_USERS = ["vigi", "andy", "juan", "bruce", "ana"]

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
        if any(word in desc for word in ["meditate", "journal", "quiet", "contemplation", "healing"]): return "Reflect"
        if any(word in desc for word in ["party", "social", "connect", "meet", "talk"]): return "Connect"
        if any(word in desc for word in ["support", "uplift", "inspire", "empower"]): return "Uplift"
        return ""

    merged["Mood/Intent"] = merged.apply(lambda row: row["Mood/Intent"] if pd.notna(row.get("Mood/Intent")) and row["Mood/Intent"].strip() != "" else infer_mood(row.get("description", "")), axis=1)
    location_cols = ["primary_loc", "primary_loc_y", "locality", "Borough", "City", "Postcode", "Location Name", "Street Address", "Address 1", "Address 2"]
    existing_cols = [col for col in location_cols if col in merged.columns]
    merged["primary_loc"] = merged[existing_cols].bfill(axis=1).iloc[:, 0].fillna("Unknown")
    title_col = "title" if "title" in merged.columns else "title_clean"

    for col in ["description", "Topical Theme", "Activity Type", "primary_loc", "Postcode", "City"]:
        if col not in merged.columns: merged[col] = ""

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

def ensure_feedback_csv():
    try:
        if not os.path.exists(FEEDBACK_CSV):
            pd.DataFrame(columns=["user", "event_id", "rating", "comment", "timestamp"]).to_csv(FEEDBACK_CSV, index=False)
        with open(FEEDBACK_CSV, 'a'): pass
        st.session_state.feedback_fallback = False
    except Exception:
        st.session_state.feedback_fallback = True
        st.session_state.feedback_memory = pd.DataFrame([
            {"user": "vigi", "event_id": "evt_park", "rating": 5, "comment": "Love park events!", "timestamp": datetime.utcnow()},
            {"user": "juan", "event_id": "evt_nature", "rating": 4, "comment": "Educational and relaxing.", "timestamp": datetime.utcnow()}
        ])
        st.warning("⚠️ Feedback fallback mode active. Feedback will not be saved permanently.")
        st.info("💡 Feedback fallback set to memory.")

ensure_feedback_csv()

# --- Sidebar Login ---
st.sidebar.title("🧑 Login")
if "user" not in st.session_state:
    username = st.sidebar.text_input("Enter your username", value="guest")
    if st.sidebar.button("Login"):
        st.session_state.user = username.lower()
        st.experimental_rerun()
else:
    st.sidebar.markdown(f"**Logged in as:** `{st.session_state.user}`")
    if st.sidebar.button("Logout"):
        del st.session_state.user
        st.experimental_rerun()

st.title(f"🌱 Welcome, {st.session_state.get('user', 'Guest').capitalize()}")
st.write("Find community events that match your goals and vibe.")

query = st.text_input("🙋‍♀️ How can I help?", "")
mood = st.selectbox("💫 Mood/Intent", ["(no preference)", "Reflect", "Connect", "Uplift"])
zip_filter = st.text_input("📍 ZIP Code (optional)", "")
submit_search = st.button("🔍 Search")

if submit_search:
    def expand_query(text):
        tokens = text.lower().split()
        expanded = set(tokens)
        for word in tokens:
            for key, synonyms in SYNONYM_MAP.items():
                if word in synonyms or word == key:
                    expanded.update(synonyms)
        return " ".join(expanded)

    query_expanded = expand_query(query)
    query_vec = vectorizer.transform([query_expanded])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[::-1][:50]
    results = final_df.iloc[top_indices]

    if zip_filter:
        results = results[results["Postcode"].astype(str).str.contains(zip_filter)]
    if mood != "(no preference)":
        results = results[results["Mood/Intent"] == mood]

    st.markdown(f"🔍 Showing {len(results)} matching events")

    for i, row in results.iterrows():
        event_id = f"{row['title'][:10]}_{i}"
        st.subheader(row.get("title", "Untitled"))
        st.caption(f"📍 {row.get('primary_loc', 'Unknown')}  —  📅 {row.get('Start Date', 'Unknown')}")
        st.markdown(f"📝 {row.get('short_description', '')}")

        with st.form(key=f"form_{event_id}_{i}"):
            rating = st.slider("Rate this event:", 1, 5, 3, key=f"slider_{event_id}")
            comment = st.text_input("Leave feedback:", key=f"comment_{event_id}")
            submit = st.form_submit_button("Submit")
            if submit:
                feedback = {
                    "user": st.session_state.get("user", "guest"),
                    "event_id": event_id,
                    "rating": rating,
                    "comment": comment,
                    "timestamp": datetime.utcnow()
                }
                try:
                    df = pd.read_csv(FEEDBACK_CSV)
                    df = pd.concat([df, pd.DataFrame([feedback])], ignore_index=True)
                    df.to_csv(FEEDBACK_CSV, index=False)
                    st.success("✅ Feedback saved.")
                except:
                    if "feedback_memory" not in st.session_state:
                        st.session_state.feedback_memory = pd.DataFrame(columns=["user", "event_id", "rating", "comment", "timestamp"])
                    st.session_state.feedback_memory = pd.concat([st.session_state.feedback_memory, pd.DataFrame([feedback])], ignore_index=True)
                    st.warning("⚠️ Feedback stored in memory only.")

        ratings_df = None
        try:
            ratings_df = pd.read_csv(FEEDBACK_CSV)
        except:
            ratings_df = st.session_state.get("feedback_memory")

        if ratings_df is not None:
            ev_ratings = ratings_df[ratings_df["event_id"] == event_id]
            if not ev_ratings.empty:
                avg_rating = ev_ratings["rating"].mean()
                count = ev_ratings.shape[0]
                st.caption(f"⭐ Avg Rating: {avg_rating:.1f} ({count} reviews)")


















