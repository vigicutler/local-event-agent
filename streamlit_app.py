import streamlit as st
import pandas as pd
import hashlib
from datetime import datetime
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === Setup Google Sheets API ===
SHEET_NAME = "event_feedback"

@st.cache_resource
def connect_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open(SHEET_NAME).sheet1
    return sheet

feedback_sheet = connect_gsheet()

# === Synonym Map ===
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
    df = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
    df.fillna("", inplace=True)
    df["title_clean"] = df["title"].str.lower().str.strip()
    df["short_description"] = df["description"].str.slice(0, 140) + "..."
    df["search_blob"] = (
        df["title_clean"] + " " + df["description"] + " " + df["Topical Theme"] + " " +
        df["Activity Type"] + " " + df["primary_loc"] + " " + df["Postcode"].astype(str)
    ).str.lower()
    return df

event_df = load_data()

# === TF-IDF Setup ===
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(event_df["search_blob"])

# === Feedback Utilities ===
def fetch_feedback_df():
    records = feedback_sheet.get_all_records()
    return pd.DataFrame(records)

def save_feedback_row(user, event_id, rating, comment):
    timestamp = datetime.utcnow().isoformat()
    rows = feedback_sheet.get_all_values()
    headers = rows[0]
    data = rows[1:]
    for i, row in enumerate(data, start=2):
        if row[0] == user and row[1] == event_id:
            feedback_sheet.update(f"C{i}:E{i}", [[rating, comment, timestamp]])
            return
    feedback_sheet.append_row([user, event_id, rating, comment, timestamp])

def get_user_feedback(user, event_id):
    df = fetch_feedback_df()
    match = df[(df.user == user) & (df.event_id == event_id)]
    if not match.empty:
        return int(match.iloc[0].rating), match.iloc[0].comment
    return None, ""

def get_event_average_rating(event_id):
    df = fetch_feedback_df()
    ratings = df[df.event_id == event_id]["rating"].astype(float)
    return round(ratings.mean(), 2) if not ratings.empty else None

def get_event_rating_count(event_id):
    df = fetch_feedback_df()
    return df[df.event_id == event_id].shape[0]

# === Recommender ===
def get_top_matches(query, top_n=50):
    expanded_terms = [query.lower()]
    for key, synonyms in SYNONYM_MAP.items():
        if key in query.lower():
            expanded_terms += synonyms
    expanded_query = " ".join(expanded_terms)
    query_vec = vectorizer.transform([expanded_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    results = event_df.iloc[top_indices].copy()
    results["relevance"] = similarity_scores[top_indices]
    return results

# === Streamlit UI ===
st.set_page_config(page_title="Local Event Agent", layout="centered")

if "user" not in st.session_state:
    st.session_state.user = "Guest"

with st.sidebar:
    st.subheader("🔐 Login")
    username = st.text_input("Username", value=st.session_state.user)
    if st.button("Login"):
        st.session_state.user = username
    st.write(f"Logged in as: `{st.session_state.user}`")
    if st.button("Logout"):
        st.session_state.user = "Guest"

st.title("🌱 NYC Community Event Agent")
st.markdown("Find local events that match your interests, mood, and community goals.")

intent_input = st.text_input("🙋 How do you want to help?", placeholder="e.g. teach kids, help dogs, plant trees")
mood_input = st.selectbox("💫 Optional — Mood/Intent", ["(no preference)", "Uplift", "Unwind", "Connect", "Empower", "Reflect"])
zipcode_input = st.text_input("📍 ZIP Code (optional)", placeholder="e.g. 10034")

if st.button("Explore"):
    if not intent_input:
        st.warning("Please describe how you want to help.")
    else:
        results = get_top_matches(intent_input)

        if mood_input != "(no preference)":
            results = results[results["Mood/Intent"].str.contains(mood_input, na=False, case=False)]

        if zipcode_input:
            results = results[results["Postcode"].astype(str).str.startswith(zipcode_input)]

        st.subheader(f"🔍 {len(results)} event(s) found")

        for i, row in results.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row['title']}")
                st.markdown(f"**Org:** {row.get('org_title_y', 'Unknown')}")
                st.markdown(f"📍 {row.get('primary_loc', 'Unknown')}")
                st.markdown(f"📅 Date: {row.get('start_date_date_y', 'N/A')}")
                st.markdown(f"📝 {row['short_description']}")

                # Tags
                tags = [row.get("Topical Theme", ""), row.get("Mood/Intent", ""), row.get("Effort Estimate", "")]
                tag_str = " ".join([f"`{t}`" for t in tags if t])
                if tag_str:
                    st.markdown(f"🏷️ {tag_str}")

                event_id = hashlib.md5((row['title'] + row.get('start_date_date_y', '')).encode()).hexdigest()

                avg_rating = get_event_average_rating(event_id)
                count = get_event_rating_count(event_id)
                if avg_rating:
                    st.markdown(f"⭐ **Community Rating:** {avg_rating} / 5 from {count} rating(s)")

                # User feedback
                existing_rating, existing_comment = get_user_feedback(st.session_state.user, event_id)
                initial_rating = existing_rating if existing_rating else 3

                with st.form(f"form_{event_id}"):
                    rating = st.slider("Rate this event:", 1, 5, value=initial_rating)
                    comment = st.text_input("Leave a comment", value=existing_comment)
                    if st.form_submit_button("Submit"):
                        save_feedback_row(st.session_state.user, event_id, rating, comment)
                        st.success("Feedback saved!")
                        st.experimental_rerun()
















