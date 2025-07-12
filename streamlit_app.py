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

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'results_per_page' not in st.session_state:
    st.session_state.results_per_page = 5

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
    st.write("🔍 Starting to load CSV...")
    enriched = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
    st.write("✅ CSV loaded successfully!")
    
    st.write("🔍 Checking data shape:", enriched.shape)
    st.write("🔍 Columns:", list(enriched.columns))
    
    # Show first few rows to debug
    st.write("🔍 First row data types:")
    st.write(enriched.dtypes.head())
    
    # Check for problematic columns
    st.write("🔍 Sample of first row:")
    first_row = enriched.iloc[0]
    for col in ["title", "description"]:
        if col in enriched.columns:
            val = first_row[col]
            st.write(f"  {col}: {type(val)} = {val}")
    
    # Try the simplest possible approach
    st.write("🔍 Creating simple dataframe...")
    
    # Convert to basic Python data first
    simple_data = []
    for i in range(min(10, len(enriched))):  # Start with just 10 rows
        row_dict = {}
        for col in enriched.columns:
            val = enriched.iloc[i][col]
            if pd.isna(val):
                val = ""
            row_dict[col] = str(val)
        simple_data.append(row_dict)
    
    # Create new dataframe from clean data
    clean_df = pd.DataFrame(simple_data)
    
    # Add derived columns
    clean_df["short_description"] = clean_df["description"].apply(lambda x: str(x)[:140] + "...")
    clean_df["title_clean"] = clean_df["title"].apply(lambda x: str(x).strip().lower())
    clean_df["search_blob"] = clean_df.apply(lambda row: (str(row["title"]) + " " + str(row["description"])).lower(), axis=1)
    clean_df["event_id"] = clean_df.apply(lambda row: hashlib.md5((str(row["title"]) + str(row["description"])).encode()).hexdigest(), axis=1)
    
    st.write("✅ Successfully created clean dataframe!")
    return clean_df

try:
    final_df = load_data()
    st.success(f"✅ Loaded {len(final_df)} events successfully!")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# === Embeddings Setup ===
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

try:
    embedder = load_embedder()
    st.success("✅ Loaded embedder successfully!")
except Exception as e:
    st.error(f"❌ Embedder error: {str(e)}")
    st.stop()

# Initialize embeddings
@st.cache_data
def compute_embeddings():
    search_texts = []
    for idx in range(len(final_df)):
        search_texts.append(final_df.iloc[idx]["search_blob"])
    return embedder.encode(search_texts, show_progress_bar=False)

try:
    corpus_embeddings = compute_embeddings()
    st.success("✅ Computed embeddings successfully!")
except Exception as e:
    st.error(f"❌ Embeddings error: {str(e)}")
    st.stop()

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

# === Search Function ===
def perform_search(query, mood_input, zipcode_input, weather_filter):
    try:
        # Expand query with synonyms
        expanded_terms = [query.lower()]
        for key, synonyms in SYNONYM_MAP.items():
            if key in query.lower():
                expanded_terms.extend(synonyms)
        expanded_query = " ".join(expanded_terms)

        results_df = final_df.copy()

        # Compute similarities
        query_vec = embedder.encode([expanded_query], show_progress_bar=False)
        similarities = cosine_similarity(query_vec, corpus_embeddings)[0]
        
        results_df["similarity"] = similarities
        results_df["score"] = similarities

        return results_df.sort_values(by="score", ascending=False).head(30)
        
    except Exception as e:
        st.error(f"❌ An error occurred while searching: {str(e)}")
        return pd.DataFrame()

# === Display Event Function ===
def display_event(row, event_index):
    """Display a single event with feedback"""
    event_id = row["event_id"]
    
    st.markdown(f"### {row.get('title', 'Untitled Event')}")
    st.markdown(f"**Org:** {row.get('org_title', 'Unknown')} | **Date:** {row.get('start_date', 'N/A')}")
    st.markdown(f"📍 {row.get('primary_loc', 'Unknown')}")
    
    # Display tags safely
    tags = []
    tag_columns = ['Topical Theme', 'Effort Estimate', 'Mood/Intent', 'Weather Badge']
    for tag_col in tag_columns:
        try:
            if tag_col in row and pd.notna(row.get(tag_col)) and row.get(tag_col):
                tags.append(f"`{row.get(tag_col)}`")
        except:
            pass
    if tags:
        st.markdown(f"🏷️ {' '.join(tags)}")
    
    st.markdown(f"{row.get('short_description', '')}")

    # Display average rating
    avg_rating = get_event_rating(event_id)
    if avg_rating:
        st.markdown(f"⭐ Community Rating: {avg_rating}/5")

    # Feedback section
    if st.button(f"📝 Rate Event #{event_index + 1}", key=f"rate_btn_{event_id}_{event_index}"):
        st.session_state[f"show_feedback_{event_id}_{event_index}"] = True

    # Show feedback form if button was clicked
    if st.session_state.get(f"show_feedback_{event_id}_{event_index}", False):
        st.markdown("**Leave your feedback:**")
        
        rating = st.slider(
            "Rating:", 
            1, 5, 3,
            key=f"rating_{event_id}_{event_index}"
        )
        
        comment = st.text_area(
            "Comment:", 
            key=f"comment_{event_id}_{event_index}",
            placeholder="What did you think about this event?"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Feedback", key=f"submit_{event_id}_{event_index}"):
                if store_feedback(event_id, rating, comment):
                    st.success("✅ Thank you for your feedback!")
                    st.session_state[f"show_feedback_{event_id}_{event_index}"] = False
                else:
                    st.error("❌ Failed to save feedback.")
        
        with col2:
            if st.button("Cancel", key=f"cancel_{event_id}_{event_index}"):
                st.session_state[f"show_feedback_{event_id}_{event_index}"] = False

# === UI ===
st.markdown("### Search for Events")
query = st.text_input("👋️ How can I help?", placeholder="e.g. dogs, clean park, teach kids")

# Simplified selectors
mood_input = st.selectbox("🌫️ Optional — Set an Intention", ["(no preference)"])
zipcode_input = st.text_input("📍 Optional — ZIP Code", placeholder="e.g. 10027")
weather_filter = st.selectbox("☀️ Filter by Weather Option", ["", "Indoors", "Outdoors", "Flexible"])

# Search button
if st.button("🔍 Explore", key="main_explore_btn"):
    if query:
        with st.spinner("Searching for events..."):
            st.session_state.search_results = perform_search(query, mood_input, zipcode_input, weather_filter)
            st.session_state.current_page = 0
    else:
        st.warning("⚠️ Please enter a search term to explore events.")

# Display results if they exist
if st.session_state.search_results is not None and not st.session_state.search_results.empty:
    results_df = st.session_state.search_results
    total_results = len(results_df)
    
    st.markdown("---")
    st.subheader(f"🔍 Found {total_results} matching events")
    
    # Pagination
    start_idx = st.session_state.current_page * st.session_state.results_per_page
    end_idx = min(start_idx + st.session_state.results_per_page, total_results)
    
    st.markdown(f"Showing events {start_idx + 1}-{end_idx} of {total_results}")
    
    # Display current page of results
    current_results = results_df.iloc[start_idx:end_idx]
    
    # Display each event individually
    for i in range(len(current_results)):
        row = current_results.iloc[i]
        with st.container():
            display_event(row, start_idx + i)
            st.markdown("---")
    
    # Pagination controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.current_page > 0:
            if st.button("⬅️ Previous", key="prev_btn"):
                st.session_state.current_page -= 1
                st.rerun()
    
    with col2:
        st.markdown(f"Page {st.session_state.current_page + 1} of {(total_results - 1) // st.session_state.results_per_page + 1}")
    
    with col3:
        if end_idx < total_results:
            if st.button("Next ➡️", key="next_btn"):
                st.session_state.current_page += 1
                st.rerun()

elif st.session_state.search_results is not None and st.session_state.search_results.empty:
    st.info("No events found matching your criteria. Try different search terms.")

else:
    st.info("Enter a topic like \"food\", \"kids\", \"Inwood\", etc. to explore events.")

# === Footer ===
st.markdown("---")
st.markdown("💡 **Tips:** Try searching for specific activities, locations, or causes you care about!")



















