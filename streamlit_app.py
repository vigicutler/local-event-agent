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
    try:
        enriched = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
        enriched.columns = enriched.columns.str.strip()
        enriched["description"] = enriched["description"].fillna("")
        enriched["short_description"] = enriched["description"].str.slice(0, 140) + "..."
        enriched["title_clean"] = enriched["title"].str.strip().str.lower()
        
        # Build search blob safely - convert everything to string first
        enriched["search_blob"] = (
            enriched["title"].fillna("").astype(str) + " " +
            enriched["description"].fillna("").astype(str)
        ).str.lower()
        
        # Try to add optional columns if they exist
        try:
            if "Topical Theme" in enriched.columns:
                enriched["search_blob"] = enriched["search_blob"] + " " + enriched["Topical Theme"].fillna("").astype(str).str.lower()
            if "Activity Type" in enriched.columns:
                enriched["search_blob"] = enriched["search_blob"] + " " + enriched["Activity Type"].fillna("").astype(str).str.lower()
            if "primary_loc" in enriched.columns:
                enriched["search_blob"] = enriched["search_blob"] + " " + enriched["primary_loc"].fillna("").astype(str).str.lower()
        except:
            pass  # If optional columns fail, continue with basic search
        
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
        if "Weather Badge" in df.columns:
            return df[df["Weather Badge"].fillna('').str.contains(tag, case=False)]
        return df
    except Exception as e:
        st.warning(f"⚠️ Error filtering by weather: {str(e)}")
        return df

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

        # Apply filters safely
        try:
            if mood_input != "(no preference)" and "Mood/Intent" in results_df.columns:
                results_df = results_df[results_df["Mood/Intent"].str.contains(mood_input, na=False, case=False)]
        except:
            pass

        try:
            if zipcode_input and "Postcode" in results_df.columns:
                results_df = results_df[results_df["Postcode"].astype(str).str.startswith(zipcode_input)]
        except:
            pass

        results_df = filter_by_weather(results_df, weather_filter)
        
        # Filter out old events safely
        try:
            if "start_date" in results_df.columns:
                results_df = results_df[~results_df["start_date"].fillna("").str.contains("2011|2012|2013|2014|2015")]
        except:
            pass

        # Compute similarities
        query_vec = embedder.encode([expanded_query], show_progress_bar=False)
        similarities = cosine_similarity(query_vec, corpus_embeddings)[0]
        
        # Get indices of filtered results
        filtered_indices = results_df.index.tolist()
        results_df = results_df.copy()
        results_df["similarity"] = [similarities[i] for i in filtered_indices]

        # Calculate scores
        results_df["score"] = results_df["similarity"]
        try:
            if mood_input != "(no preference)" and "Mood/Intent" in results_df.columns:
                mood_mask = results_df["Mood/Intent"].str.contains(mood_input, na=False, case=False)
                results_df.loc[mood_mask, "score"] += 0.1
        except:
            pass
        
        try:
            if zipcode_input and "Postcode" in results_df.columns:
                zip_mask = results_df["Postcode"].astype(str).str.startswith(zipcode_input)
                results_df.loc[zip_mask, "score"] += 0.1
        except:
            pass

        return results_df.sort_values(by="score", ascending=False).head(30)
        
    except Exception as e:
        st.error(f"❌ An error occurred while searching: {str(e)}")
        return pd.DataFrame()

# === Display Event Function ===
def display_event(row, event_index):
    """Display a single event with feedback"""
    event_id = row.event_id
    
    st.markdown(f"### {row.get('title', 'Untitled Event')}")
    
    # Display org and date safely
    org = row.get('org_title', row.get('organization', 'Unknown'))
    date = row.get('start_date', row.get('date', 'N/A'))
    st.markdown(f"**Org:** {org} | **Date:** {date}")
    
    # Display location safely
    location = row.get('primary_loc', row.get('location', row.get('venue', 'Unknown')))
    st.markdown(f"📍 {location}")
    
    # Display tags safely
    tags = []
    tag_columns = ['Topical Theme', 'Effort Estimate', 'Mood/Intent', 'Weather Badge']
    for tag_col in tag_columns:
        try:
            if tag_col in row.index and pd.notna(row.get(tag_col)) and row.get(tag_col):
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

# Mood selector - only show if column exists
mood_options = ["(no preference)"]
try:
    if "Mood/Intent" in final_df.columns:
        mood_options.extend(sorted(final_df["Mood/Intent"].dropna().unique()))
except:
    pass
mood_input = st.selectbox("🌫️ Optional — Set an Intention", mood_options)

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
    for i, (_, row) in enumerate(current_results.iterrows()):
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




















