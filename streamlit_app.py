import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import os
from datetime import datetime

FEEDBACK_CSV = "feedback_backup.csv"

st.set_page_config(page_title="🌱 NYC Community Event Agent")
st.title("🌱 NYC Community Event Agent")
st.markdown("Choose how you'd like to help and find meaningful events near you.")

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
    
    # Convert everything to strings immediately
    for col in df.columns:
        df[col] = df[col].astype(str).replace('nan', '')
    
    # Create search column
    df['search_text'] = df.get('title', '') + ' ' + df.get('description', '')
    
    # Create event IDs
    df['event_id'] = [hashlib.md5(f"{i}_{row['title']}".encode()).hexdigest() for i, (_, row) in enumerate(df.iterrows())]
    
    return df

try:
    df = load_data()
    st.success(f"✅ Loaded {len(df)} events!")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    # Show what files exist
    try:
        import os
        files = os.listdir('.')
        st.write("Files in directory:", files)
    except:
        pass
    st.stop()

# === Simple Search ===
def simple_search(query, data):
    query = query.lower()
    results = []
    
    for i, row in data.iterrows():
        search_text = str(row.get('search_text', '')).lower()
        if query in search_text:
            results.append(row)
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# === UI ===
st.markdown("### Search for Events")
query = st.text_input("👋️ How can I help?", placeholder="e.g. dogs, clean park, teach kids")

if st.button("🔍 Search"):
    if query:
        with st.spinner("Searching..."):
            results = simple_search(query, df)
            st.session_state.search_results = results
    else:
        st.warning("Please enter a search term")

# Display results
if st.session_state.search_results is not None:
    results = st.session_state.search_results
    
    if len(results) > 0:
        st.subheader(f"🔍 Found {len(results)} events")
        
        for i, (_, row) in enumerate(results.head(10).iterrows()):
            st.markdown(f"### {row.get('title', 'No Title')}")
            st.markdown(f"**Description:** {str(row.get('description', ''))[:200]}...")
            
            # Show available fields
            for col in ['org_title', 'start_date', 'primary_loc', 'Topical Theme']:
                if col in row and row[col] and str(row[col]) != 'nan':
                    st.markdown(f"**{col}:** {row[col]}")
            
            st.markdown("---")
    else:
        st.info("No events found. Try different keywords.")

else:
    st.info("Enter a search term to find events!")

# Show available columns for debugging
with st.expander("Debug: Available Data Columns"):
    st.write("Columns in your data:", list(df.columns))
    st.write("Sample row:", df.iloc[0].to_dict())



















