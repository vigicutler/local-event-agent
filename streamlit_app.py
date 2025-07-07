# 📦 Streamlit App: Updated with Metadata Merge + Card UI

import streamlit as st
import pandas as pd
from fuzzywuzzy import process

# === Load Data ===
@st.cache_data

def load_data():
    enriched = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
    raw = pd.read_csv("NYC_Service__Volunteer_Opportunities__Historical__20250626.csv")

    enriched.columns = enriched.columns.str.strip()
    raw.columns = raw.columns.str.strip()

    enriched["description"] = enriched["description"].fillna("")
    enriched["short_description"] = enriched["description"].str.slice(0, 140) + "..."
    raw["summary"] = raw["summary"].fillna("")

    merged = pd.merge(
        enriched,
        raw,
        left_on="description",
        right_on="summary",
        how="left"
    )

    return merged

final_df = load_data()

# === UI Header ===
st.set_page_config(page_title="Local Event Agent", layout="centered")
st.title("🌱 NYC Community Event Agent")
st.markdown("Choose how you'd like to help and find meaningful events near you.")

# === Input Fields ===
intent_input = st.text_input("🙋‍♀️ How can you help?", placeholder="e.g. help with homelessness, teach kids, plant trees")
mood_input = st.selectbox("💫 Optional — Set an Intention", ["(no preference)", "Uplift", "Unwind", "Connect", "Empower", "Reflect"])
zipcode_input = st.text_input("📍 Optional — ZIP Code", placeholder="e.g. 10027")

# === Action Button ===
if st.button("Explore"):
    input_clean = intent_input.strip().lower()

    # Fuzzy match using Topical Theme and Activity Type
    top_matches = process.extract(input_clean, final_df["Topical Theme"].dropna().unique(), limit=3)
    act_matches = process.extract(input_clean, final_df["Activity Type"].dropna().unique(), limit=3)

    all_matches = set([match[0] for match in top_matches + act_matches if match[1] > 60])

    filtered = final_df[
        final_df["Topical Theme"].isin(all_matches) | final_df["Activity Type"].isin(all_matches)
    ]

    # Apply mood filter
    if mood_input != "(no preference)":
        filtered = filtered[filtered["Mood/Intent"].str.contains(mood_input, case=False, na=False)]

    # Apply ZIP code filter if provided
    if zipcode_input.strip() != "":
        filtered = filtered[filtered["postalcode"].astype(str).str.startswith(zipcode_input.strip())]

    st.subheader(f"🔍 Found {len(filtered)} matching events")

    # Display Event Cards
    for _, row in filtered.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row.get('title', 'Untitled Event')}")
            st.markdown(f"**Organization:** {row.get('orgname', 'Unknown')}")
            st.markdown(f"📍 **Location:** {row.get('servicelocation', 'N/A')}")
            st.markdown(f"📅 **Date:** {row.get('startdate', 'N/A')}  \n🕒 **Time:** {row.get('starttime', '')}")
            st.markdown(f"🏷️ **Tags:** {row.get('Topical Theme', '')}, {row.get('Effort Estimate', '')}, {row.get('Mood/Intent', '')}")
            st.markdown(f"📝 {row.get('short_description', '')}")
            st.markdown("---")
