# 📦 Streamlit App: Clean Version Aligned with Capstone Vision

import streamlit as st
import pandas as pd
from fuzzywuzzy import process

# === Load Data ===
@st.cache_data

def load_data():
    df = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
    df["description"] = df["description"].fillna("")
    df["Mood/Intent"] = df["Mood/Intent"].fillna("")
    df["Topical Theme"] = df["Topical Theme"].fillna("")
    df["Activity Type"] = df["Activity Type"].fillna("")
    df["Postcode"] = df["Postcode"].fillna("").astype(str)
    df["short_description"] = df["description"].str.slice(0, 120) + "..."
    return df

final_df = load_data()

# === UI Header ===
st.title("🔎 NYC Community Event Recommender")
st.markdown("Find events based on how you want to help — choose your impact and discover local opportunities.")

# === Input Fields ===
intent_input = st.text_input("🙋‍♀️ How can you help?", placeholder="e.g. help with homelessness, teach kids, plant trees")
intent_input = intent_input.strip().lower()

mood = st.selectbox("💫 Optional - Set an Intention", ["(no preference)", "Uplift", "Unwind", "Connect", "Empower", "Reflect"])
zipcode = st.text_input("📍 Optional ZIP Code Filter", placeholder="e.g. 10027")

# === Search Button ===
if st.button("🔍 Explore"):
    input_clean = intent_input.lower()

    # Fuzzy match against Topical Theme and Activity Type
    topical_matches = process.extract(input_clean, final_df["Topical Theme"].unique(), limit=10)
    activity_matches = process.extract(input_clean, final_df["Activity Type"].unique(), limit=10)

    matched_themes = [match[0] for match in topical_matches if match[1] > 60]
    matched_activities = [match[0] for match in activity_matches if match[1] > 60]

    filtered = final_df[
        final_df["Topical Theme"].isin(matched_themes) |
        final_df["Activity Type"].isin(matched_activities)
    ]

    # Apply optional mood filter
    if mood != "(no preference)":
        filtered = filtered[
            filtered["Mood/Intent"].str.contains(mood, case=False)
        ]

    # Apply ZIP filter if provided
    if zipcode:
        filtered = filtered[filtered["Postcode"].str.startswith(zipcode.strip())]

    # Display result count
    st.subheader(f"📋 {len(filtered)} matching events:")

    # Show select columns
    st.dataframe(
        filtered[["short_description", "Topical Theme", "Effort Estimate", "Weather Badge"]].reset_index(drop=True),
        use_container_width=True
    )



