# 📦 Streamlit App: Community Event Recommender (Updated Version)
import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
    df["description"] = df["description"].fillna("")
    df["Mood/Intent"] = df["Mood/Intent"].fillna("")
    df["Topical Theme"] = df["Topical Theme"].fillna("")
    df["Activity Type"] = df["Activity Type"].fillna("")
    df["short_description"] = df["description"].str.slice(0, 120) + "..."
    return df

df = load_data()

# === Custom CSS for Key Biscayne Aesthetic ===
st.markdown("""
<style>
body {
    background-color: #f0f9f8;
}
h1 {
    color: #2b6777;
}
button[kind="primary"] {
    background-color: #52ab98 !important;
    color: white !important;
}
div[data-testid="stSelectbox"] label {
    color: #2b6777;
}
</style>
""", unsafe_allow_html=True)

# === UI Header ===
st.title("🌿 NYC Community Event Recommender")
st.markdown("Discover local ways to contribute. Start with how you want to help and filter by intent, mood, or location.")

# === User Inputs ===
intent_input = st.text_input("🙋‍♀️ How can I help?", placeholder="e.g. help with homelessness, teach kids, plant trees")
zipcode_input = st.text_input("📍 Enter your ZIP code (optional)", placeholder="e.g. 10027")

# Mood becomes an optional filter
mood_options = ["Uplift", "Unwind", "Connect", "Empower", "Reflect"]
mood_input = st.selectbox("💫 Optional: Set an intent", ["(no preference)"] + mood_options)

# === Filter + Display ===
if st.button("🔍 Explore"):
    filtered = df.copy()

    # Apply fuzzy match if intent provided
    if intent_input.strip():
        input_clean = intent_input.lower().strip()
        filtered = filtered[
            filtered["Topical Theme"].str.lower().apply(lambda x: fuzz.partial_ratio(x, input_clean)) > 60 |
            filtered["Activity Type"].str.lower().apply(lambda x: fuzz.partial_ratio(x, input_clean)) > 60
        ]

    # Apply mood filter
    if mood_input != "(no preference)":
        filtered = filtered[filtered["Mood/Intent"].str.contains(mood_input, case=False, na=False)]

    # Final display
    st.subheader(f"📋 {len(filtered)} events match your search")
    st.dataframe(
        filtered[["short_description", "Topical Theme", "Effort Estimate", "Weather Badge"]].reset_index(drop=True),
        use_container_width=True
    )


