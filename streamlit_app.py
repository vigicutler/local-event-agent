# 📦 Streamlit App: Clean Version Aligned with Capstone Vision

import streamlit as st
import pandas as pd

# === Load Data ===
@st.cache_data

def load_data():
    df = pd.read_csv("Merged_Enriched_Events_CLUSTERED.csv")
    df["description"] = df["description"].fillna("")
    df["Mood/Intent"] = df["Mood/Intent"].fillna("")
    df["Topical Theme"] = df["Topical Theme"].fillna("")
    df["short_description"] = df["description"].str.slice(0, 120) + "..."
    return df

final_df = load_data()

# === UI Header ===
st.title("🔎 NYC Community Event Recommender")
st.markdown("Find events based on how you want to help — choose your impact and discover local opportunities.")

# === Intent + Mood Filters ===
intent_options = ["Educate", "Clean", "Green", "Support", "Create"]
mood_options = ["Uplift", "Unwind", "Connect", "Empower", "Reflect"]

intent = st.selectbox("🙋‍♀️ How can you help?", intent_options)
mood = st.selectbox("💫 Optional: How do you want to feel?", ["(no preference)"] + mood_options)

# === Search Button ===
if st.button("🔍 Show Matching Events"):
    # Filter by intent → match to Topical Theme or Activity Type
    df_filtered = final_df[
        final_df["Topical Theme"].str.contains(intent, case=False) |
        final_df["Activity Type"].str.contains(intent, case=False)
    ]

    # Apply optional mood filter
    if mood != "(no preference)":
        df_filtered = df_filtered[
            df_filtered["Mood/Intent"].str.contains(mood, case=False)
        ]

    # Display result count
    st.subheader(f"📋 {len(df_filtered)} matching events:")

    # Show select columns
    st.dataframe(
        df_filtered[["short_description", "Topical Theme", "Effort Estimate", "Weather Badge"]].reset_index(drop=True),
        use_container_width=True
    )


