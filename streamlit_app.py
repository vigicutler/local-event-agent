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

    # ✅ Merge enriched tags with raw metadata
    merged = pd.merge(
        enriched,
        raw,
        left_on="description",
        right_on="summary",
        how="left",
        suffixes=("", "_y")
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

    # === Match 1: Fuzzy match Topical Theme & Activity Type
    theme_matches = process.extract(input_clean, final_df["Topical Theme"].dropna().unique(), limit=5)
    act_matches = process.extract(input_clean, final_df["Activity Type"].dropna().unique(), limit=5)

    matched_tags = set([match[0] for match in theme_matches + act_matches if match[1] >= 50])

    df_tags = final_df[
        final_df["Topical Theme"].isin(matched_tags) |
        final_df["Activity Type"].isin(matched_tags)
    ]

    # === Match 2: Search keywords in description
    df_desc = final_df[
        final_df["description"].str.lower().str.contains(input_clean, na=False)
    ]

    # === Combine both methods
    filtered = pd.concat([df_tags, df_desc]).drop_duplicates()

    # === Apply mood filter
    if mood_input != "(no preference)":
        filtered = filtered[filtered["Mood/Intent"].str.contains(mood_input, case=False, na=False)]

    # === Apply ZIP code filter
    if zipcode_input.strip():
        filtered = filtered[filtered["Postcode"].astype(str).str.startswith(zipcode_input.strip())]

    # === Display Results
    st.subheader(f"🔍 Found {len(filtered)} matching events")

    for _, row in filtered.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row.get('title_y', 'Untitled Event')}")
            st.markdown(f"**Organization:** {row.get('org_title_y', 'Unknown')}")
            st.markdown(f"📍 **Location:** {row.get('primary_loc_y', 'N/A')}")
            st.markdown(f"🗓️ **Date:** {row.get('start_date_date_y', 'N/A')}")
            st.markdown(f"🏷️ **Tags:** {row.get('Topical Theme', '')}, {row.get('Effort Estimate', '')}, {row.get('Mood/Intent', '')}")
            st.markdown(f"📝 {row.get('short_description', '')}")
            st.markdown("---")

else:
    st.info("Enter your interest and click **Explore** to find matching events.")
