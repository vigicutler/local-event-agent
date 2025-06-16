
import streamlit as st

st.set_page_config(page_title="Neighborhood AI Agent", layout="wide")

st.title("🌆 Neighborhood AI Agent")

tab1, tab2 = st.tabs(["Public View: What Should I Do Today?", "Producer View: What Event Should I Host?"])

with tab1:
    st.header("Find Local Events Near You")
    st.text_input("Neighborhood or ZIP Code", placeholder="e.g. 10034")
    st.selectbox("What kind of vibe are you in the mood for?", ["🌿 Nature", "🎭 Arts", "🎤 Open Mic", "💡 Workshop", "🧹 Volunteer"])
    st.radio("Weather preference", ["Any", "Clear skies only"])
    st.button("Get Recommendations")

with tab2:
    st.header("Plan a Local Event Based on Community Gaps")
    st.text_input("Neighborhood or ZIP Code", placeholder="e.g. 10034")
    st.selectbox("Event Format", ["🧠 Trivia", "🧘 Yoga", "🎶 Music", "👪 Family"])
    st.radio("Ideal Day", ["Friday", "Saturday", "Sunday"])
    st.button("Suggest Event Idea")
