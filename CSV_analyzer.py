import streamlit as st
import pandas as pd
from openai import OpenAI

# ---------------- OLLAMA CLIENT ----------------
client = OpenAI(
    base_url="http://localhost:11434/v1"
    
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="CSV AI Assistant", layout="wide")

st.title("CSV AI Assistant")
st.write("Upload CSV and ask questions in plain English")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ---------------- CHAT HISTORY ----------------
    if "chat" not in st.session_state:
        st.session_state.chat = []

    question = st.text_input("💬 Ask your question")

    if question:
        st.session_state.chat.append(("user", question))

        with st.spinner("🤖 Thinking..."):
            try:
                # Limit rows for performance
                data_sample = f"""
                   Columns: {df.columns.tolist()}

                   Sample Rows:
                   {df.head(15).to_string(index=False)}

                   Basic Stats:
                   {df.describe(include='all').to_string()}
                   """
                prompt = f"""
                You are a data analyst.

                Dataset:
                {data_sample}

                Question: {question}

                Instructions:
                - Answer clearly
                - Perform calculations if needed
                - Keep answer short
                """

                response = client.chat.completions.create(
                    model="llama3",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )

                answer = response.choices[0].message.content

                st.session_state.chat.append(("ai", answer))

            except Exception as e:
                st.error(f"Error: {e}")

    # ---------------- DISPLAY CHAT ----------------
    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**AI:** {msg}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🔥 Built with Streamlit + Pandas + Ollama (No API limits)")