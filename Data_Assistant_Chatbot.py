from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma

import streamlit as st
import pandas as pd

import os
import re
from dotenv import load_dotenv
import groq

import plotly.express as px

from core_utils import preprocessing

#--------------------------------------------------------------------------DATASET PREPARATION--------------------------------------------------------------------------
RAW_CSV = "Orders.csv"
CLEANED_PARQUET = "Orders_Cleaned.parquet"
df = preprocessing(RAW_CSV, CLEANED_PARQUET)
print(df.shape[0])


@st.cache_resource
def get_vector_store():
    VECTOR_DB_PATH = "chroma_db"
    embedding_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small')

    vector_store = Chroma(
        embedding_function = embedding_model,
        persist_directory = VECTOR_DB_PATH,
    )
    return vector_store

#--------------------------------------------------------------------------LLM FUNCTIONALITY--------------------------------------------------------------------------
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = groq.Groq(api_key = GROQ_API_KEY)
llm = init_chat_model('llama-3.1-8b-instant', model_provider = "groq")


def extract_quarter(query):
    match = re.search(r"quarter\s*(\d+)", query, re.IGNORECASE)
    return int(match.group(1)) if match else None

def extract_month(query):
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    for month, num in months.items():
        if month in query.lower():
            return num
    return None

def extract_year(query):
    match = re.search(r"(20\d{2})", query)
    return int(match.group(1)) if match else None

def get_procurement_data(query):
    query_lower = query.lower()

    # --- Extract filters
    month = extract_month(query_lower)
    quarter = extract_quarter(query_lower)
    year = extract_year(query_lower)

    filtered_df = df.copy()

    # --- Apply filters
    if month:
        filtered_df = filtered_df[filtered_df['Purchase Month'] == month]
    if quarter:
        filtered_df = filtered_df[filtered_df['Purchase Quarter'].str.contains(f"Q{quarter}", case=False, na=False)]
    if year:
        filtered_df = filtered_df[filtered_df['Purchase Year'] == year]

    # --- Dynamic metrics and columns ---
    # Count of orders
    if "total number of orders" in query_lower or "how many orders" in query_lower:
        return f"Total number of orders: {filtered_df.shape[0]}"

    # Highest or lowest spending period
    if "highest spending" in query_lower or "maximum spending" in query_lower:
        grouped = df.groupby(['Purchase Year', 'Purchase Quarter'], observed = True)['Total Price'].sum()
        year_q, max_total = grouped.idxmax(), grouped.max()
        return f"Quarter {year_q[1]} of {year_q[0]} had the highest spending: ${max_total:,.2f}"

    if "lowest spending" in query_lower or "minimum spending" in query_lower:
        grouped = df.groupby(['Purchase Year', 'Purchase Quarter'], observed = True)['Total Price'].sum()
        year_q, min_total = grouped.idxmin(), grouped.min()
        return f"Quarter {year_q[1]} of {year_q[0]} had the lowest spending: ${min_total:,.2f}"

    # Most frequently ordered item
    if "most frequently ordered" in query_lower or "frequently ordered" in query_lower:
        if 'Item Name' in df.columns:
            most_common = filtered_df['Item Name'].value_counts().idxmax()
            count = filtered_df['Item Name'].value_counts().max()
            return f"The most frequently ordered item is '{most_common}' with {count} orders."

    # Most expensive item
    if "most expensive item" in query_lower:
        max_row = filtered_df.loc[filtered_df['Total Price'].idxmax()]
        return f"The most expensive order is '{max_row['Item Name']}' costing ${max_row['Total Price']:,.2f}."

    # Supplier with most orders
    if "most orders by supplier" in query_lower or "most frequent supplier" in query_lower:
        if 'Supplier Name' in df.columns:
            top_supplier = filtered_df['Supplier Name'].value_counts().idxmax()
            count = filtered_df['Supplier Name'].value_counts().max()
            return f"Top supplier: {top_supplier} with {count} orders."

    # Orders from a specific department
    if "orders by department" in query_lower:
        if 'Department Name' in df.columns:
            dept_counts = filtered_df['Department Name'].value_counts()
            top_dept = dept_counts.idxmax()
            return f"The department with most orders is '{top_dept}' with {dept_counts.max()} orders."

    return "Sorry, I couldn't understand the query or the data is unavailable."


def llm_based_answer(result, query):
    examples_text = ""
    for doc in result:
        examples_text += f"""
        - Item Name: {doc.get('Item Name', 'N/A')}
          Description: {doc.get('embedded_text', 'N/A')}
          Department: {doc.get('Department Name', 'N/A')}
          Quantity: {doc.get('Quantity', 'N/A')}
          Total Price: ${doc.get('Total Price', 'N/A')}
          Purchase Date: {doc.get('Purchase Date', 'N/A')}
          Month: {doc.get('Purchase Month', 'N/A')}
          Quarter: {doc.get('Purchase Quarter', 'N/A')}
          Year: {doc.get('Purchase Year', 'N/A')}
        """

    messages = [
        {'role': 'system', 'content': """
            You are a procurement data assistant.

            You will be given procurement records in structured format (list of dictionaries). Analyze the data and answer the user's question using only that data.

            ✅ Be concise.
            ✅ Show numeric values with units (e.g., $, %, etc.)
            ✅ Format the answer as bullet points or numbered list if applicable.
            ❌ Do NOT make assumptions or hallucinate.
            """
        },

        {'role': 'user', 'content': f"""
            User Question: {query}

            Data:
            {examples_text}

            Now answer the question based strictly on the data.
            """
        }
    ]

    try:
        response = llm.invoke(messages)
    except Exception as e:
        return f"LLM error: {e}"

    return response.content.strip()


def handle_query(query, vector_store):
    structured_answer = get_procurement_data(query)

    if structured_answer and "couldn't understand" not in structured_answer.lower():
        return structured_answer

    # Fall back to LLM-powered vector search
    retrieved_docs = vector_store.similarity_search_with_score(query, k=5)
    result = []
    for doc, score in retrieved_docs:
        meta = doc.metadata
        meta['embedded_text'] = doc.page_content
        meta['score'] = score
        result.append(meta)

    return llm_based_answer(result, query)


#---------------------------------------------------------------------------STREAMLIT APP---------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    h1, h3 {
        text-align: center;
    }
    .stDataFrame {
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📊 Visual Insights", "🤖 LLM Queries"])

# -------------------------- TAB 1: GRAPHS --------------------------
with tab1:
    st.header("📈 Procurement Visualizations")

    # Total spending over years
    spending_by_year = df.groupby("Purchase Year", observed = True)["Total Price"].sum().sort_index()
    st.subheader("💰 Total Spending by Year")
    st.bar_chart(spending_by_year)

    # Spending per department
    top_departments = df.groupby("Department Name", observed = True)["Total Price"].sum().sort_values(ascending=False).head(10)
    st.subheader("🏢 Top 10 Departments by Total Spending")
    st.bar_chart(top_departments)

    st.subheader("📌 Department-wise Orders (Top 10)")
    top_depts = df['Department Name'].value_counts().head(10).reset_index()
    top_depts.columns = ['Department', 'Order Count']
    fig1 = px.pie(top_depts, values='Order Count', names='Department', title="Top Departments by Orders")
    fig1.update_traces(textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)

    # Monthly order volume trend
    monthly_trend = df.groupby(["Purchase Year", "Purchase Month"], observed = True).size().reset_index(name="Order Count")
    monthly_trend["Date"] = pd.to_datetime(dict(year=monthly_trend["Purchase Year"], month=monthly_trend["Purchase Month"], day=1))
    monthly_trend_sorted = monthly_trend.sort_values("Date")
    st.subheader("📅 Order Volume Trend Over Time")
    st.line_chart(monthly_trend_sorted.set_index("Date")["Order Count"])

    # CalCard usage breakdown
    calcard_counts = df["CalCard"].value_counts().rename(index={True: "Used", False: "Not Used"})
    st.subheader("💳 CalCard Usage")
    st.bar_chart(calcard_counts)

    st.subheader("📌 CalCard Usage Breakdown")
    calcard = df['CalCard'].map({True: 'Yes', False: 'No'}).value_counts().reset_index()
    calcard.columns = ['CalCard Used', 'Count']
    fig4 = px.pie(calcard, values='Count', names='CalCard Used', title="CalCard Usage")
    fig4.update_traces(textinfo='percent+label')
    st.plotly_chart(fig4, use_container_width=True)

    # Top 10 most frequently ordered items
    top_items = df["Item Name"].value_counts().head(10)
    st.subheader("📦 Top 10 Most Frequently Ordered Items")
    st.bar_chart(top_items)

    st.subheader("📌 Supplier Distribution (Top 10)")
    top_suppliers = df['Supplier Name'].value_counts().head(10).reset_index()
    top_suppliers.columns = ['Supplier', 'Orders']
    fig2 = px.pie(top_suppliers, values='Orders', names='Supplier', title="Top Suppliers")
    fig2.update_traces(textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📌 Acquisition Type Breakdown")
    acq_counts = df['Acquisition Type'].value_counts().reset_index()
    acq_counts.columns = ['Acquisition Type', 'Count']
    fig3 = px.pie(acq_counts, values='Count', names='Acquisition Type', title="Acquisition Type Share")
    fig3.update_traces(textinfo='percent+label')
    st.plotly_chart(fig3, use_container_width=True)


# -------------------------- TAB 2: LLM QUERIES --------------------------
with tab2:
    st.header("🤖 Ask Questions About Procurement Data")

    st.markdown("Try asking:\n- 'What is the most expensive item?'\n- 'Who is the top supplier in Q2 2014?'\n- 'How many orders were made in March 2015?'")

    query = st.text_input("Enter your query", placeholder="e.g. Show me the most expensive item in 2014")

    if st.button("Submit") and query.strip():
        with st.spinner("Processing your query..."):
            try:
                vector_store = get_vector_store()
                answer = handle_query(query, vector_store)
                st.success("Answer:")
                st.markdown(f"**{answer}**")
            except Exception as e:
                st.error(f"❌ Error: {e}")
