import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import os
import pathlib
import textwrap
import google.generativeai as genai
import PIL.Image
from IPython.display import display
from IPython.display import Markdown
from nltk import word_tokenize
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

def page1():

    # Initialize Streamlit session state
    if 'document_analyzed' not in st.session_state:
        st.session_state.document_analyzed = False
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'question_responses' not in st.session_state:
        st.session_state.question_responses = []
    if 'all_text' not in st.session_state:  # Initialize all_text in session state
        st.session_state.all_text = ""
    if 'document_search' not in st.session_state:
        st.session_state.document_search = None

    # Streamlit app title
    st.title("Document Analysis and Q&A")

    # API key input
    api_key = st.secrets["api_key"]["api"]
    os.environ["GOOGLE_API_KEY"] = api_key

    # File uploader
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "csv", "xlsx", "docx", "pptx"])
    iam = st.text_input("Who are you? (Data Scientist, Data Analyst, BI Developer)")
    context = st.text_input("What's the document about?")
    output = st.text_input("What's your expectation?")
    summary_length = st.selectbox("Select Summary Length", ("2 Sentences", "5 Sentences", "10 Sentences"))

    # Analyze document button
    if st.button("Analyze Document"):
        if uploaded_file and api_key:
            try:
                # Configure API
                genai.configure(api_key=api_key)
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

                # Save the uploaded file
                file_path = "uploaded_file" + os.path.splitext(uploaded_file.name)[1]
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Determine file type and load accordingly
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif file_extension == ".csv":
                    loader = UnstructuredCSVLoader(file_path, mode="elements", encoding="utf8")
                elif file_extension == ".xlsx":
                    loader = UnstructuredExcelLoader(file_path, mode="elements")
                elif file_extension == ".docx":
                    loader = Docx2txtLoader(file_path)
                elif file_extension == ".pptx":
                    loader = UnstructuredPowerPointLoader(file_path)
                else:
                    st.error(f"Unsupported file type: {file_extension}")
                    st.stop()

                docs = loader.load()
                st.session_state.all_text = " ".join([doc.page_content for doc in docs])  # Store all_text in session state

                # Generate summary
                prompt_template = PromptTemplate.from_template(
                    f"I am an {iam}. This file is about {context}. Answer the question based on this file: {output}. Write a {summary_length} concise summary of the following text: {{text}}"
                )
                llm_chain = LLMChain(llm=llm, prompt=prompt_template)
                stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
                response = stuff_chain.invoke(docs)

                st.session_state.summary = response["output_text"]
                st.session_state.document_analyzed = True
                st.success("Document analyzed successfully!")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

    # Show summary if analyzed
    if st.session_state.document_analyzed:
        st.subheader("Summary")
        st.write(st.session_state.summary)

        # Question input
        question = st.text_input("Ask a question about the document:", key="question_input")

        if st.button("Get Answer"):
            if question:
                try:
                    # Initialize embeddings and vector store once
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_text(st.session_state.all_text)  # Use all_text from session state
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

                    # Create FAISS vector store
                    st.session_state.document_search = FAISS.from_texts(chunks, embeddings)

                    # Query the vector store
                    query_embedding = embeddings.embed_query(question)
                    results = st.session_state.document_search.similarity_search_by_vector(query_embedding, k=3)

                    if results:
                        retrieved_texts = " ".join([result.page_content for result in results])
                        
                        # RAG template for augmented response
                        rag_template = """
                        Based on the following retrieved context:
                        "{retrieved_texts}"
                        
                        Answer the question: {question}
                        
                        Answer:"""
                        rag_prompt = PromptTemplate(input_variables=["retrieved_texts", "question"], template=rag_template)
                        rag_llm_chain = LLMChain(llm=llm, prompt=rag_prompt)

                        # Get the RAG response
                        rag_response = rag_llm_chain.run(retrieved_texts=retrieved_texts, question=question)

                        # Save the question and response to session state
                        st.session_state.question_responses.append((question, rag_response))

                        # Display the LLM response from RAG in a conversational format
                        #st.markdown("### Assistant")
                        #st.write(rag_response)
                    else:
                        st.error("No matching document found in the database.")

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")

    # Display Q&A history
    if st.session_state.question_responses:
        st.subheader("Q&A History")
        for q, a in st.session_state.question_responses:
            st.write(f"**You:** {q}")
            st.write(f"**Assistant:** {a}")




def page2():
    if 'document_analyzed' not in st.session_state:
        st.session_state.document_analyzed = False
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'question_responses' not in st.session_state:
        st.session_state.question_responses = []


    st.title("Tabular Data Analysis")

    api_key = st.secrets["api_key"]["api"]
    os.environ["GOOGLE_API_KEY"] = api_key
    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload CSV or Excel file:", type=['csv', 'xlsx'])

    def clean_data(df):
        # Remove currency symbols and convert to float
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['value', 'price', 'cost', 'amount']):
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace('$', '', regex=False)
                    df[col] = df[col].str.replace('£', '', regex=False)
                    df[col] = df[col].str.replace('€', '', regex=False)
                    df[col] = df[col].replace('[^\d.-]', '', regex=True).astype(float)
        
        # Identify columns to drop based on null percentage
        null_percentage = df.isnull().sum() / len(df)
        columns_to_drop = null_percentage[null_percentage > 0.25].index

        # Also drop columns containing 'id', 'address', 'phone', 'longitude', 'latitude'
        columns_to_drop = columns_to_drop.union(df.columns[df.columns.str.contains('id|address|phone|longitude|latitude', case=False)])

        # Drop identified columns
        df.drop(columns=columns_to_drop, inplace=True)

        # Drop object columns with more than 15 unique values
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() > 15:
                df.drop(columns=[col], inplace=True)
        
        # Fill remaining null values with median for numeric columns
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if null_percentage[col] <= 0.25:
                    if df[col].dtype in ['float64', 'int64']:
                        median_value = df[col].median()
                        df[col].fillna(median_value, inplace=True)
        
        # Convert remaining object columns to lowercase
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.lower()
        
        return df



    # Check if the file is uploaded
    if uploaded_file is not None:
        # Read the CSV file into a Pandas DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        # Show the DataFrame
        st.dataframe(df)
        df = clean_data(df)

        target = st.selectbox("Select Target Variable for Analysis", df.columns)
        question = st.text_input("Additional Question you want to ask about the dataset...")

        if st.button("Process"):
            # Initialize or reset summary if needed
            if 'summary' not in st.session_state:
                st.session_state.summary = ""

            # Function to generate plots and responses
            def generate_plot_and_response(plot_type, img_path, hue=None):
                genai.configure(api_key=api_key)
                st.markdown(f"<h2 style='text-align: center; color: black;'>{plot_type}</h2>", unsafe_allow_html=True)
                img = PIL.Image.open(img_path)
                st.image(img, caption=plot_type, use_column_width=True)
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                response = model.generate_content([question + " You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
                response.resolve()
                st.write(response.text)
                st.session_state.summary += f"{plot_type} Response:\n{response.text}\n\n"

            # Countplot Barchart
            cat_vars = [col for col in df.select_dtypes(include='object').columns if df[col].nunique() > 1 and df[col].nunique() <= 10]
            num_cols = len(cat_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()
            
            for i, var in enumerate(cat_vars):
                top_values = df[var].value_counts().head(10).index
                filtered_df = df.copy()
                filtered_df[var] = df[var].apply(lambda x: x if x in top_values else 'Other')
                sns.countplot(x=var, data=filtered_df, ax=axs[i])
                axs[i].set_title(var)
                axs[i].tick_params(axis='x', rotation=90)

            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            fig.tight_layout()
            #st.pyplot(fig)
            fig.savefig("count.png")
            st.session_state.summary += f"![Countplot Barchart](count.png)\n\n"
            generate_plot_and_response("Countplot Barchart", "count.png")


            # Multiclass Countplot Barchart
            cat_vars = [col for col in df.select_dtypes(include='object').columns if col != target and df[col].nunique() > 1 and df[col].nunique() <= 10]
            num_cols = len(cat_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            for i, var in enumerate(cat_vars):
                top_values = df[var].value_counts().head(10).index
                filtered_df = df.copy()
                filtered_df[var] = df[var].apply(lambda x: x if x in top_values else 'Other')
                sns.countplot(x=var, data=filtered_df, ax=axs[i], hue=target)
                axs[i].set_title(var)
                axs[i].tick_params(axis='x', rotation=90)

            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            fig.tight_layout()
            #st.pyplot(fig)
            fig.savefig("multiclass_count.png")
            st.session_state.summary += f"![Multiclass Countplot Barchart](multiclass_count.png)\n\n"
            generate_plot_and_response("Multiclass Countplot Barchart", "multiclass_count.png")


            # Multiclass Histplot
            num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns]
            num_cols = len(num_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            for i, var in enumerate(num_vars):
                sns.histplot(df[var], ax=axs[i], kde=True)
                axs[i].set_title(var)
                axs[i].set_xlabel('')

            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            fig.tight_layout()
            #st.pyplot(fig)
            fig.savefig("hist.png")
            st.session_state.summary += f"![Multiclass Histplot](hist.png)\n\n"
            generate_plot_and_response("Histplot", "hist.png")
            

            # Histplot
            num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns]
            num_cols = len(num_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            for i, var in enumerate(num_vars):
                sns.histplot(data=df, x=var, hue=target, kde=True, ax=axs[i])
                axs[i].set_title(var)
                axs[i].set_xlabel('')

            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            fig.tight_layout()
            #st.pyplot(fig)
            fig.savefig("hist_target.png")
            st.session_state.summary += f"![Histplot](hist_target.png)\n\n"
            generate_plot_and_response("Multiclass Histplot", "hist_target.png")

            st.session_state.document_analyzed = True
            st.success("Document analyzed successfully!")

    # Show summary if analyzed
    if st.session_state.document_analyzed == True:

        # Question input
        question = st.text_input("Ask a question about the document:", key="question_input")

        if st.button("Get Answer"):
            if question:
                try:
                    genai.configure(api_key=api_key)
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

                    # Save the uploaded file
                    file_path = "uploaded_file" + os.path.splitext(uploaded_file.name)[1]
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load the CSV or Excel using the appropriate loader
                    if uploaded_file.name.endswith('.csv'):
                        loader = UnstructuredCSVLoader(file_path, mode="elements", encoding="utf8")
                    elif uploaded_file.name.endswith('.xlsx'):
                        loader = UnstructuredExcelLoader(file_path, mode="elements")
                    docs = loader.load()
                    st.session_state.all_text = " ".join([doc.page_content for doc in docs])

                    # Initialize embeddings and vector store
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_text(st.session_state.all_text)

                    # Create FAISS vector store
                    st.session_state.document_search = FAISS.from_texts(chunks, embeddings)

                    # Query the vector store
                    query_embedding = embeddings.embed_query(question)
                    results = st.session_state.document_search.similarity_search_by_vector(query_embedding, k=3)

                    if results:
                        retrieved_texts = " ".join([result.page_content for result in results])

                        # RAG template for augmented response
                        rag_template = """
                        Based on the following retrieved context:
                        "{retrieved_texts}"
                                
                        Answer the question: {question}
                                
                        Answer:"""
                        rag_prompt = PromptTemplate(input_variables=["retrieved_texts", "question"], template=rag_template)
                        rag_llm_chain = LLMChain(llm=llm, prompt=rag_prompt)

                        # Get the RAG response
                        rag_response = rag_llm_chain.run(retrieved_texts=retrieved_texts, question=question)

                        # Save the question and response to session state
                        st.session_state.question_responses.append((question, rag_response))

                    else:
                        st.error("No matching document found in the database.")

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")

    # Display Q&A history
    if st.session_state.question_responses:
        st.subheader("Q&A History")
        for q, a in st.session_state.question_responses:
            st.write(f"**You:** {q}")
            st.write(f"**Assistant:** {a}")



# Use a session state variable to keep track of the selected page
session_state = st.session_state
if 'selected_page' not in session_state:
    session_state.selected_page = "Classification Prediction"

st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #222021;
            color: #f1f2f6;
            width: 100%;
            transition: background-color 0.3s, box-shadow 0.3s;
        }
        div.stButton > button:hover {
            background-color: #222021;
            color: #ff5c5d;  /* Adjust the text color on hover */
            width: 100%;
            box-shadow: 0 0 5px 2px #ff5c5d, 0 0 10px 5px #ff5c5d, 0 0 15px 7.5px #ff5c5d;
        }
        [data-testid="stSidebar"] {
            background: #222021;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation with styled buttons
page_1_button = st.sidebar.button("Chat with Your File", key="page1", type="primary")
page_2_button = st.sidebar.button("Tabular Analysis", key="page2", type="primary")


# Display selected page based on the button clicked
if page_1_button:
    session_state.selected_page = "Chat with Your File"

if page_2_button:
    session_state.selected_page = "Tabular Analysis"



# Inject the button style into the app
button_style = """
    <style>
        .css-1vru0uf {
            border: none !important;
            color: white;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            background-color: transparent !important;
        }
        .css-1vru0uf:hover {
            background-color: red;
        }
    </style>
"""

# Display selected page
if session_state.selected_page == "Chat with Your File":
    page1()
elif session_state.selected_page == "Tabular Analysis":
    page2()

# Inject the button style into the app
st.markdown(button_style, unsafe_allow_html=True)