import streamlit as st
from langchain_community.llms import OpenAI
import pandas as pd
import pdfplumber


st.title('Hallu App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# PDF file upload
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

#Extract text from PDF
document_text = ""
if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        document_text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())


def generate_response(input_text, context):
    if context:
        # Prepend context to the query
        input_text = context + "\n\n" + input_text

    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    response = llm.generate([input_text])  
    return response

with st.form('my_form'):
    text = st.text_area('Enter text:', 'How do you know if someone is hallucinating?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text, context=document_text)
