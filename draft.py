import streamlit as st
from datasets import load_dataset
import pandas as pd
from langchain_openai import OpenAI
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
import os


# Set key gpt-3.5-turbo-instruct
#os.environ["OPENAI_API_KEY"] = ''

# Ensure the API key is set in the environment
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

#Load the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")


# Load the dataset from Hugging Face
@st.cache_data
def load_data():
    # Load the specific subset 'attreval_gensearch' from the dataset
    dataset = load_dataset('osunlp/AttrScore', 'attreval_gensearch', split='test')
    
    # Convert the dataset to a Pandas DataFrame
    data = pd.DataFrame(dataset)
    return data

# Initialize session state for the sample index
if 'sample_index' not in st.session_state:
    st.session_state.sample_index = 0

# Custom chain using LangChain
def create_langchain_pipeline():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", api_key=os.getenv("OPENAI_API_KEY"))
    
    template = """
    Given the answer: "{answer}", the label: "{label}", and the reference: "{reference}",
    identify the relevant parts of the reference that support, contradict, or are exploratory with respect to the answer. 
    Based on the label type, retrieve the words in the answer that correspond to the supporting, contradicting, or exploratory parts of the reference. 
    Return the text segments that should be highlighted in the answer and reference based on the label type in the following format:
    Answer segments: "segment1"; "segment2"; ...
    Reference segments: "segment1"; "segment2"; ...
    """
    
    prompt = PromptTemplate(template=template, input_variables=["answer", "reference", "label"])
    chain = RunnableSequence(prompt | llm)
    
    return chain

# Function to process text with LangChain and get segments to highlight
def process_text_with_langchain(answer, reference, label, chain):
    response = chain.invoke({"answer": answer, "reference": reference, "label": label})
    st.write("LangChain Response:", response)  # Debugging line to check the response
    highlight_segments = {"answer": [], "reference": []}
    try:
        # Parse the response to get segments to highlight
        if isinstance(response, str):
            response = response.strip()
            parts = response.split("Reference segments:")
            if len(parts) == 2:
                answer_highlights = parts[0].replace("Answer segments:", "").strip().split(";")
                reference_highlights = parts[1].strip().split(";")
                highlight_segments["answer"] = [seg.strip().strip('"') for seg in answer_highlights]
                highlight_segments["reference"] = [seg.strip().strip('"') for seg in reference_highlights]
            else:
                st.write("Unexpected response format:", response)  # Debugging
    except Exception as e:
        st.write("Error parsing response:", e)  # Debugging 

    return highlight_segments

# Function to highlight text using HTML to change text color based on segments
def highlight_text_html(text, highlight_segments, color):
    for segment in highlight_segments:
        text = text.replace(segment, f'<span style="color:{color};">{segment}</span>')
    return text

# Function to display the current sample with highlighting
def display_sample(sample_index, data, chain):
    st.write(f"### Example {sample_index + 1}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Query")
        st.write(data.at[sample_index, 'query'])
        
    with col2:
        st.header("Response")
        answer = data.at[sample_index, 'answer']
        label = data.at[sample_index, 'label']
        
        highlight_segments = process_text_with_langchain(answer, data.at[sample_index, 'reference'], label, chain)
        
        if label == 'Attributable':
            color = "green"
        elif label == 'Extrapolatory':
            color = "yellow"
        elif label == 'Contradictory':
            color = "red"
        else:
            color = "white"  # Default 
        
        highlighted_answer = highlight_text_html(answer, highlight_segments.get('answer', []), color)
        st.markdown(highlighted_answer, unsafe_allow_html=True)
        
    with col3:
        st.header("Reference")
        reference = data.at[sample_index, 'reference']
        highlighted_reference = highlight_text_html(reference, highlight_segments.get('reference', []), color)
        st.markdown(highlighted_reference, unsafe_allow_html=True)
    
    # Second row for the label, centered
    st.markdown("<br>", unsafe_allow_html=True)  # Add a line break for spacing
    st.markdown(f"<h3 style='text-align: center; color: {color};'>{label}</h3>", unsafe_allow_html=True)

# Main function to display the app
def main():
    st.title("AttrScore Dataset Visualization")

    # Load data
    data = load_data()

    # Create LangChain pipeline
    chain = create_langchain_pipeline()

    # Display the current sample
    display_sample(st.session_state.sample_index, data, chain)

    # Button to display the next sample
    if st.button("Next Sample"):
        if st.session_state.sample_index < len(data) - 1:
            st.session_state.sample_index += 1
        else:
            st.session_state.sample_index = 0  # Reset to the first sample if at the end

if __name__ == "__main__":
    main()
