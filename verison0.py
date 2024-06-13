import streamlit as st
from datasets import load_dataset
import pandas as pd

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

# Function to display the current sample
def display_sample(sample_index, data):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Query")
        st.write(data.at[sample_index, 'query'])

    with col2:
        st.header("Answer")
        label = data.at[sample_index, 'label']
        answer = data.at[sample_index, 'answer']
        
        if label == 'attributable':
            st.markdown(f"<span style='color: green;'>{answer}</span>", unsafe_allow_html=True)
        elif label == 'extrapolatory':
            st.markdown(f"<span style='color: orange;'>{answer}</span>", unsafe_allow_html=True)
        elif label == 'contradictory':
            st.markdown(f"<span style='color: red;'>{answer}</span>", unsafe_allow_html=True)
        else:
            st.write(answer)

    with col3:
        st.header("Label")
        st.write(label)

    st.write("References")
    reference = data.at[sample_index, 'reference']
    st.markdown(reference)

# Main function to display the app
def main():
    st.title("AttrScore Dataset Visualization")

    # Load data
    data = load_data()

    # Display the current sample
    display_sample(st.session_state.sample_index, data)

    # Button to display the next sample
    if st.button("Next Sample"):
        if st.session_state.sample_index < len(data) - 1:
            st.session_state.sample_index += 1
        else:
            st.session_state.sample_index = 0  # Reset to the first sample if at the end

if __name__ == "__main__":
    main()
