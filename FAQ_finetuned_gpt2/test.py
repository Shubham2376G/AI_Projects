import streamlit as st

from train import train1
# Markdown content with a hyperlink
# markdown_text = """
# # Example Header
#
# This is a normal text with a [clickable link](https://www.example.com).
#
# You can also have:
# - [Streamlit Documentation](https://docs.streamlit.io)
# - [GitHub](https://github.com)
# """
#
# # Render the Markdown in Streamlit
# st.markdown(markdown_text, unsafe_allow_html=True)



# Function to respond to the query
def get_response(query):
    # A simple dictionary-based response system (you can extend this)


    # Return a default message if the query is not in the dictionary
    return train1(query)


# Streamlit app UI
st.title("Query Response App")

# Input field for the user query
query = st.text_area("Ask me something:")

# If a query is entered, display the response
if query:
    response = get_response(query)
    st.write("Response:", response)