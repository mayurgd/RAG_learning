## Importing Packages ##
import time
import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

# --------------------------------------------------------------------------------------------#

# FastAPI server URLs
FASTAPI_URL_QUERY = "http://127.0.0.1:8000/generate-response/"

# --------------------------------------------------------------------------------------------#

# Page settings
st.set_page_config(
    page_title="RAG ChatBot", layout="wide", initial_sidebar_state="expanded"
)

# Hide Streamlit default elements
hide_streamlit_style = """
                       <style>            
                       #MainMenu {visibility: hidden;}            
                       footer {visibility: hidden;}
                       .block-container {padding-top: 2.41rem;}        
                       </style>            
                       """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Hide pages display in sidebar
no_pages_display_in_sidebar_style = """
                                    <style>
                                    div[data-testid="stSidebarNav"] {display: none;}
                                    </style>
                                    """
st.markdown(no_pages_display_in_sidebar_style, unsafe_allow_html=True)

# Title display in a central column
col1, col2, col3, col4, col5 = st.columns([0.2, 0.3, 3, 0.1, 0.4])
with col1:
    pass
with col2:
    pass
with col3:
    st.write(
        '<p style="font-size:320%; color:Green; background-color:Lavender; text-align: center"><strong>RAG ChatBot</strong></p>',
        unsafe_allow_html=True,
    )
with col4:
    pass

st.write("---" * 20)
st.write("")

# --------------------------------------------------------------------------------------------#


# clear the chat history from streamlit session state
def clear_history():
    if "messages" in st.session_state:
        del st.session_state["messages"]


# --------------------------------------------------------------------------------------------#

# Sidebar with menu options
st.sidebar.write("")
st.sidebar.write("")

with st.sidebar:
    sidebar_options = option_menu(
        menu_title=None,
        options=["Ask Question", "About the App"],
        icons=["question-circle", "info-circle"],
        default_index=0,
        styles={
            "icon": {"font-size": "14px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "font-weight": "bold",
            },
        },
    )

# Sidebar actions for "Ask Question"
if sidebar_options == "Ask Question":

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])

    # user's question text input widget
    prompt = st.chat_input("Ask Something...", key="ques")

    if prompt:
        # Prepare the payload as JSON
        payload = {"query": prompt}
        print(payload)
        # Send the query to the FastAPI endpoint
        response = requests.post(
            "http://127.0.0.1:8000/generate-response/", json=payload
        )

        if response.status_code == 200:
            response_data = response.json()
            response_answer = response_data["response"]

            # display user response in chat message container
            with st.chat_message("user"):
                st.write(prompt)

            # add user message to chat history
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            # display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # Simulate stream of response with milliseconds
                for chunk in response_answer.split():
                    full_response += chunk + " "
                    time.sleep(0.05)

                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            # add assistant message to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                }
            )

            with st.sidebar:
                st.write("---" * 20)
                history_button = st.button("Clear Chat History", on_click=clear_history)


# about the app page
elif sidebar_options == "About the App":
    switch_page("about")

else:
    pass
