import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="AI Text Generator",
    layout="centered"
)

# Custom CSS
st.markdown(
    """
    <style>

    /* Hide chat avatars */
    [data-testid="stChatMessageAvatar"] {
        display: none;
    }

    /* Remove extra left spacing */
    [data-testid="stChatMessage"] {
        padding-left: 0;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown(
    '<div class="main-title">AI Text Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enter a sentence and let AI complete the text.</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:

    st.header("Generation Settings")

    max_tokens = st.slider(
        "Maximum New Tokens",
        min_value=20,
        max_value=150,
        value=50,
        step=10
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1
    )

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Load model
@st.cache_resource
def load_model():

    return pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-125M"
    )


generator = load_model()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Chat input
prompt = st.chat_input("Enter your text...")


if prompt:

    # Store user prompt
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display user prompt
    with st.chat_message("user"):
        st.write(prompt)

    # Generate text
    with st.chat_message("assistant"):

        with st.spinner("Generating..."):

            result = generator(
                prompt,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature,
                top_p=0.9
            )

            generated_text = result[0]["generated_text"]

            # Remove original prompt
            new_text = generated_text[len(prompt):].strip()

        if new_text:

            st.write(new_text)

            # Store generated text
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": new_text
                }
            )

        else:

            st.warning("No additional text was generated.")