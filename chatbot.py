import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer (DialoGPT-small)
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Custom Local Chatbot with DialoGPT")

if "chat_history_ids" not in st.session_state:
    st.session_state["chat_history_ids"] = None

if "step" not in st.session_state:
    st.session_state["step"] = 0

def generate_response(user_input):
    # encode the user input and append to chat history (if any)
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # generate a response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # save new chat history
    st.session_state.chat_history_ids = chat_history_ids

    # decode the last output tokens
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response

user_input = st.text_input("You:", key="input")

if user_input:
    response = generate_response(user_input)
    st.text_area("Bot:", value=response, height=200, max_chars=None)
