import markdown
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_id = "Narrativaai/BioGPT-Large-finetuned-chatdoctor"
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
model = AutoModelForCausalLM.from_pretrained(model_id)


def answer_question(
    prompt, temperature=0.1, top_p=0.75, top_k=40, num_beams=2, **kwargs
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, **kwargs
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return output.split(" Response:")[1]


st.set_page_config(page_title="Talk To Me", page_icon=":ambulance:", layout="wide")

colored_header(
    label="Talk To Me",
    description="Talk your way to better health",
    color_name="violet-70",
)

# st.title("Talk To Me")
# st.caption("Talk your way to better health")

# add sidebar
with open("ui/sidebar.md", "r") as sidebar_file:
    sidebar_content = sidebar_file.read()

with open("ui/styles.md", "r") as styles_file:
    styles_content = styles_file.read()


def add_sbg_from_url():
    st.markdown(
        f"""
         <style>
         .css-6qob1r {{
             background-image: url("https://images.unsplash.com/photo-1524169358666-79f22534bc6e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=3540&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


add_sbg_from_url()


def add_mbg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1536353602887-521e965eb03f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=987&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


add_mbg_from_url()


# Display the sidebar content
st.sidebar.markdown(sidebar_content)

st.write(styles_content, unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# display default message if no chat history
if not st.session_state.chat_history:
    message("Hi, I'm a medical chat bot. Ask me a question!")

# Display the chat history
for chat in st.session_state.chat_history:
    if chat["is_user"]:
        message(chat["message"], is_user=True)
    else:
        message(chat["message"])

with st.form("user_input_form"):
    st.write("Please enter your question below:")
    user_input = st.text_input("You: ")

    # Check if user has submitted a question
    if st.form_submit_button("Submit") and user_input:
        with st.spinner('Loading model and generating response...'):
        # Generate response and update chat history
            bot_response = answer_question(f"Input: {user_input}\nResponse:")
            st.session_state.chat_history.append({"message": user_input, "is_user": True})
            st.session_state.chat_history.append(
                {"message": bot_response, "is_user": False}
            )

# Display the latest chat in the chat history
if st.session_state.chat_history:
    latest_chat = st.session_state.chat_history[-1]
    if latest_chat["is_user"]:
        message(latest_chat["message"], is_user=True)
    else:
        message(latest_chat["message"])
