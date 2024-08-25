import streamlit as st
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import time

# Set up the OpenAI LLM with the given API key
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What can I do for you?"):
    # Display user input in message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    def response_generator(history):
        output_parser = StrOutputParser()
        prompt_template = PromptTemplate.from_template("""
            You are ServerGenie. You are a superintelligent AI that answers questions
            on server management like a professional.

            You are:
            - helpful, friendly, and efficient
            - good at answering both simple and complex business questions in
            simple language and professional terms.
            - an expert in debugging server management problems.
            - able to infer the intent of the user's questions.

            The user will ask about server management and you will answer as a
            professional that you are.
            You will only answer questions on server management. You must not
            answer any question that has nothing to do with server management.
            If you can't answer the question or the question is not clear, you
            are to ask for more details.
            
            conversation history: {history}

            User: {request}
        """)
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" 
                                  for msg in st.session_state.messages])
        chain = LLMChain(llm=llm, prompt=prompt_template, output_parser=output_parser)
        response = chain.run(request=prompt, history=history_text)

        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    # Display assistant's response
    with st.chat_message("assistant"):
        response = ''.join(response_generator(st.session_state.messages))
        st.markdown(response)
        
    # Add assistant's response to message history
    st.session_state.messages.append({"role": "assistant", "content": response})
