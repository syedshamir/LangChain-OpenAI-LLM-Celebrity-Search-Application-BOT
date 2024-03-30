## Integrate OPENAI API in code

import os
from constant import openai_key
#from langchain.llms import OpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["OPENAI_API_KEY"] =  openai_key
st.title('Celebrity Search Results')
input_text = st.text_input("Search the topic u want")

#First PROMPT TEMPLATE

first_input_prompt = PromptTemplate(
   input_variables =['name'],
   template = "Tell me about celebrity {name}"

)

#Memory so that LLM will remember the conversation
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')


# OPENAI LLMS
#initialize openapi llm model
llm = OpenAI(temperature = 0.8) #by default is 0.7, temperature range b/w 0 - 1, tells how much control the agent should have while providing the response, or how much balanced answer you want
chain  = LLMChain(
    llm = llm, prompt = first_input_prompt, verbose = True, output_key = 'person', memory = person_memory)

#2nd PROMPT TEMPLATES

second_input_promt = PromptTemplate(
    input_variables = ['person'],
    template = "when was {person} born"
    )

chain2 = LLMChain(

    llm = llm, prompt=second_input_promt, verbose = True, output_key = 'dob', memeory = dob_memory)

#Third Prompt template

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention 5 major events happened around {dob} in the world"
    )

chain3 = LLMChain(
    llm = llm, prompt = third_input_prompt, verbose = True, output_key = 'description', memory = descr_memory)

chain3 = LLMChain(llm = llm, prompt = third_input_prompt, verbose = True, output_key = 'description')
parent_chain = SequentialChain(
chains = [chain, chain2, chain3],input_variables = ['name'], output_variables = ['person', 'dob', 'description'], verbose = True

    )


if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)
 
