import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('Prudvireddy/gpt2-drug-info')
tokenizer.pad_token = tokenizer.eos_token

# Load models
drug_info_model = AutoModelForCausalLM.from_pretrained('Prudvireddy/gpt2-drug-info')
drug_interaction_model = AutoModelForCausalLM.from_pretrained('Prudvireddy/gpt2-drug-interaction')

def drug_info_inference(text, max_input_tokens=100, max_output_tokens=200):
    input_ids = tokenizer.encode(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_input_tokens,
        padding=True
    )

    generated_tokens_with_prompt = drug_info_model.generate(
        input_ids=input_ids,
        max_length=max_output_tokens,
        num_beams=3,  # Use a small number of beams for beam search
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )

    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)
    generated_text_answer = generated_text_with_prompt[0]

    if 'Thank you' in generated_text_answer:
        return 'Thank you' + generated_text_answer.split('Thank you')[1].split('Thank')[0]
    return generated_text_answer

def drug_interaction_inference(text1, text2, max_input_tokens=100, max_output_tokens=200):
    input_ids = tokenizer.encode(
        text1 + ' ' + text2,
        return_tensors='pt',
        truncation=True,
        max_length=max_input_tokens,
        padding=True
    )

    generated_tokens_with_prompt = drug_interaction_model.generate(
        input_ids=input_ids,
        max_length=max_output_tokens,
        num_beams=3,  # Use a small number of beams for beam search
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )

    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)
    generated_text_answer = generated_text_with_prompt[0]

    if 'Drug1:' in generated_text_answer:
        return 'Drug1:' + generated_text_answer.split('Drug1:')[1]
    return generated_text_answer

st.title("DrugWise: Info & Interactions")
st.write("Welcome to DrugWise! This tool helps you find information about individual drugs and check for potential interactions between two drugs. Simply enter the drug names and get the information you need.")

tab1, tab2 = st.tabs(["Drug Info", "Drug Interaction Info"])

with tab1:
    drug_name = st.text_input("Enter drug name:")
    if st.button("Submit"):
        result = drug_info_inference(drug_name)
        st.text_area("Drug Information:", result, height=250)

with tab2:
    drug1_name = st.text_input("Enter first drug name:")
    drug2_name = st.text_input("Enter second drug name:")
    if st.button("Submit", key="interaction"):
        result = drug_interaction_inference(drug1_name, drug2_name)
        st.text_area("Interaction Information:", result, height=250)
