import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# gpt2 tokenizer
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token

# finetuned model 1
drug_info_model = AutoModelForCausalLM.from_pretrained('drug_info')

# finetuned model 2
drug_interaction_model = AutoModelForCausalLM.from_pretrained('drug_drug_interaction')

# Enable mixed precision if available
torch.backends.cudnn.benchmark = True

def drug_info_inference(text, model=drug_info_model, tokenizer=tokenizer, max_input_tokens=100, max_output_tokens=1000):
    
    input_ids = tokenizer.encode(
    text,
    return_tensors='pt',
    truncation=True,
    max_length=max_input_tokens,
    padding=True
    )
    
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids,
        max_length=max_output_tokens,
        do_sample=False 
    )
    
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)
    generated_text_answer = generated_text_with_prompt[0]
    
    return 'Thank you' + generated_text_answer.split('Thank you')[1][:-14]

def drug_interaction_inference(text1, text2, model=drug_interaction_model, tokenizer=tokenizer, max_input_tokens=100, max_output_tokens=500):
    
    input_ids = tokenizer.encode(
    text1+ text2,
    return_tensors='pt',
    truncation=True,
    max_length=max_input_tokens,
    padding=True
    )
    
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids,
        max_length=max_output_tokens,
        do_sample=False 
    )
    
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)
    generated_text_answer = generated_text_with_prompt[0]
    
    return 'Drug1:' + generated_text_answer.split('Drug1:')[1]
#'Drug1:' + out.split('Drug1:')[1]

def get_drug_info(drug_name):

    return drug_info_inference(drug_name)

def get_interaction_info(drug1, drug2):
    
    return drug_interaction_inference(drug1, drug2)

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        "<h1 style='text-align: center; font-size: 36px; font-family: Georgia, serif;'>DrugWise: Info & Interactions</h1>"
    )
    gr.Markdown(
        "<p style='text-align: center; font-size: 18px; font-family: Arial, sans-serif;'>Welcome to DrugWise! This tool helps you find information about individual drugs and check for potential interactions between two drugs. Simply enter the drug names and get the information you need.</p>"
    )


    # Create the tab structure
    with gr.Tab("Single Drug Info"):
        drug_input = gr.Textbox(label="Enter drug name:")
        drug_output = gr.Textbox(label="Drug Information:")
        drug_button = gr.Button("Submit")
        drug_button.click(get_drug_info, drug_input, drug_output)

    with gr.Tab("Drug Interaction Info"):
        drug1_input = gr.Textbox(label="Enter first drug name:")
        drug2_input = gr.Textbox(label="Enter second drug name:")
        interaction_output = gr.Textbox(label="Interaction Information:")
        interaction_button = gr.Button("Submit")
        interaction_button.click(get_interaction_info, [drug1_input, drug2_input], interaction_output)

# Launch the interface
demo.launch()

