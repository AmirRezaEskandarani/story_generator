import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, TensorDataset
import torch


# Load the tokenizer and model from the saved directory
model_name = "D:/______________projects/story final/story-gen-model"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)     


# generate story
def generate_text(prompt,k=0,p=0.9,output_length=300,temperature=1,num_return_sequences=1,repetition_penalty=1.0):
    # print("====prompt====\n")
    # print(prompt+"\n")
    # print('====target story is as below===\n')
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    # generate story
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=output_length,
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences
    )

    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # Remove all text after eos token
        text = text[: text.find(tokenizer.eos_token)]
        # print(text)
        return text

st.header('Story Generator')

# show user to write prompt 
prompt = st.text_area("Your Prompt ", " write a prompt to generate a story based on it")

k = st.slider('k', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
p = st.slider('p', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
output_length = st.slider('story length', min_value=300, max_value=1000, value=300, step=10)


# button to generate story
if st.button("generate story"):
    generated_text = generate_text(prompt=prompt, k=k, p=p, output_length=output_length)
    st.subheader("your prompt:")
    st.write(prompt)
    st.subheader("generated story:")
    st.write(generated_text)
