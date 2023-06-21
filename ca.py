import streamlit as st
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer


@st.cache_resource
def generate(model, tokenizer, instruction, max_new_tokens=128, temperature=0.1, top_p=0.75, top_k=40,
             num_beams=4,**kwargs):
    prompt = instruction + "\n### Solution:\n"
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, **kwargs,)
    with torch.no_grad():
        generation_output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                           generation_config=generation_config, return_dict_in_generate=True,
                                           output_scores=True, max_new_tokens=max_new_tokens, early_stopping=True)
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Solution:")[1].lstrip("\n")


#logging.disable(logging.WARNING)
model_id = "mrm8488/falcoder-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda")
torch.cuda.empty_cache()
print("\n\n")
st.title("Falcoder code assist test")
st.divider()
instruction = st.text_input("Code-related Prompt:")
if instruction:
    with st.spinner("Thinking..."):
        answer = generate(model, tokenizer, instruction)
    st.text_area("Answer:", answer, height=250)
torch.cuda.empty_cache()
