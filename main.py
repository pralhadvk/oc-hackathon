from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AlbertConfig, AlbertModel
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load T5 model and tokenizer
# model_name = "t5-small"  # Choose an appropriate T5 model size
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load ALBERT model and tokenizer
# model_name = "albert-base-v2"  # Choose an appropriate ALBERT model size
# tokenizer = AlbertTokenizer.from_pretrained(model_name)
# model = AlbertForMaskedLM.from_pretrained(model_name)

model2 = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer2 = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# model1 = "gpt2"  
tokenizer1 = GPT2Tokenizer.from_pretrained("gpt2")
model1 = GPT2LMHeadModel.from_pretrained("gpt2")


app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

def generate_text1(prompt, max_length=50):
    inputs = tokenizer1(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model1.generate(input_ids=inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    
    generated_text = tokenizer1.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def generate_text2(prompt, max_length=50):
    inputs = tokenizer2(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model2.generate(input_ids=inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    
    generated_text = tokenizer2.decode(outputs[0], skip_special_tokens=True)
    return generated_text

@app.post("/generate/1")
async def generate_text_endpoint(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        response = generate_text1(prompt, max_length=50)  

        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Text generation failed.")
    
@app.post("/generate/2")
async def generate_text_endpoint(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        response = generate_text2(prompt, max_length=50)  

        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Text generation failed.")