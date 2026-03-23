from transformers import MarianMTModel, MarianTokenizer
#from huggingface_hub import login

#login(token="your_token_here")

model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"

model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

sample_text = "I'm thinking about creating a new world, What you think?"
batch = tokenizer([sample_text], return_tensors="pt")

generated_ids = model.generate(**batch)
translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(translation)