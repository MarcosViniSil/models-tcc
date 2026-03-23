from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

en_text = "hello, how are you?"

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# translate Hindi to French
tokenizer.src_lang = "en"
encoded_hi = tokenizer(en_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("pt"))
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))


