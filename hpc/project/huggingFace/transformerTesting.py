from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

# Instantiate pipeline, tokenizer, and model
model_name = "facebook/bart-large-cnn"
summarizer = pipeline(task="summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Test model the easy way
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
output = summarizer(text, max_length=40, min_length=20)
print(f"\nText corpus: \n{text}")
print(f"\nDefault Summarization: \n{output}\n")

# Test model slightly more in-depth
batch = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").input_ids
output_tokenIDs = model.generate(batch, max_length=40, min_length=20, do_sample=False)
output_decoded = tokenizer.decode(output_tokenIDs[0], skip_special_tokens=True)
print(f"\nCustom Summarization: \n{output_decoded}\n")