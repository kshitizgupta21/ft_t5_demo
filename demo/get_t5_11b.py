from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("t5-11b")

model = AutoModelWithLMHead.from_pretrained("t5-11b")

model.save_pretrained("./t5-11b")
