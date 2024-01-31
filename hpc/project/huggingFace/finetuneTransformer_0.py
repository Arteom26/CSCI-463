from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import numpy as np
import evaluate
import sys
import time


'''prefix = "summarize: "
rouge = evaluate.load("rouge")

# Instantiate pipeline, tokenizer, model, and data-collator
model_name = "facebook/bart-large-cnn"
summarizer = pipeline(task="summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)'''
'''
def tokenize_function(examples, prefix, tokenizer):
    #\print(examples["text"][0])
    #for split in examples:
    #    for doc in examples[split]["article"]:
    #        inputs = prefix + doc
    #        print(f"\n\nTest: \n{doc}")
    #        sys.exit()

    #inputs = [prefix + doc for doc in examples["train"]["article"]] # after evaluating what tokenized_dataset returns in print statement, add nested comprehension with split as an agrument for split in 
    inputs = [prefix + article for split in examples for article in examples[split]["article"]]
    #print(f"\nLast sample in inputs: \n{inputs[-1]}")
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    #gt_labels = tokenizer(text_target=examples["train"]["highlights"], max_length=128, truncation=True) # add ["train"] here to see what tokenized_dataset returns in print statement

    #for split in examples:
    #    targets = examples[split]["highlights"]
    #input_labels = [summary for split in examples for summary in examples[split]["highlights"]]
    input_labels = [examples[split]["highlights"] for split in examples]
    #print(f"\nLast sample in summary: \n{input_labels[-1]}\n")
    gt_labels = tokenizer(text_target=input_labels, max_length=128, truncation=True)
    model_inputs["labels"] = gt_labels["input_ids"]
    return model_inputs'''
'''
def tokenize_function(examples):
    #print(f"Examples object: {type(examples)}")
    #sys.exit()
    inputs = [prefix + doc for doc in examples["text"]]
    print(f"\nLast sample in inputs: \n{inputs[-1]}")
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    print(f"\nLast sample in summary: \n{examples['summary'][-1]}\n")
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs'''

'''def tokenize_function(examples):
    #print(f"Examples object: {type(examples)}")
    #sys.exit()
    inputs = [prefix + doc for doc in examples["article"]]
    print(f"\nLast sample in inputs: \n{inputs[-1]}")
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    print(f"\nLast sample in summary: \n{examples['highlights'][-1]}\n")
    labels = tokenizer(text_target=examples["highlights"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs'''

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def main():
    # Define summarization prefix and metric for evaluating model performance during training
    rouge = evaluate.load("rouge")
    prefix = "summarize: "

    # Instantiate pipeline, tokenizer, model, and data-collator
    model_name = "facebook/bart-large-cnn"
    summarizer = pipeline(task="summarization", model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name) # It is more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length
    
    # Define tokenizer pre-processing function
    def tokenize_function(examples):
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=examples["highlights"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    # Load dataset and tokenize it
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    #billsum = load_dataset("billsum", split="ca_test")
    #billsum = billsum.train_test_split(test_size=0.2)
    #print(f"\n\nBillsum sample: {billsum['train'][0]}")
    #print(f"\n\nSample: {dataset['train'][0]}\n\n")
    #tokenized_dataset = billsum.map(tokenize_function, batched=True)
    start_time = time.time()
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    #tokenized_dataset = dataset.map(tokenize_function(dataset, prefix, tokenizer), batched=True)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    
    print(tokenized_dataset)
    sys.exit()

    # Define hyperparameters
    training_args = Seq2SeqTrainingArguments(
    output_dir="outputDirectory",
    evaluation_strategy="epoch",
    #learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    #weight_decay=0.01,
    #save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    )

    # Instantiate trainer object
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train model for finetuning
    trainer.train()




if __name__ == "__main__":
    main()