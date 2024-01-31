from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import numpy as np
import evaluate
import sys
import time
import argparse

def get_argparser():
    # Instantiate parser
    parser = argparse.ArgumentParser()

    # Command line Options
    parser.add_argument("--outputDir", type=str, help="output path for trained model", 
                        required=True)

    parser.add_argument("--modelAndTokenizerName", type=str, 
                        default='facebook/bart-large-cnn')
                        
    parser.add_argument("--task", type=str, default='summarization')

    return parser


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
    # Instantiate Parser object
    args = get_argparser().parse_args()

    # Define summarization prefix and metric for evaluating model performance during training
    rouge = evaluate.load("rouge")
    prefix = "summarize: "

    # Instantiate pipeline, tokenizer, model, and data-collator
    model_name = args.modelAndTokenizerName
    summarizer = pipeline(task=args.task, model=model_name)
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
    start_time = time.time()
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    
    print(tokenized_dataset)

    # Define hyperparameters
    training_args = Seq2SeqTrainingArguments(
    output_dir=args.outputDir,
    evaluation_strategy="epoch",
    #learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
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