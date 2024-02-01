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

import nltk
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator

from tqdm.auto import tqdm
import torch



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

# Function that extracts the first three lines in an article
def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])

# Function that extracts the three sentence summary from "article" and compares it with reference "highlights"
def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["article"]]
    return metric.compute(predictions=summaries, references=dataset["highlights"])

# Function that splits the generated summaries (the predictions) into sentences that are separated by newlines
# This is the format that the ROUGE metric expects
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def main():
    # Instantiate Parser object
    args = get_argparser().parse_args()

    # Define summarization prefix and metric for evaluating model performance during training
    rouge = evaluate.load("rouge")
    prefix = "summarize: "

    # Specify output directory for model and download nltk punctuation rules
    nltk.download("punkt")
    output_dir = args.outputDir

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


    # Back to PyTorch :D
    # Create a DataLoader for each of our splits. PyTorch DataLoader expects batches of tensors, so we must set the format to "torch" in our datasets (N, C, H, W)
    tokenized_dataset.set_format("torch")
    batch_size = 8

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    val_dataloader = DataLoader(
        tokenized_dataset["validation"],
        collate_fn=data_collator,
        batch_size=batch_size
    )

    # Define back propagation optimizer and the learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_train_epochs = 10
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Instantiate accelerator, then feed model, optimizer, and dataloaders to the accelerator.prepare() method
    accelerator = Accelerator()
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # Define training loop
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]

                # If we did pad the predictions, we need to pad the labels too for uniform comparison
                labels = accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                )

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )

                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        # Compute metrics
        result = rouge_score.compute()
        # Extract the median ROUGE scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        print(f"Epoch {epoch}:", result)

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)




if __name__ == "__main__":
    main()