from datasets import load_dataset_builder
import time
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import list_metrics
from datasets import load_metric

ds_builder = load_dataset_builder("cnn_dailymail", "3.0.0")
print(f"Dataset features: {ds_builder.info.features}\n")
ds_builder = load_dataset_builder("billsum")
print(f"Dataset features: {ds_builder.info.features}\n")
billsum = load_dataset("billsum", split="ca_test")
print(billsum)
billsum = billsum.train_test_split(test_size=0.2)
print(f"\nAfter split: {billsum}")
sys.exit()

dataset = load_dataset("cnn_dailymail", "3.0.0")
dataset_train = dataset["train"]
dataset_val = dataset["validation"]
dataset_test = dataset["test"]

start_time = time.time()
article = dataset_train[0]['article']
end_time = time.time()
print(f"Elapsed time rowwise: {end_time - start_time:.4f} seconds")

start_time = time.time()
article = dataset_train['article'][0]
end_time = time.time()
print(f"Elapsed time columnwise: {end_time - start_time:.4f} seconds\n")

# Tokenization Process for a BERT Model
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenization(example):
    return tokenizer(example["article"])

#dataset_train_tokenized = dataset.map(tokenization, batched=True)

# Set the format of the dataset to be compatible with PyTorch
dataset_train.set_format(type="torch", columns=["article", "highlights", "id"])
print(f"Dataset format: {dataset_train.format['type']}")


# Print available metrics from list_metrics
metrics_list = list_metrics()
print(f"Length of metrics list: {len(metrics_list)}")
print(f"\nMetrics list: {metrics_list}\n\n")

# Load the accuracy metric associated with the cnn_dailymail dataset
metric = load_metric('accuracy', 'cnn_dailymail', trust_remote_code=True)
print(f"\nMetric Description: {metric.inputs_description}\n\n")
