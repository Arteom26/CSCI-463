# Build a BPE tokenizer from scratch on the cnn_dailymail dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
import sys


# Instantiate Tokenizer and BpeTrainer objects to allow the Tokenizer to be trained later
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# We need a pre-tokenizer to split our inputs into words. Ensures no token bigger than a word is returned by the pre-tokenizer
tokenizer.pre_tokenizer = Whitespace() # easiest pre-tokenizer possible, splits on whitespace

# Load the CNN Daily Mail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Function to convert articles in CNN dataset into iterables
def texts_generator(dataset):
    for split in dataset:
        print(f"Split: {split}")
        for text in range(len(split)):
            print(text)
            print(dataset[split][text]['article'])
            sys.exit()
            #print("\n\n")
            yield dataset[split][text]['article']

# Train tokenizer on the article iterables from the CNN Daily Mail Dataset
'''
From 'https://github.com/huggingface/tokenizers/blob/67fe59c88d4b8a6c62652766b9f266f1bb3411a2/bindings/python/py_src/tokenizers/__init__.pyi#L77'
class Tokenizer:
    def train(self, files, trainer=None):

        Train the Tokenizer using the given files.

        Reads the files line by line, while keeping all the whitespace, even new lines.
        If you want to train from data store in-memory, you can check
        :meth:`~tokenizers.Tokenizer.train_from_iterator`

        Args:
            files (:obj:`List[str]`):
                A list of path to the files that we should use for training

            trainer (:obj:`~tokenizers.trainers.Trainer`, `optional`):
                An optional trainer that should be used to train our Model
'''
tokenizer.train_from_iterator(texts_generator(dataset), trainer)

# For other ways to train a tokenizer, or to see how to train a tokenizer on a Datasets Library, see 
# https://huggingface.co/docs/tokenizers/training_from_memory

# Save the tokenizer in one file
tokenizer.save("tokenizer.cnnDailyMail.json")

# Preview tokenizer on an example:
sentence = "Hello, y'all! How are you 😁?"
output = tokenizer.encode(sentence) # To generate an emoji, hold "Windows Key + ."
print(f"Output tokens: {output.tokens}")

# Print the index of each of the tokens in the tokenizer's vocabulary:
print(f"Token index from vocabulary: {output.ids}")

# To find what caused the "[UNK]" token to appear (which is the token at index 9 in the above list), we would print the indices that correspond to the emoji in the original sentence
print(output.offsets[9])
print(f"Emoji indexed at {output.offsets[9]} is {sentence[26:27]}")

# Post processing
'''We might want our tokenizer to automatically add special tokens, like "[CLS]" or "[SEP]". To do this, we use a post-processor. 
TemplateProcessing is the most commonly used, you just have to specify a template for the processing of single sentences and pairs of 
sentences, along with the special tokens and their IDs. When we built our tokenizer, we set "[CLS]" and "[SEP]" in positions 1 and 2 
of our list of special tokens, so this should be their IDs. To double-check, we can use the Tokenizer.token_to_id method:'''
tokenizer.token_to_id("[SEP]")

# Here is how we can set the post-processing to give us the traditional BERT inputs:
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)
'''Let’s go over this snippet of code in more details. First we specify the template for single sentences: those should have the form 
"[CLS] $A [SEP]" where $A represents our sentence. Then, we specify the template for sentence pairs, which should have the form
"[CLS] $A [SEP] $B [SEP]" where $A represents the first sentence and $B the second one. The :1 added in the template represent the type 
IDs we want for each part of our input: it defaults to 0 for everything (which is why we don’t have $A:0) and here we set it to 1 for 
the tokens of the second sentence and the last "[SEP]" token. Lastly, we specify the special tokens we used and their IDs in our 
tokenizer’s vocabulary. To check out this worked properly, let’s try to encode the same sentence as before:'''
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(f"Output tokens after post processing: {output.tokens}")
print(f"Type IDs attributed to each token: {output.type_ids}")

# Encoding multiple sentences in a batch
output = tokenizer.encode_batch(["Hello, y'all! How are you 😁 ?"]) # Output is now a list of Encoding objects like the ones seen prior

# To process a batch of sentences pairs, pass two lists to the Tokenizer.encode_batch method: the list of sentences A and the list of sentences B
output = tokenizer.encode_batch(
    [["Hello, y'all!", "How are you 😁 ?"], ["Hello to you too!", "I'm fine, thank you!"]]
)

# Attention mask indicates which tokens should be attended to (non-padded tokens and potentially special character tokens)
# Padding or truncation are useful when you want to return a batch with uniform length!!!
# Also notice the attention mask output prior to any padding:
print(f"Attention mask before padding: {output[1].attention_mask}")

# When encoding multiple sentences, you can automatically pad the outputs to the longest sentence present by using Tokenizer.enable_padding, 
# with the pad_token and its ID (which we can double-check the id for the padding token with Tokenizer.token_to_id like before):
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

# We can set the direction of the padding (defaults to the right) or a given length if we want to pad every sample to that specific number 
# (here we leave it unset to pad to the size of the longest text).
output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
print(f"Output tokens after post processing and padding: {output[1].tokens}")

# In this case, the attention mask generated by the tokenizer takes the padding into account:
print(f"Attention mask after padding: {output[1].attention_mask}")

'''On top of encoding the input texts, a Tokenizer also has an API for decoding, that is converting IDs generated by your model back to a
text. This is done by the methods Tokenizer.decode (for one predicted text) and Tokenizer.decode_batch (for a batch of predictions).

The decoder will first convert the IDs back to tokens (using the tokenizer’s vocabulary) and remove all special tokens, then join those 
tokens with spaces:'''
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.ids) # Should be: [1, 37, 212, 72, 12, 82, 9, 178, 5, 37, 136, 143, 253, 0, 28, 2]

# Decode predictions aka output.ids
tokenizer_preds = tokenizer.decode([1, 37, 212, 72, 12, 82, 9, 178, 5, 37, 136, 143, 253, 0, 28, 2])
print(tokenizer_preds)


