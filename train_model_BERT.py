import sys
from datasets               import load_dataset, load_from_disk
from transformers           import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from postcoordinate_functions import get_jerarquia

## Auxiliary functions - These were taken from a HuggingFace colab ##
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def group_texts(examples):
  chunk_size = 128

  # Concatenate all texts
  concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
  # Compute length of concatenated texts
  total_length = len(concatenated_examples[list(examples.keys())[0]])
  # We drop the last chunk if it's smaller than chunk_size
  total_length = (total_length // chunk_size) * chunk_size
  # Split by chunks of max_len
  result = {
      k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
      for k, t in concatenated_examples.items()
  }
  # Create a new labels column
  result["labels"] = result["input_ids"].copy()

  return result


# Importing hierarchy
PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()

# Loading the corpus for training and testing
CORPUS_BERT_TRAINING = sys.argv[0]
CORPUS_BERT_EVAL = sys.argv[1]

# We load the dataset
print('Loading the dataset')
dataset = load_dataset('text', data_files={'train' : PATH + CORPUS + CORPUS_BERT_TRAINING,
                                           'test' : PATH + CORPUS + CORPUS_BERT_EVAL})

# We obtain the tokenizer
# This is the tokenizer for BioBERT, but it should be changed depending on our base model
print('Loading tokenizer')
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2', do_lower_case=False)

# If we already saved the preprocessed dataset, we could load it
#lm_datasets = load_from_disk(PATH + 'lm_dataset')

# Otherwise, we preprocess our dataset, tokenizing it
# and grouping up sentences
print('Preprocessing dataset')
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# We can save the preprocessed dataset to avoid
# having to preprocess it each time
#lm_datasets.save_to_disk(PATH + 'lm_dataset')

# We load the BERT model - This is BioBERT, but it can be changed depending on what we
# want to do
model = BertForMaskedLM.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
model.train()

# We create the data collator to mask some words so that we can train
# BERT to solve the task of Masked Language Modelling, although
# we don't care for the actual output, since we want to create word embeddings
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Training arguments for BERT
training_args = TrainingArguments(
    output_dir = PATH + "model",
    overwrite_output_dir = True,
    evaluation_strategy = 'steps',
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_steps=10000,
    save_steps=10000,
    save_total_limit = None
)

# Loading the training parameters into the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    data_collator=data_collator
)

# We train the model
print('Training BERT model')
trainer.train()

# After the model is trained, we can save it
print('Saving trained model')
trainer.save_model(PATH + 'model')