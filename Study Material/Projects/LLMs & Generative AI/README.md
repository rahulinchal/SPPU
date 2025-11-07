# IMDb Sentiment Analysis with DistilBERT

A sentiment analysis project using the Hugging Face Transformers library to fine-tune a DistilBERT model on the IMDb movie reviews dataset for binary sentiment classification (Positive/Negative).

## Project Overview

This project demonstrates how to fine-tune a pre-trained DistilBERT model for sentiment analysis on movie reviews. The model learns to classify movie reviews as either positive or negative sentiment.

## Dataset

- **Name**: IMDb Movie Reviews Dataset
- **Source**: Hugging Face Datasets Hub
- **Size**: 50,000 reviews
  - Training set: 25,000 reviews
  - Test set: 25,000 reviews
- **Classes**: Binary classification
  - 0: NEGATIVE
  - 1: POSITIVE
- **Features**: Text (movie review), Label (sentiment)

## Requirements

```python
transformers
datasets
evaluate
torch
numpy
```

Install dependencies:
```bash
pip install transformers datasets evaluate torch numpy
```

## Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Sequence Classification
- **Number of Labels**: 2 (Binary classification)
- **Tokenizer**: DistilBERT tokenizer with automatic padding

### Why DistilBERT?

DistilBERT is a distilled version of BERT that:
- Retains 97% of BERT's language understanding
- Is 40% smaller in size
- Runs 60% faster
- Perfect for resource-constrained environments

## Project Workflow

### 1. Data Loading

```python
from datasets import load_dataset
imdb = load_dataset("imdb")
```

The dataset is automatically downloaded and structured with train/test splits.

### 2. Preprocessing

#### Tokenization
- **Tokenizer**: `AutoTokenizer` from `distilbert-base-uncased`
- **Truncation**: Enabled to handle long reviews
- **Padding**: Dynamic padding using `DataCollatorWithPadding`

```python
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
```

#### Data Collation
- Uses `DataCollatorWithPadding` for efficient batch processing
- Dynamically pads sequences to the longest sequence in each batch
- Reduces computational overhead compared to fixed-length padding

### 3. Evaluation Metrics

- **Primary Metric**: Accuracy
- **Library**: Hugging Face Evaluate
- **Calculation**: Predictions are compared with ground truth labels

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
```

### 4. Model Configuration

#### Label Mapping
```python
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
```

#### Model Initialization
- Loaded using `AutoModelForSequenceClassification`
- Pre-trained weights from DistilBERT
- Classification head added for 2-class prediction

### 5. Training Configuration

#### Hyperparameters
- **Learning Rate**: 2e-5
- **Batch Size**: 16 (both train and eval)
- **Epochs**: 2
- **Weight Decay**: 0.01
- **Evaluation Strategy**: End of each epoch
- **Save Strategy**: End of each epoch
- **Best Model**: Automatically loaded at the end

#### Training Arguments
```python
TrainingArguments(
    output_dir="my_great_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

### 6. Training Process

The model is trained using the Hugging Face `Trainer` API:
- Automatic gradient computation and optimization
- Built-in evaluation during training
- Model checkpointing at each epoch
- Progress tracking and logging

### 7. Inference

#### Making Predictions

1. **Prepare Input Text**
   ```python
   text = "This was a masterpiece. Not completely faithful to the books, 
           but enthralling from beginning to end. Might be my favorite of the three."
   ```

2. **Tokenize**
   ```python
   inputs = tokenizer(text, return_tensors="pt")
   ```

3. **Load Model and Predict**
   ```python
   model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
   with torch.no_grad():
       logits = model(**inputs).logits
   predicted_class_id = logits.argmax().item()
   ```

4. **Get Label**
   ```python
   sentiment = model.config.id2label[predicted_class_id]
   ```

## Project Structure

```
project/
├── my_great_model/           # Model checkpoints
│   ├── checkpoint-1564/      # Best model checkpoint
│   ├── checkpoint-XXXX/      # Other checkpoints
│   └── ...
├── notebook.ipynb            # Main training notebook
└── README.md                 # This file
```

## Usage

### Training

```python
# Load dataset
imdb = load_dataset("imdb")

# Preprocess
tokenized_imdb = imdb.map(preprocess_function, batched=True)

# Train
trainer.train()
```

### Inference

```python
# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")

# Predict
with torch.no_grad():
    logits = model(**inputs).logits
    
# Get result
predicted_class_id = logits.argmax().item()
sentiment = model.config.id2label[predicted_class_id]
```

## Key Features

✅ **Transfer Learning**: Leverages pre-trained DistilBERT knowledge  
✅ **Dynamic Padding**: Efficient batch processing  
✅ **Automatic Evaluation**: Built-in accuracy metrics  
✅ **Best Model Selection**: Automatically saves the best performing checkpoint  
✅ **Easy Inference**: Simple API for making predictions  

## Performance Considerations

- **Training Time**: ~30-45 minutes on GPU for 2 epochs
- **Model Size**: ~260MB (DistilBERT base)
- **Inference Speed**: Fast inference suitable for production
- **Memory Usage**: Optimized with dynamic padding

## Expected Results

- **Baseline Accuracy**: ~50% (random guess)
- **Expected Accuracy**: ~90-93% after 2 epochs
- The model should effectively distinguish between positive and negative sentiments in movie reviews

## Customization Options

### Adjust Hyperparameters
- Increase `num_train_epochs` for potentially better performance
- Modify `learning_rate` for different convergence behavior
- Change `batch_size` based on available GPU memory

### Try Different Models
Replace `distilbert-base-uncased` with:
- `bert-base-uncased` (larger, more accurate)
- `roberta-base` (alternative architecture)
- `albert-base-v2` (parameter-efficient)

### Add More Metrics
```python
# Add precision, recall, F1-score
from evaluate import load
precision = load("precision")
recall = load("recall")
f1 = load("f1")
```

## Troubleshooting

### Out of Memory Error
- Reduce `per_device_train_batch_size`
- Enable gradient accumulation
- Use a smaller model variant

### Slow Training
- Ensure GPU is available and being used
- Increase batch size if memory permits
- Reduce max sequence length in tokenization

### Poor Performance
- Train for more epochs
- Adjust learning rate
- Check data quality and balance

## Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Text Classification Task Guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [IMDb Dataset](https://huggingface.co/datasets/imdb)
- [Hugging Face Evaluate Library](https://huggingface.co/docs/evaluate)

## Example Use Cases

- **Movie Review Analysis**: Automatically classify user reviews
- **Product Feedback**: Analyze customer sentiment
- **Social Media Monitoring**: Track brand sentiment
- **Content Moderation**: Flag negative content
- **Customer Service**: Route complaints vs compliments

## Future Improvements

- [ ] Implement cross-validation for robust evaluation
- [ ] Add confusion matrix visualization
- [ ] Include precision, recall, and F1-score metrics
- [ ] Experiment with different model architectures
- [ ] Add data augmentation techniques
- [ ] Implement early stopping
- [ ] Create a web demo using Gradio or Streamlit
- [ ] Fine-tune on domain-specific data
- [ ] Add multi-class sentiment (Very Negative, Negative, Neutral, Positive, Very Positive)

## License

This project is for educational purposes.

## Acknowledgments

- Hugging Face for the Transformers library and datasets
- Stanford for the IMDb dataset
- Google for the DistilBERT model