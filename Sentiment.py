# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# Load the data
df = pd.read_csv('data.csv')
X = df.drop('Sentiment', axis=1)  
y = df['Sentiment']

# Split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LABELS_MAPPING = {
    'neutral': 0, 'positive': 1, 'negative': 2
}
# Encoding the labels as numbers from 0-2
y_train = y_train.apply(lambda x: LABELS_MAPPING[x])
y_test = y_test.apply(lambda x: LABELS_MAPPING[x])

# Evaluating Before Training: Baseline Model 
print("Evaluating baseline model before training...")

# Check class distribution in the training set
class_distribution = y_train.value_counts()
print("Class distribution in training set:")
print(class_distribution)

# Baseline prediction: predict the majority class for all instances
majority_class = y_train.mode()[0]  # Most frequent class in training set
y_pred_baseline = [majority_class] * len(y_test)  # Predict majority class for all test samples

# Calculate accuracy of the baseline model
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print(f'Baseline Accuracy: {baseline_accuracy:.4f}')

# Generate a classification report for the baseline model
print("Baseline Classification Report:")
print(classification_report(y_test, y_pred_baseline, target_names=LABELS_MAPPING.keys()))

# Initializing the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenizing the sentences
train_encodings = tokenizer(list(X_train['Sentence']), padding=True, truncation=True)
test_encodings = tokenizer(list(X_test['Sentence']), padding=True, truncation=True)

# Combining the sentences and sentiment data 
train_enc_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': y_train,
})

test_enc_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': y_test,
})

# Initializing the GPT-2 classification model
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels = 3)
model.config.pad_token_id = model.config.eos_token_id


# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate = 2e-5,
    per_device_train_batch_size = 32,
    num_train_epochs = 3,
    weight_decay = 0.01,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_enc_dataset,
    eval_dataset = test_enc_dataset,
)

trainer.train()

# Evaluating After Training
print("\nEvaluating model after training...")

# Get predictions from the trained model
predictions = trainer.predict(test_enc_dataset)

# Convert logits to predicted class labels
y_pred = np.argmax(predictions.predictions, axis=1)

# Calculate accuracy after training
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy after Training: {accuracy:.4f}')

# Generate a classification report after training
print("Classification Report after Training:")
print(classification_report(y_test, y_pred, target_names=LABELS_MAPPING.keys()))

# Confusion Matrix Visualization
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_MAPPING.keys(), yticklabels=LABELS_MAPPING.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix after Training')
plt.show()

model_save_path = "./best_model" 
model.save_pretrained(model_save_path)