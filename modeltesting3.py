import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import numpy as np

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model
custom_objects = {"TFBertModel": TFBertModel}
model_path = "outputs/bert_model.h5"
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
print(f"Model loaded successfully from {model_path}")

# Load dataset
df = pd.read_csv("Nazario.csv")
texts = df["body"].astype(str).tolist()

# Tokenize
inputs = tokenizer(
    texts,
    max_length=128,
    truncation=True,
    padding="max_length",
    return_tensors="tf"
)

# Convert BatchEncoding to dict of tensors
input_dict = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "token_type_ids": inputs["token_type_ids"]
}

# Run predictions
predictions = model.predict(input_dict, batch_size=32)

# Apply sigmoid if needed (for binary classification)
urgency_scores = tf.nn.sigmoid(predictions).numpy().flatten()

# Save results
df["urgency_score"] = urgency_scores
df.to_csv("nazarioresults.csv", index=False)
print("Results saved to urgency_scores.csv")
