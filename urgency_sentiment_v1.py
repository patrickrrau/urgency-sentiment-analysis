from transformers import TFBertModel
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Load the tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load base BERT model (without classification head)
bert_base = TFBertModel.from_pretrained("bert-base-uncased")

# Define custom regression head
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
token_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="token_type_ids")

bert_output = bert_base(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
output = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)  # Sigmoid for output between 0 and 1

bert_model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)

#################text cleaning
def preprocess(X):
    import re
    def text_clean(text):
        temp = text.lower()
        temp = re.sub("@[A-Za-z0-9_]+","", temp)
        temp = re.sub("#[A-Za-z0-9_]+","", temp)
        temp = re.sub(r"http\S+", "", temp)
        temp = re.sub(r"www.\S+", "", temp)
        temp = re.sub("[0-9]","", temp)
        return temp
    X_cleaned = [text_clean(text) for text in X]
    return X_cleaned
############transforming raw data to an appropriate format ready to feed into the BERT model
def convert_example_to_feature(text):
    return bert_tokenizer.encode_plus(text,
            add_special_tokens = True, # add [CLS], [SEP]
            max_length = 128, # max length of the text that can go to BERT
            pad_to_max_length = True, # add [PAD] tokens
            return_attention_mask = True, # add attention mask to not focus on pad tokens
          )
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
    }, label
def encode_examples(X,y):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    for text, label in zip(X, y):
        bert_input = convert_example_to_feature(text)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

# train dataset
data = pd.read_csv("dataset.csv")
x = data['Text']  # Replace with your text column name
y = data['Score']  # Replace with your label column name

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=0)  # 40% reserved for val + test
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
ds_train_encoded = encode_examples(preprocess(X_train), y_train).shuffle(100).batch(32).repeat(2)
ds_val_encoded = encode_examples(preprocess(X_validation), y_validation).batch(32)
# test dataset
ds_test_encoded = encode_examples(preprocess(X_test), y_test).batch(32)

######### compiling the model
learning_rate = 3e-5
# choosing Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
bert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])

#############training and evaluating
bert_model.fit(ds_train_encoded, epochs=10, validation_data=ds_val_encoded)

loss, acc = bert_model.evaluate(ds_test_encoded, verbose=0)
print("accuracy: {:5.2f}%".format(100 * acc))

bert_model.save("outputs/bert_model.h5")  # Save in Keras HDF5 format

