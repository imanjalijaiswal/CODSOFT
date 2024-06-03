import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
image_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Load pre-trained word embeddings
embedding_dim = 100
word_embeddings = {}

# Define image captioning model
def create_model(vocab_size, max_caption_length):
    # Image input
    image_input = Input(shape=(4096,))
    # Image feature extractor
    image_features = Dense(256, activation='relu')(image_input)
    
    # Caption input
    caption_input = Input(shape=(max_caption_length,))
    # Word embedding layer
    word_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(caption_input)
    # LSTM for sequence processing
    caption_lstm = LSTM(256)(word_embedding)
    
    # Combine image and caption features
    combined_features = tf.keras.layers.concatenate([image_features, caption_lstm])
    # Output layer
    output = Dense(vocab_size, activation='softmax')(combined_features)
    
    # Define the model
    model = Model(inputs=[image_input, caption_input], outputs=output)
    return model

# Generate image features
def extract_image_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return image_model.predict(x).reshape(-1,)

# Generate captions
def generate_caption(image_path, model, tokenizer, max_caption_length):
    # Extract image features
    img_features = extract_image_features(image_path)
    # Start captioning process
    input_text = ['<start>']
    for i in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        predicted_word_index = np.argmax(model.predict([img_features.reshape(1, -1), sequence]))
        predicted_word = tokenizer.index_word.get(predicted_word_index, "<end>")
        if predicted_word == "<end>":
            break
        input_text.append(predicted_word)
    return ' '.join(input_text[1:])

# Example usage
image_path = 'example_image.jpg'
max_caption_length = 20
vocab_size = 10000  # Adjust according to your dataset
model = create_model(vocab_size, max_caption_length)
tokenizer = None  # Initialize tokenizer with your dataset
caption = generate_caption(image_path, model, tokenizer, max_caption_length)
print("Generated Caption:", caption)
