import os
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim import downloader as api

# Function to load, preprocess text data, and delete unmatched images
def load_and_process_data(filename, image_folder):
    captions = {}
    longest_caption_length = 0
    total_images_deleted = 0

    # Load captions into a dictionary
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[1]:
                image_name, captions_str = parts
                captions[image_name] = ' '.join(captions_str.split('|'))
                longest_caption_length = max(longest_caption_length, len(captions[image_name].split()))

    # Delete images that don't have a valid caption
    for image_file in os.listdir(image_folder):
        if image_file not in captions:
            os.remove(os.path.join(image_folder, image_file))
            total_images_deleted += 1

    print(f"Total valid image-caption pairs: {len(captions)}")
    print(f"Total images deleted: {total_images_deleted}")
    print(f"Longest caption length: {longest_caption_length}")

    return captions, longest_caption_length

# Tokenize captions and create vocabulary
def tokenize_and_pad_captions(data, max_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data.values())
    sequences = tokenizer.texts_to_sequences(data.values())
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return tokenizer, padded_sequences

# Main script
if __name__ == "__main__":
    # Load pre-trained GloVe model
    glove_model = api.load('glove-twitter-100')

    image_folder = './celeba'  
    filename = 'caps.txt'  

    # Load and process text data, and delete unmatched images
    text_data, max_caption_length = load_and_process_data(filename, image_folder)

    print(f"Total images remaining: {len(os.listdir(image_folder))}")

    # Tokenize and pad captions
    tokenizer, padded_sequences = tokenize_and_pad_captions(text_data, max_caption_length)

    # Create embedding matrix
    embedding_dim = 100
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in glove_model.key_to_index:
            embedding_matrix[i] = glove_model[word]

    # Save tokenizer and embedding matrix
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)
    with open('embedding_matrix.pkl', 'wb') as file:
        pickle.dump(embedding_matrix, file)

    # Save preprocessed captions and their corresponding image filenames
    preprocessed_data = {img: padded_sequences[idx] for idx, img in enumerate(text_data.keys())}
    with open('preprocessed_data.pkl', 'wb') as file:
        pickle.dump(preprocessed_data, file)
