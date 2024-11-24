import os
import gzip
import json
import pandas as pd
from collections import Counter
import torch


import numpy as np
import re

amazon_product_categories = ['all_beauty',
 'amazon_fashion',
 'appliances',
 'giftcards',
 'magazine_subscriptions',
 ]


lexicon = (
    (re.compile(r"\bdon't\b"), "do not"),
    (re.compile(r"\bit's\b"), "it is"),
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"\bi've\b"), "i have"),
    (re.compile(r"\bcan't\b"), "cannot"),
    (re.compile(r"\bdoesn't\b"), "does not"),
    (re.compile(r"\bthat's\b"), "that is"),
    (re.compile(r"\bdidn't\b"), "did not"),
    (re.compile(r"\bi'd\b"), "i would"),
    (re.compile(r"\byou're\b"), "you are"),
    (re.compile(r"\bisn't\b"), "is not"),
    (re.compile(r"\bi'll\b"), "i will"),
    (re.compile(r"\bthere's\b"), "there is"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bwoudn't\b"), "would not"),
    (re.compile(r"\bhe's\b"), "he is"),
    (re.compile(r"\bthey're\b"), "they are"),
    (re.compile(r"\bwe're\b"), "we are"),
    (re.compile(r"\blet's\b"), "let us"),
    (re.compile(r"\bhaven't\b"), "have not"),
    (re.compile(r"\bwhat's\b"), "what is"),
    (re.compile(r"\baren't\b"), "are not"),
    (re.compile(r"\bwasn't\b"), "was not"),
    (re.compile(r"\bwouldn't\b"), "would not"),
)

def fix_apostrophes(text):
    text = text.lower()
    
    for pattern, replacement in lexicon:
        text = pattern.sub(replacement, text)

    return text

def review_parse(path: str):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

def get_amazon_reviews(data_dir: str, ood_class: int, arch: str): 
    
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    
    amazon_reviews_data = []
    all_labels = []
    for label, product in enumerate(os.listdir(data_dir)):
        for data in review_parse(os.path.join(data_dir,product)):
            amazon_reviews_data.append(data)
            all_labels.append(label)
    
    label_map = np.unique(all_labels)
    
    label_map[[np.where(label_map==ood_class)[0][0],-1]] = label_map[[-1,np.where(label_map==ood_class)[0][0]]]
    
    labels, texts, sentiments = [],[], []
    for idx, (review, label) in enumerate(zip(amazon_reviews_data, all_labels)):
        if review['overall']==3.0 or 'overall' not in review.keys() or 'reviewText' not in review.keys():
            continue
        labels.append(label_map[label])
        texts.append(review['reviewText'])
        sentiments.append(int(review['overall']>3.0))
    
    if arch=='Roberta':
        texts = list(map(fix_apostrophes, texts))
        return np.array(texts), np.array(labels), np.array(sentiments), None

    else:
        MAX_SEQUENCE_LENGTH = 1000
        MAX_NB_WORDS = 20000

        texts = list(map(fix_apostrophes, texts))
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
            lower=False, 
            filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

        tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences(texts)

        word_index = tokenizer.word_index

        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')

        return data, np.array(labels), np.array(sentiments), word_index

def get_amazon_reviews_features(data_dir: str, ood_class: int, arch: str):
    data = torch.load(data_dir, map_location=torch.device('cuda:0'))
    
    all_labels = data['targets'].type(torch.long).cpu().detach().numpy()
    label_map = np.unique(all_labels)
    label_map[[np.where(label_map==ood_class)[0][0],-1]] = label_map[[-1,np.where(label_map==ood_class)[0][0]]]
    labels = []
    
    for idx,label in enumerate(all_labels):
        labels.append(label_map[label])

    return data['features'].cpu().detach().numpy(), np.array(labels), data['sentiments'].cpu().detach().numpy(), None

def glove_embeddings(glove_path, word_index):
    embeddings_index = {}

    with open(glove_path) as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            #values[-1] = values[-1].replace('\n', '')
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            #print (values[1:])
    
    EMBEDDING_DIM = 100

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        #embedding_vector = embeddings_index[word]
        if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix