import re
import os

import tarfile
import gzip

import tensorflow.python.platform import gfile

def download(directory, filename, url):
    

def get_train_set(dir):
    train_path = os.path.join(dir, "train")
    if not ( gfile.Exists() and gfile.Exists() ):
        corpus_file =
    return train_path

def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

def basic_tokenizer(sentence):
    words = []

def char_tokenizer():

def bpe_tokenizer():

def create_vocab():


def prepare_data(dir, max_vocab_size, tokenizer):
    train_path = get_train_set(dir)
