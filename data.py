from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os

import tarfile
import gzip
import csv, json, requests, sys
import nltk

import tokenizer
import tensorflow as tf

from nltk.corpus import state_union
from bs4 import BeautifulSoup

flags = tf.flags

FLAGS = tf.FLAGS

flags.DEFINE_string("input_file", None, "Input Raw Text")

flags.DEFINE_string("output_file", None, "Output Example File")

flags.DEFINE_string("vocab_file", None, "Vocab File")

regex_tokenizer = nltk.RegexpTokenizer("\w+")



def create_token(paragraph):
    # paragraph = str(paragraph).lower()
    paragraph = str(paragraph)
    paragraph = paragraph.encode("utf-8", "ignore").decode()

    paragraph = paragraph[4:]
    paragraph = paragraph[:-5]
    paragraph = nltk.sent_tokenize(paragraph)

    token_result = []
    for sentence in paragraph:
        r = word_tokenize(sentence)
        token_result.append(r)
    # return paragraph
    return token_result

def word_tokenize(paragraphs):
    sentences = nltk.sent_tokenize(paragraphs)

    result = []
    for sentence in sentences:
        tokenized_text = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized_text)
        result.append(tokenized_text)
    return result

def main():
    with open("data/release3.3/data/nucle3.2.sgml") as fp:
        soup = BeautifulSoup(fp, "lxml")

    result = []

    article_text = ''
    article = soup.findAll('p')
    for elements in article:
        tokenize = create_token(elements.findAll(text = True))
        result.append(tokenize)

    # print(json.dumps(result, indent=4))
    with open("result.json", "w") as outfile:
        json.dump(result, outfile)

def test():
    result = word_tokenize("Ask a new question about the step that's giving you encoding problems, and you'll learn how to specify the file encoding when processing unicode.")
    with open("test.json", "w") as outfile:
        json.dump(result, outfile)

def process_m2():


main()
