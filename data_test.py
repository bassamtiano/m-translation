from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os

import tarfile
import gzip
import csv, json, requests, sys
import nltk

# import tokenizer
import tensorflow as tf
# from absl import flags

from nltk.corpus import state_union
from bs4 import BeautifulSoup

from vocab import Vocab

# flags = tf.flags
#
# FLAGS = tf.FLAGS
#
# flags.DEFINE_string("input_file", None, "Input Raw Text")
#
# flags.DEFINE_string("output_file", None, "Output Example File")
#
# flags.DEFINE_string("vocab_file", None, "Vocab File")

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

def create_token_txt(paragraph):
    paragraph = str(paragraph)
    paragraph = paragraph.encode("utf-8", "ignore").decode()

    paragraph = word_tokenize_txt(paragraph)
    result = []
    for up_one_level in paragraph:
        result.append(up_one_level)
    return up_one_level

def word_tokenize_txt(paragraph):
    sentences = nltk.sent_tokenize(paragraph)

    result = []
    for sentence in sentences:
        tokenized_text = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized_text)
        result.append(tokenized_text)
    return result

def word_tokenize(paragraphs):
    sentences = nltk.sent_tokenize(paragraphs)

    result = []
    for sentence in sentences:
        tokenized_text = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized_text)
        result.append(tokenized_text)
    return result

def to_json(data_m2):
    result = []
    for sentence in data_m2:
        if sentence[0][0] == "S":
            sentence = sentence[2:]
            sentence = sentence[:-1]
            result.append(sentence)

    with open("resultm2.json", "w") as outfile:
        json.dump(result, outfile)

def to_txt(data_m2):
    f = open("resultm2.txt", "w+")
    for sentence in data_m2:
        if sentence[0][0] == "S":
            sentence = sentence[2:]
            f.write(sentence)

def read_m2():
    with open("data/release3.3/bea2019/nucle.train.gold.bea19.m2") as fp:
        data_m2 = fp.readlines()
    to_txt(data_m2)

def read_data_test():
    with open("fixed.txt") as fp:
        data = fp.readlines()
    result = []
    #

    for elements in data:
        tokenize = create_token_txt(elements)
        result.append(tokenize)

    # result = create_token_txt(data)
    with open("fixed.json", "w") as outfile:
        json.dump(result, outfile)

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


def corpus_test():
    voc = Vocab()
    voc.count_file(os.path.join("resultm2.txt"))
    voc.build_vocab()



read_data_test()

# if __name__ == "__main__":
#     FLAGS = flags.FLAGS
#     flgas.DEFINE_string("data_dir" , None, )
#     tf.app.run(main)
