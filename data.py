from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os

import tarfile
import gzip
import csv, json, requests, sys
import nltk

from nltk.corpus import state_union
from bs4 import BeautifulSoup
regex_tokenizer = nltk.RegexpTokenizer("\w+")


def create_token(paragraph):
    paragraph = str(paragraph).lower()
    paragraph = paragraph.rstrip("\\n")
    paragraph = paragraph.encode("utf-8", "ignore").decode()



    # remove punktuation symbols
    paragraph = " ".join(regex_tokenizer.tokenize(paragraph))
    paragraph = paragraph[1:]
    paragraph = paragraph[:-2]
    return paragraph

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



main()
