from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os

import tarfile
import gzip
import csv, json, requests, sys
import nltk

from bs4 import BeautifulSoup
regex_tokenizer = nltk.RegexpTokenizer("\w+")


def create_token(text):
    text = str(text).lower()
    # text = text.rstrip("\\n")
    text = text.encode("utf-8", "ignore").decode()
    # remove punktuation symbols
    text = " ".join(regex_tokenizer.tokenize(text))
    text = text[1:]
    text = text[:-2]
    return text

def main():
    with open("data/release3.3/data/nucle3.2.sgml") as fp:
        soup = BeautifulSoup(fp, "lxml")

    #
    # print(soup.prettify())
    # info = soup.find_all('title')

    result = []
    # for datasets in soup:
    #     # result.append(soup.find('title'));
    #     print(soup.find_all('title'))

    # title = [ data.title for data in soup.title ]
    # print(json.dumps(title, indent=2))
    # print(title)
    # print(rs)

    article_text = ''
    article = soup.findAll('p')
    for elements in article:
        # article_text += '\n' + ''.join(elements.findAll(text = True))
        tokenize = create_token(elements.findAll(text = True))
        # result.append(elements.findAll(text = True))
        result.append(tokenize)

    # print(json.dumps(result, indent=4))
    with open("result.json", "w") as outfile:
        json.dump(result, outfile)



main()
