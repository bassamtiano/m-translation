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


class Corpus(object):
    def __init__(self):
        self.vocab = Vocab()

        # Training Datasets
        self.vocab.count_file(os.path.join("resultm2.txt"))

        # Fixed Datasets

        # Test Datasets

        self.vocab.build_vocab()

        self.train = self.vocab.encode_file(os.path.join("resultm2.txt"), ordered = True)

        self.cutoffs = []


def main(unused_argv):
    


Corpus()
