import os

import collections
import re
import unicodedata
import six

import tensorflow as tf

class BasicTokenizer():

    def __init__(do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = self.convert_to_unicode(text)

    def convert_to_unicode():

    def _is_whitespace():
        

    def _clean_text(self, text):
        output = []
