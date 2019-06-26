import os
# import modeling
# import optimization

import collections
import re
import unicodedata
import six

import tensorflow as tf

"""
    Description the tokenizer

"""

# Convert Sentence to Standard Unicode
def converts_to_unicode(text):
    if six.PY3:
        # Validate if the text is string
        if isinstance(text, str):
            return text
        # Validate if the text is bytes
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python 2 or Python 3 ?")

def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

    
def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens

class FullTokenizer(object):

    def __init__(self, vocab_file, do_lower_case = true):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenizer(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for  sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_tokens_to_tokens(self, ids):
        return convert_by_vocab(self.inf_vocab, tokens)

class BasicTokenizer(object):

    def __init__(self, do_lower_case = True):
        self.do_lower_case = do_lower_case

    def tokenizer(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []

        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower() # Change Uppercase to Lowercase
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_strip_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        char = list(text)
        i = 0

        start_new_word = True


    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

class WordpieceTokenizer(object):

    def __init__(self, vocab, unk_token = "[UNK]",
                max_input_char_per_word = 200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_char_per_word = max_input_char_per_word

        def tokenize(self, text):
            text = convert_to_unicode(text)

            output_tokens = []

            for token in whitespace_tokenize(text):
                chars = list(token)
                if len(chars) > self.max_input_char_per_word:
                    output_tokens.append(self.unk_token)
                    continue

                is_bad = False
                start = 0
                sub_tokens = []

                while start < len(chars):
                    end = len(chars)
                    cur_substr = None

                    while start < end:
                        substr = "".join(chars[start:end])
                        if start > 0:
                            substr = "##" + substr
                        if substr in self.vocab:
                            cur_substr = substr
                            break
                        end -=1

                    if cur_substr is None:
                        is_bad = True
                        break

                    sub_tokens.append(cur_substr)
                    start = end

                if is_bad:
                    output_tokens.append(self.unk_token)
                else:
                    output_tokens.append(sub_token)

            return output_tokens

def _is_whitespace(char):

    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True

    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):

    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False

def _is_punctuation(char):
    cp = ord(char)

    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <=64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True

    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True

    return False
