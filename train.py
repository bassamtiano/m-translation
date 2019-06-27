import math
import random

import os
import sys

import json

import datetime

# import modeling
import tokenizer
import nltk
import random
import logging

# import optimization
# import tokenization
import tensorflow as tf


def extract_file():

def train():
    # Preparing The Data setting



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)


    train()

    n_token = corpus_info["vocab_size"]
    cutoffs = corpus_info["cuttofs"][1:-1]

    if FLAGS.do_train:

    if FLAGS.do_eval:

    # estimator = tf.contrib.tpi.TPUEstimator(
    #     use_tpu=FLAGS.use_tpu,
    #     model_fn=model_fn
    #     config=run_config,
    #     train_batch_size=FLAGS.train_batch_size,
    #     eval_batch_size=FLAGS.eval_batch_size
    # )
    #
    # tokenization.validate_case_matches_checkpoint()
    #
    # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict



if __name__ == "__main__":
    tf.app.run()
