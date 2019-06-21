import math
import random

import os
import sys

import json

import datetime
import json

import modelling
import optimization
import tokenization
import tensorflow as tf

flags = tf.flags



def create_model():

def train():



def main(_):
    train()

    estimator = tf.contrib.tpi.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size
    )

    tokenization.validate_case_matches_checkpoint()

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict



if __name__ = "__main__":
