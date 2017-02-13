import logging
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session
import tensorflow as tf

import data_utils
import seq2seq_model
import os
import numpy as np

## Web Server for connecting with alexa 

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "twitter_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "model", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", True,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size,
      FLAGS.fr_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  return model

# Loads the model
def load_model(session):
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        return model

app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)

sess = tf.Session()
model = load_model(sess)

# Runs the decoder for an utterance to get the response from the model
def get_response_from_model(session,model,sentence):
	cont_vocab_path = os.path.join(FLAGS.data_dir,
		"vocab%d.cont" % FLAGS.en_vocab_size)
	resp_vocab_path = os.path.join(FLAGS.data_dir,
		 "vocab%d.resp" % FLAGS.fr_vocab_size)
	cont_vocab, _ = data_utils.initialize_vocabulary(cont_vocab_path)
	_, rev_resp_vocab = data_utils.initialize_vocabulary(resp_vocab_path)

	# Get token-ids for the input sentence.
	token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), cont_vocab)
	print(token_ids)
	# Which bucket does it belong to?
	bucket_id = len(_buckets) - 1
	for i, bucket in enumerate(_buckets):
		if bucket[0] >= len(token_ids):
			bucket_id = i
			break
		else:
			logging.warning("Sentence truncated: %s", sentence)

	# Get a 1-element batch to feed the sentence to the model.
	encoder_inputs, decoder_inputs, target_weights = model.get_batch(
		{bucket_id: [(token_ids, [])]}, bucket_id)
	# Get output logits for the sentence.
	_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
		target_weights, bucket_id, True)
	# This is a greedy decoder - outputs are just argmaxes of output_logits.
	outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
	# If there is an EOS symbol in outputs, cut them at that point.
	if data_utils.EOS_ID in outputs:
		outputs = outputs[:outputs.index(data_utils.EOS_ID)]

	response = " ".join([tf.compat.as_str(rev_resp_vocab[output]) for output in outputs])
	return response

# Response when alexa skill is launched
@ask.launch
def ask_intent():
        response = 'Welcome to the twitter chat bot' 
        return statement(response)

# Response to any utterance to the bot (Runs the decodder of the deep neural net to get the response) 
@ask.intent("BotIntent")
def ask_intent(text):
        response = get_response_from_model(sess,model,text)
	return statement(response)

if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)
