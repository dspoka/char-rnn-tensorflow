import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        # if infer:
        #     args.batch_size = 1
        #     args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size, state_is_tuple=True)

        self.sample_bool = args.sample_bool

        self.batch_size = 1 if self.sample_bool else args.batch_size

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)

        print("Seq LENGth:", args.seq_length)

        self.input_data = tf.placeholder(tf.int32,[None, args.seq_length], name="input_data")
        self.targets = tf.placeholder(tf.int32, [None, args.seq_length], name="targets")

        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
        #     softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
        #     softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        #     softmax_w_2 = tf.get_variable("softmax_w_2", [args.vocab_size, args.vocab_size])
        #     softmax_b_2 = tf.get_variable("softmax_b_2", [args.vocab_size])

            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                encoder_inputs = tf.split(1, args.seq_length, self.input_data)
                encoder_inputs = [tf.squeeze(input_, [1]) for input_ in encoder_inputs]

                decoder_inputs = tf.split(1, args.seq_length, self.targets)
                decoder_inputs = [tf.squeeze(input_, [1]) for input_ in decoder_inputs]

                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            # tf.stop_gradient disallows the gradient to backpropagate
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            # prev symbol is index. returns vector of the symbol.
            return tf.nn.embedding_lookup(embedding, prev_symbol)
        # tensorboard
        self.embedding = embedding

        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs

        self.logits, last_state = seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                args.vocab_size,#num_encoder_symbols,
                                args.vocab_size,#num_decoder_symbols,
                                128,#embedding size
                                num_heads=1,
                                output_projection=None,
                                feed_previous=self.sample_bool,
                                scope=None,
                                initial_state_attention=False)

        # outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')


        # output = tf.reshape(tf.concat(1, outputs), [-1, args.vocab_size])
        # self.logits = tf.matmul(output, softmax_w_2) + softmax_b_2

        # output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        # self.logits = tf.matmul(output, softmax_w) + softmax_b

        # (args.seq_length, self.batch_size, args.num_decoder)
        self.probs = tf.transpose(tf.pack(self.logits), [1, 0, 2])

        if(not args.sample_bool):
            loss = seq2seq.sequence_loss(self.logits,
                    self.decoder_inputs,
                    [tf.ones(self.batch_size) for _ in range(args.seq_length)],
                    args.vocab_size)

            self.cost = tf.reduce_sum(loss) / args.seq_length
            self.final_state = last_state
            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                    args.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1, saved_args=None):
        start_targets = np.zeros((1,saved_args.seq_length))
        start_targets[0,0] = vocab['<s>']

        # state = sess.run(self.cell.zero_state(1, tf.float32))
        # for char in prime[:-1]:
        # x[0, 0] = vocab['']

        if(len(prime) > saved_args.seq_length):
            pass
            #cut off part of prime
        else:
            #prepad with zeros
            x = np.array([vocab[char] for char in prime])
            x = np.pad(x, ((saved_args.seq_length - len(prime), 0)), 'constant')
            x = np.expand_dims(x, 0)

        feed = {self.input_data: x, self.targets: start_targets}
        [probs, state] = sess.run([self.probs, self.final_state], feed)


        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        p = probs[0]
        # (self.seq_length, args.num_decoder)
        char_predictions = np.argmax(p,1)
        print "PREDICTIONS"
        print [chars[_] for _ in char_predictions]
        return ''.join([chars[_] for _ in char_predictions])


        # ret = prime
        # char = prime[-1]
        # for n in range(num):
        #     x = np.zeros((1, 1))
        #     x[0, 0] = vocab[char]

        #     feed = {self.input_data: x, self.targets: start_targets}
        #     [probs, state] = sess.run([self.probs, self.final_state], feed)
        #     p = probs[0]

        #     if sampling_type == 0:
        #         sample = np.argmax(p)
        #     elif sampling_type == 2:
        #         if char == ' ':
        #             sample = weighted_pick(p)
        #         else:
        #             sample = np.argmax(p)
        #     else: # sampling_type == 1 default:
        #         sample = weighted_pick(p)

        #     pred = chars[sample]
        #     ret += pred
        #     char = pred
        # return ret


