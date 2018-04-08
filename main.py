"""
Translation with a Sequence to Sequence Network and Attention
*************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_

In this project we will be teaching a neural network to translate from
French to English.

::

    [KEY: > input, = target, < output]

    > il est en train de peindre un tableau .
    = he is painting a picture .
    < he is painting a picture .

    > pourquoi ne pas essayer ce vin delicieux ?
    = why not try that delicious wine ?
    < why not try that delicious wine ?

    > elle n est pas poete mais romanciere .
    = she is not a poet but a novelist .
    < she not not a poet but a novelist .

    > vous etes trop maigre .
    = you re too skinny .
    < you re all alone .

... to varying degrees of success.

This is made possible by the simple but powerful idea of the `sequence
to sequence network <http://arxiv.org/abs/1409.3215>`__, in which two
recurrent neural networks work together to transform one sequence to
another. An encoder network condenses an input sequence into a vector,
and a decoder network unfolds that vector into a new sequence.

.. figure:: /_static/img/seq-seq-images/seq2seq.png
   :alt:

To improve upon this model we'll use an `attention
mechanism <https://arxiv.org/abs/1409.0473>`__, which lets the decoder
learn to focus over a specific range of the input sequence.

**Recommended Reading:**

I assume you have at least installed PyTorch, know Python, and
understand Tensors:

-  http://pytorch.org/ For installation instructions
-  :doc:`/beginner/deep_learning_60min_blitz` to get started with PyTorch in general
-  :doc:`/beginner/pytorch_with_examples` for a wide and deep overview
-  :doc:`/beginner/former_torchies_tutorial` if you are former Lua Torch user


It would also be useful to know about Sequence to Sequence networks and
how they work:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <http://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <http://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <http://arxiv.org/abs/1506.05869>`__

You will also find the previous tutorials on
:doc:`/intermediate/char_rnn_classification_tutorial`
and :doc:`/intermediate/char_rnn_generation_tutorial`
helpful as those concepts are very similar to the Encoder and Decoder
models, respectively.

And for more, read the papers that introduced these topics:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <http://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <http://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <http://arxiv.org/abs/1506.05869>`__


**Requirements**
"""
import argparse
from train import trainIters
from test import evaluateRandomly, evaluateInput
from load import prepareData, input_lang, output_lang
from model import *

def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', type=bool, default=False, help='Train the model with corpus')
    parser.add_argument('-te', '--test', type=bool, default=False, help='Test the saved model')
    parser.add_argument('-l', '--load', type=bool, default=False, help='Load the model and train')
    parser.add_argument('-c', '--corpus', type=str, default='', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('-it', '--iteration', type=int, default=50000, help='Train the model with it iterations')
    parser.add_argument('-p', '--print', type=int, default=1000, help='Print every p iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    parser.add_argument('-hi', '--hidden', type=int, default=256, help='Hidden size in encoder and decoder')
    parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-s', '--save', type=float, default=10000, help='Save every s iterations')
    parser.add_argument('-model', '--model', type=str, default='', help='Model file path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    if args.train:
        hidden_size = args.hidden
        print_every = args.print
        save_every = args.save
        iteration = args.iteration
        encoder = EncoderRNN(input_lang.n_words, hidden_size, input_lang)
        attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, output_lang, dropout_p=0.1)
        trainIters(encoder, attn_decoder, iteration, print_every=print_every, save_every = save_every)
    if args.test:
        hidden_size = args.hidden
        encoder = EncoderRNN(input_lang.n_words, hidden_size, input_lang)
        attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, output_lang, dropout_p=0.1)
        modelFile = args.model
        checkpoint = torch.load(modelFile)
        encoder.load_state_dict(checkpoint['en'])
        attn_decoder.load_state_dict(checkpoint['de'])
        evaluateInput(encoder, attn_decoder)


