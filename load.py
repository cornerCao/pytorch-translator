from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import jieba
import logging
from gensim.models import Word2Vec
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

class Lang:
    def __init__(self, name):
        self.name = name
        self.path = 'model/word2vec_'+name+'.vec'
        self.path2 = 'model/word2vec_' + name + '.txt'
        self.word2index = {'sos':0, 'eos':1}
        self.index2word = {0: "sos", 1: "eos"}
        self.word2count = {'sos':0, 'eos':0}
        self.model = None
        self.n_words = 2  # Count SOS and EOS
        self.embeddingWeight = []

    def getEmbedding(self, corpus, trainWord2Vec=True):
        print(len(corpus), corpus[0])
        if trainWord2Vec:
            self.model = word2vec.Word2Vec(corpus, size=256, window=5, min_count=1, workers=4)
            self.model.save(self.path)
            self.model.wv.save_word2vec_format(self.path2)
        else:
            self.model = word2vec.Word2Vec.load(self.path)
        for i in range(0, self.n_words):
            word = self.index2word[i]
            vec = self.model.wv[word]
            self.embeddingWeight.append(vec)

    def getVec(self, word):
        return self.model.wv[word]

    def getWord(self, vec):
        return self.model.wv.similar_by_vector(vec, topn=1)

    def contains(self, word):
        if word in self.model.wv:
            return True
        return False

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#
# 把unicode字符串转换成ascii，将unicode字符标准化
# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?？])", r" \1", s)
    s = 'sos ' + s + ' eos'
#    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def normalizeString2(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?？])", r" \1", s)
#    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

MAX_LENGTH = 30

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    corpus1 = []
    corpus2 = []
    for pair in pairs:
        corpus1.append(pair[0].split(' '))
        if lang2 == 'cmn':#中文需要分词
            cutwords = list(jieba.cut(pair[1]))
            pair[1] =  ' '.join(cutwords)
        corpus2.append(pair[1].split(' '))
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    input_lang.getEmbedding(corpus1, trainWord2Vec=False)
    output_lang.getEmbedding(corpus2, trainWord2Vec=False)
    return input_lang, output_lang, pairs


#input_lang, output_lang, pairs = prepareData('eng', 'cmn', True)
#print(random.choice(pairs))


######################################################################
# Loading data files
# ==================
#
# The data for this project is a set of many thousands of English to
# French translation pairs.
#
# `This question on Open Data Stack
# Exchange <http://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages>`__
# pointed me to the open translation site http://tatoeba.org/ which has
# downloads available at http://tatoeba.org/eng/downloads - and better
# yet, someone did the extra work of splitting language pairs into
# individual text files here: http://www.manythings.org/anki/
#
# The English to French pairs are too big to include in the repo, so
# download to ``data/eng-fra.txt`` before continuing. The file is a tab
# separated list of translation pairs:
#
# ::
#
#     I am cold.    J'ai froid.
#
# .. Note::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/data.zip>`_
#    and extract it to the current directory.

######################################################################
# Similar to the character encoding used in the character-level RNN
# tutorials, we will be representing each word in a language as a one-hot
# vector, or giant vector of zeros except for a single one (at the index
# of the word). Compared to the dozens of characters that might exist in a
# language, there are many many more words, so the encoding vector is much
# larger. We will however cheat a bit and trim the data to only use a few
# thousand words per language.
#
# .. figure:: /_static/img/seq-seq-images/word-encoding.png
#    :alt:
#
#


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
input_lang, output_lang, pairs = prepareData('eng', 'cmn', False)
