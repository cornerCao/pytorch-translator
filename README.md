# pytorch seq2seq translator

------

用pytorch实现的一个简易的translator，可以根据训练的corpus来改变翻译的语言，主要包括：

> * main.py：main函数，可以选择train/test模式，自定义hidden layer、iteration、corpus、model等。
> * model.py：定义模型结构，实现了基本的RNNEncoder, RNNDecoder, AttnDecoderRNN, 详细结构见[pytorch tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)。在原始模型上加上了中文分词（jieba）和载入word2vec pre-trained embedding（gensim）。
> * train.py：训练模型
> * test.py：用于测试模型，支持输入英文，输出对应的中文
> * load.py：用于加载语料、embedding

模仿了[pytorch-chatbot](https://github.com/ywk991112/pytorch-chatbot)和[pytorch tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)的实现
