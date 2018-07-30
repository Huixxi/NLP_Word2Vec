# NLP_Word2Vec
I have coded a classic word2vec model: skip-gram model with negative sampling as the optimization method by hand in pure python3 and use TED-Talks-Dataset as the training set.  
#### GOAL:
> **Building a skip-gram model with negative sampling to achieve that:**  
Given a specific word in the middle of a sentence (the input word), look at the words nearby and pick one at random. The network is going to tell us the probability for every word in our vocabulary of being the “nearby word” that we chose.  

#### REFERANCE:
> **Blog:**  
01.[Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)  
02.[Word2Vec Tutorial - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)  
03.[Deep Learning实战之word2vec](https://kexue.fm/usr/uploads/2017/04/146269300.pdf)  
04.[Word2Vec and FastText Word Embedding with Gensim](https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c)  
05.[A Gentle Introduction to the Bag-of-Words Model](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)  
06.[Python implementation of Word2Vec](http://www.claudiobellei.com/2018/01/07/backprop-word2vec-python/)  

> **Paper:**  
01.[Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  
02.[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)  
03.[Word2vec Parameter Learning Explained](https://arxiv.org/pdf/1411.2738.pdf)  
04.[Linguistic Regularities in Continuous Space Word Representations](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf)  
05.[Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)  
06.[Word and Phrase Translation with word2vec](https://arxiv.org/pdf/1705.03127.pdf)  
07.[word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method](https://arxiv.org/pdf/1402.3722.pdf)  

> **Video:**  
01.[Negative Sampling-Coursera Deeplearning](https://www.coursera.org/lecture/nlp-sequence-models/negative-sampling-Iwx0e)  

> **Code:**  
01.[word2vec_commented_in_C](https://github.com/chrisjmccormick/word2vec_commented)  
02.[word2vec code in python](https://radimrehurek.com/gensim/models/word2vec.html)

#### DATASET:
> 01.[TED-Talks-Dataset](https://wit3.fbk.eu/)  
**Other datasets:**  
[SNLI](https://nlp.stanford.edu/projects/snli/)、[NER]、[SQuAD]、[Coref]、[SRL]、[SST-5]、[Parsing]
