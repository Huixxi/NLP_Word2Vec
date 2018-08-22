# -*- encoding:utf-8 -*-
# !bin/env python
import lxml.etree as e
import re
import string
from collections import defaultdict
import numpy as np
import time

doc = e.parse('./dataset/ted_en-20160408.xml')
input_text = '\n'.join(doc.xpath('//content/text()'))
MAX_EXP = 6
EXP_TABLE_SIZE = 1000
EXP_TABLE = []
for i in range(0, EXP_TABLE_SIZE+1):
    EXP_TABLE.append(np.exp(((i-EXP_TABLE_SIZE/2)/(EXP_TABLE_SIZE/2)) * MAX_EXP))

def tokenize(corpus):
    # Pre-Processing Data
    ## remove parenthesis 
    clean_text = re.sub(r"\([^)]*\)", "", corpus)
    ## tokenlize
    sentences = []
    for paragraph in clean_text.split('\n'):
        if paragraph:
            for sentence in paragraph.split('.'):
                if sentence:
                    clean_sent = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]+\ *", " ", sentence)
                    tokens = clean_sent.lower().split()
                    sentences.append(tokens)
    return sentences


class Word2Vec(object):
    
    def __init__(self, sentences, wv_size=100, window=5, min_count=5, sample=1e-4, negative=15, alpha=0.36, min_alpha=0.0001, sg=1):
        np.random.seed(1)
        self.sentences = sentences
        self.wv_size = wv_size
        self.window = window
        self.min_count = min_count
        self.sample = sample
        self.negative = negative
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.sg = sg
        self.vocab = self.vocab()
        self.input_embedding = np.random.uniform(low=-0.5/(wv_size**(3/4)), high=0.5/(wv_size**(3/4)), size=(len(self.vocab), wv_size))
        self.output_weights = np.zeros([len(self.vocab), wv_size])
        self.word_oh_v = np.zeros(len(self.vocab))
        self.G0 = np.zeros_like(self.input_embedding)
        self.G1 = np.zeros_like(self.output_weights)
        self.fudge_factor = 1e-6
    
    def vocab(self, ):
        # sentences: list of sentence token lists
        # [['here', 'are', 'two', 'reasons', 'companies', 'fail', 'they', 'only', 'do', 'more', 'of', 'the', 'same', 'or', 'they', 'only', 'do', 'what', 's', 'new'], [], ...]
        sentences = self.sentences
        vocab = defaultdict(dict)
        vocab_words = ['int']
        vocab['int']['word_count'] = 0 
        vocab_size = 0
        for sent_tokens in sentences:
            vocab_size += len(sent_tokens)
            for word in sent_tokens:
                if not word.isdigit() and word not in vocab:
                    vocab[word]['word_count'] = 1
                    vocab_words.append(word)
                else:
                    if word.isdigit():
                        vocab['int']['word_count'] += 1 
                    else:
                        vocab[word]['word_count'] += 1
        low_freq_words = []
        for word in vocab:
            if vocab[word]['word_count'] < self.min_count:
                low_freq_words.append(word)
        for word in low_freq_words:
            vocab_size -= vocab[word]['word_count']
            del vocab[word]
            vocab_words.remove(word)
        sorted_vocab = []
        for word in vocab:
            sorted_vocab.append((word, vocab[word]['word_count']))
        sorted_vocab.sort(key=lambda tup: tup[1], reverse=True)
        for idx, word in enumerate(sorted_vocab):
            vocab[word[0]]['word_freq'] = vocab[word[0]]['word_count'] / vocab_size
            vocab[word[0]]['word_index'] = idx
        return vocab
       
    # Forward Propagation
    def train_batch_sg(self, ):
        sentences = self.sentences
        vocab = self.vocab
        train_step = 0
        neg_word_list = self.neg_sampling(vocab)
        for sent_tokens in sentences:
            clean_sent = []
            for word in sent_tokens:
                if word.isdigit():
                    word = 'int'
                if word not in vocab:
                    continue
                # Subsampling of High-Freq Word
                keep_prob = min((np.sqrt(vocab[word]['word_freq'] / self.sample) + 1) * (self.sample / vocab[word]['word_freq']), 1)
                keep_list = [1] * int(keep_prob * 1000) + [0] * (1000 - int(keep_prob * 1000))
                if keep_list[np.random.randint(1000)]:
                    clean_sent.append(word)
            for pos, center_word in enumerate(clean_sent):
                b = np.random.randint(0, self.window)
                for pos_c, context in enumerate(clean_sent[max(0, pos - (self.window - b)) : pos + (self.window - b) + 1], max(0, pos - (self.window - b))):
                    if pos_c != pos:
                        train_step += 1
                        # Adaptive 
                        context_idx = self.vocab[context]['word_index']
                        if np.min(self.G0) != 0 and self.alpha/np.min(self.G0) < self.min_alpha:
                            print(train_step)
                        self.train_pair_sg(center_word, context, neg_word_list=neg_word_list, neg=self.negative)

        # Save the final embedding vector matrix
        fname1 = './parameter_data/word_embedding_vector_matrix_test_%sf1.txt' % str(train_step)
        np.savetxt(fname1, self.input_embedding)
        fname2 = './parameter_data/word_embedding_vector_matrix_test_%sf2.txt' % str(train_step)
        np.savetxt(fname2, (self.input_embedding + self.output_weights) / 2)
                
    # Back Propagation
    def train_pair_sg(self, center_w, context_w, neg_word_list, neg=0):
        if neg > 0:
            context_idx = self.vocab[context_w]['word_index']
            center_idx = self.vocab[center_w]['word_index']
            neg_sample = [(center_w, 1)]
            wv_h = self.input_embedding[context_idx]
            # wv_j = self.input_embedding[self.vocab[context_w]['word_index']]
            for i in range(0, neg):
                neg_word = neg_word_list[np.random.randint(0, len(neg_word_list))] 
                if (neg_word, 0) not in neg_sample and neg_word != center_w:
                    neg_sample.append((neg_word, 0))
            # log(P(Wo|Wi)) = log(sigmoid(np.dot(Vt, Vi))) + np.sum(sigmoid(-np.dot(Vn, Vi))  for neg_w in neg_sample[1:]) / (len(neg_sample) - 1)
            # Adagrad
            dh = np.zeros(self.wv_size)
            for neg_w in neg_sample:
                target, label = neg_w[0], neg_w[1]
                neg_w_idx = self.vocab[target]['word_index']
                wv_j = self.output_weights[neg_w_idx]
                dwjh = self.sigmoid(np.dot(wv_h, wv_j)) - label
                dwj = dwjh * wv_h
                self.G1[neg_w_idx] += np.power(dwj, 2)
                dwj /= np.sqrt(self.G1[neg_w_idx]) + self.fudge_factor
                assert dwj.shape == wv_j.shape
                dh += dwjh * wv_j
                # Update the output weight matrix
                self.output_weights[neg_w_idx] -= self.alpha * dwj
            # Update the input embedding matrix
            self.G0[context_idx] += np.power(dh, 2)
            dh /= np.sqrt(self.G0[context_idx]) + self.fudge_factor
            assert dh.shape == wv_h.shape
            self.input_embedding[context_idx] -= self.alpha * dh
    
    # Negative Sampling
    def neg_sampling(self, vocab):
        NEG_SIZE = 1e6
        neg_word_list = []
        sorted_vocab = []
        freq_sum = np.sum(vocab[word]['word_freq']**0.75 for word in vocab)
        for word in vocab:
            sorted_vocab.append((word, vocab[word]['word_freq']))
        sorted_vocab.sort(key=lambda tup: tup[1], reverse=True)
        for word in sorted_vocab:
            neg_word_list.extend([word[0]] * int((word[1]**0.75 / freq_sum) * NEG_SIZE))
        return neg_word_list
    
    def sigmoid(self, x):
        if x > MAX_EXP:
            x = MAX_EXP
        if x < -MAX_EXP:
            x = -MAX_EXP
        exp_x = EXP_TABLE[int((-x / MAX_EXP) * 500 + 500)]
        return 1 / (1 + exp_x)
                        
    def cosine_distance(self, vec1, vec2):
        assert vec1.shape == vec2.shape
        return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))
    
    def most_similar(self, word, topn):
        most_similar_word = []
        word_idx = self.vocab[word]['word_index']
        for w in self.vocab:
            w_idx = self.vocab[w]['word_index']
            if w_idx != word_idx:
                cos_dis = self.cosine_distance(self.input_embedding[word_idx], self.input_embedding[w_idx])
                most_similar_word.append((w, cos_dis))
        most_similar_word.sort(key=lambda tup : tup[1], reverse=True)
        return most_similar_word[0:topn]
    
    # Test the performance with TOEFL Synonym Questions Dataset
    def eval_toefl(self, ):
        ## 生成问答对
        vocab = self.vocab
        imb_matrix = self.input_embedding
        fname = './parameter_data/word_embedding_vector_matrix_test_f3.txt'
        np.savetxt(fname, imb_matrix)
        qst = {}
        qst_ans = {}
        key = []
        val = []
        sub_val = []
        ans = []
        i = 1
        with open('./dataset/toefl-synonymset/toefl.qst', 'r') as f:
            for line in f:
                if i%6 == 1:
                    key.append(line.split()[1])
                if i%6 == 2:
                    sub_val.append((line.split()[1], line.split()[0]))
                if i%6 == 3:
                    sub_val.append((line.split()[1], line.split()[0]))
                if i%6 == 4:
                    sub_val.append((line.split()[1], line.split()[0]))
                if i%6 == 5:
                    sub_val.append((line.split()[1], line.split()[0]))
                if i%6 == 0:
                    val.append(sub_val)
                    sub_val = []
                i += 1
        with open('./dataset/toefl-synonymset/toefl.ans', 'r') as f:
            for line in f:
                if line != '\n':
                    ans.append(line.split()[-1])
        for k, v in zip(key, val):
            qst[k] = dict(v)
        for k, a in zip(key, ans):
            qst_ans[k] = a + '.'
        acc = 0
        total = 0
        for q in qst:
            if q in vocab:
                sim = []
                total += 1
                for c in qst[q]:
                    if c in vocab:
                        vec_q = imb_matrix[vocab[q]['word_index']]
                        vec_c = imb_matrix[vocab[c]['word_index']]
                        cos_dis = self.cosine_distance(vec_q, vec_c)
                        sim.append((qst[q][c], cos_dis))
                    else:
                        sim.append((qst[q][c], 0.0))
                if sim:
                    sim.sort(key=lambda tup: tup[1], reverse=True)
                    if sim[0][0] == qst_ans[q]:
                        acc += 1
        return acc, total, acc/total
    
if __name__ == '__main__':
    sentences = tokenize(input_text)
    word2vec = Word2Vec(sentences)
    vocabulary = word2vec.vocab
    b1 = time.time()
    word2vec.train_batch_sg()
    train_time = time.time() - b1
    most_similar_word_science = word2vec.most_similar('science', topn=10)
    most_similar_word_man = word2vec.most_similar('man', topn=10)
    most_similar_word_kill = word2vec.most_similar('kill', topn=10)
    most_similar_word_red = word2vec.most_similar('red', topn=10)
    most_similar_word_monday = word2vec.most_similar('monday', topn=10)
    accuracy = word2vec.eval_teofl()
    with open('most_similar_word.txt', 'w') as f:
        f.write(str(train_time))
        f.write('\n')
        f.write(str(most_similar_word_science))
        f.write('\n')
        f.write(str(most_similar_word_man))
        f.write('\n')
        f.write(str(most_similar_word_kill))
        f.write('\n')
        f.write(str(most_similar_word_red))
        f.write('\n')
        f.write(str(most_similar_word_monday))
        f.write('\n')
        f.write(str(accuracy))
