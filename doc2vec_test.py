#!/usr/bin/env python
#coding=utf-8
from gensim.utils import  simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.models as gm
import jieba
import gensim


def load_corpus(train_corpus):
    documents = []
    with open(train_corpus,'r') as f:
        for line in f.readlines():
            line = line.strip() 
            id,title = line.split("\t")
            cut_title = jieba.lcut(title)
            documents.append(gm.doc2vec.TaggedDocument(str(cut_title),[str(id)]))
    return documents


#加载已存在的doc2vec模型
model_file = "model.bin"
model = Doc2Vec.load(model_file,mmap='r')

#获取tag向量的方法
tag_1 = model.docvecs[1]
tag_2 = model.docvecs[2]
#print("tag_1:",tag_1)

# 比较两个数据的相似度
sim = model.docvecs.similarity(0,2)
print("similar:",sim)

mem_size = model.estimate_memory()
#print("mem_size:",mem_size)

#将doc2vec模型中的word2vec矩阵数据保存下来
#model.save_word2vec_format("wordvec1")
print(model)
#当前模型中预料的条数
print(model.corpus_count)

#载入增量预料
train_corpus = "add_train_file"
corpus_doc = load_corpus(train_corpus)

print("new dict len:",len(corpus_doc))
#增量训练模型，total_examples是必须的，其值为当前预料和增量预料之和
model.train(corpus_doc,total_examples=model.corpus_count+len(corpus_doc), epochs=40)
print(model)
#测试一下增量训练的模型能不能直接得到增量预料中tag的向量值，结果报错了。
vec_ch = model.docvecs[102]
#再次保存增量训练后的模型的word2vec矩阵，但是增量预料数据并没有对应的vec;
model.save_word2vec_format("wordvec2")

