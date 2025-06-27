# 2025Tianchi-LLM-QA

from langchain.schema import Document
from langchain.vectorstores import Chroma,FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pdf_parse import DataProcess
import torch
# from bm25_retriever import BM25


device = 'cuda:3'

class FaissRetriever(object):
    def __init__(self, model_path, data):
        self.embeddings  = HuggingFaceEmbeddings(
                               model_name = model_path,
                               model_kwargs = {"device": device}
                           )
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        # self.vector_store.save_local(PERSIST_DIR)
        # del self.embeddings
        torch.cuda.empty_cache()

    def GetTopK(self, query, k):
       context = self.vector_store.similarity_search_with_score(query, k=k)
       return context

    def GetvectorStore(self):
        return self.vector_store


if __name__ == "__main__":
    # base = ""
    model_name="/DATA/disk0/public_model_weights/bge-large-zh-v1.5" #text2vec-large-chinese
    data = open('./data/all_text_split.txt','r',encoding='utf-8').read().splitlines()
    data = [d for d in data if '知识库检索' not in d or '发布时间' not in d]
    print(len(data))



    faissretriever = FaissRetriever(model_name, data)
    # bm25 = BM25(data)
    faiss_ans = faissretriever.GetTopK("根据2022年度报告，中国联通的企业定位是什么？", 10)









#!/usr/bin/env python
# coding: utf-8


from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from pdf_parse import DataProcess
import jieba


'''
这段代码实现了一个基于 BM25 (Best Matching 25) 算法的文档检索系统。BM25 是一种常用于信息检索系统中的排序算法，
用于根据查询（query）和文档之间的匹配程度来计算文档的相关性分数。它考虑了词频和逆文档频率等因素，并为每个文档分配一个分数，表示该文档与查询的匹配程度。
'''
class BM25(object):

    def __init__(self, documents:list):
        '''
        构造函数初始化了一个 BM25 对象，接受一个文档列表 documents 作为输入。
        对于每个文档（以行的形式提供），先进行一些清理操作（去除多余的换行符和空格）。
        如果文档内容长度小于5个字符，则跳过该文档。
        使用 jieba.cut_for_search(line) 对文档进行中文分词，并将分词结果连接为一个空格分隔的字符串。
        分词后的文档存储为 Document 对象，并附加一个 metadata 字典，其中包含文档的唯一 id（通过 idx 索引获得）。
        对于原始文档（未经过分词），也将其保存到 full_docs 列表中，以便后续使用原始文档内容进行输出。
        self.documents 保存了分词后的文档集合，self.full_documents 保存了未处理的原始文档。
        最后，调用 _init_bm25 方法初始化 BM25 检索器。
        '''
        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            line = line.strip("\n").strip()
            if(len(line)<5):
                continue
            tokens = " ".join(jieba.cut_for_search(line))
            # docs.append(Document(page_content=tokens, metadata={"id": idx, "cate":words[1],"pageid":words[2]}))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            # full_docs.append(Document(page_content=words[0], metadata={"id": idx, "cate":words[1], "pageid":words[2]}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.documents = docs
        self.full_documents = full_docs
        self.retriever = self._init_bm25()

    # 初始化BM25的知识库
    def _init_bm25(self):
        '''
        这个方法初始化 BM25 检索器（BM25Retriever）并返回它。from_documents 方法可能是将输入的分词后文档列表（self.documents）
        转换为 BM25 检索器可以理解的格式。这使得 BM25 算法能够使用该集合进行检索操作。
        '''
        return BM25Retriever.from_documents(self.documents)

    # 获得得分在topk的文档和分数
    def GetBM25TopK(self, query, topk):
        '''
        该方法接受一个查询 query 和一个参数 topk，返回与查询最相关的前 topk 个文档。
        首先，设置 BM25 检索器的 k 参数为 topk，表示检索返回的文档数量。
        将查询 query 进行中文分词（与文档的分词方式一致），并将分词结果连接为一个空格分隔的字符串。
        调用 self.retriever.get_relevant_documents(query) 获取与查询相关的文档。get_relevant_documents 返回的是与查询最相关的文档索引列表（可能按相关性排序）。
        然后，通过这些索引，查找原始的文档 full_documents 并将其保存到 ans 列表中。
        最后，返回 ans 列表，其中包含了最相关的文档。
        '''
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans

if __name__ == "__main__":
    # bm2.5
    # dp =  DataProcess(pdf_path = "/DATA/disk0/lzl/other/NLPTask/05文本数据知识库检索/Tianchi-LLM-QA-main/data/train_a.pdf")
    # dp.ParseBlock(max_seq = 1024)
    # dp.ParseBlock(max_seq = 512)
    # print(len(dp.data))
    # dp.ParseAllPage(max_seq = 256)
    # dp.ParseAllPage(max_seq = 512)
    # print(len(dp.data))
    # dp.ParseOnePageWithRule(max_seq = 256)
    # dp.ParseOnePageWithRule(max_seq = 512)
    # print(len(dp.data))
    # data = dp.data

    data = open('./data/all_text_split.txt','r',encoding='utf-8').read().splitlines()
    data = [d for d in data if '知识库检索' not in d or '发布时间' not in d]
    print(len(data))

    bm25 = BM25(data)
    res = bm25.GetBM25TopK("根据2022年度报告，中国联通的企业定位是什么？", 10)
    print(res)
