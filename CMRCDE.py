from sklearn.feature_extraction.text import TfidfVectorizer
import synonyms
import json
import jieba


def loadCorpus(filepath):
    with open(filepath, 'r') as load_f:
        load_dict = json.load(load_f)
    corpus = [item["context_text"] for item in load_dict]
    sent_words = [list(jieba.cut(sent0)) for sent0 in corpus]
    splitCorpus = [" ".join(sent0) for sent0 in sent_words]
    return load_dict, splitCorpus


def getCorpusTfIdf(corpus):
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(corpus).toarray()
    tfidf_vocab = tfidf_vec.vocabulary_
    return tfidf_matrix, tfidf_vocab


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def getKeyWords(tfidf_matrix, tfidf_vocab):
    keyWords = [get_key(tfidf_vocab, list(tfidfs).index(max(tfidfs)))[0] for tfidfs in tfidf_matrix]
    return keyWords


def replaceKeyword(keyword):
    syWords = synonyms.nearby(keyword)
    if len(syWords[0]) != 0:
        reKeyword = syWords[0][1]
        return keyword, reKeyword
    else:
        return keyword


def replaceDocument(keyword, reKeyword, document):

    context_text = document["context_text"]
    document["context_text"] = context_text.replace(keyword, reKeyword)
    for qa in enumerate(document['qas']):
        query_text = qa[1]['query_text']
        if keyword in query_text:
            document['qas'][qa[0]]["query_text"] = query_text.replace(keyword, reKeyword)
        if qa[1]["answers"]:
            answer = qa[1]["answers"][0]
            # TODO answer为float的情况
            if keyword in str(answer):
                document['qas'][qa[0]]["answers"][0] = answer.replace(keyword, reKeyword)
    return document


def replace(loadFilepath, dumpFilepath):
    validReCount = 0
    reKeyWords = []
    dump_dict = []
    print("开始加载旧文档处理语料......")
    load_dict, splitCorpus = loadCorpus(loadFilepath)
    print("语料处理完毕......")
    print("开始生成tfidf值......")
    tfidf_matrix, tfidf_vocab = getCorpusTfIdf(splitCorpus)
    print("tfidf值生成完毕......")
    print("开始提取关键词......")
    keyWords = getKeyWords(tfidf_matrix, tfidf_vocab)
    print("关键词提取完毕......")
    print("开始转换近义词......")
    for keyWord in keyWords:
        reKeyWord = replaceKeyword(keyWord)
        if type(reKeyWord) == tuple:
            validReCount += 1
            reKeyWords.append(reKeyWord[1])
        else:
            reKeyWords.append(reKeyWord)
    print("近义词转换完毕......")
    print("开始替换文档......")
    for index, document in enumerate(load_dict):
        dump_dict.append(replaceDocument(keyWords[index], reKeyWords[index], document))
    print("文档替换完毕......")
    print("新文档写入文件......")
    with open(dumpFilepath, 'w') as dump_f:
        json.dump(dump_dict, dump_f)
    print("新文档写入完毕......")
    return validReCount


if __name__ == '__main__':
    # print(replaceKeyword("开心"))
    loadFilepath = "data/cmrc2018_train.json"
    dumpFilepath = "data/cmrc2018_train_DE.json"
    validReCount = replace(loadFilepath, dumpFilepath)
    print("替换近义词有效文档数为: ", validReCount)