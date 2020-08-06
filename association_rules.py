from pyspark.mllib.fpm import FPGrowth
from pyspark import SparkContext,SparkConf
# from pyspark.mllib.fpm import
# dir(pyspark.mllib.fpm)

import os
conf=SparkConf().setMaster("local").setAppName("local_test")
sc=SparkContext(conf=conf)
sc.setLogLevel("ERROR")


data = sc.textFile("D:/data_set/asso_data/sample_fpgrowth.txt")
transactions = data.map(lambda line: line.strip().split(' '))
model = FPGrowth.train(transactions, minSupport=0.3, numPartitions=10)
# help(FPGrowth.train)
# model.freqItemsets().show()
# model.associationRules.show()
# model.transform(transactions).show()

result = model.freqItemsets()

tmp = result.collect()
for fi in tmp:
    print(fi)
print('频繁项集数量',len(tmp))

freqDict = result.map(lambda x:[tuple(x[0]), x[1]]).collectAsMap()
print(freqDict)

import itertools

def subSet(listVariable):  # 定义求列表所有非空真子集的函数
    newList = []
    for i in range(1, len(listVariable)):
        newList.extend(list(itertools.combinations(listVariable, i)))
    return newList


def computeConfidence(freqItemset):
    itemset = freqItemset[0]
    freq = freqItemset[1]
    subItemset = subSet(itemset)
    rules = []
    for i in subItemset:
        complement = tuple(set(itemset).difference(set(i)))  # 取补集
        confidence = float(freq) / freqDict[tuple(i)]  # 求置信度
        itemLink = str(complement) + '->' + str(i)
        rule = [itemLink, freq, confidence]
        rules.append(rule)
    return rules


confidence = result.flatMap(computeConfidence) # 使用flatMap可将map映射后的结果“摊平”
tmp_confidence = confidence.collect()
for i in tmp_confidence:
    print(i)
print('计算完成置信度',len(tmp_confidence))
# print(confidence)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

minSupportConfidence = confidence.filter(lambda x: x[2] > 0.5)
for rules in (minSupportConfidence.collect()):
    print(rules)
print('过滤后：',len(minSupportConfidence.collect()))

# print(minSupportConfidence)