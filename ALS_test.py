# -*- coding: UTF-8 -*-
from pyspark import SparkConf, SparkContext, HiveContext, StorageLevel
from pyspark.ml.recommendation import ALS
import os
import sys
import datetime
import pandas as pd

from pyspark.ml.evaluation import RegressionEvaluator

conf = SparkConf().setMaster("yarn-client").setAppName("alsre")\
        .set("spark.yarn.historyServer.address","http://PUSH-MSG-CACHE-02:18088")\
        .set("spark.yarn.historyServer.allowTracking","true")\
        .set("spark.eventLog.dir","hdfs://PUSH-MSG-CACHE-02:8020/user/spark/applicationHistory")\
        .set("spark.default.parallelism",800)
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
hive_context = HiveContext(sc)




item_pd = pd.read_csv('./result/itemId.csv')
user_pd = pd.read_csv('./result/userId.csv')
ratings_pd = pd.read_csv('./result/ratings.csv')

ratings=hive_context.createDataFrame(ratings_pd)
os_app=['com.lenovo.music','com.lenovo.calendar','com.lenovo.browser',
        'com.lenovo.menu_assistant','com.lenovo.themecenter','com.zui.calculator',
        'com.lenovo.safecenter','com.zui.filemanager','com.lenovo.leos.cloud.sync','com.lenovo.scg']


def mapsolve1(imei, recomend):
    res = ""
    num = 0
    tmplist = sorted(dict(recomend).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    tmp_install = list(os_app)
    if imei in installed_dict:
        tmp_install.extend(installed_dict[imei])
    tmp_install = set(tmp_install)
    for k_v in tmplist:
        if dictItem[k_v[0]] not in tmp_install and num < 10:
            res = res + str(dictItem[k_v[0]]) + ':' + str(round(float(1.0 - num * 0.1), 1)) + ";"
            num += 1
    return res




installed_sql = "select * from test.user_download_package_yesterday"
installed_dict = {}
for row in hive_context.sql(installed_sql).collect():
    installed_dict[row[0]] = row[1].split('|')


als = ALS(maxIter=15, regParam=0.02, userCol="userId", itemCol="itemId", ratingCol="rating",rank=13,numUserBlocks=30, numItemBlocks=30,implicitPrefs=True,coldStartStrategy="drop",nonnegative=True)

model = als.fit(ratings)
recs = model.recommendForAllUsers(15).persist(StorageLevel.MEMORY_AND_DISK)


dictUser = user_pd.set_index('userNum')['userId'].to_dict()
dictItem = item_pd.toPandas().set_index('itemNum')['itemId'].to_dict()
recs1=recs.toPandas()
recs1['user']=recs1['userId'].map(dictUser)

result=recs1[['user','recommendations']]

print('start_filter:',datetime.datetime.now())
result['rec_filter_install']=result[['user','recommendations']].apply(lambda x :mapsolve1(x['user'],x['recommendations']),axis=1)
print('finish_filter:',datetime.datetime.now())

finalrec=result[['user','rec_filter_install']]

finalrec.to_csv('./result/test/rec_result_30days_10app_2019_10_24.csv')