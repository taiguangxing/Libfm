from pyspark.sql import SQLContext,HiveContext
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
import numpy as np
from pyspark.sql.types import IntegerType
import pandas as pd
spark = SparkSession.builder.appName('LSH').config('spark.executor.memory','2g').getOrCreate()
# pd_data = pd.read_csv('./data/toutiao_title_bert_feature.csv')
# pd_data['features'] = pd_data.applymap([])
# sc = SparkContext(appName="read_mysql", master='local')
# sc.setLogLevel("ERROR")  #
# sqlContext = SQLContext(sc)
# testDF = sc.parallelize([(1,'a'),(1,'aa'),(2,'b'),(2,'bb'),(3,'c')],3).toDF(['ind','state'])
# testDF.show()
# testDF.registerTempTable('tmp_table')
# sqlContext.sql('select * from tmp_table').show()
#
jdbcDf = spark.read.format('jdbc').options(url = 'jdbc:mysql://localhost:3306/rec',driver = 'com.mysql.jdbc.Driver',dbtable ="rec_article_pool",user = 'root',password ='root').load()
# jdbcDf.show()
dfA=jdbcDf.selectExpr('id','title_feature as features')
dfA.show()
# dfA = dfA.withColumn('id',dfA['id'].ast(IntegerType()))
# dfA.show()
def trans_str_to_vector(row):
        return row.id,[float(x) for x in np.array(np.mat(row.features))[0]]

dfA = dfA.rdd.map(trans_str_to_vector).toDF(['id','features'])
print(dfA)
def _trans_to_vetor_dense(partitions):
        for row in partitions:
                yield row.id,Vectors.dense(row.features)

dfA = dfA.rdd.mapPartitions(_trans_to_vetor_dense).toDF(['id','features'])
dfA.show()
print(dfA)

brp = BucketedRandomProjectionLSH().setInputCol('features').setOutputCol('hashes').setNumHashTables(4).setBucketLength(10)
model = brp.fit(dfA)
model.transform(dfA).show()
# #
resDF = model.approxSimilarityJoin(dfA, dfA, 20.0, "EuclideanDistance").selectExpr('int(datasetA.id) as id1','int(datasetB.id) as id2','EuclideanDistance') #datasetA.features'
resDF.sort("EuclideanDistance").show()
resDF.count()
resDF= resDF.withColumnRenamed('title_feature','features')
#
# resDF.write.jdbc('jdbc:mysql://localhost:3306/rec?useSSL=false','sim_vector',mode = 'overwrite',properties={"user":'root',"password":"root"})
spark.stop()