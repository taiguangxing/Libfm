from pyspark.sql import SQLContext,HiveContext
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName('test').config('spark.executor.memory','2g').getOrCreate()


# sc = SparkContext(appName="read_mysql", master='local')
# sc.setLogLevel("ERROR")  #
# sqlContext = SQLContext(sc)
# testDF = sc.parallelize([(1,'a'),(1,'aa'),(2,'b'),(2,'bb'),(3,'c')],3).toDF(['ind','state'])
# testDF.show()
# testDF.registerTempTable('tmp_table')
# sqlContext.sql('select * from tmp_table').show()
#
# jdbcDf = spark.read.format('jdbc').options(url = 'jdbc:mysql://localhost:3306/rec',driver = 'com.mysql.jdbc.Driver',dbtable ="rec_ori_article",user = 'root',password ='root').load()
# jdbcDf.show()


dfA = spark.createDataFrame([(0, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (2, Vectors.dense(-1.0, -1.0)),
      (3, Vectors.dense(-1.0, 1.0))]).toDF('id','features')

dfA.show()

dfB = spark.createDataFrame([
    (4, Vectors.dense(1.0, 1.0)),
    (5, Vectors.dense(-1.0, 0.0)),
    (6, Vectors.dense(0.0, 1.0)),
    (7, Vectors.dense(0.0, -1.0))
]).toDF('id','features')



def mul_vector(row):
    return row.id,2*row.features

dfA=dfA.rdd.map(mul_vector).toDF(['id','features'])
dfA.show()


#
# brp = BucketedRandomProjectionLSH().setInputCol('features').setOutputCol('hashes').setNumHashTables(3).setBucketLength(2.0)
# model = brp.fit(dfA)
# model.transform(dfA).show()
#
# resDF = model.approxSimilarityJoin(dfA, dfB, 2.5, "EuclideanDistance").select('datasetA.id','datasetB.id','EuclideanDistance','datasetA.features','datasetB.features')
# resDF.describ()
#
# resDF = resDF.selectExpr('datasetA.id as id1','datasetB.id as id2','round(EuclideanDistance)','datasetA.features')
# resDF.show()
#
#     # .toDF('dfaId','dfbId','EuclideanDistance')
#
# resDF.write.jdbc('jdbc:mysql://localhost:3306/rec?useSSL=false','sim_vector',mode = 'overwrite',properties={"user":'root',"password":"root"})
#
# spark.stop()