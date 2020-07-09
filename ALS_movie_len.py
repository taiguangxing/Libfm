
from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark import SparkContext,SparkConf
import pandas as pd
import datetime

# import os
# path = os.getcwd()
# print(path)
# print(__file__)
import os

def save_mode(model,sc,path):
    try:
        model.save(sc,path)
    except Exception:
        print ("保存模型出错")



def create_spark_context():
    conf = SparkConf().setMaster("local").setAppName("local_test")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    #
    # spark_conf = SparkConf()\
    #     .setAppName('Python_Spark_WordCount')\
    #     .setMaster('local[4]') \
    #     .set("spark.driver.extraJavaOptions", "-Xss4096k")
    #
    # spark_context = SparkContext(conf=spark_conf) # 获取SparkContext实例对象,
    # spark_context.setLogLevel('WARN')             # 设置日志级别
    return sc

package = 'mllib'


if __name__ == '__main__':
    sc = create_spark_context()
    spark = SparkSession.builder.getOrCreate()
    data = pd.read_csv('D:/data_set/movie-len/ml-latest-small/ml-latest-small/ratings.csv')
    print(data.head(5))
    data.drop('timestamp',axis=1,inplace=True)
    ratings = spark.createDataFrame(data)
    (training, test) = ratings.randomSplit([0.8, 0.2])

    if package == 'mllib':
        from pyspark.mllib.recommendation import MatrixFactorizationModel, ALS
        # Build the recommendation model using ALS on the training data
        model = ALS.train(training,5, iterations=20, lambda_=0.1)
        model = ALS.trainImplicit(training,5, iterations=20, lambda_=0.1)
        save_mode(model,sc,'D:/python_project/Libfm/model/mlib_als_model_{0}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')))
        # model.recommendProductsForUsers(5)
        # predictions = model.predictAll()
        #
        # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
        #                                 predictionCol="prediction")
        # rmse = evaluator.evaluate(predictions)
        # print("Root-mean-square error = " + str(rmse))
    else:
        from pyspark.ml.recommendation import ALS
        als = ALS(rank=20, maxIter=20, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
        model = ALS.fit(training)
        model.save('D:/python_project/Libfm/model/ml_als_model')
        predictions = model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print("Root-mean-square error = " + str(rmse))




    #利用模型为每一个用户推荐5个视频
    recs = model.recommendForAllUsers(5)
    recommend_res = recs.toPandas()
    recommend_res.to_csv('D:/python_project/Libfm/res/als_recommend.csv')
    sc.stop()