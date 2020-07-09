from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml.recommendation import ALS

from pyspark.sql import SparkSession
from pyspark import SparkContext,SparkConf
import pandas as pd
from pyspark.mllib.recommendation import MatrixFactorizationModel
import sys

def create_spark_context():
    conf = SparkConf().setMaster("local").setAppName("local_test")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    return sc

def load_model(sc,model_path):                    # 加载模型
    # try:
    model = MatrixFactorizationModel.load(sc,model_path)
    return model
    # except Exception:
    #         print ("加载模型出错")




def recommend_movies(als, movies, user_id):
    rmd_movies = als.recommendProducts(user_id, 10)
    print('推荐的电影为：{}'.format(rmd_movies))
    for rmd in rmd_movies:
        print("为用户{}推荐的电影为：{}".format(rmd[0], movies[rmd[1]]))
    return rmd_movies

def recommend_users(als, movies, movie_id):      # 为每个电影推荐10个用户
    rmd_users = als.recommendUsers(movie_id, 10)
    print('针对电影ID：{0},电影名:{1},推荐十个用户为:'.format(movie_id, movies[movie_id]))
    for rmd in rmd_users:
        print("推荐用户ID：{},推荐评分：{}".format(rmd[0], rmd[2]))


def recommend(als_model, movie_dic):
    if sys.argv[1] == '--U':                     # 推荐电影给用户
        recommend_movies(als_model, movie_dic, int(sys.argv[2]))
    if sys.argv[1] == '--M':                     # 推荐用户给电影
        recommend_users(als_model, movie_dic, int(sys.argv[2]))


def get_movie_title_dict(data_path):
    import pandas as pd
    movie = pd.read_csv(data_path)
    print(movie.head(5))
    res_dict ={}
    for ind,row in movie.iterrows():
        res_dict[row[0]]=row[1]
    # movie_dict = movie[['movieId','title']].to_dict(into='dict')
    return res_dict


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("请输入2个参数, 要么是: --U user_id,  要么是: --M movie_id")
        exit(-1)
    sc = create_spark_context()
    # als= ALS()
    # als_model= als.load('D:/python_project/Libfm/model/ml_als_model')
    als_model = load_model(sc,'D:/python_project/Libfm/model/als_model')
    user_feature = als_model.userFeatures().collect()
    print(user_feature[:10])
    movie_dict = get_movie_title_dict('D:/data_set/movie-len/ml-latest-small/ml-latest-small/movies.csv')
    recommend(als_model,movie_dict)
    sc.stop()





