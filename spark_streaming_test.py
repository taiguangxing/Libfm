from __future__ import print_function
import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import json
# {'timestamp': 1594846806854,
#  'value': b'download\x01868591041692871\x01com.tencent.mobileqq\x012020-07-16 05:00:06\x01bizIdentity:com.tencent.mobileqq|'
#           b'bizType:updateType|appType:1|imei:868591041692871|org:-1\x01leapp://ptn/appmanager.do?page=update',
#  'partition': 2, 'topic': 'recommend-engine-like', 'key': b'159484680685424', 'offset': 138537387}


if __name__ == "__main__":

    sc = SparkContext(appName="streamingkafka",master='local[1]')
    sc.setLogLevel("ERROR") #
    ssc = StreamingContext(sc, 10) #
    # ssc.checkpoint('hdfs://10.0.4.151:8022/user/ire/user_interest_data/')
    # brokers='10.0.133.106:3200,10.0.133.105:3200,10.0.133.104:3200'
    # topic = {'recommend-engine-like':1}
    # group_id = "test"
    # lines = KafkaUtils.createStream(ssc,zkQuorum=brokers,groupId=group_id,topics=topic)
    lines = ssc.textFileStream('log/')
    # lines.show()
    print(lines.count())
    lines.pprint(5)

    #
    # linesTmp = lines.map(lambda x: json.loads(x[1].split("|")[1])).map(lambda x: (x["cm"]["md"], 1)) \
    #     .reduceByKey(lambda a, b: a + b).updateStateByKey(updateFunc)
    #
    #     # createDirectStream(ssc, [topic], {"metadata.broker.list": brokers})
    # lines_rdd = kafka_streaming_rdd.map(lambda x: x[1])
    # counts = lines.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)

    # counts.pprint()
    ssc.start()
    ssc.awaitTermination(10)
