
# coding: utf-8

#import findspark
#findspark.init()

from pyspark.sql import SQLContext
from pyspark.sql.types import *

from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from operator import add
import sys
import numpy as np
import pandas as pd
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import DoubleType,IntegerType
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

sc = SparkContext("local",'app')
spark = SparkSession.builder.appName('name').config('spark.sql.shuffle.partitions',10).getOrCreate()

# # 1. load data

data=spark.read.csv('./data/train_flight.csv',header=True,inferSchema=True)


dataset_unprocess1=data

airlines=data.select('AIRLINE').distinct().rdd.map(lambda row:row['AIRLINE']).collect()

for airline in airlines:
    print "begin"
    dataset_unprocess2=dataset_unprocess1.filter(dataset_unprocess1['AIRLINE']==airline)
    dataset_unprocess2=dataset_unprocess2.filter(dataset_unprocess2['DEPARTURE_DELAY']<60)

    dataset=dataset_unprocess2

    udf = UserDefinedFunction(lambda x: x*1.0, DoubleType())
    new_data=dataset.select('*',udf(dataset['DEPARTURE_DELAY']).alias('double_labels'))
    dataset=new_data.drop('DEPARTURE_DELAY')
    dataset=dataset.withColumnRenamed('double_labels','DEPARTURE_DELAY')

    categoricalColumns = ['ORIGIN_AIRPORT']  # to add
    numericCols = ['NEW_SCHEDULED_DEPARTURE']  # to add

    cols=dataset.columns

    stages = []
    feature_names=[]
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol=categoricalCol,
            outputCol=categoricalCol+"Index")
        encoder = OneHotEncoder(inputCol=categoricalCol+"Index",
            outputCol=categoricalCol+"classVec")
        stages += [stringIndexer, encoder]

    assemblerInputs = map(lambda c: c + "classVec", categoricalColumns) + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="Features")
    stages += [assembler]
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(dataset)
    dataset_transformed = pipelineModel.transform(dataset)

    pipelineModel.write().overwrite().save(str(airline)+'_pipeline')

    selectedcols = ['DEPARTURE_DELAY', "features"]
    dataset_transformed = dataset_transformed .select(selectedcols)
    dataset_transformed=dataset_transformed.select('*').withColumnRenamed('DEPARTURE_DELAY','label')

    trainingData=dataset_transformed

    #linear regression
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.5,labelCol="label", featuresCol="features")
    model = lr.fit(trainingData)
    model.write().overwrite().save(str(airline)+'_model')
    print 'done'

