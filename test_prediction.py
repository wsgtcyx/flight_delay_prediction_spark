
# coding: utf-8

from pyspark.ml.regression import LinearRegression,LinearRegressionModel
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.ml.param import Param, Params
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import numpy as np
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, IntegerType
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pickle

models={}

def init_model():
    all_airlines = ['AA', 'AS','B6',"F9","DL", 'EV', 'HA', 'MQ', 'NK', 'OO', "UA", "US", "VX", "WN"]
    for airline in all_airlines:
        pipeline = PipelineModel.load('model/'+str(airline) + "_pipeline")
        models[str(airline) + "_pipeline"] = pipeline
        model = LinearRegressionModel.load('model/'+str(airline) + "_model")
        models[str(airline) + "_model"] = model
    airport_information=pickle.load(open("model/airport_information.pkl",'r'))
    models["airport"]=airport_information
    print "load finish"

def prediction(input):
    input=input.split(",")
    hours = (int)(input[1])
    minutes = (int)(input[2])
    times = hours * 60 + minutes
    airport = input[0]
    result={}
    for airline in all_airlines:
        schema = StructType([
            StructField("ORIGIN_AIRPORT", StringType(), nullable=False),
            StructField("AIRLINE", StringType(), nullable=True),
            StructField("DEPARTURE_DELAY", DoubleType(), nullable=True),
            StructField("NEW_SCHEDULED_DEPARTURE", IntegerType(), nullable=True)])
        data = []
        for minute in range(-5, 5, 1):
            data.append((airport, airline, 0.0, times + minute))
        df = spark.createDataFrame(data, schema)
        pipeline = models[str(airline + "_pipeline")]
        if airport not in models["airport"][airline]:
            continue
        data_transformed = pipeline.transform(df)

        data_transformed = data_transformed.withColumnRenamed('Features', 'features')
        selectedcols = ['DEPARTURE_DELAY', "features"]
        dataset_transformed = data_transformed.select(selectedcols)
        dataset_transformed = dataset_transformed.select('*').withColumnRenamed('DEPARTURE_DELAY', 'label')

        model=models[str(airline+"_model")]
        temp_result = model.transform(dataset_transformed).select('prediction').rdd.map(
                lambda element: element['prediction']).collect()
        result[airline]=np.array(temp_result).mean()
        print airline,result[airline]

    return result

if __name__ == "__main__":

    sc = SparkContext("local",'app')
    spark = SparkSession.builder.appName('name').config('spark.sql.shuffle.partitions',10).getOrCreate()

    all_airlines=['AA','AS',"B6","F9","DL",'EV','HA','MQ','NK','OO',"UA","US","VX","WN"]

    init_model()

    all_airport=[]
    for airline in all_airlines:
        all_airport.extend(models['airport'][airline])
    all_airport=set(all_airport)
    input="BNA,4,20"  #origin_airport+hours+minutes

    for airport in all_airport:
        input=airport+","+'21'+",20"
        print input
        result=prediction(input)
        #lines=[origin_airport, airline, schedule_departure_hout,schedule_minute]
        print(sorted(result.items(),cmp=lambda x, y: cmp(x[1], y[1])))


