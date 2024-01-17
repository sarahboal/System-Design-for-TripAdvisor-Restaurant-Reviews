import warnings
warnings.filterwarnings("ignore")


import kaggle
import pandas as pd
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.sql import Row
from sparknlp.pretrained import PretrainedPipeline
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import pdb
import datetime
import os
def download_data(fraction=1,path_download = './data/'):

    PATH_file = f'{path_download}/trip_advisor_restaurents_10k_-_trip_rest_neywork_1.csv'

    if 'trip_advisor_restaurents_10k_-_trip_rest_neywork_1.csv' in os.listdir(path_download):
        df = pd.read_csv(PATH_file)
        df=df.sample(frac=fraction,random_state=42)
        print(f'Dimension of the data {df.shape}')
        return df
    
    # Make sure to download the kaggle token from your kaggle profile -> Settings - > create new token
    # then add that to your path : /home/USER_NAME/.kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('rayhan32/trip-advisor-newyork-city-restaurants-dataset-10k/',\
                                       path=path_download, unzip=True)
    
    df = pd.read_csv(PATH_file)
    df=df.sample(frac=fraction,random_state=42)
    print(f'Dimension of the data {df.shape}')
    return df

def setup_SPARK_NLP():
    lemmas = requests.get('https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/lemma-corpus-small/lemmas_small.txt').text
    with open('lemmas_small.txt','w') as f:
        f.write(lemmas)
    sent = requests.get('https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sentiment-corpus/default-sentiment-dict.txt').text
    with open('default-sentiment-dict.txt','w') as f:
        f.write(sent)
    return None

def spark_nlp(df,spark,REVIEW_COLUMN='Reveiw Comment'):
    if type(df)==pd.DataFrame:
        spark_df = spark.createDataFrame(df)
    else:
        spark_df =df
    # Step 1: Transforms raw texts to `document` annotation
    document_assembler = (
        DocumentAssembler()
        .setInputCol(REVIEW_COLUMN)
        .setOutputCol("document")
    )

    # Step 2: Sentence Detection
    sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

    # Step 3: Tokenization
    tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

    # Step 4: Lemmatization
    lemmatizer= Lemmatizer().setInputCols("token").setOutputCol("lemma").\
                setDictionary("lemmas_small.txt", key_delimiter="->", value_delimiter="\t")

    # Step 5: Sentiment Detection
    sentiment_detector= (
        SentimentDetector()
        .setInputCols(["lemma", "sentence"])
        .setOutputCol("sentiment_score")
        .setDictionary("default-sentiment-dict.txt", ",")
    )

    # Step 6: Finisher
    finisher= (
        Finisher()
        .setInputCols(["sentiment_score"]).setOutputCols("sentiment")
    )

    # Define the pipeline
    pipeline = Pipeline(
        stages=[
            document_assembler,
            sentence_detector, 
            tokenizer, 
            lemmatizer, 
            sentiment_detector, 
            finisher
        ]
    )
    result = pipeline.fit(spark_df).transform(spark_df)
    result =result.withColumn("SPARK sentiment", F.concat_ws(",", F.col("sentiment")))
    result = result.drop('sentiment')
    return result

def viviken(df,spark,REVIEW_COLUMN='Reveiw Comment'):
    #pdb.set_trace()
    if type(df)==pd.DataFrame:
        spark_df = spark.createDataFrame(df)
    else:
        spark_df =df
    # Step 1: Transforms raw texts to `document` annotation
    document_assembler = (
        DocumentAssembler()
        .setInputCol(REVIEW_COLUMN)
        .setOutputCol("document")
    )

    # Step 2: Sentence Detection
    sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

    # Step 3: Tokenization
    tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

    # Step 4: Normalization
    normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalizer")

    # Step 5: Viviken
    viviken = ViveknSentimentModel.pretrained().setInputCols(["document", "normalizer"]).setOutputCol("viviken") 

    # Step 6: Finisher
    finisher= (
        Finisher()
        .setInputCols(["viviken"]).setOutputCols("sentiment_viviken")
    )

    # Define the pipeline
    pipeline = Pipeline(
        stages=[
            document_assembler,
            sentence_detector, 
            tokenizer, 
            normalizer, 
            viviken, 
            finisher
        ]
    )
    result_viviken = pipeline.fit(spark_df).transform(spark_df)
    result_viviken = result_viviken.withColumn("Viviken", F.concat_ws(",", F.col("sentiment_viviken")))
    result_viviken = result_viviken.drop('sentiment_viviken')
    return result_viviken

def vader_func(row,REVIEW_COLUMN='Reveiw Comment'):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(row[REVIEW_COLUMN])
    if vs['compound']>0:
        sentiment = 'positive'#print(row['Reveiw Comment'],vs)
    elif vs['compound']<0 :
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    # add new item to row
    temp = row.asDict()
    temp["VADER sentiment"] = sentiment    
    return Row(**temp)    
    
def vader_nlp(df,spark,REVIEW_COLUMN='Reveiw Comment'):
    # sentiment analysis based on VADER NLP: https://vadersentiment.readthedocs.io/en/latest/index.html
    # create regular Spark session
    if type(df)==pd.DataFrame:
        spark_df = spark.createDataFrame(df)
    else:
        spark_df =df

    rdd = spark.sparkContext.parallelize(spark_df.collect())
    results = rdd.map(vader_func).collect()
    cols =df.columns
    cols.append("VADER sentiment")
    results = spark.createDataFrame(results, cols)
    return results
    
    
def main(fraction=1):
    # Download the data
    df= download_data(fraction=fraction)
    # setup SPARK NLP libs
    setup_SPARK_NLP()
    #pdb.set_trace()
    # Start Spark Session
    spark = sparknlp.start()
    # perform spark nlp
    start_spark = datetime.datetime.now()
    result = spark_nlp(df,spark,REVIEW_COLUMN='Reveiw Comment')
    end_spark = datetime.datetime.now()
    # perform VADER NLP ()
    start_vader = datetime.datetime.now()
    result_with_vader = vader_nlp(result,spark,REVIEW_COLUMN='Reveiw Comment')
    end_vader= datetime.datetime.now()
    # perform viviken
    start_viviken = datetime.datetime.now()
    result_viviken = viviken(df,spark,REVIEW_COLUMN='Reveiw Comment')
    end_viviken = datetime.datetime.now()
    vk = end_viviken - start_viviken
    sn = end_spark-start_spark
    vn = end_vader - start_vader
    print('First 20 Rows:')
    print(result_with_vader.show())
    print('---------------------------------------------------')
    print(f'SPARK NLP total elapsed time (seconds) {sn.total_seconds()}')
    print(f'VADER NLP total elapsed time (seconds) {vn.total_seconds()}')
    print(f'VIVIKEN MODEL total elapsed time (seconds) {vk.total_seconds()}')
    return sn.total_seconds(),vn.total_seconds(),vk.total_seconds()

if __name__=='__main__':
    fractions = [0.1,0.25,0.5,0.75,1]
    run_times = pd.DataFrame(columns=['SPARK','VADER','VIVIKEN'],index=fractions)
    for fraction in fractions:
        sn,vn,vk = main(fraction)
        run_times.loc[fraction,'SPARK']=sn
        run_times.loc[fraction,'VADER']=vn
        run_times.loc[fraction,'VIVIKEN']=vk
    print(run_times)

    

