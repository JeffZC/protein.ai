import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
from string import digits
# Import data


result = ""

def extract(df):
    temp = df[" q3"].to_json()
    temp = temp.replace("\":\"", " ")
    temp = temp.replace("\",\"", "")
    table = temp.maketrans('', '', digits)
    temp = temp.translate(table)
    return temp

count = 0
path = "data/pre_processing/*/*/*.csv"
for fname in glob.glob(path):
    data = pd.read_csv(fname)
    result = result + extract(data)


f = open("train_ds", "w")
f.write(result)
f.close()


f = open("token.numpy", "w")

df = pd.DataFrame([], columns=list('AB'))

def process(tar):
    count = result.count(tar)
    f = open("token.numpy", "a")
    f.write(tar)
    f.write(str(count))
    return count



for i in "CHE":
    for j in "CHE":
        target = " " + i + " " + j
        count = process(target)
        df = df.append(pd.DataFrame([[target, count]], columns=list('AB')))


for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            target = " " + i + " " + j + " " + k
            count = process(target)
            df = df.append(pd.DataFrame([[target, count]], columns=list('AB')))



for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            for l in "CHE":
                target = " " + i + " " + j + " " + k + " " + l
                count = process(target)
                df = df.append(pd.DataFrame([[target, count]], columns=list('AB')))



for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            for l in "CHE":
                for m in "CHE":
                    target = " " + i + " " + j + " " + k + " " + l + " " + m
                    count = process(target)
                    df = df.append(pd.DataFrame([[target, count]], columns=list('AB')))



for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            for l in "CHE":
                for m in "CHE":
                    for n in "CHE":
                        target = " " + i + " " + j + " " + k + " " + l + " " + m + " " + n
                        count = process(target)
                        df = df.append(pd.DataFrame([[target, count]], columns=list('AB')))



for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            for l in "CHE":
                for m in "CHE":
                    for n in "CHE":
                        for o in "CHE":
                            target = " " + i + " " + j + " " + k + " " + l + " " + m + " " + n + " " + o
                            count = process(target)
                            df = df.append(pd.DataFrame([[target, count]], columns=list('AB')))



for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            for l in "CHE":
                for m in "CHE":
                    for n in "CHE":
                        for o in "CHE":
                            for p in "CHE":
                                target = " " + i + " " + j + " " + k + " " + l + " " + m + " " + n + " " + o + " " + p
                                count = process(target)
                                df = df.append(pd.DataFrame([[target, count]], columns=list('AB')))


f.close()
df.sort_values(by=['B'])
df.to_csv('./token.csv')

#print(data.sum())
#vectorizer = TfidfVectorizer(input = result, ngram_range = (2,6))

#X = vectorizer.fit_transform(corpus)
#func = vectorizer.build_tokenizer()
#print(vectorizer)
