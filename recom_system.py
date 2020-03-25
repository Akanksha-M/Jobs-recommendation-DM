

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:40:29 2020

@author: Akanksha
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



df = pd.read_csv('Jobs_data.csv')

#select features for recommendation
features = ['Job_Title','Location','Job_Salary']

#combine these new features into a column in DF
def combine_features(row):
    return row['Job_Title']+""+row['Location']+""+row['Job_Salary']
for feature in features:
    df[feature] = df[feature].fillna('') #filling all NaNs with blank string

 #applying combined_features() method over each rows of dataframe and storing the combined string in "combined_features" column
df["combined_features"] = df.apply(combine_features,axis=1)


df.iloc[0].combined_features

#create a count matrix of this new combined data

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

print(count_matrix.toarray())

#find the cosine similarity

cosine_sim = cosine_similarity(count_matrix)
print(cosine_sim)


#get index of job from req
def get_job_from_index(Index):
	return df[df.Index == Index]["Company_Name"].values[0],df[df.Index == Index]["Job_Title"].values[0],df[df.Index == Index]["Location"].values[0],df[df.Index == Index]["Job_Salary"].values[0]
def get_index_from_job(Job_Title,Location,Job_Salary):
	return df[df.Job_Title == Job_Title]["Index"].values[0]

#user requirements
req_title = "Software Development Engineer"
req_location = "Hyderabad"
req_salary = "Rs 150000"

req_index = get_index_from_job(req_title,req_location,req_salary)
#accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it
similar_jobs = list(enumerate(cosine_sim[req_index])) 

# to get a list of all the similar movies in descending order of similarity score
sorted_similar_jobs = sorted(similar_jobs,key=lambda x:x[1],reverse=True)

#printing the list of similar movies
i=0
#print("Top 5 similar jobs to "+req_title+" are:\n")
for job in sorted_similar_jobs:
    print(get_job_from_index(job[0]))
    i=i+1
    if i>5:
        break



