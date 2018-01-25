#Recommender Systems
#TWO Types: 
#1,Content-Based 
#2,Collaborative Filtering(CF):Memory Based CF and Model-Based CF


#Recommender Systems

#Import Libraries
import numpy as np
import pandas as pd

#Get the Data
import os
os.getcwd()
os.chdir("C:\\Users\\abhishek.b.jaiswal\\Desktop\\DataScience\\sem 2\\BD 3\\codes")
os.getcwd()
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
print(df.head())

movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()

#We can merge them together:
df = pd.merge(df,movie_titles,on='item_id')
print(df.head())

#EDA
#Visualization Imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

#Let's create a ratings dataframe with average rating and number of ratings:
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
df.groupby('title')['rating'].count().sort_values(ascending=False).head()

#Data Frame
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

#Plot
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)





#Recommending Similar Movies
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()
#Most rated movie:
ratings.sort_values('num of ratings',ascending=False).head(10)

#Let's choose two movies: starwars, a sci-fi movie. And Liar Liar, a comedy.
ratings.head()

#Now let's grab the user ratings for those two movies:
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()

#We can then use corrwith() method to get correlations between two pandas series:
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

#Let's clean this by removing NaN values and using a DataFrame instead of a series:
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()
corr_starwars.sort_values('Correlation',ascending=False).head(10)
#Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()
#Now sort the values and notice how the titles make a lot more sense:
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()

#Now the same for the comedy Liar Liar:
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()

