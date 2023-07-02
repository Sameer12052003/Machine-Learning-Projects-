# %% [markdown]
# Libraries 

# %%
import numpy as np
import pandas as pd
import ast



# %%
credits_df = pd.read_csv('D:\IITM academics and courses\IITM courses\Internship ke khoj\credits.csv')
movies_df = pd.read_csv("D:\IITM academics and courses\IITM courses\Internship ke khoj\movies.csv")

# %%
movies_df

# %%
credits_df

# %%
movies_df.head()

# %%
movies_df.tail()

# %%
movies_df = movies_df.merge(credits_df, on = 'title')

# %%
print(movies_df.shape)


# %%

movies_df.info()

# %% [markdown]
# Selecting columns on which we are working

# %%
movies_df = movies_df[['movie_id','title','overview','genres','keywords','cast','crew']]
movies_df.head()

# %% [markdown]
# Finding number of missing values

# %%
movies_df.isnull().sum()

# %% [markdown]
# Removing missing values using dropna method

# %%
movies_df.dropna(inplace= True)
movies_df.duplicated()

# %%
movies_df.iloc[0].genres

# %%
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# %%
movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)


# %%
movies_df.head()

# %%
movies_df.iloc[0].cast
movies_df['cast'] = movies_df['cast'].apply(convert)
movies_df.head()

# %%
def fetch_director(obj):
    L= []
    for i in ast.literal_eval(obj):
        if i['job']== 'Director':
            L.append(i['name'])
    return L

# %%
movies_df['crew'] = movies_df['crew'].apply(fetch_director)


# %%
movies_df.head()

# %% [markdown]
# Splitting the overview into commas. Separating the words in overview 

# %%
movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())
movies_df.head()


# %% [markdown]
# Removing all the white spaces

# %%
movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ","") for i in x])


# %%
movies_df.head()

# %%
movies_df['tags'] = movies_df['overview']+ movies_df['genres']+movies_df['crew']+movies_df['cast']+movies_df['keywords']

# %%
movies_df.head()

# %% [markdown]
# Creating new dataframe

# %%
new_df = movies_df[['title','movie_id','tags']]


# %%
new_df

# %%
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))



# %%
new_df['tags'] = new_df['tags'].apply(lambda X: X.lower())

# %%
new_df.head()

# %%
new_df['tags'][0]

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words= 'english')
cv.fit_transform(new_df['tags']).toarray().shape

# %%
vectors = cv.fit_transform(new_df['tags']).toarray()

# %%
vectors[0]

# %%
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# %%
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# %%
new_df['tags'] = new_df['tags'].apply(stem)


# %%
new_df['tags'] = new_df['tags'].apply(stem)


# %%
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors)

# %%
similarity = cosine_similarity(vectors)
similarity[0]

# %%
sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x:x[1])[1:6]


# %%
def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x : x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        
    

# %%
recommend('Avatar')

# %% [markdown]
# 


