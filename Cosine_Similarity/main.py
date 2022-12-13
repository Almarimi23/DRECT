import pandas as pd  
import numpy as np  
import re
import nltk
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from sklearn import preprocessing
from nltk.stem import SnowballStemmer
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from utils import replace_fn, utils_preprocess_desc, utils_preprocess_skills

# Read the full dataset
full_data = pd.read_csv('full_data.csv')

# Create a unique id
full_data['ID'] = full_data.index
print('\nOriginal full data:\n', full_data.head())
df = full_data.copy()
# Drop missing values 
df.dropna(inplace=True)

# Create a dataframe for description & skills
df_desc = df[['ID','member description','challenge description']]
df_skills = df[['ID','member_skills','task_skills']]

############ Descriptions - Cosine similarity score ##################
# Remove single word memebr descriptions
df_desc['member description'] = [desc if len(desc.split())>1 else np.nan for desc in df_desc['member description'].tolist()]
df_desc = df_desc.dropna().reset_index(drop=True)

# Take a copy of the data
df_cleaned1 = df_desc.copy()

# Get stopwords
lst_stopwords = nltk.corpus.stopwords.words("english")

# Apply rename_fn
df_cleaned1 = replace_fn(df_cleaned1)
# Apply utils_preprocess_text
df_cleaned1["member description"] = df_cleaned1["member description"].apply(lambda x:
                                          utils_preprocess_desc(x, flg_stemm=False, flg_lemm=True,
                                                                lst_stopwords=lst_stopwords))
df_cleaned1["challenge description"] = df_cleaned1["challenge description"].apply(lambda x:
                                          utils_preprocess_desc(x, flg_stemm=False, flg_lemm=True,
                                                                lst_stopwords=lst_stopwords))
print('\nCleaned descriptions data:\n', df_cleaned1.head())
member_descriptions = df_cleaned1["member description"].str.split()
task_descriptions =df_cleaned1["challenge description"].str.split()

# Create a train corpus
train_corpus=[]
train_corpus.extend([TaggedDocument(doc, [i]) for i, doc in enumerate(member_descriptions)])
train_corpus.extend([TaggedDocument(doc, [i]) for i, doc in enumerate(task_descriptions)])

# Build a Doc2Vec model & a Vocabulary
model = Doc2Vec(dm = 1, min_count=1, window=3, vector_size=500,  epochs=10, workers=16)
model.build_vocab(train_corpus)

# Train the model 
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
# Save model
fname = './models/descriptions/doc2vec_model_descriptions'
model.save(fname)
# Load the model 
model = Doc2Vec.load(fname)

scores_desc = []
for i in range(len(df_cleaned1)):
    try:
        score = model.wv.n_similarity(member_descriptions[i],task_descriptions[i])
        scores_desc.append(score)
    except:
        scores_desc.append(np.nan)

df_desc['Cosine_similarity_descriptions'] = scores_desc
df_desc = df_desc[df_desc['Cosine_similarity_descriptions'].notnull()].reset_index(drop=True)
df_desc['Cosine_similarity_descriptions'] = df_desc['Cosine_similarity_descriptions'].astype(float)
df_desc = df_desc[['ID','Cosine_similarity_descriptions']]

############ Skills - Cosine similarity score ##################

# Drop missing values
df_skills = df_skills.dropna().reset_index(drop=True)

df_skills = df_skills[df_skills['member_skills'].notnull()].reset_index(drop=True)
# Replace React.js by ReactJS
df_skills['member_skills'] = [df_skills['member_skills'].loc[i].replace('React.js','ReactJS') for i in range(len(df_skills))]
# Take a copy of the data
df_cleaned2 = df_skills.copy()

# Apply utils_preprocess_text
df_cleaned2["member_skills"] = df_cleaned2["member_skills"].apply(lambda x:
                                          utils_preprocess_skills(x))
df_cleaned2["task_skills"] = df_cleaned2["task_skills"].apply(lambda x:
                                          utils_preprocess_skills(x))
print('\nCleaned skills data:\n', df_cleaned2.head())
member_skills = df_cleaned2["member_skills"].str.split()
task_skills =df_cleaned2["task_skills"].str.split()

# Create a train corpus
skills_corpus=[]
skills_corpus.extend([TaggedDocument(doc, [i]) for i, doc in enumerate(member_skills)])
skills_corpus.extend([TaggedDocument(doc, [i]) for i, doc in enumerate(task_skills)])
# Build a Doc2Vec model & a Vocabulary
model_skills = Doc2Vec(dm = 1, min_count=1, window=3, vector_size=500,  epochs=10, workers=16)
model_skills.build_vocab(skills_corpus)
# Train the model
model_skills.train(skills_corpus, total_examples=model_skills.corpus_count, epochs=model_skills.epochs)

# Save model
fname = "./models/skills/doc2vec_model_skills"
model_skills.save(fname)
# Load the model 
model_skills = Doc2Vec.load(fname)

scores = []
for i in range(len(df_cleaned2)):
    try:
        score = model_skills.wv.n_similarity(member_skills[i],task_skills[i])
        scores.append(score)
    except:
        scores.append(np.nan)

df_skills['Cosine_similarity_skills'] = scores
df_skills = df_skills[df_skills['Cosine_similarity_skills'].notnull()].reset_index(drop=True)
df_skills['Cosine_similarity_skills'] = df_skills['Cosine_similarity_skills'].astype(float)
df_skills = df_skills[['ID','Cosine_similarity_skills']]

# Join full dataset & Cosine similarity scores
# Join Cosine_similarity_descriptions
join_df = full_data.merge(df_desc, on='ID', how='outer')
# Join Cosine_similarity_skills
join_df = join_df.merge(df_skills, on='ID', how='outer')
join_df = join_df.drop('ID',axis=1)
join_df = join_df[join_df['Cosine_similarity_descriptions'].notnull()]
join_df = join_df[join_df['Cosine_similarity_skills'].notnull()]
join_df = join_df.reset_index(drop=True)
print('\nFull data with Cosine similarity score: \n', join_df)
# Save data
join_df.to_csv('full_data_updated.csv', index=False)



