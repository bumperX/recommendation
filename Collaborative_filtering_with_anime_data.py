#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
import operator
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


anime = pd.read_csv('./input/anime.csv')
rating = pd.read_csv('./input/rating.csv')


# In[3]:


# avoid to distort the average of rating
rating.rating.replace({-1: np.nan}, inplace=True)
rating.head()


# In[4]:


# Focus on TV category
anime_TV = anime[anime.type == 'TV']
anime_TV.head()


# In[5]:


# Merge tables
merged = rating.merge(anime_TV, on='anime_id', suffixes= ['_user', ''])
merged.rename(columns={'rating_user':'user_rating'}, inplace=True)


# In[6]:


# Consider the computer memory so that only take the first 10000 users
merged = merged[['user_id', 'name', 'user_rating']]
merged_sub = merged[merged.user_id<10000]
merged_sub.head()


# In[7]:


piv = merged_sub.pivot_table(index=['user_id'], values=['user_rating'], columns=['name'])


# In[96]:


piv.columns = [j for i,j in piv.columns]
print(piv.shape)
piv.head()


# In[97]:


# Normalize user_rating
piv_norm = piv.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
piv_norm.fillna(0, inplace=True)
piv_norm = piv_norm.T
piv_norm = piv_norm.loc[:, (piv_norm!=0).any(axis=0)]


# In[98]:


# Prepare for the following functions and convert data in a sparse matrix format 
piv_sparse = sp.sparse.csr_matrix(piv_norm.values)


# In[99]:


# Compute cosine similarity values between each user/user array pair and item/item array pair
item_similarity = cosine_similarity(piv_sparse)
user_similarity = cosine_similarity(piv_sparse.T)


# In[100]:


item_sim_df = pd.DataFrame(item_similarity, index=piv_norm.index, columns=piv_norm.index)
user_sim_df = pd.DataFrame(user_similarity, index=piv_norm.columns, columns=piv_norm.columns)


# In[108]:


# Return the top 10 TVs with the highest similarity value
def top_animes(anime_name):
    count = 1
    print("Similar shows to {} include:\n".format(anime_name))
    for item in item_sim_df.sort_values(by=[anime_name], ascending=False).index[1:11]:
        print("No. {}: {}".format(count, item))
        count += 1


# In[109]:


top_animes('Fate/Zero')


# In[110]:


# Return the top 5 users with the highest similarity value
def top_users(user):
    if user not in piv_norm.columns:
        return("No data available on user {}".format(user))
    print("Most Similar Users:\n")
    sim_values = user_sim_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:11]
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:11]
    zipped = zip(sim_users, sim_values)
    for user, sim in zipped:
        print("User #{0}, Similarity value: {1:.2f}".format(user, sim))


# In[111]:


top_users(3)


# In[114]:


# Construct a list of lists containing the highest rated TVs per similar user 
# and return the name of TVs along with the frequency it appears in the list
def similar_user_recs(user):
    if user not in piv_norm.columns:
        return("No data available on user {}".format(user))
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:11]
    best = []
    most_common = {}
    
    for i in sim_users:
        max_score = piv_norm.loc[:, i].max()
        best.append(piv_norm[piv_norm.loc[:, i]==max_score].index.tolist())

    for i in range(len(best)):
        for j in best[i]:
            if j in most_common:
                most_common[j] += 1
            else:
                most_common[j] = 1
    sorted_list = sorted(most_common.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_list[:5]


# In[115]:


similar_user_recs(3)


# In[130]:


# Calculate the weighted average of similar users to determine a potential rating for an input user and show
def predict_rating(anime_name, user):
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:1000]
    user_values = user_sim_df.sort_values(by=user, ascending=False).loc[:, user].tolist()[1:1000]
    rating_list = []
    weight_list = []
    for idx, usr in enumerate(sim_users):
        rating = piv.loc[usr, anime_name]
        similarity = user_values[idx]
        if np.isnan(rating):
            continue
        elif not np.isnan(rating):
            rating_list.append(rating*similarity)
            weight_list.append(similarity)

    return sum(rating_list) / sum(weight_list)


# In[138]:


predict_rating('Fate/Zero', 3)


# In[ ]:




