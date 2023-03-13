#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import os
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('full_data_updated.csv')
df = df.sort_values('challengeId').reset_index(drop=True)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


# Check NaN values in task_recency(in days) column
df['task_recency (in days)'].isnull().sum()


# This means there are 2464 mebers who have less than 15 tasks in the data.

# In[7]:


# Unique handles & challenge_ids
handles = df['handle'].unique()
challenge_ids = df['challengeId'].unique()


# In[8]:


# Extract developers & challenges to calculate the count of interactions
dev1 = []
dev2 = []
ids = []
for i in challenge_ids:
  challenge_df = df[df['challengeId']==i]
  unique_handles = challenge_df['handle'].unique()
  for handle1 in unique_handles:
    for handle2 in unique_handles:
      if handle1!=handle2:
        ids.append(i)
        dev1.append(handle1)
        dev2.append(handle2)


# In[9]:


df1 = pd.DataFrame(ids, columns=['challengeId'])
df1['Developer1'] = dev1
df1['Developer2'] = dev2


# In[10]:


df1.head()


# In[11]:


df1.shape


# In[12]:


# Function to drop duplicates
def drop_duplicates_devs(df):
    df2 = df.copy()
    df2['devs'] = df2['Developer1']+ ' '+ df2['Developer2']
    df2['devs'] = df2['devs'].apply(lambda x: ' '.join(sorted(x.split())))
    # drop duplicates 
    new_df = []
    for id in df2['challengeId'].unique():
      id_df = df2[df2['challengeId']==id]
      id_df.drop_duplicates(subset=['devs'], inplace=True)
      new_df.append(id_df)
    new_df = pd.concat(new_df)
    new_df = new_df.reset_index(drop=True)
    return new_df


# In[13]:


# Apply above function 
final_df = drop_duplicates_devs(df1)


# In[14]:


final_df.head()


# In[15]:


# Calculate interaction count
interaction_df = final_df.groupby('devs')['challengeId'].count().reset_index(drop=False).rename(columns={'challengeId':'Interaction_count'}).sort_values('Interaction_count', ascending=True).reset_index(drop=True)
interaction_df.head()


# In[16]:


interaction_df['Developer1'] = interaction_df['devs'].apply(lambda x:x.split()[0])
interaction_df['Developer2'] = interaction_df['devs'].apply(lambda x:x.split()[1])
# Add challengeId to intercation_df 
interaction_df = interaction_df.merge(final_df[['challengeId', 'devs']], on='devs', how='outer')


# In[17]:


interaction_df.head()


# In[18]:


# Save the interaction_df in csv 
# interaction_df.to_csv('developer_interaction_data.csv', index=False)


# ### Weighted Social Networks

# In this we add weight to the network, each edge has a weight signifying the the strength of collaboration among them as a count of the number of interactions in terms of the frequency of the number of challenges participated by the developers..

# In[19]:


# create an empty undirected graph
G_weighted = nx.Graph()

# add edge to the graph
dev1_list = interaction_df['Developer1'].tolist()
dev2_list = interaction_df['Developer2'].tolist()
weight_list = interaction_df['Interaction_count'].tolist()

for d1, d2, w in zip(dev1_list, dev2_list, weight_list):
  G_weighted.add_edge(d1, d2,   weight=w)
print(nx.info(G_weighted))


# In[20]:


# Sum of the weights on the edges in graph
sum_weights_edges = G_weighted.size(weight='weight')
print(sum_weights_edges)


# In[21]:


# Graph
Vp = G_weighted.number_of_nodes()    # Total number of nodes/developers 
Ep =G_weighted.number_of_edges()    # Total number of edges
print(f'Total number of nodes : {Vp}')
print(f'Total number of edges : {Ep}')


# Now, let's compute following 2 measures for each subgraph:      
# 1. Sub-graph connectivity 
# 2. Sum of the weights on the edges 

# ### Subgraphs

# In[22]:


# extract subgraphs
sub_graphs = [G_weighted.subgraph(c) for c in nx.connected_components(G_weighted)]
print(f'Number of Subgraphs : {len(sub_graphs)}')

print('Nodes in each Subgraph : ')
for i, sg in enumerate(sub_graphs):
  print(f'Subgraph : {i+1}')
  print(sg.nodes)


# #### Plot Subgraph 1

# In[23]:


pos = nx.spring_layout(G_weighted)  #setting the positions with respect to G
SG1 = sub_graphs[0]  

plt.figure()
nx.draw_networkx(SG1, pos=pos)

print('Sum of the weights on the edges in Subgraph 1 :')
sum_weights_edges1 = SG1.size(weight='weight')
print(sum_weights_edges1)


#  **Sub-graph connectivity**   
# We need the total number of edges & the total number of nodes in the sub graph to find the sub-graph connectivity.

# In[24]:


Vp1 = SG1.number_of_nodes()    # Total number of nodes/developers 
Ep1 =SG1.number_of_edges()    # Total number of edges
print(f'Total number of nodes : {Vp1}')
print(f'Total number of edges : {Ep1}')
denominator = abs(Vp1)*(abs(Vp1)-1)/2 
coefficient_value1 = abs(Ep1)/denominator
print(f'Sub-graph connectivity value : {coefficient_value1}')


# In[25]:


# Collaborative Preference CP score for sub graph 1
CP_score1 = coefficient_value1*sum_weights_edges1
print(CP_score1)


# In[26]:


# Groupby Developer 1 to get the number of interactions with other other developers 
df_intr = final_df.groupby(['Developer1','challengeId'])['devs'].count().reset_index(drop=False)
df_intr = df_intr.rename(columns={'Developer1':'handle','devs':'number_interactions'})
df_intr.head()


# In[27]:


# Check weights(Interaction_count) on each edge for Sub graph 1
sg1_df = df_intr[df_intr['handle'].isin(SG1.nodes())]
# Add coefficient_value & sum_weights_edges
sg1_df['sum_weights_edges'] = sum_weights_edges1
sg1_df['Coefficient_value'] = coefficient_value1
sg1_df['CP_score'] = sg1_df['Coefficient_value']*sg1_df['number_interactions']
sg1_df


# In[28]:


df_final = sg1_df.copy()
df_final = df_final.reset_index(drop=True)
df_final.drop('Coefficient_value', axis=1, inplace=True)
df_final


# In[29]:


# Join df_final & df (full data)
join_df = df.merge(df_final, on=['challengeId','handle'], how='outer')
join_df = join_df.sort_values('challengeId').reset_index(drop=True)
# Drop duplicates
join_df = join_df.drop_duplicates().reset_index(drop=True)
join_df.head()


# In[30]:


join_df.isnull().sum()


# In[31]:


join_df.shape


# In[32]:


join_df.to_csv('full_dataset_updated_with_NANs.csv', index=False)


# In[33]:


# Drop null values
join_df1 = join_df.dropna().reset_index(drop=True)
join_df1.head()


# In[34]:


join_df1.head(500)


# In[35]:


join_df1.shape


# In[36]:


# Save the full data in csv 
join_df1.to_csv('full_data_updated_with_no_NANs.csv', index=False)


# In[ ]:




