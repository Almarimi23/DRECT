from google.colab import drive 
drive.mount('/content/drive')

import networkx as nx
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings 
warnings.filterwarnings('ignore')

os.chdir('/content/drive/MyDrive/Nuri Almarimi')

df = pd.read_csv('/content/drive/MyDrive/Nuri Almarimi/full_data_updated.csv')

df.head()

df.shape

df.columns

# Unique handles & challenge_ids
handles = df['handle'].unique()
challenge_ids = df['challengeId'].unique()

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

df1 = pd.DataFrame(ids, columns=['challengeId'])
df1['Developer1'] = dev1
df1['Developer2'] = dev2

df1.head()

df1.shape

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

# Apply above function 
final_df = drop_duplicates_devs(df1)

final_df.head()

# Calculate interaction count
interaction_df = final_df.groupby('devs')['challengeId'].count().reset_index(drop=False).rename(columns={'challengeId':'Interaction_count'}).\
sort_values('Interaction_count', ascending=True).reset_index(drop=True)
interaction_df.head()

interaction_df['Developer1'] = interaction_df['devs'].apply(lambda x:x.split()[0])
interaction_df['Developer2'] = interaction_df['devs'].apply(lambda x:x.split()[1])
# Add challengeId to intercation_df 
interaction_df = interaction_df.merge(final_df[['challengeId', 'devs']], on='devs', how='outer')

interaction_df.head()

# Save the interaction_df in csv 
# interaction_df.to_csv('developer_interaction_data.csv', index=False)

"""### Weighted Social Networks

In this we add weight to the network, each edge has a weight signifying the the strength of collaboration among them as a count of the number of interactions in terms of the frequency of the number of challenges participated by the developers..
"""

# create an empty undirected graph
G_weighted = nx.Graph()

# add edge to the graph
dev1_list = interaction_df['Developer1'].tolist()
dev2_list = interaction_df['Developer2'].tolist()
weight_list = interaction_df['Interaction_count'].tolist()

for d1, d2, w in zip(dev1_list, dev2_list, weight_list):
  G_weighted.add_edge(d1, d2,   weight=w)
print(nx.info(G_weighted))

# Sum of the weights on the edges in graph
sum_weights_edges = G_weighted.size(weight='weight')
print(sum_weights_edges)

# Graph
Vp = G_weighted.number_of_nodes()    # Total number of nodes/developers 
Ep =G_weighted.number_of_edges()    # Total number of edges
print(f'Total number of nodes : {Vp}')
print(f'Total number of edges : {Ep}')

"""Now, let's compute following 2 measures for each subgraph:      
1. Sub-graph connectivity 
2. Sum of the weights on the edges

### Subgraphs
"""

# extract subgraphs
sub_graphs = [G_weighted.subgraph(c) for c in nx.connected_components(G_weighted)]
print(f'Number of Subgraphs : {len(sub_graphs)}')

print('Nodes in each Subgraph : ')
for i, sg in enumerate(sub_graphs):
  print(f'Subgraph : {i+1}')
  print(sg.nodes)

"""#### Plot Subgraph 1"""

pos = nx.spring_layout(G_weighted)  #setting the positions with respect to G
SG1 = sub_graphs[0]  

plt.figure()
nx.draw_networkx(SG1, pos=pos)

print('Sum of the weights on the edges in Subgraph 1 :')
sum_weights_edges1 = SG1.size(weight='weight')
print(sum_weights_edges1)

""" **Sub-graph connectivity**   
We need the total number of edges & the total number of nodes in the sub graph to find the sub-graph connectivity.
"""

Vp1 = SG1.number_of_nodes()    # Total number of nodes/developers 
Ep1 =SG1.number_of_edges()    # Total number of edges
print(f'Total number of nodes : {Vp1}')
print(f'Total number of edges : {Ep1}')
denominator = abs(Vp1)*(abs(Vp1)-1)/2 
coefficient_value1 = abs(Ep1)/denominator
print(f'Sub-graph connectivity value : {coefficient_value1}')

# Collaborative Preference CP score for sub graph 1
CP_score1 = coefficient_value1*sum_weights_edges1
print(CP_score1)

# Check weights(Interaction_count) on each edge for Sub graph 1
sg1_df = interaction_df[interaction_df['Developer1'].isin(SG1.nodes())]
# Add coefficient_value 
sg1_df['Coefficient_value'] = coefficient_value1
sg1_df['CP_score'] = sg1_df['Coefficient_value']*sg1_df['Interaction_count']
sg1_df

"""#### Plot Subgraph 2"""

pos = nx.spring_layout(G_weighted)  #setting the positions with respect to G
SG2 = sub_graphs[1]  

plt.figure()
nx.draw_networkx(SG2, pos=pos)

print('Sum of the weights on the edges in Subgraph 2 :')
sum_weights_edges2 = SG2.size(weight='weight')
print(sum_weights_edges2)

# Sub-graph connectivity
Vp2 = SG2.number_of_nodes()    # Total number of nodes/developers 
Ep2 =SG2.number_of_edges()    # Total number of edges
print(f'Total number of nodes : {Vp2}')
print(f'Total number of edges : {Ep2}')
denominator = abs(Vp2)*(abs(Vp2)-1)/2 
coefficient_value2 = abs(Ep2)/denominator
print(f'Sub-graph connectivity value : {coefficient_value2}')

# Collaborative Preference CP score for sub graph 2
CP_score2 = coefficient_value2*sum_weights_edges2
print(CP_score2)

# Check weights(Interaction_count) on each edge for Sub graph 1
sg2_df = interaction_df[interaction_df['Developer1'].isin(SG2.nodes())]
# Add coefficient_value 
sg2_df['Coefficient_value'] = coefficient_value2
sg2_df['CP_score'] = sg2_df['Coefficient_value']*sg2_df['Interaction_count']
sg2_df

"""#### Plot Subgraph 3"""

pos = nx.spring_layout(G_weighted)  #setting the positions with respect to G
SG3 = sub_graphs[2]  

plt.figure()
nx.draw_networkx(SG3, pos=pos)

print('Sum of the weights on the edges in Subgraph 3 :')
sum_weights_edges3 = SG3.size(weight='weight')
print(sum_weights_edges3)

# Sub-graph connectivity
Vp3 = SG3.number_of_nodes()    # Total number of nodes/developers 
Ep3 =SG3.number_of_edges()    # Total number of edges
print(f'Total number of nodes : {Vp3}')
print(f'Total number of edges : {Ep3}')
denominator = abs(Vp3)*(abs(Vp3)-1)/2 
coefficient_value3 = abs(Ep3)/denominator
print(f'Sub-graph connectivity value : {coefficient_value3}')

# Collaborative Preference CP score for sub graph 3
CP_score3 = coefficient_value3*sum_weights_edges3
print(CP_score3)

# Check weights(Interaction_count) on each edge for Sub graph 3
sg3_df = interaction_df[interaction_df['Developer1'].isin(SG3.nodes())]
# Add coefficient_value 
sg3_df['Coefficient_value'] = coefficient_value3
sg3_df['CP_score'] = sg3_df['Coefficient_value']*sg3_df['Interaction_count']
sg3_df

# Add all subgraphs data 
df_final = pd.concat([sg1_df, sg2_df, sg3_df]) 
df_final = df_final.reset_index(drop=True)
df_final.drop('Coefficient_value', axis=1, inplace=True)
df_final

# Join df_final & df (full data)
join_df = df.merge(df_final, on='challengeId', how='outer')
join_df.head()

join_df.shape

join_df.tail()

join_df[832057:]

# Drop duplicates
join_df.drop_duplicates(inplace=True)

join_df = join_df.reset_index(drop=True)

join_df[832057:]

# Drop null values if needed
#join_df.dropna(inplace=True)

# Save the full data in csv 
join_df.to_csv('/content/drive/MyDrive/Nuri Almarimi/full_data_updated_V1.csv', index=False)

"""### betweenness Centrality"""

pos = nx.spring_layout(G_weighted)
betCent = nx.betweenness_centrality(G_weighted, normalized=True, endpoints=True)
node_color = [20000.0 * G_weighted.degree(v) for v in G_weighted]
node_size =  [v * 10000 for v in betCent.values()]
plt.figure(figsize=(15,15))
nx.draw_networkx(G_weighted, pos=pos, with_labels=False,
                 node_color=node_color,
                 node_size=node_size )
plt.axis('off')

"""### Degree Centrality"""

pos = nx.spring_layout(G_weighted)
degCent = nx.degree_centrality(G_weighted)
node_color = [20000.0 * G_weighted.degree(v) for v in G_weighted]
node_size =  [v * 10000 for v in degCent.values()]
plt.figure(figsize=(15,15))
nx.draw_networkx(G_weighted, pos=pos, with_labels=False,
                 node_color=node_color,
                 node_size=node_size )
plt.axis('off')

"""### Closeness Centrality"""

pos = nx.spring_layout(G_weighted)
cloCent = nx.closeness_centrality(G_weighted)
node_color = [20000.0 * G_weighted.degree(v) for v in G_weighted]
node_size =  [v * 10000 for v in cloCent.values()]
plt.figure(figsize=(15,15))
nx.draw_networkx(G_weighted, pos=pos, with_labels=False,
                 node_color=node_color,
                 node_size=node_size )
plt.axis('off')

"""### Eigenvector Centrality"""

pos = nx.spring_layout(G_weighted)
eigCent = nx.eigenvector_centrality(G_weighted)
node_color = [20000.0 * G_weighted.degree(v) for v in G_weighted]
node_size =  [v * 10000 for v in eigCent.values()]
plt.figure(figsize=(15,15))
nx.draw_networkx(G_weighted, pos=pos, with_labels=False,
                 node_color=node_color,
                 node_size=node_size )
plt.axis('off')

