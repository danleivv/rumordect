#!/usr/bin/env python3

import json

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

jpath = 'rumor/3567665295750659.json'
data = json.load(open(jpath))
G = nx.Graph()

cnt = 0
for item in data:
    if item['parent']:
        G.add_edge(item['parent'], item['mid'])
    else:
        cnt += 1
print(cnt)

pos = graphviz_layout(G, prog='twopi')
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=10, alpha=0.5, node_color='blue', with_labels=False)
plt.axis('equal')
plt.show()
