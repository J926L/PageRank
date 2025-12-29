# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

def run_pagerank(data_dir: Path):
    # 数据加载
    emails = pd.read_csv(data_dir / "Emails.csv")
    # 读取别名文件
    aliases_df = pd.read_csv(data_dir / "Aliases.csv")
    aliases = {}
    for index, row in aliases_df.iterrows():
        aliases[row['Alias']] = row['PersonId']
    # 读取人名文件
    persons_df = pd.read_csv(data_dir / "Persons.csv")
    persons = {}
    for index, row in persons_df.iterrows():
        persons[row['Id']] = row['Name']

    def unify_name(name):
        name = str(name).lower()
        name = name.replace(",","").split("@")[0]
        if name in aliases:
            return persons.get(aliases[name], name)
        return name

    def show_graph(graph, filename):
        # 1. 布局优化：进一步加大 k 值，并忽略边权重对布局的影响（避免强连接拉得太近）
        # 使用 weight=None 让布局只取决于拓扑结构，k=2.0 提供极大的弹力
        positions = nx.spring_layout(graph, k=2.0, iterations=150, seed=42, weight=None)
        
        # 2. 节点大小优化：使用对数缩放 (log scaling)
        # 原始 PageRank 差异巨大，直接线性缩放会导致中心节点过大。对数缩放能平衡视觉重心。
        pageranks = np.array([x['pagerank'] for v, x in graph.nodes(data=True)])
        # 映射到 300 - 3000 的范围
        nodesize = (np.log1p(pageranks * 100) / np.log1p(pageranks.max() * 100)) * 3000 + 300
        
        # 3. 边粗细优化
        edgesize = [np.sqrt(e[2]['weight']) * 0.5 for e in graph.edges(data=True)]
        
        plt.figure(figsize=(20, 20)) # 进一步加大画布
        
        # 4. 分层绘制：先画边，再画节点，最后画标签
        nx.draw_networkx_edges(graph, positions, width=edgesize, alpha=0.15, edge_color='gray', 
                             arrowsize=12, connectionstyle='arc3,rad=0.1')
        
        nx.draw_networkx_nodes(graph, positions, node_size=nodesize, alpha=0.6, 
                             node_color='skyblue', edgecolors='white', linewidths=1)
        
        # 5. 标签优化：添加半透明背景(bbox)使文字更清晰，并稍微偏移
        labels = {node: node for node in graph.nodes()}
        for node, (x, y) in positions.items():
            plt.text(x, y + 0.02, s=node, fontsize=10, 
                    horizontalalignment='center', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        
        plt.axis('off')
        plt.tight_layout()
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        save_path = output_dir / filename
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")
        # plt.show() # Uncomment if running in interactive environment

    emails.MetadataFrom = emails.MetadataFrom.apply(unify_name)
    emails.MetadataTo = emails.MetadataTo.apply(unify_name)

    edges_weights_temp = defaultdict(int)
    for row in zip(emails.MetadataFrom, emails.MetadataTo):
        temp = (row[0], row[1])
        edges_weights_temp[temp] += 1

    edges_weights = [(key[0], key[1], val) for key, val in edges_weights_temp.items()]

    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges_weights)
    pagerank = nx.pagerank(graph)
    nx.set_node_attributes(graph, name='pagerank', values=pagerank)

    # show_graph(graph, "pagerank_full.png")
    
    pagerank_threshold = 0.005
    small_graph = graph.copy()
    for n, p_rank in graph.nodes(data=True):
        if p_rank['pagerank'] < pagerank_threshold: 
            small_graph.remove_node(n)
            
    show_graph(small_graph, "pagerank_core.png")
