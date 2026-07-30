import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import networkx as nx
import plotly.graph_objects as go
def generate_wordcloud(papers):
    text = " ".join([p.get("abstract", "") for p in papers if p.get("abstract")])
    if not text:
        st.info("No abstracts to build a word cloud.")
        return

    custom_stopwords = set(STOPWORDS).union(stop_words)
    wc = WordCloud(
        width=800,
        height=400,
        stopwords=custom_stopwords,
        background_color="white",
        colormap="viridis"
    ).generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# ========== CO-AUTHOR GRAPH ==========
def build_coauthor_graph(papers):
    G = nx.Graph()

    for p in papers:
        authors = p.get("authors", ["Unknown"])
        # normalize
        authors = [a.strip() for a in authors if a and isinstance(a, str)]
        # filter obvious affiliation tokens
        authors = [a for a in authors if not looks_like_affiliation(a) and a.lower() != "unknown"]

        # add nodes
        for a in authors:
            G.add_node(a)

        # add edges (with weight)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                u, v = authors[i], authors[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    if G.number_of_nodes() == 0:
        st.info("No author data available to build co-author graph.")
        return

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_text = [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y); node_text.append(n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="top center",
        marker=dict(size=12, color='skyblue')
    ))
    st.plotly_chart(fig, use_container_width=True)

