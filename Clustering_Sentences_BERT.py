
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embedder = SentenceTransformer('nli-bert-large')

# Corpus with example sentences
# corpus = ['A man is eating food.',
#           'A man is eating a piece of bread.',
#           'A man is eating pasta.',
#           'The girl is carrying a baby.',
#           'The baby is carried by the woman',
#           'A man is riding a horse.',
#           'A man is riding a white horse on an enclosed ground.',
#           'A monkey is playing drums.',
#           'Someone in a gorilla costume is playing a set of drums.',
#           'A cheetah is running behind its prey.',
#           'A cheetah chases prey on across a field.'
#           ]

corpus=[  'A man is riding a horse.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.',
          'A cheetah is running behind its prey.',
          'A cheetah chases prey on across a field.',
          'AR Rehman loves to play the Piano',
          'Virat Kohli plays cricket'
]
corpus_embeddings = embedder.encode(corpus)

# Normalize the embeddings to unit length
corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

clustering_model = AgglomerativeClustering(affinity='cosine',n_clusters=None, distance_threshold=0.6,linkage='average') 
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in clustered_sentences.items():
    print("Cluster ", i+1)
    print(cluster)
    print("")