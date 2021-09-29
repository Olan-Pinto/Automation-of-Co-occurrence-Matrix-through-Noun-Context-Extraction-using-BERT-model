'''Given a block of text, we want to have a function or model that is able to extract important keywords.
We might specify as a parameter how many keywords we want to extract from the given text'''

'''We can use this logic for our porject in the sense that the "Block of text" would be all clustered sentences
So then after keyword extraction each cluster will have one major noun phrase arising from it.'''

'''first step to keyword extraction is producing a set of plausible keyword candidates. (using ngrams)
 keywords are either single words or two words. Rarely do we see long keywords: after all, long, 
complicated keywords are self-defeating since the very purpose of a keyword is to be impressionable, 
short, and concise. Using scikit-learn’s count vectorizer, we can specify the n-gram range parameter, 
then obtain the entire list of n-grams that fall within the specified range.'''

text="""A monkey is playing drums.
"""

# text="""Someone in a gorilla costume is playing a set of drums.

# """

# text="""
#         AR Rehman loves to play the Piano.
# """
from sklearn.feature_extraction.text import CountVectorizer

n_gram_range = (1, 2)
stop_words = "english"

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
all_candidates = count.get_feature_names()

# print(all_candidates)


"""
One glaring problem with the list of all candidates above is that there are some verbs or verb phrases that
we do not want included in the list. Most often or not, keywords are nouns or noun phrases. To remove 
degenerate candidates we need to some basic part-of-speech or POS tagging. 
"""

import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
noun_phrases = set(chunk.text.strip().lower() for chunk in doc.noun_chunks)

nouns = set()
for token in doc:
    if token.pos_ == "NOUN":
        nouns.add(token.text)

all_nouns = nouns.union(noun_phrases)

'''
filter the earlier list of all candidates and including only those that are in the all nouns set we 
obtained through spaCy.
'''
candidates = list(filter(lambda candidate: candidate in all_nouns, all_candidates))

# print(candidates)

"""
a good keyword is one that which accurately captures the semantics of the main text.
The intuition behind embedding-based keyword extraction is the following: if we can embed both the text
and keyword candidates into the same latent embeeding space, best keywords are most likely ones whose 
embeddings live in close proximity to the text embedding itself. In other words, keyword extraction 
simply amounts to calculating some distance metric between the text embedding and candidate keyword  
embeddings, and finding the top k candidates that are closest to the full text.
"""

"""
we use a knowledge-distilled version of RoBERTa. But really, any BERT-based model, or even simply 
autoencoding, embedding-generating transformer model should do the job.
"""
from transformers import AutoModel, AutoTokenizer
model_name = "distilroberta-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""
tokenize the candidate keywords, then pass them through the model itself. BERT-based models typically 
output a pooler output, which is a 768-dimensional vector for each input text.
"""
candidate_tokens = tokenizer(candidates, padding=True, return_tensors="pt")
candidate_embeddings = model(**candidate_tokens)["pooler_output"]

# print(candidate_embeddings.shape)

text_tokens = tokenizer([text], padding=True, return_tensors="pt")
text_embedding = model(**text_tokens)["pooler_output"]

'''
Calculating Distance Measurements

obtain the cosine similarity between the text embedding and candidate embeddings, 
perform an argsort operation to obtain the indices of the keywords that are closest to the text embedding,
slice the top k keywords from the candidates list.
'''
candidate_embeddings = candidate_embeddings.detach().numpy()
text_embedding = text_embedding.detach().numpy()

from sklearn.metrics.pairwise import cosine_similarity

top_k = 5
distances = cosine_similarity(text_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_k:]]

print(keywords)

# from wordwise import Extractor
# text="""A monkey is playing drums.

#     """
# extractor = Extractor()
# keywords = extractor.generate(text, 3)
# print(keywords)

# Look into KeyBERT and check results