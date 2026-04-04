## TF IDF (Term Frequency Inverse Document Frequency)
 
### How It Works
 
TF IDF builds on BoW by weighting words. It asks: how important is this word in this document relative to the entire corpus? Common words get downweighted, rare informative words get upweighted.
 
### The Formulas
 
```
TF(word, D) = occurrences of word in D / total words in D
 
IDF(word) = log(total documents / documents containing the word)
 
TF IDF = TF × IDF
```
 
**TF** captures local importance (within a document).
**IDF** captures global importance (across all documents). Common words that appear everywhere get a low IDF.
 
### Why Log in IDF?
 
Without log, values get extreme. With 1000 documents: a rare word (1 doc) gets IDF = 1000, medium word (10 docs) gets 100, common word (500 docs) gets 2. Too wide a range. With log: log(1000) ≈ 6.9, log(100) ≈ 4.6, log(2) ≈ 0.69. Much more balanced and stable.
 
### Worked Example
 
```
D1 = "people watch cricket"
D2 = "cricket watch cricket"
D3 = "people give comment"
D4 = "cricket give comment"
 
Vocabulary = ['comment', 'cricket', 'give', 'people', 'watch']
```
 
Final TF IDF matrix:
 
```
           comment   cricket    give     people    watch
D1   →     0         0.096      0        0.231     0.231
D2   →     0         0.191      0        0         0.231
D3   →     0.231     0          0.231    0.231     0
D4   →     0.231     0.096      0.231    0         0
```
 
"cricket" in D2 is 0.191 (appears twice) vs 0.096 in D1 (appears once). But it is not doubled because IDF penalizes "cricket" for appearing in multiple documents.
 
### Pros
 
Captures importance not just counts, reduces impact of common words, better than BoW for search and retrieval, no training required, simple and interpretable.
 
### Cons
 
Still ignores word order, no semantic understanding ("car" vs "automobile" treated as different), still sparse and high dimensional, cannot handle context ("bank" as river vs finance gets same vector), OOV problem remains.
 
---
 
## Embeddings (Brief)
 
Embeddings solve the core problems above. They produce dense, low dimensional vectors (100 to 300 dimensions) where similar words end up close together.
 
"I like this movie" vs "I love this film":
With OHE/BoW/TF IDF → "like" ≠ "love", "movie" ≠ "film" → model thinks these are different.
With embeddings → "like" ≈ "love", "movie" ≈ "film" → model understands similarity.
 
Evolution: Word2Vec → Transformer based models (BERT, GPT, etc.)
 