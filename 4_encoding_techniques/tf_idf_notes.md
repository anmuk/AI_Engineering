## TF IDF (Term Frequency Inverse Document Frequency)
 
### What It Does
 
TF IDF builds on BoW by asking: how important is this word in this specific document compared to the entire corpus? It does not just count words. It weighs them. Common words that appear everywhere get downweighted. Rare, informative words get upweighted.
 
### The Formulas
 
**TF (Term Frequency):**
 
```
TF(word, D) = occurrences of the word in the sentence / total words in the sentence
```
 
If a word shows up more in a document, it is probably more relevant to that document.
 
**IDF (Inverse Document Frequency):**
 
```
IDF(word) = log(total number of documents / number of documents containing the word)
```
 
IDF penalizes words that show up in many documents (they are not useful for distinguishing documents) and rewards rare words.
 
**TF IDF:**
 
```
TF IDF = TF × IDF
```
 
TF captures local importance (within a document). IDF captures global importance (across all documents). Multiplying them gives a balanced score.
 
### Why TF?
 
More occurrences of a word in a document means that word is more important to that document.
 
### Why IDF?
 
Common words like "is", "the", "and" appear in almost every document. They are noise. IDF reduces their weight so they do not dominate the results.
 
### Why Multiply TF × IDF?
 
TF alone tells you importance within one document. IDF alone tells you importance across the corpus. Combining them gives you the full picture: a word that is frequent locally but rare globally is the most informative.
 
### Why Log in IDF?
 
Without log, IDF values become extreme and the model becomes unstable.
 
Example with 1000 documents:
 
```
Without log:
  Rare word (appears in 1 doc):    IDF = 1000 / 1   = 1000
  Medium word (appears in 10 docs): IDF = 1000 / 10  = 100
  Common word (appears in 500 docs): IDF = 1000 / 500 = 2
 
  Range: 2 to 1000. The rare word completely dominates.
 
With log:
  Rare word:   log(1000) ≈ 6.9
  Medium word:  log(100)  ≈ 4.6
  Common word:  log(2)    ≈ 0.69
 
  Range: 0.69 to 6.9. Still preserves the ranking but the gap is manageable.
```
 
Log compresses the scale while keeping the importance order intact. Rare words still score higher, but they do not overpower everything else. This is essentially normalization.
 
sklearn uses natural log (ln / log base e) by default. The base does not matter conceptually because all log bases compress the same way and preserve the same ranking.
 
### Worked Example
 
```
D1 = "people watch cricket"
D2 = "cricket watch cricket"
D3 = "people give comment"
D4 = "cricket give comment"
 
Vocabulary = ['comment', 'cricket', 'give', 'people', 'watch']
```
 
Computing TF × IDF for each cell (showing the formula before the final number):
 
```
           comment          cricket          give             people           watch
D1 →       0                (1/3)*log(4/3)   0                (1/3)*log(4/2)   (1/3)*log(4/2)
D2 →       0                (2/3)*log(4/3)   0                0                (1/3)*log(4/2)
D3 →       (1/3)*log(4/2)   0                (1/3)*log(4/2)   (1/3)*log(4/2)   0
D4 →       (1/3)*log(4/2)   (1/3)*log(4/3)   (1/3)*log(4/2)   0                0
```
 
Where:
log(4/3) ≈ 0.28768
log(4/2) = log(2) ≈ 0.69315
 
Final TF IDF matrix:
 
```
           comment   cricket    give     people    watch
D1 →       0         0.096      0        0.231     0.231
D2 →       0         0.191      0        0         0.231
D3 →       0.231     0          0.231    0.231     0
D4 →       0.231     0.096      0.231    0         0
```
 
"cricket" in D2 is 0.191 vs 0.096 in D1 because "cricket" appears twice in D2 (higher TF). But it is not simply doubled because IDF drags it down since "cricket" appears in 3 out of 4 documents.
 
"people" gets 0.231 where it appears because it only shows up in 2 out of 4 documents (higher IDF = more distinctive).
 
### Pros
 
Captures word importance, not just counts. Rare words get higher weight, common words get lower weight. Better than BoW for relevance tasks like search engines, information retrieval, and document ranking. Google used TF IDF for indexing pages before neural networks. BM25 (used in RAG systems today) is an evolution of TF IDF. No training required, just direct calculation. Simple and mathematically interpretable.
 
### Cons
 
Still ignores word order. "dog bites man" and "man bites dog" produce the same representation.
 
No semantic understanding. "car" and "automobile" are treated as completely different words. "love" and "like" have no captured relationship.
 
Still sparse and high dimensional. Vocabulary size = vector size. Many zeros, inefficient memory usage.
 
Cannot handle context. The word "bank" gets the same vector whether it means a river bank or a financial institution.
 
OOV problem. New words not in the vocabulary are simply ignored.
 
**Bottom line:** TF IDF improves by adding importance but still fails to understand meaning and context.

---

## TF IDF Code Walkthrough
 
```python
from sklearn.feature_extraction.text import TfidfVectorizer
 
documents = ["people watch movie and watch movie again", 
             "people watch cricket and watch cricket",
             "people like movie and like movie a lot",
             "people like cricket"]
```
 
**Create the TF IDF vectorizer:**
 
```python
tf_idf = TfidfVectorizer()
```
 
TfidfVectorizer works just like CountVectorizer (BoW) but instead of raw counts, it computes TF IDF scores. It handles tokenization, vocabulary building, TF calculation, IDF calculation, and the final multiplication all in one step.
 
**Fit and transform:**
 
```python
tf_idf_vector = tf_idf.fit_transform(documents)
```
 
`fit_transform()` does two things at once: learns the vocabulary and IDF values from the data (fit), then converts each document into its TF IDF vector (transform).
 
**What the output looks like:**
 
```python
tf_idf_vector
# <4x8 sparse matrix of type 'numpy.float64'
#   with 17 stored elements in Compressed Sparse Row format>
```
 
It returns a sparse matrix by default (4 documents × 8 vocabulary words). Only 17 out of 32 cells have nonzero values, which is why sklearn stores it as sparse to save memory.
 
**Convert to readable array:**
 
```python
tf_idf_vector.toarray()
```
 
Output:
 
```
         again    and      cricket  like     lot      movie    people   watch
D1 →  [0.3877,  0.2475,  0.0000,  0.0000,  0.0000,  0.6114,  0.2023,  0.6114]
D2 →  [0.0000,  0.2685,  0.6632,  0.0000,  0.0000,  0.0000,  0.2195,  0.6632]
D3 →  [0.0000,  0.2475,  0.0000,  0.6114,  0.3877,  0.6114,  0.2023,  0.0000]
D4 →  [0.0000,  0.0000,  0.6404,  0.6404,  0.0000,  0.0000,  0.4239,  0.0000]
```
 
Notice how "people" has relatively low scores (0.2023 in D1, D3 and 0.2195 in D2) because it appears in all 4 documents, so IDF penalizes it heavily. Meanwhile "again" in D1 gets 0.3877 because it only appears in one document, making it distinctive.
 
Also note that sklearn's TfidfVectorizer applies L2 normalization by default, so the values differ slightly from manual TF × IDF calculations. The relative ordering and logic remain the same.
 
**Check the vocabulary:**
 
```python
tf_idf.get_feature_names_out()
# ['again', 'and', 'cricket', 'like', 'lot', 'movie', 'people', 'watch']
```
 
Sorted alphabetically, same convention as CountVectorizer.
 
### The sklearn Pattern (Same for OHE, BoW, and TF IDF)
 
1. **Create** the object (OneHotEncoder / CountVectorizer / TfidfVectorizer)
2. **Fit** on training data (learns vocabulary and any parameters like IDF values)
3. **Transform** new data using what was learned
 
Or use `fit_transform()` to do steps 2 and 3 together on training data.

