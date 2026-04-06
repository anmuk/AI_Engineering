# Encoding and Embedding in NLP
 
The whole point of encoding is to convert text into numbers so machines can process it. There are several approaches, each with its own tradeoffs.
 
| Technique | Core Question |
|---|---|
| One Hot Encoding (OHE) | Is the word present or not? |
| Bag of Words (BoW) | How many times does the word appear? |
| TF IDF | How important is this word in this document vs all documents? |
| Embeddings | What is the meaning of this text in context? |
 
OHE, BoW, and TF IDF are traditional, rule based methods. Embeddings (Word2Vec, transformers) capture semantic meaning and belong to the deep learning era.

## One Hot Encoding (OHE)
 
### How It Works
 
OHE marks each word as 1 (present) or 0 (absent). Every unique word in the vocabulary gets its own column.
 
Given four documents:
 
```
D1 = "people watch movie"
D2 = "people watch cricket"
D3 = "people like movie"
D4 = "people like cricket"
```
 
Step 1: Build vocabulary from all unique words.
 
```
Vocabulary = {people, watch, like, movie, cricket}   (size = 5)
```
 
Step 2: Encode. 1 if the word is in the document, 0 if not.
 
```
         people  watch  like  movie  cricket
D1   →     1       1      0     1       0
D2   →     1       1      0     0       1
D3   →     1       0      1     1       0
D4   →     1       0      1     0       1
```
 
This is the **document level** representation (1xN vector per document).
 
### Word Level (Sequence) Representation
 
Each word becomes its own one hot vector. A 3 word document becomes a 3x5 matrix.
 
D1 = "people watch movie":
 
```
people  → [1, 0, 0, 0, 0]
watch   → [0, 1, 0, 0, 0]
movie   → [0, 0, 0, 1, 0]
```
 
Document level vectors go into traditional ML (Logistic Regression, Naive Bayes). Word level sequences go into deep learning models (RNN, LSTM, GRU, attention).
 
### Pros
 
Easy to implement, simple binary representation, no training required, direct mapping.
 
### Cons
 
**Sparse matrices:** vocab of 10,000 words means vectors of length 10,000 with mostly zeros. Memory waste.
 
**High dimensionality:** vector size = vocabulary size. Grows linearly.
 
**No semantic understanding:** "like" and "love" are treated as completely unrelated.
 
**OOV problem:** new words not in the vocabulary cannot be represented at all.
 
---
 
## Bag of Words (BoW)
 
### How It Works
 
BoW counts how many times each word appears. Unlike OHE which only checks presence, BoW tracks frequency.
 
```
D1 = "people watch movie and watch movie again"
D2 = "people watch cricket and watch cricket"
D3 = "people like movie and like movie a lot"
D4 = "people like cricket"
 
Vocabulary = ['again', 'and', 'cricket', 'like', 'lot', 'movie', 'people', 'watch']
```
 
```
         again  and  cricket  like  lot  movie  people  watch
D1   →     1     1      0       0    0     2       1       2
D2   →     0     1      2       0    0     0       1       2
D3   →     0     1      0       2    1     2       1       0
D4   →     0     0      1       1    0     0       1       0
```
 
"movie" appears twice in D1 so it gets a count of 2. OHE would have just said 1.
 
### Difference from OHE
 
OHE: "people watch cricket people" → [1, 1, 1] (just presence)
BoW: "people watch cricket people" → [2, 1, 1] (actual counts)
 
### Pros
 
Easy to implement, captures word frequency, works well for text classification / spam detection / basic sentiment analysis. No training needed.
 
### Cons
 
**Ignores word order:** "dog bites man" and "man bites dog" produce identical vectors [1, 1, 1].
 
**No semantic understanding:** "like" and "love" are completely separate features.
 
**High dimensionality and sparsity:** same problems as OHE.
 
**OOV problem:** same as OHE.
 
**Overemphasizes frequent words:** "movie movie movie" gets high importance even if the repetition is meaningless.

---
 
## Code Walkthrough
 
### OHE with sklearn
 
```python
from sklearn.preprocessing import OneHotEncoder
import numpy
 
documents = ["people watch movie", 
             "people watch cricket", 
             "people like movie", 
             "people like cricket"]
 
# Tokenize: split each sentence into words
tokens = [sentence.lower().split() for sentence in documents]
# [['people', 'watch', 'movie'], ['people', 'watch', 'cricket'], ...]
 
# Reshape for sklearn: each word needs to be its own list
all_words = [[word] for sentence in tokens for word in sentence]
# [['people'], ['watch'], ['movie'], ['people'], ...]
 
# Create encoder. sparse_output=False returns NumPy array instead of sparse matrix.
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(all_words)  # learns the vocabulary
 
# Check vocabulary
print("Vocabulary:", encoder.categories_[0])
# ['cricket' 'like' 'movie' 'people' 'watch']
 
# Encode each sentence (word level representation)
for sentence in tokens:
    encoded_sentence = encoder.transform([[word] for word in sentence])
    print("Sentence:", sentence)
    print(encoded_sentence)
```
 
### BoW with sklearn
 
```python
from sklearn.feature_extraction.text import CountVectorizer
 
documents = ["people watch movie and watch movie again", 
             "people watch cricket and watch cricket",
             "people like movie and like movie a lot",
             "people like cricket"]
 
# CountVectorizer handles tokenization + vocabulary + counting in one step
bow = CountVectorizer()
bow.fit(documents)  # learns vocabulary
 
# Check vocabulary
print(bow.get_feature_names_out())
 
# Transform to BoW vectors
print(bow.transform(documents).toarray())
# .toarray() converts sparse matrix to regular NumPy array
 
# OOV demonstration
new_doc = ["cows are friendly animals"]
bow.transform(new_doc)
# All zeros. None of these words are in the vocabulary. This is the OOV problem.
```
 
### The sklearn Pattern
 
Both OHE and BoW follow the same workflow:
 
1. **Create** the object (OneHotEncoder or CountVectorizer)
2. **Fit** on training data (learns the vocabulary)
3. **Transform** new data using the learned vocabulary
 
You can combine steps 2 and 3 with `fit_transform()` on training data.
