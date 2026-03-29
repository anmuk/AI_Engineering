Encoding - numerical representation
Embedding - meaningful numerical representation - vector of meaningful number
Embedding is a dense(non sparse), high-dimensional numerical vector that represents the semantic meaning of data

High dimensional: typical ranged from 100 to several thousand dimensions depending on model

---

**What is encoding ?**

Emcoding is converting raw data into numerical form ( oftern sparse) without necessarily preserving semantic meaning of data

-> This is count based, matrix factorization based

---

**Methods of encoding**

one-hot encoding
Bag of Words (BOW)
Term frequency - Inverse Document Frequency (TF-IDF)
BM25: BM25 is an improved extension of TF-IDF, but technically it is a ranking algorithm, not just an encoding method
GloVe: Global Vectors is a count-based + matrix factoriztion-based

---

**Vectors and Matrix**

Vector - set of numerical values

5 -> not a vector, this is a scalar value
[5,10] -> this is a vector
[5 10] -> this is a matrix
[2  4]

Matrix - collection of vectors
vector - special case of matrix (1 row or 1 column)
Matrix - a table of numbers (grid)

A vector is a numerical representation of data in n-dimensional space.
In mathematics vector comes from linear algebra
vector - array of numbers [1.7, -.3, 3.1], can be float values
in NLP - embedding = vector with meaning
Capture semantic relationships

Matrix - row x column, so 3x2 means 3 rows and 2 columns

In 2 dimension graph: P(4,5) would represent a 2D vector or a position of point P at coordinates 4,5 in top right quadrant

in 3 dimension graph: P(x,y,z)

4 dimension: [1, 2, 3, 4]
.
.
.
n-dimension : [1, 2, 5, 8, ....n]

---

**OpenAI Embedding model**

3072 dimensions of the model

This means inside vector there will be 3072 numbers - [1,2,3,4 .... 3072]

->
if point P is at 4,5:

magnitude or distance from 0,0:
using Baudhayana theoream: magnitude = hypoteneuse = root of 41

concept of similarity - 

vectors at (1,0) and (0,1) - dot product value = 0 
angle between two = 90 degree, cos90 = 0

---

**More details on encoding and embedding**

Encoding: Data -> text -> vector
Embedding: data (like text, image, video, audio) -> vector

Transformers used nowadays - State of Art Model, i.e., the latest model - SOTA technique

Corpus: Collection of documents
Document: Collection of paragraphs
Paragraph: Collection of sentences
Sentence: Collection of words/tokens
Word(token): May split into tokens
Character: Smallest unit

Embedding early on in 2018/19/20  used word2vec or fasttext
these were based on neural networks

Nowadays SOTA used, which are based on transformers
word/document/images/audio/video/complete corpus or book converted into embedding
Any kind of data converted

How to check similarity ?

1) Dot product 
2) Cosine value
3) ED

---

**Classical word embedding models vs new models**

Word2Vec, FastText

Limitations:
- Create embeddings at the word level only
- Product static embeddings (fixed vectors)
- Do not understand the meaning

Example:

sentence1: I sat on the river bank
sentence2: I deposited money in the bank

Word: bank

In a static embedding model like Word2Vec, the word is represented by the same vector in both sentences -> [0.21, -0.44, 0.89...]

in SOTA transformer-based embedding model, the word will be represented by different vectors
bank -> [0.21, -0.44, 0.89...]
bank -> [0.93, 0.01, 0.75,...]

BECAUSE IT UNDERSTANDS CONTEXT

---

**Difference in Encoding and Embedding**

frequency based  vs network based
Convert data to numbers vs  covert data to numbers (represent meaning)
Oftern sparse (count based) vs dense(NN or Transformer based)
Semantic Meaning not preserved vs preserved
Context Awareness not preserved vs preserved

There are several SOTA models - paid and open source both available

---

**Keyword vs Semantic Search (similarity search):**

Keyword search matches words.
Semantic matches meaning: find out similarity using matrix like cosine, euclidian, dot product, etc.

Keyword search fails when:
"How to reduce heart risk"
"Ways to lower cadiovascular disease chances"

But system will mainly look for exact words

A traditional keyword search may not match it

reduce not equal to lower
heart risk not equal to cardiovascular disease

But semantic search will give these a strong match









