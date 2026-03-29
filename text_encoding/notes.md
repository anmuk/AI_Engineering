Encoding - numerical representation
Embedding - meaningful numerical representation - vector of meaninful number

**What is encoding ?**

Emcoding is converting raw data into numerical form ( oftern sparse) without necessarily preserving semantic meaning of data

-> This is count based, matrix factorization based

**Methods of encoding**

one-hot encoding
Bag of Words (BOW)
Term frequency - Inverse Document Frequency (TF-IDF)
BM25: BM25 is an improved extension of TF-IDF, but technically it is a ranking algorithm, not just an encoding method
GloVe: Global Vectors is a count-based + matrix factoriztion-based

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
vector - list of numbers [1.7, -.3, 3.1]
in NLP - embedding = vector with meaning
Capture semantic relationships

Matrix - row x column, so 3x2 means 3 rows and 2 columns


**How to run python venv for setup**

mp note: use python 3.11 for this practical

uv python list

uv python install 3.11

uv venv env —python <mention python version> (use only cpython interpreter)

command for installting the requirement file: uv pip install -r requirements.txt