## Embeddings
 
Embeddings solve the fundamental problems that OHE, BoW, and TF IDF all share. Instead of sparse, high dimensional vectors full of zeros, embeddings produce dense, low dimensional vectors where similar words end up close together in vector space.
 
### Why Embeddings Matter
 
Consider two sentences:
 
```
"I like this movie"
"I love this film"
```
 
With OHE / BoW / TF IDF:
"like" ≠ "love" and "movie" ≠ "film". The model thinks these sentences are completely different.
 
With Embeddings:
"like" ≈ "love" and "movie" ≈ "film". The model understands they are similar.
 
### Two Categories of Embeddings
 
**Classical method:** Word2Vec (Google, 2013). Fundamental, still asked about in interviews.
 
**SOTA (State of the Art):** Transformer based models. HuggingFace Sentence Transformers, OpenAI Embeddings, Gemini, etc. These are what you use in production today via APIs.
 
---
 
## Word2Vec
 
### What It Is
 
Word2Vec was created by Google in 2013. It is a neural network that learns to convert words into dense numerical vectors (typically 300 dimensions). Unlike OHE/BoW/TF IDF which are rule based, Word2Vec actually trains a model to learn word representations.
 
### The Agenda
 
The goal is: Word → Number (Vector)
 
More specifically: Word → Features → Model
 
Each word gets mapped to a dense vector where each dimension represents some learned feature of that word.
 
### How Features Work (Conceptual Example)
 
Imagine we have 5 features: Gender, Wealth, Power, Weight, Speak
 
```
            Gender  Wealth  Power  Weight  Speak
King    →     1      1.0     0.99    0.7     1
Queen   →     1      0.8     0.7     0.8     1
Man     →     1      0.5     0.4     0.7     1
Woman   →     1      0.4     0.3     0.8     1
Monkey  →     1      0       0       0.1     0
```
 
Each word becomes a vector of these feature values:
 
```
King  → [1, 1, 0.99, 0.7, 1]
Man   → [1, 0.5, 0.4, 0.7, 1]
Woman → [1, 0.4, 0.3, 0.8, 1]
```
 
In reality, Word2Vec uses 300 features (dimensions), and the model learns what those features represent automatically. You do not define them manually. The values are all between 0 and 1.
 
Google's Word2Vec specifically uses 300 dimensions, meaning every word becomes a 1x300 vector.
 
### Vector Arithmetic (The Famous Example)
 
Because embeddings capture meaning as numbers, you can do math with words:
 
```
Queen = King - Man + Woman
```
 
In vector form:
 
```
Queen = [1, 1, 0.99, 0.7, 1] - [1, 0.5, 0.4, 0.7, 1] + [1, 0.4, 0.3, 0.8, 1]
```
 
This works because the model has learned that the relationship between King and Man is the same as the relationship between Queen and Woman. Word embeddings enable vector operations that capture semantic relationships.
 
### Visualization
 
If you plot word vectors in 3D space (using 3 features like Strong, Human, Hardword):
 
```
Men   → [5, 6, 4]
Women → [6, 6, 6]
Child → [2, 6, 3]
```
 
Words with similar meanings cluster together. Men and Women would be close to each other but far from Child on certain axes.
 
### How Word2Vec Actually Learns
 
Word2Vec uses a neural network:
 
**Architecture:** Input layer → Hidden layer → Output layer (Single layer perceptron or multilayer perceptron)
 
**Single Perceptron:** Takes inputs (x), multiplies by weights (w), adds bias (b), passes through activation function.
 
```
Output = Activation(Wx + b)
```
 
**Neural Network layers:**
1. Input layer
2. Hidden layer(s)
3. Output layer
 
**Multilayer perceptron** = multiple hidden layers.
 
**Training process:**
1. Forward Propagation: FP → (Wx + b) with activation. Compute prediction.
2. Loss: Output(Prediction) minus Actual = Loss
3. Backward Propagation: BP → Optimizer → Gradient Descent → Adjust the weights
4. One complete FP + BP = 1 Epoch
5. Weights are initialized randomly, then through many epochs the loss decreases until the model finds the best possible weights.
 
Word2Vec trains this neural network on large amounts of text. The learned weights in the hidden layer become the word vectors.
 
### FastText
 
FastText is an improved version of Word2Vec, also called Avg Word2Vec. It handles subword information, which helps with the OOV problem that Word2Vec still has.
 
---
 
## Transformer Based Embeddings (SOTA)
 
Transformers have become the dominant approach for embeddings. They are what powers modern AI.
 
Available through APIs and libraries:
HuggingFace → Sentence Transformers
OpenAI → OpenAI Embeddings
Google → Gemini Embeddings
 
These models produce contextual embeddings, meaning the same word gets different vectors depending on the sentence it appears in. "Bank" in "river bank" gets a different embedding than "bank" in "I went to the bank to deposit money."
 
This is the key difference from Word2Vec, where every word always gets the same vector regardless of context.
 
### Connection to GenAI
 
All of this encoding and embedding knowledge feeds directly into GenAI concepts:
Fine Tuning, RAG, Agents, MCP, etc. all rely on transformer based embeddings under the hood.

---

## Summary: The Evolution
 
```
OHE       → Presence (0 or 1)
BoW       → Count (frequency)
TF IDF    → Count + Importance (weighted frequency)
BM25      → Evolution of TF IDF (used in RAG today)
Word2Vec  → Dense vectors via neural network (semantic meaning, 2013)
FastText  → Improved Word2Vec (handles subwords)
Transformers → Contextual embeddings (SOTA, what powers modern AI)
```
 
OHE, BoW, TF IDF, and BM25 are all encoding methods (rule based, no training).
Word2Vec, FastText, and Transformers produce embeddings (learned via neural networks, capture semantic meaning).