# Natural Language Processing Portfolio
**  will update the part 1/2 and part 5 **

My NLP Portfolio, this repository is meant to showcase projects that I have worked on Natural Language Processing Applictions. This include any applied explorations and/or project.

## Part 1: Pre-Trained Models Total XP: 10  Completed All core practice and Applied exploration

- 1/28: F1_1 Using the Transformers Library

- 2/2: F1_2 Text Classification Data and Evaluation (Applied exploration)

- 2/4: F1_3 ROUGE and Summarization 

- 2/9 Demo Day: Presented to Zack, Agha, Mitchell


## Part 2: Total XP: 10 Completed All core practice and Applied exploration

- 2/9 F2_1 Chat & Instruct Models

- 2/11 F2_2 Large language Models via Web API

- 2/16 F2_3 Markov Models (Applied Exploration)

- 2/18 F2_4 Subword Tokenization and Byte-Pair Encoding

- 2/23 Demo Day: Presented to Evan scherrer and Sagar Bhandari

## Part 3: Machine Learning with Text | Total XP: 10

---

### Core Practice

**2/25 — F3_1 Machine Learning with Text Data**
Explored integer encoding, Bag-of-Words (CountVectorizer), and TF-IDF encodings 
using scikit-learn classifiers on a spam detection dataset.

**3/2 — F3_2 Logistic Regression and PyTorch**
Reviewed integer encoding, bag-of-words, and TF-IDF from the previous class. 
Introduction to PyTorch tensors, loss functions, and the training loop.

**3/4 — F3_3 Multiclass, Adam, and Neural Networks**
Built a fully-connected neural network using PyTorch's nn.Sequential with a ReLU 
hidden layer, applied to the dair-ai/emotion dataset for 6-class emotion classification.

---

### Applied Exploration — F3_2

For this exploration I used the papluca/language-identification dataset, which contains 
90k samples of text across 20 languages. The dataset is perfectly balanced with 3500 
samples per language in the training set.

The biggest thing I noticed is that without Adam, the model's guesses tend to be all 
over the place. With Adam it struggles more on languages with very similar dialects, like 
Spanish and Portuguese, Russian and Bulgarian, and Chinese and Japanese.

I also found that TF-IDF with only 5000 word features loses a lot of character-level 
information, which is key for distinguishing similar languages.

### Demo Day
Presented to: Evan, Sakar


## Part 4: Embeddings | Total XP: 15

### Core Practice

**3/9 — F4_1 Embeddings**  
Explored skip-gram word embeddings using one-hot encoding and negative sampling on toy sentences and the AG News dataset. Learned how embeddings capture semantic similarity by comparing cosine distances between words.

**3/11 — F4_2 PyTorch Embeddings**  
Refactored the embedding pipeline using PyTorch’s `nn.Embedding` layer and applied it to AG News classification, showing how dense embeddings improve efficiency and performance compared to TF-IDF.

**3/25 — F4_3 Semantic Search with Embeddings**  
Used a pretrained SentenceTransformer model to perform semantic search using cosine similarity and ranking with top-k results, retrieving semantically relevant documents even without exact word overlap.

---

### Applied Exploration — F4_1

For this exploration I built word embeddings using a skip-gram model trained on the AG News dataset with one-hot encoding and negative sampling.

The main thing I noticed is that some semantic relationships were captured correctly, like “stock” and “market”, but many others were weak or missing, such as “team” and “game” or “technology” and “computer”. This suggests the model struggled due to limited data and a simplified training setup.

I also found that using a small vocabulary and limited training epochs led to incomplete embeddings, where many words did not appear often enough to learn strong relationships.

---

### Creative Synthesis — F4 Project

For this project I built a semantic song finder using pretrained sentence embeddings (`all-MiniLM-L6-v2`) over lyrics from 21 artists.

The main thing I noticed is that cosine similarity alone was not enough to properly rank results, but using `torch.topk` made the system return songs that matched the mood of a query much more reliably (e.g. sad songs for “heartbroken” and upbeat songs for “happy”). I also found that embeddings worked well for matching meaning across different artists and writing styles without relying on exact word overlap.

---

### Demo Day  
Presented to: Declan

## Part 5: RAG, More on Similarity, and Neural Language Modeling
 
- 3/30 F5_1 Retrival Augmented Generation (RAG)

- 4/1 F5_2 Evan Scherrer WordNet

- 4/6 F5_3 Recurrent Neural Networks and Language Modeling
