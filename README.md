# Natural Language Processing Portfolio

My NLP Portfolio, this repository is meant to showcase projects that I have worked on Natural Language Processing Applictions. This include any applied explorations and/or project.

## Part 1: Pre-Trained Models | Total XP: 10

### Core Practice

**1/28 — F1_1 Using the Transformers Library**

Explored the Hugging Face Transformers library, learned about sentiment analysis pipelines, hardware acceleration with GPUs/TPUs, batch processing of text, and specifying different models like roberta-base-go_emotions.

**2/2 — F1_2 Text Classification Data and Evaluation**

Loaded and explored datasets from Hugging Face, specifically the go_emotions dataset, evaluated classification models using scikit-learn metrics such as accuracy, precision, recall, F1 score, and confusion matrices.

**2/4 — F1_3 ROUGE and Summarization**

Learned about ROUGE metrics for evaluating summarization, used the BART summarization model on news articles and legislative bills, computed ROUGE scores to assess summary quality.

---

### Applied Exploration — F1_2

For this exploration, I evaluated a different text classification model on a sentiment analysis dataset. I used the cardiffnlp/twitter-xlm-roberta-base-sentiment model on the syedkhalid076/Sentiment-Analysis dataset, which contains 83,989 training samples across three sentiment categories. The model achieved an accuracy of 0.4 on the test set, but performed perfectly on the validation set, suggesting potential overfitting or dataset issues.

### Demo Day

Presented to: Zack, Agha, Mitchell


## Part 2: Working with small, large, and classic language models | Total XP: 10

### Core Practice

**2/9 — F2_1 Chat & Instruct Models**

Introduced to chat and instruct models using SmolLM2, learned about multi-turn conversations, benchmarks for evaluation, and conducted group exercises comparing different instruct models.

**2/11 — F2_2 Large Language Models via Web API**

Explored using large language models via APIs, specifically OpenAI's GPT models, and learned about tools like web search for enhanced responses.

**2/16 — F2_3 Markov Models**

Built and trained Markov models for text generation, used NLTK corpora like Shakespeare and Moby Dick, generated text using random next word prediction.

**2/18 — F2_4 Subword Tokenization and Byte-Pair Encoding**

Learned about subword tokenization, Byte-Pair Encoding (BPE), used Hugging Face tokenizers, and retrieved text from the web for tokenization.

---

### Applied Exploration — F2_1

For this exploration, I compared two instruct models of similar size: SmolLM2-360M-Instruct and Qwen/Qwen2.5-0.5B-Instruct. I created 5 prompts and had group members vote on the best responses. The SmolLM2 model generally performed better on technical and creative tasks, while Qwen excelled in straightforward questions.

### Applied Exploration — F2_3

For this exploration, I enhanced the MarkovModel class to support higher-order models (2nd, 3rd order). I trained on Shakespeare Hamlet and Alice in Wonderland corpora, generating text at different orders. Higher-order models produced more coherent text but required more data to avoid unseen states.

### Applied Exploration — F2_4

For this exploration, I implemented a basic BPE tokenization algorithm and used it on text retrieved from Project Gutenberg and Wikipedia. I then used the tokenized text to train a Markov model, comparing the generated text quality with word-level tokenization. Subword tokenization led to more diverse but sometimes fragmented text generation.

### Demo Day

Presented to: Evan Scherrer, Sagar Bhandari

## Part 3: Machine Learning with Text | Total XP: 10

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

## Part 5: RAG, More on Similarity, and Neural Language Modeling | Total XP: 10

### Core Practice

**3/30 — F5_1 Retrieval-Augmented Generation (RAG)**

Built a retrieval-augmented generation system that combines semantic search with language model generation. The pipeline loads Drake University course information, encodes courses into embedding vectors using pretrained sentence transformers, and performs semantic retrieval using cosine similarity to find relevant courses given user queries. Retrieved courses are then formatted as context and passed to a small language model to generate grounded answers about the course schedule.

**4/1 — F5_2 Semantic Similarity with WordNet**

Explored WordNet, a lexical database organizing English words into synsets (synonym sets) structured around single concepts. Learned how synsets connect through relationships like hypernyms (more general concepts) and hyponyms (more specific concepts), forming a hierarchy where similar words appear closer together. Implemented semantic similarity metrics including path similarity (based on shortest distance in the hierarchy) and Wu-Palmer similarity (accounting for depth and lowest common ancestor).

**4/6 — F5_3 Recurrent Neural Networks and Language Modeling**

Built neural language models using recurrent neural networks to predict the next token in a sequence. Implemented a small RNN language model with embedding, RNN, and output layers trained on a small corpus of sentences. The model processes sequences one token at a time, maintaining a hidden state that carries information forward through time, enabling it to learn patterns and generate coherent text by repeatedly predicting the most likely next token.

---

### Applied Exploration — F5_1

For this exploration, I evaluated three SmolLM models of different sizes (135M, 360M, 1.7B) on the same RAG pipeline using Drake course information. For each model, I created 5 benchmark questions covering different query types (asking about specific instructors, course attributes, timing, prerequisites, and topics). I evaluated how well each model's responses stayed grounded in the retrieved context versus hallucinating course details. The larger 1.7B model generated more detailed and accurate responses, while the 135M model sometimes invented course information. Importantly, all models struggled with out-of-scope questions, sometimes attempting to answer with general knowledge rather than acknowledging missing context. This highlighted the challenge of training models to recognize the boundaries of their training data and the value of combining RAG with model alignment.

---

### Demo Day

Presented to: Declan, Alex

## Part 6: Going Deeper with RNNs, Transformers, and Transfer Learning | Total XP: 10

### Core Practice

**4/8 — F6_1 LSTM and Encoder-Decoder Architectures**

Implemented sequence-to-sequence models using GRU-based encoder-decoder architectures for a synthetic reverse-sequence task. Built an Encoder that reads an input sequence and produces a final hidden state, and a Decoder that generates an output sequence starting from that hidden state. Implemented teacher forcing during training, where the decoder receives correct previous tokens rather than its own predictions, which significantly accelerates training. Explored the encoder-decoder bottleneck problem where a single context vector must contain all information about the input.

**4/10 — F6_2 Attention**

Extended the encoder-decoder architecture with attention mechanisms to address the bottleneck problem. Instead of passing only the final encoder hidden state, attention allows the decoder to access all encoder hidden states with learned weights determining which input positions are most relevant for generating each output token. Implemented attention scoring via dot-product similarity, computed softmax weights, and created weighted sums of encoder states. Demonstrated how attention enables the decoder to focus on relevant input information when generating each output token.

**4/15 — F6_3 Transformers**

Learned the transformer architecture that replaces recurrent structures with self-attention layers for parallel computation. Studied key transformer components including self-attention for extracting information from large contexts, multi-headed attention where multiple attention heads learn different relationships between words, and positional encodings to preserve word order. Explored how stacked transformer blocks allow multiple rounds of refinement, with early layers learning simple patterns and later layers combining them into abstract understanding.

**4/17 — F6_4 Fine-Tuning Transformers**

Practiced supervised fine-tuning of pretrained transformer models on domain-specific data. Used LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning, which adds small trainable adapter paths inside transformer layers while keeping original pretrained weights frozen. Fine-tuned Qwen2.5-0.5B-Instruct on a Drake course schedule question-answering dataset, automatically generating training examples from structured course data, and observed how the model learns to answer domain-specific questions while retaining general language understanding.

---

### Applied Exploration — F6_3

For this exploration, I researched the BERT family of transformer models (BERT, DistilBERT, RoBERTa), learning about their architecture and applications. BERT was developed by Google in 2018 as an encoder-only transformer that reads text bidirectionally, meaning it attends to both left and right context simultaneously. This bidirectional reading enables BERT to deeply understand word meaning and context. DistilBERT, created by Hugging Face, reduces BERT's size through distillation while maintaining most of its performance. RoBERTa, developed by Meta, improves upon BERT's training procedure. All three are encoder-only models optimized for understanding rather than generation, making them excellent for text classification, question answering, named entity recognition, and sentence similarity. I found that bidirectional attention is a key difference from decoder-only models like GPT, and that smaller variants like DistilBERT sacrifice relatively little accuracy for significant speed improvements, making them practical for deployment. One remaining question is how much information about specific domains can be retained during fine-tuning before the model loses its general language understanding.

---

### Demo Day

Presented to: Declan, Jacob

## Part 7: Agents, Alignment, and Test-Time Compute | Total XP: 35


### Core Practice

**4/27 — F7_1 Tool Calling**

Explored agentic AI by building a tool-calling assistant that can decide when to use tools, generate structured tool calls, and use tool results to answer questions. Built a calendar management system where the model decides between listing events or creating new events based on user requests.

**4/29 — F7_2 Alignment with Preference Tuning**

Learned about alignment and preference optimization, exploring how to make language models behave in ways that match human preferences. Studied RLHF (Reinforcement Learning from Human Feedback) and implemented DPO (Direct Preference Optimization) for training a course advising assistant to prefer grounded, honest answers over hallucinations.

**5/4 — F7_3 Thinking Models and Quantization**

Investigated two topics: chain-of-thought prompting where models show intermediate reasoning steps before answering, and quantization where model weights are stored with fewer bits to reduce memory usage. Experimented with different prompting styles and compared models loaded in 16-bit, 8-bit, and 4-bit formats.

---

### Applied Exploration — F7_1

For this exploration, I built a tool-calling calendar assistant using Qwen2.5-0.5B-Instruct that could parse user requests for calendar operations, generate JSON-structured tool calls with appropriate arguments, and execute Python functions to modify or query a calendar. The main challenge I encountered was handling natural language date references like "tomorrow" and handling ambiguous time specifications, which required preprocessing the user input before passing it to the tool execution layer.

---

### Creative Synthesis — Large Project

**WordNet Connections Game** (20xp)

Built a word puzzle game leveraging WordNet's hypernym hierarchy. The core mechanic involves finding common ancestors between two words in WordNet's semantic tree. The key insight is that word sense ambiguity becomes a feature: by selecting the optimal synset for each word, seemingly unrelated words can share surprisingly deep common ancestors. For example, "love" and "gun" appear unconnected using their default meanings, but under different senses they both resolve to "person" in the hierarchy. The game uses a scoring system where the depth of the found ancestor determines points, making it challenging to discover deep connections while still being able to safely guess "entity" at the root.

**Movie Recommendation System with Preference Learning** (5xp)

Built an end-to-end pipeline combining RAG and DPO. The system loads 5000 movies from Kaggle, encodes movie descriptions and keywords using pretrained sentence embeddings, and performs semantic retrieval using cosine similarity. A language model generates personalized recommendations from the retrieved candidates, and a Gradio interface collects user preference votes between pairs of recommendations. These preferences are then used to fine-tune the model with DPO, teaching it to improve its recommendation quality based on actual user feedback.

---

### Demo Day

Presented to: Declan, Ujan, Mitchell, Jacob 
 