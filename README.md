# Multi-task-Neural-Networks-for-Hierarchical-Text-Classification-Machine-Learning-Research-Project-

# Datasets 
- Amazon Product Reviews https://www.kaggle.com/kashnitsky/hierarchical-text-classification  
- DBPedia https://www.kaggle.com/danofer/dbpedia-classes  
- Coronavirus Tweets https://www.kaggle.com/smid80/coronavirus-covid19-tweets  

# Experiment Design
- Accuracy is used for evaluation metric
- Randomly train test split with percentages: 80%, 20% 
- NLTK is used for text preprocessing and tokenization
- Word2Vec for word embedding 

# Models tried
- Fully Connected Network as Baseline
- Shallow/Deep CNN
- Shallow/Deep LSTM
- With/Without Multitask network structure

# Results
- All the Multitask networks outperform the vanilla networks
- Shallow LSTM achieve the highest accuracy, but didn't benefit from stacking more layers
- CNN benefit from stacking more layers with residual connections

---

# Evaluating-Student-Writing-kaggle (Experiments with Transformer based model)
https://www.kaggle.com/c/feedback-prize-2021
