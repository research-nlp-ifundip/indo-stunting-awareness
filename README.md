# indo-stunting-awareness: An Indonesian Twitter Dataset on Stunting Awareness

## Introduction
This dataset consists of **tweets from Twitter discussing the topic of Stunting In the Indonesian language**. The dataset is designed to support research and data analysis using unsupervised learning approaches. For general information, Stunting refers to impaired growth and development in children due to chronic malnutrition, recurrent infections, and insufficient psychosocial stimulation. This issue significantly impacts children's physical and cognitive development and is a critical public health indicator. Twitter is widely used to share information and raise community awareness about health issues, including Stunting. This dataset aims to facilitate further analysis of public perceptions and discussions about stunting on social media.

There are **45.722 documents** in the `dataset` folder. The documents are available in **CSV format**.


## How It Works?

1. **Classification**
   - Initially, the crawling data will be manually labeled, with 500 samples designated for training and 50 for testing. Each label represents the sentiment polarity of the data: "positive," "negative," or "neutral."
   - The data will first be preprocessed to ensure it is clean. Various machine learning algorithms (including Naive Bayes, Random Forest, Logistic Regression, Support Vector Machine, and Hierarchical Classification) will be used to train the model. The best-performing model will then predict the sentiment of the remaining unlabeled data. You can find the code for this stage in the `dataset/ST_Classification` folder.

3. **Topic Modeling**
   - Topic modeling will be performed to identify topics within the documents using Latent Dirichlet Allocation (LDA). The code for this section is located in the `dataset/ST_Topic_Modeling` folder.

4. **Sentiment Analysis**
   - Each topic identified in the previous stage will undergo sentiment analysis to provide a comprehensive understanding of public sentiment related to each topic. The code for this analysis can be found in the `dataset/ST_Sentiment_Analysis` folder.

**Note:**
- The training and testing data used in sentiment analysis are derived from the dominant topics identified in the topic modeling stage.


## Dataset Details
The dataset contains Twitter data collected from January 1, 2021, to September 3, 2021. It includes three columns: `date` representing the tweet's date, `content` with the tweet's text, and `username` indicating the user's Twitter handle. The data is unsupervised and is not labeled for any specific classification or analysis.


## License
This dataset is free; you do not need our permission to use it. However, please be aware that creating and distributing copies of this dataset in your repository without our consent is not allowed. 
