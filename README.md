# Sentiment Analysis

Sentiment analysis aims to analyze and predict the opinion, sentiment and view towards the given subject. This project tends to show mainly two concepts, comparison of different Machine learning models and how different models can be developed to show the sentiment behind the tweets as positive, negative or, neutral.
There is a room for improvement in every aspect of development. In this regard, the accuracy of all these models can always be improved. Accuracy improvement generally involves two methods, parameters tuning and dataset cleaning.

There are altogether 3 lexicon based methods and 7 machine learning models in this project. For lexicon method, SentiWordnet, VADER and Textblob are used. And for machine learning models, Logistic Regression, Decision Tree, Support Vector Machine, Random Forest, Long Short Term Memory (LSTM), Naive Bayes and XG Boost are used.

**Dataset:**

The dataset used in the project has been downloaded from Kaggle [Link to dataset](https://www.kaggle.com/kazanova/sentiment140?select=training.1600000.processed.noemoticon.csv). The total of 1000,000 rows have been used for this project.

The dataset has been cleaned using the custom defined function where special characters, symbols, links and twitter handle are removed. The null values have been removed which left the total of 99950 usable cleaned rows.


## Lexicon Based Approach

Generally speaking, in lexicon-based approaches a piece of text message is represented as a bag of words. Following this representation of the message, sentiment values from the dictionary are assigned to all positive and negative words or phrases within the message. A combining function, such as sum or average, is applied in order to make the final prediction regarding the overall sentiment for the message. Apart from a sentiment value, the aspect of the local context of a word is usually taken into consideration, such as negation or intensification.

**SentiWordnet:**
SentiWordNet is made up of tens of thosands of words, there meanings, partof speech represented and the degree of positivity and negativity of the word, ranging from 0 to 1.These words were all derived from the WordNet 2.0 database, which is a database of english wordsand their meanings where terms are organized according tosemantic relations or meanings. Thesewords are all grouped by there synonyms into what is called synsets.So basically, SentiWordNet extends the WordNet by addition of subjectivity information ( + or - ) toevery word in the database.

**TextBlob:**
TextBlob is a Python (2 and 3) library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.
A good thing about TextBlob is that they are just like python strings. So, you can transform and play with it same like we did in python. Below, I have shown you below some basic tasks. Don’t worry about the syntax, it is just to give you an intuition about how much-related TextBlob is to Python strings.

**VADER (Valence Aware Dictionary and Sentiment Reasoner):**
It uses a list of lexical features (e.g. word) which are labeled as positive or negative according to their semantic orientation to calculate the text sentiment. Vader sentiment returns the probability of a given input sentence to be positive, negative, and neutral.
Vader is optimized for social media data and can yield good results when used with data from Twitter, Facebook, etc. As the above result shows the polarity of the word and their probabilities of being pos, neg neu, and compound.

