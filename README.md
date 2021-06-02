# Sentiment Analysis

Sentiment analysis aims to analyze and predict the opinion, sentiment and view towards the given subject. This project tends to show mainly two concepts, comparison of different Machine learning models and how different models can be developed to show the sentiment behind the tweets as positive, negative or, neutral.
There is a room for improvement in every aspect of development. In this regard, the accuracy of all these models can always be improved. Accuracy improvement generally involves two methods, parameters tuning and dataset cleaning.

There are altogether 3 lexicon based methods and 7 machine learning models in this project. For lexicon method, SentiWordnet, VADER and Textblob are used. And for machine learning models, Logistic Regression, Decision Tree, Support Vector Machine, Random Forest, Long Short Term Memory (LSTM), Naive Bayes and XG Boost are used.

**Dataset:**

The dataset used in the project has been downloaded from [Kaggle](https://www.kaggle.com/kazanova/sentiment140?select=training.1600000.processed.noemoticon.csv). The total of 1000,000 rows have been used for this project.

The dataset has been cleaned using the custom defined function where special characters, symbols, links and twitter handle are removed. The null values have been removed which left the total of 99950 usable cleaned rows.

![alt text](https://raw.githubusercontent.com/sauravrox/sentiment-analysis/main/images/dataset.png)

*Negative > 33392, Positive > 33316, Neutral  > 33242*

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

## Machine Learning Models

Machine Learning involves training the model by certain portion of data making the model to learn from it so that it predicts the result based on what it had learned. There are many different classic as well as newly developed models that can be used acording to the requirement.  

**Logistic Regression:**

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Some of the examples of classification problems are Email spam or not spam, Online transactions Fraud or not Fraud, Tumor Malignant or Benign. Logistic regression transforms its output using the logistic sigmoid function to return a probability value [4](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148).

**Support Vector Machine(SVM)**

SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes[5](https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989).

**Decision Tree**

Decision tree is one of the most common machine learning algorithms. Used in statistics and data analysis for predictive models. A decision tree is a flowchart like structure in which each internal node represents a “test” on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes)[6](https://towardsdatascience.com/understanding-decision-trees-once-and-for-all-2d891b1be579).

**Random Forest**

Random forest methods combines multiple decision trees, trains each one on a slightly different set of observations. Splitting nodes in each tree considering a limited number of features. The final predictions of the random forest are made by averaging the predictions of each individual tree Citation Citation [[7]](https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76).

**XGBoost**

XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks. However, when it comes to small-to-medium structured/tabular data, decision tree based algorithms are considered best-in-class right now [[8]](https://www.kdnuggets.com/2019/05/xgboost-algorithm.html).


**Long Short-Term Memory (LSTM)**

Long Short-Term Memory (LSTM) networks are a modified version of recurrentneural networks, which makes it easier to remember past data in memory. The van-ishing gradient problem of RNN is resolved here. LSTM is well-suited to classify,process and predict time series given time lags of unknown duration. It trains themodel by using back-propagation [[9]](https://towardsdatascience.com/understanding-rnn-and-lstm-f7cdf6dfc14e).


**Naive Bayes**

Naive Bayes methods are a set of supervised learning algorithms based on applyingBayes’ theorem with the “naive” assumption of conditional independence betweenevery pair of features given the value of the class variable. In spite of their appar-ently over-simplified assumptions, naive Bayes classifiers have worked quite well inmany real-world situations, famously document classification and spam filtering.They require a small amount of training data to estimate the necessary parameters[10](https://scikit-learn.org/stable/modules/naive-bayes.html).