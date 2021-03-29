from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
import matplotlib.pyplot as plt
import sys
import os

# Initialising Objects
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()
cv = CountVectorizer()
mnb = MultinomialNB()
bnb = BernoulliNB()
gnb = GaussianNB()


# Creating a function for cleaning, tokenizing, removing stopwords and stemming the reviews
def getProcessedText(reviews):
    reviews = reviews.lower()

    # Cleaning reviews data that is removing the <br /><br /> from the text
    reviews = reviews.replace("<br /><br />", " ")

    # Tokenize
    tokens = tokenizer.tokenize(reviews)

    # Stopword removal
    new_tokens = [token for token in tokens if token not in en_stopwords]

    # Stemming
    stemmed_tokens = [ps.stem(token) for token in new_tokens]

    # Joining the above list of words
    cleaned_review = ' '.join(stemmed_tokens)

    return cleaned_review


X = [
    "This was an awesome movie",
    "Great movie! I like it a lot",
    "Happy ending! awesome acting by the hero",
    "loved it! truly great",
    "bad not upto the mark",
    "could have been better",
    "Surely a Disappointing movie"
]
Y = [1, 1, 1, 1, 0, 0, 0]
X_test = [
    "I was happy and I loved the acting in the movie",
    "The movie I saw was bad",
    "The movie I saw was not bad",
    "The movie I saw was not good"
]

X_clean = [getProcessedText(review) for review in X]
X_test_clean = [getProcessedText(review) for review in X_test]

# Vectorization of the data
X_vec = cv.fit_transform(X_clean).toarray()
print(X_vec)
print(X_vec.shape)
print(cv.vocabulary_)
X_test_vec = cv.transform(X_test_clean).toarray()
print(X_test_vec)
print(X_test_vec.shape)

# Multinomial Naive Bayes
mnb.fit(X_vec, Y)
print(mnb.predict(X_test_vec))
# Getting the posterior probability for each class
print(mnb.predict_proba(X_test_vec))
# Finding the score
print(mnb.score(X_vec, Y))

# Mutlivariate Bernoulli event model
bnb.fit(X_vec, Y)
print(bnb.predict(X_test_vec))
# Getting the posterior probability for each class
print(bnb.predict_proba(X_test_vec))
print(bnb.score(X_vec, Y))

# Generate Confusion Matrix
cfn = confusion_matrix([1, 0, 1, 0], bnb.predict(X_test_vec))
print(cfn)
# Plotting confusion matrix
plot_confusion_matrix(bnb, X_test_vec, [1, 0, 1, 0], normalize=None)
plt.show()
# You can also use confusion matrix for cases with more than one classes.

# We are getting 100% score in both because of the small data that we are considering .
# This is a case of overfitting

# You can add bigram/trigram/ngram to make prediction more accurate . Here we have used a smaller data
# for training but you can use imdb_X_train file if you have the time to let it train
