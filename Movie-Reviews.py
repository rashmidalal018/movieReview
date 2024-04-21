# import nltk
# nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
import random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
import string
import nltk
from nltk import NaiveBayesClassifier

print("Categories are : ",movie_reviews.categories())
print("Length is : ",len(movie_reviews.fileids()))
print(movie_reviews.words(movie_reviews.fileids()[5]))

documents = []
for category in movie_reviews.categories():
    for field in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(field),category))
# print(documents[0:5])

# shuffle them 
random.shuffle(documents)

# funtion to know pos 
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

    
# w = "better"
# pos_tag([w])  # pass w as a array not a single word otherwise it will perform pos on every 

# to create list of stop words
stops = set(stopwords.words('english'))
punctuations = list(string.punctuation)
stops.update(punctuations)

# function for cleaning
lemmatizer = WordNetLemmatizer()
def clean_review(words):
    output_words = []
    for w in words:
        if w.lower() not in stops:
            pos = pos_tag([w])
            clean_word = lemmatizer.lemmatize(w, pos = get_simple_pos(pos[0][1]))
            output_words.append(clean_word.lower())
    return output_words
    
# go through documents 
documents = [(clean_review(document),category) for document, category in documents]
# print(documents[0])

# split
training_documents = documents[0:1500]
testing_documents = documents[1500:]


# building features
all_words = []
for doc in training_documents:
    all_words += doc[0]

freq = nltk.FreqDist(all_words)
common = freq.most_common(15)  #top 15 most common words
features = [i[0] for i in common]

def get_features_dict(words):
    current_features = {}
    words_set = set(words)
    for w in features:
        current_features[w] = w in words_set
    return current_features

get_features_dict(training_documents[0][0])
training_data = [(get_features_dict(doc), category) for doc, category in training_documents]
testing_data = [(get_features_dict(doc), category) for doc, category in testing_documents]

# in nltk we have in build classifier
# nltk require the format - array of tuples where each tuple has a dictinory where each tuple has dictinory 
# that have feature and feature value  and the category of that particular data point is part of 

# using naive bayes classifier
classifier =  NaiveBayesClassifier.train(training_data)
print(nltk.classify.accuracy(classifier, testing_data))

classifier.show_most_informative_features(15)   # it will show most informative features 

import pickle as pk
with open('model.pkl', 'wb') as file:
    pk.dump(classifier, file)