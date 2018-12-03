'''
Created on Nov 18, 2018

@author: gkeys
'''

'''
Created on Nov 17, 2018

@author: gkeys

USING:
https://stackoverflow.com/questions/47438313/how-to-predict-after-training-data-using-naive-bayes-with-python
https://www.digitalocean.com/community/tutorials/how-to-build-a-machine-learning-classifier-in-python-with-scikit-learn
https://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/
https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html
https://medium.com/tensorist/classifying-yelp-reviews-using-nltk-and-scikit-learn-c58e71e962d9
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from nltk.stem.snowball import FrenchStemmer
stemmer = FrenchStemmer()

import pandas, string, re  

filepath = "/Users/gkeys/DEV/data/ML-data/ag_news_csv/test.csv"
data = open(filepath).read()

strict_stopwords=[]
stopwordfile = open("/Users/gkeys/DEV/data/ML-data/special/stopwords-nopunct-strict.txt").read()
for line in stopwordfile.split("\n"): 
    strict_stopwords.append(line)

class_labels = ["World" , "Sports", "Business", "Sci/Tech"]


classes, titles, summaries = [], [], []
for i, line in enumerate(data.split("\n")):
    line = line.decode('utf-8') 
    content = line.split("\",\"")
    if len(content) == 3:
        classes.append(re.sub(r'[^a-z\d\s:]', " ", content[0]))
        titles.append(re.sub(r'[^a-z\d\s:]', " ", stemmer.stem(content[1].lower())))
        summaries.append(re.sub(r'[^a-z\d\s:]', " ", stemmer.stem(content[2].lower())))
    
print "number of records = ", len(titles)
print "titles\n", titles[:3]
print "summaries\n", summaries[:3]

# create a dataframe using texts and classes
trainDF = pandas.DataFrame()
trainDF['text'] = summaries
trainDF['classes'] = classes
print trainDF['text'][:3]

count_vect = CountVectorizer(stop_words=strict_stopwords)
x = count_vect.fit_transform(trainDF['text'])
encoder = LabelEncoder()
y = encoder.fit_transform(trainDF['classes'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#--- test set
nb = MultinomialNB()
nb.fit(x_train, y_train)
y_predicted = nb.predict(x_test)

print "training accuracy = {:.2f}".format(accuracy_score(y_test, y_predicted))



#--- new
new_text = "that was an expensive purchase economy"
print "\nnew text to categorize: "+new_text
newDF = pandas.DataFrame()

newDF['text'] = [new_text]

n = count_vect.transform(newDF['text'])

n_predicted = nb.predict(n)
class_index = n_predicted[0] 

probs = nb.predict_proba(n)[0]
print "prediction: " + class_labels[class_index] + " (pr={:.2f}".format(probs[class_index]) +")"
tmp = ""
for i in range(0, len(class_labels)): 
    tmp = tmp + class_labels[i] + " = {:.2f}".format(probs[i]) + "   "
print tmp

exit()

#----------------------stats
from sklearn.metrics import confusion_matrix, classification_report



print "\n0:World  1:Sport, 2:Business, 3:Sci/Tech\n"
confusion = confusion_matrix(y_test, y_predicted, labels=[0,1,2,3])
print confusion

print classification_report(y_test, y_predicted, labels=[0,1,2,3])
#plt.figure()
#hm=sn.heatmap(cm)

import numpy as np
norm_confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
print norm_confusion


import matplotlib.pyplot as plt
import itertools
plt.imshow(norm_confusion, cmap=plt.get_cmap("Blues"))
plt.title("0:World  1:Sport, 2:Business, 3:Sci/Tech")
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.xticks(np.arange(4), [0,1,2,3])
plt.yticks(np.arange(4), [0,1,2,3])
plt.tight_layout()
plt.xlabel('true label')
plt.ylabel('predicted label');
for i, j in itertools.product(range(norm_confusion.shape[0]), range(norm_confusion.shape[1])):
    plt.text(j, i, "{:0.2f}".format(norm_confusion[i, j]),
             horizontalalignment="center",
             color="white" if norm_confusion[i, j] > (norm_confusion.max() / 2) else "black")
plt.figure()


#--------------------explore
# word counts
bag_of_words = count_vect.transform(trainDF['text'])
sum_words = bag_of_words.sum(axis=0)
words_freq=[(word, sum_words[0, idx]) for word, idx in count_vect.vocabulary_.items()]
words_freq_sorted = sorted(words_freq, key = lambda x: x[1], reverse=True)
top = words_freq_sorted[:25]

import matplotlib.pyplot as plt
plt.bar([i[0] for i in top], [i[1] for i in top])
plt.title("top words")
plt.ylabel("freq")
plt.xticks(rotation='vertical')
plt.show()

count_vect_ngram = CountVectorizer(ngram_range=(2,3), stop_words=strict_stopwords)
bag_of_words = count_vect_ngram.fit_transform(trainDF['text'])
sum_words = bag_of_words.sum(axis=0)
words_freq=[(word, sum_words[0, idx]) for word, idx in count_vect_ngram.vocabulary_.items()]
words_freq_sorted = sorted(words_freq, key = lambda x: x[1], reverse=True)
top = words_freq_sorted[:25]
print top

import matplotlib.pyplot as plt
plt.bar([i[0] for i in top], [i[1] for i in top])
plt.title("top phrases")
plt.ylabel("freq")
plt.xticks(rotation='vertical')
plt.show()


#--------------- do more with this: https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-4-count-vectorizer-b3f4944e51b5
