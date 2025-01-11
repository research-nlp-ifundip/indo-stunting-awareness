"""# Import Library"""

!pip install sastrawi
!pip install swifter
import csv
import nltk
import numpy as np
import pandas as pd
import re
import swifter
import sys
import warnings
warnings.filterwarnings("ignore")
from nltk.tokenize import RegexpTokenizer
from pickle import dump
from pickle import load
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

np.linspace(0.5, 1.5, 6)

"""# Classification BC 1"""

df=pd.read_csv('data_training01.csv', names=["date", "content", "user", "label"])
df.insert(2,'content_prepo',"")
df['content_prepo'] = df['content'].map(lambda x: x.replace("'", ''))

"""### Data Training Preprocessing"""

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    result = "<hashtag> "+ hashtag_body
    return result

def preprocess(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"\\n*","") #enter
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<emoji>") #smile
    text = re_sub(r"{}{}p+".format(eyes, nose), "<emoji>") #lolface
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<emoji>") #sadface
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<emoji>") #neutralface
    text = re_sub(r"<3","<emoji>") #heart
    text = re_sub(r"\d+", "") #number
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"[^\w\d\s\<>-]+", " ") #symbol
    text = re_sub(r"lt|gt|amp|nbsp","") #enter
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2") #ughhh -> ugh lebih dari 2 dibelakang
    text = re_sub(r"((?<=^)|(?<= )).((?=$)|(?= ))", "") #single char remove
    return text.lower()

df['content_prepo'] = df['content_prepo'].apply(preprocess)

#Tokenizing
tokenizer = RegexpTokenizer(r'\<\w+\>|\w+|[,;.!?]')
df['content_prepo'] = df['content_prepo'].apply(tokenizer.tokenize)

#create stopwords
factory = StopWordRemoverFactory()
ind_stopwords = factory.get_stop_words()

# append additional stopword
ind_stopwords.extend(["yg", "dg", "rt", "dgn", "dr", "sm", "ny", "d", 'klo',
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 'udh',
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'kok',
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'heh',
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'ky',
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', 'td', 'pd', 'kan', 'lo', 'lah'])

# convert list to dictionary
ind_stopwords = set(ind_stopwords)

#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in ind_stopwords]

df['content_prepo'] = df['content_prepo'].apply(stopwords_removal)

import swifter

# create stemmer
factory = StemmerFactory()
ind_stemmer = factory.create_stemmer()
stem_except = ["<user>", "<hashtag>", "<url>", "<emoji>"]
def stemming(texts):
    return [ind_stemmer.stem(doc) if doc not in stem_except else doc for doc in texts]

df['content_prepo'] = df['content_prepo'].swifter.apply(stemming)

df.to_csv("datatraining01_preprocessing.csv", index=False, header=False) # Change Here

"""###Data Testing Preprocessing"""

df=pd.read_csv('data_testing01.csv', names=["date", "content", "user", "label"])
df.insert(2,'content_prepo',"")
df['content_prepo'] = df['content'].map(lambda x: x.replace("'", ''))

# Cleaning
df['content_prepo'] = df['content_prepo'].apply(preprocess)

#Tokenizing
df['content_prepo'] = df['content_prepo'].apply(tokenizer.tokenize)

# Stopword Removal
df['content_prepo'] = df['content_prepo'].apply(stopwords_removal)

# Stemming
df['content_prepo'] = df['content_prepo'].swifter.apply(stemming)

df.to_csv("datatesting01_preprocessing.csv", index=False, header=False)

"""###TF-IDF"""

df=pd.read_csv('datatraining01_preprocessing.csv', names=["date", "content", "content_prepro", "user", "label"])
df_test=pd.read_csv('datatesting01_preprocessing.csv', names=["date", "content", "content_prepro", "user", "label"])

tdm_transformer = CountVectorizer().fit(df['content_prepro'])
df_tdm = tdm_transformer.transform(df['content_prepro'])

tfidf_transformer = TfidfTransformer().fit(df_tdm)
df_tfidf = tfidf_transformer.transform(df_tdm)

"""###Model Training"""

# create a pipeline to convert the data into tfidf form
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer())])

# Specify X and y
extract = pipeline.fit(df.content_prepro)
X_train = extract.transform(df.content_prepro)
y_train = df.label
X_test = extract.transform(df_test.content_prepro)
y_test = df_test.label

"""Naive Bayes"""

model = MultinomialNB()
params = {'alpha': np.linspace(0.5, 1.5, 6), 'fit_prior': [True, False]}
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
model_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv)
result = model_cv.fit(X_train, y_train)
print('\n\nNAIVE BAYES')
print('Best Score = %s' % result.best_score_)
print('Std = %s' % result.cv_results_['std_test_score'][result.best_index_])
print('Best Hyperparameter = %s' % result.best_params_)

# Testing
bestModel = result.best_estimator_
print('Testing')
print("R2: {:.4f}".format(bestModel.score(X_test, y_test)))
print()
y_true, y_pred = y_test, result.predict(X_test)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

"""Random Forest"""

params = {'n_estimators': [10, 50, 100],
          'max_features': ['auto', 'sqrt', 'log2'],
          'max_depth' : [6,7,8],
          'criterion' :['gini', 'entropy']}

model = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
model_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv)
result = model_cv.fit(X_train, y_train)
print('\n\n RANDOM FOREST')
print('Best Score = %s' % result.best_score_)
print('Std = %s' % result.cv_results_['std_test_score'][result.best_index_])
print('Best Hyperparameter = %s' % result.best_params_)

# Testing
bestModel = result.best_estimator_
print('Testing')
print("R2: {:.4f}".format(bestModel.score(X_test, y_test)))
print()
y_true, y_pred = y_test, result.predict(X_test)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

"""LR"""

params = {'solver':['newton-cg', 'lbfgs', 'liblinear'],
          'penalty':['none', 'l1', 'l2', 'elasticnet'],
          'C': [0.1,1,10,100,1000],}
model = LogisticRegression()
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
model_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv)
result = model_cv.fit(X_train, y_train)
print('\n\n LOGISTIC REGRESSION')
print('Best Score = %s' % result.best_score_)
print('Std = %s' % result.cv_results_['std_test_score'][result.best_index_])
print('Best Hyperparameter = %s' % result.best_params_)

# Testing
bestModel = result.best_estimator_
print('Testing')
print("R2: {:.4f}".format(bestModel.score(X_test, y_test)))
print()
y_true, y_pred = y_test, result.predict(X_test)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

"""SVM"""

params = {'C': [0.1,1,10,100],
          'gamma':[0.01,0.1,1,10, 100],
          'degree':[0,1,2,3,4,5,6],
          'kernel':['linear','rbf','poly','sigmoid']}
model = SVC()
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
model_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv)
result = model_cv.fit(X_train, y_train)
print('\n\n SVM')
print('Best Score = %s' % result.best_score_)
print('Std = %s' % result.cv_results_['std_test_score'][result.best_index_])
print('Best Hyperparameter = %s' % result.best_params_)

# Testing
bestModel = result.best_estimator_
print('Testing')
print("R2: {:.4f}".format(bestModel.score(X_test, y_test)))
print()
y_true, y_pred = y_test, result.predict(X_test)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

"""Save Model"""

final_model = LogisticRegression(C=1000, penalty='l2', solver='newton-cg')
final_model.fit(X_train,y_train)

# save model
print('Saving Model...')
dump(final_model, open('modelclassfication01.pkl', 'wb'))
print('Saving Pipeline...')
dump(extract, open('modelpipeline01.pkl', 'wb'))

"""# Classification BC 2"""

print('Classification BC 2')
df = pd.read_csv('datatraining01_preprocessing.csv', names=["date", "content", "content_prepro", "user", "label"])
df = df.drop(["label"], axis=1)
data = pd.read_csv('data_training.csv', names=["date", "content", "user", "label"])
df_train = pd.concat([df, data["label"]], axis=1)
df_train = df_train.loc[df_train['label'].isin([1,2])]

df = pd.read_csv('datatesting01_preprocessing.csv', names=["date", "content", "content_prepro", "user", "label"])
df = df.drop(["label"], axis=1)
data = pd.read_csv('data_testing.csv', names=["date", "content", "user", "label"])
df_test = pd.concat([df, data["label"]], axis=1)
df_test = df_test.loc[df_test['label'].isin([1,2])]

"""###TF-IDF"""

from sklearn.feature_extraction.text import CountVectorizer
tdm_transformer = CountVectorizer().fit(df_train['content_prepro'])
df_tdm = tdm_transformer.transform(df_train['content_prepro'])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(df_tdm)
df_tfidf = tfidf_transformer.transform(df_tdm)

# create a pipeline to convert the data into tfidf form
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer())])

# Specify X and y
extract = pipeline.fit(df_train.content_prepro)
X_train = extract.transform(df_train.content_prepro)
y_train = df_train.label
X_test = extract.transform(df_test.content_prepro)
y_test = df_test.label

"""###Model Training

Naive Bayes
"""

model = MultinomialNB()
params = {'alpha': np.linspace(0.5, 1.5, 6), 'fit_prior': [True, False]}
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
model_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv)
result = model_cv.fit(X_train, y_train)

print('\n\n NAIVE BAYES')
print('Best Score = %s' % result.best_score_)
print('Std = %s' % result.cv_results_['std_test_score'][result.best_index_])
print('Best Hyperparameter = %s' % result.best_params_)

# Testing
bestModel = result.best_estimator_
print('Testing')
print("R2: {:.4f}".format(bestModel.score(X_test, y_test)))
print()
y_true, y_pred = y_test, result.predict(X_test)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

"""Random Forest"""

params = {'n_estimators': [10, 50, 100],
          'max_features': ['auto', 'sqrt', 'log2'],
          'max_depth' : [6,7,8],
          'criterion' :['gini', 'entropy']}

model = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
model_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv)
result = model_cv.fit(X_train, y_train)
print('\n\n RANDOM FOREST')
print('Best Score = %s' % result.best_score_)
print('Std = %s' % result.cv_results_['std_test_score'][result.best_index_])
print('Best Hyperparameter = %s' % result.best_params_)

# Testing
bestModel = result.best_estimator_
print('Testing')
print("R2: {:.4f}".format(bestModel.score(X_test, y_test)))
print()
y_true, y_pred = y_test, result.predict(X_test)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

"""LR"""

params = {'solver':['newton-cg', 'lbfgs', 'liblinear'],
          'penalty':['none', 'l1', 'l2', 'elasticnet'],
          'C': [0.1,1,10,100,1000],}
model = LogisticRegression()
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
model_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv)
result = model_cv.fit(X_train, y_train)
print('\n\n LOGISTIC REGRESSION')
print('Best Score = %s' % result.best_score_)
print('Std = %s' % result.cv_results_['std_test_score'][result.best_index_])
print('Best Hyperparameter = %s' % result.best_params_)

# Testing
bestModel = result.best_estimator_
print('Testing')
print("R2: {:.4f}".format(bestModel.score(X_test, y_test)))
print()
y_true, y_pred = y_test, result.predict(X_test)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

"""SVM"""

params = {'C': [0.1,1,10,100],
          'gamma':[0.01,0.1,1,10, 100],
          'degree':[0,1,2,3,4,5,6],
          'kernel':['linear','rbf','poly','sigmoid']}
model = SVC()
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
model_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv)
result = model_cv.fit(X_train, y_train)
print('\n\n SVM')
print('Best Score = %s' % result.best_score_)
print('Std = %s' % result.cv_results_['std_test_score'][result.best_index_])
print('Best Hyperparameter = %s' % result.best_params_)

# Testing
bestModel = result.best_estimator_
print('Testing')
print("R2: {:.4f}".format(bestModel.score(X_test, y_test)))
print()
y_true, y_pred = y_test, result.predict(X_test)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

"""Save Model"""

final_model = MultinomialNB(alpha=0.5, fit_prior=True)
final_model.fit(X_train,y_train)

# save model
print('Saving Model...')
dump(final_model, open('modelclassfication02.pkl', 'wb'))
print('Saving Pipeline...')
dump(extract, open('modelpipeline02.pkl', 'wb'))

"""#Hierarchial Classification"""

df = pd.read_csv('datatesting01_preprocessing.csv', names=["date", "content", "content_prepro", "user", "label"])
df = df.drop(["label"], axis=1)
data = pd.read_csv('data_testing.csv', names=["date", "content", "user", "label"])
df_test = pd.concat([df, data["label"]], axis=1)

#save GT to list
ground_truth = df_test["label"].tolist()
#save Data to list
data = df_test["content_prepro"].tolist()
# Create Prediction List
pred_list = [0] * 100

# Load BC 1
pipeline1 = load(open('modelpipeline01.pkl', 'rb'))
model1 = load(open('modelclassification01.pkl', 'rb'))

# Load BC 2
pipeline2 = load(open('modelpipeline02.pkl', 'rb'))
model2 = load(open('modelclassification02.pkl', 'rb'))

for dt in data:
  #BC 1
  X_bc1 = pipeline1.transform([dt])
  y_bc1 = model1.predict(X_bc1)
  if y_bc1 == 0: #unrelated
      pred_list[data.index(dt)] = y_bc1
  else: #related
      X_bc2 = pipeline2.transform([dt])
      y_bc2 = model2.predict(X_bc2)
      pred_list[data.index(dt)] = y_bc2

print('Confusion Matrix')
print(confusion_matrix(ground_truth, pred_list))
print('Classification Report')
print(classification_report(ground_truth, pred_list))

"""# Save Data Test Prediction

BC 1
"""

df = pd.read_csv('datatesting01_preprocessing.csv', names=["date", "content", "content_prepro", "user", "label"])
df = df.drop(["label"], axis=1)
data = pd.read_csv('data_testing01.csv', names=["date", "content", "user", "label"])
df_test = pd.concat([df, data["label"]], axis=1)

# Load Pipeline
pipeline = load(open('modelpipeline01.pkl', 'rb'))

# Specify X
X = pipeline.transform(df_test.content_prepro)

# Load Model
final_model = load(open('modelclassification01.pkl', 'rb'))

# Predict
predictions = final_model.predict(X)

# Dataframe
y_predicts = pd.DataFrame(data = predictions, columns = ['preds'], index = df_test.index.copy())
df_out = pd.merge(df_test, y_predicts, how = 'left', left_index = True, right_index = True)

df_out.to_csv("datatesting_prediction1.csv", index=False, header=False)

"""BC 2"""

df = pd.read_csv('datatesting01_preprocessing.csv', names=["date", "content", "content_prepro", "user", "label"])
df = df.drop(["label"], axis=1)
data = pd.read_csv('data_testing.csv', names=["date", "content", "user", "label"])
df_test = pd.concat([df, data["label"]], axis=1)
df_test = df_test.loc[df_test['label'].isin([1,2])]

# Load Pipeline
pipeline = load(open('modelpipeline02.pkl', 'rb'))

# Specify X
X = pipeline.transform(df_test.content_prepro)

# Load Model
final_model = load(open('modelclassification02.pkl', 'rb'))

# Predict
predictions = final_model.predict(X)

# Dataframe
y_predicts = pd.DataFrame(data = predictions, columns = ['preds'], index = df_test.index.copy())
df_out = pd.merge(df_test, y_predicts, how = 'left', left_index = True, right_index = True)

df_out.to_csv("datatesting_prediction2.csv", index=False, header=False)