import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from sklearn.metrics import precision_recall_curve

### Reading CSV file
train = pd.read_csv("train.csv")
train.head()

train_eda = train.sample(frac = 0.2)
train_eda.shape

### Plotting Pie chart for data imbalance
train_0 = (train['target'] == 1)
train_1_1= sum(train_0 == True)
train_1_0= sum(train_0 == False)

labels = 'Flagged insincere', 'Normal'
sizes = [train_1_1, train_1_0]
colors = ['gold', 'lightskyblue']
explode = (0.1, 0)  # explode 1st slice
 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Pie-chart of Data distribution across classes')
plt.axis('equal')
plt.show()

### Wordcloud
all_questions = " ".join(x for x in train.question_text)
stopwords = set(STOPWORDS) 
cloud_1 = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(all_questions)

plt.imshow(cloud_1, interpolation='bilinear')
plt.axis("off")
plt.show()


### Vectorizing
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(train, test_size=0.10, random_state=42)

train_df.shape #(1175509, 3)
test_df.shape       #(130613, 3)


from sklearn.feature_extraction.text import TfidfVectorizer

vect_word = TfidfVectorizer(max_features=10000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,2),dtype=np.float32)
vect_char = TfidfVectorizer(max_features=30000, lowercase=True, analyzer='char',stop_words= 'english',ngram_range=(1,6),dtype=np.float32)

train_original = train_df
test_original = test_df
train_df = train_original.sample(frac = 0.5)
test_df = test_original.sample(frac = 0.5)

#vect_word.fit(list(train['comment_text']) + list(test['comment_text']))
tr_vect = vect_word.fit_transform(train_df['question_text'])
ts_vect = vect_word.transform(test_df['question_text'])

#vect_char.fit(list(train['comment_text']) + list(test['comment_text']))
tr_vect_char = vect_char.fit_transform(train_df['question_text'])
ts_vect_char = vect_char.transform(test_df['question_text'])

y = train_df['target']

from scipy import sparse
X = sparse.hstack([tr_vect, tr_vect_char])
x_test = sparse.hstack([ts_vect, ts_vect_char])

y_test = test_df['target']

### Classifier: Logistic Regression 
prd = np.zeros((x_test.shape[0],y.shape[1]))
cv_score =[]
for i,col in enumerate(target_col):
    lr = LogisticRegression(C=4,random_state = i)
    print('Building {} model for column:{''}'.format(i,col)) 
    lr.fit(X,y[col])
    #cv_score.append(lr.score)
    prd[:,i] = lr.predict_proba(x_test)[:,1]
    
#pred = np.zeros((x_test.shape[0],y.shape[0]))
cv_score =[]
model = LogisticRegression()
model.fit(tr_vect,y)

prd = model.predict_proba(ts_vect)[:,1]
y_score = model.decision_function(ts_vect)

#confusion matrix for test dataset 
pred =  model.predict(ts_vect)
print('\nConfusion matrix\n',confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#The reported averages include:
#1. micro average (averaging the total true positives, false negatives and false positives), 
#2. macro average (averaging the unweighted mean per label), 
#3. weighted average (averaging the support-weighted mean per label) and 


Note: in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.

#### Hence, Sensitivity = 0.36, specificity = 0.99

x_test = ts_vect
y_score = model.decision_function(x_test)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

precision, recall, _ = precision_recall_curve(y_test, y_score)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision Recall Curve: Avg precision-recall score={0:0.2f}'.format(average_precision))