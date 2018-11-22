import glob2
import glob
import re
import collections
import numpy as np
import os

#Path of all the positive/negative reviews
main_path = glob2.glob('aclImdb/test/*/**.txt')  # list of all .txt files in both pos and neg

#Concatenation of all the positive/negative reviews
with open('pos_rev.txt', 'w') as f, open('neg_rev.txt', 'w') as g:
    for file in main_path:
        with open(file) as infile:
          if os.path.basename(os.path.dirname(file)) == 'pos':
            f.write(infile.read() + '\n')
          else:
            g.write(infile.read() + '\n')

#Cleaning up the pos_rev.txt and neg_rev.txt for MORE accuracy
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews

pos_rev_clean = []  #create an empty list
for line in open('pos_rev.txt', 'r'):
    pos_rev_clean.append(line.strip())

neg_rev_clean = []  #create an empty list
for line in open('neg_rev.txt', 'r'):
    neg_rev_clean.append(line.strip())

#List of words in each review class
wordListPos = re.sub("[^\w]", " ", str(preprocess_reviews(pos_rev_clean))).split()
wordListNeg = re.sub("[^\w]", " ", str(preprocess_reviews(neg_rev_clean))).split()

#Counting words in positive reviews
cntPos = collections.Counter()
for word in wordListPos:
  cntPos[word] += 1
#Counting words in negative reviews
cntNeg = collections.Counter()
for word in wordListNeg:
  cntNeg[word] += 1
#Total words in each review
word_count_Pos = len(re.findall(r'\w+', str(preprocess_reviews(pos_rev_clean))))
word_count_Neg = len(re.findall(r'\w+', str(preprocess_reviews(neg_rev_clean))))
#Half of total number of reviews across all input dataset
tot_rev = len(list(glob.iglob("aclImdb/test/pos/*.txt", recursive=True)))

#Training the model based on naive Bayes theory
#Test dataset chosen from both positive and negative reviews
accuracy_pos_unit = 0
accuracy_neg_unit = 0
for file in main_path:
    test_review = []
    for line in open(file):
       test_review.append(line.strip())
    wordListReview = re.sub("[^\w]", " ", str(preprocess_reviews(test_review))).split()
    ReviewWordCount = len(wordListReview)
    cntReview = collections.Counter()
#The size of vocabulary
    Voc = 10000
#Looping over the words in a test review
    for word in wordListReview:
      cntReview[word] += 1
    ReviewKeys = cntReview.keys()
#Regularized naive Bayes rule to calcualte the probability of the review with a positive tone
    LoglikelihoodPos = 0
    for word in ReviewKeys:
        LoglikelihoodPos = LoglikelihoodPos + cntReview[word] * np.log(
            (float(cntPos[word]) + 1) / (word_count_Pos + Voc))
#Regularized naive Bayes rule to calcualte the probability of the review with a negative tone
    LoglikelihoodNeg = 0
    for word in ReviewKeys:
        LoglikelihoodNeg = LoglikelihoodNeg + cntReview[word] * np.log(
            (float(cntNeg[word]) + 1) / (word_count_Neg + Voc))
#Comparing against the correct label and computing the accuracy
    if (LoglikelihoodPos > LoglikelihoodNeg and os.path.basename(os.path.dirname(file)) == 'pos'):
        #print("Naive Bayes says the review is positive!")
        accuracy_pos_unit = accuracy_pos_unit + 1
     #elif LoglikelihoodPos == LoglikelihoodNeg:
        #print("Naive Bayes says the review is half-half!")
    elif (LoglikelihoodPos < LoglikelihoodNeg and os.path.basename(os.path.dirname(file)) == 'neg'):
        #print("Naive Bayes says the review is negative!")
        accuracy_neg_unit = accuracy_neg_unit + 1
    elif LoglikelihoodPos > LoglikelihoodNeg and os.path.basename(os.path.dirname(file)) == 'neg':
        accuracy_pos_unit = accuracy_pos_unit - 1
    elif LoglikelihoodPos < LoglikelihoodNeg and os.path.basename(os.path.dirname(file)) == 'pos':
        accuracy_neg_unit = accuracy_neg_unit - 1
    else:
        continue
print("Total error is {}".format(100*(1-(accuracy_neg_unit+accuracy_pos_unit)/(2*tot_rev))) + '%')      #total error rate
