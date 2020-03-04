

# Multinomial Bayes algorithm for spam filter for SMS messasges (data set obatiend from: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection


import pandas as pd 
import numpy as np 
import re

raw_SMS_data = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['Label', 'SMS'])

print(raw_SMS_data.shape)
raw_SMS_data.head()
print(raw_SMS_data['Label'].value_counts())


# Training and validation data sets
# sampling the entire data set to genenrate training and validation data 
# sets 

rand_SMS_data = raw_SMS_data.sample(frac=1, random_state=1)
training_SMS_data = rand_SMS_data.iloc[0:4458,:].reset_index(drop=True)
validation_SMS_data = rand_SMS_data.iloc[4458:,:].reset_index(drop=True)

print(training_SMS_data['Label'].value_counts(normalize = True))
print(validation_SMS_data['Label'].value_counts(normalize = True))

# Performing some data cleaning to get the SMS data in a format 
# that we can compute the probabilities of spam 

training_SMS_data['SMS'] = training_SMS_data['SMS'].str.replace('\W', ' ')
training_SMS_data['SMS'] = training_SMS_data['SMS'].str.lower()
training_SMS_data.head()

training_SMS_data['SMS'] = training_SMS_data['SMS'].str.split()
vocabulary = [] 

for sms_iter in training_SMS_data['SMS']: 
    for word in sms_iter: 
        vocabulary.append(word)
        
vocabulary = list(set(vocabulary))
print(len(vocabulary))


word_counts_per_sms = {} 
word_counts_per_sms = {unique_word: [0] * len(training_SMS_data['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(training_SMS_data['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1

    
word_counts = pd.DataFrame(word_counts_per_sms)
training_set_clean = pd.concat([training_SMS_data, word_counts], axis=1)
print(training_set_clean.head())


# In[6]:


spam_traning_sms = training_set_clean[training_set_clean['Label'] == 'spam']
ham_traning_sms = training_set_clean[training_set_clean['Label'] == 'ham']

p_spam = len(spam_traning_sms)/len(training_set_clean)
p_ham = len(ham_traning_sms) /len(training_set_clean)
# print(p_spam,p_ham)

n_words_per_spam_message = spam_traning_sms['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

n_words_per_ham_message = ham_traning_sms['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

n_vocabulary = len(vocabulary)
# print(n_spam,n_ham,N_vocabulary)

alpha = 1

print(spam_traning_sms.head())


param_w_spam  = {word: 0 for word in vocabulary} 
param_w_ham = {word: 0 for word in vocabulary} 

for word in vocabulary: 
    n_w_given_spam = spam_traning_sms[word].sum()  
    p_w_given_spam = (n_w_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
    param_w_spam[word] = p_w_given_spam
    
    n_w_given_ham = ham_traning_sms[word].sum()
    p_w_given_ham = (n_w_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
    param_w_ham[word] = p_w_given_ham

def classify(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message: 
        if word in param_w_spam: 
            p_spam_given_message *= param_w_spam[word]
        
        if word in param_w_ham: 
            p_ham_given_message *= param_w_ham[word]
              

    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)

    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')
        
        

classify('WINNER!! This is the secret code to unlock the money: C3421.')
classify("Sounds good, Tom, then see u there")  
    

def classify_test_set(message):    
#     '''
#     message: a string
#     '''
    
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in param_w_spam:
            p_spam_given_message *= param_w_spam[word]
            
        if word in param_w_ham:
            p_ham_given_message *= param_w_ham[word]
    
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message > p_ham_given_message:
        return 'spam'
    else:
        return 'needs human classification'


correct = 0 
total = len(validation_SMS_data)

validation_SMS_data['predicted'] = validation_SMS_data['SMS'].apply(classify_test_set)


# In[30]:


for row in validation_SMS_data.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1
        
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)


# In[ ]:




