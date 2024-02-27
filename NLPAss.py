# -*- coding: utf-8 -*-
"""


@author: Shraddha Ghuge
"""


'''text=
Look for data to help you address the question. Governments are good
sources because data from public research is often freely available. Good
places to start include http://www.data.gov/, and http://www.science.
gov/, and in the United Kingdom, http://data.gov.uk/.
Two of my favorite data sets are the General Social Survey at http://www3.norc.org/gss+website/, 
and the European Social Survey at http://www.europeansocialsurvey.org/.

Using tokenization , Extract all money transaction from below sentence 
along with currency. Output should be,
wo $
500 €
2.
1.Use stemming for following docs
doc = nlp("Mando talked for 3 hours although talking isn't his thing")
doc = nlp("eating eats eat ate adjustable rafting ability meeting better")

2.  convert these list of words into base form using Stemming and Lemmatization and observe the transformations 
 
#using stemming in nltk
lst_words = ['running', 'painting', 'walking', 'dressing', 'likely', 'children', 'whom', 'good', 'ate', 'fishing']
#using lemmatization in spacy
doc = nlp("running painting walking dressing likely children who good ate fishing")


3.convert the given text into it's base form using both stemming and lemmatization
text = """Latha is very multi talented girl.She is good at many skills like dancing, running, singing, playing.She also likes eating Pav Bhagi. she has a 
habit of fishing and swimming too.Besides all this, she is a wonderful at cooking too.
"""
3.You are parsing a news story from cnbc.com. News story is stores in news_story.txt which is on whatsapp. You need to, 
Extract all NOUN tokens from this story. You will have to read the file in python first to collect all the text and then extract NOUNs in a python list
Extract all numbers (NUM POS type) in a python list
Print a count of all POS tags in this story
'''



'''
1.Use stemming for following docs
doc = nlp("Mando talked for 3 hours although talking isn't his thing")
doc = nlp("eating eats eat ate adjustable rafting ability meeting better")

'''



#1.1
#import nltk library
import nltk
#doc = nlp("Mando talked for 3 hours although talking isn't his thing")
sentence="Mando talked for 3 hours although talking isn't his thing"

#here we are perform stemming using PorterStemmer
from nltk.stem.porter import PorterStemmer

ps_stemmer=PorterStemmer()
#PorterStemmer it is class
#ps_stemmer is object of PorterStemmer
words=sentence.split()
" ".join([ps_stemmer.stem(wd) for wd in words])


#1.2
#import nltk library
import nltk
sentence="eating eats eat ate adjustable rafting ability meeting better"

#Perform stemming using PorterStemmer
from nltk.stem.porter import PorterStemmer
#split sentence into list of words 
words=sentence.split()
#perform stemming of each word and join them again
" ".join([ps_stemmer.stem(wd) for wd in words])

####
#extra
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
lst_words = ['running', 'painting', 'walking', 'dressing', 'likely', 'children', 'whom', 'good', 'ate', 'fishing']
for word in lst_words:
    print(word,"|",stemmer.stem(word))


#######################################3
'''2.  convert these list of words into base form 
using Stemming and Lemmatization and observe the 
transformations 
 
#using stemming in nltk
lst_words = ['running', 'painting', 'walking', 'dressing', 'likely', 'children', 'whom', 'good', 'ate', 'fishing']
#using lemmatization in spacy
doc = nlp("running painting walking dressing likely children who good ate fishing")
'''

#2.1

#using stemming in nltk
#lst_words = ['running', 'painting', 'walking', 'dressing', 'likely', 'children', 'whom', 'good', 'ate', 'fishing']
from nltk.stem import PorterStemmer
#stemming using PorterStemmer
stemmer=PorterStemmer()
lst_words = ['running', 'painting', 'walking', 'dressing', 'likely', 'children', 'whom', 'good', 'ate', 'fishing']
for word in lst_words:
    print(word,"|",stemmer.stem(word))


#2.2
#using lemmatization in spacy
#doc = nlp("running painting walking dressing likely children who good ate fishing")


import spacy
nlp=spacy.load("en_core_web_sm")
#error :-'en_core_web_sm'. It doesn't seem to be a Python package 
doc = nlp("running painting walking dressing likely children who good ate fishing")
for token in doc:
    print(token,"|",token.lemma_)

#or

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
lemmatizer=WordNetLemmatizer()
sentence="running painting walking dressing likely children who good ate fishing"
words=word_tokenize(sentence)
" ".join([lemmatizer.lemmatize(word) for word in words])

#####################################

'''
3.convert the given text into it's base form using 
both stemming and lemmatization
text = """Latha is very multi talented girl.She is 
good at many skills like dancing, running, singing, 
playing.She also likes eating Pav Bhagi. she has a 
habit of fishing and swimming too.Besides all this,
 she is a wonderful at cooking too.
"""
'''
#stemming

import nltk
from nltk.stem.porter import PorterStemmer
text = """Latha is very multi talented girl. She is 
good at many skills like dancing , running , singing , 
playing .She also likes eating Pav Bhagi. she has a 
habit of fishing and swimming too. Besides all this,
 she is a wonderful at cooking too."""
ps_stemmer=PorterStemmer()
words=text.split()
" ".join([ps_stemmer.stem(wd) for wd in words])


#lemmatization
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
lemmatizer=WordNetLemmatizer()
text = """Latha is very multi talented girl.She is 
good at many skills like dancing, running, singing, 
playing.She also likes eating Pav Bhagi. she has a 
habit of fishing and swimming too.Besides all this,
 she is a wonderful at cooking too."""
words=word_tokenize(text)
" ".join([lemmatizer.lemmatize(word) for word in words])



"""
3.You are parsing a news story from cnbc.com. 
News story is stores in news_story.txt which is on 
whatsapp. You need to, 
Extract all NOUN tokens from this story. You will 
have to read the file in python first to collect all 
the text and then extract NOUNs in a python list
Extract all numbers (NUM POS type) in a python list
Print a count of all POS tags in this story
'''
"""

#read file
with open('C:/Assignments/NLP/news_story.txt','r') as file_object:
    raw_data=file_object.read()
raw_data

from nltk.tokenize import word_tokenize
result=word_tokenize(raw_data)
result
final=nltk.pos_tag(result)
final
print(final)
nltk.download('tagsets')
nltk.help.upenn_tagset('DT')
nltk.help.upenn_tagset('NN')
nltk.help.upenn_tagset('NNP')

for i in range(0,len(final)):
    if(final[i][1]=='NNP'):
        print(final[i][0])



'''
key_products=raw_data[10:19]
key_products
key_products=[row.strip().replace(')','').split('(') for row in key_products]
key_products
'''





