import os
import nltk
import nltk.corpus



# from nltk.corpus import brown

# word_list = brown.words()

# words = ' '.join(list(word_list))


from nltk.tokenize import word_tokenize

# brown_tokens = word_tokenize(words)

# print(brown_tokens)


# from nltk.probability import FreqDist
# fdist = FreqDist()

# for word in brown_tokens:
#     fdist[word.lower()] += 1

# # print(len(fdist))


# fdist_top10 = fdist.most_common(10)
# print(fdist_top10)


from nltk.util import bigrams, trigrams, ngrams


quote = 'The quick brown fox jumped over the lazy brown dog.'

quote_tokens = word_tokenize(quote)

quote_bigrams = list(nltk.bigrams(quote_tokens))

print(quote_bigrams)


from nltk.stem import PorterStemmer
pst = PorterStemmer()


print(pst.stem("given"))



from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
word_lem = WordNetLemmatizer()

print(word_lem.lemmatize('fishes'))


from nltk.corpus import stopwords

# print(stopwords.words('english'))

import re
punctuation = re.compile(r'[-.?!,:;()|0-9]')

post_punctuation=[]

for words in quote_tokens:
    word = punctuation.sub("",words)
    if len(word)>0:
        post_punctuation.append(word)

print(post_punctuation)

sent = 'Timothy is a natural when it comes to drawing'
sent_tokens = word_tokenize(sent)

for token in sent_tokens:
    print(nltk.pos_tag([token]))



from nltk import ne_chunk
NE_sent = 'The US President stays in the White House'

NE_tokens = word_tokenize(NE_sent)
NE_tags = nltk.pos_tag(NE_tokens)
NE_NER = ne_chunk(NE_tags)
print(NE_NER)



new = 'the big cat ate the little mouse who was after fresh cheese'

new_tokens = nltk.pos_tag(word_tokenize(new))
print(new_tokens)

grammar_np = r'NP: {<DT>?<JJ>*<NN>}'
chunk_parser = nltk.RegexpParser(grammar_np)
chunk_result = chunk_parser.parse(new_tokens)
print(chunk_result)


