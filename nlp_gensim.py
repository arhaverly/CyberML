
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation


text = '''
Our Data Policy explains how we collect and use your personal data to determine some of the ads you see and provide all of the other services described below. You can also go to your settings at any time to review the privacy choices you have about how we use your data.
Return to top
1. The services we provide
Our mission is to give people the power to build community and bring the world closer together. To help advance this mission, we provide the Products and services described below to you:
Provide a personalized experience for you:
Your experience on Facebook is unlike anyone else's: from the posts, stories, events, ads, and other content you see in News Feed or our video platform to the Pages you follow and other features you might use, such as Trending, Marketplace, and search. We use the data we have - for example, about the connections you make, the choices and settings you select, and what you share and do on and off our Products - to personalize your experience.
Connect you with people and organizations you care about:
We help you find and connect with people, groups, businesses, organizations, and others that matter to you across the Facebook Products you use. We use the data we have to make suggestions for you and others - for example, groups to join, events to attend, Pages to follow or send a message to, shows to watch, and people you may want to become friends with. Stronger ties make for better communities, and we believe our services are most useful when people are connected to people, groups, and organizations they care about.
'''



stopwords = list(STOP_WORDS)

nlp = spacy.load('en_core_web_sm')

doc = nlp(text)

tokens = [token.text for token in doc]

# print(tokens)


punctuation = punctuation + '\n'
# print(punctuation)


word_frequencies = {}
for word in doc:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1


# print(word_frequencies)


max_frequency = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency


# print(word_frequencies)

sentence_tokens = [sent for sent in doc.sents]
# print(sentence_tokens)




sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]



# print(sentence_scores)

from heapq import nlargest

select_length = int(len(sentence_tokens)*0.3)
# print('select_length:', select_length)


summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

# print(summary)


final_summary = [word.text for word in summary]
summary = ' '.join(final_summary)


print('\nFinal Summary:')
print(summary)

print('text length:', len(text))
print('summary length:', len(summary))








