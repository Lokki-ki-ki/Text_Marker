import pandas as pd
import numpy as np
import nltk
import string
import language_tool_python
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
from textstat.textstat import textstatistics
from textacy import preprocessing
import spacy
import collections

class DataEng:
    def __init__(self, input) -> None:
        # self.data = pd.DataFrame
        self.input = input
        self.words = nltk.word_tokenize(input)
        self.sent = nltk.sent_tokenize(input)
        


    def Engineering(self):
        num_of_words = len(self.input.split())
        stopwords_freq = self.stopword_count() / num_of_words
        av_word_per_sen = num_of_words / len(self.sent)
        punctuations = self.punctuation()
        ARI = self.ARI(num_of_words)

        # Frequency of diff words
        tagging = self.POS_Tagging()
        freq_of_verb = tagging[0] / num_of_words
        freq_of_adj = tagging[1] / num_of_words
        freq_of_adv = tagging[2] / num_of_words
        freq_of_distinct_adj = tagging[6] / num_of_words
        freq_of_distinct_adv = tagging[7] / num_of_words
        freq_of_wrong_words = self.wrongwords()
        freq_of_noun = tagging[5] / num_of_words
        freq_of_transition = tagging[4] / num_of_words
        freq_of_pronoun = tagging[3] / num_of_words
        noun_to_adj = freq_of_adj / freq_of_noun
        verb_to_adv = freq_of_distinct_adv / freq_of_verb

        # Sentiments Score
        sentiments = self.sentiment_Score()
        sentiment_compound = sentiments[0]
        sentiment_positive = sentiments[1]
        sentiment_negative = sentiments[2]

        # Grammar
        num_of_grammar_errors = self.grammarerrors()
        corrected_text = self.fixgrammar()
        num_of_short_forms = self.shortforms(corrected_text)
        Incorrect_form_ratio = (num_of_grammar_errors + num_of_short_forms) / num_of_short_forms

        # Readability
        flesch_reading_ease = textstat.flesch_reading_ease(preprocessing.normalize.whitespace(self.input))
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(preprocessing.normalize.whitespace(self.input))
        dale_chall_readability_score = textstat.dale_chall_readability_score(preprocessing.normalize.whitespace(self.input))
        text_standard = textstat.text_standard(preprocessing.normalize.whitespace(self.input), float_output=True)
        mcalpine_eflaw = 3.3
        
        # mcalpine_eflaw = 25 - (textstat.mcalpine_eflaw(preprocessing.normalize.whitespace(self.input)))
        # module 'textstat' has no attribute 'mcalpine_eflaw'
       
        # freq_diff_words = self.difficult_words() / len(self.words)
        # [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.
        number_of_diff_words = 43.29
        freq_diff_words = 0.126
        ttr = self.ttrscore(corrected_text)
        coherence_score = 0.37 # hv't edit
        lexrank_avg_min_diff = 0.48
        lexrank_interquartile = 0.22
        phrase_diversity = 0.155
        sentence_complexity = 1.5


        output = [[num_of_words, stopwords_freq, av_word_per_sen, punctuations, ARI,
        freq_of_verb, freq_of_adj, freq_of_adv, freq_of_distinct_adj, freq_of_distinct_adv,
        sentence_complexity, freq_of_wrong_words, sentiment_compound, sentiment_positive,
        sentiment_negative, num_of_grammar_errors,
        num_of_short_forms, Incorrect_form_ratio, flesch_reading_ease, flesch_kincaid_grade,
        dale_chall_readability_score, text_standard, mcalpine_eflaw, number_of_diff_words,
        freq_diff_words, ttr, coherence_score,
        lexrank_avg_min_diff, lexrank_interquartile, freq_of_noun, freq_of_transition, freq_of_pronoun,
        noun_to_adj, verb_to_adv, phrase_diversity]]

        return pd.DataFrame(output)

### Helper functions
    #TTR
    def ttrscore(self, sample):
        sample_words=sample.split()
        n_words = len(sample_words)

        # remove all punctuations
        for i in range(n_words):
            for c in string.punctuation:
                sample_words[i] = sample_words[i].replace(c,'')

        # remove empty words
        sample_words = list(filter(None, sample_words))
        n_words = len(sample)

        # count each word
        word_count = collections.Counter(sample_words)

        # get the sorted list of unique words
        unique_words = list(word_count.keys())
        unique_words.sort()

        n_unique = len(unique_words)
        ttr = len(word_count)/float(n_words)
        return ttr

    #Difficult Word Extraction
    def syllables_count(self, word):
        return textstatistics().syllable_count(word)

    def break_sentences(self, text):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        return list(doc.sents)

    def difficult_words(self): # python -m spacy download en 
        nlp = spacy.load('en_core_web_sm')
        # doc = nlp(self.input)
        # Find all words in the text
        words = []
        sentences = self.break_sentences(self.input)
        for sentence in sentences:
            words += [str(token) for token in sentence]
    
        diff_words_set = set()
        
        for word in words:
            syllable_count = self.syllables_count(word)
            if word not in nlp.Defaults.stop_words and syllable_count >= 2:
                diff_words_set.add(word)
    
        return len(diff_words_set)
    

    def shortforms(self, correct_text):
        collections = {'u':'you', 'b':'B'}
        num_of_error=0
        splitted_text=correct_text.split()
        for word in splitted_text:
            if word in collections.keys():
                num_of_error += 1
        return num_of_error 

    def fixgrammar(self):# It returns the correct text
        tool = language_tool_python.LanguageTool('en-US')
        return tool.correct(self.input.replace('\n', ''))

    def grammarerrors(self):
        tool = language_tool_python.LanguageTool('en-US')
        return len(tool.check(self.input.replace('\n', '')))

    def wrongwords(self):
        spell = SpellChecker()
        wrong = spell.unknown(self.words)
        return len(wrong) / len(self.input.split())

    def punctuation(self):
        count = 0
        for word in self.words:
            count += len([c for c in word if c in list(string.punctuation)])
        return count

    def stopword_count(self):
        count = 0
        for word in self.words:
            if word in stopwords.words('english'):
                count += 1
        return count

    def ARI(self, num_of_words):
        ttl_characters = 0
        for word in self.input.split():
            ttl_characters += len(word)
        ari = 4.71*(ttl_characters / num_of_words) + 0.5*(num_of_words / len(self.sent)) - 21.43
        return ari

    def sentiment_Score(self):
        # nltk.download('vader_lexicon') # Check if there is vader_lexicon -> if not download
        sia=SentimentIntensityAnalyzer()
        com = sia.polarity_scores(self.input).get('compound')
        pos = sia.polarity_scores(self.input).get('pos')
        neg = sia.polarity_scores(self.input).get('neg')
        return [com, pos, neg]

    def POS_Tagging(self):
        # Set the initial count num as 1 to avoid infinity in operation
        num_of_verb=1
        num_of_adj=1
        num_of_adv=1
        num_of_pron = 1
        num_of_tran = 1
        num_of_noun=1
        adj_list=[]
        adv_list=[]

        sample = self.words
        ps = PorterStemmer()
        sample_tokens=[ps.stem(word) for word in sample]

        tag=nltk.pos_tag(sample_tokens)
        for j in range(len(tag)):
            if tag[j][1] == 'VB':
                num_of_verb+=1
                continue
            if tag[j][1][:2] == 'JJ':
                num_of_adj+=1
                adj_list.append(tag[j][0])
                continue
            if tag[j][1][:2] == 'RB':
                num_of_adv+=1
                adv_list.append(tag[j][0])
                continue
            if tag[j][1] == 'NN' or tag[j][1]=='NNS':
                num_of_noun+=1
                continue
            if tag[j][1][:3] == 'PRP':
                num_of_pron += 1
            if tag[j][1] == 'CC':
                num_of_tran += 1
            if j < len(tag) -2 and tag[j][1] == 'IN' and tag[j+1][1] == 'NN' and tag[j+2][1] == ',':
                num_of_tran += 1

        num_of_distinct_adj = 1 + len(np.unique(np.array(adj_list)))
        num_of_distinct_adv = 1 + len(np.unique(np.array(adv_list)))

        return [
            num_of_verb,
            num_of_adj,
            num_of_adv,
            num_of_pron,
            num_of_tran,
            num_of_noun,
            num_of_distinct_adj,
            num_of_distinct_adv
        ]
    
# Used for testing
input = "I think that students would benefit from learning at home,because they wont have to change and get up early in the morning to shower and do there hair. taking only classes helps them because at there house they'll be pay more attention. they will be comfortable at home.The hardest part of school is getting ready. you wake up go brush your teeth and go to your closet and look at your cloths. after you think you picked a outfit u go look in the mirror and youll either not like it or you look and see a stain. Then you'll have to change. with the online classes you can wear anything and stay home and you wont need to stress about what to wear.most students usually take showers before school. they either take it before they sleep or when they wake up. some students do both to smell good. that causes them do miss the bus and effects on there lesson time cause they come late to school. when u have online classes u wont need to miss lessons cause you can get everything set up and go take a shower and when u get out your ready to go.when your home your comfortable and you pay attention. it gives then an advantage to be smarter and even pass there classmates on class work. public schools are difficult even if you try. some teacher dont know how to teach it in then way that students understand it. that causes students to fail and they may repeat the class."
data = DataEng(input).Engineering()
print(data)
# print(len(nltk.sent_tokenize(input)))

# split() -> punctuation with word
# nltk -> seperate punctuation



        # self.data['number_of_words'] = num_of_words
        # self.data['stopwords_frequency'] = stopwords_freq
        # self.data['av_word_per_sen'] = av_word_per_sen
        # self.data['punctuations'] = punctuations
        # self.data['ARI'] = ARI
        # self.data['freq_of_verb'] = freq_of_verb
        # self.data['freq_of_adj'] = freq_of_adj
        # self.data['freq_of_adv'] = freq_of_adv
        # self.data['freq_of_distinct_adj'] = freq_of_distinct_adj
        # self.data['freq_of_distinct_adv'] = freq_of_distinct_adv
        # self.data['freq_of_wrong_words'] = freq_of_wrong_words
        # self.data['sentiment_compound'] = sentiment_compound
        # self.data['sentiment_positive'] = sentiment_positive
        # data['sentiment_negative']
        # data['num_of_grammar_errors']
        # data['corrected_text']
        # data['num_of_short_forms']
        # data['Incorrect_form_ratio']
        # data['flesch_reading_ease']
        # data['flesch_kincaid_grade']
        # data['dale_chall_readability_score']
        # data['text_standard']
        # data['mcalpine_eflaw']
        # data['number_of_diff_words']
        # data['freq_diff_words']
        # data['ttr']
        # data['coherence_score']
        # data['lexrank_avg_min_diff']
        # data['lexrank_interquartile']
        # data['freq_of_noun']
        # data['freq_of_transition']
        # data['freq_of_pronoun']
        # data['noun_to_adj']
        # data['verb_to_adv']