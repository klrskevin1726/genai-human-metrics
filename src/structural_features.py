import re
import statistics
import string

def get_words(text):
    return re.findall(r'\b\w+\b', text.lower())

def sentence_stats(text):
    sentences = re.split(r'[.!?]+', text)
    
    word_count = []

    #Word count per sentence
    for sentence in sentences:
        sentence = sentence.strip()

        if sentence:
            word_count.append(len(sentence.split()))

    #Standar Deviation
    if len(word_count) > 1:
        std = statistics.stdev(word_count)
    else:
        std = 0.0

    
    avg = sum(word_count) / len(word_count)

    return avg, std

def lexical_diversity(text):
    words = get_words(text)

    total_words = len(words)
    unique_words = len(set(words))

    if total_words > 0:
        ttr = unique_words / total_words
    else:
        ttr = 0
    return ttr

def avg_word_length(text):
    words = get_words(text)

    if len(words) > 0:
        return sum(len(w) for w in words) / len(words)
    else:
        return 0.0
    
def punctuation_rate(text):
    punctuation_count = 0
    total_chr = 0

    for c in text:
        if c in string.punctuation:
            punctuation_count += 1

    if len(text) > 0:
        return punctuation_count / len(text)
    else:
        return 0.0
    
print(punctuation_rate("Hello, world! How are you?"))