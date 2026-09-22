import re
import statistics

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