import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from textblob import TextBlob
import numpy as np
import os

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

def measure_text_complexity(text):

    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]

    num_words = len(words)
    num_sentences = len(sentences)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    unique_words = set(words)
    ttr = len(unique_words) / num_words if num_words > 0 else 0

    d = cmudict.dict()


    def count_syllables(word):
        try:
            return max([len([y for y in x if y[-1].isdigit()]) for x in d[word.lower()]])
        except KeyError:
            return sum([1 for char in word if char in 'aeiouy'])


    syllable_count = sum(count_syllables(word) for word in words)
    avg_syllables_per_word = syllable_count / num_words if num_words > 0 else 0
    flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)


    pos_tags = pos_tag(words)
    pos_counts = {tag: 0 for tag in ['NN', 'VB', 'JJ', 'RB']}
    for _, tag in pos_tags:
        if tag[:2] in pos_counts:
            pos_counts[tag[:2]] += 1
    pos_distribution = {key: count / num_words for key, count in pos_counts.items()}


    blob = TextBlob(text)
    sentiment = blob.sentiment

    return {
        "Number of Words": num_words,
        "Number of Sentences": num_sentences,
        "Average Sentence Length": avg_sentence_length,
        "Vocabulary Richness (TTR)": ttr,
        "Average Syllables per Word": avg_syllables_per_word,
        "Flesch Reading Ease Score": flesch_reading_ease,
        "POS Distribution": pos_distribution,
        "Sentiment Polarity": sentiment.polarity,
        "Sentiment Subjectivity": sentiment.subjectivity,
    }

def compute_pairwise_similarity(lines):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(lines)
    similarities = []
    for i in range(len(lines) - 1):
        sim = 1 - cosine(embeddings[i].toarray(), embeddings[i + 1].toarray())
        similarities.append(sim)
    return np.mean(similarities) if similarities else 0


def analyze_file(file_path):
    if not os.path.exists(file_path):
        print("File not found.")
        return

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]


    print(
        f"{'Line Number':<12} {'Words':<8} {'Sentences':<10} {'Avg Sent Len':<15} {'TTR':<6} {'Syll/Wrd':<10} "
        f"{'FRE':<6} {'Polarity':<10} {'Subjectivity':<12}")
    print("-" * 100)


    for i, line in enumerate(lines, start=1):
        metrics = measure_text_complexity(line)
        print(f"{i:<12} {metrics['Number of Words']:<8} {metrics['Number of Sentences']:<10} "
              f"{metrics['Average Sentence Length']:<15.2f} {metrics['Vocabulary Richness (TTR)']:<6.2f} "
              f"{metrics['Average Syllables per Word']:<10.2f} {metrics['Flesch Reading Ease Score']:<6.2f} "
              f"{metrics['Sentiment Polarity']:<10.2f} {metrics['Sentiment Subjectivity']:<12.2f}")


    avg_similarity = compute_pairwise_similarity(lines)
    print(f"\nAverage Sentence Embedding Similarity Across Lines: {avg_similarity:.4f}")


file_path = "src/temperatures.txt"
analyze_file(file_path)
