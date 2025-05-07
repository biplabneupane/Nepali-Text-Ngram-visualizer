import os
import re
from collections import Counter
from nltk.util import ngrams
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set Devanagari-compatible font
rcParams['font.family'] = 'Noto Sans Devanagari'


# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Step 1: Load and clean text
with open('data/sample.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()
    text = re.sub(r'[^\w\s]', '', text)

# Step 2: Tokenize
tokens = text.split()

# Step 3: N-gram functions
def get_ngrams(tokens, n):
    return list(ngrams(tokens, n))

def save_ngram_plot(tokens, n):
    ngram_list = get_ngrams(tokens, n)
    freq = Counter(ngram_list)
    most_common = freq.most_common(10)

    if not most_common:
        print(f"No {n}-grams found.")
        return

    labels, values = zip(*most_common)
    labels = [' '.join(t) for t in labels]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color='skyblue')
    plt.xlabel("Frequency")
    plt.title(f"Top 10 {n}-grams")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    filename = f'output/{n}_gram_chart.png'
    plt.savefig(filename)
    plt.close()
    print(f"✅ {n}-gram chart saved to {filename}")

def print_ngrams(tokens, n):
    ngram_list = get_ngrams(tokens, n)
    freq = Counter(ngram_list)

    print(f"\n🔁 {n}-GRAM FREQUENCIES:")
    for k, v in freq.most_common():
        print(f"{' '.join(k)}: {v}")

    save_ngram_plot(tokens, n)

# Step 4: Run for unigrams, bigrams, trigrams
print_ngrams(tokens, 1)
print_ngrams(tokens, 2)
print_ngrams(tokens, 3)
