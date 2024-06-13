import requests
from collections import Counter
import matplotlib.pyplot as plt
import re

# reade online !!! 
url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
response = requests.get(url)
text = response.text

# delete non alphanumeric and do lowercase
words = re.findall(r'\b\w+\b', text.lower())

# get frequency from text
word_freq = Counter(words)

sorted_words = word_freq.most_common()
top_50_words = sorted_words[:50]

# print top 50
for i, (word, freq) in enumerate(top_50_words, start=1):
    print(f"{i}. {word}: {freq}")

# calculate mean const
const = [freq * rank for rank, (word, freq) in enumerate(sorted_words, start=1)]
average_const = sum(const) / len(const)
print(f"average const: {average_const}")

# plot term vs frequncy
terms = [word for word, freq in top_50_words]
frequencies = [freq for word, freq in top_50_words]

plt.figure(figsize=(17, 10))
plt.bar(terms, frequencies)
plt.xticks(rotation=75)
plt.xlabel('Term')
plt.ylabel('Frequency')
plt.savefig('TermFreq.pdf')
plt.close()
