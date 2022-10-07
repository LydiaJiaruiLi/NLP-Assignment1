import os
import nltk
from nltk.tokenize import TweetTokenizer
from collections import Counter
import re
from nltk.corpus import stopwords

def read_files(dir):
    files = os.listdir(dir)
    docs = []
    for file in files:
        f = open(dir + file, 'r', encoding='utf-8')
        docs.append(f.read().lower())
        f.close()
    return docs

docs = read_files('CUAD_v1/full_contract_txt/')

# Tokenization
tknzr = TweetTokenizer()
tokens = []
for doc in docs:
    tokens += tknzr.tokenize(doc)
print(len(tokens))

# Save tokens to output.txt
f = open('outputs/output.txt', 'w', encoding='utf-8')
tmp_tokens = tokens
for i in range(len(tmp_tokens)):
    if '\n' in tmp_tokens[i]:
        tmp_tokens[i] = tmp_tokens[i].replace('\n', ' ')
SEP = '\n'
f.write(SEP.join(tmp_tokens))
f.close

# with open('outputs/output.txt', 'r', encoding='utf-8') as fp:
#     x = len(fp.readlines())
#     print('Total lines:', x) # 8

# Count types
counts = Counter(tokens)

print(len(counts))
print('Question b:', len(counts)/len(tokens))

# Sort tokens by occurrence
print('The top 20 tokens with the most occurrences:')
print(counts.most_common(20))

f = open('outputs/tokens.txt', 'w', encoding='utf-8')
tmp_counts = counts
for each in tmp_counts.most_common():
    if '\n' in each[0]:
        each[0] = each[0].replace('\n', ' ')
    f.write(each[0] + '\t\t' + str(each[1]) + '\n')
f.close()

# Print tokens appeared only once
once = 0
for each in counts.most_common():
    if each[1] == 1:
        once += 1
print(once, 'tokens only appeared once')

# e)     From the list of tokens, extract only words, by excluding punctuation and other symbols, including forum-specific
# symbols, if any. Please pay attention to end of sentence dot (full stops). How many words did you find? List the top 20
# most frequent words in your report, with their frequencies. What is the type/token ratio when you use only words
# (called lexical diversity)?
tokens_without_symbols = []
for token in tmp_tokens:
    # detect if a word contains symbols
    if(bool(re.match('^[a-zA-Z0-9]*$',token))==True):
        tokens_without_symbols.append(token)
print("the number of only words without symbols is: ", len(tokens_without_symbols))

# Count types of the words without symbols
types_without_symbols = Counter(tokens_without_symbols)
print("The types of the words without symbols:", len(types_without_symbols))
print('The type/token ratio of the words without symbols:', len(types_without_symbols)/len(tokens_without_symbols))

# Sort tokens by occurrence
print('The top 20 tokens without symbols:')
print(types_without_symbols.most_common(20))

# f)     From the list of words, exclude stopwords. List the top 20 most frequent words and their frequencies in your report.
# You can use this list of stopwords (or any other that you consider adequate). Also compute the type/token ratio when you use only
# word tokens without stopwords (called lexical density)?
nltk.download('stopwords')
tokens_without_stopwords = []
stop_words = set(stopwords.words('english'))
for token in tmp_tokens:
    # detect if a word is a stopword
    if token not in stop_words:
        tokens_without_stopwords.append(token)

print("the number of only words without stopwords is: ", len(tokens_without_stopwords))

# Count types of the words without symbols
types_without_stopwords = Counter(tokens_without_stopwords)
print('The types of the words without stopwords:', len(types_without_stopwords))
print('The type/token ratio of the words without stopwords:', len(types_without_stopwords)/len(tokens_without_stopwords))

# Sort tokens by occurrence
print('The top 20 tokens without stopwords:')
print(types_without_stopwords.most_common(20))

# g)    Compute all the pairs of two consecutive words (bigrams) (excluding stopwords and punctuation).
# List the most frequent 20 pairs and their frequencies in your report.
tokens_without_stopwords_symbols = []
for token in tokens_without_stopwords:
    if (bool(re.match('^[a-zA-Z0-9]*$',token))==True):
        tokens_without_stopwords_symbols.append(token)

print("the number of only words without stopwords and symbols is: ", len(tokens_without_stopwords_symbols))
bigrams = nltk.bigrams(tokens_without_stopwords_symbols)
types_bigrams = Counter(list(bigrams))
print(types_bigrams.most_common(20))

# results for question e f g
# the number of only words without symbols is:  3922314
# The types of the words without symbols: 29657
# The type/token ratio of the words without symbols: 0.0075610978621293455
# The top 20 tokens without symbols:
# [('the', 257211), ('of', 156123), ('and', 132850), ('to', 129884), ('or', 108938), ('in', 79954), ('any', 62239), ('a', 53118), ('shall', 48794), ('by', 44317), ('agreement', 43631), ('this', 39989), ('be', 39702), ('for', 38727), ('such', 36171), ('with', 33886), ('as', 32911), ('party', 30494), ('that', 27653), ('other', 26310)]
# the number of only words without stopwords is:  3115940
# The types of the words without stopwords: 39958
# The type/token ratio of the words without stopwords: 0.012823738582899542
# The top 20 tokens without stopwords:
# [(',', 241445), ('.', 139317), (')', 76081), ('(', 75059), ('*', 61928), ('shall', 48794), ('"', 44393), ('agreement', 43631), ('party', 30494), ('-', 22942), ('[', 21494), (']', 21154), (':', 20503), (';', 15714), ('/', 15163), ('may', 13596), ('parties', 13514), ('section', 13296), ('company', 11259), ('information', 10937)]
# the number of only words without stopwords and symbols is:  2246174
# [(('set', 'forth'), 6152), (('third', 'party'), 4099), (('agreement', 'shall'), 3682), (('confidential', 'information'), 3604), (('party', 'shall'), 3286), (('intellectual', 'property'), 2932), (('effective', 'date'), 2822), (('written', 'notice'), 2385), (('either', 'party'), 2320), (('terms', 'conditions'), 2087), (('prior', 'written'), 1814), (('without', 'limitation'), 1781), (('forth', 'section'), 1744), (('shall', 'deemed'), 1685), (('term', 'agreement'), 1682), (('time', 'time'), 1674), (('shall', 'mean'), 1637), (('including', 'without'), 1554), (('party', 'may'), 1547), (('confidential', 'treatment'), 1539)]
#
# Process finished with exit code 0