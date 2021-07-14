# <u>Approaching any NLP Problem</u>

## <u>Stemming & Lemmatization</u>

Stemming and lemmatization are very useful and used interchangeably , However they are different.

They both reduce words to their base/origin.

**Stemming**: It reduces words to its base, words that are stemmed may no longer be real words (not understandable or not in the dictionary).

There are many stemming algorithms. Abhishek says Snowball stemmer is famous

**Lemmatization**: It reduces words to its base, but the lemmatized words will still be understandable to us.

There isn't a lot of Lemmatizers. example of lemmatizers is WordNetLematizer.

****

## **<u>Tokenization</u>**

Used for splitting text into single entities (words/sentences).

There  are many ways to tokenize sentences

```python
# split text to words
from ntlk.tokenize import word_tokenize
# split text to words with different algorithm
from ntlk.tokenize import word_tokenize
# split text into sentences
from ntlk.tokenize import sent_tokenize 

# you can create your own tokenizers using regexp + split
# or simply just using split() is a simple tokenizer
tokens = sentence.split()

import re
# removes all punctuation from a sentence then splits
re.sub(r'[^\w]', ' ', sentence).split()

# nltk also has a regexp_tokenizer that functions as above
```

****

