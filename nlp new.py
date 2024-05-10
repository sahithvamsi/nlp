from nltk.tokenize  import word_tokenize, sent_tokenize

from nltk import ne_chunk,pos_tag

from nltk.corpus import stopwords

from nltk.stem  import PorterStemmer, WordNetLemmatizer


text = "ramesh is good boy. suresh is best boy"


new_Se = sent_tokenize(text)

print(new_Se)


for sen in new_Se:
    print(sen)
    wor_t = word_tokenize(sen)
    print(wor_t)
    stop_w = set(stopwords.words("english"))
    fi_w = [word for word in wor_t if word.lower() not in stop_w]
    print(fi_w)
    porter = PorterStemmer()
    ste = [porter.stem(word) for word in fi_w]
    print(ste)
    lem = WordNetLemmatizer()
    lemmza = [lem.lemmatize(word) for word in fi_w]
    print(lemmza)
    pos = pos_tag(lemmza)
    print(pos)
    ner = ne_chunk(pos)
    print(ner)
    
#ngram
from nltk import ngrams

from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps over the lazy dog."




new_t = word_tokenize(text)

ng = list(ngrams(new_t, 2))

print(ng)

#tfidf

from sklearn.feature_extraction.text import TfidfVectorizer

# Example corpus
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
]

model = TfidfVectorizer()

tf_mat = model.fit_transform(corpus)

names = model.get_feature_names_out()

print(tf_mat.toarray())

