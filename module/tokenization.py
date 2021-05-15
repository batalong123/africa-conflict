import string
import spacy

#create function tokenizer
def tokenizer(sentence):
    
    # Create our list of punctuation marks
    punctuations = string.punctuation

    # Create our list of stopwords
    nlp = spacy.load("en_core_web_sm")
    stop_words = spacy.lang.en.stop_words.STOP_WORDS

    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

    # Removing stop words
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

    # return preprocessed list of tokens
    return mytokens


def word_tokenizer(text):

    def is_token_allowed(token):
        if(not token or token.is_stop or token.is_punct):
            return False
        return True
    
    def preprocess_token(token):
        return token.lemma_.strip().lower()
    
    nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(text)
    
    filtered_token = [preprocess_token(token) for token in doc if is_token_allowed(token)]
    
    return filtered_token  