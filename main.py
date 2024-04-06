from bs4 import BeautifulSoup
import requests

page = requests.get('https://economictimes.indiatimes.com/news/economy/indicators/growth-of-eight-core-sectors-at-6-7-in-feb-as-against-4-1-in-jan/articleshow/108852450.cms').text

content = BeautifulSoup(page, 'html.parser')

script = content.find_all('script', {'type': 'application/ld+json'})

import json

# Initialize an empty list to store the desired script block
newsScriptBlock = []

# Iterate over each script block
for s in script:
    # Parse the JSON content of the script block
    content = json.loads(s.string)
    
    # Check if the '@type' key exists and if its value is 'NewsArticle'
    if '@type' in content and content['@type'] == 'NewsArticle':
        # If it is, append the script block to the list
        newsScriptBlock.append(s)
        
# Initialize an empty list to store the extracted data
extractedArticle = []

# Iterate over each script block
for s in newsScriptBlock:
    # Parse the JSON content of the script block
    content = json.loads(s.string)
    
    # Extract the desired keys and append them to the list
    extractedArticle.append({
        content.get('articleBody', '')
    })

import re
import nltk
import pandas as pd

# Initialize an empty list to store the sentences
sentences = []

# Iterate over each article in extractedArticle
for article in extractedArticle:
    # Extract the string from the set
    article_string = next(iter(article))
    # Clean the sentences using regular expressions
    cleaned_string = re.sub(r'Rs\.', 'Rs', article_string)
    # Use nltk.sent_tokenize to split the article into sentences
    article_sentences = nltk.sent_tokenize(cleaned_string)
    # Append the sentences to the list
    sentences.append(article_sentences)

# Flatten the list of sentences
sentences = [sentence for sublist in sentences for sentence in sublist]
print(sentences)

# Convert the list of sentences into a pandas DataFrame
df = pd.DataFrame(sentences, columns=['sentence'])

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scipy
import torch

X = df['sentence'].to_list()
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

preds = []
preds_proba = []
scoresList = []
tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}
for x in X:
    with torch.no_grad():
        input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)
        logits = model(**input_sequence).logits
        scores = {
        k: v
        for k, v in zip(
            model.config.id2label.values(),
            scipy.special.softmax(logits.numpy().squeeze()),
        )
    }
    sentimentFinbert = max(scores, key=scores.get)
    probabilityFinbert = max(scores.values())
    scoresList.append(scores)
    preds.append(sentimentFinbert)
    preds_proba.append(probabilityFinbert)
    print(x, scores)
    
Output = pd.DataFrame({
    'X': X,
    'preds': preds,
    'preds_proba': preds_proba
})
scoresdf = pd.DataFrame(scoresList)
Output = pd.concat([Output, scoresdf], axis=1)

print(Output)