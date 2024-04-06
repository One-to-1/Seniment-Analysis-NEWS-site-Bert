# **Explanations**

## Importing necessary libraries

```python
from bs4 import BeautifulSoup
import requests
```

`BeautifulSoup` is a Python library used for web scraping purposes to pull the data out of HTML and XML files. `requests` is a Python library used for making HTTP requests.

## Fetching the webpage

```python
page = requests.get('https://economictimes.indiatimes.com/news/economy/indicators/growth-of-eight-core-sectors-at-6-7-in-feb-as-against-4-1-in-jan/articleshow/108852450.cms').text
```

`requests.get()` sends a GET request to the specified URL and returns a response. `.text` returns the text content of the response.

## Parsing the webpage

```python
content = BeautifulSoup(page, 'html.parser')
```

`BeautifulSoup()` is a function that parses the HTML content. It takes two arguments: the HTML content to be parsed and the parser library to be used ('html.parser' in this case).

## Finding all script blocks of type 'application/ld+json'

```python
script = content.find_all('script', {'type': 'application/ld+json'})
```

`find_all()` is a BeautifulSoup function that finds all instances of a tag in the parsed HTML. It takes two arguments: the tag to be found ('script' in this case) and a dictionary of attributes to match ({'type': 'application/ld+json'} in this case).

## Importing the `json` library

```python
import json
```

`json` is a Python library used for working with JSON data.

## Initializing an empty list to store the desired script block

```python
newsScriptBlock = []
```

This line initializes an empty list to store the script blocks that match the desired criteria.

## Iterating over each script block

```python
for s in script:
    content = json.loads(s.string)
    if '@type' in content and content['@type'] == 'NewsArticle':
        newsScriptBlock.append(s)
```

`json.loads()` is a function that parses a JSON string and converts it into a Python object. `s.string` returns the string content of the script block. The `if` statement checks if the '@type' key exists in the content and if its value is 'NewsArticle'. If both conditions are met, the script block is appended to `newsScriptBlock`.

## Initializing an empty list to store the extracted data

```python
extractedArticle = []
```

This line initializes an empty list to store the extracted data.

## Iterating over each script block in `newsScriptBlock`

```python
for s in newsScriptBlock:
    content = json.loads(s.string)
    extractedArticle.append({
        content.get('articleBody', '')
    })
```

This block of code is similar to the previous one, but instead of checking for a specific '@type', it extracts the 'articleBody' key from the content and appends it to `extractedArticle`. If 'articleBody' does not exist, an empty string is appended instead.

## Importing more libraries

```python
import re
import nltk
import pandas as pd
```

`re` is a Python library used for working with regular expressions. `nltk` (Natural Language Toolkit) is a Python library used for working with human language data. `pandas` is a Python library used for data manipulation and analysis.

## Initializing an empty list to store the sentences

```python
sentences = []
```

This line initializes an empty list to store the sentences.

## Iterating over each article in `extractedArticle`

```python
for article in extractedArticle:
    article_string = next(iter(article))
    cleaned_string = re.sub(r'Rs\.', 'Rs', article_string)
    article_sentences = nltk.sent_tokenize(cleaned_string)
    sentences.append(article_sentences)
```

`next(iter(article))` returns the first item in `article`. `re.sub()` is a function that replaces all occurrences of a pattern in a string. We are using this because we don't want the period in `Rs.` to be taken as as the ending of a sentence `nltk.sent_tokenize()` is a function that splits a text into sentences. The sentences are then appended to `sentences`.

## Flattening the list of sentences

```python
sentences = [sentence for sublist in sentences for sentence in sublist]
```

This line uses a list comprehension to flatten the list of sentences.
Flattening a list in Python refers to the process of converting a list of lists (or a nested list) into a single, flat list.

The `sentences` list is a list of lists, where each sublist contains sentences from a single article. The line of code:

```python
sentences = [sentence for sublist in sentences for sentence in sublist]
```

is using a list comprehension to "flatten" this list of lists. It iterates over each sublist in `sentences` and then over each `sentence` in that sublist. The result is a new list that contains all the sentences from all the articles, without any nested structure.

For example, if `sentences` was initially:

```python
[['sentence 1 from article 1', 'sentence 2 from article 1'], ['sentence 1 from article 2', 'sentence 2 from article 2']]
```

After flattening, `sentences` would be:

```python
['sentence 1 from article 1', 'sentence 2 from article 1', 'sentence 1 from article 2', 'sentence 2 from article 2']
```

This makes it easier to process all the sentences together in the subsequent steps of the code.

## Converting the list of sentences into a pandas DataFrame

```python
df = pd.DataFrame(sentences, columns=['sentence'])
```

`pd.DataFrame()` is a function that creates a DataFrame from a list. The `columns` argument specifies the column labels.

## Importing more libraries

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scipy
import torch
```

`transformers` is a Python library used for state-of-the-art natural language processing. `scipy` is a Python library used for scientific computing. `torch` is a Python library used for machine learning.

## Loading the pretrained FinBERT model and tokenizer

```python
X = df['sentence'].to_list()
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

`df['sentence'].to_list()` converts the 'sentence' column of the DataFrame to a list. `AutoTokenizer.from_pretrained()` and `AutoModelForSequenceClassification.from_pretrained()` are functions that load the pretrained FinBERT tokenizer and model, respectively.

## Making sentiment predictions

```python
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
```

This block of code tokenizes each sentence, feeds it into the model, computes the softmax of the output logits, and stores the predicted sentiment and its probability. `torch.no_grad()` is a context manager that disables gradient calculation, which is not needed during inference. `tokenizer()` tokenizes the input sentence. `model()` feeds the tokenized input into the model. `scipy.special.softmax()` applies the softmax function to the output logits. `max()` returns the key/value with the highest value.

17.1. `preds = []`, `preds_proba = []`, `scoresList = []`: These lines initialize three empty lists. `preds` will store the predicted sentiment for each sentence, `preds_proba` will store the probability of the predicted sentiment, and `scoresList` will store the scores for each sentiment.

17.2. `tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}`: This line sets up the arguments for the tokenizer. Padding ensures that all sequences in a batch have the same length, truncation cuts off sequences longer than the specified maximum length, and the maximum length is set to 512 tokens.

17.3. `for x in X:`: This line starts a loop over each sentence in `X`.

17.4. `with torch.no_grad():`: This line starts a context where gradient computation is disabled.

`torch` is a Python library, part of the PyTorch framework, that provides multi-dimensional arrays (tensors) and an extensive collection of operations for these tensors. It's widely used in machine learning and deep learning for tasks such as training neural networks.

`torch.no_grad()` is a context manager in PyTorch that disables the calculation of gradients. Gradients are measures of how much the output of a function changes if you change the inputs a little bit. They are essential for optimization tasks, such as training a neural network, where you want to adjust the parameters of the network to minimize the error on your training data.

However, when you're just applying the network (i.e., doing inference, like predicting the sentiment of a sentence), you don't need to compute gradients. Disabling gradient computation with `torch.no_grad()` can save memory and make your code run faster.

In the given code, `torch.no_grad()` is used because the model is being used for inference, not training. The model's parameters aren't being updated, so there's no need to keep track of the gradients.

17.5. `input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)`: This line tokenizes the sentence. The `return_tensors="pt"` argument tells the tokenizer to return PyTorch tensors.

17.6. `logits = model(**input_sequence).logits`: This line feeds the tokenized sentence into the model and retrieves the logits (raw prediction values) from the output.

17.7. `scores = {...}`: This block of code computes the softmax of the logits to get probabilities, and maps each probability to its corresponding sentiment label. The softmax function is used to convert the logits into probabilities that sum to 1.

17.8. `sentimentFinbert = max(scores, key=scores.get)`: This line determines the sentiment with the highest probability.

17.9. `probabilityFinbert = max(scores.values())`: This line retrieves the highest probability.

17.10. `scoresList.append(scores)`, `preds.append(sentimentFinbert)`, `preds_proba.append(probabilityFinbert)`: These lines append the scores, predicted sentiment, and probability to their respective lists.

17.11. `print(x, scores)`: This line prints the sentence and its scores for each sentiment.

## Creating the output DataFrame

```python
Output = pd.DataFrame({
    'X': X,
    'preds': preds,
    'preds_proba': preds_proba
})
scoresdf = pd.DataFrame(scoresList)
Output = pd.concat([Output, scoresdf], axis=1)
```

This block of code creates a DataFrame that contains the original sentences, their predicted sentiments, and the probabilities of each sentiment. `pd.concat()` concatenates the two DataFrames along the columns (`axis=1`).

## Printing the output DataFrame:

```python
print(Output)
```

This line prints the output DataFrame.
