# A Sentiment Analysis Web App
## Using PyTorch and SageMaker

---

This is a sentiment analysis model trained on 25000 movie reviews, and was built as a part of Udacity's Deep Learning Nanodegree.


## The Outline

The project was broadly built on the following lines, and this notebook has been divided into the same sections.

1. Download or otherwise retrieve the data.
2. Process / Prepare the data.
3. Upload the processed data to S3.
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job).
6. Deploy the trained model.
7. Use the deployed model.


## Downloading the data

I will be using the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

> Maas, Andrew L., et al. [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/). In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_. Association for Computational Linguistics, 2011.


```python
%mkdir ../data
!wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zxf ../data/aclImdb_v1.tar.gz -C ../data
```

    mkdir: cannot create directory ‘../data’: File exists
    --2020-11-21 17:49:51--  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    Resolving ai.stanford.edu (ai.stanford.edu)... 171.64.68.10
    Connecting to ai.stanford.edu (ai.stanford.edu)|171.64.68.10|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 84125825 (80M) [application/x-gzip]
    Saving to: ‘../data/aclImdb_v1.tar.gz’
    
    ../data/aclImdb_v1. 100%[===================>]  80.23M  5.96MB/s    in 19s     
    
    2020-11-21 17:50:11 (4.19 MB/s) - ‘../data/aclImdb_v1.tar.gz’ saved [84125825/84125825]
    
    

## Preparing and Processing the data

Once I downloaded the data, I did some initial data processing: I read in each of the reviews, combined them into a single input structure, then split the data into a training set and a testing set of 25,000 observations each, with positive and negative reviews distributed equally.


```python
import os
import glob

def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}
    
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}
        
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # A positive review is '1' and a negative review is '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
                    
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)
                
    return data, labels
```


```python
data, labels = read_imdb_data()
print("IMDB reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
            len(data['train']['pos']), len(data['train']['neg']),
            len(data['test']['pos']), len(data['test']['neg'])))
```

    IMDB reviews: train = 12500 pos / 12500 neg, test = 12500 pos / 12500 neg
    

I then combined positive and negative reviews into test and train dataframes, and used Scikit-learn's shuffle method to 
shuffle them around.


```python
from sklearn.utils import shuffle

def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""
    
    #Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']
    
    #Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test
```


```python
train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))
```

    IMDb reviews (combined): train = 25000, test = 25000
    

Here's what a single datapoint in the set looks like. It's a good idea to see, for yourself, what the processing pipeline has done, and whether or not it worked as expected.


```python
print(train_X[100])
print(train_y[100])
```

    An executive, very successful in his professional life but very unable in his familiar life, meets a boy with down syndrome, escaped from a residence . Both characters feel very alone, and the apparently less intelligent one will show to the executive the beauty of the small things in life... With this argument, the somehow Amelie-like atmosphere and the sentimental music, I didn't expect but a moralistic disgusting movie. Anyway, as there were some interesting scenes (the boy is sometimes quite a violent guy), and the interpretation of both actors, Daniel Auteil and Pasqal Duquenne, was very good, I decided to go on watching the movie. The French cinema, in general, has the ability of showing something that seems quite much to life, opposed to the more stereotyped American cinema. But, because of that, it is much more disappointing to see after the absurd ending, with the impossible death of the boy, the charming tone, the happiness of the executive's family, the cheap moral, the unbearable laughter of the daughters, the guy waving from heaven as Michael Landon... Really nasty, in my humble opinion.
    0
    

I removed all html tags, and tokenized the input using a stemmer. This  way words such as *entertained* and *entertaining* are considered the same.


```python
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup

def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words
```

The `review_to_words` method above uses `BeautifulSoup` to remove any html tags that appear and uses the `nltk` package to tokenize the reviews.

Here's what it looks like applied to a single review:


```python
# Apply review_to_words to a review
review_to_words(train_X[99])
```




    ['poor',
     'perform',
     'sinatra',
     'martin',
     'hyer',
     'grossli',
     'underdevelop',
     'support',
     'charact',
     'annoy',
     'talki',
     'real',
     'plot',
     'end',
     'leav',
     'flatter',
     'pancak',
     'loos',
     'end',
     'could',
     'tie',
     'four',
     'sequel',
     'even',
     'care',
     'wooden',
     'charact',
     'maclain',
     'real',
     'asset',
     'penultim',
     'sequenc',
     'chicago',
     'hood',
     'search',
     'sinatra',
     'charact',
     'laughabl',
     'music',
     'sequenc',
     'also',
     'poor',
     'final',
     'scene',
     'martin',
     'charact',
     'remov',
     'hat',
     'woman',
     'call',
     'pig',
     'almost',
     'made',
     'go',
     'outsid',
     'find',
     'stone',
     'throw',
     'televis',
     'screen']




```python
len(set(review_to_words(train_X[99])))
```




    51




```python
print(len(train_X[99]))
print(len(review_to_words(train_X[99])))
```

    642
    60
    

The above used Porter stemmer not only gets the stems of words, but also converts all of them to lower case. Besides this, it also removes all punctuation, whitespace and returns a list of words. 

Since my `review_to_words` function works as expected, I'll apply it to the entire dataset.


```python
import pickle

cache_dir = os.path.join("../cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, data_train))
        #words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test
```


```python
# Preprocess data
train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)
```

    Read preprocessed data from cache file: preprocessed_data.pkl
    

## Transform the data

The next set is tp transform the data from its word representation to a bag-of-words feature representation. First, I will represent each word as an integer. 

Next, I set a working vocabulary with a fixed size. This is because some of the words that appear in the reviews occur very infrequently and likely don't contain much information anyway. The vocabulary will only contain the most frequently occuring words. I will combine all the infrequent words into a single category and, label it `1`.

Besides, the model is a recurrent neural network. It will be convenient if the length of reviews is the same. To do this, we will fix a size for our reviews and then pad short reviews with the category 'no word' (which we will label `0`) and truncate long reviews.

### Create a word dictionary

Here I build a dictionary to map words that appear in the reviews to integers. I fix the size of the vocabulary (including the 'no word' and 'infrequent' categories) to be `5000`.


```python
import numpy as np
from collections import Counter

def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    # Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    # sentence is a list of words.
    
    word_count = Counter(np.concatenate(data, axis = 0)) # A dict storing the words that appear in the reviews along with how often they occur
    
    # Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    # sorted_words[-1] is the least frequently appearing word.
    
    sorted_words = sorted(word_count, key = word_count.get, reverse = True)
    
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict
```


```python
word_dict = build_dict(train_X)
```


```python
# TODO: Use this space to determine the five most frequently appearing words in the training set.
word_count_t = Counter(np.concatenate(train_X, axis = 0))

sorted_words_t = sorted(word_count_t, key = word_count_t.get, reverse = True)

print(sorted_words_t[:5])
```

    ['movi', 'film', 'one', 'like', 'time']
    

I ran the function to understand the most frequently appearing phrases as a sanity check. These words are movie related, and form the phrase *once upon a time*. Makes sense.

```['movi', 'film', 'one', 'like', 'time']```

### Save `word_dict`

I'm made a pickle dump for the `word_dict` method. I'll use it later, once I build an endpoint to processes a submitted review.


```python
data_dir = '../data/pytorch' # The folder we will use for storing data
if not os.path.exists(data_dir): # Make sure that the folder exists
    os.makedirs(data_dir)
```


```python
with open(os.path.join(data_dir, 'word_dict.pkl'), "wb") as f:
    pickle.dump(word_dict, f)
```

### Transform the reviews

I will use the word dictionary to transform reviews to integer sequence representation, making sure to pad or truncate to a fixed length, which in this case is `500`.


```python
def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)

def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []
    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)
```


```python
train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
test_X, test_X_len = convert_and_pad_data(word_dict, test_X)
```

As a quick check to make sure that things are working as intended.


```python
print(len(train_X[99]))
print(train_X_len[99])
```

    500
    114
    


```python
d = np.array(train_X[99])
np.count_nonzero(d)
```




    114



**Comments:** 
* At the face of it, implementing the ```preprocess_data``` method looks like a good practice. This procees the raw data and dumps the pre-processed data into a pkl file. We can retrieve processed data from this dump going forward. Thought this was a time counsuming method to run, this function was a one-time call.

* The ```convert_and_pad_data``` is an absolute must since we're training a Neural Network. The network will expect reviews of uniform length and this function ensure that this happens.

## Upload the data to S3

I will need to upload the training dataset to S3.

### Save the processed training dataset locally

I will save the dataset locally, first. Each row of the dataset has the form `label`, `length`, `review[500]` where `review[500]` is a sequence of `500` integers representing the words in the review.


```python
import pandas as pd
    
pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1) \
        .to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
```

### Uploading the training data


I will upload the training data to the SageMaker default S3 bucket so that we can provide access to it while training our model.


```python
import sagemaker

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/sentiment_rnn'

role = sagemaker.get_execution_role()
```


```python
input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)
```

## Build and Train the PyTorch Model

A model in the SageMaker framework consists of three objects:

 - Model Artifacts,
 - Training Code, and
 - Inference Code,
 
Here I used containers provided by Amazon, and added some custom code.

I will start by building a neural network in PyTorch along with a training script. Code to the model object is written in another script, but can be found below:


```python
!pygmentize train/model.py
```

    [34mimport[39;49;00m [04m[36mtorch[39;49;00m[04m[36m.[39;49;00m[04m[36mnn[39;49;00m [34mas[39;49;00m [04m[36mnn[39;49;00m
    
    [34mclass[39;49;00m [04m[32mLSTMClassifier[39;49;00m(nn.Module):
        [33m"""[39;49;00m
    [33m    This is the simple RNN model we will be using to perform Sentiment Analysis.[39;49;00m
    [33m    """[39;49;00m
    
        [34mdef[39;49;00m [32m__init__[39;49;00m([36mself[39;49;00m, embedding_dim, hidden_dim, vocab_size):
            [33m"""[39;49;00m
    [33m        Initialize the model by settingg up the various layers.[39;49;00m
    [33m        """[39;49;00m
            [36msuper[39;49;00m(LSTMClassifier, [36mself[39;49;00m).[32m__init__[39;49;00m()
    
            [36mself[39;49;00m.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=[34m0[39;49;00m)
            [36mself[39;49;00m.lstm = nn.LSTM(embedding_dim, hidden_dim)
            [36mself[39;49;00m.dense = nn.Linear(in_features=hidden_dim, out_features=[34m1[39;49;00m)
            [36mself[39;49;00m.sig = nn.Sigmoid()
            
            [36mself[39;49;00m.word_dict = [34mNone[39;49;00m
    
        [34mdef[39;49;00m [32mforward[39;49;00m([36mself[39;49;00m, x):
            [33m"""[39;49;00m
    [33m        Perform a forward pass of our model on some input.[39;49;00m
    [33m        """[39;49;00m
            x = x.t()
            lengths = x[[34m0[39;49;00m,:]
            reviews = x[[34m1[39;49;00m:,:]
            embeds = [36mself[39;49;00m.embedding(reviews)
            lstm_out, _ = [36mself[39;49;00m.lstm(embeds)
            out = [36mself[39;49;00m.dense(lstm_out)
            out = out[lengths - [34m1[39;49;00m, [36mrange[39;49;00m([36mlen[39;49;00m(lengths))]
            [34mreturn[39;49;00m [36mself[39;49;00m.sig(out.squeeze())


There are three parameters that I can tweak to alter model performance: embedding dimension, the hidden dimension and the size of the vocabulary.

First I load a portion of the training data set as a sample, and train on this.


```python
import torch
import torch.utils.data

# Read in only the first 250 rows
train_sample = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None, names=None, nrows=250)

# Turn the input pandas dataframe into tensors
train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze()
train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long()

# Build the dataset
train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
# Build the dataloader
train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)
```

Writing the training method

Here I write some classic code training my neural network over several epochs:


```python
def train(model, train_loader, epochs, optimizer, loss_fn, device):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            optimizer.zero_grad()
            
            output = model(batch_X)
            
            loss = loss_fn(output, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
```

I now test over a small number of epochs as a sanity check


```python
import torch.optim as optim
from train.model import LSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(32, 100, 5000).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()

train(model, train_sample_dl, 5, optimizer, loss_fn, device)
```

    Epoch: 1, BCELoss: 0.6932283639907837
    Epoch: 2, BCELoss: 0.6832670569419861
    Epoch: 3, BCELoss: 0.6746159672737122
    Epoch: 4, BCELoss: 0.6651097059249877
    Epoch: 5, BCELoss: 0.6537458539009094
    

### Training the model

I initialize an estimator on SageMaker and get ready to train


```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point="train.py",
                    source_dir="train",
                    role=role,
                    framework_version='0.4.0',
                    py_version = 'py3',
                    train_instance_count=1,
                    train_instance_type='ml.p2.xlarge',
                    hyperparameters={
                        'epochs': 10,
                        'hidden_dim': 200,
                    })
```


```python
estimator.fit({'training': input_data})
```

    'create_image_uri' will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.
    's3_input' class will be renamed to 'TrainingInput' in SageMaker Python SDK v2.
    'create_image_uri' will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.
    

    2020-11-21 18:15:53 Starting - Starting the training job...
    2020-11-21 18:15:56 Starting - Launching requested ML instances......
    2020-11-21 18:17:16 Starting - Preparing the instances for training.........
    2020-11-21 18:18:36 Downloading - Downloading input data...
    2020-11-21 18:19:10 Training - Downloading the training image...
    2020-11-21 18:19:41 Training - Training image download completed. Training in progress.[34mbash: cannot set terminal process group (-1): Inappropriate ioctl for device[0m
    [34mbash: no job control in this shell[0m
    [34m2020-11-21 18:19:41,386 sagemaker-containers INFO     Imported framework sagemaker_pytorch_container.training[0m
    [34m2020-11-21 18:19:41,414 sagemaker_pytorch_container.training INFO     Block until all host DNS lookups succeed.[0m
    [34m2020-11-21 18:19:42,028 sagemaker_pytorch_container.training INFO     Invoking user training script.[0m
    [34m2020-11-21 18:19:42,284 sagemaker-containers INFO     Module train does not provide a setup.py. [0m
    [34mGenerating setup.py[0m
    [34m2020-11-21 18:19:42,284 sagemaker-containers INFO     Generating setup.cfg[0m
    [34m2020-11-21 18:19:42,284 sagemaker-containers INFO     Generating MANIFEST.in[0m
    [34m2020-11-21 18:19:42,285 sagemaker-containers INFO     Installing module with the following command:[0m
    [34m/usr/bin/python -m pip install -U . -r requirements.txt[0m
    [34mProcessing /opt/ml/code[0m
    [34mCollecting pandas (from -r requirements.txt (line 1))
      Downloading https://files.pythonhosted.org/packages/74/24/0cdbf8907e1e3bc5a8da03345c23cbed7044330bb8f73bb12e711a640a00/pandas-0.24.2-cp35-cp35m-manylinux1_x86_64.whl (10.0MB)[0m
    [34mCollecting numpy (from -r requirements.txt (line 2))[0m
    [34m  Downloading https://files.pythonhosted.org/packages/b5/36/88723426b4ff576809fec7d73594fe17a35c27f8d01f93637637a29ae25b/numpy-1.18.5-cp35-cp35m-manylinux1_x86_64.whl (19.9MB)[0m
    [34mCollecting nltk (from -r requirements.txt (line 3))
      Downloading https://files.pythonhosted.org/packages/92/75/ce35194d8e3022203cca0d2f896dbb88689f9b3fce8e9f9cff942913519d/nltk-3.5.zip (1.4MB)[0m
    [34mCollecting beautifulsoup4 (from -r requirements.txt (line 4))
      Downloading https://files.pythonhosted.org/packages/d1/41/e6495bd7d3781cee623ce23ea6ac73282a373088fcd0ddc809a047b18eae/beautifulsoup4-4.9.3-py3-none-any.whl (115kB)[0m
    [34mCollecting html5lib (from -r requirements.txt (line 5))
      Downloading https://files.pythonhosted.org/packages/6c/dd/a834df6482147d48e225a49515aabc28974ad5a4ca3215c18a882565b028/html5lib-1.1-py2.py3-none-any.whl (112kB)[0m
    [34mRequirement already satisfied, skipping upgrade: python-dateutil>=2.5.0 in /usr/local/lib/python3.5/dist-packages (from pandas->-r requirements.txt (line 1)) (2.7.5)[0m
    [34mCollecting pytz>=2011k (from pandas->-r requirements.txt (line 1))[0m
    [34m  Downloading https://files.pythonhosted.org/packages/12/f8/ff09af6ff61a3efaad5f61ba5facdf17e7722c4393f7d8a66674d2dbd29f/pytz-2020.4-py2.py3-none-any.whl (509kB)[0m
    [34mRequirement already satisfied, skipping upgrade: click in /usr/local/lib/python3.5/dist-packages (from nltk->-r requirements.txt (line 3)) (7.0)[0m
    [34mCollecting joblib (from nltk->-r requirements.txt (line 3))
      Downloading https://files.pythonhosted.org/packages/28/5c/cf6a2b65a321c4a209efcdf64c2689efae2cb62661f8f6f4bb28547cf1bf/joblib-0.14.1-py2.py3-none-any.whl (294kB)[0m
    [34mCollecting regex (from nltk->-r requirements.txt (line 3))
      Downloading https://files.pythonhosted.org/packages/2e/e4/3447fed9ab29944333f48730ecff4dca92f0868c5b188d6ab2b2078e32c2/regex-2020.11.13.tar.gz (694kB)[0m
    [34mCollecting tqdm (from nltk->-r requirements.txt (line 3))
      Downloading https://files.pythonhosted.org/packages/02/bc/857fff709f7ce9eabdc502d6fa71f4b7e964200b1bcd00f0a2f59667d1bf/tqdm-4.53.0-py2.py3-none-any.whl (70kB)[0m
    [34mCollecting soupsieve>1.2; python_version >= "3.0" (from beautifulsoup4->-r requirements.txt (line 4))
      Downloading https://files.pythonhosted.org/packages/6f/8f/457f4a5390eeae1cc3aeab89deb7724c965be841ffca6cfca9197482e470/soupsieve-2.0.1-py3-none-any.whl[0m
    [34mRequirement already satisfied, skipping upgrade: six>=1.9 in /usr/local/lib/python3.5/dist-packages (from html5lib->-r requirements.txt (line 5)) (1.11.0)[0m
    [34mCollecting webencodings (from html5lib->-r requirements.txt (line 5))
      Downloading https://files.pythonhosted.org/packages/f4/24/2a3e3df732393fed8b3ebf2ec078f05546de641fe1b667ee316ec1dcf3b7/webencodings-0.5.1-py2.py3-none-any.whl[0m
    [34mBuilding wheels for collected packages: nltk, train, regex
      Running setup.py bdist_wheel for nltk: started[0m
    [34m  Running setup.py bdist_wheel for nltk: finished with status 'done'
      Stored in directory: /root/.cache/pip/wheels/ae/8c/3f/b1fe0ba04555b08b57ab52ab7f86023639a526d8bc8d384306
      Running setup.py bdist_wheel for train: started
      Running setup.py bdist_wheel for train: finished with status 'done'
      Stored in directory: /tmp/pip-ephem-wheel-cache-ltcnozv6/wheels/35/24/16/37574d11bf9bde50616c67372a334f94fa8356bc7164af8ca3
      Running setup.py bdist_wheel for regex: started[0m
    [34m  Running setup.py bdist_wheel for regex: finished with status 'done'
      Stored in directory: /root/.cache/pip/wheels/27/f6/66/a4243e485a0ebc73dc59033ae26c48e82526f77dbfe158ac59[0m
    [34mSuccessfully built nltk train regex[0m
    [34mInstalling collected packages: pytz, numpy, pandas, joblib, regex, tqdm, nltk, soupsieve, beautifulsoup4, webencodings, html5lib, train
      Found existing installation: numpy 1.15.4
        Uninstalling numpy-1.15.4:[0m
    [34m      Successfully uninstalled numpy-1.15.4[0m
    [34mSuccessfully installed beautifulsoup4-4.9.3 html5lib-1.1 joblib-0.14.1 nltk-3.5 numpy-1.18.5 pandas-0.24.2 pytz-2020.4 regex-2020.11.13 soupsieve-2.0.1 tqdm-4.53.0 train-1.0.0 webencodings-0.5.1[0m
    [34mYou are using pip version 18.1, however version 20.3b1 is available.[0m
    [34mYou should consider upgrading via the 'pip install --upgrade pip' command.[0m
    [34m2020-11-21 18:20:04,991 sagemaker-containers INFO     Invoking user script
    [0m
    [34mTraining Env:
    [0m
    [34m{
        "module_dir": "s3://sagemaker-ap-south-1-615025276797/sagemaker-pytorch-2020-11-21-18-15-53-195/source/sourcedir.tar.gz",
        "log_level": 20,
        "num_gpus": 1,
        "channel_input_dirs": {
            "training": "/opt/ml/input/data/training"
        },
        "job_name": "sagemaker-pytorch-2020-11-21-18-15-53-195",
        "hyperparameters": {
            "epochs": 10,
            "hidden_dim": 200
        },
        "current_host": "algo-1",
        "input_config_dir": "/opt/ml/input/config",
        "output_intermediate_dir": "/opt/ml/output/intermediate",
        "additional_framework_parameters": {},
        "model_dir": "/opt/ml/model",
        "network_interface_name": "eth0",
        "resource_config": {
            "hosts": [
                "algo-1"
            ],
            "current_host": "algo-1",
            "network_interface_name": "eth0"
        },
        "output_dir": "/opt/ml/output",
        "framework_module": "sagemaker_pytorch_container.training:main",
        "module_name": "train",
        "output_data_dir": "/opt/ml/output/data",
        "input_data_config": {
            "training": {
                "RecordWrapperType": "None",
                "S3DistributionType": "FullyReplicated",
                "TrainingInputMode": "File"
            }
        },
        "num_cpus": 4,
        "hosts": [
            "algo-1"
        ],
        "input_dir": "/opt/ml/input",
        "user_entry_point": "train.py"[0m
    [34m}
    [0m
    [34mEnvironment variables:
    [0m
    [34mSM_HP_HIDDEN_DIM=200[0m
    [34mPYTHONPATH=/usr/local/bin:/usr/lib/python35.zip:/usr/lib/python3.5:/usr/lib/python3.5/plat-x86_64-linux-gnu:/usr/lib/python3.5/lib-dynload:/usr/local/lib/python3.5/dist-packages:/usr/lib/python3/dist-packages[0m
    [34mSM_USER_ENTRY_POINT=train.py[0m
    [34mSM_FRAMEWORK_PARAMS={}[0m
    [34mSM_LOG_LEVEL=20[0m
    [34mSM_NUM_CPUS=4[0m
    [34mSM_MODULE_NAME=train[0m
    [34mSM_OUTPUT_DATA_DIR=/opt/ml/output/data[0m
    [34mSM_RESOURCE_CONFIG={"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"}[0m
    [34mSM_CHANNEL_TRAINING=/opt/ml/input/data/training[0m
    [34mSM_INPUT_DATA_CONFIG={"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}[0m
    [34mSM_MODEL_DIR=/opt/ml/model[0m
    [34mSM_HPS={"epochs":10,"hidden_dim":200}[0m
    [34mSM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"training":"/opt/ml/input/data/training"},"current_host":"algo-1","framework_module":"sagemaker_pytorch_container.training:main","hosts":["algo-1"],"hyperparameters":{"epochs":10,"hidden_dim":200},"input_config_dir":"/opt/ml/input/config","input_data_config":{"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","job_name":"sagemaker-pytorch-2020-11-21-18-15-53-195","log_level":20,"model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-ap-south-1-615025276797/sagemaker-pytorch-2020-11-21-18-15-53-195/source/sourcedir.tar.gz","module_name":"train","network_interface_name":"eth0","num_cpus":4,"num_gpus":1,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"},"user_entry_point":"train.py"}[0m
    [34mSM_HOSTS=["algo-1"][0m
    [34mSM_USER_ARGS=["--epochs","10","--hidden_dim","200"][0m
    [34mSM_INPUT_DIR=/opt/ml/input[0m
    [34mSM_CHANNELS=["training"][0m
    [34mSM_MODULE_DIR=s3://sagemaker-ap-south-1-615025276797/sagemaker-pytorch-2020-11-21-18-15-53-195/source/sourcedir.tar.gz[0m
    [34mSM_HP_EPOCHS=10[0m
    [34mSM_CURRENT_HOST=algo-1[0m
    [34mSM_NUM_GPUS=1[0m
    [34mSM_OUTPUT_DIR=/opt/ml/output[0m
    [34mSM_FRAMEWORK_MODULE=sagemaker_pytorch_container.training:main[0m
    [34mSM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate[0m
    [34mSM_NETWORK_INTERFACE_NAME=eth0[0m
    [34mSM_INPUT_CONFIG_DIR=/opt/ml/input/config
    [0m
    [34mInvoking script with the following command:
    [0m
    [34m/usr/bin/python -m train --epochs 10 --hidden_dim 200
    
    [0m
    [34mUsing device cuda.[0m
    [34mGet train data loader.[0m
    [34mModel loaded with embedding_dim 32, hidden_dim 200, vocab_size 5000.[0m
    [34mEpoch: 1, BCELoss: 0.6715617362333803[0m
    [34mEpoch: 2, BCELoss: 0.5966293142766369[0m
    [34mEpoch: 3, BCELoss: 0.5062848998575794[0m
    [34mEpoch: 4, BCELoss: 0.4297685410295214[0m
    [34mEpoch: 5, BCELoss: 0.3867273573972741[0m
    [34mEpoch: 6, BCELoss: 0.35295430312351306[0m
    [34mEpoch: 7, BCELoss: 0.3220365698240241[0m
    [34mEpoch: 8, BCELoss: 0.3051981919882249[0m
    [34mEpoch: 9, BCELoss: 0.29086542585674596[0m
    [34mEpoch: 10, BCELoss: 0.2794098930091274[0m
    [34m2020-11-21 18:23:02,495 sagemaker-containers INFO     Reporting training SUCCESS[0m
    
    2020-11-21 18:23:10 Uploading - Uploading generated training model
    2020-11-21 18:23:10 Completed - Training job completed
    Training seconds: 274
    Billable seconds: 274
    

## Deploy the model for testing


```python
# Deploy the trained model
predictor = estimator.deploy(initial_instance_count = 1,
                            instance_type = 'ml.m4.xlarge')
```

    Parameter image will be renamed to image_uri in SageMaker Python SDK v2.
    'create_image_uri' will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.
    

    ---------------!

## Use the model for testing

Once deployed, I will read in the test data, send it to the deployed model, and use the results to determine accuracy.


```python
test_X = pd.concat([pd.DataFrame(test_X_len), pd.DataFrame(test_X)], axis=1)
```


```python
# Split the data into chunks and send each chunk seperately, accumulating the results.

def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = np.array([])
    for array in split_array:
        predictions = np.append(predictions, predictor.predict(array))
    
    return predictions
```


```python
predictions = predict(test_X.values)
predictions = [round(num) for num in predictions]
```


```python
from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)
```




    0.83912



Not too bad, 83%

### More testing

So the model works well on the processed reviews. I'll now pass a stray review to see how the network deals with unseen data points.


```python
test_review = 'The simplest pleasures in life are the best, and this film is one of them. Combining a rather basic storyline of love and adventure this movie transcends the usual weekend fair with wit and unmitigated charm.'
```

I do the same preprocessing here. Using the `review_to_words` and `convert_and_pad` methods from section one, convert `test_review` into a numpy array `test_data` suitable to send to our model.


```python
# TODO: Convert test_review into a form usable by the model and save the results in test_data
test_data = None

test_review_words = review_to_words(test_review)

test_data, test_data_len = convert_and_pad_data(word_dict, test_review_words)

test_data = pd.concat([pd.DataFrame(test_data_len), pd.DataFrame(test_data)], axis = 1)
```


```python
predictor.predict(test_data)
```




    array([0.37846673, 0.58298856, 0.27177435, 0.54650784, 0.41799745,
           0.4034507 , 0.610804  , 0.51276743, 0.46907002, 0.5319264 ,
           0.23136131, 0.6319357 , 0.34935707, 0.6316328 , 0.68350583,
           0.38812724, 0.46733084, 0.41334122, 0.6815751 , 0.5702613 ],
          dtype=float32)



Since the value returned by the model is close to `1`, it think the review we submitted is positive.

### Delete the endpoint

Now I will delete my endpoint, we don't want AWS bills maxing out my credit card, and pushing me into debt.


```python
estimator.delete_endpoint()
```

    estimator.delete_endpoint() will be deprecated in SageMaker Python SDK v2. Please use the delete_endpoint() function on your predictor instead.
    

## Deploy the model for the web app

The model works fine. I made a simple custome interface to submit reviews and have the model predict its sentiment.


```python
!pygmentize serve/predict.py
```

    [34mimport[39;49;00m [04m[36margparse[39;49;00m
    [34mimport[39;49;00m [04m[36mjson[39;49;00m
    [34mimport[39;49;00m [04m[36mos[39;49;00m
    [34mimport[39;49;00m [04m[36mpickle[39;49;00m
    [34mimport[39;49;00m [04m[36msys[39;49;00m
    [34mimport[39;49;00m [04m[36msagemaker_containers[39;49;00m
    [34mimport[39;49;00m [04m[36mpandas[39;49;00m [34mas[39;49;00m [04m[36mpd[39;49;00m
    [34mimport[39;49;00m [04m[36mnumpy[39;49;00m [34mas[39;49;00m [04m[36mnp[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m[04m[36m.[39;49;00m[04m[36mnn[39;49;00m [34mas[39;49;00m [04m[36mnn[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m[04m[36m.[39;49;00m[04m[36moptim[39;49;00m [34mas[39;49;00m [04m[36moptim[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m[04m[36m.[39;49;00m[04m[36mutils[39;49;00m[04m[36m.[39;49;00m[04m[36mdata[39;49;00m
    
    [34mfrom[39;49;00m [04m[36mmodel[39;49;00m [34mimport[39;49;00m LSTMClassifier
    
    [34mfrom[39;49;00m [04m[36mutils[39;49;00m [34mimport[39;49;00m review_to_words, convert_and_pad
    
    [34mdef[39;49;00m [32mmodel_fn[39;49;00m(model_dir):
        [33m"""Load the PyTorch model from the `model_dir` directory."""[39;49;00m
        [36mprint[39;49;00m([33m"[39;49;00m[33mLoading model.[39;49;00m[33m"[39;49;00m)
    
        [37m# First, load the parameters used to create the model.[39;49;00m
        model_info = {}
        model_info_path = os.path.join(model_dir, [33m'[39;49;00m[33mmodel_info.pth[39;49;00m[33m'[39;49;00m)
        [34mwith[39;49;00m [36mopen[39;49;00m(model_info_path, [33m'[39;49;00m[33mrb[39;49;00m[33m'[39;49;00m) [34mas[39;49;00m f:
            model_info = torch.load(f)
    
        [36mprint[39;49;00m([33m"[39;49;00m[33mmodel_info: [39;49;00m[33m{}[39;49;00m[33m"[39;49;00m.format(model_info))
    
        [37m# Determine the device and construct the model.[39;49;00m
        device = torch.device([33m"[39;49;00m[33mcuda[39;49;00m[33m"[39;49;00m [34mif[39;49;00m torch.cuda.is_available() [34melse[39;49;00m [33m"[39;49;00m[33mcpu[39;49;00m[33m"[39;49;00m)
        model = LSTMClassifier(model_info[[33m'[39;49;00m[33membedding_dim[39;49;00m[33m'[39;49;00m], model_info[[33m'[39;49;00m[33mhidden_dim[39;49;00m[33m'[39;49;00m], model_info[[33m'[39;49;00m[33mvocab_size[39;49;00m[33m'[39;49;00m])
    
        [37m# Load the store model parameters.[39;49;00m
        model_path = os.path.join(model_dir, [33m'[39;49;00m[33mmodel.pth[39;49;00m[33m'[39;49;00m)
        [34mwith[39;49;00m [36mopen[39;49;00m(model_path, [33m'[39;49;00m[33mrb[39;49;00m[33m'[39;49;00m) [34mas[39;49;00m f:
            model.load_state_dict(torch.load(f))
    
        [37m# Load the saved word_dict.[39;49;00m
        word_dict_path = os.path.join(model_dir, [33m'[39;49;00m[33mword_dict.pkl[39;49;00m[33m'[39;49;00m)
        [34mwith[39;49;00m [36mopen[39;49;00m(word_dict_path, [33m'[39;49;00m[33mrb[39;49;00m[33m'[39;49;00m) [34mas[39;49;00m f:
            model.word_dict = pickle.load(f)
    
        model.to(device).eval()
    
        [36mprint[39;49;00m([33m"[39;49;00m[33mDone loading model.[39;49;00m[33m"[39;49;00m)
        [34mreturn[39;49;00m model
    
    [34mdef[39;49;00m [32minput_fn[39;49;00m(serialized_input_data, content_type):
        [36mprint[39;49;00m([33m'[39;49;00m[33mDeserializing the input data.[39;49;00m[33m'[39;49;00m)
        [34mif[39;49;00m content_type == [33m'[39;49;00m[33mtext/plain[39;49;00m[33m'[39;49;00m:
            data = serialized_input_data.decode([33m'[39;49;00m[33mutf-8[39;49;00m[33m'[39;49;00m)
            [34mreturn[39;49;00m data
        [34mraise[39;49;00m [36mException[39;49;00m([33m'[39;49;00m[33mRequested unsupported ContentType in content_type: [39;49;00m[33m'[39;49;00m + content_type)
    
    [34mdef[39;49;00m [32moutput_fn[39;49;00m(prediction_output, accept):
        [36mprint[39;49;00m([33m'[39;49;00m[33mSerializing the generated output.[39;49;00m[33m'[39;49;00m)
        [34mreturn[39;49;00m [36mstr[39;49;00m(prediction_output)
    
    [34mdef[39;49;00m [32mpredict_fn[39;49;00m(input_data, model):
        [36mprint[39;49;00m([33m'[39;49;00m[33mInferring sentiment of input data.[39;49;00m[33m'[39;49;00m)
    
        device = torch.device([33m"[39;49;00m[33mcuda[39;49;00m[33m"[39;49;00m [34mif[39;49;00m torch.cuda.is_available() [34melse[39;49;00m [33m"[39;49;00m[33mcpu[39;49;00m[33m"[39;49;00m)
        
        [34mif[39;49;00m model.word_dict [35mis[39;49;00m [34mNone[39;49;00m:
            [34mraise[39;49;00m [36mException[39;49;00m([33m'[39;49;00m[33mModel has not been loaded properly, no word_dict.[39;49;00m[33m'[39;49;00m)
        
        [37m# TODO: Process input_data so that it is ready to be sent to our model.[39;49;00m
        [37m#       You should produce two variables:[39;49;00m
        [37m#         data_X   - A sequence of length 500 which represents the converted review[39;49;00m
        [37m#         data_len - The length of the review[39;49;00m
    
        data_X = [34mNone[39;49;00m
        data_len = [34mNone[39;49;00m
        
        data_words = review_to_words(input_data)
        data_X, data_len = convert_and_pad(model.word_dict, data_words)
    
        [37m# Using data_X and data_len we construct an appropriate input tensor. Remember[39;49;00m
        [37m# that our model expects input data of the form 'len, review[500]'.[39;49;00m
        data_pack = np.hstack((data_len, data_X))
        data_pack = data_pack.reshape([34m1[39;49;00m, -[34m1[39;49;00m)
        
        data = torch.from_numpy(data_pack)
        data = data.to(device)
    
        [37m# Make sure to put the model into evaluation mode[39;49;00m
        model.eval()
    
        [37m# TODO: Compute the result of applying the model to the input data. The variable `result` should[39;49;00m
        [37m#       be a numpy array which contains a single integer which is either 1 or 0[39;49;00m
        
        [34mwith[39;49;00m torch.no_grad():
            output = model.forward(data)
    
        result = np.round(output.numpy())
    
        [34mreturn[39;49;00m result


### Deploying the model

Now that the custom inference code has been written, I will create and deploy the model. To begin I construct a new PyTorchModel object which points to the model artifacts created during training and to the inference code, then I call the deploy method to launch the deployment container.


```python
from sagemaker.predictor import RealTimePredictor
from sagemaker.pytorch import PyTorchModel

class StringPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(StringPredictor, self).__init__(endpoint_name, sagemaker_session, content_type='text/plain')

model = PyTorchModel(model_data=estimator.model_data,
                     role = role,
                     framework_version='0.4.0',
                     py_version = 'py3',
                     entry_point='predict.py',
                     source_dir='serve',
                     predictor_cls=StringPredictor)
predictor = model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

    Parameter image will be renamed to image_uri in SageMaker Python SDK v2.
    'create_image_uri' will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.
    

    ---------------!

### Testing the model

With the model deployed, I test to see if everything is working by loading the first `250` positive and negative reviews, send them to the endpoint, then collect the results.


```python
import glob

def test_reviews(data_dir='../data/aclImdb', stop=250):
    
    results = []
    ground = []
    
    # We make sure to test both positive and negative reviews    
    for sentiment in ['pos', 'neg']:
        
        path = os.path.join(data_dir, 'test', sentiment, '*.txt')
        files = glob.glob(path)
        
        files_read = 0
        
        print('Starting ', sentiment, ' files')
        
        # Iterate through the files and send them to the predictor
        for f in files:
            with open(f) as review:
                # First, we store the ground truth (was the review positive or negative)
                if sentiment == 'pos':
                    ground.append(1)
                else:
                    ground.append(0)
                # Read in the review and convert to 'utf-8' for transmission via HTTP
                review_input = review.read().encode('utf-8')
                # Send the review to the predictor and store the results
                results.append(int(float(predictor.predict(review_input))))
                
            # Sending reviews to our endpoint one at a time takes a while so we
            # only send a small number of reviews
            files_read += 1
            if files_read == stop:
                break
            
    return ground, results
```


```python
ground, results = test_reviews()
```

    Starting  pos  files
    Starting  neg  files
    


```python
from sklearn.metrics import accuracy_score
accuracy_score(ground, results)
```




    0.848




```python
predictor.predict(test_review)
```




    b'1.0'



The last step is to create a simple web page for the app.

## Use the model for the web app
On the far left is our web app that collects a user's movie review, sends it off and expects a positive or negative sentiment in return.

To get a user submitted movie review to our SageMaker model, I will use a Lambda function with permission to send and receive data from the endpoint. Next, I will use the API Gateway to create a new endpoint. 


### Setting up a Lambda function

Here's what the Lambda function looks like: 

```python
import boto3

def lambda_handler(event, context):

    # The SageMaker runtime is what allows us to invoke the endpoint created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName = '**ENDPOINT NAME HERE**',    # The name of the endpoint we created
                                       ContentType = 'text/plain',                 # The data format that is expected
                                       Body = event['body'])                       # The actual review

    # The response is an HTTP response whose body contains the result of the inference
    result = response['Body'].read().decode('utf-8')

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : result
    }
```

Once you have copy and pasted the code above into the Lambda code editor, replace the `**ENDPOINT NAME HERE**` portion with the name of the endpoint that we deployed earlier. You can determine the name of the endpoint using the code cell below.


```python
predictor.endpoint
```




    'sagemaker-pytorch-2020-11-21-18-36-20-803'



### Setting up API Gateway

Now that our Lambda function is set up, it is time to create a new API using API Gateway that will trigger the Lambda function we have just created.

## Deploying our web app

Now that we have a publicly available API, we can start using it in a web app. I have a simple html web page, you can find it at `index.html`.

For testing, I took the following review of *Joker* from Rotten Tomatoes; I wanted to see how the algorithm performs when the sentiment of the review is not overt:

> Joker is a subversion of the trope of the hero's journey, made for a villain.

**Result**: *POSITIVE*

Looks like it did a good job! 



```python
predictor.delete_endpoint()
```

That's all folks!
