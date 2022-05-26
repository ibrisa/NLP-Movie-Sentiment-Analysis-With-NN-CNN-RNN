# NLP Movie Sentiment Analysis with a Simple Neural Network, Convolutional Neural Network and Recurrent Neural Network and Various Lemmatizers
For Machine Learning for Science and Profit class, May 2021

## Motivation
Natural Language Processing is an area of machine learning that is fascinating because of its wide spread applicability and in the area of sentiment analysis, the ability to extract emotions from a sentence, something even humans sometimes fail to do. As day by day, more communication is replaced with text instead of conversations, text analysis and NLP's role can grow even larger.

For this assignment for my Machine Learning for Science and Profit class, I use the IMDB Dataset obtained from Kaggle, which contains 50,000 movie reviews of which there are an equal amount of negative or positive polarizing reviews. The goal of this assignment is two-fold, obtain the highest classification accuracy by testing a variety of models while also testing different stemming and lemmatization methods to see how this can influence the classification accuracy.

As NLP deals with text, the same word might be used a variety of times inflected in different forms (e.g. befuddle, befuddled, befuddling) which are called a lexeme. Ideally, the algorithm should treat all of the occurrences of that word as the same word (e.g. befuddle), or the base word that you would find in the dictionary, also called the lemma (hence lemmatization). There are many options to translate the inflections into a single word - lemmatization techniques find the lemma, while stemming techniques find the part of the word that is responsible for its lexical meaning, which can often result in truncated looking words. In this assignment, I test the following stemming and lemmatization techniques:

- Snowball Stemmer
- Lancaster Stemmer
- Word Net Lemmatizer
- SpaCy's Lemmatizer

For the model section of this assignment, I test each of the stemming/lemmatization techniques on:
- A simple neural network
- A convolutional neural network
- A recurrent neural network with LSTM


## Models
The following portions of the model implementation are the same across each of the neural networks, so the choices are rationalized here:

- Model: I use the sequential model as I only have one input, the movie review, and one output, the binary classification of sentiment.
- Embedding Layer: The first layer is the embedding layer which turns integers into vectors. It has a vocabulary size of whichever lemmatizer option is being used. For the weights, I use the word embedding matrix that was built previously, which also constrains the output dimensions, as I used the 100-dimensional version of GloVe.
- Dropout Layers: I added dropout layers to the models in an attempt to prevent overfitting, which randomly selects 20% of the neurons to be ignored.
- Final Layer: For all of the neural networks, I use a final Dense layer with sigmoid activation for the output layer, as the final goal is to predict a probability as an output, and the sigmoid function transforms the input into a range between 0 and 1.
- Optimizer: I initially tested both the SGD and Adam optimizer after a literature review noted Adam was faster, but sometimes has errors which SGD doesn't, but there wasn't that much variation in the accuracy scores, so I ultimately used the Adam optimizer.
- Loss Function: I use binary cross entropy for the loss function, as we're attempting a binary classification.
- Fit: I fit the model with a validation split, setting aside 20% of the training data to use as a validation set, so after every epoch, the loss and accuracy is evaluated on this portion of the data.
- Epochs: I train with few epochs to prevent the model from memorizing the data and overfitting

#### Simple Neural Network

The first model that I'll train is a simple neural network. It uses 1 of each of the previously explained layers.

#### Convolutional Neural Network

The second model I train is a convolutional neural network which even though traditionally applied to images, can also perform well with NLP tasks. I used a 1D CNN with an embedding layer constructed like the simple NN and a mix of convolutional layers with the non linear ReLu function as the activation function and pooling layers. The pooling layers help prevent overfitting through down-sampling, since the reduction of size in this layer leads to some information loss.

#### Recurrent Neural Network

Recurrent neural networks work well with sequential data, by reusing activations of previous nodes or later nodes in the sequence to influence the output, so it's well suited to be applied to detect sentiment in a sequence of words in a review. The Long Short Term Memory network has been classically applied to text-related problems and here, I apply it with a combination of the same embedding layer, droupout layers and dense output layer as described previously.


## Results

Select screenshots. For additional results see the ipynb.

Comparison of Model Classification Accuracy

<img width="148" alt="Screen Shot 2022-05-26 at 9 26 31 AM" src="https://user-images.githubusercontent.com/55218727/170532604-a082d509-5c98-4e1f-baa6-be0811ba9e21.png">

Comparison of Stemmers and Lemmatizers

<img width="281" alt="Screen Shot 2022-05-26 at 9 26 20 AM" src="https://user-images.githubusercontent.com/55218727/170532620-2b458e94-693e-4edb-b93c-81ef53c2a0fb.png">

CNN Model Accuracy using Snowball Stemmer

<img width="300" alt="image" src="https://user-images.githubusercontent.com/55218727/170532747-71ac5d71-696e-4058-8955-33dca3ca6be8.png">

CNN Model Accuracy using WordNet Lemmatizer

<img width="300" alt="image" src="https://user-images.githubusercontent.com/55218727/170532887-8a4c14b3-f844-42a2-9ae4-6f2ccd1adfc0.png">
