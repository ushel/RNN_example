# RNN_example


What is word embedding -> is the term used for representation of words for text analysis, typically in the form of real-valued vector that encodes the meaning of the word such that the words 

that are closer in the vector space are expected to be similar in meaning.


Word embedding techniques :- 

1. Count or Frequency (onehotencoding, bagofwords, TF-IDF)

2. Deep Learning trained models (Word2Vec --> 1. CBOW, 2. Skipgram)


Word2Vec uses neural network model to learn word associations from a large corpus of text. Once trained, such model can detect synonymous words or suggest additional words for a 

partial sentence. Word2Vec represents each distinct word with a particular list of numbers called a vector.

Feature representation :- each and every word present in the vocabulary will be converted into a feature representation. features can be (Gender, Royal , Age, Food ..n ) ->  300 Dimensions

when we are training large Word2Vec model we will not able to see features entirely.

|v| -> unique words in my corpus


            Boy         Girl         King       Queen       Apple       Mango

Gender       -1         +1          -0.92      +0.93        0.01         0.05

Royal        0.01      0.02          0.95       0.96        -0.02        +0.02

Age          0.03      0.03          0.75       0.68        0.95         0.96

Food                                                        0.91        0.92

.
.
.
.
.
n


[King - Boy + Queen] = Girl

King [0.95, 0.96]  Man [0.95, 0.98]

Queen [-0.96, 0.95] Women [-0.84, -0.98]


King - Man + Queen = Women


Cosine similarity -> calculate the distance between 2 vectors say king and queen, we use distance formula which says Distance  = 1 - Cosine similarity


Cosine similarity is angle between these 2 vectors say 45 cos(45) = 1/sqrt(2) = 0.7071

Distance = 1 - 0.7071 = 0.29 we can say almost these 2 words are similar. if distance is nearer to 0 then we can say these 2 vectors are similar to each other.

Recommendation movie we can use the cosine using feature representation. 


How word2vec models are created: -

pretrained model

trained model from scratch

1. CBOW -> [continuous bag of words] corpus -> dataset [Deeplearning.ai company is related to data science.]

window size = 5 [is used to create your input data and output data.]

window size 5 means how many word we are selecting initially.

from this 5 words will take the center word 

I/p                                                     O/P
    
Deeplearning.ai, company, related, to                   is           [it should be knowing that what all words are in forward context and what all words are in backward context]

move window by 1 step and take next 5 words

company, is, to, data                                   related

is, related, data, science                               to


you can take any window size, but try to take odd number so that will have equal number of words in forward as well as backward context.

next step is to train model with above input and output

here we can send text directly as input, we need to convert this text into some vectors, to send it as input to NN. 

there are 7 words in vocabulary, so we can use onehot encoding technique.

deeplearning.ai  1 0 0 0 0 0 0

related          0 0 0 1 0 0 0    


CBOW -> Fully connected NN

here input is fix 4 words

input layer -> 7 inputs per word and total number of words going as inputs are 4 words so input size is 7*4 = 28

deeplearning.ai = [1,0,0,0,0,0,0] -> 7 d vector

company = [0,1,0,0,0,0,0]

related = [0,0,0,1,0,0,0]

to = [0,0,0,0,1,0,0]

hidden layer -> window size if 5 -> 5 d vector 7*5 weights

output layer -> 1 word, each word is represented by vector of 7  5*7 weights

each and every node is connected to each and every other node. will have weights

we get output = [0,0,1,0,0,0,0]

window size is 5 means will get output of size 5 for every word word->vector -> size will be 5 vectors.

window size of 5 means want to feature representation with the vector size of 5

deeplearning.ai size of vector will be 5 [0.92,0.94,0.25,0.36,0.45]

Will be using IMDB dataset, 

text reviews and o/p will be positive or negative review using RNN.


Input dataset ---> Feature transformation ---> simple RNN --(.h5)-> streamlit webapp ---> Deployment 


simple RNN --> Embedding Layer and Simple RNN


here we will pass input as text to RNN but RNN does not get text and hence need to convert it into vectors

how words are converted into vectors, here embedding layers comes into picture.

This embedding layer is responsible in converting words into vectors. it uses word embeddings(word2Vec)


Embedding layers

Word embedding -->

technique converting words into vectors.

it will take word embedding technique where it will take words as input and will give embedding as output.

word embedding --> Feature representation

Dataset

Text                        output
<x11,x12,x13,x14>              0  

<x21,x22,x23,x24>              1

----------------               -

----------------               -


xit --> convert these words to vector using embedding layers.

techniques

1. OneHotEncoding

|v| = 10,000(vocabulary size) consider onehot represetantion man = 10,000 dimensions [0,0,1,0.....0], here in this vocabulary whichever index man word is present that will be one 

and remaining all will be 0. 

Boy = [0,0,0,0,0,1,0,0,.......0] index 2000 = 1


sparse metrics -> problem -> sparse matrix leads to overfitting as you just have 0's and 1's so it not an efficient techniques if vocabulary size is large.


2. word embedding:- [word2vec] can be used in embedding layer

it creates feature representation for every word that is available in dataset

|v| = 10,000

Boy    [2000]

Girl   [5000]

King   [6000]

Queen  [9000]

Apple  [1000]

Mango  [7000]

How every word is converted into vectors 

select feature representation with 300 dimension and represent each word in 300 dimensions.

            Boy         Girl        King        Queen       Apple       Mango

Gender      -1          +1          -0.92       +0.92        0.0        0.1     

Royal       0.01        0.03        0.95        0.96         -0.02      0.01

Age

Food

.
.
.
300 dimensions

