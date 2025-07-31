Building an LLM from scratch


Lecture Notes (Only the most important things):

LLM building methodology: 
Gathering of large amount of unlabelled text → Pretraining model on this data (unsupervised learning forms a base/foundational model) → Fine-tuning the LLM on labelled data (supervised learning) for specific applications 
Pretrained model can be trained for as simple task as sentence or text completion and be able to perform diversity of tasks 
Fine Tuning: 1. Instruction based 2. Classification (labelled data required for both)
Transformers: :
Most LLMs rely on the transformer architecture
Transformer is based on deep neural network architecture
Original transformer: for language translation tasks
Transformer architecture: Input à Preprocessing (breaking down of input word or sentence into tokens or smaller pieces and assign unique id to each) à Encoder (vector embedding: capture the semantic relation/meaning between the words or tokens vectorized representations) à Vector Embeddings à Decoder (generate output text from partial input text and embedded vector)
Right hand – side: Input (Translated text / Partial Input text) à Preprocessing à Decoder (generate translated text one word at a time) à Output layer
Tasks are done work/ token at a time. So along with the last token it also receives as input already translated or worked upon tokens (Partial output text)
Attention mechanism: allows us to model dependencies between different words despite the distance between them
Self-attention mechanism: allows the model to weigh the importance of different words relative to each other in a model. Enables long range dependencies
Long range dependencies: to predict next word or sentence check for its dependencies on other words far back
So which word to pay more attention to in order to predict next word
BERT: Bidirectional encoder representations from transformers – predicts hidden words in a given sentence (fills missing words). Random words can be masked
Looks at sentence from both directions, relationships between words understood; does not have decoder only encoder
GPT: Generative pre-trained transformers – generates new words, looks only left to right (unidirectional way of looking at sentence); produces one word at a time; does not have encoder only decoder
Transformers: also used for CV
LLMs: can be based on other architectures recurrent or convolutional or LSTMs
Transformers != LLMs
GPT = transformers + unsupervised learning data
Zero shot: no description is provided with the task
Few shots: some description or supporting text provided with the task. If only one example provided then it is one shot learning 
GPT-3 is a few shot learner, examples help generate accurate predictions
Token: unit of text that which model reads
Autoregressive model: label itself is the next word predicted, previous output is used as input for the future predictions, we use the structure of the data itself to get labels
Emergent behaviour: GPT is trained only for next word prediction but along with this it can perform other tasks as well it developed these on its own; being able to perform tasks it was not trained for
Part of sentence used for training next for testing or predicting next word 
Tokenization: 
split the text into words and subwords as tokens 
Assign token ids to each token
Encode token ids into vector representations 
Removing whitespaces during tokenization is a matter of application while it saves computation it can be useful for being more sensitive to the exact structure
Vocabulary is a list of our token with tokens sorted in alphabetical order
Assign a unique token id to each item in the list a unique integer to each
Vocabulary is a dictionary mapping sorted tokens to ids
Unk or unknown is a token for unknown words. Any unknown word encountered gets this token id.
End of text indicates the start or end of a segment for different text sources as there are any training sources 
Chatgpt tokenizer only uses <endoftext> tokens for simplicity
Byte pair encoding for unknown tokens for making subwords
Byte Pair Encoding:
There are 3 types of Tokenizer algorithms: word, sub-word, character based
Word-based: each individual word is a token. Words not present in the training set or source are called out of vocabulary (OOV) words. They are difficult to deal with here as it increases the demand of the sources 
Similarity between words is not captured and each word is treated as separate despite being closely related 
Character based: individual characters are considered as tokens
Character based tokenizers have very small vocabulary considering the limited number of characters (256 for English). This solves the OOV problem
Problem: Meanings associated with words is completely lost.Tokenized sequence is also much longer than the initial raw text
Sub-word tokenization: Byte pair tokenization is an example of this
Rules: 1. Do not split frequently used words in subwords
2. Split rare words into smaller meaningful subwords
Pros: helps models understand root words and the similarity in their meanings. Understands uses of common suffix and use in syntactic situations
Byte pair encoding (BPE) algorithm: replace the byte pair that occurs the most in the data with the one that doesn’t occur in the data. Repeat till no byte pair occurs more than once. This compresses the data. This process is called Encoding.
Rest rules stay the same for bite pair encoding 
A preprocessing step exists before tokenization which means that at the end of word an end token </w> is added to symbolize the end of the word 
Step 1: split the words into individual characters and calculate frequency in a frequency table 
Step 2: when we find common pairs, merge them and perform same iteration till token limit is reached 
Merging the pair is what creates a subword. Now this is a single token now, find other pairs occurring a lot and merge and create new token. After doing this also subtract the frequency from the characters frequency 
Only merge the characters when they have enough frequency in tokens
Stopping criteria for token formation could be reaching token limit or just the number of iterations 
Subowrd encoding reduces the number of tokens. Also solves the OOV problem.
50257 tokens int gpt 2,3
OOV problem does not occur as it breaks down new words into sub words or even characters 
Creating input target or input-output pairs:
The word or input given to LLM the target or output is the words which are predicted in that iteration or will be predicted in that iteration. This output word along with previous input is fed as input in the next iteration. Thus, it is an autoregressive model.
During training process words past the target are masked
Context size: number of words we want to/need to give as input for model to predict
Create 2 arrays namely x and y as the input output arrays 
how many words or context the model should pay attention to at a time for predicted output
Data loader: iterates over the input dataset and returns the inputs and targets as PyTorch tensors
Datasets and data loaders help efficient loading or processing of data in a compact manner
Data loader made in a sliding door kind of manner
Input and targets are arranged in the form of tensors
Target or output tensors are just made by shifting the input tensor by one
_getitem will tell the dataloader the type of input and output data is needed
Dataloader would help us do parallel processing and analyze multiple batches at a time 
Batch size is the number of batches that the model can operate at at once; basically the data operated at before updating the parameters 
While actual training of models the context size is set to at least 256
Small batch size leads to quicker updates but they are noisier 
Large batch size will take a lot of time
More overlap between batches could lead to overfitting
So more stride helps avoid overfitting 
Token Embeddings:
We cannot assign random numbers to words/subwords as it does not capture the semantic relationship between the two.
We need to exploit the fact that some words are similar or closer to meaning than others
One-hot encoding also fails to capture the semantic relationship between words
Every word encoded as a vector 
Dimension of vector determined by features. Vectors can capture semantic meanings
Every token converted into a vector: Vector/Token embedding
Neural network is trained for making token/vector embeddings
GPT building did not use a pre trained word2vec but trained it along with the model
Token ids → embedding layer weight matrix which consists of embedding vector dimensions and the number of weights corresponding to it for each token
These weights are randomly initialized at first 
This is the start of the LLM learning process
These parameters are optimized during the training process
The matrix row is the number of tokens while the number of columns is number of characteristics/ features of the vector
Lookup table: access the respective token vector using its vector id
Embedding layer is just like a lookup operation where we can access vector embeddings using token id
The creation of embedding layer weights or lookup operation is similar to that of how neural network works
Embedding layer is much more efficient than neural network operation which is why it is used nn.linear meh computationally inexpensive 
Positional Embedding 
If we do not take the positioning of a word into consideration a word will be assigned to same vector for different sentences/scenarios despite its varying position 
Same token id results in the same embedding vector despite the difference in position. The position information is not exploited 
It is important to inject additional position information 
Two types of positional embeddings: Absolute and Relative 
Absolute is more often used type
Absolute: a unique embedding is added to each token embedding conveying its position 
Final input embedding is the sum of token and positional embedding consisting of positional information 
Both the positional and token embedding have same dimension
FRelative: how far apart different words are not their exact position 
This can generalize better to sequence or embeddings of varying lengths even if those are not encountered during training. Useful for longer paragraphs 
Absolute good when actual position info is necessary like sequence generation, GPT uses this 
Relative good for long sequences where same phrase can occur again and again 
We need to optimize values for both embeddings and it's a part of training process
Encoding or embedding for higher dimensional space 
Dataloader just makes the task of loading data much easier and convenient 
For a dataloader of size 8 by 4 the batch size is 8 but input at a time is 4. This is why positional embedding tensor will be of 4 by 256 which is vector dimension 
For positional embedding vector the row size is context length
Attention Mechanism:
For long complicated sentences it is important for the model to pay attention to only the key details 
For translating or any other type of working with long sequences the direct word by word translation or similar methods cannot contextual understanding and grammar alignment are necessary
Thus model should retain the memory of what came before for context in order to better predict and work
Two submodules in NN encoder and decoder 
A context vector is generated by encoder which is then passed on to the decoder 
The context vector consists the meaning and context of what the input really means and represents 
RNN employed this architecture best before. Hidden state in the RNN consists of memory. As each word proceeds hidden state is updated and finally a final hidden captures everything (this is the context vector) this is then sent to the decoder
The problem here is that the decoder has no access to previous hidden states, only the final hidden state. This results in a loss of context for texts with long sequences and long range dependencies 
Complex sentences contain longer dependencies. RNN struggles with this because of the one final hidden layer and single context vector leading to loss of context
A modified RNN where a decoder can selectively access each different part of the input at different steps. So, that we can decide which token to pay more attention to
Each word is assigned an attention weight which tells how much attention to pay to each token
So basically for example for translation tasks the model can take a look at the entire input context and pay attention to words more important to carry out the task. This is called Dynamic focus. 
While training the attention mechanism understands how to carry out tasks effectively like for translation better arrangement of words rather than mindless word to word translation
Self attention: Rather than looking at different sequences we are looking at a single sequence and tokens within to understand relation between tokens and which tokens to give more attention to carry out certain task
‘Self’ is associated with the ability to compute weights or understand relation between words based on a single (input) sequence thus the word self (input)
Traditional attention mechanisms compare two or more sequences to understand such relations while modern mechanisms use only single input sequences 
Simplified self attention mechanism without trainable weights
The embedding captures the meaning of the but not really its relation with other words in the sentence. 
Goal of attention mechanism is to take the embedding vector and then transform it into context vector 
Context vector is more like an enriched context vector containing relation of the word with other words along with semantic meaning. So it contains information of other tokens too
These are inputs to LLMs
For every word how much attention should be given to other words is given by attention weights 
The word we are working on to find its context vector from embedding vector is called query
How much attention should be paid to each of the words for the query this is quantified by a matrix called attention score 
To Find which tokens to give more importance to we see how token embeddings are aligned with each other. This can be done using dot product 
Dot product to find the attention scores so dot product between query and rest of the tokens. The vectors which are more aligned to each other attend more to each other
After calculating attention scores we normalize the vector, for interpretability to determine how much percentage attention to give the rest of the tokens. Also for maintaining training stability
Attention weights are the normalised scores they sum up to 1
Softmax is preferred for normalization. Normalization should be such that values close to 0 or 1 should be approximate to those so as to not confuse the back propagation model 
Softmax formula: divide the exponent of the values by the summation of the exponents 
Pytorch softmax: Subtract the maximum value from the exponent power value. Done to get away from numerical instabilities which occur during computation like overflow for large values and underflow for small values
The positive output of the exponential function is necessary for interpretability 
Next we multiply each embedding vector by respective attention weights. All of these resultant vectors summed up gives the context vector for that specific query
Compute attention scores → attention weights → context vectors
Trainable weights are necessary because the importance to a particular cannot be given just because of the semantic meaning the current context in which the words are used also matters
Without trainable weights we might end up only looking at words which are similar to the query without taking in account the current context
Self Attention mechanism with trainable weights:
Self attention mechanism is also called as ‘scaled dot product attention’
Trainable weight matrices which can be optimized while training of the model and can create good context vectors 
Self attention mechanism using three weight matrices: Key, Query and Value
First step in converting Embedding vectors to context vectors is converting embedding vectors to key, query and value.
These three matrices are randomly initialised and while training process are optimized
These matrices change the dimension of the input matrix. So by multiplication of input with query weight matrix we get resultant query matrix
Every row of the three vectors corresponds to a singlet token
We are finding the attention scores between the query and the key. How  a particular query attends to different key vectors 
A key step done before normalization of the attention scores is that it is scaled by the square root of the embedding dimension of keys. This is why it is called scaled dot product attention.
Then applied softmax to achieve interpretable attention weights
The scaling is applied because the softmax function is sensitive to its inputs. If a very high input is provided the softmax becomes peaky (provides an unnecessarily high value) 
A very high value of queries or keys leads to very high value of attention scores. This can result in the model becoming way too confident in a key (giving it more attention). To make sure this does not happen scaling is done. Stability in learning
Multiplying two random numbers can increase the variance; dividing by square root specifically can keep the variance close to 1. As the dimensions of key and query increase so does the variance proportional to the dimension value. Diving by sqrt keeps it close to 1. Helps with better learning and computation 
We calculate the context vector by multiplying the value tensor by attention weights 
Query and key can be thought of as referring to a particular token 
Value is the actual representation of the token embedding 
“Once the model retrieves which keys are most important to the query it retrieves the corresponding values.
Causal Attention:
Causal attention also known as masked attention is a form of self-attention 
In this rather than looking at how much the query attends to the entire input sequence we look at how much the query attends to the tokens or inputs before it
Only consider attention scores with the words which come before or at the word 
All the tokens after this word are masked out (made to zero)
For this the context size also matters
Attention weights above the diagonal are set to zero
The weights are again normalized to make sure that the weights sum is zero
All the weights which are masked above the diagonal are called as causal attention masks
We get the attention weights as we did previously then we just mask the upper diagonal of weights and then renormalize s that sum is zero
Data leakage problem: even if we change the weights if future tokens to zero while applying softmax to all the entries the influence of other tokens including future ones is already recorded 
This can be worked out using an upper triangular infinity mask. This means that mask all the values above the triangle with negative infinity then when softmax is applied the values above the triangle will automatically be zero while the rest will sum up to 1
This cancels the influence of future tokens 
Dropout is a neural network technique that ensures that all the neurons are working efficiently so it randomly turns off a few neurons like this all of them participate and provide better generalisation 
This applied to the model tice once after the calculation of the attention scores then after the attention weights are applied to the values of the input vector 
A dropout mask is implemented which tells which neurons are going to be turned off. There is a scale to that it can 50% so at a time 50% of the attention weights will be turned off 
In GPT models the dropout rate is around 0.1 or 0.2 
Since the weights are switched the rest of the on ones are rescaled either halved or doubled according to the dropout rate 
Scaling is done by 1/1-p where p is dropout rate 
Along with all this we will also be creating batched as we want the model to compute multiple sentences at once so multiple batches will compute multiple sentences 
When multiple causal head attentions are stacked together they give the multihead attention 
Multihead Mechanism:
The multi head mechanism refers to dividing the attention mechanism itself in multiple into multiple heads where each head will be operating independently 
The set of one key, value and query matrix comprises of one head 
We stack these multiple single head attention layers on top of each other. Combine all the outputs 
We concatenate all the context vectors received and get the ultimate context vector
Number of columns increase with the stacking of the context vectors
We are running multiple attention mechanisms in parallel and then aggregate them 
Multihead attention with weight splits:
Reducing the number of matrix multiplications by creating one big weights matrix which consists all heads dimensions and then split them based on the number of heads.
This is done to reduce computation power
Ways to interpret dimensions start with the outermost layer where 1 is the batch size then 3 is number of tokens 2 is the number of heads for each token and 3 is dimension of heads.
First the grouping is done according to the number of tokens but now for further calculations we want to group in terms of the number of heads 
The above step is done to find separate attention scores for separate heads 
After the formation of the context vector we are just going to merge or concatenate the two heads so reverse process first we will switch dimensions num_heads with num_tokens just like it was before 
Flatten each token output in a row basically combine heads 
LLM architecture:
Masked multi head attention forms an important part of the transformer block 
First part of Transformer block is layer normalization block then masked multi-head attention 
Then comes a dropout layer this entire thing is a shortcut connection and the output from this goes to another shortcut connection 
this entire architecture consists of trainable weights which are optimized once the model is pretrained 
124 million GPT-2 model will be used 
N_layers is number of transformer layers 
Qkv_bias is query key value bias
Logit output of the model because we have 4 tokens and the output is basically the probabilities of the 50257 tokens occurring after the given token. The word with the most probability is chosen as final predicted word 
So the output is number of tokens times the columns corresponds to number of vocabulary size 
The embedding vector is multiplied by embedding dimension * vocabulary size neural network to gain logit matrix
Till final normalization layer there will be output with number of tokens per batch times embedding dimension 
Output head is the final neural net after which there will be 4 tokens corresponding to which there will be 50257 probabilities of each token
Layer normalization:
The layer normalization comes several ties within the transformer a=block as well as outside it which why a separate class for it makes sense 
Before and after the attention head there are layer normalizations and one after coming out of the transformer block 
These are are used to make the training process more efficient 
Training multi layer neural network can be challenging due to two problems the vanishing or exploding gradient 
Gradient of each layer is highly dependent on the output layer because of back propagation therefore the output being too large or small affect the gradient as well 
Already too large gradient as it back propagates would become too large and this is called exploding gradient 
The same goes for a gradient which is very small. For a very small gradient the learning would stagnate and parameters will not update (parameters updation is dependent on the magnitude on the gradient)
So both large and small affect the learning of the parameters. 
Thus to maintain stability of learning batch normalization is implemented 





