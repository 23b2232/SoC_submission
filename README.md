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
