# GPT Synopsis

- Split into 2 parts, the summary and the main rubric requirements that I gleaned as I read.

## Concepts

Key Concepts

- Objective - Learn a general way way to transfer the meaning of words with little fine tuning to a bunch of tasks.
- They choose to focus on tasks: natural language inference, question answering, semantic similarity, and text classification
- Unsupervised Pretraining - to find a good initializeation point. Used in previous works, like image classification, regression, and it acts as a good regularization scheme
- They also chose to keep the transformers (self attention) and word embedding plus positional encoding feature.

Why did the authors do this?

- They choose to use the transformer for their architecture as it’s a strong foundation and it outperforms alternatives like RNNs, and are able to capture longer range of language, compared to something like LSTM.

Strengths

- Unsupervised Pretraining: The use of unsupervised pretraining allows the model to learn a general representation of language from vast amounts of unlabeled data. This serves as an excellent initialization point for fine-tuning on specific tasks, leading to better performance and data efficiency.

- Minimal Task-Specific Changes: Unlike previous models that required substantial architectural changes and new parameters for each specific task, GPT-1 needs minimal modifications. This makes it more efficient and versatile across different NLP tasks.

- Transformer Architecture: The choice of the transformer architecture, with its self-attention mechanism, provides a strong foundation. Transformers outperform traditional RNNs and LSTMs, especially in capturing long-range dependencies in text, which is crucial for understanding context and meaning.

Novelty

- Improving upon NLP tasks, using unsupervised learning.
- Previous models, when fine tuning to a specific task required a lot of new parameters for each task. But GPT requires minimal changes to the model architecture when fine tuning to a specific task. And it captures long range of language thanks to the transformer architecture.

Previous Works

- Transformers (except they used sinusoidal and GPT ditched the encoders all together) also revolutionized NLP by using self-attention mechanisms to process sequences of text. Also demonstratably better than RNNs and LSTMs at NLP tasks.
- perhaps previous didn’t use task aware input transformation
- pre trained word embeddings. In the introduction they talk about how even if supervised learning was feasible, unsupervised learning can still provide a performance boost, and they even give an example with pre-trained word embeddings.
- Building upon word embeddings, so they can train on unlabeled texts.

Limitations

- Are the word embeddings they’re using, the most effective way?
- Well maybe the way they’re using supervised learning (by at the end just feeding it into a linear layer to predict the next token) could be improved upon. Is a linear layer the best way, or are there better ways to learn to predict next token after the activation function of the final decoder blocks.

Challenges / Weaknesses

- Natural Language inference was a pretty big challenge as there’s a lot of deepness to the text, that makes it hard to model correctly.
- Performed worse than a biLSTM model in RTE datasets, although they believe can improve it with multi task training

# Synopsis

There are a lot of unlabeled text corpora. But they’re not labeled. The challenge that they are solving, is that they can train models generative pre training, followed by discriminative fine tuning.

With generative pre-training, they made huge gains on

textual entailment, question answering, semantic similarity assessment, and
document classification.

## Introduction

Unsupervised learning in NLP, is very helpful, and there’s been research in the past, showing how this is the case `pre-trained word embeddings`

It’s hard to gain useful information to feed into the machine from straight up unlabeled texts. The problem is what are we trying to get from the words.

And even with the methods we do have, it’s hard to know if the preexisting methods are even effective at solving their respective problem.

There using `unsupervised pre-training` and `supervised fine-tuning`

- There had been previous papers that did this type of training and had improvements across NLP benchmarks. One which they even compared to was ELMo, and another was BERT . These were models that utilized this approach, and were state of the art, making their decision to use this strategy seem more compelling.
- The model basically generalizes during the pretraining and transfer learns the fine tuning context.

They choose the transformer as it provides strong results on language modeling tasks. Better than RNNs and LSTMs used previously.

## Framework

Two stages

1st stage - Training it on a large corpus

2nd - Fine tuning it on a specific tasks, this time supervised.

They’re trying to maximize the amount of words predicted correctly by maximizing the probability of token prediction by using the equation

$$
L_1(\mu) =\Sigma logP(u_i|u_{i-k}\dots,u_{i-1};\Theta)
$$

They use a multi-layer decoder blocks for the language model. Basically a multi-headed self attention operation, followed with feed forward layer to produce and output distribution over target tokens.

Equations is basically the transformer embeddings, plus the positional encoding,

Put that through the transformer blocks perform a SoftMax.

In my code I chose to use `nn.Embedding` for easier use

```cpp
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # This is the token embedding
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table=nn.Embedding(block_size, n_embed)

       # Forward Methods where tok and pos embedding are used
     def forward(self, index, targets=None):

        B, T = index.shape
        #
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # print(f'Pre')
        x = tok_emb + pos_emb
```

### Supervised Fine Tuning

Now after training with the previous steps (word embeddings) and transformer decoding, they now move on to the supervised target tasks.

Now what they do, they have labeled tokens. with label y. They feed these tokens to the activation, and then they feed it into another linear layer with parameters Wy to predict y.

That equation looks something like

$$
P(y|x^1, \dots, x^m) = \text{softmax}(h_l^m W_y)
$$

And maximizing objective equation looks like. Basically maximizing probability of predicting the next tokens given the previous series of tokens.

$$
L_2(C) = \Sigma logP(y|x^1, \dots, x^m )
$$

To maximize the objective;

They also added a secondary objective to the finetuning process. In hopes of improving generalization of the supervised model, and to accelerate convergence. **Previous work** shows improved performance with this type of an auxiliary objective. They optimize for this objective using

$$
L_3(C) = L_2(C) + \lambda * L_1(C)
$$

(I chose to ignore this auxiliary objective as I implemented the base model and have not focused on any fine tuning yet.)

For each task they have a specific way to fine tune for it. E.g.

### Task Specific Input Transformations

For some tasks like text classifying they can just directly fine tune the model. But something else like question answering or textual entailment, they have structured inputs such as ordered sentence pairs.

There are some modifications.

**Entailment** - concatenate premise and hypothesis with delimiter

**Similarity** - no ordering, but modify sequence to contain both possible orderings with delimiter and process each independently and then add them element wise before going into the linear output layer

**Question and Answering and Reasoning. -** They have a context document, question q, set of possible answers. Concatenate each document, question, and possible answers with delimiters. And process these independently and normalize with a SoftMax to get output distribution over possible answers.

## Experiments

They used books corpus for training the language model. Basically bunch of books, and used because of long stretches of contiguous text. Which was one of their main objectives from the beginning ( a language model that’s general, if it was fine tuned to say medical scenarios I assume a lot of it would be trained on textbooks)

### Specs they used

- 12 blocks of decoders (with 12 attention heads each and 768 dimensional states)
- FFN - 3072 dimensional inner states
- Adam Optimizer
- Max learning rate 2.5e-4 increased linearly from zero over first 2000 updates.
- Train for 100 epochs on minibatches of 64 contiguous seq of 512 tokens.
- `Used learned positional encodings instead of the sinusoidal used in attention is all you need`
  - Why did they do this?
    - Learned Position embeddings allow more flexibility, unlike fixed sinusoidal patterns
    - The model can also learn to optimize these as well, so more information that can help the model represent language better
    - Embeddings also generally increases performance of the model
- tokenized using spacy, cleaned using ftfy
- add dropout to the classifier with a rate of 0.1.
- learning rate of 6.25e-5 and a batch size of 32. Our model finetunes quickly and
- 3 epochs of training was sufficient for most cases.
- use a linear learning rate decay schedule with warmup over 0.2% of training. λ
  was set to 0.

### Supervised Fine Tuning

Performed experiments on a variety of tasks, (such as GLUE benchmark

- `The task of Natural language inference was challenging as there is a lot of deepness to the text like lexical entalment, coreference, and lexical and syntactic ambiguity`
  - Why did they say this?
    - Language is complicated and even for humans it’s hard to absorb and truly understand text
    - There are relationships across hundreds of pages, language can become vague if improperly or intentionally written.
    - There are many nuances and special cases for the model to learn. Like words spelled the same can mean different things based on the context. So the model has to track tokens across sentences and even pages to model the true meaning accurately
- The Model is better than previous benchmarks’ baselines, and is better able to reason over multiple sentences and even handle some linguistic ambiguity
- **Results**
  - 56% accuracy om RTE
    - below biLSTM model
  - They have `not explored multi task training yet but they believe it will benefit in the future`
  - Why will it benefit with multi task training?
    - They say this because multi task training can help the model generalize better by learning shared representations across different tasks, leading to improved performance on each individual task, as it leverages the commonalities between tasks to enhance the model’s overall understanding and capabilities

## Results Tables

MNLI-m beating every other model (that they provided in the paper)

MNLI-mm - beating every model that they listed

SNLI - again beating models like CAFE and ESIM + ELMo but with smaller leads than others.

QNLI - large margin between GPT and other compared models

RTE - Does worse than GenSen and the BiLSTM + Attn

Beating all models in common sense reasoning in RACE-m RACE-h RACE and Story Cloze

Achieves 91.3 % accuracy on SST-2

Lots of new records in the 9 out of 12 datasets they evaluated on.

**Interesting parts from the analysis**

- They say that the LSTM has higher variance in zero-shot performance, and theorize that the transformer is doing better due to it’s inductive bias.

Through the provided framework, and pretraining on a diverse corpus with long stretches of text with long range dependencies using the decoder transformers, them model is able to transfer the learning and apply this to tasks like question answering, semantic similarity assessment, entailment determination, and text classification.

They managed to improve on 9 of the 12 datasets. They suggest that unsupervised pretraining as a baseline and the finetuning is what can work best in the future.

## Conclusion

The framework provided is a way to train a model on learning natural language through generative pretraining and supervised fine tuning after. They are able to improve on the best results in 9 of the 12 datasets that they have evaluated. They say their work suggests that achieving significant performance gains is possible and transformers will work best in the approach of natural language understanding and other areas of NLP.
