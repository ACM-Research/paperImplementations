## Key Concept

Bidirectional Contextualization:

BERT reads the entire sequence of words at once (bidirectionally, and in a deep way, not just concatenating left to right and right to left vectors [using masked learning]). (unlike previous models) This allows it to capture context from both directions, providing a deeper understanding of word meaning based on surrounding words.

## Strengths

- only one output layer to create models for a wide range of tasks
  > BERT is the first finetuning based representation model that achieves
  > state-of-the-art performance on a large suite
  > of sentence-level and token-level tasks, outperforming many task-specific architectures.
- Next Sentence Prediction
  - BERT does not just transfer sentence embeddings. Instead, BERT transfers all its learned parameters to the downstream tasks. This means BERT uses the entire pre-trained model to initialize the new task, which helps in better performance because it leverages the full context and knowledge learned during pre-training.

## Weaknesses

- BERT models are large and can be slow during inference, which might be impractical for real-time applications or on devices with limited computational power.
  Fixed Input Size:

- BERT has a maximum input size limitation (typically 512 tokens). Longer texts need to be truncated or split, which can result in a loss of context or information.
  Interpretability:

- Like many deep learning models, BERT can be seen as a "black box" with limited interpretability. Understanding how BERT makes specific predictions can be challenging.

## Novelties

- Training deeply on unlabeled text through left and right context (Bidirectional pretraining for language represenmtations)
- Only one
- Using a Masked Language Model (inspiered the Cloze task)

# Summary

- The following is the summary of the bert paper and basically what I gleaned from it. It might have some references

### Prev Lit?

- ELMo generalized trad word embeddings research by extracting context sensitive features from a left to right and right to left language model

## Comparison

- previous works were unidirectional and according to the authors (this limits the choice of architectures that can be used during pretraining)
- IN GPT they only do left to right (because it only predicts based on the context already the sames)

## Code

- Abstract introduce a new language
  Bidirectional Encoder Representation
  Two existing strategies for pretrained language representations to down stream tasks
  **Feature based & Fine Tuning**

Basically masks random words and the model tries to predict the masked words based on only the context of the other words.

## Reviewing Approaches of Pretrained General language representations

2 Steps - pretraining and fine tuning

Pre Training

- trained on unlabeled data over different pretraining tasks

Finetuning

- init with pretrained aparams and then fine tuned using labeled data from down stream tasks

Based on the encoder.

L - encoder blocks

hidden size - H

Heads - A

Uses word piece embeddings

Sentence Pairs - packed in a single sequence. Differentiate sentences in 2 ways

First with a separation token. And then adding learned embeddings to every token indicating it’s sentence locatrion

Adds segment embeddings and position embeddings to the token embeddings (wordpiece emeddings)

Final hidden vectors corresponding to mask tokens are fed into and output softmax over the vocabulary as.

**They masked 15% of wordpiece tokens**

- only reconstruct masked worsd
-
- **80% of the time**, we replace "wo" with [MASK]:
  - Original: "wo rd"
  - Modified: "[MASK] rd"
- **10% of the time**, we replace "wo" with a random token:
  - Original: "wo rd"
  - Modified: "xx rd" (where "xx" is some random token in the vocabulary)
- **10% of the time**, "wo" remains unchanged:
  - Original: "wo rd"
  - Modified: "wo rd"

Using Cross Entropy Loss function for predicting original tokens during pretraining\

## Pre Training

Masked LM.

### Task 2

Next sentence predicityon

take two sentences A - B 50% B is the next sentence from the selected corpus, 50% it’s just a random sentence labeled isNext and NotNext respectively

Beneficial for QA and NLI

- why did they pretrain here?
- maybe the initial parameters were important to the task of question answering and natural language inference

Pretraining they used the 800M word BooksCorpus and English Wikipedia (ignoring lists tables and header)

- Why? Contingious - because they want to extract long contigious sentences, likely to make the model perform better in language understanding and how each word flows to the next to be included in it’s training data.

## Fine Tuning

- swapping inputs and outputs for fine tuning
  - `previously, sentence were independently encoded, then applied bidirectional cross attention, BERT though, concatanates these, and then uses self attention, which effectively includes bideirectional cross attention between two sentences`
  -

# Experiments

## GLUE (general language understanding eval benchm)

collection of diverse nat lang understanding task

descriptions of glue datasets are included in appendix b1

new params added, are classification layer werights - where K is num of labels

$$
W \in \R^{K \times H}
$$

Loss function is

$$
\log(\text{softmax}(CW^T))
$$

For each glue task they use 32 batch size and fine tune for 3 epochs over data for all glue tasks

and chose the best learning rate either `5e-5, 4e-5, 3e-5, and 2e-5`

## SQUAD - standford ques ans dataset

Question and Passage are using the sentence A & B embedding.

They introduce a start and end vector during fine tuning

Probability of word i being the start of answer span is computed as dot product betwee nTi and S followed by a softmax over all the words in the apragraph

$$
P_i = \frac{e^{S*T_i}}{\sum_je^{S*T_j}}
$$

- The question and passage are combined into one sequence.
- The model learns special vectors to mark the start and end of the answer.
- It calculates the probability for each word to be the start or end of the answer.
- It scores possible answer spans and picks the one with the highest score.
- The model is fine-tuned to improve its predictions over 3 epochs. with a lr 5e-5

## SQUAD 2.0

Basically SQUAD 1.1 except no short answer exists in the paragraph.

The score for no-answer (snull) is calculated using the start and end vectors (S and E) with the [CLS] token's hidden state (C).

$$
snull=S⋅C+E⋅C
$$

For spans that potentially contain an answer, the score (ŝi,j) is calculated as before:

$$
s^i,j=maxj≥i(S⋅Ti+E⋅Tj)
$$

- Some questions might not have an answer in the paragraph.
- BERT treats unanswered questions as having the answer at the [CLS] token.
- It calculates scores for both possible answers and no-answer.
- If the best possible answer score is higher than the no-answer score by a certain threshold, it predicts that answer.
- The model is fine-tuned and shows improved performance over previous system

**Fine-tuning and Results**:

- The model is fine-tuned for 2 epochs with a learning rate of 5e-5 and a batch size of 48.

## SWAG

Sentence pare completion examples that eval grounded sence inference

or the SWAG dataset:

1. The task is to pick the most logical continuation of a given sentence from four options.
2. BERT creates four sequences, each combining the given sentence with one of the possible continuations.
3. A special vector scores each sequence by interacting with the [CLS] token.
4. These scores are turned into probabilities using a softmax function.
5. The model is fine-tuned with specific settings (3 epochs, learning rate of 2e-5, batch size of 16).
6. BERTLARGE shows significantly better performance compared to previous models, indicating its strong capability in understanding and predicting sentence continuations.

## Ablation Studies

**No NSP**

Biderectional model trained using mask lm

without using next sentence prediction task

**LTR & NO NSP**

Left context only

- **NSP Task Importance**: Including NSP alongside MLM significantly improves model performance on various tasks by helping the model understand relationships between sentences.
- **Bidirectional vs. LTR**: Bidirectional MLM models outperform LTR models because they can utilize both left and right context during training, making them more effective for tasks requiring comprehensive understanding of context.

### Effect of Model Size on Fine-Tuning

1. **Experimental Setup**:
   - BERT models of varying sizes were trained with different configurations of layers, hidden units, and attention heads.
   - All models used the same hyperparameters and training procedure as described previously.
2. **Results on GLUE Tasks**:
   - Results are reported on selected GLUE tasks, showing the average Dev Set accuracy across 5 random restarts of fine-tuning.
   - Larger BERT models consistently show higher accuracy across all datasets in GLUE tasks.
   - Even on datasets with small training sets, like MRPC with only 3,600 labeled examples, larger models demonstrate significant accuracy improvements.
3. **Comparison with Previous Work**:
   - Previous literature typically used Transformer models of smaller sizes (e.g., (L=6, H=1024, A=16) with 100M parameters).
   - BERTBASE has 110M parameters, and BERTLARGE has 340M parameters, surpassing previous benchmarks in model size.
4. **Impact of Model Size**:

   - Increasing model size has historically improved performance on large-scale tasks such as machine translation and language modeling.
   - This study demonstrates that scaling up to extreme model sizes also leads to substantial improvements on smaller-scale tasks when models are well-pre-trained.
   - Unlike feature-based approaches in previous studies, BERT shows that directly fine-tuning on downstream tasks with larger, pre-trained representations benefits task-specific models significantly.

   ## Finetune vs Features

   - **Fine-tuning Approach**:
     - **Method**: A classification layer is added on top of BERT, and all model parameters are fine-tuned on the NER task.
     - **Advantages**: Allows BERT to adapt its representations specifically for the NER task, potentially improving performance.
   - **Feature-based Approach**:
     - **Method**: Contextual embeddings are extracted from one or more layers of BERT without fine-tuning any parameters. These embeddings are then fed into a two-layer BiLSTM and followed by a classification layer.
     - **Advantages**: Useful when tasks require specific architectural modifications that are not easily represented by the BERT architecture alone. Also offers computational benefits by pre-computing representations once and running cheaper models on to
