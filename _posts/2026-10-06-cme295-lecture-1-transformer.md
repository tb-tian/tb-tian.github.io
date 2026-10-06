---
layout: post
title: "My notes on CME295 Lecture 1: The Transformer"
---

This is my notes on [CME295: Transformers & Large Language Models](https://www.youtube.com/watch?v=Ub3GoFaUcds&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy)

**Outline**

1. [NLP overview](#nlp-overview)
2. [Tokenization](#tokenization)
3. [Word representation](#word-representation)
4. [RNNs and LSTMs](#rnns-and-lstms)
5. [Self-attention](#self-attention)
6. [Transformer architecture](#transformer-architecture)
7. [Key takeaways](#key-takeaways)

## NLP overview

### Tasks

| Type | What it predicts | Examples |
|---|---|---|
| Classification | One label for the whole text | Sentiment analysis, intent detection, language detection |
| Token classification (sequence labeling) | One label per token | Named entity recognition (NER), part-of-speech (POS) tagging |
| Generation | A sequence of text | Machine translation, question answering, summarization, text generation |

### Metrics

- **Classification / NER:** accuracy, precision, recall, F1.
- **BLEU:** n-gram **precision** of the generated text against reference text. Mostly used for machine translation.
- **ROUGE:** n-gram **recall** against reference text. Mostly used for summarization.
- **Perplexity:** how surprised a language model is by held-out text. Lower is better.

## Tokenization

Before a model sees text, the text is split into **tokens**.

| Type | Pros | Cons | Example: "teddy bear" |
|---|---|---|---|
| Word | Easy to interpret; short sequences | Large vocabulary; word variations (e.g. *run*, *runs*, *running*) not handled | `teddy` `bear` |
| Subword (BPE, WordPiece) | Reuses word roots; embeddings are intuitive | Longer sequences; tokenization is more complex | `ted` `##dy` `bear` |
| Character / Byte | No out-of-vocabulary words; small vocabulary | Much longer sequences; patterns are too low-level to interpret | `t` `e` `d` `d` `y` … |

Most modern LLMs use **subword** tokenization.

Special tokens:

- `[UNK]`: a token that is not in the vocabulary.
- `[PAD]`: fills shorter sequences in a batch so they all have the same length.
- `[BOS]` / `[EOS]`: the beginning and end of a sequence.

## Word representation

### Why not one-hot encoding?

With one-hot vectors, every pair of distinct words is **orthogonal**, so their cosine similarity is 0:

$$\cos(u, v) = \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert}$$

This means *cat* is as far from *dog* as it is from *carburetor*. We want **learned embeddings**, where similar words end up close together.

### Word2Vec

Goal: learn word vectors that capture meaning by training on a simple prediction task.

- **Continuous bag of words (CBOW):** use the surrounding context words to predict the target word.
- **Skip-gram:** use the target word to predict the surrounding context words.

The network is shallow: a one-hot input of size $V$, a hidden layer of size $d$ (the embedding), and an output of size $V$ that is a probability distribution over the vocabulary.

![Word2Vec network: an input layer of size V, a hidden layer of size d, and an output layer of size V](/assets/images/cme295-lecture-1/word2vec-architecture.png)

**Example (next-word prediction).** The input is the one-hot vector for "A", and the model should predict "cute". The hidden activations $[0.2, 0.9]$ are the learned embedding of "A". After training, the hidden-layer weights are the word embeddings.

![The one-hot vector for "A" passes through a 2-dimensional hidden layer and gives an output distribution whose highest probability is "cute"](/assets/images/cme295-lecture-1/word2vec-next-word-example.png)

## RNNs and LSTMs

### Recurrent neural networks

RNNs were introduced in the 1980s. They are networks whose connections form a **temporal sequence**: at each step $t$, the hidden state $a^{\langle t \rangle}$ combines the current input $x^{\langle t \rangle}$ with the previous state $a^{\langle t-1 \rangle}$.

![An unrolled RNN: each cell takes the input x⟨t⟩ and the previous hidden state a⟨t−1⟩, and outputs y⟨t⟩](/assets/images/cme295-lecture-1/rnn-general-form.png)

- **Pros:** handles sequences of any length; the same weights are shared across all time steps.
- **Cons:**
  - The whole past is squeezed into one hidden state, so long-range information is lost.
  - **Vanishing gradients** make long dependencies hard to learn.
  - Computation is **sequential**: step $t$ cannot start until step $t-1$ finishes, so training doesn't parallelize.

### Long short-term memory (LSTM)

LSTMs were introduced in *Long Short-Term Memory* (1997). They add a structured **cell state** $c^{\langle t \rangle}$ controlled by gates: forget $\Gamma_f$, update $\Gamma_u$, relevance $\Gamma_r$ and output $\Gamma_o$.

![An LSTM cell: the cell state c flows across the top and is changed by the forget, update, relevance and output gates](/assets/images/cme295-lecture-1/lstm-cell.png)

The gates **reduce** the vanishing-gradient problem, but the LSTM is **still sequential** and still slow on long sequences. That limitation is what motivates attention.

## Self-attention

Introduced in *Attention Is All You Need* (Vaswani et al., 2017).

**Idea:** each token builds a **query** $q$ and compares it to the **keys** $k$ of every token in the sequence. The more similar a key is, the more weight its **value** $v$ gets in the output.

![The query for "teddy bear" is compared with the key of every token in "a cute teddy bear is reading."](/assets/images/cme295-lecture-1/attention-query-keys.png)

With all tokens stacked into matrices $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$ and $V \in \mathbb{R}^{n \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- Dividing by $\sqrt{d_k}$ keeps the dot products from growing with the dimension, so the softmax doesn't saturate.
- **Why it beats RNNs:**
  1. All tokens are processed **in parallel** with matrix multiplications.
  2. Any two tokens are connected **directly**, not through a long chain of hidden states.
- **Multi-head attention:** $h$ heads run in parallel, each with its own learned projections of $Q$, $K$ and $V$. Each head can learn a different kind of relationship between tokens, such as syntax or coreference. The outputs are concatenated and projected back to $d_{\text{model}}$.

## Transformer architecture

For more interactive explainer: 
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [The Transformer Model (MachineLearningMastery)](https://machinelearningmastery.com/the-transformer-model/)

![The full Transformer: an encoder stack on the left and a decoder stack on the right, followed by a linear layer and a softmax](/assets/images/cme295-lecture-1/transformer-architecture.png)

### Hyperparameters

| Symbol | Meaning | Original paper (base) |
|---|---|---|
| $V$ | Vocabulary size | ~37k (shared BPE) |
| $d_{\text{model}}$ | Embedding / hidden dimension | 512 |
| $N$ | Number of stacked encoder (and decoder) layers | 6 |
| $h$ | Number of attention heads | 8 |
| $d_k = d_v$ | Per-head key/value dimension ($d_{\text{model}} / h$) | 64 |
| $d_{\text{ff}}$ | Inner dimension of the feed-forward network | 2048 |

### Input embedding and positional encoding

1. The input text is tokenized, and each token is mapped to a learned vector of size $d_{\text{model}}$ (a $V \times d_{\text{model}}$ lookup table).
2. Attention by itself has no notion of order, so a **positional encoding** is **added** to each embedding.
   - It can be learned, or fixed. The original paper uses fixed **sinusoids** of different frequencies.
   - Sinusoids encode absolute position, and their structure also makes **relative** offsets easy for the model to use.

### Encoder

![The encoder block highlighted: multi-head attention and a feed-forward network, each followed by Add & Norm, stacked N times](/assets/images/cme295-lecture-1/transformer-encoder.png)

The encoder rewrites each input token as a function of all the other input tokens. Each of the $N$ layers has:

1. **Multi-head self-attention** over the input sequence.
2. **Position-wise feed-forward network:** $d_{\text{model}} \to d_{\text{ff}} \to d_{\text{model}}$, applied to each token independently.
3. Around each sub-layer, a **residual connection + layer norm** ("Add & Norm"): $\text{LayerNorm}(x + \text{Sublayer}(x))$.

### Decoder input: outputs shifted right

- The target tokens are embedded with their own learned $V \times d_{\text{model}}$ table, plus positional encoding.
- The sequence starts with `[BOS]` and is **shifted right by one**, so the model predicts token $t$ from tokens $< t$.

### Decoder

![The decoder block highlighted: masked multi-head attention, cross-attention over the encoder output and a feed-forward network, each followed by Add & Norm](/assets/images/cme295-lecture-1/transformer-decoder.png)

Each of the $N$ decoder layers has three sub-layers, each wrapped in Add & Norm:

1. **Masked multi-head self-attention:** a causal mask stops each position from attending to *future* tokens, so the model can't see the answer during training.
2. **Encoder–decoder (cross) attention:** queries come from the decoder; keys and values come from the **encoder output**. This is where the decoder "reads" the source sentence.
3. **Position-wise feed-forward network.**

### Output

- A **linear layer** projects $d_{\text{model}} \to V$, giving one logit per vocabulary token.
- A **softmax** turns the logits into a probability distribution over the next token.
- At inference, decoding is autoregressive: choose a token, append it to the decoder input, and repeat until `[EOS]`.

## Key takeaways

- Subword tokenization balances vocabulary size against sequence length.
- Learned embeddings (e.g. Word2Vec) place similar words close together; one-hot vectors can't.
- RNNs and LSTMs process tokens one by one, which limits both long-range memory and parallelism.
- Self-attention connects every pair of tokens directly and runs as a few matrix multiplications.
- The Transformer has an encoder (self-attention + FFN) and a decoder (masked self-attention + cross-attention + FFN), with residual connections and layer norm throughout.
