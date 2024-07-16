### Research Question

Can synthetic, ML-generated text-audio pairs be used to train a SOTA CLAP model for music?

### Summary

My project aims to efficiently train a SOTA model that encodes music and text descriptions into the same latent space using synthetic data from generative models trained with orders of magnitude more resources--essentially a form of knowledge distillation. Just like CLIP with images, this kind of model could be applied to a myriad of downstream tasks, from simple genre classification to semantic music search and content recommendation. 

The core of the project first requires a procedure for generating a synthetic dataset that is diverse and representative enough of the musical space to yield useful results, and of course training the CLAP model itself. If this yields promising results and time permits, we will then move on to experiment with other architectures (e.g., experimenting with different encoders), integrating the model with an LLM which you could talk to about music, and more comprehensive evaluation on downstream tasks.

Dr. Yapeng Tian has agreed to be our faculty advisor for this team.

### Methodology

Firstly, the synthetic dataset will need to be prepared. I plan to do this by aggregating a list of many different qualities a song could have, e.g., the X most common instruments, Y most common moods, Z most common genres, etc. (I do not have an exact list of these different attributes and how many of each to select. I planned to have this be one of the first details my team and I work out together, but can also do this ahead of time if preferred.) This list of keywords would then be randomly combined to yield N captions, which would then be fed into the generative model (MusicGen) to get a song. I would also likely augment this data with some high quality public domain data from, e.g., wikimedia commons.

Secondly, a CLAP model would be trained on this synthetic dataset. Models of the same size will both be finetuned from base CLAP models and trained from scratch. In addition to standard contrastive loss, we will use genre classification and semantic retrieval as proxy metrics for model performance, as the fuzzy nature of such a semantic problem makes direct and quantitative evaluation exceedingly difficult. Models will also be trained with differing learning rates on their pretrained text and audio encoders to see which configuration yields the best results.

As mentioned above, alternative architectures will be implemented if the above is completed with satisfactory results and time to spare. This will involve experimenting with alternative encoders, like encodec, in addition to any other suitable ideas proposed by the team. The same training data, model size, etc. will be used as in the previous models trained for fair comparison.


### Resources Needed
The only resource needed here will be compute, likely in the form of GPU credits. I have a homelab rig with a 4090, beefy processor, and lots of RAM that I am more than willing to let my team use, but additional resources may be needed for the bigger, final, models and perhaps for the initial large synthetic data generation.

### Annotated Bibliography

- [Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687)
    - This paper introduces a CLIP-like dual-encoder model for audio an text, trained on text descriptions and ~4200 hours of audio from publicly available, free to use online sources. It uses two pre-trained encoders on each end, and two MLP layers to align their embeddings to the same space and is trained contrastively with the same loss function as CLIP. The paper achieves SOTA results in audio classification and retrieval. Since their dataset was limited due to it only comprising free-to-use data, and since many of those captions were not maximally rich (many were simply the clips' filenames), I speculate that significant improvements can be made with this same architecture and better data.
- [CLAP: Learning Audio Concepts From Natural Language Supervision](https://arxiv.org/abs/2206.04769)
    - This paper is by Microsoft and appears to have been released independently and very shortly after the LAION CLAP paper. Its architecture is practically identical save for it being scaled up and using a purely convolutional audio encoder as opposed to LAION's transformer-based HTSTAT. The authors report that their fine-tuned model did achieve SOTA results in genre classification by a large margin, but only the weights for these fine-tuned models were not released and thus this performance cannot be reproduced nor verified.
- [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)
    - This paper by Meta introduces MusicGen, a language model which generates music clips from textual descriptions. It uses encodec (a hi-fi neural audio encoder which uses residual vector quantization and interleaved token streams) for tokenization, and a T5 encoder for text embedding. The authors train an autoregressive transformer to generate encodec audio tokens conditioned on the text embedding provided and an optional melody. Autoregressive loss is calculated with respect to the tokens output by encodec when fed the target audio.