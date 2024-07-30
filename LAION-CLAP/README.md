# Synopses

## Paper 1 - LAION-CLAP

#### Intro
This paper introduces a model trained using contrastive learning to encode text and audio into the same latent space--a very similar approach to that which OpenAI took with their CLIP, whose success the authors acknowledge as inspiration and motivation for their work. The model is trained on over 4200 hours (~630k pairs) of publicly available, free to use online audio (to respect rightsholders) sources with various and often naive methods of assigning captions. The authors demonstrate that the model achieves SOTA results in many audio classification tasks.

#### Architecture
The CLAP model, like CLIP, is a dual encoder model. Multiple audio and text encoders were tested. They empirically established that the best performing pair was RoBERTa and HTSAT (a transformer-based model that uses SWIN transformers over the 2D spectrogram of the audio). This configuration was used for all evaluations. The model also included a feature-fusion mechanism (based on a 2D convolutional layer and an attention-based weighted sum) which allowed the model to take into account both fine-grained local features, as well as more coarse global ones. This allows the model to work with longer audio clips that don't fit in the smaller local context window.

#### Evaluation
As discussed before, the results for this model were very impressive. However, this paper does present a few limitations and raise a few questions. 

Firstly, there is the quality and quantity of the data used. Richly descriptive caption-audio pairs are already hard enough to find on the internet, or elsewhere (people don't just go into YouTube or soundcloud comments and wax poetic about the sonic qualities of a sound or song). Since the authors were limited to just publicly available sounds, this made constructing a data set even harder--as detailed in the paper, many of the captions for the sounds were even just the clips' filenames. This being the case, it would not be farfetched at all to think that significant performance increase is attainable with solely a richer dataset and no architecture modifications. This is especially true for more specific applications like music. 

Secondly, both audio encoders tested in the paper operated on mel-spectrograms, which are 2D representations of audio showing frequency and time. It may be worth experimenting with an encoder that uses a more linear tokenization as is commonly used for language (maybe something built upon Meta's encodec), though I have no predictions for which would perform better.