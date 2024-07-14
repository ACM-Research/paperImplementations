# Advertisement Proj

**How effective are transformer-based models in generating visual advertisements from textual descriptions, and how does fine-tuning with manually collected copywriting enhance ad creativity and effectiveness?**

Basically a text to image transformer model built from scratch (first pretrained on general images) then with a fine tuning on advertisements and and copywrite. It will be integrated with a text interpretation model (think BERT-like) and a cross modal transformer to fuse the embeddings the best and so the self attention operation performs better. The idea is have an easier way to create image advertisements by taking a few pictures of a product and having the model spit out the product in custom scenarios. Why? I think it’s a great way to learn the latest in transformer text-image text (likely implementing something similar to stable diffusion), and I think it’s something that can actually provide value in the real world.

- Methodology
  - Pytorch for making the model
  - Dataset: First broad corpus’ text and images, then fine tuned on advertisement datasets (either manually scraped [i’ve seen some bots that might’ve done this] or predefined datasets] )
    - If the data is too hard to find, (which is looking like it is) might switch to a different project
    - (https://www.facebook.com/ads/library/api/?source=nav-header Meta API for ads)
      - All EU Ads ever served this past year apparently can be accessed through this (thank u eu lawmakers | also wondering if translation necessary since if ads are not in english)
      - Maybe webscraping possible through their own ads library tool?
  - Stable Diffusion paper (if anything later than it cameout implement that)
  - BERT or a better MLM model to interpret the text
- Resources
  - Since training a text to image model (likely gpu credits) or maybe if utd has gpus probably.

## PAPERS

- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
  - **Source**: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. _arXiv preprint arXiv:1810.04805_.
  - **Description**: The BERT paper introduces a novel transformer-based model for natural language understanding, which pretrains on large text corpora using a masked language model (MLM) and next sentence prediction (NSP). BERT significantly improves performance on various NLP tasks through its bidirectional training approach, capturing context from both directions.
  - **Evaluation**: BERT's success in understanding and generating text makes it a strong candidate for interpreting textual descriptions in this project. Its ability to capture nuanced meaning from context will enhance the quality of text-to-image transformations, making it a foundational component of the proposed advertisement generation model.
- **Stable Diffusion**
  - **Source**: Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., & Sutskever, I. (2022). Stable Diffusion: High-Resolution Image Synthesis with Latent Diffusion Models. _arXiv preprint arXiv:2107.09679_.
  - **Description**: This paper presents Stable Diffusion, a model for high-resolution image synthesis that leverages latent diffusion models to generate images from text descriptions. It demonstrates state-of-the-art performance in generating coherent and detailed images based on textual input, significantly advancing text-to-image generation techniques.
  - **Evaluation**: The principles and techniques from Stable Diffusion will be instrumental in guiding the development of the text-to-image transformer model for this project. Its approach to generating high-quality images from text will serve as a blueprint for achieving similar results in the context of advertisement generation.
- **ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks**
  - **Source**: Lu, J., Batra, D., Parikh, D., & Lee, S. (2019). ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks. _arXiv preprint arXiv:1908.02265_.
  - **Description**: ViLBERT extends the BERT architecture to handle vision-and-language tasks by processing visual and textual inputs through separate streams and integrating them using co-attention layers. This model shows improved performance on various tasks requiring joint understanding of images and text.
  - **Evaluation**: ViLBERT's approach to integrating visual and textual representations will inform the design of the cross-modal transformer in this project. Its ability to handle diverse vision-and-language tasks makes it a relevant reference for developing robust visiolinguistic embeddings for advertisement generation.
- **VIOLET: Visual and Textual Learning with Lots of Embeddings Together**
  - **Source**: Fu, T., Liao, W., Zhang, X., & Cheng, H. (2021). VIOLET: Visual and Textual Learning with Lots of Embeddings Together. _arXiv preprint arXiv:2102.03334_.
  - **Description**: VIOLET is a model designed to integrate visual and textual data by combining multiple embeddings in a unified framework. It employs a cross-modal transformer architecture to effectively capture the interactions between visual and textual elements.
  - **Evaluation**: The techniques and insights from VIOLET will be valuable for developing the cross-modal transformer in this project. By leveraging its approach to embedding and integrating visual and textual data, the project can achieve more accurate and contextually rich text-to-image transformations for advertisement generation. VIOLET's framework will guide the model's design to ensure effective fusion of text and image embeddings, enhancing the creativity and effectiveness of the generated ads.

## Some Other Ideas That Haven’t Really Came About Yet

These are some other ideas I have but haven’t put that much time (it’s basically some brainstorm stuff). But could be cool to work on

# Graph Database Optimization

Summary: Basically using a transformer model to better predict when to move data from a database into an in-memory cache (so you can have better read speeds). I feel like transformers could work because the self attention in a highly interlinked databases (graph databases) is in my head similar to how language also interlinks. I was thinking to have a model predict when to move data into an in memory cache in a way that’s (hopefully) better than some of the current algorithms (most frequently used gets moved, some other DL algos) and wonder if the better context understanding of transformers can better predict what data is going to be used.

Research Question:

**Can transformer models improve the prediction of data movement from a graph database to an in-memory cache compared to traditional algorithms, thereby enhancing read speeds?**

### Methodology

1. **Model Development**:
   - **Integration of Graphformer and Longformer**: Develop a model that combines the capabilities of Graphformer (designed for graph data) and Longformer (utilizing a sliding window mechanism to handle long sequences).
   - **Data Processing**: Preprocess graph database data to convert it into a format suitable for transformer input, ensuring the model can effectively capture the relational context.
2. **Dataset and Training**:
   - **Dataset**: Use a combination of synthetic graph database data and real-world datasets that exhibit complex interlinked structures.
   - **Training**: Train the model to predict which data nodes should be moved to an in-memory cache based on past access patterns. Use supervised learning with labeled data indicating optimal caching decisions.
3. **Evaluation**:
   - **Baseline Comparison**: Compare the transformer model's performance against traditional caching algorithms (e.g., Least Recently Used, Most Frequently Used) and other deep learning algorithms.
   - **Metrics**: Evaluate the model based on cache hit rate, read speeds, and overall system performance. Conduct experiments to measure improvements in these metrics.
4. **Optimization and Fine-Tuning**:
   - Fine-tune the model to balance the trade-off between cache size and read speed improvements.
   - Experiment with different configurations of the Graphformer and Longformer integration to identify the most effective architecture.

### Resources Needed

- **GPU Credits**: Required for training the transformer models, especially given the computational complexity of handling large graph datasets.
- **Datasets**: Access to both synthetic and real-world graph database datasets for training and evaluation purposes.
- **Software**: PyTorch for model development, along with libraries for data processing, graph manipulation, and model evaluation.

### Annotated Bibliography

1. **GraphFormers: Transformers for Graph Representation Learning**
   - **Source**: Wu, Z., Zhang, S., Li, J., & Yu, P. (2021). GraphFormers: Transformers for Graph Representation Learning. _arXiv preprint arXiv:2106.03234_.
   - **Description**: The paper introduces GraphFormers, a transformer-based model designed for graph data. It leverages the self-attention mechanism to capture the complex relationships within graphs, demonstrating superior performance in graph representation tasks compared to traditional graph neural networks.
   - **Evaluation**: The principles and techniques from GraphFormers will guide the development of the transformer model for this project. Its ability to effectively capture and represent graph data is critical for predicting optimal caching decisions in graph databases.
2. **Longformer: The Long-Document Transformer**
   - **Source**: Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. _arXiv preprint arXiv:2004.05150_.
   - **Description**: Longformer introduces a transformer model that can handle long sequences using a sliding window attention mechanism. This approach reduces computational complexity while maintaining the ability to capture long-range dependencies.
   - **Evaluation**: Integrating Longformer's sliding window mechanism with GraphFormers will allow the model to handle large and complex graph datasets more efficiently, making it suitable for real-time caching decisions.
3. **Transformers for Graphs: An Overview**
   - **Source**: Dwivedi, V. P., Joshi, C. K., Laurent, T., Bengio, Y., & Bresson, X. (2020). Transformers for Graphs: An Overview. _arXiv preprint arXiv:2012.09699_.
   - **Description**: This paper provides an overview of transformer models applied to graph data, discussing various architectures and their performance on graph-related tasks. It highlights the potential of transformers to revolutionize graph representation learning.
   - **Evaluation**: The insights from this overview will help in understanding the strengths and limitations of different transformer architectures for graph data, informing the design and optimization of the model for this project.

# Text-to-Video Generation Project Proposal

**Research Question**:
Can transformer models, particularly those integrating visual and textual embeddings, generate high-quality videos from textual descriptions?

Summary: This would just be me trying to implement a text to video model, because it obviously is one of the most cutting edge parts of transformers out there.

- (I don’t know how to exactly frame this part)
- But a use case could be for the advertising thing
- Or something like stock video generation?
- Deepfakes??(i don’t think this one would even be ethical)

What would make this one particularly difficult

- Compute - yeah, this would take a lot of computing power for smoething
- Recency - not many papers on this, and lack of resources could be annoying
- A Labeled dataset (easy or hard find) this like a pretty easy find.

Papers:
VIOLET

Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis
