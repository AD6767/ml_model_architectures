# CLIP / VLM Basics

## Architecture
CLIP aligns images and text into a shared embedding space. 
`image -> image encoder -> image projection -> image embedding`, `text  -> text encoder  -> text projection  -> text embedding`.
Then image-text similarity is computed with a dot product: `similarity = image_embedding @ text_embedding.T`
If embeddings are L2-normalized, dot product becomes cosine similarity.

## Problem
Traditional classifiers predict from a fixed label set. CLIP learns from paired image-text data, allowing the model to connect visual concepts with natural language.

## Input / output
Image encoder output: image_features: `[B, image_feature_dim]`
Text encoder output: text_features: `[B, text_feature_dim]`
Projection heads map both into the same shared dimension: image_embedding: `[B, embed_dim]`, text_embedding: `[B, embed_dim]`.
Similarity matrix: logits_per_image: `[B, B]`, logits_per_text: `[B, B]`

## Key blocks
### Image Encoder
Extracts visual features from an image. Common choices: ResNet, ViT, Swin Transformer

### Text Encoder
Extracts language features from text. Common choices: bidirectional (encoder-style) Transformer instead of a causal decoder

### Projection Heads
Projection heads map image and text features into the same embedding space.
`image_features -> Linear -> image_embedding`, `text_features  -> Linear -> text_embedding`. 
This is needed because image and text encoders may output different feature dimensions.

### L2 Norm
Before similarity, embeddings are usually normalized: `embedding = embedding / ||embedding||`
This makes dot product behave like cosine similarity.

### Contrastive Loss
For a batch of paired image-text examples: `image_0 <-> text_0`, `image_1 <-> text_1`, `image_2 <-> text_2`. The diagonal pairs are positives. Off-diagonal pairs are negatives. The target labels come from batch order: labels = `[0, 1, 2, ..., B-1]`

## Loss
CLIP uses symmetric contrastive loss.
Image-to-text loss: Each image should classify its matching text among all texts in the batch.
Text-to-image loss: Each text should classify its matching image among all images in the batch.
Final loss: loss = (image_to_text_loss + text_to_image_loss) / 2

## Why it was introduced
CLIP was introduced to learn visual concepts from natural language supervision instead of fixed manually labeled classes. This made models more flexible and enabled zero-shot transfer to new categories.

## Strengths
* Learns open-vocabulary visual representations.
* Enables zero-shot classification.
* Supports image-text retrieval.
* Useful foundation for VLMs.
* Scales well with large image-text datasets.

## Weaknesses
* Depends heavily on data quality and caption quality.
* Can learn dataset biases.
* Contrastive training needs large and diverse batches.

## Production concerns
* Normalize embeddings before retrieval.
* Choose embedding dimension based on latency, memory, and retrieval quality.
* For retrieval, cache embeddings instead of recomputing them.

## Important questions
1. What is the main idea of CLIP?
CLIP learns a shared embedding space where matching images and texts are close together, and non-matching pairs are far apart.
2. What are positives and negatives in CLIP?
For a batch of paired image-text examples, the matching image-text pairs on the diagonal are positives. All off-diagonal pairs are treated as negatives.
3. Where do the labels come from?
Labels come from batch ordering. If `image_i` matches `text_i`, then the correct label for row `i` is `i`. `labels = torch.arange(batch_size)`
4. Why is the similarity matrix [B, B]?
Each image is compared against every text in the batch. For B images and B texts: `image_embeddings @ text_embeddings.T -> [B, B]`
5. Why normalize embeddings?
Normalization makes dot product equivalent to cosine similarity and prevents vector magnitude from dominating similarity.
6. What does temperature do?
Temperature controls the sharpness of the similarity distribution. Lower temperature makes the model focus more strongly on the highest-similarity pairs.
7. How does CLIP do zero-shot classification?
Class names are converted into text prompts such as: "a photo of a cat", "a photo of a dog". The image embedding is compared with each text embedding. The closest text prompt becomes the predicted class.
8. Is CLIP generative?
No. CLIP is mainly an embedding/alignment model. It retrieves, ranks, or classifies by similarity. It does not generate text by itself.
