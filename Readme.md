# Image Captioning with Encoder-Attention-Decoder Architecture

explaining the implementation of an image captioning system using an encoder-attention-decoder architecture on the Flickr8k dataset. The notebook walks through the entire process from data preparation to model evaluation and comparison with a pre-trained model.

## Project Overview

Image captioning combines computer vision and natural language processing to generate textual descriptions of images. The implemented architecture consists of:

1. **Encoder**: A CNN (ResNet-50) that extracts image features
2. **Attention**: A mechanism that helps the model focus on relevant parts of the image
3. **Decoder**: An LSTM that generates captions word by word

## Dataset

The project uses the Flickr8k dataset, which contains:
- 8,091 images, each with 5 different captions
- Split into train (70%), validation (15%), and test (15%) sets
- Approximately 40,455 image-caption pairs in total

## Data Preprocessing

### Image Transformations
- **Training**: Applies random horizontal flips, rotations, color jittering, resizing to 224×224, and normalization
- **Testing/Validation**: Only applies resizing and normalization for consistent evaluation

### Text Processing
- Custom tokenizer to break captions into words
- Vocabulary building with frequency threshold (words appearing less than 10 times are replaced with `<unk>`)
- Special tokens: `<pad>`, `<sos>` (start of sequence), `<eos>` (end of sequence), `<unk>` (unknown)
- Option to use GloVe pre-trained embeddings

### Dataset and DataLoader
- Custom `FlickrDataset` class for handling image-caption pairs
- Custom `collate_fn` to handle variable-length captions through padding
- Dataloaders for training, validation, and testing with batch size of 16

## Model Architecture

### Encoder
- Based on ResNet-50 pre-trained on ImageNet
- Removes classification layer and uses the convolutional features
- Adaptive pooling to ensure consistent output size (7×7×2048)
- Fine-tunes only the last two blocks of ResNet
- Output: Features of shape (batch_size, 49, 2048)

### Attention Mechanism
- Soft attention that computes weights for each of the 49 image regions
- Uses three linear layers to calculate relevance scores between text and image
- Creates a context vector as a weighted sum of visual features

### Decoder
- LSTM-based decoder with attention
- Embedding layer to convert word indices to vectors
- At each time step:
  - Takes the previous word and context vector as input
  - Uses attention to focus on relevant image regions
  - Outputs probability distribution over vocabulary

### EncoderDecoder
- Combines the encoder, attention, and decoder components
- Forward pass: processes images and generates captions
- Inference method: generates captions for new images

## Model Training

### Training Configuration
- Embedding dimension: 128
- Attention dimension: 128
- Decoder dimension: 256
- Loss function: Cross-entropy with padding ignored
- Optimizer: Adam with learning rate 3e-4
- Batch size: 16
- Early stopping based on validation loss

### Training Process
- Training loop for a specified number of epochs (2-3 in this implementation)
- Mixed-precision training when available for speed
- Training and validation losses tracked for each epoch
- BLEU scores calculated on validation set
- Best model saved based on validation loss

## Model Evaluation

### Metrics
- BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores
- Visualization of attention weights
- Comparison with reference captions

### Pretrained Model Comparison
- Compares custom model with a pre-trained ViT-GPT2 image captioning model
- Side-by-side visualization of generated captions
- Comparison of BLEU scores between models

## Interactive UI for Testing

An interactive UI allows uploading new images and generating captions using both models:
- Upload an image
- Generate captions from both custom and pre-trained models
- Visualize attention weights from the custom model
- Display side-by-side results for comparison

## Results and Visualization

- Training and validation loss curves
- BLEU score progression during training
- Attention maps showing which parts of the image the model focuses on for each word
- Comparative bar charts of BLEU scores between custom and pre-trained models

## Insights from the Implementation

1. The custom model, despite being trained on limited resources, learns to focus on relevant image regions.
2. The attention mechanism provides interpretability by showing which parts of the image influence each generated word.
3. The pre-trained model generally achieves higher BLEU scores due to its larger training dataset and more complex architecture.

## How to Use the Notebook

1. Run the data loading and preprocessing cells to set up the dataset
2. Execute the model definition cells to create the encoder-attention-decoder architecture
3. Run the training cells to train the model (adjust epochs as needed)
4. Evaluate the model and compare with the pre-trained model
5. Use the interactive UI to test with new images

This implementation demonstrates the fundamental concepts of image captioning while providing visualization tools to understand the model's behavior.