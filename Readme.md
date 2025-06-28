# Hybrid CNN Architecture

A custom deep learning architecture that combines the best features from multiple proven CNN architectures including ResNet, Inception, DenseNet, and Self-Attention mechanisms.

## Architecture Overview

The Hybrid CNN combines:
- **ResNet Bottleneck Blocks**: For deep feature learning with residual connections
- **Inception Blocks**: For multi-scale feature extraction using different kernel sizes
- **DenseNet Blocks**: For feature reuse and gradient flow
- **Self-Attention Mechanisms**: For spatial attention and long-range dependencies

## Features

- 🔧 **Modular Design**: Easy to customize and extend
- 🎯 **Multi-Scale Feature Extraction**: Captures features at different scales
- 🔄 **Attention Mechanisms**: Self-attention for better spatial understanding
- 📊 **Feature Reuse**: DenseNet-style connections for efficient learning
- ⚡ **Optimized Performance**: Balanced architecture for accuracy and efficiency

## Installation

```bash
pip install tensorflow>=2.8.0
```

## Quick Start

```python
from hybrid_cnn import create_hybrid_cnn, get_recommended_callbacks

# Create model for your dataset
num_classes = 10  # Replace with your number of classes
model = create_hybrid_cnn(
    num_classes=num_classes,
    input_shape=(224, 224, 3)
)

# Print model summary
model.summary()

# Get recommended training callbacks
callbacks = get_recommended_callbacks('best_model.keras')

# Train your model (assuming you have train_dataset and val_dataset)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=callbacks
)
```

## Advanced Usage

### Custom Model Configuration

```python
from hybrid_cnn import HybridCNN

# Create custom model instance
hybrid_cnn = HybridCNN(num_classes=100, input_shape=(256, 256, 3))

# Build the model
model = hybrid_cnn.build_model()

# Custom compilation
hybrid_cnn.compile_model(
    optimizer='sgd',
    learning_rate=0.01,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_5_accuracy']
)
```

### Using Individual Components

```python
from hybrid_cnn import HybridCNN
from tensorflow.keras import layers, models

# Use individual blocks in your custom architecture
inputs = layers.Input(shape=(224, 224, 3))
x = HybridCNN.inception_block(inputs, filters=64, block_name='custom_inception')
x = HybridCNN.self_attention_block(x)
x = HybridCNN.bottleneck_block(x, filters=128, block_name='custom_bottleneck')
# ... continue building your custom model
```

## Architecture Details

### Model Structure

1. **Initial Convolution**: 7x7 conv, stride 2, followed by BatchNorm and MaxPool
2. **Stage 1**: 2 Bottleneck blocks (64 filters) + Self-attention
3. **Stage 2**: Inception block + Bottleneck block (128 filters) + Self-attention
4. **Stage 3**: Dense block (4 layers, growth rate 32) + Transition
5. **Stage 4**: Bottleneck + Inception (256 filters) + Self-attention
6. **Stage 5**: Dense block (4 layers, growth rate 32) + Transition
7. **Final Stage**: 2 Bottleneck blocks (512 filters) + Self-attention
8. **Classification**: Global Average Pooling + Dropout + Dense

### Key Components

#### Bottleneck Block
```
Input → 1x1 Conv → 3x3 Conv → 1x1 Conv → Add with shortcut → Output
```

#### Inception Block
```
Input → [1x1, 3x3, 5x5, MaxPool] → Concatenate → Output
```

#### Dense Block
```
Input → [BN→ReLU→Conv] × num_layers with concatenation → Output
```

#### Self-Attention Block
```
Input → [Query, Key, Value] → Attention weights → Scaled output + residual → Output
```

## Performance

The model has been tested on various image classification tasks and shows competitive performance:

- **Parameters**: ~25M parameters (depending on num_classes)
- **Memory**: Efficient memory usage due to attention mechanisms
- **Speed**: Balanced inference speed with high accuracy

## Training Tips

1. **Learning Rate**: Start with 0.001, use ReduceLROnPlateau callback
2. **Batch Size**: 32 works well for most cases
3. **Data Augmentation**: Recommended for better generalization
4. **Input Size**: 224x224 is optimal, but can be adjusted
5. **Callbacks**: Use the provided `get_recommended_callbacks()` function

## Example

### Image Classification

```python
import tensorflow as tf
from hybrid_cnn import create_hybrid_cnn, get_recommended_callbacks

# Prepare your data
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/train',
    image_size=(224, 224),
    batch_size=32
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/validation',
    image_size=(224, 224),
    batch_size=32
)

# Create model
num_classes = len(train_dataset.class_names)
model = create_hybrid_cnn(num_classes=num_classes)

# Train
callbacks = get_recommended_callbacks('my_model.keras')
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks
)
```
