"""
Hybrid CNN Architecture
A custom deep learning architecture combining ResNet, Inception, DenseNet, and Self-Attention mechanisms.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Optional


class HybridCNN:
    """
    A hybrid CNN architecture that combines multiple proven architectures:
    - ResNet bottleneck blocks for deep feature learning
    - Inception blocks for multi-scale feature extraction
    - DenseNet blocks for feature reuse
    - Self-attention mechanisms for spatial attention
    """
    
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """
        Initialize the Hybrid CNN model.
        
        Args:
            num_classes (int): Number of output classes
            input_shape (tuple): Input image shape (height, width, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
    
    @staticmethod
    def self_attention_block(x):
        """
        Self-attention block for spatial attention mechanism.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with self-attention applied
        """
        batch_size, height, width, channels = x.shape
        
        # Create query, key, and value projections
        f = layers.Conv2D(channels // 8, 1, name='attention_f')(x)
        g = layers.Conv2D(channels // 8, 1, name='attention_g')(x)
        h = layers.Conv2D(channels, 1, name='attention_h')(x)
        
        # Reshape for matrix multiplication
        f = layers.Reshape((height * width, channels // 8))(f)
        g = layers.Reshape((height * width, channels // 8))(g)
        h = layers.Reshape((height * width, channels))(h)
        
        # Transpose g for dot product
        g = layers.Permute((2, 1))(g)
        
        # Compute attention weights
        attention = layers.Dot(axes=(2, 1))([f, g])
        attention = layers.Activation('softmax')(attention)
        
        # Apply attention to values
        out = layers.Dot(axes=(2, 1))([attention, h])
        out = layers.Reshape((height, width, channels))(out)
        
        # Learnable scaling parameter
        gamma = tf.Variable(0., trainable=True, name='attention_gamma')
        out = layers.Lambda(lambda x: x * gamma)(out)
        
        # Residual connection
        return layers.Add()([x, out])
    
    @staticmethod
    def bottleneck_block(x, filters: int, stride: int = 1, block_name: str = ""):
        """
        ResNet-style bottleneck block.
        
        Args:
            x: Input tensor
            filters (int): Number of output filters
            stride (int): Stride for convolution
            block_name (str): Name prefix for layers
            
        Returns:
            Output tensor
        """
        shortcut = x
        
        # Bottleneck path: 1x1 -> 3x3 -> 1x1
        x = layers.Conv2D(filters//4, 1, strides=stride, padding='same', 
                         name=f'{block_name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
        x = layers.ReLU(name=f'{block_name}_relu1')(x)
        
        x = layers.Conv2D(filters//4, 3, padding='same', 
                         name=f'{block_name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
        x = layers.ReLU(name=f'{block_name}_relu2')(x)
        
        x = layers.Conv2D(filters, 1, padding='same', 
                         name=f'{block_name}_conv3')(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn3')(x)
        
        # Adjust shortcut if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                   name=f'{block_name}_shortcut_conv')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{block_name}_shortcut_bn')(shortcut)
        
        # Add residual connection
        x = layers.Add(name=f'{block_name}_add')([shortcut, x])
        x = layers.ReLU(name=f'{block_name}_relu_out')(x)
        
        return x
    
    @staticmethod
    def inception_block(x, filters: int, block_name: str = ""):
        """
        Inception-style block with multiple kernel sizes.
        
        Args:
            x: Input tensor
            filters (int): Number of output filters
            block_name (str): Name prefix for layers
            
        Returns:
            Output tensor
        """
        # 1x1 convolution
        conv1 = layers.Conv2D(filters // 4, 1, padding='same', 
                             name=f'{block_name}_1x1_conv')(x)
        conv1 = layers.BatchNormalization(name=f'{block_name}_1x1_bn')(conv1)
        conv1 = layers.ReLU(name=f'{block_name}_1x1_relu')(conv1)
        
        # 3x3 convolution
        conv3 = layers.Conv2D(filters // 4, 1, padding='same', 
                             name=f'{block_name}_3x3_reduce')(x)
        conv3 = layers.BatchNormalization(name=f'{block_name}_3x3_reduce_bn')(conv3)
        conv3 = layers.ReLU(name=f'{block_name}_3x3_reduce_relu')(conv3)
        conv3 = layers.Conv2D(filters // 4, 3, padding='same', 
                             name=f'{block_name}_3x3_conv')(conv3)
        conv3 = layers.BatchNormalization(name=f'{block_name}_3x3_bn')(conv3)
        conv3 = layers.ReLU(name=f'{block_name}_3x3_relu')(conv3)
        
        # 5x5 convolution
        conv5 = layers.Conv2D(filters // 4, 1, padding='same', 
                             name=f'{block_name}_5x5_reduce')(x)
        conv5 = layers.BatchNormalization(name=f'{block_name}_5x5_reduce_bn')(conv5)
        conv5 = layers.ReLU(name=f'{block_name}_5x5_reduce_relu')(conv5)
        conv5 = layers.Conv2D(filters // 4, 5, padding='same', 
                             name=f'{block_name}_5x5_conv')(conv5)
        conv5 = layers.BatchNormalization(name=f'{block_name}_5x5_bn')(conv5)
        conv5 = layers.ReLU(name=f'{block_name}_5x5_relu')(conv5)
        
        # Max pooling path
        pool = layers.MaxPooling2D(3, strides=1, padding='same', 
                                  name=f'{block_name}_pool')(x)
        pool = layers.Conv2D(filters // 4, 1, padding='same', 
                           name=f'{block_name}_pool_conv')(pool)
        pool = layers.BatchNormalization(name=f'{block_name}_pool_bn')(pool)
        pool = layers.ReLU(name=f'{block_name}_pool_relu')(pool)
        
        # Concatenate all paths
        return layers.Concatenate(name=f'{block_name}_concat')([conv1, conv3, conv5, pool])
    
    @staticmethod
    def dense_block(x, num_layers: int, growth_rate: int, block_name: str = ""):
        """
        DenseNet-style dense block.
        
        Args:
            x: Input tensor
            num_layers (int): Number of dense layers
            growth_rate (int): Growth rate for feature maps
            block_name (str): Name prefix for layers
            
        Returns:
            Output tensor
        """
        features = [x]
        
        for i in range(num_layers):
            x = layers.BatchNormalization(name=f'{block_name}_bn_{i}')(x)
            x = layers.ReLU(name=f'{block_name}_relu_{i}')(x)
            x = layers.Conv2D(growth_rate, 3, padding='same', 
                             name=f'{block_name}_conv_{i}')(x)
            features.append(x)
            x = layers.Concatenate(name=f'{block_name}_concat_{i}')(features)
        
        return x
    
    @staticmethod
    def transition_block(x, reduction: float = 0.5, block_name: str = ""):
        """
        Transition block to reduce feature map size.
        
        Args:
            x: Input tensor
            reduction (float): Reduction factor for channels
            block_name (str): Name prefix for layers
            
        Returns:
            Output tensor
        """
        channels = int(x.shape[-1] * reduction)
        x = layers.BatchNormalization(name=f'{block_name}_bn')(x)
        x = layers.ReLU(name=f'{block_name}_relu')(x)
        x = layers.Conv2D(channels, 1, padding='same', 
                         name=f'{block_name}_conv')(x)
        x = layers.AveragePooling2D(2, strides=2, 
                                   name=f'{block_name}_pool')(x)
        return x
    
    def build_model(self):
        """
        Build the complete hybrid CNN model.
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape, name='input')
        x = inputs
        
        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding='same', name='initial_conv')(x)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.ReLU(name='initial_relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same', name='initial_pool')(x)
        
        # Stage 1: Bottleneck blocks with attention
        x = self.bottleneck_block(x, 64, block_name='stage1_block1')
        x = self.bottleneck_block(x, 64, block_name='stage1_block2')
        x = self.self_attention_block(x)
        
        # Stage 2: Inception + Bottleneck with attention
        x = self.inception_block(x, 128, block_name='stage2_inception')
        x = self.bottleneck_block(x, 128, stride=2, block_name='stage2_bottleneck')
        x = self.self_attention_block(x)
        
        # Stage 3: Dense block with transition
        x = self.dense_block(x, num_layers=4, growth_rate=32, block_name='stage3_dense')
        x = self.transition_block(x, reduction=0.5, block_name='stage3_transition')
        
        # Stage 4: Bottleneck + Inception with attention
        x = self.bottleneck_block(x, 256, stride=2, block_name='stage4_bottleneck')
        x = self.inception_block(x, 256, block_name='stage4_inception')
        x = self.self_attention_block(x)
        
        # Stage 5: Dense block with transition
        x = self.dense_block(x, num_layers=4, growth_rate=32, block_name='stage5_dense')
        x = self.transition_block(x, reduction=0.5, block_name='stage5_transition')
        
        # Final stage: Bottleneck blocks with attention
        x = self.bottleneck_block(x, 512, stride=2, block_name='final_bottleneck1')
        x = self.bottleneck_block(x, 512, block_name='final_bottleneck2')
        x = self.self_attention_block(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dropout(0.5, name='dropout')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = models.Model(inputs, outputs, name='HybridCNN')
        return self.model
    
    def compile_model(self, 
                     optimizer: str = 'adam',
                     learning_rate: float = 0.001,
                     loss: str = 'sparse_categorical_crossentropy',
                     metrics: list = ['accuracy']):
        """
        Compile the model with specified parameters.
        
        Args:
            optimizer (str): Optimizer name
            learning_rate (float): Learning rate
            loss (str): Loss function
            metrics (list): List of metrics to track
        """
        if self.model is None:
            self.build_model()
        
        if optimizer.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def get_model(self):
        """
        Get the built model.
        
        Returns:
            Keras model
        """
        if self.model is None:
            self.build_model()
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        return self.model.summary()


# Example usage and utility functions
def create_hybrid_cnn(num_classes: int, 
                     input_shape: Tuple[int, int, int] = (224, 224, 3),
                     compile_model: bool = True,
                     **compile_kwargs) -> tf.keras.Model:
    """
    Convenience function to create and optionally compile a Hybrid CNN model.
    
    Args:
        num_classes (int): Number of output classes
        input_shape (tuple): Input image shape
        compile_model (bool): Whether to compile the model
        **compile_kwargs: Additional arguments for model compilation
        
    Returns:
        Compiled Keras model
    """
    hybrid_cnn = HybridCNN(num_classes, input_shape)
    model = hybrid_cnn.build_model()
    
    if compile_model:
        # Default compilation parameters
        default_compile_args = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss': 'sparse_categorical_crossentropy',
            'metrics': ['accuracy']
        }
        default_compile_args.update(compile_kwargs)
        hybrid_cnn.compile_model(**default_compile_args)
    
    return model


def get_recommended_callbacks(model_save_path: str = 'best_model.keras'):
    """
    Get recommended callbacks for training.
    
    Args:
        model_save_path (str): Path to save the best model
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    return callbacks


# Example usage
if __name__ == "__main__":
    # Create model for 20 classes
    model = create_hybrid_cnn(num_classes=20, input_shape=(224, 224, 3))
    
    # Print model summary
    model.summary()
    
    # Get recommended callbacks
    callbacks = get_recommended_callbacks('my_model.keras')
    
    print("Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")