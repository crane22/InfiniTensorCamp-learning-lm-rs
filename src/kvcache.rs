use std::vec;

use crate::tensor::Tensor;

/// KVCache holds the key-value cache for each layer during model inference.
/// This structure is designed to store the past key-value pairs to efficiently
/// handle attention mechanisms during generation or inference.
pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // Key cache (max_seq_len, n_kv_head * dqkv) per layer.
    v_cache: Vec<Tensor<T>>, // Value cache (max_seq_len, n_kv_head * dqkv) per layer.
    max_seq_len: usize,      // Maximum sequence length the cache can store.
    dim: usize,              // Dimension of each key or value vector.
    length: usize,           // Current length of the stored sequence in the cache.
}

impl<T: Default + Copy> KVCache<T> {
    /// Creates a new KVCache for a given number of layers, max sequence length,
    /// and dimension size. Initializes key and value caches for all layers.
    ///
    /// # Parameters:
    /// - `n_layers`: Number of transformer layers.
    /// - `max_seq_len`: Maximum sequence length for the cache.
    /// - `dim`: Dimension of the key and value vectors.
    /// - `init_len`: Initial length of the sequence (e.g., 0 for a new cache).
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            // Initialize key cache for all layers with default values.
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            // Initialize value cache for all layers with default values.
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            max_seq_len,
            dim,
            length: init_len,
        }
    }

    /// Returns a slice of the key cache for a specific layer starting from a given position.
    /// The result includes all key vectors from the `start` position to the current cache length.
    ///
    /// # Parameters:
    /// - `layer`: Index of the transformer layer.
    /// - `start`: Starting position in the sequence for slicing.
    ///
    /// # Returns:
    /// - A tensor slice from the key cache for the given layer.
    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    /// Returns a slice of the value cache for a specific layer starting from a given position.
    /// The result includes all value vectors from the `start` position to the current cache length.
    ///
    /// # Parameters:
    /// - `layer`: Index of the transformer layer.
    /// - `start`: Starting position in the sequence for slicing.
    ///
    /// # Returns:
    /// - A tensor slice from the value cache for the given layer.
    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    /// Increments the current sequence length in the cache by the given sequence length (`seq_len`).
    /// This is called when a new sequence of tokens is added to the cache.
    ///
    /// # Parameters:
    /// - `seq_len`: Number of tokens to add to the sequence.
    pub fn increment(&mut self, seq_len: usize) {
        self.length += seq_len;
    }

    /// Returns the current sequence length in the cache.
    ///
    /// # Returns:
    /// - The number of tokens stored in the cache.
    pub fn len(&self) -> usize {
        self.length
    }
}
