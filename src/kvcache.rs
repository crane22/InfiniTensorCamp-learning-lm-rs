use num_traits::{Float, FromPrimitive};
use std::{usize, vec};

use crate::tensor::Tensor;

pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize,
    length: usize, // length of the current sequence
}

impl<T: Default + Copy + Float + FromPrimitive> KVCache<T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            max_seq_len,
            dim,
            length: init_len,
        }
    }

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        assert!(start < self.length, "Start index out of bounds");
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        assert!(start < self.length, "Start index out of bounds");
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn increment(&mut self, seq_len: usize) {
        assert!(
            self.length + seq_len <= self.max_seq_len,
            "Sequence length exceeds maximum cache size"
        );
        self.length += seq_len;
    }

    pub fn len(&self) -> usize {
        self.length
    }
}
