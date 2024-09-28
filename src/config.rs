use serde::{Deserialize, Serialize};

/// Configuration for the LLaMA model, parsed from JSON.
/// This structure holds all the essential hyperparameters for the model.
#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct LlamaConfigJson {
    /// Beginning of sequence (BOS) token ID.
    pub bos_token_id: u32,
    /// End of sequence (EOS) token ID.
    pub eos_token_id: u32,
    /// Size of the hidden states in the model.
    pub hidden_size: usize,
    /// Intermediate size for feed-forward layers.
    pub intermediate_size: usize,
    /// Maximum sequence length (position embeddings).
    pub max_position_embeddings: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of key-value heads for attention.
    pub num_key_value_heads: usize,
    /// Size of the vocabulary.
    pub vocab_size: usize,
    /// Epsilon value for RMS normalization. Defaults to `1e-5`.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE (Rotary Positional Embedding) theta value. Defaults to `1e4`.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Data type of tensors in the Torch model (e.g., float32, float16).
    pub torch_dtype: String,
    /// Whether to tie the word embeddings and output layer weights. Defaults to `false`.
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

/// Provides a default epsilon value for RMS normalization (1e-5).
#[inline(always)]
const fn default_rms_norm_eps() -> f32 {
    1e-5
}

/// Provides a default value for RoPE theta (1e4).
#[inline(always)]
const fn default_rope_theta() -> f32 {
    1e4
}

/// Provides a default value for whether word embeddings should be tied (false).
#[inline(always)]
const fn default_tie_word_embeddings() -> bool {
    false
}
