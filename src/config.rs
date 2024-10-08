use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum TensorDType {
    Float32,
    Float16,
}

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
    #[serde(default = "default_torch_dtype")]
    pub torch_dtype: TensorDType,
    /// Whether to tie the word embeddings and output layer weights. Defaults to `false`.
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

impl LlamaConfigJson {
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err("hidden_size cannot be zero".to_string());
        }
        if self.max_position_embeddings < 2 {
            return Err("max_position_embeddings should be at least 2".to_string());
        }
        if self.num_attention_heads == 0 || self.num_attention_heads > self.hidden_size {
            return Err(
                format!(
                    "num_attention_heads must be non-zero and less than or equal to hidden_size ({}). Found: {}",
                    self.hidden_size, self.num_attention_heads
                )
            );
        }
        if self.vocab_size == 0 {
            return Err("vocab_size cannot be zero".to_string());
        }
        if self.intermediate_size == 0 {
            return Err("intermediate_size cannot be zero".to_string());
        }
        Ok(())
    }
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

/// Provides a default value for the tensor data type (float32).
#[inline(always)]
fn default_torch_dtype() -> TensorDType {
    TensorDType::Float32
}
