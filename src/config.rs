use serde::{
    de::{self, Deserializer, Unexpected},
    Deserialize, Serialize,
};

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum TensorDType {
    Float32,
    Float16,
    BFloat16,
}

impl<'de> Deserialize<'de> for TensorDType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s: String = Deserialize::deserialize(deserializer)?;
        match s.as_str() {
            "float32" => Ok(TensorDType::Float32),
            "float16" => Ok(TensorDType::Float16),
            "bfloat16" => Ok(TensorDType::BFloat16),
            _ => {
                // If the value doesn't match any variant, return a default or custom error
                Err(de::Error::invalid_value(
                    Unexpected::Str(&s),
                    &"valid tensor dtype",
                ))
            }
        }
    }
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

/// Provides a default value for the tensor data type (float32).
#[inline(always)]
fn default_torch_dtype() -> TensorDType {
    TensorDType::Float32
}

/// Provides a default value for whether word embeddings should be tied (false).
#[inline(always)]
const fn default_tie_word_embeddings() -> bool {
    false
}
