use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors};

/// A struct representing the parameters of a LLaMA model loaded from SafeTensors format.
/// Each field holds a tensor corresponding to a specific part of the model.
pub struct LLamaParams<T> {
    pub embedding_table: Tensor<T>, // (vocab_size, dim) - Embedding lookup table for token IDs
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size,) x layers - RMS normalization weights for attention layers
    pub wq: Vec<Tensor<T>>, // (n_heads * head_size, hidden_size) x layers - Query projection weights
    pub wk: Vec<Tensor<T>>, // (n_kv_heads * head_size, hidden_size) x layers - Key projection weights
    pub wv: Vec<Tensor<T>>, // (n_kv_heads * head_size, hidden_size) x layers - Value projection weights
    pub wo: Vec<Tensor<T>>, // (hidden_size, n_heads * head_size) x layers - Output projection weights
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size,) x layers - RMS normalization weights for FFN layers
    pub w_up: Vec<Tensor<T>>, // (intermediate_size, hidden_size) x layers - FFN up-projection weights
    pub w_gate: Vec<Tensor<T>>, // (intermediate_size, hidden_size) x layers - FFN gating projection weights
    pub w_down: Vec<Tensor<T>>, // (hidden_size, intermediate_size) x layers - FFN down-projection weights
    pub rms_out_w: Tensor<T>,   // (hidden_size,) - Final layer RMS normalization weights
    pub lm_head: Tensor<T>,     // (vocab_size, dim) - Final projection to vocabulary (output layer)
}

impl LLamaParams<f32> {
    /// Loads the parameters of the LLaMA model from a SafeTensors file.
    /// Converts the SafeTensor data into `Tensor<f32>` format and initializes the `LLamaParams` struct.
    ///
    /// # Parameters:
    /// - `safetensor`: A SafeTensors instance that holds the model's weights.
    /// - `config`: A configuration object providing the model's settings (like layers, dimensions, etc.).
    ///
    /// # Returns:
    /// - An initialized `LLamaParams` struct containing all the model parameters.
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let layers = config.num_hidden_layers; // Number of transformer layers in the model

        /// Helper function to extract a tensor from the SafeTensors object using a given name.
        /// Converts the raw data into `Tensor<f32>` format.
        ///
        /// # Parameters:
        /// - `name`: The name of the tensor in the SafeTensors file.
        /// - `safetensor`: The SafeTensors object containing the data.
        ///
        /// # Returns:
        /// - A tensor of type `Tensor<f32>`.
        fn get_tensor(name: &str, safetensor: &SafeTensors) -> Tensor<f32> {
            let tensor_view = safetensor.tensor(name).unwrap();
            let vec_f32s = tensor_view
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())) // Converts 4-byte chunks into f32
                .collect::<Vec<f32>>();
            Tensor::new(vec_f32s, &tensor_view.shape().to_vec()) // Create a `Tensor<f32>` from raw data
        }

        /// Helper function to extract tensors for each layer from the SafeTensors object.
        /// It loads a vector of tensors corresponding to a particular weight type (e.g., wq, wk).
        ///
        /// # Parameters:
        /// - `safetensor`: The SafeTensors object containing the data.
        /// - `layers`: Number of layers in the model.
        /// - `name`: The common name for the tensor (layer-specific names will be constructed).
        ///
        /// # Returns:
        /// - A vector of tensors, one for each layer.
        fn get_tensor_vec(safetensor: &SafeTensors, layers: usize, name: &str) -> Vec<Tensor<f32>> {
            (0..layers)
                .map(|layer_idx| {
                    let tensor_name = format!("model.layers.{layer_idx}.{name}");
                    get_tensor(&tensor_name, safetensor)
                })
                .collect()
        }

        LLamaParams {
            // Load embedding lookup table and final projection layer
            embedding_table: get_tensor("lm_head.weight", safetensor), // (vocab_size, dim)
            rms_out_w: get_tensor("model.norm.weight", safetensor),    // (hidden_size,)
            lm_head: get_tensor("lm_head.weight", safetensor),         // (vocab_size, dim)

            // Load transformer layer-specific weights
            rms_att_w: get_tensor_vec(safetensor, layers, "input_layernorm.weight"),
            rms_ffn_w: get_tensor_vec(safetensor, layers, "post_attention_layernorm.weight"),
            wq: get_tensor_vec(safetensor, layers, "self_attn.q_proj.weight"),
            wk: get_tensor_vec(safetensor, layers, "self_attn.k_proj.weight"),
            wv: get_tensor_vec(safetensor, layers, "self_attn.v_proj.weight"),
            wo: get_tensor_vec(safetensor, layers, "self_attn.o_proj.weight"),
            w_up: get_tensor_vec(safetensor, layers, "mlp.up_proj.weight"),
            w_gate: get_tensor_vec(safetensor, layers, "mlp.gate_proj.weight"),
            w_down: get_tensor_vec(safetensor, layers, "mlp.down_proj.weight"),
        }
    }
}
