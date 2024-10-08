use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use half::f16;
use num_traits::{Float, FromPrimitive};
use safetensors::{Dtype, SafeTensors};

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl<T: Float + Default + Copy + FromPrimitive> LLamaParams<T> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            let tensor_view = safetensor
                .tensor(name)
                .unwrap_or_else(|_| panic!("Tensor `{}` not found in safetensors", name));
            let dtype = tensor_view.dtype();
            let element_size = dtype.size();
            match dtype {
                Dtype::F32 => {
                    let data = tensor_view
                        .data()
                        .chunks(element_size)
                        .map(|chunk| {
                            f32::from_le_bytes(chunk.try_into().expect("Chunk is not 4 bytes long"))
                        })
                        .map(|v| T::from_f32(v).unwrap())
                        .collect();
                    let shape = tensor_view.shape().to_vec();
                    Tensor::<T>::new(data, &shape)
                }
                Dtype::F16 => {
                    let data = tensor_view
                        .data()
                        .chunks(element_size)
                        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
                        .map(|f| T::from_f32(f.to_f32()).unwrap())
                        .collect();
                    let shape = tensor_view.shape().to_vec();
                    Tensor::<T>::new(data, &shape)
                }
                _ => panic!("Unsupported dtype `{:?}` found in safetensors", dtype),
            }
        };

        let n_layers = config.num_hidden_layers;
        LLamaParams {
            embedding_table: if config.tie_word_embeddings {
                get_tensor("lm_head.weight")
            } else {
                get_tensor("model.embed_tokens.weight")
            },
            rms_att_w: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.input_layernorm.weight")))
                .collect(),
            wq: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.o_proj.weight")))
                .collect(),
            rms_ffn_w: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.mlp.up_proj.weight")))
                .collect(),
            w_gate: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.mlp.gate_proj.weight")))
                .collect(),
            w_down: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.mlp.down_proj.weight")))
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
