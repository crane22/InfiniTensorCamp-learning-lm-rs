use std::fs::File;
use std::path::Path;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, matmul_transb, rms_norm, silu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;

/// Llama Model structure for text generation
/// # Parameters:
/// - `vocab`: Vocabulary size.
/// - `n_layers`: Number of layers in the transformer.
/// - `n_q_h`: Number of attention heads for queries.
/// - `n_kv_h`: Number of attention heads for keys and values.
/// - `d`: Dimension of hidden states.
/// - `dqkv`: Dimension of a single Q, K, or V vector.
/// - `di`: Dimension of intermediate states.
/// - `eps`: Epsilon value for RMS normalization.
/// - `rope_theta`: Scaling factor for rotary positional embeddings.
/// - `max_seq_len`: Maximum sequence length.
/// - `params`: The model parameters including weights and embeddings.
/// - `bos_token_id`: Token ID for the beginning of a sentence.
/// - `eos_token_id`: Token ID for the end of a sentence.
pub struct Llama<T> {
    vocab: usize,
    n_layers: usize,
    n_q_h: usize,
    n_kv_h: usize,
    d: usize,
    dqkv: usize,
    di: usize,
    eps: f32,
    rope_theta: f32,
    max_seq_len: usize,
    params: LLamaParams<T>,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl Llama<f32> {
    /// Load a Llama model from a safetensor file
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    /// Initialize a new KV cache for attention layers
    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    /// Forward pass through the model to compute logits
    /// # Parameters:
    /// - `input`: Input tensor of token IDs.
    /// - `cache`: KV cache for storing intermediate attention results.
    /// # Returns:
    /// - A tensor of logits for the next token prediction.
    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len); // Extend cache for the current sequence
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h; // Compute number of groups for multi-head attention

        // Pre-allocate buffers
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        // Process each transformer layer
        for layer in 0..self.n_layers {
            // Apply RMSNorm to residual
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            // Prepare Q, K, V buffers
            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]);
            let k = &mut cache.k_cache(layer, past_seq_len); // Retrieve keys from cache
            let v = &mut cache.v_cache(layer, past_seq_len); // Retrieve values from cache

            // Linear projection for Q, K, V
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);

            // Apply RoPE (rotary positional embeddings)
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            // Perform self-attention
            let full_k = &mut cache.k_cache(layer, 0); // Full keys
            let full_v = &mut cache.v_cache(layer, 0); // Full values
            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            // Output projection
            OP::matmul_transb(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
            );

            // Apply feed-forward network (MLP)
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // Apply final RMSNorm and output projection
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits // Return logits for the next token prediction
    }

    /// Generate a sequence of tokens using the model
    /// # Parameters:
    /// - `token_ids`: Initial sequence of token IDs.
    /// - `max_len`: Maximum number of tokens to generate.
    /// - `top_p`: Top-p sampling probability threshold.
    /// - `top_k`: Top-k sampling threshold.
    /// - `temperature`: Temperature for controlling randomness.
    /// # Returns:
    /// - A vector of generated token IDs.
    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = token_ids.to_vec();
        let mut cache = self.new_cache();
        let mut input_tensor = Tensor::<u32>::new(result.clone(), &vec![result.len()]);

        while result.len() < max_len {
            let logits = self.forward(&input_tensor, &mut cache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }
            input_tensor = Tensor::<u32>::new(vec![next_token], &vec![1]);
        }

        result
    }
}

/// Perform self-attention computation
/// # Parameters:
/// - `hidden_states`: Output tensor for the hidden states.
/// - `att_scores`: Tensor for storing attention scores.
/// - `q`: Query tensor.
/// - `k`: Key tensor.
/// - `v`: Value tensor.
/// - `n_kv_h`: Number of key-value heads.
/// - `n_groups`: Number of attention groups.
/// - `seq_len`: Sequence length.
/// - `total_seq_len`: Total sequence length (including past and current tokens).
/// - `dqkv`: Dimension of a single Q, K, or V vector.
fn self_attention(
    hidden_states: &mut Tensor<f32>,
    att_scores: &mut Tensor<f32>,
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let unit_len = dqkv;
    let q_row_len = n_kv_h * n_groups * unit_len;
    let k_row_len = n_kv_h * unit_len;

    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let hidden_states_data = unsafe { hidden_states.data_mut() };

    // Calculate attention scores (Q @ K^T)
    for m in 0..seq_len {
        for n in 0..total_seq_len {
            for i in 0..n_kv_h {
                let q_offset = m * q_row_len + i * n_groups * unit_len;
                let k_offset = n * k_row_len + i * unit_len;

                for j in 0..n_groups {
                    let q_slice = &q_data[q_offset + j * unit_len..q_offset + (j + 1) * unit_len];
                    let k_slice = &k_data[k_offset..k_offset + unit_len];

                    // Compute dot product between q_slice and k_slice
                    let score = q_slice
                        .iter()
                        .zip(k_slice)
                        .map(|(&q, &k)| q * k)
                        .sum::<f32>()
                        / (unit_len as f32).sqrt();

                    let attn_offset = (i * n_groups + j) * seq_len * total_seq_len;
                    unsafe {
                        att_scores.data_mut()[attn_offset + m * total_seq_len + n] = score;
                    }
                }
            }
        }
    }

    // Apply softmax to attention scores
    OP::masked_softmax(att_scores);

    // Calculate attention output (attn @ V)
    let att_scores_data = att_scores.data();
    let v_row_len = n_kv_h * unit_len;
    let hidden_row_len = n_kv_h * n_groups * unit_len;

    for m in 0..seq_len {
        for i in 0..n_kv_h {
            let v_start = i * unit_len;

            for j in 0..n_groups {
                let attn_offset = (i * n_groups + j) * seq_len * total_seq_len;
                let hidden_offset = m * hidden_row_len + i * n_groups * unit_len + j * unit_len;

                for n in 0..unit_len {
                    let mut sum = 0.0;
                    for k in 0..total_seq_len {
                        let score = att_scores_data[attn_offset + m * total_seq_len + k];
                        sum += score * v_data[v_start + k * v_row_len + n];
                    }
                    hidden_states_data[hidden_offset + n] = sum;
                }
            }
        }
    }
}

/// Feed-forward network (MLP) layer
/// # Parameters:
/// - `residual`: Residual connection tensor.
/// - `hidden_states`: Hidden states output tensor.
/// - `gate`: Gate tensor.
/// - `up`: Up tensor for feed-forward network.
/// - `w_up`: Weight tensor for up projection.
/// - `w_down`: Weight tensor for down projection.
/// - `w_gate`: Weight tensor for gate projection.
/// - `rms_w`: Weight tensor for RMS normalization.
/// - `eps`: Epsilon for RMS normalization.
fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    rms_norm(hidden_states, &residual, rms_w, eps);
    matmul_transb(gate, 0.0, &hidden_states, &w_gate, 1.0);
    matmul_transb(up, 0.0, &hidden_states, w_up, 1.0);
    silu(up, &gate);
    matmul_transb(hidden_states, 0.0, &up, w_down, 1.0);
    *residual = hidden_states.add(residual);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}

#[test]
fn test_self_attention() {
    let seq_len = 2;
    let total_seq_len = 4;
    let n_kv_h = 2;
    let n_groups = 1;
    let dqkv = 3;

    // Initialize simple test tensors for Q, K, and V
    let q_data = vec![
        0.1, 0.2, 0.3, // Q for seq_idx 0, head 0
        0.4, 0.5, 0.6, // Q for seq_idx 1, head 0
        0.7, 0.8, 0.9, // Q for seq_idx 0, head 1
        1.0, 1.1, 1.2, // Q for seq_idx 1, head 1
    ];
    let q = Tensor::<f32>::new(q_data, &vec![seq_len, n_kv_h * n_groups * dqkv]);

    let k_data = vec![
        0.1, 0.2, 0.3, // K for total_seq_idx 0, head 0
        0.4, 0.5, 0.6, // K for total_seq_idx 1, head 0
        0.7, 0.8, 0.9, // K for total_seq_idx 2, head 0
        1.0, 1.1, 1.2, // K for total_seq_idx 3, head 0
        1.3, 1.4, 1.5, // K for total_seq_idx 0, head 1
        1.6, 1.7, 1.8, // K for total_seq_idx 1, head 1
        1.9, 2.0, 2.1, // K for total_seq_idx 2, head 1
        2.2, 2.3, 2.4, // K for total_seq_idx 3, head 1
    ];
    let k = Tensor::<f32>::new(k_data, &vec![total_seq_len, n_kv_h * dqkv]);

    let v_data = vec![
        0.1, 0.2, 0.3, // V for total_seq_idx 0, head 0
        0.4, 0.5, 0.6, // V for total_seq_idx 1, head 0
        0.7, 0.8, 0.9, // V for total_seq_idx 2, head 0
        1.0, 1.1, 1.2, // V for total_seq_idx 3, head 0
        1.3, 1.4, 1.5, // V for total_seq_idx 0, head 1
        1.6, 1.7, 1.8, // V for total_seq_idx 1, head 1
        1.9, 2.0, 2.1, // V for total_seq_idx 2, head 1
        2.2, 2.3, 2.4, // V for total_seq_idx 3, head 1
    ];
    let v = Tensor::<f32>::new(v_data, &vec![total_seq_len, n_kv_h * dqkv]);

    // Initialize attention score tensor and hidden_states
    let mut att_scores = Tensor::<f32>::default(&vec![n_kv_h, n_groups, seq_len, total_seq_len]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, n_kv_h * n_groups * dqkv]);

    // Run self_attention
    self_attention(
        &mut hidden_states,
        &mut att_scores,
        &q,
        &k,
        &v,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );

    // Check the results (example expected results, calculated manually)
    let expected_hidden_states = Tensor::<f32>::new(
        vec![
            0.7825454, 0.8825454, 0.9825454, // Output for seq_idx 0, head 0
            1.1990090, 1.2990088, 1.3990089, // Output for seq_idx 0, head 1
            1.5267198, 1.6267197, 1.7267196, // Output for seq_idx 1, head 0
            1.9442390, 2.0442388, 2.1442390, // Output for seq_idx 1, head 1
        ],
        &vec![seq_len, n_kv_h * n_groups * dqkv],
    );

    assert!(hidden_states.close_to(&expected_hidden_states, 1e-3));
}

#[test]
fn test_forward() {
    // Define a small dummy model configuration
    let vocab_size = 10;
    let num_hidden_layers = 2;
    let num_attention_heads = 4;
    let num_key_value_heads = 4;
    let hidden_size = 8;
    let intermediate_size = 16;
    let rms_norm_eps = 1e-6;
    let rope_theta = 10000.0;
    let max_position_embeddings = 4;
    let params = LLamaParams {
        embedding_table: Tensor::<f32>::new(
            vec![0.1; vocab_size * hidden_size],
            &vec![vocab_size, hidden_size],
        ),
        rms_att_w: vec![
            Tensor::<f32>::new(vec![0.1; hidden_size], &vec![hidden_size]);
            num_hidden_layers
        ],
        wq: vec![
            Tensor::<f32>::new(
                vec![0.1; hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        wk: vec![
            Tensor::<f32>::new(
                vec![0.1; hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        wv: vec![
            Tensor::<f32>::new(
                vec![0.1; hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        wo: vec![
            Tensor::<f32>::new(
                vec![0.1; hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        w_up: vec![
            Tensor::<f32>::new(
                vec![0.1; intermediate_size * hidden_size],
                &vec![intermediate_size, hidden_size]
            );
            num_hidden_layers
        ],
        w_down: vec![
            Tensor::<f32>::new(
                vec![0.1; hidden_size * intermediate_size],
                &vec![hidden_size, intermediate_size]
            );
            num_hidden_layers
        ],
        w_gate: vec![
            Tensor::<f32>::new(
                vec![0.1; intermediate_size * hidden_size],
                &vec![intermediate_size, hidden_size]
            );
            num_hidden_layers
        ],
        rms_ffn_w: vec![
            Tensor::<f32>::new(vec![0.1; hidden_size], &vec![hidden_size]);
            num_hidden_layers
        ],
        rms_out_w: Tensor::<f32>::new(vec![0.1; hidden_size], &vec![hidden_size]),
        lm_head: Tensor::<f32>::new(
            vec![0.1; vocab_size * hidden_size],
            &vec![vocab_size, hidden_size],
        ),
    };
    let bos_token_id = 1;
    let eos_token_id = 2;

    // Define the test model with the parameters
    let model = Llama {
        vocab: vocab_size,
        n_layers: num_hidden_layers,
        n_q_h: num_attention_heads,
        n_kv_h: num_key_value_heads,
        d: hidden_size,
        dqkv: hidden_size / num_attention_heads,
        di: intermediate_size,
        eps: rms_norm_eps,
        rope_theta: rope_theta,
        max_seq_len: max_position_embeddings,
        params: params,
        bos_token_id: bos_token_id,
        eos_token_id: eos_token_id,
    };

    // Create dummy input and cache for testing
    let input = Tensor::<u32>::new(vec![1, 2, 3], &vec![3]); // A simple input sequence [1, 2, 3]
    let mut cache = KVCache::<f32>::new(
        num_hidden_layers,
        max_position_embeddings,
        num_key_value_heads * (hidden_size / num_key_value_heads),
        0,
    );

    // Call forward function
    let output = model.forward(&input, &mut cache);

    // Print output for manual inspection
    println!("Output: {:?}", output.data());

    // Basic checks to ensure the output is the correct shape and type
    assert_eq!(
        output.shape(),
        &vec![1, vocab_size],
        "Output shape should match [1, vocab_size]"
    );
}

#[test]
fn test_generate() {
    use std::path::PathBuf;
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("story");
    let model = Llama::from_safetensors(&model_dir);
    // 输入的 token 序列
    let token_ids = vec![0, 123, 456]; // 一些初始的token序列

    // 调用 generate 函数进行文本生成
    let generated_tokens = model.generate(&token_ids, 50, 0.9, 50, 1.0);

    // 打印生成的 token 序列
    println!("Generated Tokens: {:?}", generated_tokens);

    // 可添加一些简单的断言进行检查
    assert!(generated_tokens.len() <= 50, "生成的token数不应超过max_len");
    assert!(
        generated_tokens.contains(&model.eos_token_id),
        "生成结果应包含EOS token"
    );
}
