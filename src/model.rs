use crate::config::LlamaConfigJson;
use crate::float::FloatLike; // Importing the FloatLike trait
use crate::kvcache::KVCache;
use crate::operators::{self as OP, matmul_transb, rms_norm, silu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::vec;

/// Llama Model structure for text generation
pub struct Llama<T: FloatLike> {
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

impl<T: FloatLike> Llama<T> {
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
            params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    /// Initialize a new KV cache for attention layers
    pub fn new_cache(&self) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    /// Forward pass through the model to compute logits
    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<T> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len); // Extend cache for the current sequence
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h; // Compute number of groups for multi-head attention

        // Pre-allocate buffers
        let mut residual = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<T>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<T>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<T>::default(&vec![seq_len, self.di]);

        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        // Process each transformer layer
        for layer in 0..self.n_layers {
            // Apply RMSNorm to residual
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                T::from_f32(self.eps),
            );

            // Prepare Q, K, V buffers
            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]);
            let k = &mut cache.k_cache(layer, past_seq_len); // Retrieve keys from cache
            let v = &mut cache.v_cache(layer, past_seq_len); // Retrieve values from cache

            // Linear projection for Q, K, V
            OP::matmul_transb(
                q,
                T::zero(),
                &hidden_states,
                &self.params.wq[layer],
                T::one(),
            );
            OP::matmul_transb(
                k,
                T::zero(),
                &hidden_states,
                &self.params.wk[layer],
                T::one(),
            );
            OP::matmul_transb(
                v,
                T::zero(),
                &hidden_states,
                &self.params.wv[layer],
                T::one(),
            );

            // Apply RoPE (rotary positional embeddings)
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                T::from_f32(self.rope_theta),
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                T::from_f32(self.rope_theta),
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
                T::one(),
                &hidden_states,
                &self.params.wo[layer],
                T::one(),
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
                T::from_f32(self.eps),
            );
        }

        // Apply final RMSNorm and output projection
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            T::from_f32(self.eps),
        );

        let mut logits = Tensor::<T>::default(&vec![1, self.vocab]);
        OP::matmul_transb(
            &mut logits,
            T::zero(),
            &hidden_states,
            &self.params.lm_head,
            T::one(),
        );

        logits // Return logits for the next token prediction
    }

    /// Generate a sequence of tokens using the model
    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: T,
        top_k: u32,
        temperature: T,
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

    /// Stream-based generation of tokens, outputting one token at a time.
    pub fn stream_generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: T,
        top_k: u32,
        temperature: T,
        tokenizer: &tokenizers::Tokenizer, // Use tokenizer to decode token IDs to text
    ) {
        let mut result = token_ids.to_vec(); // Store initial token IDs (e.g., from user input)
        let mut cache = self.new_cache(); // Initialize KVCache
        let mut input_tensor = Tensor::<u32>::new(result.clone(), &vec![result.len()]); // Create input tensor

        // Loop until the generated sequence reaches the max length or the EOS token is generated
        while result.len() < max_len {
            let logits = self.forward(&input_tensor, &mut cache); // Forward pass
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature); // Sample next token

            // Check if we reached the EOS token, stop generation
            if next_token == self.eos_token_id {
                break;
            }

            // Add new token to result
            result.push(next_token);

            // Decode the latest token to its corresponding text and print it
            let output_text = tokenizer.decode(&[next_token], true).unwrap();
            print!("{}", output_text); // Stream the output to the terminal
            io::stdout().flush().unwrap(); // Ensure the text is displayed immediately

            // Prepare next input for the next forward pass
            input_tensor = Tensor::<u32>::new(vec![next_token], &vec![1]);
        }

        println!(); // Ensure the terminal cursor moves to a new line after streaming
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
fn self_attention<T: FloatLike>(
    hidden_states: &mut Tensor<T>,
    att_scores: &mut Tensor<T>,
    q: &Tensor<T>,
    k: &Tensor<T>,
    v: &Tensor<T>,
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
                    let score = q_slice.iter().zip(k_slice).map(|(&q, &k)| q * k).sum::<T>()
                        / T::from_f32((unit_len as f32).sqrt());

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
                    let mut sum = T::zero();
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
fn mlp<T: FloatLike>(
    residual: &mut Tensor<T>,
    hidden_states: &mut Tensor<T>,
    gate: &mut Tensor<T>,
    up: &mut Tensor<T>,
    w_up: &Tensor<T>,
    w_down: &Tensor<T>,
    w_gate: &Tensor<T>,
    rms_w: &Tensor<T>,
    eps: T,
) {
    rms_norm(hidden_states, &residual, rms_w, eps);
    matmul_transb(gate, T::zero(), &hidden_states, &w_gate, T::one());
    matmul_transb(up, T::zero(), &hidden_states, w_up, T::one());
    silu(up, &gate);
    matmul_transb(hidden_states, T::zero(), &up, w_down, T::one());
    *residual = hidden_states.add(residual);
}

#[test]
pub fn test_mlp_floatlike() {
    // Helper function to run the test for a generic FloatLike type (f32 or f16)
    fn run_mlp_test<T: FloatLike + Default + Copy + PartialOrd + std::fmt::Debug>() {
        let seq_len = 4;
        let d = 2;
        let di = 3;

        let mut residual = Tensor::<T>::new(
            vec![
                T::from_f32(1.0),
                T::from_f32(1.0),
                T::from_f32(1.0),
                T::from_f32(1.0),
                T::from_f32(1.0),
                T::from_f32(1.0),
                T::from_f32(1.0),
                T::from_f32(1.0),
            ],
            &vec![seq_len, d],
        );

        let mut hidden_states = Tensor::<T>::default(&vec![seq_len, d]);
        let mut gate_buf = Tensor::<T>::default(&vec![seq_len, di]);
        let mut up_buf = Tensor::<T>::default(&vec![seq_len, di]);

        let w_up = Tensor::<T>::new(
            vec![
                T::from_f32(0.1),
                T::from_f32(0.2),
                T::from_f32(0.3),
                T::from_f32(0.4),
                T::from_f32(0.5),
                T::from_f32(0.6),
            ],
            &vec![di, d],
        );

        let w_down = Tensor::<T>::new(
            vec![
                T::from_f32(0.1),
                T::from_f32(0.2),
                T::from_f32(0.3),
                T::from_f32(0.4),
                T::from_f32(0.5),
                T::from_f32(0.6),
            ],
            &vec![d, di],
        );

        let w_gate = Tensor::<T>::new(
            vec![
                T::from_f32(0.1),
                T::from_f32(0.2),
                T::from_f32(0.3),
                T::from_f32(0.4),
                T::from_f32(0.5),
                T::from_f32(0.6),
            ],
            &vec![di, d],
        );

        let rms_w = Tensor::<T>::new(vec![T::from_f32(1.0), T::from_f32(1.0)], &vec![d]);
        let eps = T::from_f32(1e-6);

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
            &Tensor::<T>::new(
                vec![
                    T::from_f32(1.3429964),
                    T::from_f32(1.7290739),
                    T::from_f32(1.3429964),
                    T::from_f32(1.7290739),
                    T::from_f32(1.3429964),
                    T::from_f32(1.7290739),
                    T::from_f32(1.3429964),
                    T::from_f32(1.7290739),
                ],
                &vec![seq_len, d]
            ),
            1e-3
        ));
    }

    // Run the MLP test for both f32 and f16
    run_mlp_test::<f32>();
    run_mlp_test::<half::f16>();
}

#[test]
pub fn test_load_safetensors_floatlike() {
    // Helper function to test loading safetensors for generic type T (f32 or f16)
    fn run_load_safetensors_test<T: FloatLike + Default + Copy + PartialOrd + std::fmt::Debug>() {
        use std::path::PathBuf;

        // Path to the model directory
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("story");

        // Load model using generic FloatLike type
        let model: Llama<T> = Llama::from_safetensors(model_dir);

        // Assertions on model configuration parameters
        assert_eq!(model.vocab, 2048);
        assert_eq!(model.n_layers, 2);
        assert_eq!(model.n_q_h, 8);
        assert_eq!(model.n_kv_h, 4);
        assert_eq!(model.d, 128);
        assert_eq!(model.dqkv, 16);
        assert_eq!(model.di, 384);

        // Assertions on specific parameter values
        assert!(model.params.embedding_table.data()[50].float_eq(T::from_f32(0.14453125), 1e-6));
        assert!(
            model.params.lm_head.data()[10].float_eq(model.params.embedding_table.data()[10], 1e-6)
        );
        assert!(model.params.rms_att_w[0].data()[10].float_eq(T::from_f32(0.18652344), 1e-6));
        assert!(model.params.rms_ffn_w[1].data()[10].float_eq(T::from_f32(0.32421875), 1e-6));
        assert!(model.params.rms_out_w.data()[100].float_eq(T::from_f32(0.73046875), 1e-6));
        assert!(model.params.w_down[0].data()[100].float_eq(T::from_f32(-0.0625), 1e-6));
        assert!(model.params.w_up[0].data()[100].float_eq(T::from_f32(1.46875), 1e-6));
        assert!(model.params.w_gate[1].data()[100].float_eq(T::from_f32(0.296875), 1e-6));
        assert!(model.params.wq[1].data()[100].float_eq(T::from_f32(0.032226563), 1e-6));
        assert!(model.params.wk[1].data()[100].float_eq(T::from_f32(-0.21386719), 1e-6));
        assert!(model.params.wv[0].data()[100].float_eq(T::from_f32(0.041015625), 1e-6));
        assert!(model.params.wo[0].data()[100].float_eq(T::from_f32(0.01965332), 1e-6));
    }

    // Run the test for both f32 and f16
    run_load_safetensors_test::<f32>();
    run_load_safetensors_test::<half::f16>();
}

#[test]
pub fn test_self_attention_floatlike() {
    // Helper function to test self_attention for generic type T (f32 or f16)
    fn run_self_attention_test<T: FloatLike + Default + Copy + PartialOrd + std::fmt::Debug>() {
        let seq_len = 2;
        let total_seq_len = 4;
        let n_kv_h = 2;
        let n_groups = 1;
        let dqkv = 3;

        // Initialize simple test tensors for Q, K, and V
        let q_data = vec![
            T::from_f32(0.1),
            T::from_f32(0.2),
            T::from_f32(0.3), // Q for seq_idx 0, head 0
            T::from_f32(0.4),
            T::from_f32(0.5),
            T::from_f32(0.6), // Q for seq_idx 1, head 0
            T::from_f32(0.7),
            T::from_f32(0.8),
            T::from_f32(0.9), // Q for seq_idx 0, head 1
            T::from_f32(1.0),
            T::from_f32(1.1),
            T::from_f32(1.2), // Q for seq_idx 1, head 1
        ];
        let q = Tensor::<T>::new(q_data, &vec![seq_len, n_kv_h * n_groups * dqkv]);

        let k_data = vec![
            T::from_f32(0.1),
            T::from_f32(0.2),
            T::from_f32(0.3), // K for total_seq_idx 0, head 0
            T::from_f32(0.4),
            T::from_f32(0.5),
            T::from_f32(0.6), // K for total_seq_idx 1, head 0
            T::from_f32(0.7),
            T::from_f32(0.8),
            T::from_f32(0.9), // K for total_seq_idx 2, head 0
            T::from_f32(1.0),
            T::from_f32(1.1),
            T::from_f32(1.2), // K for total_seq_idx 3, head 0
            T::from_f32(1.3),
            T::from_f32(1.4),
            T::from_f32(1.5), // K for total_seq_idx 0, head 1
            T::from_f32(1.6),
            T::from_f32(1.7),
            T::from_f32(1.8), // K for total_seq_idx 1, head 1
            T::from_f32(1.9),
            T::from_f32(2.0),
            T::from_f32(2.1), // K for total_seq_idx 2, head 1
            T::from_f32(2.2),
            T::from_f32(2.3),
            T::from_f32(2.4), // K for total_seq_idx 3, head 1
        ];
        let k = Tensor::<T>::new(k_data, &vec![total_seq_len, n_kv_h * dqkv]);

        let v_data = vec![
            T::from_f32(0.1),
            T::from_f32(0.2),
            T::from_f32(0.3), // V for total_seq_idx 0, head 0
            T::from_f32(0.4),
            T::from_f32(0.5),
            T::from_f32(0.6), // V for total_seq_idx 1, head 0
            T::from_f32(0.7),
            T::from_f32(0.8),
            T::from_f32(0.9), // V for total_seq_idx 2, head 0
            T::from_f32(1.0),
            T::from_f32(1.1),
            T::from_f32(1.2), // V for total_seq_idx 3, head 0
            T::from_f32(1.3),
            T::from_f32(1.4),
            T::from_f32(1.5), // V for total_seq_idx 0, head 1
            T::from_f32(1.6),
            T::from_f32(1.7),
            T::from_f32(1.8), // V for total_seq_idx 1, head 1
            T::from_f32(1.9),
            T::from_f32(2.0),
            T::from_f32(2.1), // V for total_seq_idx 2, head 1
            T::from_f32(2.2),
            T::from_f32(2.3),
            T::from_f32(2.4), // V for total_seq_idx 3, head 1
        ];
        let v = Tensor::<T>::new(v_data, &vec![total_seq_len, n_kv_h * dqkv]);

        // Initialize attention score tensor and hidden_states
        let mut att_scores = Tensor::<T>::default(&vec![n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut hidden_states = Tensor::<T>::default(&vec![seq_len, n_kv_h * n_groups * dqkv]);

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

        // Check the results (example expected results, calculated manually for both f32 and f16)
        let expected_hidden_states = Tensor::<T>::new(
            vec![
                T::from_f32(0.7825454),
                T::from_f32(0.8825454),
                T::from_f32(0.9825454), // Output for seq_idx 0, head 0
                T::from_f32(1.1990090),
                T::from_f32(1.2990088),
                T::from_f32(1.3990089), // Output for seq_idx 0, head 1
                T::from_f32(1.5267198),
                T::from_f32(1.6267197),
                T::from_f32(1.7267196), // Output for seq_idx 1, head 0
                T::from_f32(1.9442390),
                T::from_f32(2.0442388),
                T::from_f32(2.1442390), // Output for seq_idx 1, head 1
            ],
            &vec![seq_len, n_kv_h * n_groups * dqkv],
        );

        // Use float_eq for comparison of floating-point values
        assert!(hidden_states.close_to(&expected_hidden_states, 1e-3));
    }

    // Run the test for both f32 and f16
    run_self_attention_test::<f32>();
    run_self_attention_test::<half::f16>();
}

#[test]
fn test_self_attention_print() {
    let seq_len = 2;
    let total_seq_len = 4;
    let n_kv_h = 2;
    let n_groups = 1;
    let dqkv = 3;

    // Initialize simple test tensors for Q, K, and V
    let q_data_f32 = vec![
        0.1, 0.2, 0.3, // Q for seq_idx 0, head 0
        0.4, 0.5, 0.6, // Q for seq_idx 1, head 0
        0.7, 0.8, 0.9, // Q for seq_idx 0, head 1
        1.0, 1.1, 1.2, // Q for seq_idx 1, head 1
    ];
    let q_f32 = Tensor::<f32>::new(q_data_f32.clone(), &vec![seq_len, n_kv_h * n_groups * dqkv]);

    let k_data_f32 = vec![
        0.1, 0.2, 0.3, // K for total_seq_idx 0, head 0
        0.4, 0.5, 0.6, // K for total_seq_idx 1, head 0
        0.7, 0.8, 0.9, // K for total_seq_idx 2, head 0
        1.0, 1.1, 1.2, // K for total_seq_idx 3, head 0
        1.3, 1.4, 1.5, // K for total_seq_idx 0, head 1
        1.6, 1.7, 1.8, // K for total_seq_idx 1, head 1
        1.9, 2.0, 2.1, // K for total_seq_idx 2, head 1
        2.2, 2.3, 2.4, // K for total_seq_idx 3, head 1
    ];
    let k_f32 = Tensor::<f32>::new(k_data_f32.clone(), &vec![total_seq_len, n_kv_h * dqkv]);

    let v_data_f32 = vec![
        0.1, 0.2, 0.3, // V for total_seq_idx 0, head 0
        0.4, 0.5, 0.6, // V for total_seq_idx 1, head 0
        0.7, 0.8, 0.9, // V for total_seq_idx 2, head 0
        1.0, 1.1, 1.2, // V for total_seq_idx 3, head 0
        1.3, 1.4, 1.5, // V for total_seq_idx 0, head 1
        1.6, 1.7, 1.8, // V for total_seq_idx 1, head 1
        1.9, 2.0, 2.1, // V for total_seq_idx 2, head 1
        2.2, 2.3, 2.4, // V for total_seq_idx 3, head 1
    ];
    let v_f32 = Tensor::<f32>::new(v_data_f32.clone(), &vec![total_seq_len, n_kv_h * dqkv]);

    // Initialize attention score tensor and hidden_states
    let mut att_scores_f32 =
        Tensor::<f32>::default(&vec![n_kv_h, n_groups, seq_len, total_seq_len]);
    let mut hidden_states_f32 = Tensor::<f32>::default(&vec![seq_len, n_kv_h * n_groups * dqkv]);

    // Run self_attention for f32
    self_attention(
        &mut hidden_states_f32,
        &mut att_scores_f32,
        &q_f32,
        &k_f32,
        &v_f32,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );

    // Print the hidden states for f32
    println!("f32 hidden_states: {:?}", hidden_states_f32.data());

    use half::f16;
    // Now run the same for f16
    let q_f16 = Tensor::<f16>::new(
        q_data_f32.iter().map(|&x| f16::from_f32(x)).collect(),
        &vec![seq_len, n_kv_h * n_groups * dqkv],
    );
    let k_f16 = Tensor::<f16>::new(
        k_data_f32.iter().map(|&x| f16::from_f32(x)).collect(),
        &vec![total_seq_len, n_kv_h * dqkv],
    );
    let v_f16 = Tensor::<f16>::new(
        v_data_f32.iter().map(|&x| f16::from_f32(x)).collect(),
        &vec![total_seq_len, n_kv_h * dqkv],
    );

    let mut att_scores_f16 =
        Tensor::<f16>::default(&vec![n_kv_h, n_groups, seq_len, total_seq_len]);
    let mut hidden_states_f16 = Tensor::<f16>::default(&vec![seq_len, n_kv_h * n_groups * dqkv]);

    self_attention(
        &mut hidden_states_f16,
        &mut att_scores_f16,
        &q_f16,
        &k_f16,
        &v_f16,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );

    // Print the hidden states for f16
    println!("f16 hidden_states: {:?}", hidden_states_f16.data());

    // Now you can assert equality with expected results or continue with the test logic as required.
}
#[test]
fn test_forward_floatlike() {
    use half::f16;

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

    // Prepare test parameters for both f32 and f16
    let params_f32 = LLamaParams {
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

    let params_f16 = LLamaParams {
        embedding_table: Tensor::<f16>::new(
            vec![f16::from_f32(0.1); vocab_size * hidden_size],
            &vec![vocab_size, hidden_size],
        ),
        rms_att_w: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size],
                &vec![hidden_size]
            );
            num_hidden_layers
        ],
        wq: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        wk: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        wv: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        wo: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        w_up: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); intermediate_size * hidden_size],
                &vec![intermediate_size, hidden_size]
            );
            num_hidden_layers
        ],
        w_down: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size * intermediate_size],
                &vec![hidden_size, intermediate_size]
            );
            num_hidden_layers
        ],
        w_gate: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); intermediate_size * hidden_size],
                &vec![intermediate_size, hidden_size]
            );
            num_hidden_layers
        ],
        rms_ffn_w: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size],
                &vec![hidden_size]
            );
            num_hidden_layers
        ],
        rms_out_w: Tensor::<f16>::new(vec![f16::from_f32(0.1); hidden_size], &vec![hidden_size]),
        lm_head: Tensor::<f16>::new(
            vec![f16::from_f32(0.1); vocab_size * hidden_size],
            &vec![vocab_size, hidden_size],
        ),
    };

    let bos_token_id = 1;
    let eos_token_id = 32000;

    // Define the test model with the parameters for f32
    let model_f32 = Llama {
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
        params: params_f32,
        bos_token_id: bos_token_id,
        eos_token_id: eos_token_id,
    };

    // Define the test model with the parameters for f16
    let model_f16 = Llama {
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
        params: params_f16,
        bos_token_id: bos_token_id,
        eos_token_id: eos_token_id,
    };

    // Create dummy input and cache for testing
    let input_f32 = Tensor::<u32>::new(vec![1, 2, 3], &vec![3]); // A simple input sequence [1, 2, 3]
    let input_f16 = Tensor::<u32>::new(vec![1, 2, 3], &vec![3]); // Same input sequence for f16
    let mut cache_f32 = KVCache::<f32>::new(
        num_hidden_layers,
        max_position_embeddings,
        num_key_value_heads * (hidden_size / num_key_value_heads),
        0,
    );
    let mut cache_f16 = KVCache::<f16>::new(
        num_hidden_layers,
        max_position_embeddings,
        num_key_value_heads * (hidden_size / num_key_value_heads),
        0,
    );

    // Call forward function for f32
    let output_f32 = model_f32.forward(&input_f32, &mut cache_f32);
    println!("f32 Output: {:?}", output_f32.data());

    // Call forward function for f16
    let output_f16 = model_f16.forward(&input_f16, &mut cache_f16);
    println!("f16 Output: {:?}", output_f16.data());

    // Basic checks to ensure the output is the correct shape and type for f32
    assert_eq!(
        output_f32.shape(),
        &vec![1, vocab_size],
        "f32: Output shape should match [1, vocab_size]"
    );

    // Basic checks to ensure the output is the correct shape and type for f16
    assert_eq!(
        output_f16.shape(),
        &vec![1, vocab_size],
        "f16: Output shape should match [1, vocab_size]"
    );
}

#[test]
fn test_generate_floatlike() {
    use half::f16;
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

    // Prepare test parameters for both f32 and f16
    let params_f32 = LLamaParams {
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

    let params_f16 = LLamaParams {
        embedding_table: Tensor::<f16>::new(
            vec![f16::from_f32(0.1); vocab_size * hidden_size],
            &vec![vocab_size, hidden_size],
        ),
        rms_att_w: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size],
                &vec![hidden_size]
            );
            num_hidden_layers
        ],
        wq: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        wk: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        wv: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        wo: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size * hidden_size],
                &vec![hidden_size, hidden_size]
            );
            num_hidden_layers
        ],
        w_up: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); intermediate_size * hidden_size],
                &vec![intermediate_size, hidden_size]
            );
            num_hidden_layers
        ],
        w_down: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size * intermediate_size],
                &vec![hidden_size, intermediate_size]
            );
            num_hidden_layers
        ],
        w_gate: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); intermediate_size * hidden_size],
                &vec![intermediate_size, hidden_size]
            );
            num_hidden_layers
        ],
        rms_ffn_w: vec![
            Tensor::<f16>::new(
                vec![f16::from_f32(0.1); hidden_size],
                &vec![hidden_size]
            );
            num_hidden_layers
        ],
        rms_out_w: Tensor::<f16>::new(vec![f16::from_f32(0.1); hidden_size], &vec![hidden_size]),
        lm_head: Tensor::<f16>::new(
            vec![f16::from_f32(0.1); vocab_size * hidden_size],
            &vec![vocab_size, hidden_size],
        ),
    };

    let bos_token_id = 1;
    let eos_token_id = 2;

    // Define the test model with the parameters for f32
    let model_f32 = Llama {
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
        params: params_f32,
        bos_token_id: bos_token_id,
        eos_token_id: eos_token_id,
    };

    // Define the test model with the parameters for f16
    let model_f16 = Llama {
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
        params: params_f16,
        bos_token_id: bos_token_id,
        eos_token_id: eos_token_id,
    };

    // Input token sequence for both f32 and f16
    let token_ids = vec![0, 114, 514];

    // Call generate function for f32
    let generated_f32 = model_f32.generate(&token_ids, 10, 0.9, 5, 1.0);
    println!("f32 Generated Tokens: {:?}", generated_f32);

    // Call generate function for f16
    let generated_f16 =
        model_f16.generate(&token_ids, 10, f16::from_f32(0.9), 5, f16::from_f32(1.0));
    println!("f16 Generated Tokens: {:?}", generated_f16);

    // Check if both generated sequences contain the EOS token and have a reasonable length
    assert!(
        generated_f32.len() <= 10,
        "f32: Generated token length exceeds limit"
    );
    assert!(
        generated_f32.contains(&model_f32.eos_token_id),
        "f32: EOS token not present"
    );

    assert!(
        generated_f16.len() <= 10,
        "f16: Generated token length exceeds limit"
    );
    assert!(
        generated_f16.contains(&model_f16.eos_token_id),
        "f16: EOS token not present"
    );
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
    ));
}

// #[test]
// pub fn test_load_safetensors() {
//     use crate::tensor::float_eq;
//     use std::path::PathBuf;
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     let model = Llama::from_safetensors(model_dir);
//     assert_eq!(model.vocab, 2048);
//     assert_eq!(model.n_layers, 2);
//     assert_eq!(model.n_q_h, 8);
//     assert_eq!(model.n_kv_h, 4);
//     assert_eq!(model.d, 128);
//     assert_eq!(model.dqkv, 16);
//     assert_eq!(model.di, 384);

//     assert!(float_eq(
//         &model.params.embedding_table.data()[50],
//         &0.14453125,
//         1e-6
//     ));
//     assert_eq!(
//         model.params.lm_head.data()[10],
//         model.params.embedding_table.data()[10]
//     );
//     assert!(float_eq(
//         &model.params.rms_att_w[0].data()[10],
//         &0.18652344,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.rms_ffn_w[1].data()[10],
//         &0.32421875,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.rms_out_w.data()[100],
//         &0.73046875,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.w_down[0].data()[100],
//         &-0.0625,
//         1e-6
//     ));
//     assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
//     assert!(float_eq(
//         &model.params.w_gate[1].data()[100],
//         &0.296875,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.wq[1].data()[100],
//         &0.032226563,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.wk[1].data()[100],
//         &-0.21386719,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.wv[0].data()[100],
//         &0.041015625,
//         1e-6
//     ));
//     assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
// }

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
