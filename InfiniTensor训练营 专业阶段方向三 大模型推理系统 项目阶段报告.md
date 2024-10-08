北京大学 郭曜玮
项目地址：https://github.com/crane22/InfiniTensorCamp-learning-lm-rs
# 1. 作业阶段
## 1.1 算子：SiLU 函数
推导：
	算子需求来源于Feed-Forward Network中的计算：
$$\text{FFN}_\text{SwiGLU}(X, W^G, W^{\uparrow}, W^{\downarrow}) = \text{SwiGLU}(X, W^G, W^{\uparrow}, \vec{0}, \vec{0}, 1)W^{\downarrow}$$
	其中：
$$\text{SwiGLU}(\vec{x}, W, V, \vec{b}, \vec{c}, \beta) = \text{Swish}_{\beta}(\vec{x}W+\vec{b}) \odot (\vec{x}V+\vec{c})$$
	（$\odot$表示Hadamard积, 亦即逐元素乘积）
	对于SwiGLU的第一部分：
	$$\text{Swish}_{\beta}(\vec{x}) = \vec{x} \odot \sigma(\beta\vec{x})$$
	其中$\sigma$表示Sigmoid函数, $\sigma(x) = \frac{1}{1+e^{-x}}$. 取$\beta = 1$, Swish函数就是SiLU函数：
	$$\text{SiLU}(\vec{x}) = \vec{x} \odot \sigma(\vec{x})$$
	取$\vec{b}=\vec{c}=\vec{0}$，考虑到SwiGLU函数的计算过程$\text{SwiGLU}(X, {W^G}, W^{\uparrow}) = \text{SiLU}(XW^G) \odot (XW^{\uparrow})$, 将实际代码中的SiLU函数写为以下版本：
$$\text{SiLU}(XW^G,XW^{\uparrow}) = (XW^G) \odot \sigma(XW^G) \odot XW^{\uparrow}$$
实现：
```Rust
// y = sigmoid(x) * x * y
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    assert!(
        y.shape() == x.shape(),
        "Input and output tensors must have the same shape"
    );
    let length = y.size();
    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();
    for i in 0..length {
        let x_sigmoid = 1.0 / (1.0 + (-x_data[i]).exp());
        y_data[i] *= x_sigmoid * x_data[i];
    }
}
```
说明：
	最简单的算子之一，保证输入的张量X和Y形状相同即可

## 1.2 算子：RMS Normalization
推导：
	对行向量$\vec{x_i}^T$和$\vec{w_{\text{RMS}}}^T$：
$$\text{RMSNorm}_{\vec{w_{\text{RMS}}}^T}(\vec{x_i}^T) = \frac{\vec{x_i}^T}{\text{RMS}(\vec{x_i})} \odot {\vec{w_{\text{RMS}}}^T} = \frac{[x_{i, 1}w_1, x_{i, 2}w_2, \ldots, x_{i, d_{model}}w_{d_{model}}]}{\sqrt{\frac{1}{d_{model}} \sum_{j = 1}^{d_{model}} x_{i, j}^2}}$$
	其中$d_{model}$为超参数`hidden_size`.
	实际运算中对矩阵$X$，可以将其分割成数个行向量的slice分别运算，最后进行拼接
实现：
```Rust
// rms = sqrt((sum(x^2) / n) + epsilon)
// y = x * w / rms
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // Ensure the input tensors have the correct sizes
    assert!(
        y.shape() == x.shape(),
        "Input and output tensors must have the same shape"
    );
    assert!(w.shape().len() == 1, "Weight tensor w must be 1D");
    let x_last_dim = x.shape().last().copied().unwrap_or(0);
    if x_last_dim == 0 {
        return;
    }
    assert!(
        w.size() == x_last_dim,
        "Weight tensor must match the last dimension of the input tensor"
    );

    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();
    let w_data = w.data();

    let batch = x.size() / x_last_dim;

    for i in 0..batch {
        let left = i * x_last_dim;
        let right = left + x_last_dim;
        let x_slice = &x_data[left..right];
        let y_slice = &mut y_data[left..right];

        // Compute the RMS value for the current slice
        let rms = (x_slice.iter().map(|&val| val * val).sum::<f32>() / x_last_dim as f32 + epsilon)
            .sqrt();

        // Normalize and apply the weights
        for j in 0..x_last_dim {
            y_slice[j] = w_data[j] * x_slice[j] / rms;
        }
    }
}
```
说明：
	注意边界条件，判断张量$Y$, $X$, $W$是否满足要求；其余按要求计算即可

## 1.3 算子：矩阵乘法
说明：
	转置B矩阵的矩阵乘法
$$C \leftarrow \alpha AB^T + \beta C$$
实现：
```Rust
// C = beta * C + alpha * A @ B^T
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // Ensure the input tensors have the correct sizes
    let (a_row, a_col) = (a.shape()[0], a.shape()[1]);
    let (b_row, b_col) = (b.shape()[0], b.shape()[1]);
    let (c_row, c_col) = (c.shape()[0], c.shape()[1]);

    assert!(a_col == b_col, "Inner dimensions of A and B must match");
    assert!(
        a_row == c_row && b_row == c_col,
        "Output matrix C must have shape (a_row, b_row)"
    );

    let c_data = unsafe { c.data_mut() };
    let a_data = a.data();
    let b_data = b.data();

    // Compute alpha * A @ B^T + beta * C
    for i in 0..c_row {
        for j in 0..c_col {
            let mut sum = 0.0;
            for k in 0..a_col {
                sum += a_data[i * a_col + k] * b_data[j * b_col + k];
            }
            c_data[i * c_col + j] = beta * c_data[i * c_col + j] + alpha * sum;
        }
    }
}
```
说明：
	很基本的实现，因为太基本了反而没什么好说的；存在很多优化空间

## 1.4 模型结构：Feed-Forward神经网络
推导：
	对前级输出的张量$X_{residual}$归一化之后作为本级的输入$X$，归一化函数`RMSNorm`详见`1.2`节
$$X = \text{RMSNorm}_{\vec{w_{\text{RMSFFN}}}^T}(X_{residual})$$
	其中$\vec{w_{\text{RMSFFN}}}$是模型参数`model.layers.*.post_attention_layernorm.weight`
	网络结构推导详见`1.1`节
$$\text{FFN}_\text{SwiGLU}(X, W^G, W^{\uparrow}, W^{\downarrow}) = \text{SiLU}(XW^G,XW^{\uparrow}) W^{\downarrow}$$
实现：
```Rust
// MLP Layer
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
    // Apply RMS normalization to the residual tensor and store the result in hidden_states
    OP::rms_norm(hidden_states, &residual, &rms_w, eps);
    // Calculate gate = hidden @ w_gate.T
    OP::matmul_transb(gate, 0.0, &hidden_states, &w_gate, 1.0);
    // Calculate up = hidden @ w_up.T
    OP::matmul_transb(up, 0.0, &hidden_states, &w_up, 1.0);
    // Apply SiLU activation to gate and up and store the result in up
    OP::silu(up, &gate);
    // Calculate output = gate @ w_down.T and Add the output to residual (residual connection)
    OP::matmul_transb(residual, 1.0, &up, &w_down, 1.0);
}
```
说明：
	按要求实现即可。在最后实现残差连接功能时，利用了`1.3`节矩阵乘法的$C \leftarrow \alpha AB^T + \beta C$，令$\beta = 1$ 计算最终结果的同时，将计算结果加到残差张量中。

## 1.5 LLaMA模型参数加载
实现：
```Rust
use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::Dtype;
use safetensors::SafeTensors;

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

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            let tensor_view = safetensor
                .tensor(name)
                .unwrap_or_else(|_| panic!("TensorNotFound({})", name));
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
                        .collect();
                    let shape = tensor_view.shape().to_vec();
                    Tensor::<f32>::new(data, &shape)
                }
                _ => panic!("Unsupported dtype: {:?}", dtype),
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

```
说明：
	首先读取`model.safetensors`，依照`story`模型和`chat`模型各自的实际情况手动提取了张量名称。
	预先获取了相应张量的数据类型，利用数据类型和向量形状创建向量并分配内存空间；调用`safetensors`库，通过张量名称分别读取张量，注意了一些边界情况；并为后续实现半精度做准备。同时，对`tie_word_embeddings`分别为`true`和`false`时做了不同调整。
# 2. 项目阶段
## 2.1 模型结构：Self-Attention
推导：
	LLaMA模型使用的注意力为Grouped Query Attention (GQA)。假定总注意力头数为$H$，对注意力头$h$，如果不考虑KVCache，可以得到每个头的注意力分数$A_h$为：
$$A_h = \text{masked\_softmax}(\frac{Q_h K_g^T}{\sqrt{d_{head}}})$$
	其中，注意力头$h$所在的组$g$可由$g = \lceil \frac{h}{G} \rceil$ （$G$为总组数，亦即key-value头的总数）得到。
	而每个头的注意力输出$Z_h$为：
$$Z_h=A_h V_g = \text{masked\_softmax}(\frac{Q_h K_g^T}{\sqrt{d_{head}}}) V_g$$
	此时，$Q_h$，$K_g$和$V_g$的维度均为$[seq\_len, d_{head}]$，$A_h$的维度为$[seq\_len, seq\_len]$，$Z_h$的维度为$[seq\_len, d_{head}]$。
	首先，考虑到使用了KVCache，事实上$K$和$V$矩阵在计算投影（与$K$计算位置编码）之后，与KVCache中之前的数据进行了拼接，所以$K_g$和$V_g$的实际维度均为$[total\_seq\_len, d_{head}]$，计算注意力分数得到的矩阵$A_h$的维度为$[seq\_len, total\_seq\_len]$，$Z_h$的维度仍为$[seq\_len, d_{head}]$。
	其次，考虑GQA使得$Q$，$K$，和$V$矩阵分别是由$H$个$Q_h$，$G$个$K_g$和$G$个$V_g$矩阵分别拼接而成，所以输入Self-Attention结构的$Q$，$K$，和$V$矩阵的维度分别为$[seq\_len, H \times d_{head}]$，$[total\_seq\_len, G \times d_{head}]$和$[total\_seq\_len, G \times d_{head}]$。对每个头$h$拼接得到的最终注意力输出$Z$的维度为$[seq\_len, H \times d_{head}]$。
实现：
```Rust
fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let hidden_states_data = unsafe { hidden_states.data_mut() };
    // let att_scores_data = unsafe { att_scores.data_mut() };
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();

    let d_head = dqkv;
    let num_key_value_heads = n_kv_h;
    let num_attention_heads = n_kv_h * n_groups;
    let num_query_heads_per_kv_group = n_groups; // n_groups = num_attention_heads / num_key_value_heads
    let total_d_q = num_attention_heads * d_head;
    let total_d_kv = num_key_value_heads * d_head;
    let total_d_atts_3 = num_query_heads_per_kv_group * seq_len * total_seq_len;
    let total_d_atts_2 = seq_len * total_seq_len;
    let total_d_atts_1 = total_seq_len;

    // Calculate attention scores for every head h (score = Q @ K.T / sqrt(dim))
    for curr_k_head in 0..num_key_value_heads {
        let offset_matrix_k_g = curr_k_head * d_head;
        for curr_q_in_group in 0..num_query_heads_per_kv_group {
            let curr_att_head = curr_k_head * num_query_heads_per_kv_group + curr_q_in_group;
            let offset_matrix_q_h = curr_att_head * d_head;
            for curr_idx_seq in 0..seq_len {
                let begin_vec_q = curr_idx_seq * total_d_q + offset_matrix_q_h;
                for curr_idx_total_seq in 0..total_seq_len {
                    let begin_vec_k = curr_idx_total_seq * total_d_kv + offset_matrix_k_g;

                    let vec_q = &q_data[begin_vec_q..][..d_head]; // [begin_vec_q:begin_vec_q+d_head] in python
                    let vec_k = &k_data[begin_vec_k..][..d_head];
                    // Calculate the dot product between Q and K, then normalize by sqrt(d_head)
                    let score = vec_q.iter().zip(vec_k).map(|(&q, &k)| q * k).sum::<f32>()
                        / (d_head as f32).sqrt();
                    // Calculate the index for attention scores and assign the score
                    let curr_idx_attscore = curr_k_head * total_d_atts_3
                        + curr_q_in_group * total_d_atts_2
                        + curr_idx_seq * total_d_atts_1
                        + curr_idx_total_seq;
                    unsafe {
                        att_scores.data_mut()[curr_idx_attscore] = score;
                    }
                }
            }
        }
    }

    // Apply masked softmax to the attention scores
    OP::masked_softmax(att_scores);

    let att_scores_data = att_scores.data();
    let total_d_z_3 = num_key_value_heads * num_query_heads_per_kv_group * d_head;
    let total_d_z_2 = num_query_heads_per_kv_group * d_head;
    let total_d_z_1 = d_head;

    // Compute the output hidden states by applying attention scores to the values (V)
    for curr_v_head in 0..num_key_value_heads {
        let offset_matrix_v_g = curr_v_head * d_head;
        for curr_q_in_group in 0..num_query_heads_per_kv_group {
            let offset_matrix_a_h = curr_q_in_group * total_d_atts_2 + curr_v_head * total_d_atts_3;
            for curr_idx_seq in 0..seq_len {
                let begin_vec_a = curr_idx_seq * total_d_atts_1 + offset_matrix_a_h;
                for curr_idx_dhead in 0..d_head {
                    let begin_vec_v = curr_idx_dhead + offset_matrix_v_g;
                    let mut sum = 0.0f32;
                    for curr_idx_total_seq in 0..total_seq_len {
                        let curr_idx_vec_a = begin_vec_a + curr_idx_total_seq;
                        let curr_idx_vec_v = begin_vec_v + curr_idx_total_seq * total_d_kv;
                        sum += att_scores_data[curr_idx_vec_a] * v_data[curr_idx_vec_v];
                    }
                    let curr_idx_hidden = curr_idx_seq * total_d_z_3
                        + curr_v_head * total_d_z_2
                        + curr_q_in_group * total_d_z_1
                        + curr_idx_dhead;
                    hidden_states_data[curr_idx_hidden] = sum;
                }
            }
        }
    }
}
```
## 2.2 功能：文本生成
推导：
	完善`forward`函数`for`循环中每层的Attention Block，首先对前级输出的张量$X_{residual}$归一化之后作为本级的输入$X$，归一化函数`RMSNorm`详见`1.2`节
$$X = \text{RMSNorm}_{\vec{w_{\text{RMSAtt}}}^T}(X_{residual})$$
	其中$\vec{w_{\text{RMSAtt}}}$是模型参数`model.layers.*.input_layernorm.weight`
	其次，为$Q$，$K$，和$V$矩阵分配空间，并完成$Q$，$K$，和$V$矩阵的投影计算，同时使用RoPE对$Q$和$K$矩阵进行位置编码，得到最终`Self-Attention`所需的$Q_{buf}^{rot}$，$K_{cache}^{rot}$，和$V_{cache}$
	使残差矩阵通过`Self-Attention`层，经过特征融合矩阵$W^O$后得到新的输入矩阵。再将其输入到`Feed-Forward Network`之后，完成一层的运算
实现：
```Rust
    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
    // 前略
    for layer in 0..self.n_layers {
    // 前略
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

            OP::matmul_transb(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
            );

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
	    let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![1, self.d]); //Here
	// 后略
    }
```
说明：
	`forward`函数的设计在`推导`中已详细说明。但是最后三行`logits`, `hidden_states`和`residual`的维度在设计过程中，`residual`的维度缺少一个高维（hidden_states: [1, 128], residual: [128]），我也没有实现广播`RMSNorm`，导致输出维度中`hidden_states`和`residual`维度不匹配，按照这种做法已经修复。
```Rust
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
        let mut input = Tensor::<u32>::new(result.clone(), &vec![result.len()]);

        while result.len() < max_len {
            let logits = self.forward(&input, &mut cache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }
            input = Tensor::<u32>::new(vec![next_token], &vec![1]);
        }

        result
    }
```
说明：
	用了最简单的办法去逐个将输出的结果输入，形成自回归（直到遇到`eos_token`视为停止）。
`story`模型输出结果为：
```Text
Once upon a time
Once upon a time, a little girl named Lily found a big, shiny palm on the ground. She picked it up and showed it to her mommy. She was very happy and played with her palm every day.
One day, Lily found a big, shiny pedal. She was so excited to show her friends. She took the pedal to show it. Lily was very happy. She wanted to show her mommy how to show it, but she didn't want to share her pedal.
Lily went to her mom and said, "Mom, can I have this pedal." Her mom smiled and said, "That's great, Lily! Let's share it together."
They played with the pedals and had lots of fun. They had lots of fun. The pink flower was gone, and the pink flower became even more pink pedal. They had lots of fun, and Lily learned that sharing can make her feel better than being mean.<|end_story|>
```

## 2.3 功能：命令行AI多轮对话
实现：
```Rust
struct Message {
    role: String,
    content: String,
}

impl Message {
    fn format(&self) -> String {
        format!("<|im_start|>{}{}<|im_end|>", self.role, self.content)
    }
}
```
说明：
	依照Jinja格式定义了消息模板。
实现：
```Rust
/// model.rs
impl Llama<f32> {
	// 前略
    pub fn streaming_generate<'a>(
        &'a self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kvcache: &'a mut KVCache<f32>,
    ) -> impl Iterator<Item = u32> + 'a {
        let mut result_tokens = token_ids.to_vec();
        let mut input_tensors =
            Tensor::<u32>::new(result_tokens.clone(), &vec![result_tokens.len()]);

        std::iter::from_fn(move || {
            if result_tokens.len() >= max_len {
                return None;
            }

            let logits = self.forward(&input_tensors, kvcache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result_tokens.push(next_token);
            input_tensors = Tensor::<u32>::new(vec![next_token], &vec![1]);

            if next_token == self.eos_token_id {
                None
            } else {
                Some(next_token)
            }
        })
    }
}
```
说明：
	在`model.rs`中添加`streaming_generate`函数实现了流式生成 token 的功能。使用 `std::iter::from_fn`创建一个迭代器，每次迭代生成一个新 token。每次调用迭代器时，首先检查当前生成的 token 数量是否超过了 `max_len`，如果超过了则停止生成。执行前向计算得到生成下一个 token 的 `logits`后，采样下一个 token。将生成的 token加入 `result_tokens` 中，更新 `input_tensors` 以便用于下一次生成。
实现：
```Rust
/// main.rs
fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    chat(&llama, &tokenizer);
}

fn chat(llama: &model::Llama<f32>, tokenizer: &Tokenizer) {
    let mut kvcache = llama.new_cache();
    let mut messages: Vec<Message> = vec![];

    loop {
        // Get user input
        print!("User: ");
        io::stdout().flush().unwrap();
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();
        if user_input.eq_ignore_ascii_case("exit") {
            break;
        }

        // Append user message to conversation history
        messages.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        // Format the input using Jinja-like template for the model
        let conversation_input: String =
            messages.iter().map(|msg| msg.format()).collect::<String>() + "<|im_start|>assistant";

        let binding = tokenizer.encode(conversation_input, true).unwrap();
        let input_ids = binding.get_ids();

        // Stream the model's response
        print!("Assistant: ");
        io::stdout().flush().unwrap();

        let response_tokens = llama.streaming_generate(input_ids, 512, 0.9, 4, 1.0, &mut kvcache);
        for token in response_tokens {
            let word = tokenizer.decode(&[token], true).unwrap() + " ";
            print!("{}", word);
            io::stdout().flush().unwrap();
        }

        println!();

        // Append assistant message to conversation history
        messages.push(Message {
            role: "assistant".to_string(),
            content: "".to_string(), // We'll update this after generating the response
        });
    }
}
```
说明：
	`chat` 函数实现了一个交互式的命令行应用，允许用户与 AI 模型进行多轮对话。`chat` 函数是一个基于命令行的对话循环，允许用户输入文本并与模型进行多轮对话，通过维护一段对话的上下文，使得模型能够理解用户输入的历史内容并生成上下文相关的回应。
`chat`模型输出结果为：
```Text
User: Tell me a story of a cat Lily.
Assistant: 
 It was a beautiful summer evening when the sun was shining on the horizon , and I could feel myself getting closer to the moon light , and my eyes were filled with a we . I was walking home from work , my heart pound ing in my chest . 
 
 I could feel the warmth of the sun in my eyes , and I could feel the warmth of the sun on my face . I could feel the warmth and warmth of the sun on my skin , and I could feel the warmth and warmth of the sun on my face . 
 
 As I made my way back to the sun set , the sun was setting , and I could feel the sun beating down on my face . It was as if the sun was shining in my chest , and I could feel it in my chest . 
 
 As I walked , I could feel the warmth of the sun on my face as I made my way to the moon light . I could feel the warmth of the sun on my face , and I could feel the warm sun on my face as I looked up at the sky . I could feel the warmth of the sun on my face , and I could feel the warmth of the sun on my face . 
 
 As I walked , my heart pound ing in my chest , and I knew that I was there to share with me . I had found a way to make a difference in myself , and I was ready to take on the world . 
User: What does the fox say?
Assistant:  assistant 
 The f ox says that the f ox thinks it is not a bad idea . They say , " The f ox says that the f ox says they have been in a bad relationship with the dog . The dog is a good friend , but they say they want to give him a good deal ." They say that the f ox believes that they have been a good friend and that they can make the most of the f ox . They say that the f ox is a good friend and a good friend who loves to talk about their own life . They say they want to give the f ox a good deal . They say the f ox is the f ox saying that it is not a bad thing , but it is not a good friend . They say that the f ox believes that the f ox thinks they have a good relationship with the dog and that the f ox is the f ox . They say that the f ox believes that the f ox believes that the f ox is a good friend and that the f ox believes that the dog is a good friend . 
User: exit
 *  Terminal will be reused by tasks, press any key to close it.
```
## 2.4 功能：FP16半精度推理
说明：
	使用`half::f16`和`num_traits::Float`两个库，修改基本上所有的文件以支持泛型`<T: Float>`。代码太多粘不过来了，请老师自己看吧。

## 2.5 功能：CPU分布式推理
说明：
	来不及做了，之后在`GitHub`上更新吧。