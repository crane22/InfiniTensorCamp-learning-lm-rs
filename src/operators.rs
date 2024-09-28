use crate::tensor::Tensor;

/// Gather function to extract (row) vectors from a 2D table based on given indices.
/// # Parameters:
/// - `y`: Output tensor where the gathered rows are stored.
/// - `indices`: A tensor containing the indices of the rows to gather.
/// - `table`: A 2D tensor from which rows are gathered.
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert_eq!(table_shape.len(), 2, "Table must be 2D");
    let dim = table_shape[1];
    assert_eq!(y.size(), length * dim, "Output tensor y size mismatch");
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

/// Rotary Positional Embedding (RoPE) function to apply position-based embeddings.
/// # Parameters:
/// - `y`: The tensor to which RoPE is applied.
/// - `start_pos`: The starting position for rotary embedding.
/// - `theta`: A scaling factor for position frequency.
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert_eq!(shape.len(), 3, "RoPE input must be 3D");
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

/// Masked softmax function to apply softmax with masking.
/// Softmax(x) = exp(x - max) / sum(exp(x - max))
/// # Parameters:
/// - `y`: The tensor on which softmax is applied, with masking.
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2, "Tensor must be at least 2D");
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let sum: f32 = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

/// Root Mean Square (RMS) normalization function.
/// # Parameters:
/// - `y`: Output tensor after applying normalization.
/// - `x`: Input tensor to normalize.
/// - `w`: Weight tensor for normalization.
/// - `epsilon`: Small value to ensure numerical stability.
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    assert_eq!(y.shape(), x.shape(), "Shape mismatch between y and x");
    assert_eq!(w.shape().len(), 1, "Weight tensor w must be 1D");
    assert_eq!(
        w.shape()[0],
        *x.shape().last().unwrap(),
        "Shape mismatch in w"
    );

    let x_data = x.data();
    let w_data = w.data();
    let y_data: &mut [f32] = unsafe { y.data_mut() };

    let length = w.size();
    let batch = x.size() / length;

    for i in 0..batch {
        let left = i * length;
        let x_batch = &x_data[left..left + length];
        let sum_of_x_sq: f32 = x_batch.iter().map(|&x_value| x_value * x_value).sum();
        let rms = (sum_of_x_sq / (length as f32) + epsilon).sqrt();
        for j in 0..length {
            y_data[left + j] = x_batch[j] * w_data[j] / rms;
        }
    }
}

/// SiLU (Sigmoid-Weighted Linear Unit) activation function.
/// Applies the element-wise operation: y = sigmoid(x) * x * y
/// # Parameters:
/// - `y`: Output tensor after applying SiLU.
/// - `x`: Input tensor.
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let length = y.size();
    assert_eq!(length, x.size(), "Shape mismatch between y and x");

    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();

    for i in 0..length {
        let x_value = x_data[i];
        let x_sigmoid = 1.0 / (1.0 + (-x_value).exp());
        y_data[i] *= x_sigmoid * x_value;
    }
}

/// Matrix multiplication with the second matrix transposed.
/// C = beta * C + alpha * A @ B^T
/// # Parameters:
/// - `c`: Output tensor.
/// - `beta`: Scalar multiplier for C.
/// - `a`: Input tensor A.
/// - `b`: Input tensor B (transposed).
/// - `alpha`: Scalar multiplier for the product A @ B^T.
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let a_row = a.shape()[0];
    let a_col = a.shape()[1];
    let b_row = b.shape()[0];
    let b_col = b.shape()[1];
    let c_row = c.shape()[0];
    let c_col = c.shape()[1];

    assert_eq!(a_row, c_row, "Shape mismatch between A and C rows");
    assert_eq!(b_row, c_col, "Shape mismatch between B and C columns");
    assert_eq!(a_col, b_col, "Shape mismatch between A and B");

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

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

/// Struct to represent a probability associated with a token
#[derive(Clone, Copy, PartialEq, Debug)]
struct Probability {
    val: f32,
    tok: u32,
}

impl Eq for Probability {}

impl PartialOrd for Probability {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Probability {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.val
            .partial_cmp(&other.val)
            .unwrap()
            .reverse()
            .then_with(|| self.tok.cmp(&other.tok))
    }
}

impl From<(usize, &f32)> for Probability {
    #[inline]
    fn from((i, p): (usize, &f32)) -> Self {
        Self {
            val: *p,
            tok: i as u32,
        }
    }
}

/// Random sampling from a probability vector.
/// # Parameters:
/// - `x`: Input tensor representing a probability distribution.
/// - `top_p`: Top-p sampling threshold.
/// - `top_k`: Top-k sampling threshold.
/// - `temperature`: Temperature to control randomness.
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();

    if temperature <= 0.0 || top_k < 2 || top_p <= 0.0 {
        return logits[0].tok;
    }

    let max_logit = logits[0].val;
    let sum_exp: f32 = logits.iter_mut().map(|p| (p.val - max_logit).exp()).sum();

    let mut accum = 0.0;
    let rand_thresh = rand::random::<f32>() * sum_exp;

    for prob in &logits {
        accum += prob.val.exp();
        if accum >= rand_thresh {
            return prob.tok;
        }
    }

    logits[0].tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
