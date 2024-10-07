use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let idx = indices.data()[i] as usize * dim;
        assert!(idx + dim <= table.size(), "Index out of bounds");
        let src = &table.data()[idx..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
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

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
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
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

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

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

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
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// // Parallelize these operators
// use rayon::prelude::*;
// // get (row) vectors from a 2D table given a list of indices
// pub fn gather_parallel(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
//     let length = indices.size();
//     let table_shape = table.shape();
//     assert!(table_shape.len() == 2);
//     let dim = table_shape[1];
//     assert!(y.size() == length * dim);

//     let table_data = table.data();
//     let indices_data = indices.data();

//     unsafe {
//         y.data_mut()
//             .par_chunks_mut(dim)
//             .enumerate()
//             .for_each(|(i, dst)| {
//                 let src = &table_data[indices_data[i] as usize * dim..][..dim];
//                 dst.copy_from_slice(src);
//             });
//     }
// }

// // RoPE: Rotary Positional Embedding
// pub fn rope_parallel(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
//     todo!();
//     let shape = y.shape();
//     assert!(shape.len() == 3);
//     let seq_len = shape[0];
//     let n_heads = shape[1];
//     let d = shape[2];
//     let data = unsafe { y.data_mut() };
//     for tok in 0..seq_len {
//         let pos = start_pos + tok;
//         for head in 0..n_heads {
//             for i in 0..d / 2 {
//                 let a = data[tok * n_heads * d + head * d + i];
//                 let b = data[tok * n_heads * d + head * d + i + d / 2];
//                 let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
//                 let (sin, cos) = freq.sin_cos();
//                 data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
//                 data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
//             }
//         }
//     }
// }

// // rms = sqrt((sum(x^2) / n) + epsilon)
// pub fn rms_norm_parallel(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
//     // Ensure the input tensors have the correct sizes
//     assert!(
//         y.size() == x.size(),
//         "Input and output tensors must have the same size"
//     );
//     assert!(w.shape().len() == 1, "Weight tensor w must be 1D");
//     assert!(
//         w.size() == x.shape().last().copied().unwrap_or(0),
//         "Weight tensor must match the last dimension of the input tensor"
//     );

//     let x_data = x.data();
//     let w_data = w.data();

//     let last_dim = x.shape().last().copied().unwrap_or(0);
//     if last_dim == 0 {
//         return;
//     }

//     // Use parallel iteration for better performance
//     unsafe {
//         y.data_mut()
//             .par_chunks_mut(last_dim)
//             .zip(x_data.par_chunks(last_dim))
//             .for_each(|(y_slice, x_slice)| {
//                 // Compute the RMS value for the current slice
//                 let rms = (x_slice.iter().map(|&val| val * val).sum::<f32>() / last_dim as f32
//                     + epsilon)
//                     .sqrt();

//                 // Normalize and apply the weights
//                 for j in 0..last_dim {
//                     y_slice[j] = w_data[j] * x_slice[j] / rms;
//                 }
//             });
//     }
// }

// // y = sigmoid(x) * x * y
// pub fn silu_parallel(y: &mut Tensor<f32>, x: &Tensor<f32>) {
//     assert!(
//         y.shape() == x.shape(),
//         "Input and output tensors must have the same shape"
//     );

//     let y_data = unsafe { y.data_mut() };
//     let x_data = x.data();

//     // Use parallel iteration for better performance
//     y_data
//         .par_iter_mut()
//         .zip(x_data.par_iter())
//         .for_each(|(y_elem, &x_elem)| {
//             let x_sigmoid = 1.0 / (1.0 + (-x_elem).exp());
//             *y_elem *= x_sigmoid * x_elem;
//         });
// }

// // C = beta * C + alpha * A @ B^T
// pub fn matmul_transb_parallel(
//     c: &mut Tensor<f32>,
//     beta: f32,
//     a: &Tensor<f32>,
//     b: &Tensor<f32>,
//     alpha: f32,
// ) {
//     todo!();
//     // Ensure the input tensors have the correct sizes
//     let (a_row, a_col) = (a.shape()[0], a.shape()[1]);
//     let (b_row, b_col) = (b.shape()[0], b.shape()[1]);
//     let (c_row, c_col) = (c.shape()[0], c.shape()[1]);

//     assert!(a_col == b_col, "Inner dimensions of A and B must match");
//     assert!(
//         a_row == c_row && b_row == c_col,
//         "Output matrix C must have shape (a_row, b_row)"
//     );

//     let a_data = a.data();
//     let b_data = b.data();
//     let c_data = unsafe { c.data_mut() };

//     // Compute alpha * A @ B^T + beta * C
//     for i in 0..c_row {
//         for j in 0..c_col {
//             let mut sum = 0.0;
//             for k in 0..a_col {
//                 sum += a_data[i * a_col + k] * b_data[j * b_col + k];
//             }
//             c_data[i * c_col + j] = beta * c_data[i * c_col + j] + alpha * sum;
//         }
//     }
// }

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
