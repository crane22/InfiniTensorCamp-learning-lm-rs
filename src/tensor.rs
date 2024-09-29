use crate::float::FloatLike;
use std::{fmt::Debug, slice, sync::Arc};

/// A struct representing a multidimensional tensor, where `data` holds the underlying array
/// and `shape` describes its dimensions. The `offset` field allows for slicing, and `length`
/// represents the number of elements in the current view of the tensor.
pub struct Tensor<T> {
    data: Arc<Box<[T]>>, // Shared, immutable data stored in an Arc to allow safe clones
    shape: Vec<usize>,   // Shape (dimensions) of the tensor
    offset: usize,       // Offset for the starting position (used in slicing)
    length: usize,       // Number of elements in the tensor
}

impl<T: Copy + Clone + Default> Tensor<T> {
    /// Creates a new tensor with the provided data and shape. Ensures that the data length matches
    /// the product of the dimensions in the shape.
    ///
    /// # Parameters:
    /// - `data`: The actual data values as a flat vector.
    /// - `shape`: The dimensions of the tensor.
    ///
    /// # Panics:
    /// - If the size of `data` doesn't match the total size implied by `shape`.
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        assert_eq!(
            length,
            shape.iter().product::<usize>(),
            "Data size does not match shape"
        );
        Tensor {
            data: Arc::new(
                data.into_boxed_slice()
                    .try_into()
                    .expect("Failed to convert Vec<T> to Box<[T]>"),
            ),
            shape: shape.clone(),
            offset: 0,
            length,
        }
    }

    /// Creates a default tensor filled with the default value of type `T` and the provided shape.
    ///
    /// # Parameters:
    /// - `shape`: The dimensions of the tensor.
    ///
    /// # Returns:
    /// - A tensor filled with default values of type `T`.
    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    /// Returns a reference to the tensor's underlying data slice.
    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    /// Returns a mutable reference to the underlying data.
    /// This function is unsafe because it circumvents Rust's usual aliasing guarantees.
    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// Returns the total number of elements in the tensor.
    pub fn size(&self) -> usize {
        self.length
    }

    /// Reshapes the tensor without changing its data. It panics if the new shape's size
    /// doesn't match the current number of elements.
    ///
    /// # Parameters:
    /// - `new_shape`: The new shape (dimensions) of the tensor.
    ///
    /// # Panics:
    /// - If the product of the new shape's dimensions doesn't equal the current number of elements.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            panic!(
                "New shape {:?} does not match tensor of length {}",
                new_shape, self.length
            );
        }
        self.shape = new_shape.clone();
        self
    }

    /// Slices the tensor to create a new tensor that references a subset of the original.
    /// The result shares the same underlying data but has a different offset and length.
    ///
    /// # Parameters:
    /// - `start`: Starting index in the flattened tensor data.
    /// - `shape`: The shape of the resulting slice.
    ///
    /// # Returns:
    /// - A new tensor representing the sliced view.
    ///
    /// # Panics:
    /// - If the requested slice exceeds the bounds of the tensor.
    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(
            self.offset + start + new_length <= self.data.len(),
            "Slice exceeds tensor bounds"
        );
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }
}

impl<T> Tensor<T>
where
    T: FloatLike + Copy + Clone + Default + std::ops::Add<Output = T>,
{
    /// Adds two tensors element-wise and returns a new tensor containing the results.
    ///
    /// # Parameters:
    /// - `other`: The other tensor to add.
    ///
    /// # Returns:
    /// - A new tensor where each element is the sum of corresponding elements from `self` and `other`.
    ///
    /// # Panics:
    /// - If the two tensors do not have the same shape.
    pub fn add(&self, other: &Tensor<T>) -> Tensor<T> {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensors must have the same shape for addition"
        );
        let result_data: Vec<T> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| *a + *b)
            .collect();

        Tensor::new(result_data, &self.shape)
    }
}

impl<T: FloatLike + Clone> Clone for Tensor<T> {
    /// Clones the tensor, creating a new instance that shares the same underlying data
    /// but can be independently modified in terms of shape and offset.
    fn clone(&self) -> Self {
        Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
            offset: self.offset,
            length: self.length,
        }
    }
}

// Helper functions for numerical comparisons and debugging.
impl<T: FloatLike + Default + Debug> Tensor<T> {
    /// Compares two tensors to check if they are approximately equal within a specified relative tolerance.
    ///
    /// # Parameters:
    /// - `other`: The other tensor to compare.
    /// - `rel`: The relative tolerance for comparison.
    ///
    /// # Returns:
    /// - `true` if all elements are approximately equal within the tolerance; otherwise, `false`.
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            println!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape());
            return false;
        }
        let a = self.data();
        let b = other.data();
        a.iter().zip(b).all(|(x, y)| x.float_eq(*y, rel))
    }

    /// Prints the tensor's shape, offset, and length. Also prints the tensor's data in batches according
    /// to the last dimension for easier visualization of multidimensional data.
    pub fn print(&self) {
        println!(
            "shape: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_new_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::<f32>::new(data.clone(), &shape);
        assert_eq!(tensor.data(), &data[..]);
        assert_eq!(tensor.shape(), &shape);
    }

    #[test]
    fn test_new_f16() {
        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let shape = vec![2, 2];
        let tensor = Tensor::<f16>::new(data.clone(), &shape);
        assert_eq!(tensor.data(), &data[..]);
        assert_eq!(tensor.shape(), &shape);
    }

    #[test]
    fn test_default_f32() {
        let shape = vec![2, 2];
        let tensor = Tensor::<f32>::default(&shape);
        assert_eq!(tensor.data(), &[0.0f32, 0.0, 0.0, 0.0]);
        assert_eq!(tensor.shape(), &shape);
    }

    #[test]
    fn test_default_f16() {
        let shape = vec![2, 2];
        let tensor = Tensor::<f16>::default(&shape);
        let zero = f16::from_f32(0.0);
        assert_eq!(tensor.data(), &[zero, zero, zero, zero]);
        assert_eq!(tensor.shape(), &shape);
    }

    #[test]
    fn test_reshape_f32() {
        let mut tensor = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &vec![2, 2]);
        tensor.reshape(&vec![4]);
        assert_eq!(tensor.shape(), &vec![4]);
    }

    #[test]
    fn test_reshape_f16() {
        let mut tensor = Tensor::<f16>::new(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            &vec![2, 2],
        );
        tensor.reshape(&vec![4]);
        assert_eq!(tensor.shape(), &vec![4]);
    }

    #[test]
    fn test_slice_f32() {
        let tensor = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &vec![2, 2]);
        let sliced = tensor.slice(1, &vec![1, 2]);
        assert_eq!(sliced.data(), &[2.0, 3.0]);
        assert_eq!(sliced.shape(), &vec![1, 2]);
    }

    #[test]
    fn test_slice_f16() {
        let tensor = Tensor::<f16>::new(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            &vec![2, 2],
        );
        let sliced = tensor.slice(2, &vec![1, 2]);
        assert_eq!(sliced.data(), &[f16::from_f32(3.0), f16::from_f32(4.0)]);
        assert_eq!(sliced.shape(), &vec![1, 2]);
    }

    #[test]
    fn test_add_f32() {
        let tensor1 = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &vec![2, 2]);
        let tensor2 = Tensor::<f32>::new(vec![5.0, 6.0, 7.0, 8.0], &vec![2, 2]);
        let result = tensor1.add(&tensor2);
        assert_eq!(result.data(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_f16() {
        let tensor1 = Tensor::<f16>::new(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            &vec![2, 2],
        );
        let tensor2 = Tensor::<f16>::new(
            vec![
                f16::from_f32(5.0),
                f16::from_f32(6.0),
                f16::from_f32(7.0),
                f16::from_f32(8.0),
            ],
            &vec![2, 2],
        );
        let result = tensor1.add(&tensor2);
        assert_eq!(
            result.data(),
            &[
                f16::from_f32(6.0),
                f16::from_f32(8.0),
                f16::from_f32(10.0),
                f16::from_f32(12.0)
            ]
        );
    }

    #[test]
    fn test_close_to_f32() {
        let tensor1 = Tensor::<f32>::new(vec![1.0001, 2.0001, 3.0001, 4.0001], &vec![2, 2]);
        let tensor2 = Tensor::<f32>::new(vec![1.0002, 2.0002, 3.0002, 4.0002], &vec![2, 2]);
        assert!(tensor1.close_to(&tensor2, 0.001));
    }

    #[test]
    fn test_close_to_f16() {
        let tensor1 = Tensor::<f16>::new(
            vec![
                f16::from_f32(1.0001),
                f16::from_f32(2.0001),
                f16::from_f32(3.0001),
                f16::from_f32(4.0001),
            ],
            &vec![2, 2],
        );
        let tensor2 = Tensor::<f16>::new(
            vec![
                f16::from_f32(1.0002),
                f16::from_f32(2.0002),
                f16::from_f32(3.0002),
                f16::from_f32(4.0002),
            ],
            &vec![2, 2],
        );
        assert!(tensor1.close_to(&tensor2, 0.001));
    }

    #[test]
    fn test_print_f32() {
        let tensor = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &vec![2, 2]);
        tensor.print(); // Visually check the output in the test logs
    }

    #[test]
    fn test_print_f16() {
        let tensor = Tensor::<f16>::new(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            &vec![2, 2],
        );
        tensor.print(); // Visually check the output in the test logs
    }
}
