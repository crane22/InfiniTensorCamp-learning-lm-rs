use half::f16;
use num_traits::Float;
use std::{slice, sync::Arc, vec};

pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        assert_eq!(
            length,
            shape.iter().product::<usize>(),
            "Data size does not match shape"
        );
        Tensor {
            data: Arc::new(data.into_boxed_slice()),
            shape: shape.clone(),
            offset: 0,
            length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

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

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(start < self.length, "Start index is out of bounds");
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

impl<T: Float + Copy + Clone + Default + std::fmt::Debug> Tensor<T> {
    pub fn close_to(&self, other: &Self, rel: T) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();

        a.iter().zip(b).all(|(x, y)| float_eq(*x, *y, rel))
    }

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

impl Tensor<f16> {
    pub fn float_to_f16(input: &Tensor<f32>) -> Self {
        let data = input
            .data()
            .iter()
            .map(|&val| f16::from_f32(val))
            .collect::<Vec<_>>();
        Tensor::new(data, &input.shape().clone())
    }

    pub fn f16_to_float(&self) -> Tensor<f32> {
        let data = self
            .data()
            .iter()
            .map(|&val| val.to_f32())
            .collect::<Vec<_>>();
        Tensor::new(data, &self.shape().clone())
    }
}

impl<T: Float + Clone> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
            offset: self.offset,
            length: self.length,
        }
    }
}

#[inline]
pub fn float_eq<T: Float>(x: T, y: T, rel: T) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / T::from(2.0).unwrap()
}
