pub mod device;
mod node;
mod tensor;

use std::borrow::Borrow;

use crate::Shape;

pub trait BasicFunctions
where
    Self: Sized,
{
    // core

    fn shape(&self) -> Shape;
    fn to_vec(&self) -> Vec<f32>;
    fn to_float(&self) -> f32;

    // utility

    fn argmax(&self, dim: u32) -> Vec<u32>;
    fn argmin(&self, dim: u32) -> Vec<u32>;
    fn argsort(&self, dim: u32) -> Vec<u32>;

    // basic

    //fn pow<T: Borrow<Self>>(&self, k: T) -> Self;
    //fn powf(&self, k: u32) -> Self;
    //fn powi(&self, k: i32) -> Self;
    fn sqrt(&self) -> Self;
    fn abs(&self) -> Self;

    // trigonometric

    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;

    // exp

    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn tanh(&self) -> Self;
    fn sigmoid(&self) -> Self;
    fn softplus(&self) -> Self;

    // reduction

    fn sum(&self, dim: u32) -> Self;
    fn max(&self, dim: u32) -> Self;
    fn min(&self, dim: u32) -> Self;
    fn broadcast(&self, dim: u32, size: u32) -> Self;
    fn logsumexp(&self, dim: u32) -> Self;
    fn ln_softmax(&self, dim: u32) -> Self;
    fn softmax(&self, dim: u32) -> Self;
    fn softmax_cross_entropy<T: Borrow<Self>>(&self, t: T, dim: u32) -> Self;
    fn sparse_softmax_cross_entropy(&self, ids: &[u32], dim: u32) -> Self;

    // matrix

    fn matmul<T: Borrow<Self>>(&self, rhs: T) -> Self;
    fn transpose(&self) -> Self;
    //fn permute_dims(&self, perm: &[u32]) -> Self;
    //fn flip(&self, dim: u32) -> Self;
    fn triangular_l(&self, k: u32) -> Self;
    fn triangular_u(&self, k: u32) -> Self;

    // ramp

    fn relu(&self) -> Self;
    fn lrelu(&self) -> Self;
    fn prelu(&self, a: f32) -> Self;
    fn elu(&self, a: f32) -> Self;

    // manipulation

    fn slice(&self, dim: u32, lower: u32, upper: u32) -> Self;
    fn split(&self, dim: u32, n: u32) -> Vec<Self>;
    fn pick(&self, ids: &[u32], dim: u32) -> Self;
    fn concat(xs: &[&Self], dim: u32) -> Self;
    fn reshape(&self, shape: Shape) -> Self;
    fn flatten(&self) -> Self;

    // batch

    fn batch_sum(&self) -> Self;
    fn batch_slice(&self, lower: u32, upper: u32) -> Self;
    fn batch_split(&self, n: u32) -> Vec<Self>;
    fn batch_pick(&self, ids: &[u32]) -> Self;
    fn batch_concat(xs: &[&Self]) -> Self;

    // others

    //fn reshape(&self, shape: Shape) -> Self;
    //fn flatten(&self) -> Self;

    //fn stop_gradient(&self, dim: u32) -> Self;

    // convolution

    // fn conv2d<T: Borrow<Self>>(&self, w: T, padding0: u32, padding1: u32, stride0: u32, stride1: u32, dilation0: u32, dilation1: u32) -> Self;
    // fn max_pool2d<T: Borrow<Self>>(&self, w: T, padding0: u32, padding1: u32, stride0: u32, stride1: u32, dilation0: u32, dilation1: u32) -> Self;
}
