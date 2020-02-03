use std::borrow::Borrow;

use crate::functions::{BasicDeviceFunctions, BasicFunctions};
use crate::{Shape, Tensor};

impl<'arg, 'dev> BasicFunctions for Tensor<'dev> {
    // core

    fn shape(&self) -> Shape {
        self.shape
    }

    fn to_vec(&self) -> Vec<f32> {
        self.device().tensor_to_vector(self)
    }

    fn to_float(&self) -> f32 {
        self.device().tensor_to_float(self)
    }

    // utility

    fn argmax(&self, dim: u32) -> Vec<u32> {
        self.device().argmax(self, dim)
    }

    fn argmin(&self, dim: u32) -> Vec<u32> {
        self.device().argmin(self, dim)
    }

    fn argsort(&self, dim: u32) -> Vec<u32> {
        self.device().argsort(self, dim)
    }

    // basic

    fn pow<T: Borrow<Self>>(&self, k: T) -> Self {
        self.device().powf_fw(self, k.borrow())
    }

    fn powf(&self, k: f32) -> Self {
        self.device().powf_const_r_fw(self, k)
    }

    fn powi(&self, k: i32) -> Self {
        self.device().powi_fw(self, k)
    }

    fn sqrt(&self) -> Self {
        self.device().sqrt_fw(self)
    }

    fn abs(&self) -> Self {
        self.device().abs_fw(self)
    }

    // trigonometric

    fn sin(&self) -> Self {
        self.device().sin_fw(self)
    }

    fn cos(&self) -> Self {
        self.device().cos_fw(self)
    }

    fn tan(&self) -> Self {
        self.device().tan_fw(self)
    }

    // exp

    fn exp(&self) -> Self {
        self.device().exp_fw(self)
    }

    fn ln(&self) -> Self {
        self.device().ln_fw(self)
    }

    fn tanh(&self) -> Self {
        self.device().tanh_fw(self)
    }

    fn sigmoid(&self) -> Self {
        self.device().sigmoid_fw(self)
    }

    fn softplus(&self) -> Self {
        self.device().softplus_fw(self)
    }

    // reduction

    fn sum(&self, dim: u32) -> Self {
        self.device().sum_fw(self, dim)
    }

    fn max(&self, dim: u32) -> Self {
        self.device().max_fw(self, dim)
    }

    fn min(&self, dim: u32) -> Self {
        self.device().min_fw(self, dim)
    }

    fn broadcast(&self, dim: u32, size: u32) -> Self {
        self.device().broadcast_fw(self, dim, size)
    }

    fn logsumexp(&self, dim: u32) -> Self {
        self.device().logsumexp_fw(self, dim)
    }

    fn ln_softmax(&self, dim: u32) -> Self {
        self - self.logsumexp(dim).broadcast(dim, self.shape()[dim])
    }

    fn softmax(&self, dim: u32) -> Self {
        self.ln_softmax(dim).exp()
    }

    fn softmax_cross_entropy<T: Borrow<Self>>(&self, t: T, dim: u32) -> Self {
        -(t.borrow() * self.ln_softmax(dim)).sum(dim)
    }

    fn sparse_softmax_cross_entropy(&self, ids: &[u32], dim: u32) -> Self {
        -self.ln_softmax(dim).pick(ids, dim)
    }

    // matrix

    fn matmul<T: Borrow<Self>>(&self, rhs: T) -> Self {
        self.device().matmul_fw(self, rhs.borrow())
    }

    fn transpose(&self) -> Self {
        self.device().transpose_fw(self)
    }

    fn triangular_l(&self, k: u32) -> Self {
        self.device().triangular_l_fw(self, k)
    }

    fn triangular_u(&self, k: u32) -> Self {
        self.device().triangular_u_fw(self, k)
    }

    // ramp

    fn relu(&self) -> Self {
        self.device().prelu_fw(self, 0.)
    }

    fn lrelu(&self) -> Self {
        self.device().prelu_fw(self, 0.01)
    }

    fn prelu(&self, a: f32) -> Self {
        self.device().prelu_fw(self, a)
    }

    fn elu(&self, a: f32) -> Self {
        self.device().elu_fw(self, a)
    }

    // manipulation

    fn slice(&self, dim: u32, lower: u32, upper: u32) -> Self {
        self.device().slice_fw(self, dim, lower, upper)
    }

    fn split(&self, dim: u32, n: u32) -> Vec<Self> {
        let mut ret = vec![];
        let total = self.shape()[dim];
        let skip = total / n;
        assert!(skip * n == total);
        for i in 0..n {
            ret.push(self.slice(dim, i * skip, (i + 1) * skip));
        }
        ret
    }

    fn pick(&self, ids: &[u32], dim: u32) -> Self {
        self.device().pick_fw(self, ids, dim)
    }

    fn concat(xs: &[&Self], dim: u32) -> Self {
        xs[0].device().concat_fw(xs, dim)
    }

    fn reshape(&self, shape: Shape) -> Self {
        let mut t = self.device().copy_tensor(self);
        t.update_shape(shape);
        t
    }

    fn flatten(&self) -> Self {
        let s = self.shape();
        self.reshape(shape![s.volume(); s.batch()])
    }

    // batch

    fn batch_sum(&self) -> Self {
        self.device().batch_sum_fw(self)
    }

    fn batch_slice(&self, lower: u32, upper: u32) -> Self {
        self.device().batch_slice_fw(self, lower, upper)
    }

    fn batch_split(&self, n: u32) -> Vec<Self> {
        let mut ret = vec![];
        let total = self.shape().batch();
        let skip = total / n;
        assert!(skip * n == total);
        for i in 0..n {
            ret.push(self.batch_slice(i * skip, (i + 1) * skip));
        }
        ret
    }

    fn batch_pick(&self, ids: &[u32]) -> Self {
        self.device().batch_pick_fw(self, ids)
    }

    fn batch_concat(xs: &[&Self]) -> Self {
        assert!(xs.len() != 0);
        xs[0].device().batch_concat_fw(xs)
    }
}
