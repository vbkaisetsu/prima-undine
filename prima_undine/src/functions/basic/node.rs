use std::borrow::Borrow;

use crate::functions::{BasicDeviceFunctions, BasicFunctions};
use crate::{operators as op, Node, Shape};

impl<'arg, 'dev> BasicFunctions for Node<'arg, 'dev> {
    // core

    fn shape(&self) -> Shape {
        self.inner_value().shape
    }

    fn to_vec(&self) -> Vec<f32> {
        self.forward();
        self.inner_value().to_vec()
    }

    fn to_float(&self) -> f32 {
        self.forward();
        self.inner_value().to_float()
    }

    // utility

    fn argmax(&self, dim: u32) -> Vec<u32> {
        self.forward();
        self.device().argmax(&self.inner_value(), dim)
    }

    fn argmin(&self, dim: u32) -> Vec<u32> {
        self.forward();
        self.device().argmin(&self.inner_value(), dim)
    }

    fn argsort(&self, dim: u32) -> Vec<u32> {
        self.forward();
        self.device().argsort(&self.inner_value(), dim)
    }

    // basic

    fn sqrt(&self) -> Self {
        Node::create(op::Sqrt::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    fn abs(&self) -> Self {
        Node::create(op::Abs::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    // trigonometric

    fn sin(&self) -> Self {
        Node::create(op::Sin::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    fn cos(&self) -> Self {
        Node::create(op::Cos::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    fn tan(&self) -> Self {
        Node::create(op::Tan::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    // exp

    fn exp(&self) -> Self {
        Node::create(op::Exp::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    fn ln(&self) -> Self {
        Node::create(op::Ln::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    fn tanh(&self) -> Self {
        Node::create(op::Tanh::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    fn sigmoid(&self) -> Self {
        Node::create(op::Sigmoid::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    fn softplus(&self) -> Self {
        Node::create(op::Softplus::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    // reduction

    fn sum(&self, dim: u32) -> Self {
        Node::create(op::Sum::new(self.device(), dim), &[self])
            .pop()
            .unwrap()
    }

    fn max(&self, dim: u32) -> Self {
        Node::create(op::Max::new(self.device(), dim), &[self])
            .pop()
            .unwrap()
    }

    fn min(&self, dim: u32) -> Self {
        Node::create(op::Min::new(self.device(), dim), &[self])
            .pop()
            .unwrap()
    }

    fn broadcast(&self, dim: u32, size: u32) -> Self {
        Node::create(op::Broadcast::new(self.device(), dim, size), &[self])
            .pop()
            .unwrap()
    }

    fn logsumexp(&self, dim: u32) -> Self {
        Node::create(op::Logsumexp::new(self.device(), dim), &[self])
            .pop()
            .unwrap()
    }

    fn ln_softmax(&self, dim: u32) -> Self {
        self - self.logsumexp(dim).broadcast(dim, self.shape()[dim])
    }

    fn softmax(&self, dim: u32) -> Self {
        self.ln_softmax(dim).exp()
    }

    fn softmax_cross_entropy<T: Borrow<Self>>(&self, t: T, dim: u32) -> Self {
        Node::create(
            op::SoftmaxCrossEntropy::new(self.device(), dim),
            &[self, t.borrow()],
        )
        .pop()
        .unwrap()
    }

    fn sparse_softmax_cross_entropy(&self, ids: &[u32], dim: u32) -> Self {
        Node::create(
            op::SparseSoftmaxCrossEntropy::new(self.device(), ids, dim),
            &[self],
        )
        .pop()
        .unwrap()
    }

    // matrix

    fn matmul<T: Borrow<Self>>(&self, rhs: T) -> Self {
        let rhs = rhs.borrow();
        Node::create(op::Matmul::new(self.device()), &[self, rhs])
            .pop()
            .unwrap()
    }

    fn transpose(&self) -> Self {
        Node::create(op::Transpose::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    fn triangular_l(&self, k: u32) -> Self {
        Node::create(op::TriangularL::new(self.device(), k), &[self])
            .pop()
            .unwrap()
    }

    fn triangular_u(&self, k: u32) -> Self {
        Node::create(op::TriangularU::new(self.device(), k), &[self])
            .pop()
            .unwrap()
    }

    // ramp

    fn relu(&self) -> Self {
        Node::create(op::PReLU::new(self.device(), 0.), &[self])
            .pop()
            .unwrap()
    }

    fn lrelu(&self) -> Self {
        Node::create(op::PReLU::new(self.device(), 0.01), &[self])
            .pop()
            .unwrap()
    }

    fn prelu(&self, a: f32) -> Self {
        Node::create(op::PReLU::new(self.device(), a), &[self])
            .pop()
            .unwrap()
    }

    fn elu(&self, a: f32) -> Self {
        Node::create(op::ELU::new(self.device(), a), &[self])
            .pop()
            .unwrap()
    }

    // manipulation

    fn slice(&self, dim: u32, lower: u32, upper: u32) -> Self {
        Node::create(op::Slice::new(self.device(), dim, lower, upper), &[self])
            .pop()
            .unwrap()
    }

    fn split(&self, dim: u32, n: u32) -> Vec<Self> {
        Node::create(op::Split::new(self.device(), dim, n), &[self])
    }

    fn pick(&self, ids: &[u32], dim: u32) -> Self {
        Node::create(op::Pick::new(self.device(), ids, dim), &[self])
            .pop()
            .unwrap()
    }

    fn concat(xs: &[&Self], dim: u32) -> Self {
        assert!(xs.len() != 0);
        Node::create(op::Concat::new(xs[0].device(), dim), xs)
            .pop()
            .unwrap()
    }

    fn reshape(&self, shape: Shape) -> Self {
        Node::create(op::Reshape::new(self.device(), shape), &[self])
            .pop()
            .unwrap()
    }

    fn flatten(&self) -> Self {
        let s = self.shape();
        self.reshape(shape![s.volume(); s.batch()])
    }

    // batch

    fn batch_sum(&self) -> Self {
        Node::create(op::BatchSum::new(self.device()), &[self])
            .pop()
            .unwrap()
    }

    fn batch_slice(&self, lower: u32, upper: u32) -> Self {
        Node::create(op::BatchSlice::new(self.device(), lower, upper), &[self])
            .pop()
            .unwrap()
    }

    fn batch_split(&self, n: u32) -> Vec<Self> {
        Node::create(op::BatchSplit::new(self.device(), n), &[self])
    }

    fn batch_pick(&self, ids: &[u32]) -> Self {
        Node::create(op::BatchPick::new(self.device(), ids), &[self])
            .pop()
            .unwrap()
    }

    fn batch_concat(xs: &[&Self]) -> Self {
        assert!(xs.len() != 0);
        Node::create(op::BatchConcat::new(xs[0].device()), xs)
            .pop()
            .unwrap()
    }
}
