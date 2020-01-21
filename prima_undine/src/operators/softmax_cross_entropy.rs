use std::cell::RefCell;

use crate::functions::{BasicDeviceFunctions, BasicFunctions};
use crate::{shape_ops, Device, Operator, Shape, Tensor};

define_operator_struct!(SoftmaxCrossEntropy, dim, u32);
impl<'arg, 'dev> Operator<'arg, 'dev> for SoftmaxCrossEntropy<'dev> {
    fn name(&self) -> String {
        "SoftmaxCrossEntropy(dim=".to_string() + &self.dim.to_string() + ")"
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![shape_ops::elementwise(x[0], x[1]).resize_dim(self.dim, 1)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(x[0].softmax_cross_entropy(x[1], self.dim));
    }

    fn backward(&self, x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        let ln_softmax_x = x[0].ln_softmax(self.dim);
        let bcast_gy = gy[0].broadcast(self.dim, x[0].shape()[self.dim]);
        *gx[0].borrow_mut() += (ln_softmax_x.exp() - x[1]) * &bcast_gy;
        *gx[1].borrow_mut() -= ln_softmax_x * &bcast_gy;
    }
}

pub struct SparseSoftmaxCrossEntropy<'dev> {
    device: &'dev crate::Device<'dev>,
    ids: Vec<u32>,
    dim: u32,
}

impl<'dev> SparseSoftmaxCrossEntropy<'dev> {
    pub fn new(
        device: &'dev Device<'dev>,
        ids: &[u32],
        dim: u32,
    ) -> SparseSoftmaxCrossEntropy<'dev> {
        SparseSoftmaxCrossEntropy {
            device: device,
            ids: ids.to_vec(),
            dim: dim,
        }
    }
}

impl<'arg, 'dev> Operator<'arg, 'dev> for SparseSoftmaxCrossEntropy<'dev> {
    fn name(&self) -> String {
        "SparseSoftmaxCrossEntropy(dim=".to_string() + &self.dim.to_string() + ")"
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![shape_ops::pick(x[0], &self.ids, self.dim)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(x[0].sparse_softmax_cross_entropy(&self.ids, self.dim));
    }

    fn backward(&self, x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        *gx[0].borrow_mut() +=
            x[0].softmax(self.dim) * gy[0].broadcast(self.dim, x[0].shape()[self.dim]);
        gy[0]
            .device()
            .pick_bw(&-gy[0], &self.ids, self.dim, &mut *gx[0].borrow_mut());
    }
}
