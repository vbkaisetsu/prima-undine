use std::cell::RefCell;

use crate::functions::{BasicDeviceFunctions, BasicFunctions};
use crate::{Device, Operator, Shape, Tensor};

define_operator_struct!(Logsumexp, dim, u32);
impl<'arg, 'dev> Operator<'arg, 'dev> for Logsumexp<'dev> {
    fn name(&self) -> String {
        "Logsumexp(dim=".to_string() + &self.dim.to_string() + ")"
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![x[0].resize_dim(self.dim, 1)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device().logsumexp_fw(x[0], self.dim));
    }

    fn backward(&self, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        let size = x[0].shape().dims()[self.dim as usize];
        *gx[0].borrow_mut() +=
            (x[0] - y[0].broadcast(self.dim, size)).exp() * gy[0].broadcast(self.dim, size);
    }
}
