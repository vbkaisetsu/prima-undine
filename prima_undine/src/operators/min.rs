use std::cell::RefCell;

use crate::functions::BasicDeviceFunctions;
use crate::{Device, Operator, Shape, Tensor};

define_operator_struct!(Min, dim, u32);
impl<'arg, 'dev> Operator<'arg, 'dev> for Min<'dev> {
    fn name(&self) -> String {
        "Min(dim=".to_string() + &self.dim.to_string() + ")"
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![x[0].resize_dim(self.dim, 1)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device().min_fw(x[0], self.dim));
    }

    fn backward(&self, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        self.device()
            .min_bw(x[0], y[0], gy[0], self.dim, &mut *gx[0].borrow_mut());
    }
}
