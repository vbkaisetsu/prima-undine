use std::cell::RefCell;

use crate::functions::{BasicDeviceFunctions, BasicFunctions};
use crate::{shape_ops, Device, Operator, Shape, Tensor};

define_operator_struct!(Concat, dim, u32);
impl<'arg, 'dev> Operator<'arg, 'dev> for Concat<'dev> {
    fn name(&self) -> String {
        "Concat".to_string() + "(dim=" + &self.dim.to_string() + ")"
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![shape_ops::concat(x, self.dim)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device().concat_fw(x, self.dim));
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        let mut offset = 0;
        for gxi in gx {
            let mut gxi = gxi.borrow_mut();
            let span = gxi.shape()[self.dim];
            *gxi += gy[0].slice(self.dim, offset, offset + span);
            offset += span;
        }
    }
}
