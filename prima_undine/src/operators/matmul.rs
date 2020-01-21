use std::cell::RefCell;

use crate::functions::BasicDeviceFunctions;
use crate::{shape_ops, Device, Operator, Shape, Tensor};

define_operator_struct!(Matmul);
impl<'arg, 'dev> Operator<'arg, 'dev> for Matmul<'dev> {
    fn name(&self) -> String {
        "Matmul".to_string()
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![shape_ops::matmul(x[0], x[1])]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device().matmul_fw(x[0], x[1]));
    }

    fn backward(&self, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        self.device()
            .matmul_bw_a(x[0], x[1], y[0], gy[0], &mut *gx[0].borrow_mut());
        self.device()
            .matmul_bw_b(x[0], x[1], y[0], gy[0], &mut *gx[1].borrow_mut());
    }
}
