use std::cell::RefCell;

use crate::functions::BasicDeviceFunctions;
use crate::{Device, Operator, Shape, Tensor};

define_operator_struct!(Powi, k, i32);
impl<'arg, 'dev> Operator<'arg, 'dev> for Powi<'dev> {
    fn name(&self) -> String {
        "Powi(k=".to_string() + &self.k.to_string() + ")"
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![x[0]]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device().powi_fw(x[0], self.k));
    }

    fn backward(&self, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        self.device()
            .powi_bw(x[0], y[0], gy[0], self.k, &mut *gx[0].borrow_mut());
    }
}
