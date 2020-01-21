use std::cell::RefCell;

use crate::functions::BasicDeviceFunctions;
use crate::{Device, Operator, Shape, Tensor};

define_operator_struct!(Softplus);
impl<'arg, 'dev> Operator<'arg, 'dev> for Softplus<'dev> {
    fn name(&self) -> String {
        "Softplus".to_string()
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![x[0]]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device().softplus_fw(x[0]));
    }

    fn backward(&self, x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        *gx[0].borrow_mut() += self.device().sigmoid_fw(x[0]) * gy[0];
    }
}
