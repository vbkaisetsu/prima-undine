use std::cell::RefCell;

use crate::functions::BasicDeviceFunctions;
use crate::{Device, Operator, Shape, Tensor};

define_operator_struct!(BatchSum);
impl<'arg, 'dev> Operator<'arg, 'dev> for BatchSum<'dev> {
    fn name(&self) -> String {
        "BatchSum".to_string()
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![x[0].resize_batch(1)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device().batch_sum_fw(x[0]));
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        *gx[0].borrow_mut() += gy[0];
    }
}
