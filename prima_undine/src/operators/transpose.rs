use std::cell::RefCell;

use crate::functions::BasicDeviceFunctions;
use crate::{shape_ops, Device, Operator, Shape, Tensor};

define_operator_struct!(Transpose);
impl<'arg, 'dev> Operator<'arg, 'dev> for Transpose<'dev> {
    fn name(&self) -> String {
        "Transpose".to_string()
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![shape_ops::transpose(x[0])]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device.transpose_fw(x[0]));
    }

    fn backward(&self, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        self.device
            .transpose_bw(x[0], y[0], gy[0], &mut *gx[0].borrow_mut());
    }
}
