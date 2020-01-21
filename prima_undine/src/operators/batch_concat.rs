use std::cell::RefCell;

use crate::functions::{BasicDeviceFunctions, BasicFunctions};
use crate::{shape_ops, Device, Operator, Shape, Tensor};

define_operator_struct!(BatchConcat);
impl<'arg, 'dev> Operator<'arg, 'dev> for BatchConcat<'dev> {
    fn name(&self) -> String {
        "BatchConcat()".to_string()
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![shape_ops::batch_concat(x)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device().batch_concat_fw(x));
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        let mut offset = 0;
        for gxi in gx {
            let mut gxi = gxi.borrow_mut();
            let span = gxi.shape().batch();
            *gxi += gy[0].batch_slice(offset, offset + span);
            offset += span;
        }
    }
}
