use std::cell::RefCell;

use crate::functions::BasicDeviceFunctions;
use crate::{shape_ops, Device, Operator, Shape, Tensor};

define_operator_struct!(BatchSlice, lower, u32, upper, u32);
impl<'arg, 'dev> Operator<'arg, 'dev> for BatchSlice<'dev> {
    fn name(&self) -> String {
        "BatchSlice".to_string()
            + "(lower="
            + &self.lower.to_string()
            + ",upper="
            + &self.upper.to_string()
            + ")"
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![shape_ops::batch_slice(x[0], self.lower, self.upper)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device().batch_slice_fw(x[0], self.lower, self.upper));
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        self.device()
            .batch_slice_bw(gy[0], self.lower, &mut *gx[0].borrow_mut());
    }
}
