use std::cell::RefCell;

use crate::functions::{BasicDeviceFunctions, BasicFunctions};
use crate::{shape_ops, Device, Operator, Shape, Tensor};

define_operator_struct!(Broadcast, dim, u32, size, u32);
impl<'arg, 'dev> Operator<'arg, 'dev> for Broadcast<'dev> {
    fn name(&self) -> String {
        "Broadcast".to_string()
            + "(dim="
            + &self.dim.to_string()
            + ",size="
            + &self.size.to_string()
            + ")"
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![shape_ops::broadcast(x[0], self.dim, self.size)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(self.device().broadcast_fw(x[0], self.dim, self.size));
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        *gx[0].borrow_mut() += gy[0].sum(self.dim);
    }
}
