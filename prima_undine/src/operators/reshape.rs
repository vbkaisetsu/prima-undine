use std::cell::RefCell;

use crate::{Device, Operator, Shape, Tensor};

define_operator_struct!(Reshape, shape, Shape);
impl<'arg, 'dev> Operator<'arg, 'dev> for Reshape<'dev> {
    fn name(&self) -> String {
        format!("Reshape(shape={:?})", self.shape)
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, _x: &[Shape]) -> Vec<Shape> {
        vec![self.shape]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        let mut t = self.device.copy_tensor(x[0]);
        t.update_shape(self.shape);
        y[0].replace(t);
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        let mut gx = gx[0].borrow_mut();
        let mut t = self.device.copy_tensor(gy[0]);
        t.update_shape(gx.shape);
        *gx += t;
    }
}
