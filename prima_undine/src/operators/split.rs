use std::cell::RefCell;

use crate::functions::BasicDeviceFunctions;
use crate::{Device, Operator, Shape, Tensor};

define_operator_struct!(Split, dim, u32, n, u32);
impl<'arg, 'dev> Operator<'arg, 'dev> for Split<'dev> {
    fn name(&self) -> String {
        "Split".to_string() + "(dim=" + &self.dim.to_string() + ",n=" + &self.n.to_string() + ")"
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        let total = x[0][self.dim];
        let size = total / self.n;
        assert!(size * self.n == total);
        vec![x[0].resize_dim(self.dim, size); self.n as usize]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        let total = x[0].shape[self.dim];
        let skip = total / self.n;
        for i in 0..self.n {
            y[i as usize].replace(
                self.device()
                    .slice_fw(x[0], self.dim, i * skip, (i + 1) * skip),
            );
        }
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        let total = gx[0].borrow().shape[self.dim];
        let skip = total / self.n;
        for i in 0..self.n {
            self.device()
                .slice_bw(gy[i as usize], self.dim, i * skip, &mut *gx[0].borrow_mut());
        }
    }
}
