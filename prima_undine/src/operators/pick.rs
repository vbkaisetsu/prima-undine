use std::cell::RefCell;

use crate::functions::{BasicDeviceFunctions, BasicFunctions};
use crate::{shape_ops, Device, Operator, Shape, Tensor};

pub struct Pick<'dev> {
    device: &'dev crate::Device<'dev>,
    ids: Vec<u32>,
    dim: u32,
}

impl<'dev> Pick<'dev> {
    pub fn new(device: &'dev Device<'dev>, ids: &[u32], dim: u32) -> Pick<'dev> {
        Pick {
            device: device,
            ids: ids.to_vec(),
            dim: dim,
        }
    }
}

impl<'arg, 'dev> Operator<'arg, 'dev> for Pick<'dev> {
    fn name(&self) -> String {
        "Pick(dim=".to_string() + &self.dim.to_string() + ")"
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![shape_ops::pick(x[0], &self.ids, self.dim)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(x[0].pick(&self.ids, self.dim));
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        gy[0]
            .device()
            .pick_bw(gy[0], &self.ids, self.dim, &mut *gx[0].borrow_mut());
    }
}
