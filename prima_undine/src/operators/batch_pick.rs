use std::cell::RefCell;

use crate::functions::{BasicDeviceFunctions, BasicFunctions};
use crate::{shape_ops, Device, Operator, Shape, Tensor};

pub struct BatchPick<'dev> {
    device: &'dev crate::Device<'dev>,
    ids: Vec<u32>,
}

impl<'dev> BatchPick<'dev> {
    pub fn new(device: &'dev Device<'dev>, ids: &[u32]) -> Self {
        Self {
            device: device,
            ids: ids.to_vec(),
        }
    }
}

impl<'arg, 'dev> Operator<'arg, 'dev> for BatchPick<'dev> {
    fn name(&self) -> String {
        "BatchPick()".to_string()
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.device
    }

    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape> {
        vec![shape_ops::batch_pick(x[0], &self.ids)]
    }

    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].replace(x[0].batch_pick(&self.ids));
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]) {
        gy[0]
            .device()
            .batch_pick_bw(gy[0], &self.ids, &mut *gx[0].borrow_mut());
    }
}
