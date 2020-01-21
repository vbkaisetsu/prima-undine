use std::cell::RefCell;

use crate::{Device, Operator, Shape, Tensor};

pub struct Input<'arg, 'dev> {
    value: &'arg Tensor<'dev>,
}

impl<'arg, 'dev> Input<'arg, 'dev> {
    pub fn new(value: &'arg Tensor<'dev>) -> Input<'arg, 'dev> {
        Input { value: value }
    }
}

impl<'arg, 'dev> Operator<'arg, 'dev> for Input<'arg, 'dev> {
    fn name(&self) -> String {
        "Input".to_string()
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.value.device()
    }

    fn forward_shape(&self, _x: &[Shape]) -> Vec<Shape> {
        vec![self.value.shape]
    }

    fn forward(&self, _x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].refer(self.value);
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], _gy: &[&Tensor], _gx: &[&RefCell<Tensor>]) {
        // NOP
    }
}

pub struct InputOwner<'dev> {
    value: Tensor<'dev>,
}

impl<'dev> InputOwner<'dev> {
    pub fn new(value: Tensor<'dev>) -> InputOwner<'dev> {
        InputOwner { value: value }
    }
}

impl<'arg, 'dev> Operator<'arg, 'dev> for InputOwner<'dev> {
    fn name(&self) -> String {
        "InputOwner".to_string()
    }

    fn device(&self) -> &'dev Device<'dev> {
        self.value.device()
    }

    fn forward_shape(&self, _x: &[Shape]) -> Vec<Shape> {
        vec![self.value.shape]
    }

    fn forward(&self, _x: &[&Tensor], y: &mut [&mut Tensor<'arg>]) {
        y[0].alloc();
        y[0].device().reset_tensor_by_tensor(y[0], &self.value);
    }

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], _gy: &[&Tensor], _gx: &[&RefCell<Tensor>]) {
        // NOP
    }
}
