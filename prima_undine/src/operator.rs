use std::cell::RefCell;

use crate::{Device, Shape, Tensor};

pub trait Operator<'arg, 'dev> {
    fn name(&self) -> String;
    fn device(&self) -> &'dev Device<'dev>;
    fn forward_shape(&self, x: &[Shape]) -> Vec<Shape>;
    fn forward(&self, x: &[&Tensor], y: &mut [&mut Tensor<'arg>]);
    fn backward(&self, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]);
}
