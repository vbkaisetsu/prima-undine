use std::cell::RefCell;

use crate::{Device, Operator, Shape, Tensor};

pub struct Parameter<'arg, 'dev> {
    value: &'arg Tensor<'dev>,
    gradient: RefCell<&'arg mut Tensor<'dev>>,
}

impl<'arg, 'dev> Parameter<'arg, 'dev> {
    pub fn new(parameter: &'arg mut crate::Parameter<'dev>) -> Parameter<'arg, 'dev> {
        Parameter {
            value: &parameter.value,
            gradient: RefCell::new(&mut parameter.gradient),
        }
    }
}

impl<'arg, 'dev> Operator<'arg, 'dev> for Parameter<'arg, 'dev> {
    fn name(&self) -> String {
        "Parameter".to_string()
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

    fn backward(&self, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], _gx: &[&RefCell<Tensor>]) {
        **self.gradient.borrow_mut() += gy[0];
    }
}
