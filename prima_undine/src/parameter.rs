use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;

use serde::{Deserialize, Serialize};

use crate::{Device, Shape, Tensor};

#[derive(Serialize, Deserialize)]
pub struct Parameter<'dev> {
    pub value: Tensor<'dev>,
    pub gradient: Tensor<'dev>,
    pub stats: HashMap<String, RefCell<Tensor<'dev>>>,
}

impl<'dev> Parameter<'dev> {
    pub fn new(value: Tensor<'dev>, gradient: Tensor<'dev>) -> Parameter<'dev> {
        assert_eq!(value.shape.batch(), 1);
        assert_eq!(gradient.shape.batch(), 1);
        Parameter {
            value: value,
            gradient: gradient,
            stats: HashMap::new(),
        }
    }

    pub fn move_to_device(&mut self, device: &'dev Device<'dev>) {
        self.value.move_to_device(device);
        self.gradient.move_to_device(device);
        for (_, stat) in self.stats.iter_mut() {
            stat.borrow_mut().move_to_device(device);
        }
    }

    pub fn shape(&self) -> Shape {
        self.value.shape
    }

    pub fn reset_gradient(&mut self) {
        self.gradient.reset(0.);
    }

    pub fn add_stat(&mut self, name: &str, shape: Shape) {
        assert!(!self.stats.contains_key(name));
        let stat = self.value.device().new_tensor_by_constant(shape, 0.);
        self.stats.insert(name.to_string(), RefCell::new(stat));
    }

    pub fn has_stat(&self, name: &str) -> bool {
        self.stats.contains_key(name)
    }

    pub fn parameters(&self) -> Vec<&Parameter<'dev>> {
        vec![self]
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<'dev>> {
        vec![self]
    }
}

impl<'dev> Deref for Parameter<'dev> {
    type Target = Tensor<'dev>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
