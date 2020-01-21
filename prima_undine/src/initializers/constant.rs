use crate::{Initializer, Tensor};

pub struct Constant {
    k: f32,
}

impl Constant {
    pub fn new(k: f32) -> Constant {
        Constant { k: k }
    }
}

impl Initializer for Constant {
    fn apply(&self, tensor: &mut Tensor) {
        tensor.alloc();
        tensor.reset(self.k);
    }
}
