use crate::functions::RandomDeviceFunctions;
use crate::{Initializer, Tensor};

pub struct Normal {
    mean: f32,
    sd: f32,
}

impl Normal {
    pub fn new(mean: f32, sd: f32) -> Normal {
        Normal { mean: mean, sd: sd }
    }
}

impl Initializer for Normal {
    fn apply(&self, tensor: &mut Tensor) {
        let new_tensor = tensor
            .device()
            .random_normal(tensor.shape, self.mean, self.sd);
        tensor.replace(new_tensor);
    }
}
