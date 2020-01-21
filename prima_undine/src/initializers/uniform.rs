use crate::functions::RandomDeviceFunctions;
use crate::{Initializer, Tensor};

pub struct Uniform {
    lower: f32,
    upper: f32,
}

impl Uniform {
    pub fn new(lower: f32, upper: f32) -> Uniform {
        Uniform {
            lower: lower,
            upper: upper,
        }
    }
}

impl Initializer for Uniform {
    fn apply(&self, x: &mut Tensor) {
        let new_tensor = x.device().random_uniform(x.shape, self.lower, self.upper);
        x.replace(new_tensor);
    }
}
