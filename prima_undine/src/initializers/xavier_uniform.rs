use crate::functions::RandomDeviceFunctions;
use crate::{Initializer, Tensor};

pub struct XavierUniform {
    scale: f32,
}

impl XavierUniform {
    pub fn new(scale: f32) -> XavierUniform {
        XavierUniform { scale: scale }
    }
}

impl Initializer for XavierUniform {
    fn apply(&self, x: &mut Tensor) {
        let s = x.shape;
        assert!(s.is_matrix());
        let bound = self.scale * (6. / (s[0] + s[1]) as f32).sqrt();
        let new_tensor = x.device().random_uniform(s, -bound, bound);
        x.replace(new_tensor);
    }
}
