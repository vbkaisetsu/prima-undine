use crate::Tensor;

pub trait Initializer {
    fn apply(&self, tensor: &mut Tensor);
}
