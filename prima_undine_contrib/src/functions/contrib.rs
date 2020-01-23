mod node;
mod tensor;

pub trait ContribFunctions {
    type Output;
    fn mean(&self, dim: u32) -> Self;
    fn batch_mean(&self) -> Self;
    fn dropout(&self, rate: f32, enabled: bool) -> Self;
    fn dropout_dim(&self, dim: u32, rate: f32, enabled: bool) -> Self;
    fn slice_sum(xs: &[&Self]) -> Self;
    fn batch_normalization(&self, g: &Self, b: &Self, eps: f32, enabled: bool) -> Self;
    fn layer_normalization(&self, g: &Self, b: &Self, eps: f32) -> Self;
}
