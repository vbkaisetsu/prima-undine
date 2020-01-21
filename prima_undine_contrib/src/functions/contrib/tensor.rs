use prima_undine::functions::BasicFunctions;
use prima_undine::functions::RandomDeviceFunctions;
use prima_undine::Tensor;

use super::ContribFunctions;

impl<'dev> ContribFunctions for Tensor<'dev> {
    type Output = Self;

    fn mean(&self, dim: u32) -> Self {
        self.sum(dim) / self.shape()[dim] as f32
    }

    fn batch_mean(&self) -> Self {
        self.batch_sum() / self.shape().batch() as f32
    }

    fn dropout(&self, rate: f32, enabled: bool) -> Self {
        if !enabled {
            1. * self
        } else if rate >= 1. {
            0. * self
        } else {
            let p = 1. - rate;
            (1. / p) * self * Self::from(self.device().random_bernoulli(self.shape(), p))
        }
    }

    fn slice_sum(xs: &[&Self]) -> Self {
        assert!(xs.len() != 0);
        let mut ret = Self::from(xs[0].device().new_tensor_by_constant(xs[0].shape(), 0.));
        for x in xs {
            ret = ret + *x;
        }
        ret
    }

    fn batch_normalization(&self, g: &Self, b: &Self, eps: f32, enabled: bool) -> Self {
        if enabled {
            let ref m = self.batch_mean();
            let ref shift = self - m;
            let v = (shift * shift).batch_mean() + eps;
            let x = (self - m) / v.sqrt();
            g * x + b
        } else {
            1. * self
        }
    }

    fn layer_normalization(&self, g: &Self, b: &Self, eps: f32) -> Self {
        let ref m = self.flatten().mean(0);
        let ref shift = self - m;
        let v = (shift * shift).flatten().mean(0) + eps;
        let x = (self - m) / v.sqrt();
        g * x + b
    }
}
