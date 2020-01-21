use crate::{Device, Shape, Tensor};

pub trait RandomDeviceFunctions {
    fn random_bernoulli(&self, shape: Shape, p: f32) -> Tensor;
    fn random_normal(&self, shape: Shape, mean: f32, sd: f32) -> Tensor;
    fn random_uniform(&self, shape: Shape, lower: f32, upper: f32) -> Tensor;
}

impl<'dev> RandomDeviceFunctions for Device<'dev> {
    fn random_bernoulli(&self, shape: Shape, p: f32) -> Tensor {
        let mut y = self.new_tensor(shape);
        y.alloc();
        self.call_fw_impl("random_bernoulli_impl", &[], &[], &[p], &mut [&mut y]);
        y
    }

    fn random_normal(&self, shape: Shape, mean: f32, sd: f32) -> Tensor {
        let mut y = self.new_tensor(shape);
        y.alloc();
        self.call_fw_impl("random_normal_impl", &[], &[], &[mean, sd], &mut [&mut y]);
        y
    }

    fn random_uniform(&self, shape: Shape, lower: f32, upper: f32) -> Tensor {
        let mut y = self.new_tensor(shape);
        y.alloc();
        self.call_fw_impl(
            "random_uniform_impl",
            &[],
            &[],
            &[lower, upper],
            &mut [&mut y],
        );
        y
    }
}
