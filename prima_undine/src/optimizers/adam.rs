use crate::functions::BasicFunctions;
use crate::optimizer::OptimizerBase;
use crate::{Optimizer, Parameter};

define_optimizer_struct!(Adam, alpha, f32, beta1, f32, beta2, f32, eps, f32);
impl Optimizer for Adam {
    fn configure_parameter(&self, parameter: &mut Parameter) {
        if !parameter.has_stat("Adam.m1") {
            parameter.add_stat("Adam.m1", parameter.shape());
        }
        if !parameter.has_stat("Adam.m2") {
            parameter.add_stat("Adam.m2", parameter.shape());
        }
    }

    fn update_parameter(&self, scale: f32, parameter: &mut Parameter) {
        let epoch = (self.epoch + 1) as f32;
        let g = &parameter.gradient;
        let mut m1 = parameter.stats["Adam.m1"].borrow_mut();
        let mut m2 = parameter.stats["Adam.m2"].borrow_mut();
        *m1 *= self.beta1;
        *m1 += (1. - self.beta1) * g;
        *m2 *= self.beta2;
        *m2 += (1. - self.beta2) * g * g;
        let mm1 = &*m1 / (1. - self.beta1.powf(epoch));
        let mm2 = &*m2 / (1. - self.beta2.powf(epoch));
        parameter.value -= (scale * self.alpha) * mm1 / (mm2.sqrt() + self.eps);
    }
}
