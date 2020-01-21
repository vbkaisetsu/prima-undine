use crate::optimizer::OptimizerBase;
use crate::{Optimizer, Parameter};

define_optimizer_struct!(SGD, eta, f32);
impl Optimizer for SGD {
    fn configure_parameter(&self, _parameter: &mut Parameter) {}
    fn update_parameter(&self, scale: f32, parameter: &mut Parameter) {
        let diff = (scale * self.eta) * &parameter.gradient;
        parameter.value -= &diff;
    }
}
