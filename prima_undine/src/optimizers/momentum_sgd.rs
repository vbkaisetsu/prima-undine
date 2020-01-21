use crate::optimizer::OptimizerBase;
use crate::{Optimizer, Parameter};

define_optimizer_struct!(MomentumSGD, eta, f32, momentum, f32);
impl Optimizer for MomentumSGD {
    fn configure_parameter(&self, parameter: &mut Parameter) {
        if !parameter.has_stat("MomentumSGD.m") {
            parameter.add_stat("MomentumSGD.m", parameter.shape());
        }
    }

    fn update_parameter(&self, scale: f32, parameter: &mut Parameter) {
        let mdiff = (scale * self.eta) * &parameter.gradient;
        let mut m = parameter.stats["MomentumSGD.m"].borrow_mut();
        *m *= self.momentum;
        *m -= &mdiff;
        parameter.value += &*m;
    }
}
