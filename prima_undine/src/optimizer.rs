use crate::functions::BasicFunctions;
use crate::{Model, Parameter};

pub trait OptimizerBase {
    fn epoch(&mut self) -> &mut u32;
    fn get_learning_rate_scaling(&self) -> f32;
    fn set_learning_rate_scaling(&mut self, scale: f32);
    fn get_weight_decay(&self) -> f32;
    fn set_weight_decay(&mut self, strength: f32);
    fn get_gradient_clipping(&self) -> f32;
    fn set_gradient_clipping(&mut self, threshold: f32);
}

pub trait Optimizer: OptimizerBase {
    fn configure_parameter(&self, parameter: &mut Parameter);
    fn update_parameter(&self, scale: f32, parameter: &mut Parameter);
    fn configure_parameters(&self, parameters: &mut [&mut Parameter]) {
        for param in parameters {
            self.configure_parameter(param);
        }
    }
    fn update_parameters(&mut self, parameters: &mut [&mut Parameter]) {
        let l2_strength = self.get_weight_decay();
        if l2_strength > 0. {
            for param in parameters.iter_mut() {
                let diff = l2_strength * &param.value;
                param.gradient += &diff;
            }
        }
        let clip_threshold = self.get_gradient_clipping();
        if clip_threshold > 0. {
            let mut sq_norm = 0.;
            for param in parameters.iter_mut() {
                let g = &param.gradient;
                sq_norm += (g * g).flatten().sum(0).to_float();
            }
            if sq_norm > clip_threshold * clip_threshold {
                let clip_scale = clip_threshold / sq_norm.sqrt();
                for param in parameters.iter_mut() {
                    param.gradient *= clip_scale;
                }
            }
        }
        for param in parameters.iter_mut() {
            self.update_parameter(self.get_learning_rate_scaling(), param);
            param.reset_gradient();
        }
        *self.epoch() += 1;
    }
    fn configure_model<'dev, M: Model<'dev>>(&self, model: &mut M) {
        self.configure_parameters(&mut model.parameters_mut());
    }
    fn update_model<'dev, M: Model<'dev>>(&mut self, model: &mut M) {
        self.update_parameters(&mut model.parameters_mut());
    }
}
