macro_rules! define_optimizer_struct {
    ( $name:ident $(, $attr:ident, $type:ty )* ) => {
        pub struct $name {
            epoch: u32,
            lr_scale: f32,
            l2_strength: f32,
            clip_threshold: f32,
            $( $attr: $type, )*
        }

        impl $name {
            pub fn new( $( $attr: $type, )* ) -> $name {
                $name {
                    epoch: 0,
                    lr_scale: 1.,
                    l2_strength: 0.,
                    clip_threshold: 0.,
                    $( $attr: $attr, )*
                }
            }
        }

        impl OptimizerBase for $name {
            fn epoch(&mut self) -> &mut u32 {
                &mut self.epoch
            }

            fn get_learning_rate_scaling(&self) -> f32 {
                self.lr_scale
            }

            fn set_learning_rate_scaling(&mut self, scale: f32) {
                assert!(scale >= 0.);
                self.lr_scale = scale;
            }

            fn get_weight_decay(&self) -> f32 {
                self.l2_strength
            }

            fn set_weight_decay(&mut self, strength: f32) {
                assert!(strength >= 0.);
                self.l2_strength = strength;
            }

            fn get_gradient_clipping(&self) -> f32 {
                self.clip_threshold
            }

            fn set_gradient_clipping(&mut self, threshold: f32) {
                assert!(threshold >= 0.);
                self.clip_threshold = threshold;
            }
        }
    };
}

mod adam;
mod momentum_sgd;
mod sgd;

pub use self::adam::Adam;
pub use self::momentum_sgd::MomentumSGD;
pub use self::sgd::SGD;
