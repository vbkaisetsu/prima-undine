use rand::distributions::{Bernoulli, Distribution, Uniform};
use rand_distr::{LogNormal, Normal};

pub trait Randomizer: Send + Sync {
    fn fill_bernoulli(&mut self, p: f32, data: &mut [f32]);
    fn fill_uniform(&mut self, lower: f32, upper: f32, data: &mut [f32]);
    fn fill_normal(&mut self, mean: f32, sd: f32, data: &mut [f32]);
    fn fill_log_normal(&mut self, mean: f32, sd: f32, data: &mut [f32]);
}

pub struct DefaultRandomizer;

impl DefaultRandomizer {
    pub fn new() -> DefaultRandomizer {
        DefaultRandomizer {}
    }
}

impl Randomizer for DefaultRandomizer {
    fn fill_bernoulli(&mut self, p: f32, data: &mut [f32]) {
        let mut rng = rand::thread_rng();
        let dist = Bernoulli::new(p as f64).unwrap();
        for d in data {
            *d = dist.sample(&mut rng) as u32 as f32;
        }
    }

    fn fill_uniform(&mut self, lower: f32, upper: f32, data: &mut [f32]) {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(lower, upper);
        for d in data {
            *d = dist.sample(&mut rng);
        }
    }

    fn fill_normal(&mut self, mean: f32, sd: f32, data: &mut [f32]) {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(mean, sd).unwrap();
        for d in data {
            *d = dist.sample(&mut rng);
        }
    }

    fn fill_log_normal(&mut self, mean: f32, sd: f32, data: &mut [f32]) {
        let mut rng = rand::thread_rng();
        let dist = LogNormal::new(mean, sd).unwrap();
        for d in data {
            *d = dist.sample(&mut rng);
        }
    }
}
