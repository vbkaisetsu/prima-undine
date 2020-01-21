use crate::{Device, Parameter};

pub trait Model<'dev> {
    fn parameters(&self) -> Vec<&Parameter<'dev>>;
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<'dev>>;
    fn move_to_device(&mut self, device: &'dev Device<'dev>) {
        for param in self.parameters_mut() {
            param.move_to_device(device);
        }
    }
}

impl<'dev, M> Model<'dev> for Vec<M>
where
    M: Model<'dev>,
{
    fn parameters(&self) -> Vec<&Parameter<'dev>> {
        let mut params = vec![];
        for model in self {
            params.append(&mut model.parameters());
        }
        params
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<'dev>> {
        let mut params = vec![];
        for model in self {
            params.append(&mut model.parameters_mut());
        }
        params
    }
    fn move_to_device(&mut self, device: &'dev Device<'dev>) {
        for model in self {
            for param in model.parameters_mut() {
                param.move_to_device(device);
            }
        }
    }
}

impl<'dev> Model<'dev> for Vec<Parameter<'dev>> {
    fn parameters(&self) -> Vec<&Parameter<'dev>> {
        self.iter().collect()
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<'dev>> {
        self.iter_mut().collect()
    }
    fn move_to_device(&mut self, device: &'dev Device<'dev>) {
        for param in self {
            param.move_to_device(device);
        }
    }
}
