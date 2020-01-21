# Prima-undine ⛵ : A Neural Network Toolkit

## Backends

* CPU (Naive)
* [OpenCL](https://github.com/vbkaisetsu/prima-undine-opencl)

## Example

```rust
use std::fs;

use prima_undine::functions::ArithmeticFunctions;
use prima_undine::functions::BasicFunctions;
use prima_undine::{
    devices as D, initializers as I, optimizers as O, shape, Device, Model, Node, Optimizer,
    Parameter, Tensor,
};
use prima_undine_contrib::functions::ContribFunctions;

use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Model, Serialize, Deserialize)]
struct XORModel<'dev> {
    pw1: Parameter<'dev>,
    pb1: Parameter<'dev>,
    pw2: Parameter<'dev>,
    pb2: Parameter<'dev>,
}

impl<'dev> XORModel<'dev> {
    fn new(device: &'dev Device) -> Self {
        Self {
            pw1: device.new_parameter(shape![8, 2], &I::Normal::new(0.5, 1.)),
            pb1: device.new_parameter(shape![8], &I::Normal::new(0.5, 1.)),
            pw2: device.new_parameter(shape![1, 8], &I::Normal::new(0.5, 1.)),
            pb2: device.new_parameter(shape![], &I::Normal::new(0.5, 1.)),
        }
    }
}

fn forward<'arg, 'dev, T>(x: &T, params: &'arg mut XORModel<'dev>) -> T
where
    'dev: 'arg,
    T: From<&'arg mut Parameter<'dev>> + ArithmeticFunctions<T> + BasicFunctions,
{
    let w1 = T::from(&mut params.pw1);
    let b1 = T::from(&mut params.pb1);
    let w2 = T::from(&mut params.pw2);
    let b2 = T::from(&mut params.pb2);
    let h = (w1.matmul(x) + b1).tanh();
    w2.matmul(h) + b2
}

fn main() {
    let dev = D::Naive::new();

    // Prepare training data
    let ref x_data = dev.new_tensor_by_slice(shape![2; 4], &[0., 0., 0., 1., 1., 0., 1., 1.]);
    let ref t_data = dev.new_tensor_by_slice(shape![; 4], &[0., 1., 1., 0.]);

    let train = true;

    if train {
        // Initialize parameters
        let mut model = XORModel::new(&dev);

        // Use SGD Optimizer
        let mut optimizer = O::SGD::new(0.1);
        optimizer.configure_model(&mut model);

        // Train data
        for _ in 0..100 {
            {
                let ref t = Node::from(t_data);
                let ref x = Node::from(x_data);
                let ref y = forward(x, &mut model);
                let ref diff = t - y;
                let ref loss = (diff * diff).batch_mean();
                println!("loss: {}", loss.to_float());
                println!("  y: {:?}", y.to_vec());
                loss.backward();
            }
            optimizer.update_model(&mut model);
        }

        // Save parameters using Serde
        let model_json = json!(model).to_string();
        fs::write("./model.json", &model_json).unwrap();

    } else {
        // Load parameters using Serde
        let model_json = fs::read_to_string("./model.json").unwrap();
        let mut model: XORModel = serde_json::from_str(&model_json).unwrap();

        // Move parameters to the device
        model.move_to_device(&dev);

        // Calculate and print the result
        let ref y = forward(x_data, &mut model);
        let ref diff = t_data - y;
        let loss = (diff * diff).batch_mean();
        println!("loss: {}", loss.to_float());
        println!("  y: {:?}", y.to_vec());
    }
}
```
