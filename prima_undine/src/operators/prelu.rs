use crate::functions::BasicDeviceFunctions;

define_operator_x!(PReLU, prelu_fw, prelu_bw, a, f32);
