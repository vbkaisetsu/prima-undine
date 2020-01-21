use crate::functions::BasicDeviceFunctions;

define_operator_x!(ELU, prelu_fw, prelu_bw, a, f32);
