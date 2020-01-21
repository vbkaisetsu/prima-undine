#[macro_use]
mod common;

mod abs;
mod add;
mod add_assign;
mod argmax;
mod argmin;
mod argsort;
mod batch_concat;
mod batch_pick;
mod batch_slice;
mod batch_sum;
mod broadcast;
mod concat;
//mod conv2d;
//mod copy_tensor;
mod cos;
mod div;
mod elu;
mod exp;
mod flip;
mod identity;
mod ln;
mod logsumexp;
mod matmul;
mod max;
mod min;
mod mul;
mod mul_assign;
mod neg;
mod permute_dims;
mod pick;
mod powf;
mod powi;
mod prelu;
mod random;
mod reset_tensor;
mod sigmoid;
mod sin;
mod slice;
mod softplus;
mod sqrt;
mod sub;
mod sub_assign;
mod sum;
mod tan;
mod tanh;
mod tensor_to_vector;
mod transpose;
mod triangular_l;
mod triangular_u;

use crate::device_impl::DeviceImpl;
use crate::random::DefaultRandomizer;
use crate::{Device, Randomizer};
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

pub struct Naive {}

impl DeviceImpl for Naive {
    fn identifier(&self) -> String {
        "Naive".to_string()
    }

    fn new_handle(&self, size: u32) -> AtomicPtr<c_void> {
        let data = Box::new(vec![0f32; size as usize]);
        AtomicPtr::new(Box::into_raw(data) as *mut c_void)
    }

    fn drop_handle(&self, handle: &AtomicPtr<c_void>) {
        unsafe {
            Box::from_raw(handle.load(Ordering::Acquire) as *mut Vec<f32>);
        }
    }
}

impl<'dev> Naive {
    pub fn new() -> Device<'dev> {
        let mut dev = Device::new(Naive {});

        let randomizer: Arc<Mutex<Box<dyn Randomizer>>> =
            Arc::new(Mutex::new(Box::new(DefaultRandomizer::new())));

        // initializers

        dev.register_fw_impl("reset_tensor_impl", reset_tensor::ResetTensorImpl::new());
        dev.register_fw_impl(
            "reset_tensor_by_slice_impl",
            reset_tensor::ResetTensorBySliceImpl::new(),
        );
        dev.register_fw_impl(
            "reset_tensor_by_tensor_impl",
            reset_tensor::ResetTensorByTensorImpl::new(),
        );

        dev.register_fw_f32_impl(
            "tensor_to_vector_impl",
            tensor_to_vector::TensorToVectorImpl::new(),
        );

        dev.register_fw_impl("identity_impl", identity::IdentityImpl::new());

        dev.register_fw_impl(
            "random_bernoulli_impl",
            random::RandomBernoulliImpl::new(Arc::clone(&randomizer)),
        );
        dev.register_fw_impl(
            "random_normal_impl",
            random::RandomNormalImpl::new(Arc::clone(&randomizer)),
        );
        dev.register_fw_impl(
            "random_uniform_impl",
            random::RandomUniformImpl::new(Arc::clone(&randomizer)),
        );

        // assign

        dev.register_fw_impl("add_assign_impl", add_assign::AddAssignImpl::new());

        dev.register_fw_impl("sub_assign_impl", sub_assign::SubAssignImpl::new());

        dev.register_fw_impl(
            "mul_assign_const_impl",
            mul_assign::MulAssignConstImpl::new(),
        );

        // utility

        dev.register_fw_u32_impl("argmax_impl", argmax::ArgmaxImpl::new());

        dev.register_fw_u32_impl("argmin_impl", argmin::ArgminImpl::new());

        dev.register_fw_u32_impl("argsort_impl", argsort::ArgsortImpl::new());

        // arithmetic

        dev.register_fw_impl("neg_fw_impl", neg::NegFwImpl::new());

        dev.register_fw_impl("add_fw_impl", add::AddFwImpl::new());
        dev.register_bw_impl("add_bw_a_impl", add::AddBwAImpl::new());
        dev.register_bw_impl("add_bw_b_impl", add::AddBwBImpl::new());
        dev.register_fw_impl("add_const_fw_impl", add::AddConstFwImpl::new());
        dev.register_bw_impl("add_const_bw_impl", add::AddConstBwImpl::new());
        dev.register_fw_impl("add_scalar_fw_impl", add::AddScalarFwImpl::new());

        dev.register_fw_impl("sub_fw_impl", sub::SubFwImpl::new());
        dev.register_bw_impl("sub_bw_a_impl", sub::SubBwAImpl::new());
        dev.register_bw_impl("sub_bw_b_impl", sub::SubBwBImpl::new());
        dev.register_fw_impl("sub_const_l_fw_impl", sub::SubConstLFwImpl::new());
        dev.register_bw_impl("sub_const_l_bw_impl", sub::SubConstLBwImpl::new());
        dev.register_fw_impl("sub_const_r_fw_impl", sub::SubConstRFwImpl::new());
        dev.register_bw_impl("sub_const_r_bw_impl", sub::SubConstRBwImpl::new());
        dev.register_fw_impl("sub_scalar_l_fw_impl", sub::SubScalarLFwImpl::new());
        dev.register_fw_impl("sub_scalar_r_fw_impl", sub::SubScalarRFwImpl::new());

        dev.register_fw_impl("mul_fw_impl", mul::MulFwImpl::new());
        dev.register_bw_impl("mul_bw_a_impl", mul::MulBwAImpl::new());
        dev.register_bw_impl("mul_bw_b_impl", mul::MulBwBImpl::new());
        dev.register_fw_impl("mul_const_fw_impl", mul::MulConstFwImpl::new());
        dev.register_bw_impl("mul_const_bw_impl", mul::MulConstBwImpl::new());
        dev.register_fw_impl("mul_scalar_fw_impl", mul::MulScalarFwImpl::new());

        dev.register_fw_impl("div_fw_impl", div::DivFwImpl::new());
        dev.register_bw_impl("div_bw_a_impl", div::DivBwAImpl::new());
        dev.register_bw_impl("div_bw_b_impl", div::DivBwBImpl::new());
        dev.register_fw_impl("div_const_l_fw_impl", div::DivConstLFwImpl::new());
        dev.register_bw_impl("div_const_l_bw_impl", div::DivConstLBwImpl::new());
        dev.register_fw_impl("div_const_r_fw_impl", div::DivConstRFwImpl::new());
        dev.register_bw_impl("div_const_r_bw_impl", div::DivConstRBwImpl::new());
        dev.register_fw_impl("div_scalar_l_fw_impl", div::DivScalarLFwImpl::new());
        dev.register_fw_impl("div_scalar_r_fw_impl", div::DivScalarRFwImpl::new());

        // basic

        dev.register_fw_impl("powf_fw_impl", powf::PowfFwImpl::new());
        dev.register_bw_impl("powf_bw_a_impl", powf::PowfBwAImpl::new());
        dev.register_bw_impl("powf_bw_b_impl", powf::PowfBwBImpl::new());
        dev.register_fw_impl("powf_const_l_fw_impl", powf::PowfConstLFwImpl::new());
        dev.register_bw_impl("powf_const_l_bw_impl", powf::PowfConstLBwImpl::new());
        dev.register_fw_impl("powf_const_r_fw_impl", powf::PowfConstRFwImpl::new());
        dev.register_bw_impl("powf_const_r_bw_impl", powf::PowfConstRBwImpl::new());
        dev.register_fw_impl("powf_scalar_l_fw_impl", powf::PowfScalarLFwImpl::new());
        dev.register_fw_impl("powf_scalar_r_fw_impl", powf::PowfScalarRFwImpl::new());

        dev.register_fw_impl("sqrt_fw_impl", sqrt::SqrtFwImpl::new());
        dev.register_bw_impl("sqrt_bw_impl", sqrt::SqrtBwImpl::new());

        dev.register_fw_impl("abs_fw_impl", abs::AbsFwImpl::new());
        dev.register_bw_impl("abs_bw_impl", abs::AbsBwImpl::new());

        dev.register_fw_impl("powi_fw_impl", powi::PowiFwImpl::new());
        dev.register_bw_impl("powi_bw_impl", powi::PowiBwImpl::new());

        // trigonometric

        dev.register_fw_impl("sin_fw_impl", sin::SinFwImpl::new());
        dev.register_bw_impl("sin_bw_impl", sin::SinBwImpl::new());

        dev.register_fw_impl("cos_fw_impl", cos::CosFwImpl::new());
        dev.register_bw_impl("cos_bw_impl", cos::CosBwImpl::new());

        dev.register_fw_impl("tan_fw_impl", tan::TanFwImpl::new());
        dev.register_bw_impl("tan_bw_impl", tan::TanBwImpl::new());

        // exp

        dev.register_fw_impl("exp_fw_impl", exp::ExpFwImpl::new());
        dev.register_bw_impl("exp_bw_impl", exp::ExpBwImpl::new());

        dev.register_fw_impl("ln_fw_impl", ln::LnFwImpl::new());
        dev.register_bw_impl("ln_bw_impl", ln::LnBwImpl::new());

        dev.register_fw_impl("tanh_fw_impl", tanh::TanhFwImpl::new());
        dev.register_bw_impl("tanh_bw_impl", tanh::TanhBwImpl::new());

        dev.register_fw_impl("sigmoid_fw_impl", sigmoid::SigmoidFwImpl::new());
        dev.register_bw_impl("sigmoid_bw_impl", sigmoid::SigmoidBwImpl::new());

        dev.register_fw_impl("softplus_fw_impl", softplus::SoftplusFwImpl::new());

        // reduction

        dev.register_fw_impl("sum_fw_impl", sum::SumFwImpl::new());

        dev.register_fw_impl("logsumexp_fw_impl", logsumexp::LogsumexpFwImpl::new());

        dev.register_fw_impl("max_fw_impl", max::MaxFwImpl::new());
        dev.register_bw_impl("max_bw_impl", max::MaxBwImpl::new());

        dev.register_fw_impl("min_fw_impl", min::MinFwImpl::new());
        dev.register_bw_impl("min_bw_impl", min::MinBwImpl::new());

        dev.register_fw_impl("broadcast_fw_impl", broadcast::BroadcastFwImpl::new());

        // matrix

        dev.register_fw_impl("matmul_fw_impl", matmul::MatmulFwImpl::new());
        dev.register_bw_impl("matmul_bw_a_impl", matmul::MatmulBwAImpl::new());
        dev.register_bw_impl("matmul_bw_b_impl", matmul::MatmulBwBImpl::new());

        dev.register_fw_impl("transpose_fw_impl", transpose::TransposeFwImpl::new());
        dev.register_bw_impl("transpose_bw_impl", transpose::TransposeBwImpl::new());

        dev.register_fw_impl(
            "permute_dims_fw_impl",
            permute_dims::PermuteDimsFwImpl::new(),
        );
        dev.register_bw_impl(
            "permute_dims_bw_impl",
            permute_dims::PermuteDimsBwImpl::new(),
        );

        dev.register_fw_impl("flip_fw_impl", flip::FlipFwImpl::new());
        dev.register_bw_impl("flip_bw_impl", flip::FlipBwImpl::new());

        dev.register_fw_impl(
            "triangular_l_fw_impl",
            triangular_l::TriangularLFwImpl::new(),
        );
        dev.register_bw_impl(
            "triangular_l_bw_impl",
            triangular_l::TriangularLBwImpl::new(),
        );

        dev.register_fw_impl(
            "triangular_u_fw_impl",
            triangular_u::TriangularUFwImpl::new(),
        );
        dev.register_bw_impl(
            "triangular_u_bw_impl",
            triangular_u::TriangularUBwImpl::new(),
        );

        // ramp

        dev.register_fw_impl("prelu_fw_impl", prelu::PReLUFwImpl::new());
        dev.register_bw_impl("prelu_bw_impl", prelu::PReLUBwImpl::new());

        dev.register_fw_impl("elu_fw_impl", elu::EluFwImpl::new());
        dev.register_bw_impl("elu_bw_impl", elu::EluBwImpl::new());

        // manipulation

        dev.register_fw_impl("slice_fw_impl", slice::SliceFwImpl::new());
        dev.register_bw_impl("slice_bw_impl", slice::SliceBwImpl::new());

        dev.register_fw_impl("pick_fw_impl", pick::PickFwImpl::new());
        dev.register_bw_impl("pick_bw_impl", pick::PickBwImpl::new());

        dev.register_fw_impl("concat_fw_impl", concat::ConcatFwImpl::new());

        // batch

        dev.register_fw_impl(
            "batch_concat_fw_impl",
            batch_concat::BatchConcatFwImpl::new(),
        );

        dev.register_fw_impl("batch_pick_fw_impl", batch_pick::BatchPickFwImpl::new());
        dev.register_bw_impl("batch_pick_bw_impl", batch_pick::BatchPickBwImpl::new());

        dev.register_fw_impl("batch_slice_fw_impl", batch_slice::BatchSliceFwImpl::new());
        dev.register_bw_impl("batch_slice_bw_impl", batch_slice::BatchSliceBwImpl::new());

        dev.register_fw_impl("batch_sum_fw_impl", batch_sum::BatchSumFwImpl::new());

        dev
    }
}
