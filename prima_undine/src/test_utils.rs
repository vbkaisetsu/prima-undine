macro_rules! assert_vector_ulps_eq {
    ( $lhs:expr, $rhs:expr $(, $opt:ident = $val:expr)*) => {
        if $lhs.len() != $rhs.len() {
            println!("lhs = {:?}, rhs = {:?}", $lhs, $rhs);
        }
        assert!($lhs.len() == $rhs.len());
        for i in 0..$lhs.len() {
            if !approx::ulps_eq!($lhs[i], $rhs[i] $(, $opt = $val)*) {
                println!("lhs = {:?}, rhs = {:?}", $lhs, $rhs);
            }
            assert!(approx::ulps_eq!($lhs[i], $rhs[i] $(, $opt = $val)*));
        }
    };
}

macro_rules! generate_fw_testset {
    ( $x:expr, $x_f:expr ) => {{
        $x.iter()
            .map(|&x| $x_f(x as f64) as f32)
            .collect::<Vec<f32>>()
    }};
}

macro_rules! generate_bw_testset {
    ( $x:expr, $gy:expr, $y_f:expr, $gx_f:expr ) => {{
        let x_f64 = $x.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let y_f64 = x_f64.iter().map(|&x| $y_f(x)).collect::<Vec<f64>>();
        let gy_f64 = $gy.iter().map(|&gy| gy as f64).collect::<Vec<f64>>();
        let y = y_f64.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let gx = x_f64
            .iter()
            .zip(y_f64.iter())
            .zip(gy_f64.iter())
            .map(|((&x, &y), &gy)| $gx_f(x, y, gy) as f32)
            .collect::<Vec<f32>>();
        (y, gx)
    }};
}
