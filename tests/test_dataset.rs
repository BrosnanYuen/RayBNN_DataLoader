#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_dataset() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let inx_cpu: [f64; 5] = [0.0, -4.1, 1.7, -0.9, 0.3];

    let instr: String = String::from("0.0, -4.1,1.7, -0.9, 0.3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::CSV::str_to_vec_cpu::<f64>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);



    let inx_cpu: [i32; 5] = [0, -4, 1, -9, 3];

    let instr: String = String::from("0,-4, 1,   -9,  3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::CSV::str_to_vec_cpu::<i32>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);




    let inx_cpu: [f32; 5] = [0.0, -4.1, 1.7, -0.9, 0.3];

    let instr: String = String::from("0.0, -4.1,1.7, -0.9, 0.3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::CSV::str_to_vec_cpu::<f32>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);





    let inx_cpu: [i64; 5] = [0, -4, 1, -9, 3];

    let instr: String = String::from("0,-4, 1,   -9,  3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::CSV::str_to_vec_cpu::<i64>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);



    let inx_cpu: [u64; 5] = [0, 4, 1, 9, 3];

    let instr: String = String::from("0, 4, 1, 9, 3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::CSV::str_to_vec_cpu::<u64>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);



}
