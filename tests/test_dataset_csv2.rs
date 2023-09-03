#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_dataset_csv2() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    let inx_cpu: Vec<i32> = vec![0, -4, 1, -9,   3];

    let outstr: String = String::from("0,-4,1,-9,3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::CSV::vec_cpu_to_str::<i32>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);


}
