#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;

use std::collections::HashMap;


#[test]
fn test_parse_YAML() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let mut modeldata_string:  HashMap<String, String> = HashMap::new();
    let mut modeldata_float:  HashMap<String, f64> = HashMap::new();
    let mut modeldata_int:  HashMap<String, u64> = HashMap::new();

    RayBNN_DataLoader::Model::YAML::read(
        "./test_data/test.yaml",
    
        &mut modeldata_string,
        &mut modeldata_float,
        &mut modeldata_int,
    );


}
