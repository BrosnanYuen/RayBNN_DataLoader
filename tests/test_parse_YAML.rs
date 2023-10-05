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

    assert!(modeldata_string.contains_key("model_filename"));
    assert_eq!(modeldata_string["model_filename"].clone(), "/opt/test/");


    assert!(modeldata_string.contains_key("data_filename"));
    assert_eq!(modeldata_string["data_filename"].clone(), "/tmp/test/");


    assert!(modeldata_float.contains_key("version"));
    assert_eq!(modeldata_float["version"].clone(), 1.5);

    assert!(modeldata_float.contains_key("add_ratio"));
    assert_eq!(modeldata_float["add_ratio"].clone(), 4.7);


    assert!(modeldata_int.contains_key("active_size"));
    assert_eq!(modeldata_int["active_size"].clone(), 1552);

    assert!(modeldata_int.contains_key("input_size"));
    assert_eq!(modeldata_int["input_size"].clone(), 15);



}
