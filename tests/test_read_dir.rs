#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;

use std::collections::HashMap;


#[test]
fn test_read_dir() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);
    
    let mut modeldata_string:  HashMap<String, String> = HashMap::new();
    let mut modeldata_float:  HashMap<String, f64> = HashMap::new();
    let mut modeldata_int:  HashMap<String, u64> = HashMap::new();
    

    RayBNN_DataLoader::Model::Misc::read_network_dir("./test_data/network_batch21/", 
        modeldata_string, 
        modeldata_float, 
        modeldata_int, 
        WValues, 
        WRowIdxCSR, 
        WColIdx, 
        H, 
        A, 
        B, 
        C, 
        D, 
        E, 
        glia_pos, 
        neuron_pos, 
        neuron_idx
    );

}