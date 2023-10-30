#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;

use rand::{distributions::Standard, Rng};
use std::collections::HashMap;

#[test]
fn test_dataset_csv3() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let mut neuron_pos = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    	"./test_data/neuron_pos3.csv",
    );


	arrayfire::print_gen("neuron_pos".to_string(), &neuron_pos,Some(6));

}
