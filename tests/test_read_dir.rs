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
    
	let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);

    
	let mut glia_pos = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut neuron_pos = arrayfire::constant::<f64>(0.0,temp_dims);



	
	let mut H = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut A = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut B = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut C = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut D = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut E = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut neuron_idx = arrayfire::constant::<i32>(0,temp_dims);





	let mut WValues = arrayfire::constant::<f64>(0.0,temp_dims);
	let mut WRowIdxCSR = arrayfire::constant::<i32>(0,temp_dims);
	let mut WColIdx = arrayfire::constant::<i32>(0,temp_dims);



    RayBNN_DataLoader::Model::Network::read_network_dir("./test_data/network_batch21/", 
        &mut modeldata_string, 
        &mut modeldata_float, 
        &mut modeldata_int, 
        &mut WValues, 
        &mut WRowIdxCSR, 
        &mut WColIdx, 
        &mut H, 
        &mut A, 
        &mut B, 
        &mut C, 
        &mut D, 
        &mut E, 
        &mut glia_pos, 
        &mut neuron_pos, 
        &mut neuron_idx
    );

    assert!(modeldata_int.contains_key("output_size"));
    assert_eq!(modeldata_int["output_size"].clone(), 3);

    assert!(modeldata_int.contains_key("neuron_size"));
    assert_eq!(modeldata_int["neuron_size"].clone(), 600);

    assert!(modeldata_int.contains_key("step_num"));
    assert_eq!(modeldata_int["step_num"].clone(), 10);

    assert!(modeldata_int.contains_key("proc_num"));
    assert_eq!(modeldata_int["proc_num"].clone(), 4);

    assert!(modeldata_int.contains_key("input_size"));
    assert_eq!(modeldata_int["input_size"].clone(), 4);

    assert!(modeldata_int.contains_key("space_dims"));
    assert_eq!(modeldata_int["space_dims"].clone(), 3);

    assert!(modeldata_int.contains_key("batch_size"));
    assert_eq!(modeldata_int["batch_size"].clone(), 105);

    assert!(modeldata_int.contains_key("active_size"));
    assert_eq!(modeldata_int["active_size"].clone(), 25);

    assert!(modeldata_float.contains_key("spring_const"));
    assert_eq!(modeldata_float["spring_const"].clone(), 0.01);

    assert!(modeldata_float.contains_key("con_rad"));
    assert_eq!(modeldata_float["con_rad"].clone(), 0.6);


    
    assert_eq!(WValues.dims()[0], 3136);
    assert_eq!(WValues.dims()[1], 1);


    assert_eq!(WRowIdxCSR.dims()[0], 601);
    assert_eq!(WRowIdxCSR.dims()[1], 1);

    assert_eq!(WColIdx.dims()[0], 3136);
    assert_eq!(WColIdx.dims()[1], 1);

    assert_eq!(A.dims()[0], 600);
    assert_eq!(A.dims()[1], 1);

    assert_eq!(B.dims()[0], 600);
    assert_eq!(B.dims()[1], 1);

    assert_eq!(C.dims()[0], 600);
    assert_eq!(C.dims()[1], 1);

    assert_eq!(glia_pos.dims()[0], 197);
    assert_eq!(glia_pos.dims()[1], 3);

    assert_eq!(neuron_pos.dims()[0], 203);
    assert_eq!(neuron_pos.dims()[1], 3);

    assert_eq!(neuron_idx.dims()[0], 203);
    assert_eq!(neuron_idx.dims()[1], 1);


}