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
fn test_dataset_csv2() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    let inx_cpu: Vec<i32> = vec![0, -4, 1, -9,   3];

    let outstr: String = String::from("0,-4,1,-9,3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::CSV::vec_cpu_to_str::<i32>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);





    let inx_cpu: Vec<i64> = vec![0, -4, 1, -9,   3];

    let outstr: String = String::from("0,-4,1,-9,3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::CSV::vec_cpu_to_str::<i64>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);



    let inx_cpu: Vec<f64> = vec![0.0, -4.1,1.7, -0.9, 0.3];

    let outstr: String = String::from("0.0,-4.1,1.7,-0.9,0.3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::CSV::vec_cpu_to_str::<f64>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);



    let inx_cpu: Vec<f32> = vec![0.0, -4.1,1.7, -0.9, 0.3];

    let outstr: String = String::from("0.0,-4.1,1.7,-0.9,0.3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::CSV::vec_cpu_to_str::<f32>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);








    let inx_cpu: Vec<u32> = vec![0, 4, 1, 9,   3];

    let outstr: String = String::from("0,4,1,9,3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::CSV::vec_cpu_to_str::<u32>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);



    let inx_cpu: Vec<u64> = vec![0, 4, 1, 9,   3];

    let outstr: String = String::from("0,4,1,9,3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::CSV::vec_cpu_to_str::<u64>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);








    let mut metadata: HashMap<&str,u64> = HashMap::new();
	let randvec: Vec<i32> = rand::thread_rng().sample_iter(Standard).take(3*11).collect();

    metadata.insert("dim0", 11);
    metadata.insert("dim1", 3);
	
	RayBNN_DataLoader::Dataset::CSV::write_vec_cpu_to_csv::<i32>(
		"./randvec2.csv",
		&randvec,
        &metadata
	);


    let (mut read_test2,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<i32>(
    	"./randvec2.csv"
    );

    assert_eq!(metadata[&"dim0"], 11);
    assert_eq!(metadata[&"dim1"], 3);

    assert_eq!(randvec,read_test2);

    std::fs::remove_file("./randvec2.csv");











    let mut metadata: HashMap<&str,u64> = HashMap::new();
	let randvec: Vec<u32> = rand::thread_rng().sample_iter(Standard).take(3*11).collect();

    metadata.insert("dim0", 11);
    metadata.insert("dim1", 3);
	
	RayBNN_DataLoader::Dataset::CSV::write_vec_cpu_to_csv::<u32>(
		"./randvec2.csv",
		&randvec,
        &metadata
	);


    let (mut read_test2,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<u32>(
    	"./randvec2.csv"
    );

    assert_eq!(metadata[&"dim0"], 11);
    assert_eq!(metadata[&"dim1"], 3);

    assert_eq!(randvec,read_test2);

    std::fs::remove_file("./randvec2.csv");














    let mut metadata: HashMap<&str,u64> = HashMap::new();
	let randvec: Vec<f32> = rand::thread_rng().sample_iter(Standard).take(3*11).collect();

    metadata.insert("dim0", 11);
    metadata.insert("dim1", 3);
	
	RayBNN_DataLoader::Dataset::CSV::write_vec_cpu_to_csv::<f32>(
		"./randvec2.csv",
		&randvec,
        &metadata
	);


    let (mut read_test2,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<f32>(
    	"./randvec2.csv"
    );

    assert_eq!(metadata[&"dim0"], 11);
    assert_eq!(metadata[&"dim1"], 3);

    assert_eq!(randvec,read_test2);

    std::fs::remove_file("./randvec2.csv");
}
