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










    let mut metadata: HashMap<&str,u64> = HashMap::new();
    let write_vec: Vec<i32> = vec![
        1,-2,3,-4,
        -5,6,-7,8,
        9,10,11,12,
        13,14,15,16,
        -17,18,-19,20,
        21,-22,23,-24,
        25,26,27,28
    ];

    metadata.insert("dim0", 7);
    metadata.insert("dim1", 4);
	
	RayBNN_DataLoader::Dataset::CSV::write_vec_cpu_to_csv::<i32>(
		"./test_write.csv",
		&write_vec,
        &metadata
	);


    let mut read_test = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<i32>(
    	"./test_write.csv"
    );

    assert_eq!(read_test.dims()[0], 7);
    assert_eq!(read_test.dims()[1], 4);

    std::fs::remove_file("./test_write.csv");

    read_test = arrayfire::sum(&read_test, 0);

    //arrayfire::print_gen("read_test".to_string(), &read_test,Some(6));

    let mut row0_cpu = vec!(i32::default();read_test.elements());
	read_test.host(&mut row0_cpu);

    let row0_act = vec![47,  50,  53,  56];
    assert_eq!(row0_cpu, row0_act);












	let randarrz_dims = arrayfire::Dim4::new(&[5,11,1,1]);
	let randarrz = arrayfire::randn::<f64>(randarrz_dims);

	RayBNN_DataLoader::Dataset::CSV::write_arrayfire_to_csv::<f64>(
		"./randvec.csv",
		&randarrz
	);

    //arrayfire::print_gen("randarrz".to_string(), &randarrz,Some(6));

	let arrfromfile = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
		"./randvec.csv"
    );

    //arrayfire::print_gen("arrfromfile".to_string(), &arrfromfile,Some(6));

	let subarr = randarrz-arrfromfile;
	let absval = arrayfire::abs(&subarr);
	let (r0,r1) = arrayfire::mean_all(&absval);

	assert!(r0 < 1e-6);
	assert!(r1 < 1e-6);


    std::fs::remove_file("./randvec.csv");









	let randarrz_dims = arrayfire::Dim4::new(&[5,11,1,1]);
	let randarrz = arrayfire::randu::<u32>(randarrz_dims);

	RayBNN_DataLoader::Dataset::CSV::write_arrayfire_to_csv::<u32>(
		"./randvec.csv",
		&randarrz
	);

    //arrayfire::print_gen("randarrz".to_string(), &randarrz,Some(6));

	let arrfromfile = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<u32>(
		"./randvec.csv"
    );

    //arrayfire::print_gen("arrfromfile".to_string(), &arrfromfile,Some(6));

	let subarr = randarrz-arrfromfile;
	let absval = arrayfire::abs(&subarr);
	let (r0,r1) = arrayfire::mean_all(&absval);

	assert!(r0 < 1e-6);
	assert!(r1 < 1e-6);


    std::fs::remove_file("./randvec.csv");














	let randarrz_dims = arrayfire::Dim4::new(&[5,11,1,1]);
	let randarrz = arrayfire::randu::<i32>(randarrz_dims);

	RayBNN_DataLoader::Dataset::CSV::write_arrayfire_to_csv::<i32>(
		"./randvec.csv",
		&randarrz
	);

    //arrayfire::print_gen("randarrz".to_string(), &randarrz,Some(6));

	let arrfromfile = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<i32>(
		"./randvec.csv"
    );

    //arrayfire::print_gen("arrfromfile".to_string(), &arrfromfile,Some(6));

	let subarr = randarrz-arrfromfile;
	let absval = arrayfire::abs(&subarr);
	let (r0,r1) = arrayfire::mean_all(&absval);

	assert!(r0 < 1e-6);
	assert!(r1 < 1e-6);


    std::fs::remove_file("./randvec.csv");


}
