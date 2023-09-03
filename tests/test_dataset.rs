#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



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


    let inx_cpu: [u32; 5] = [0, 4, 1, 9, 3];

    let instr: String = String::from("0, 4, 1, 9, 3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::CSV::str_to_vec_cpu::<u32>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);



    let instr: String = String::from("aaa, bbb, ccc");
    let outvec_cpu = RayBNN_DataLoader::Dataset::CSV::str_to_vec_cpu::<u64>(&instr);
    assert_eq!(outvec_cpu.len(), 0);









    let mut read_test = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<f64>(
    	"./test_data/read_test.dat"
    );


	let mut read_act: Vec<f64> = vec![
		-0.004866,-0.0018368,0.0049874,0.0023202,-4.9179e-05,-0.0033278,
		-0.0082358,-0.006966,-0.0033703,0.0038264,0.0047417,0.0017643,
		0.0013178,-0.00061582,0.008669,3.5362e-05,-0.00080587,0.0044014,
		0.00012772,-0.00088359,-0.0072174,0.0043621,0.0046395,2.6826e-05
	];

	read_act = read_act.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();

	read_test = read_test.par_iter().map(|x|  (x * 1.0e10).round() / 1.0e10 ).collect::<Vec<f64>>();


	assert_eq!(read_test, read_act);









    let mut read_test2 = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<i64>(
    	"./test_data/read_test2.dat"
    );

    let mut read_act2: Vec<i64> = vec![
        233,-4233,234,631,
        24, 222,-1,23,
        45,3,1,100,
        -2,3,  5,61,
        344,222,33,-10
    ];
    assert_eq!(read_test2, read_act2);

}
