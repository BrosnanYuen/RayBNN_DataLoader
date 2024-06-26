use arrayfire;


use rayon::prelude::*;


use std::fs;

use std::collections::HashMap;

use std::fs::File;
use std::io::Write;



use std::io::{self, BufRead};
use std::path::Path;


use crate::Model::YAML::read;
use crate::Dataset::CSV::file_to_arrayfire;

/*








use std::fs;


use std::fs::File;
use std::io::Write;


use rayon::prelude::*;

use std::io::{self, BufRead};
use std::path::Path;


pub fn write(
	filename: &str,

    modeldata_string: &HashMap<String, String>,
	modeldata_float: &HashMap<String, f64>,
    modeldata_int: &HashMap<String, u64>,
	)
{
	let mut strvec: Vec<String> = Vec::new();

	for (key, value) in modeldata_int {
		let tmp = format!("{}: {}\n", key.clone(), value.clone());
		strvec.push(tmp.clone());
	}

	for (key, value) in modeldata_float {
		let tmp = format!("{}: {}\n", key.clone(), value.clone());
		strvec.push(tmp.clone());
	}

	for (key, value) in modeldata_string {
		let tmp = format!("{}: '{}'\n", key.clone(), value.clone());
		strvec.push(tmp.clone());
	}

	let tmpstr: String = strvec.into_par_iter().collect::<String>();


	let mut file0 = File::create(filename).unwrap();
	writeln!(file0, "{}", tmpstr);
}





pub fn vec_cpu_to_str<Z: arrayfire::HasAfEnum + Sync + Send>(
	invec: &[Z]
	) -> String  {

	let mut s0 = format!("{:?}",invec.clone());
	s0 = s0.replace("[", "");
	s0 = s0.replace("]", "");
	s0 = s0.replace(" ", "");

	s0
}






pub fn write_vec_cpu_to_csv<Z: arrayfire::HasAfEnum + Sync + Send>(
	filename: &str,
	invec: &Vec<Z>,
	metadata: &HashMap<String,u64>,
	)
{

	let dim0 = metadata["dim0"];
    let dim1 = metadata["dim1"];

	//let mut wtr0 = vec_cpu_to_str::<Z>(invec);
	let mut tmp: String = invec.par_chunks_exact(dim1 as usize).map(vec_cpu_to_str ).map(|x| x+"\n").collect();
	tmp.pop();


	let mut file0 = File::create(filename).unwrap();
	writeln!(file0, "{}", tmp);
}







pub fn write_arrayfire_to_csv<Z: arrayfire::HasAfEnum + Sync + Send>(
	filename: &str,
	arr: &arrayfire::Array<Z>
	)
{

	let mut metadata: HashMap<String,u64> = HashMap::new();

	metadata.insert("dim0".to_string(), arr.dims()[0]);
    metadata.insert("dim1".to_string(), arr.dims()[1]);


	let tmp = arrayfire::transpose(arr, false);

	let mut invec = vec!(Z::default();tmp.elements());
	tmp.host(&mut invec);

	write_vec_cpu_to_csv::<Z>(
		filename,
		&invec,
		&metadata
	);
}











let mut modeldata_int: HashMap<String,u64> = HashMap::new();
let mut modeldata_float: HashMap<String,f64> = HashMap::new();
let mut modeldata_string: HashMap<String,String> = HashMap::new();

modeldata_int.insert("neuron_size".to_string(), netdata.neuron_size.clone());
modeldata_int.insert("input_size".to_string(), netdata.input_size.clone());
modeldata_int.insert("output_size".to_string(), netdata.output_size.clone());
modeldata_int.insert("proc_num".to_string(), netdata.proc_num.clone());
modeldata_int.insert("active_size".to_string(), netdata.active_size.clone());
modeldata_int.insert("space_dims".to_string(), netdata.space_dims.clone());
modeldata_int.insert("step_num".to_string(), netdata.step_num.clone());
modeldata_int.insert("batch_size".to_string(), netdata.batch_size.clone());
modeldata_int.insert("del_unused_neuron".to_string(), netdata.del_unused_neuron.clone() as u64);

modeldata_float.insert("time_step".to_string(), netdata.time_step.clone());
modeldata_float.insert("nratio".to_string(), netdata.nratio.clone());
modeldata_float.insert("neuron_std".to_string(), netdata.neuron_std.clone());
modeldata_float.insert("sphere_rad".to_string(), netdata.sphere_rad.clone());
modeldata_float.insert("neuron_rad".to_string(), netdata.neuron_rad.clone());
modeldata_float.insert("con_rad".to_string(), netdata.con_rad.clone());
modeldata_float.insert("init_prob".to_string(), netdata.init_prob.clone());
modeldata_float.insert("add_neuron_rate".to_string(), netdata.add_neuron_rate.clone());
modeldata_float.insert("del_neuron_rate".to_string(), netdata.del_neuron_rate.clone());
modeldata_float.insert("center_const".to_string(), netdata.center_const.clone());
modeldata_float.insert("spring_const".to_string(), netdata.spring_const.clone());
modeldata_float.insert("repel_const".to_string(), netdata.repel_const.clone());



let dir_path = "/tmp/network_batch21/";

let filename = format!("{}/model.yaml",dir_path);
write(
    &filename,

    &modeldata_string,
    &modeldata_float,
    &modeldata_int,
);

let filename = format!("{}/WValues.csv",dir_path);
write_arrayfire_to_csv(&filename,&WValues);

let filename = format!("{}/WRowIdxCSR.csv",dir_path);
write_arrayfire_to_csv(&filename,&WRowIdxCSR);

let filename = format!("{}/WColIdx.csv",dir_path);
write_arrayfire_to_csv(&filename,&WColIdx);

let filename = format!("{}/H.csv",dir_path);
write_arrayfire_to_csv(&filename,&H);

let filename = format!("{}/A.csv",dir_path);
write_arrayfire_to_csv(&filename,&A);

let filename = format!("{}/B.csv",dir_path);
write_arrayfire_to_csv(&filename,&B);

let filename = format!("{}/C.csv",dir_path);
write_arrayfire_to_csv(&filename,&C);

let filename = format!("{}/D.csv",dir_path);
write_arrayfire_to_csv(&filename,&D);

let filename = format!("{}/E.csv",dir_path);
write_arrayfire_to_csv(&filename,&E);

let filename = format!("{}/glia_pos.csv",dir_path);
write_arrayfire_to_csv(&filename,&glia_pos);

let filename = format!("{}/neuron_pos.csv",dir_path);
write_arrayfire_to_csv(&filename,&neuron_pos);

let filename = format!("{}/neuron_idx.csv",dir_path);
write_arrayfire_to_csv(&filename,&neuron_idx);



*/




pub fn read_network_dir<Z: std::str::FromStr + arrayfire::FloatingPoint  + Send + Sync >(
	dir_path: &str,

    modeldata_string: &mut HashMap<String, String>,
	modeldata_float: &mut HashMap<String, f64>,
    modeldata_int: &mut HashMap<String, u64>,


	WValues: &mut arrayfire::Array<Z>,
	WRowIdxCSR: &mut arrayfire::Array<i32>,
	WColIdx: &mut arrayfire::Array<i32>,
	H: &mut arrayfire::Array<Z>,
	A: &mut arrayfire::Array<Z>,
	B: &mut arrayfire::Array<Z>,
	C: &mut arrayfire::Array<Z>,
	D: &mut arrayfire::Array<Z>,
	E: &mut arrayfire::Array<Z>,
	glia_pos: &mut arrayfire::Array<Z>,
	neuron_pos: &mut arrayfire::Array<Z>,
	neuron_idx: &mut arrayfire::Array<i32>
	){


    let filename = format!("{}/model.yaml",dir_path);
    read(
        &filename, 
        modeldata_string, 
        modeldata_float, 
        modeldata_int
    );

	let filename = format!("{}/WValues.csv",dir_path);
	*WValues = file_to_arrayfire::<Z>(&filename);

	let filename = format!("{}/WRowIdxCSR.csv",dir_path);
	*WRowIdxCSR = file_to_arrayfire::<i32>(&filename);

	let filename = format!("{}/WColIdx.csv",dir_path);
	*WColIdx = file_to_arrayfire::<i32>(&filename);

	let filename = format!("{}/H.csv",dir_path);
	*H = file_to_arrayfire::<Z>(&filename);

	let filename = format!("{}/A.csv",dir_path);
	*A = file_to_arrayfire::<Z>(&filename);

	let filename = format!("{}/B.csv",dir_path);
	*B = file_to_arrayfire::<Z>(&filename);

	let filename = format!("{}/C.csv",dir_path);
	*C = file_to_arrayfire::<Z>(&filename);

	let filename = format!("{}/D.csv",dir_path);
	*D = file_to_arrayfire::<Z>(&filename);

	let filename = format!("{}/E.csv",dir_path);
	*E = file_to_arrayfire::<Z>(&filename);

	let filename = format!("{}/glia_pos.csv",dir_path);
	*glia_pos = file_to_arrayfire::<Z>(&filename);

	let filename = format!("{}/neuron_pos.csv",dir_path);
	*neuron_pos = file_to_arrayfire::<Z>(&filename);

	let filename = format!("{}/neuron_idx.csv",dir_path);
	*neuron_idx = file_to_arrayfire::<i32>(&filename);

}




