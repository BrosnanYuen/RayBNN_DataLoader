use arrayfire;


use rayon::prelude::*;


use std::fs;

use std::collections::HashMap;

use std::fs::File;
use std::io::Write;

pub fn str_to_vec_cpu<Z: std::str::FromStr>(
	instr: &str
) -> Vec<Z>  {

	let mut vecZ: Vec<Z> = Vec::new();


	let mut newline = instr.replace("\n", "");
	newline = newline.replace(" ", "");

	if newline.len() > 0
	{
		let strvec: Vec<&str> = newline.split(",").collect();
		
		for i in 0..strvec.len()
		{
            match strvec[i as usize].parse::<Z>() {
                Ok(n) => vecZ.push(n),
                Err(..) => {}
            }
		}
	}

	vecZ
}





pub fn file_to_vec_cpu<Z: std::str::FromStr + Send + Sync>(
	filename: &str
) -> (Vec<Z>, HashMap<&str,u64>)  {

    let mut metadata = HashMap::new();


	let contents = fs::read_to_string(filename).expect("error");

	let tmp = contents.par_split('\n').map(str_to_vec_cpu );

    metadata.insert("dims", 2);

    let dim0 = (tmp.clone().count() as u64) - 1;
    metadata.insert("dim0", dim0.clone());

    let result: Vec<Z> = tmp.flatten_iter().collect();

    let dim1 = (result.len() as u64)/dim0;

    metadata.insert("dim1", dim1.clone());
    
    (result,metadata)
}




pub fn file_to_arrayfire<Z: std::str::FromStr + arrayfire::HasAfEnum + Send + Sync>(
	filename: &str,
	) -> arrayfire::Array<Z>  {

	let (vector,metadata) = file_to_vec_cpu::<Z>(filename);

    let dim0 = metadata[&"dim0"];
    let dim1 = metadata[&"dim1"];

	let arr_dims = arrayfire::Dim4::new(&[dim1, dim0, 1, 1]);
	let outarr = arrayfire::Array::new(&vector, arr_dims);


	arrayfire::transpose(&outarr,false)
}




pub fn vec_cpu_to_str<Z: arrayfire::HasAfEnum>(
	invec: &Vec<Z>
	) -> String  {

	let mut s0 = format!("{:?}",invec.clone());
	s0 = s0.replace("[", "");
	s0 = s0.replace("]", "");
	s0 = s0.replace(" ", "");

	s0
}





pub fn write_vec_cpu_to_csv<Z: arrayfire::HasAfEnum>(
	filename: &str,
	invec: &Vec<Z>,
	metadata: HashMap<&str,u64>,
	)
{

	let mut wtr0 = vec_cpu_to_str::<Z>(invec);

	
	let mut file0 = File::create(filename).unwrap();
	writeln!(file0, "{}", wtr0);
}



