use arrayfire;


use rayon::prelude::*;


use std::fs;

use std::collections::HashMap;


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

