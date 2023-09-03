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
    metadata.insert("dim0", (tmp.clone().count() as u64) - 1);
    
    (tmp.flatten_iter().collect(),metadata)
}


