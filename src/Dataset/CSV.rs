use arrayfire;


use rayon::prelude::*;


use std::fs;




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
) -> Vec<Z>  {
	let contents = fs::read_to_string(filename).expect("error");

	contents.par_split('\n').map(str_to_vec_cpu ).flatten_iter().collect()
}


