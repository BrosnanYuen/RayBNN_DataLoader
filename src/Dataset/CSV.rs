use arrayfire;


use rayon::prelude::*;


use std::fs;




pub fn str_to_vec_cpu(
	instr: &str
) -> Vec<f64>  {

	let mut vecf64: Vec<f64> = Vec::new();


	let mut newline = instr.replace("\n", "");
	newline = newline.replace(" ", "");

	if newline.len() > 0
	{
		let strvec: Vec<&str> = newline.split(",").collect();
		let ssize: u64 = strvec.len() as u64;

		
		for i in 0u64..ssize
		{
			let value:f64 = strvec[i as usize].parse::<f64>().unwrap();
			vecf64.push(value);
		}
	}

	vecf64
}




