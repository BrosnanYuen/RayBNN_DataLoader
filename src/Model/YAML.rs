use arrayfire;


use rayon::prelude::*;


use std::fs;

use std::collections::HashMap;

use std::fs::File;
use std::io::Write;



use std::io::{self, BufRead};
use std::path::Path;


fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}






pub fn read(
	filename: &str,

    modeldata_string: &mut HashMap<String, String>,
	modeldata_float: &mut HashMap<String, f64>,
    modeldata_int: &mut HashMap<String, u64>,
	)
{

	if let Ok(lines) = read_lines(filename) {
        // Consumes the iterator, returns an (Optional) String
        for line in lines {
            if let Ok(data) = line {
				if data.contains("#")
				{
					continue;
				}

				if data.contains("'")
				{
					let datasplit: Vec<&str> = data.split(":").collect();
					let key = datasplit[0].clone().to_string();

					let mut value = datasplit[1].clone().to_string();

					let datasplit: Vec<&str> = value.split("'").collect();

					value = datasplit[1].clone().to_string();

					println!("value V{}V",value);
				}
				else if data.contains(".")
				{

				}
				else 
				{
					
				}


            }
        }
    }

}












