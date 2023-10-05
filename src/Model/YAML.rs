use arrayfire;


use rayon::prelude::*;


use std::fs;

use std::collections::HashMap;

use std::fs::File;
use std::io::Write;


use nohash_hasher;






pub fn read<Z: arrayfire::HasAfEnum + Sync + Send>(
	filename: &str,

	modeldata_float: &mut HashMap<&str,f64>,
    modeldata_int: &mut HashMap<&str,f64>,
	)
{


}












