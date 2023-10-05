use arrayfire;


use rayon::prelude::*;


use std::fs;

use std::collections::HashMap;

use std::fs::File;
use std::io::Write;


use nohash_hasher;






pub fn read<Z: arrayfire::RealFloating, V: arrayfire::Scanable>(
	filename: &str,

    modeldata_string: &mut HashMap<String, String>,
	modeldata_float: &mut HashMap<String, Z>,
    modeldata_int: &mut HashMap<String, V>,
	)
{



}












