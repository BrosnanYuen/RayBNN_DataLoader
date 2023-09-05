use arrayfire;


use rayon::prelude::*;


use std::fs;

use std::collections::HashMap;

use std::fs::File;
use std::io::Write;


use nohash_hasher;

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
	metadata: &HashMap<&str,u64>,
	)
{

	let dim0 = metadata[&"dim0"];
    let dim1 = metadata[&"dim1"];

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

	let mut metadata: HashMap<&str,u64> = HashMap::new();

	metadata.insert("dim0", arr.dims()[0]);
    metadata.insert("dim1", arr.dims()[1]);


	let tmp = arrayfire::transpose(arr, false);

	let mut invec = vec!(Z::default();tmp.elements());
	tmp.host(&mut invec);

	write_vec_cpu_to_csv::<Z>(
		filename,
		&invec,
		&metadata
	);
}










pub fn file_to_hash_cpu<Z: std::str::FromStr + Send + Sync + Clone>(
	filename: &str,
	sample_size: u64,
	batch_size: u64
	) -> nohash_hasher::IntMap<u64, Vec<Z> >  {

	
	

	let (arr,metadata) = file_to_vec_cpu(filename);

	let arr_size = arr.len() as u64;
	let item_num = (arr_size/(sample_size*batch_size));

	let mut lookup: nohash_hasher::IntMap<u64, Vec<Z> >  = nohash_hasher::IntMap::default();
	let mut start:usize = 0;
	let mut end:usize = 0;
	for i in 0..item_num
	{
		start = (i*(sample_size*batch_size)) as usize;
		end = ((i+1)*(sample_size*batch_size)) as usize;
		lookup.insert(i, (&arr[start..end]).to_vec() );
	}

	lookup
}

