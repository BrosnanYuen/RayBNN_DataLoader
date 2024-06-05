
use rayon::prelude::*;






pub fn single_random_uniform() -> f64
{

	let single_rand_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let singlerand = arrayfire::randu::<f64>(single_rand_dims);

	let mut singlerand_cpu: [f64 ; 1] = [0.0];
	singlerand.host(&mut singlerand_cpu);
	

	return singlerand_cpu[0]  
}









pub fn random_uniform_range(
	max_size: u64,
) -> u64
{

	let rand_number = single_random_uniform();

	let min_idx  =  (rand_number  * (max_size as f64) ) as u64;

	return min_idx 
}







