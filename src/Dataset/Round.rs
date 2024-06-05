
use rayon::prelude::*;


use num_traits;





const TEN: f64 = 10.0;

pub fn rscalar<Z:  num_traits::Float>(
	input: Z,
	decimal: u64
	) -> Z  {

	let places = TEN.powf(decimal as f64);
	let places: Z = num_traits::cast(places).unwrap();

	let ret = (input * places).round() / places;

	ret
}







pub fn rvector<Z:  num_traits::Float + Send + Sync>(
	input: &Vec<Z>,
	decimal: u64
	) -> Vec<Z>  {


	input.par_iter().map(|&x|  rscalar(x , decimal) ).collect::<Vec<Z>>()
}









