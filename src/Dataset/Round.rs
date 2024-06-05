
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







pub fn rvector(
	input: &Vec<f64>,
	decimal: u64
	) -> Vec<f64>  {


	input.par_iter().map(|&x|  rscalar(x , decimal) ).collect::<Vec<f64>>()
}









