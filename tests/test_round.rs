#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;

use std::collections::HashMap;


#[test]
fn test_round() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let value =  RayBNN_DataLoader::Dataset::Round::rscalar(31.12356234, 4);

    assert_eq!(value, 31.1236);








    let value =  RayBNN_DataLoader::Dataset::Round::rscalar(0.231543275, 6);

    assert_eq!(value, 0.231543);







    let value =  RayBNN_DataLoader::Dataset::Round::rscalar(254123.21232, 2);

    assert_eq!(value, 254123.21);














    let vec0: Vec<f64> = vec![324.34156354, 7.34398734, 0.231243538, -0.3412443];
    let vec1 =  RayBNN_DataLoader::Dataset::Round::rvector(&vec0, 4);

    let vec2: Vec<f64> = vec![324.3416, 7.344, 0.2312, -0.3412];
    assert_eq!(vec1, vec2 );






    let max_iter = 1000000;
    let mut total_sum = 0.0;
    for zz in 0..max_iter
    {
        let rand_number = RayBNN_DataLoader::Dataset::Random::single_random_uniform();

        assert!(rand_number >= 0.0 );
        assert!(1.0 >= rand_number);

        total_sum = total_sum + rand_number;
    }

    assert!( (max_iter as f64)/1.8  >= total_sum);
    assert!(total_sum >= (max_iter as f64)/2.2 );

    //println!("total_sum {}", total_sum);








    let max_num = 6;
    let max_iter = max_num*10000;

    let mut  zero_vec = vec![0u64; max_num as usize];

    for qq in 0..max_iter
    {

        let rand_num =  RayBNN_DataLoader::Dataset::Random::random_uniform_range(max_num);

        zero_vec[rand_num as usize] = zero_vec[rand_num as usize] + 1;
        //println!("rand_num {}",rand_num);

        assert!( rand_num <  max_num  );
        assert!( 0 <= max_num );
    }

    for vv in 0..max_num
    {
        assert!( zero_vec[vv as usize]  >  3000 );
        //println!("vv {}  {}",vv, zero_vec[vv as usize]);
    }

}
