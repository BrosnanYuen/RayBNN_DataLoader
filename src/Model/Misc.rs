use arrayfire;


use rayon::prelude::*;


/*


let mut modeldata_int: HashMap<String,u64> = HashMap::new();
let mut modeldata_float: HashMap<String,f64> = HashMap::new();
let mut modeldata_string: HashMap<String,String> = HashMap::new();

modeldata_int.insert("neuron_size".to_string(), netdata.neuron_size.clone());
modeldata_int.insert("input_size".to_string(), netdata.input_size.clone());
modeldata_int.insert("output_size".to_string(), netdata.output_size.clone());
modeldata_int.insert("proc_num".to_string(), netdata.proc_num.clone());
modeldata_int.insert("active_size".to_string(), netdata.active_size.clone());
modeldata_int.insert("space_dims".to_string(), netdata.space_dims.clone());
modeldata_int.insert("step_num".to_string(), netdata.step_num.clone());
modeldata_int.insert("batch_size".to_string(), netdata.batch_size.clone());
modeldata_int.insert("del_unused_neuron".to_string(), netdata.del_unused_neuron.clone() as u64);

modeldata_float.insert("time_step".to_string(), netdata.time_step.clone());
modeldata_float.insert("nratio".to_string(), netdata.nratio.clone());
modeldata_float.insert("neuron_std".to_string(), netdata.neuron_std.clone());
modeldata_float.insert("sphere_rad".to_string(), netdata.sphere_rad.clone());
modeldata_float.insert("neuron_rad".to_string(), netdata.neuron_rad.clone());
modeldata_float.insert("con_rad".to_string(), netdata.con_rad.clone());
modeldata_float.insert("init_prob".to_string(), netdata.init_prob.clone());
modeldata_float.insert("add_neuron_rate".to_string(), netdata.add_neuron_rate.clone());
modeldata_float.insert("del_neuron_rate".to_string(), netdata.del_neuron_rate.clone());
modeldata_float.insert("center_const".to_string(), netdata.center_const.clone());
modeldata_float.insert("spring_const".to_string(), netdata.spring_const.clone());
modeldata_float.insert("repel_const".to_string(), netdata.repel_const.clone());



write(
    "/tmp/network/model.yaml",

    &modeldata_string,
    &modeldata_float,
    &modeldata_int,
);



*/