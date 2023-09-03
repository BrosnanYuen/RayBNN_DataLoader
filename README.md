# RayBNN_DataLoader
Data Loader for RayBNN

Read CSV, numpy, and binary files to Rust vectors of f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64

Read CSV, numpy, and binary files to Arrayfire GPU arrays of f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64


# List of Examples



# Read a CSV file to a floating point 64 bit CPU Vector
The vector is completely flat
```
let (mut cpu_vector,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<f64>(
    "./test_data/read_test.dat"
);
```

# Read a CSV file to a integer 64 bit CPU Vector
The vector is completely flat
```
let (mut cpu_vector,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<i64>(
    "./test_data/read_test2.dat"
);
```

# Read a CSV file to a floating point 64 bit arrayfire
The array is 2D existing in GPU or OpenCL
```
let read_test = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    "./test_data/read_test.dat"
);
```
