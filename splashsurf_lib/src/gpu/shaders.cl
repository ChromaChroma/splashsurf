
kernel void fill_arr ( global int* out ){
    const size_t id = get_global_id(0);
    out[id] = id;
}