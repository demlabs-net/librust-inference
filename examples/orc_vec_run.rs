//! Арбитр Ф8.1g: компилирует ЭТАЛОННЫЙ исходник vec-кернела (fa.metal оракула,
//! FC спец. как nsg=1/nwg=32/ns10=256/has_kvpad) и гоняет на входах оракула.
//! Совпал с y — расход в НАШЕМ исходнике; нет — в бинаре оракула.
#![cfg(feature = "metal")]
use metal::*;

fn main() {
    let device = Device::system_default().unwrap();
    let src = include_str!("/tmp/llama_ref/orc_vec_run.metal");
    let opts = CompileOptions::new();
    let lib = device
        .new_library_with_source(src, &opts)
        .expect("compile oracle source");
    let func = lib.get_function("orc_vec", None).unwrap();
    let pipe = device.new_compute_pipeline_state_with_function(&func).unwrap();

    let q = std::fs::read("/tmp/llama_ref/dec_q.bin").unwrap();
    let k = std::fs::read("/tmp/llama_ref/dec_k.bin").unwrap();
    let v = std::fs::read("/tmp/llama_ref/dec_v.bin").unwrap();
    let ywant = std::fs::read("/tmp/llama_ref/dec_y.bin").unwrap();
    let tkv = k.len() / (2 * 128 * 4);
    let nh = 32usize;
    let hd = 128usize;
    // f16 KV
    let kf: Vec<half::f16> = q.chunks_exact(4).map(|c| half::f16::from_f32(f32::from_le_bytes(c.try_into().unwrap()))).collect();
    let _ = kf;
    let k16: Vec<half::f16> = k.chunks_exact(4).map(|c| half::f16::from_f32(f32::from_le_bytes(c.try_into().unwrap()))).collect();
    let v16: Vec<half::f16> = v.chunks_exact(4).map(|c| half::f16::from_f32(f32::from_le_bytes(c.try_into().unwrap()))).collect();
    let qf: Vec<f32> = q.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let ywant: Vec<f32> = ywant.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    let mb = |data: &[u8]| device.new_buffer_with_data(data.as_ptr().cast(), data.len() as u64, MTLResourceOptions::StorageModeShared);
    let qb = mb(bytemuck_f32(&qf));
    let kb = mb(bytemuck_f16(&k16));
    let vb = mb(bytemuck_f16(&v16));
    // pad buffer: nb11*C*nkv*2 + 32*tq*2 bytes (mask absent)
    let nb11 = 2 * 128 * 2u64;
    let pad_bytes = nb11 * 32 * 2 + 32 * 1 * 2;
    let pad = device.new_buffer(pad_bytes, MTLResourceOptions::StorageModeShared);
    let nrows = 1usize * nh;
    let tmp = device.new_buffer((nrows * (hd * 32 + 64) * 4) as u64, MTLResourceOptions::StorageModeShared);
    let yb = device.new_buffer((nh * hd * 4) as u64, MTLResourceOptions::StorageModeShared);

    #[repr(C)]
    struct A {
        ne01: i32, ne02: i32, ne03: i32,
        nb01: u64, nb02: u64, nb03: u64,
        ne11: i32, ne12: i32, ne13: i32,
        ns10: i32,
        nb11: u64, nb12: u64, nb13: u64,
        ns20: i32,
        nb21: u64, nb22: u64, nb23: u64,
        ne31: i32, ne32: i32, ne33: i32,
        nb31: u64, nb32: u64, nb33: u64,
        ne1: i32, ne2: i32, ne3: i32,
        scale: f32,
        max_bias: f32, m0: f32, m1: f32, n_head_log2: i32, logit_softcap: f32,
        n_kv_max_padded: i32,
    }
    let a = A {
        ne01: 1, ne02: nh as i32, ne03: 1,
        nb01: (nh * hd * 4) as u64, nb02: (hd * 4) as u64, nb03: (nh * hd * 4) as u64,
        ne11: tkv as i32, ne12: 2, ne13: 1,
        ns10: 256,
        nb11, nb12: (hd * 2) as u64, nb13: (hd * 2) as u64,
        ns20: 256,
        nb21: nb11, nb22: (hd * 2) as u64, nb23: (hd * 2) as u64,
        ne31: 1, ne32: 1, ne33: 1,
        nb31: (tkv * 2) as u64, nb32: 0, nb33: 0,
        ne1: nh as i32, ne2: 1, ne3: 1,
        scale: 1.0f32 / (hd as f32).sqrt(),
        max_bias: 0.0, m0: 1.0, m1: 1.0, n_head_log2: 5, logit_softcap: 0.0,
        n_kv_max_padded: 0,
    };
    let q = queue();
    let cmdb = q.new_command_buffer().to_owned();
    let enc = cmdb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_bytes(0, std::mem::size_of::<A>() as u64, &a as *const A as *const _);
    enc.set_buffer(1, Some(&qb), 0);
    enc.set_buffer(2, Some(&kb), 0);
    enc.set_buffer(3, Some(&vb), 0);
    enc.set_buffer(4, Some(&qb), 0); // mask dummy
    enc.set_buffer(5, Some(&qb), 0); // sinks dummy
    enc.set_buffer(6, Some(&pad), 0);
    enc.set_buffer(7, Some(&tmp), 0);
    enc.set_threadgroup_memory_length(0, 1024);
    let grid = MTLSize::new(32, nh as u64, 32);
    let tg = MTLSize::new(32, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmdb.commit();
    cmdb.wait_until_completed();

    // reduce
    let rsrc = include_str!("/tmp/llama_ref/orc_reduce.metal");
    let rlib = device.new_library_with_source(rsrc, &CompileOptions::new()).unwrap();
    let rpipe = device.new_compute_pipeline_state_with_function(&rlib.get_function("orc_reduce", None).unwrap()).unwrap();
    let cmdb2 = q.new_command_buffer().to_owned();
    let enc2 = cmdb2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&rpipe);
    let nrows_i = (nh * 1) as i32;
    enc2.set_bytes(0, 4, &nrows_i as *const i32 as *const _);
    enc2.set_buffer(1, Some(&tmp), 0);
    enc2.set_buffer(2, Some(&yb), 0);
    enc2.dispatch_thread_groups(MTLSize::new(nrows_i as u64, 1, 1), MTLSize::new(1024, 1, 1));
    enc2.end_encoding();
    cmdb2.commit();
    cmdb2.wait_until_completed();

    let got = unsafe {
        std::slice::from_raw_parts(yb.contents() as *const f32, nh * hd)
    };
    let bad = (0..got.len()).filter(|i| got[*i].to_bits() != ywant[*i].to_bits()).count();
    println!("ORACLE-SOURCE kernel vs oracle y: bad={bad}/{}", got.len());

    // ── теперь НАШ standalone-кернел, тот же лаунч ──
    let osrc = include_str!("/tmp/llama_ref/our_vec_run.metal");
    let olib = device.new_library_with_source(osrc, &CompileOptions::new()).expect("our src");
    let opipe = device.new_compute_pipeline_state_with_function(&olib.get_function("our_vec", None).unwrap()).unwrap();
    let tmp2 = device.new_buffer((nrows * (hd * 32 + 64) * 4) as u64, MTLResourceOptions::StorageModeShared);
    let yb2 = device.new_buffer((nh * hd * 4) as u64, MTLResourceOptions::StorageModeShared);
    let cmdb3 = q.new_command_buffer().to_owned();
    let enc3 = cmdb3.new_compute_command_encoder();
    enc3.set_compute_pipeline_state(&opipe);
    enc3.set_bytes(0, std::mem::size_of::<A>() as u64, &a as *const A as *const _);
    enc3.set_buffer(1, Some(&qb), 0);
    enc3.set_buffer(2, Some(&kb), 0);
    enc3.set_buffer(3, Some(&vb), 0);
    enc3.set_buffer(4, Some(&qb), 0);
    enc3.set_buffer(5, Some(&qb), 0);
    enc3.set_buffer(6, Some(&pad), 0);
    enc3.set_buffer(7, Some(&tmp2), 0);
    enc3.set_threadgroup_memory_length(0, 1024);
    enc3.dispatch_thread_groups(MTLSize::new(32, nh as u64, 32), MTLSize::new(32, 1, 1));
    enc3.end_encoding();
    cmdb3.commit();
    cmdb3.wait_until_completed();
    let cmdb4 = q.new_command_buffer().to_owned();
    let enc4 = cmdb4.new_compute_command_encoder();
    enc4.set_compute_pipeline_state(&rpipe);
    enc4.set_bytes(0, 4, &nrows_i as *const i32 as *const _);
    enc4.set_buffer(1, Some(&tmp2), 0);
    enc4.set_buffer(2, Some(&yb2), 0);
    enc4.dispatch_thread_groups(MTLSize::new(nrows_i as u64, 1, 1), MTLSize::new(1024, 1, 1));
    enc4.end_encoding();
    cmdb4.commit();
    cmdb4.wait_until_completed();
    let got2 = unsafe { std::slice::from_raw_parts(yb2.contents() as *const f32, nh * hd) };
    let bad2 = (0..got2.len()).filter(|i| got2[*i].to_bits() != ywant[*i].to_bits()).count();
    println!("OUR-SOURCE standalone vs oracle y: bad={bad2}/{}", got2.len());
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn bytemuck_f16(v: &[half::f16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}
fn queue() -> CommandQueue {
    Device::system_default().unwrap().new_command_queue()
}
