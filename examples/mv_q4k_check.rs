//! Точечная сверка q4_K matvec/GEMM движка с эталоном: x = oracle
//! attn_norm-0[:,5] (дамп oracle_dump.cpp), срав с oracle Qcur-0 (сырая
//! q-проекция). LLM_MM=1 — прогон при m=9 (порог use_mm), по умолчанию m=1.
use rust_inference::backend::Backend as _;
use std::io::Read as _;

fn parse(path: &str, step: u32, name: &str, ne: &str) -> Vec<f32> {
    let mut s = String::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_string(&mut s)
        .unwrap();
    let mut vals = Vec::new();
    let mut grab = false;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("STEP ") {
            let mut it = rest.split_whitespace();
            let st: u32 = it.next().unwrap().parse().unwrap();
            assert_eq!(it.next(), Some("TENSOR"));
            let nm = it.next().unwrap();
            let shape = it.next().unwrap_or("").trim_matches(|c: char| !c.is_ascii_digit() && c != ',');
            grab = st == step && nm == name && shape == ne;
            if grab {
                vals.clear();
            }
            continue;
        }
        if grab {
            for t in line.split_whitespace() {
                if t.len() == 8 && t.bytes().all(|b| b.is_ascii_hexdigit()) {
                    vals.push(f32::from_bits(u32::from_str_radix(t, 16).unwrap()));
                }
            }
        }
    }
    vals
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (xall, oq) = if std::env::var("LLM_ISO").is_ok() {
        let rd = |p: &str| -> Vec<f32> {
            let b = std::fs::read(p).unwrap();
            b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        ({ let p: String = std::env::var("LLM_X").unwrap_or_else(|_| "/tmp/x14.bin".into()); rd(&p) }, rd("/tmp/iso_out.bin"))
    } else if let Ok(p) = std::env::var("LLM_BIN") {
        let rd = |p: &str| -> Vec<f32> {
            let b = std::fs::read(p).unwrap();
            b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let mut it = p.split(',');
        (rd(it.next().unwrap()), rd(it.next().unwrap()))
    } else if std::env::var("LLM_MM_BIN").is_ok() {
        let rd = |p: &str| -> Vec<f32> {
            let b = std::fs::read(p).unwrap();
            b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        (rd("/tmp/x14.bin"), rd("/tmp/oq14.bin"))
    } else {
        (parse(&a[2], 0, "attn_norm-0", "2048,6,1,1"), parse(&a[3], 0, "Qcur-0", "2048,6,1,1"))
    };
    let mm = std::env::var("LLM_MM").ok();
    let m = mm.as_deref().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let m = std::env::var("LLM_M").ok().and_then(|s| s.parse().ok()).unwrap_or(m);
    let from_bin = mm.is_some_and(|s| s.parse::<usize>().is_ok());

    let bytes = std::fs::read(&a[1]).unwrap();
    let gf = rust_inference::gguf::parse(&bytes).unwrap();
    let wm = rust_inference::weights_map::WeightsMap::from_gguf(&gf).unwrap();
    let be = rust_inference::metal::MetalBackend::new().unwrap();
    let w = wm.get("blk.0.attn_q.weight").unwrap();
    let wt = be
        .from_quantized(w.raw(), vec![2048, 2048], w.format)
        .unwrap();
    // x — строки токенов 5..5+m (для m>1 берём токен 5 m раз — сравнение
    // только строки 0 корректна при m>1, но нас интересует побитовая
    // совпааемость matvec-пути: при m=9 движок пойдёт в mm, и это контроль).
    let xs: Vec<f32> = if std::env::var("LLM_ISO").is_ok() || std::env::var("LLM_BIN").is_ok() || std::env::var("LLM_MM_BIN").is_ok() {
        xall[..m * 2048].to_vec()
    } else {
        let mut v = Vec::new();
        for _ in 0..m {
            v.extend_from_slice(&xall[5 * 2048..6 * 2048]);
        }
        v
    };
    let xt = be.from_cpu_f32(&xs, vec![m, 2048]).unwrap();
    let c = be.zeros(vec![m, 2048]).unwrap();
    be.gemm_nt(&xt, &wt, &c).unwrap();
    let got = be.to_cpu_f32(&c).unwrap();
    let row = std::env::var("LLM_ROW").ok().and_then(|s| s.parse().ok()).unwrap_or(if std::env::var("LLM_MM_BIN").is_ok() || std::env::var("LLM_BIN").is_ok() || std::env::var("LLM_ISO").is_ok() { 0 } else { 5 });
    let oq5 = &oq[row * 2048..(row + 1) * 2048];
    let nd = (0..2048).filter(|i| oq5[*i].to_bits() != got[*i].to_bits()).count();
    println!("m={m}: bitwise-различий с oracle: {nd}/2048");
    for i in 0..6 {
        println!(
            "[{i}] oracle={:10.8} eng={:10.8} {}",
            oq5[i],
            got[i],
            if oq5[i].to_bits() == got[i].to_bits() { "==" } else { "!!" }
        );
    }
}
