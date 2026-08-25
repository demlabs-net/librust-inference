//! Диагностика DFlash: диффузия t=1 (AR-шаг драфтера) vs таргет-декод.
//! Использование: dflash_probe <target.gguf> <draft.gguf>
#![cfg(feature = "metal")]
use std::error::Error;
use std::fs::File;
use rust_inference::backend::Backend as _;
use rust_inference::dflash::DFlashModel;
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::weights_map::{dequant_f32, WeightsMap};

fn top5(row: &[f32]) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = row.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.into_iter().take(5).collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (tgt, drf) = (&args[1], &args[2]);
    let backend = MetalBackend::new()?;
    let mmap_t = unsafe { memmap2::Mmap::map(&File::open(tgt)?)? };
    let gf_t = gguf::parse(&mmap_t)?;
    let wm_t = WeightsMap::from_gguf(&gf_t)?;
    let vocab = wm_t.get("token_embd.weight").unwrap().shape[0] as usize;
    let cfg_t = Qwen3Config::from_gguf(&gf_t, vocab)?;
    let mut tgt_model = Qwen3Model::load(&backend, &wm_t, cfg_t.clone())?;

    let mmap_d = unsafe { memmap2::Mmap::map(&File::open(drf)?)? };
    let gf_d = gguf::parse(&mmap_d)?;
    let wm_d = WeightsMap::from_gguf(&gf_d)?;
    let mut dmod = DFlashModel::load_gguf(&backend, &gf_d, &wm_d, vocab)?;

    let ew = dequant_f32(wm_t.get("token_embd.weight").unwrap());
    let embed_t = backend.from_cpu_f32(&ew, vec![vocab, cfg_t.dim])?;
    let ow = dequant_f32(wm_t.get("output.weight").unwrap_or(wm_t.get("token_embd.weight").unwrap()));
    let f16: Vec<half::f16> = ow.iter().map(|&x| half::f16::from_f32(x)).collect();
    let lm_head = backend.from_cpu_f16(&f16, vec![vocab, cfg_t.dim])?;
    dmod.set_target_embed(embed_t);

    // токен для пробы: <|im_start|> (151644) — известный контекст
    let id = 151644u32;
    // таргет: decode(id) → top-5
    let lg_t = tgt_model.decode(&backend, id)?;
    let v_t = backend.to_cpu_f32(&lg_t)?;
    let n = cfg_t.vocab_size;
    let last = &v_t[v_t.len() - n..];
    eprintln!("TARGET decode({id}): top5={:?}", top5(last));
    // драфтер: диффузия [id] (t=1, AR-шаг)
    let r = dmod.diffuse(&backend, &[id], &lm_head, 1)?;
    let v_d = backend.to_cpu_f32(&r.logits)?;
    let row = &v_d[..n];
    eprintln!("DRAFT  diffuse([{id}]): top5={:?}", top5(row));
    // драфтер: диффузия [id, MASK×15] — первая позиция
    let mut ids = vec![id];
    ids.resize(16, dmod.mask_token);
    let r2 = dmod.diffuse(&backend, &ids, &lm_head, 1)?;
    let v2 = backend.to_cpu_f32(&r2.logits)?;
    let row0 = &v2[..n];
    eprintln!("DRAFT  diffuse(block) pos0: top5={:?}", top5(row0));

    // 4B-проверка: Anbeeld-GGUF vs avneetsb-safetensors (валидность конвертаций)
    {
        let p4 = "/Users/dmitriygerasimov/.vassista-models/dflash/qwen35-4b-dflash-Q8_0.gguf";
        let p4s = "/Users/dmitriygerasimov/.vassista-models/dflash/qwen35-4b-dflash-oq8.safetensors";
        if std::path::Path::new(p4).exists() && std::path::Path::new(p4s).exists() {
            let f4 = File::open(p4)?;
            let m4 = unsafe { memmap2::Mmap::map(&f4)? };
            let g4 = gguf::parse(&m4)?;
            let w4 = WeightsMap::from_gguf(&g4)?;
            let st4 = match rust_inference::weights::read_safetensors_f32(std::path::Path::new(p4s)) {
                Ok(s) => s,
                Err(e) => { eprintln!("W4: read_safetensors: {e}"); return Ok(()); }
            };
            let keys: Vec<String> = st4.keys().cloned().collect();
            eprintln!("ST4-n={} first={:?}", keys.len(), &keys[..keys.len().min(5)]);
            for (gn, sn) in [
                ("blk.0.attn_q.weight", "layers.0.self_attn.q_proj.weight"),
                ("blk.0.attn_k.weight", "layers.0.self_attn.k_proj.weight"),
                ("blk.0.ffn_gate.weight", "layers.0.mlp.gate_proj.weight"),
            ] {
                let g = rust_inference::weights_map::dequant_f32(w4.get(gn).unwrap());
                if let Some((s, sh)) = st4.get(sn) {
                    let ncmp = g.len().min(s.len());
                    let (mut corr, mut ns, mut ng) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..ncmp {
                        let (sv, gv) = (s[i] as f64, g[i] as f64);
                        corr += sv * gv; ns += sv * sv; ng += gv * gv;
                    }
                    eprintln!("W4 {gn}: st={sh:?} corr={:.4}", corr / (ns.sqrt() * ng.sqrt()).max(1e-9));
                }
            }
        } else {
            eprintln!("W4: файлы 4B отсутствуют");
        }
    }
    // эмбеддинг MASK-токена (не нулевой ли)
    {
        let ew = dequant_f32(wm_t.get("token_embd.weight").unwrap());
        let d = cfg_t.dim;
        let norms: Vec<(u32, f64)> = [151669u32, 151644, 151645, 151646, 1234, 5, 198]
            .iter()
            .map(|&i| {
                let v = &ew[i as usize * d..(i + 1) as usize * d];
                let n = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
                (i, n)
            })
            .collect();
        eprintln!("embed norms: {norms:?}");
    }
    // куски токенов драфтера
    {
        let tokz = rust_inference::tokenizer::Tokenizer::from_gguf(&gf_t)?;
        for t in [11u32, 13, 279, 419, 67, 15136] {
            eprintln!("piece({t}) = {:?}", String::from_utf8_lossy(tokz.piece(t)));
        }
    }
    // ГИПОТЕЗА LAYOUT: транспонированная загрузка 2D-весов
    {
        if let Ok(mut d2) = DFlashModel::load_gguf_transposed(&backend, &gf_d, &wm_d, vocab) {
            d2.set_target_embed(backend.from_cpu_f32(
                &dequant_f32(wm_t.get("token_embd.weight").unwrap()),
                vec![vocab, cfg_t.dim],
            )?);
            let r = d2.diffuse(&backend, &[151644], &lm_head, 1)?;
            let v = backend.to_cpu_f32(&r.logits)?;
            eprintln!("DRAFT-TRANSPOSED diffuse([151644]): top5={:?}", top5(&v[..n]));
        } else {
            eprintln!("DRAFT-TRANSPOSED: load не удался");
        }
    }
    // СРАВНЕНИЕ ВЕСОВ: safetensors z-lab (HF [out,in]) vs GGUF-деквант
    {
        let st = rust_inference::weights::read_safetensors_f32(
            std::path::Path::new("/Users/dmitriygerasimov/.vassista-models/dflash/qwen3-8b-dflash.safetensors"),
        )?;
        let keys: Vec<String> = st.keys().cloned().collect();
        eprintln!("ST-n={} first={:?}", keys.len(), &keys[..keys.len().min(8)]);
        // корреляция с ПРАВИЛЬНЫМИ именами: st "layers.K.*" ↔ gguf "blk.K.*"
        // гипотеза сдвига: GGUF blk.5 vs safetensors layers.4; blk.4 vs layers.3
        let sn_owned: Vec<(String, String)> = [
            ("blk.5.attn_q.weight", "layers.4.self_attn.q_proj.weight"),
            ("blk.4.attn_q.weight", "layers.3.self_attn.q_proj.weight"),
            ("blk.5.ffn_gate.weight", "layers.4.mlp.gate_proj.weight"),
        ].iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        for (gn, sn) in &sn_owned {
            if let Some(t) = wm_d.get(gn) {
                let g = rust_inference::weights_map::dequant_f32(t);
                if let Some((s, sh)) = st.get(sn) {
                    let ncmp = g.len().min(s.len());
                    let (mut corr, mut ns, mut ng) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..ncmp {
                        let (sv, gv) = (s[i] as f64, g[i] as f64);
                        corr += sv * gv; ns += sv * sv; ng += gv * gv;
                    }
                    eprintln!("SHIFT {gn} vs {sn}: st={sh:?} corr={:.4}", corr / (ns.sqrt() * ng.sqrt()).max(1e-9));
                }
            }
        }
        for k in 0..2usize {
            for (gn, sn) in [
                (format!("blk.{k}.attn_q.weight"), format!("layers.{k}.self_attn.q_proj.weight")),
                (format!("blk.{k}.attn_k.weight"), format!("layers.{k}.self_attn.k_proj.weight")),
                (format!("blk.{k}.ffn_gate.weight"), format!("layers.{k}.mlp.gate_proj.weight")),
                (format!("blk.{k}.attn_output.weight"), format!("layers.{k}.self_attn.o_proj.weight")),
            ] {
                let g = rust_inference::weights_map::dequant_f32(wm_d.get(&gn).unwrap());
                match st.get(&sn) {
                    Some((s, sh)) => {
                        // raw и транспонированный GGUF-деквант (st — HF [out,in])
                        let (r, c) = (sh[0], sh[1]);
                        let ncmp = g.len().min(s.len());
                        let (mut corr, mut ns, mut ng) = (0.0f64, 0.0f64, 0.0f64);
                        for i in 0..ncmp {
                            let (sv, gv) = (s[i] as f64, g[i] as f64);
                            corr += sv * gv; ns += sv * sv; ng += gv * gv;
                        }
                        let mut corr_t = 0.0f64;
                        let mut ng_t = 0.0f64;
                        if g.len() == r * c {
                            for i in 0..r {
                                for j in 0..c {
                                    let sv = s[i * c + j] as f64;
                                    let gv = g[j * r + i] as f64;
                                    corr_t += sv * gv;
                                    ng_t += gv * gv;
                                }
                            }
                        }
                        eprintln!("WEIGHTS {gn}: st={sh:?} corr_raw={:.4} corr_trans={:.4}",
                            corr / (ns.sqrt() * ng.sqrt()).max(1e-9),
                            if ng_t > 0.0 { corr_t / (ns.sqrt() * ng_t.sqrt()).max(1e-9) } else { 0.0 });
                    }
                    None => eprintln!("WEIGHTS {gn}: нет {sn} в safetensors"),
                }
            }
        }
    }
    // rope-позиции: диффузия [151644] на разных pos
    for pos2 in [0usize, 1, 2, 5, 100] {
        let r = dmod.diffuse(&backend, &[151644], &lm_head, pos2)?;
        let v = backend.to_cpu_f32(&r.logits)?;
        let row = &v[..n];
        eprintln!("DRAFT diffuse([151644]@pos{pos2}): top3={:?}", &top5(row)[..3]);
    }
    // контроль: диффузия t=1 на РАЗНЫХ id — влияет ли контекст?
    for id2 in [151644u32, 151645, 151646, 1234, 5] {
        let r = dmod.diffuse(&backend, &[id2], &lm_head, 1)?;
        let v = backend.to_cpu_f32(&r.logits)?;
        let row = &v[..n];
        eprintln!("DRAFT diffuse([{id2}]): top3={:?}", &top5(row)[..3]);
    }
    Ok(())
}
