//! Дамп первого блока диффузии DFlash на реальных моделях (8B-Base + z-lab
//! драфтер) — бывший ignored-тест. Пишет f32-дампы в /tmp для числового
//! сравнения с эталоном dflash/model.py и проверяет инварианты:
//! все дампы конечны, блок диффузии детерминирован, приёмка первого блока
//! не нулевая.
//!
//! Запуск: INFER_MODELS_DIR=…/models cargo run --release --features metal \
//!   --example dflash_block_dump
#![cfg(feature = "metal")]
use rust_inference::backend::Backend as _;
use rust_inference::metal::MetalBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
        use rust_inference::gguf;
        let backend = MetalBackend::new().unwrap();
        let home = std::env::var("HOME").unwrap();
        // Модели переехали в метарепу: models/ (реестр + gguf). Инф-переменная
        // для кастомного пути.
        let mdir = std::env::var("INFER_MODELS_DIR")
            .unwrap_or_else(|_| format!("{home}/work/ai/inference/models"));
        let tp = format!("{mdir}/llm/Qwen3-8B-Base-Q8_0.gguf");
        // ОРИГИНАЛ z-lab (Q8_0.gguf старше — другие веса, приёмка 0 в
        // принципе). Требует config.json рядом в ~/models/dflash/.
        let dp = format!("{mdir}/dflash/qwen3-8b-dflash-zlab.safetensors");
        let tb = std::fs::read(&tp).unwrap();
        let tgf = gguf::parse(&tb).unwrap();
        let twm = rust_inference::weights_map::WeightsMap::from_gguf(&tgf).unwrap();
        let voc = rust_inference::tokenizer::Tokenizer::from_gguf(&tgf).unwrap();
        let vocab = voc.vocab_size();
        let tc = rust_inference::qwen3::Qwen3Config::from_gguf(&tgf, vocab).unwrap();
        let mut target = rust_inference::qwen3::Qwen3Model::load(&backend, &twm, tc.clone()).unwrap();
        // осмысленный промпт (tg-мусор из 'a' вырождал модель в
        // детерминированную петлю [220,170,222,107,...] — приёмка на нём
        // бессмысленна); фичи = ВСЕ строки префилла (z-lab первый блок).
        let ids: Vec<u32> = voc.encode(
            "Расскажи подробно, как работает спекулятивное декодирование в трансформерах: какие есть методы и в чём их отличия?",
        );
        let tlayers = vec![1, 9, 17, 25, 33];
        let (logits, extract) = target.prefill_extract(&backend, &ids, &tlayers).unwrap();
        let lv = backend.to_cpu_f32(&logits).unwrap();
        let v = tc.vocab_size;
        let t_prompt = ids.len();
        let anchor = lv
            .chunks(v)
            .last()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0 as u32
            })
            .unwrap();
        eprintln!("anchor={anchor} vocab={v} t_prompt={t_prompt}");
        let mut dr = rust_inference::dflash::load_drafter_gguf(
            &backend,
            std::path::Path::new(&dp),
            target.token_embed_table().clone(),
            target.output_projection().clone(),
            vocab,
        )
        .expect("drafter load");
        // фичи (t_prompt строк × 5 слоёв, выходы слоёв) — ТОКЕН-МАЖОР
        // (z-lab: cat по последней оси): строка t = [l0[t]; l1[t]; …; l4[t]].
        let d = tc.dim;
        let n_layers = extract.len();
        let mut feat: Vec<f32> = vec![0f32; n_layers * t_prompt * d];
        for (li, t) in extract.iter().enumerate() {
            let rows = backend.to_cpu_f32(t).unwrap();
            for r in 0..t_prompt {
                let src = r * d;
                let dst = r * n_layers * d + li * d;
                feat[dst..dst + d].copy_from_slice(&rows[src..src + d]);
            }
        }
        let ft = backend.from_cpu_f32(&feat, vec![t_prompt, 5 * d]).unwrap();
        dr.features = Some(ft.clone());
        // блок: [anchor, MASK×15], anchor на позиции t_prompt
        let block = dr.model.block_size;
        let mask = dr.model.mask_token;
        let mut bids = vec![anchor];
        bids.resize(block, mask);
        let pos = t_prompt;
        let res = dr
            .model
            .diffuse(
                &backend,
                &bids,
                &dr.lm_head.clone(),
                pos,
                dr.features.as_ref(),
            )
            .expect("diffuse");
        // дампы
        let write_f32 = |path: &str, data: &[f32]| {
            let mut b: Vec<u8> = Vec::with_capacity(data.len() * 4);
            for x in data {
                b.extend_from_slice(&x.to_le_bytes());
            }
            std::fs::write(path, b).unwrap();
        };
        write_f32("/tmp/dump_features.bin", &feat);
        // эмбеддинги блока (строки target_embed)
        let emb = backend.to_cpu_f32(target.token_embed_table()).unwrap();
        let mut bte = Vec::with_capacity(block * d);
        for &id in &bids {
            bte.extend_from_slice(&emb[id as usize * d..(id as usize + 1) * d]);
        }
        write_f32("/tmp/dump_embeds.bin", &bte);
        let lg_cpu = backend.to_cpu_f32(&res.logits).unwrap();
        write_f32("/tmp/dump_logits.bin", &lg_cpu);
        // lm_head таргета в f32 — для python-эталона приёмки (top-K IOU):
        // dequant из GGUF (тот же путь, что и загрузка движка).
        if let Some(t) = twm.get("output.weight") {
            let hd = rust_inference::weights_map::dequant_f32(t);
            if !hd.is_empty() {
                write_f32("/tmp/dump_head.bin", &hd);
                eprintln!("head dumped: {} f32", hd.len());
            }
        }
        // Веса драфтера Q8-деквант (слой 0 q/k + fc) — сверка против bf16.
        if let Some(t) = twm.get("blk.0.attn_q.weight") {
            write_f32(
                "/tmp/dump_q_attn_q.bin",
                &rust_inference::weights_map::dequant_f32(t),
            );
        }
        if let Some(t) = twm.get("blk.0.attn_k.weight") {
            write_f32(
                "/tmp/dump_q_attn_k.bin",
                &rust_inference::weights_map::dequant_f32(t),
            );
        }
        if let Some(t) = twm.get("fc.weight") {
            write_f32("/tmp/dump_q_fc.bin", &rust_inference::weights_map::dequant_f32(t));
        }
        eprintln!("q8 weight dumps written");
        let hd = backend.to_cpu_f32(&res.hidden).unwrap();
        write_f32("/tmp/dump_hidden.bin", &hd);
        let meta = format!(
            "{{\"vocab\":{v},\"dim\":{d},\"block\":{block},\"pos\":{pos},\"mask\":{mask},\"anchor\":{anchor},\"n_layers\":5,\"tlayers\":[1,9,17,25,33],\"t_prompt\":{t_prompt}}}"
        );
        std::fs::write("/tmp/dump_meta.json", meta).unwrap();
        eprintln!("dumps written");
        // ── приёмка первого блока: черновик vs фактические токены таргета ──
        use rust_inference::spec::Drafter;
        let mut lg = logits.clone();
        let mut actual = Vec::with_capacity(12);
        for _ in 0..12 {
            let rv = backend.to_cpu_f32(&lg).unwrap();
            let row = &rv[..v];
            let tok = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            actual.push(tok);
            lg = target.decode(&backend, tok).unwrap();
        }
        eprintln!("anchor={anchor} actual(1..)={:?}", &actual[..7]);
        // свежий драфтер, тот же контекст: draft() работает по history
        let mut dr2 = rust_inference::dflash::load_drafter_gguf(
            &backend,
            std::path::Path::new(&dp),
            target.token_embed_table().clone(),
            target.output_projection().clone(),
            vocab,
        )
        .expect("drafter2");
        dr2.features = Some(ft.clone());
        dr2.commit(&[anchor]);
        let mut history = ids.clone();
        history.push(anchor);
        let (tokens, probs) = dr2.draft(&backend, &history, 6);
        eprintln!("actual  = {:?}", &actual[..7]);
        eprintln!("draft   = {:?}", tokens);
        eprintln!(
            "probs   = {:?}",
            probs.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>()
        );
        let matched = tokens
            .iter()
            .zip(actual.iter().skip(1))
            .filter(|(a, b)| a == b)
            .count();
        println!(
            "first-block acceptance: {matched}/{} (с лагом 1, как z-lab)",
            tokens.len()
        );
        // ── инварианты (раньше — только дамп): конечность, детерминизм,
        //    ненулевая приёмка (полный ноль = сломанный путь диффузии).
        assert!(
            lg_cpu.iter().all(|v| v.is_finite()),
            "logits блока содержат NaN/Inf"
        );
        assert!(
            hd.iter().all(|v| v.is_finite()),
            "hidden блока содержит NaN/Inf"
        );
        assert!(matched >= 1, "приёмка первого блока нулевая — путь диффузии сломан");
        let res2 = dr
            .model
            .diffuse(&backend, &bids, &dr.lm_head.clone(), pos, dr.features.as_ref())
            .expect("diffuse #2");
        let lg2 = backend.to_cpu_f32(&res2.logits).unwrap();
        assert_eq!(
            lg_cpu.len(),
            lg2.len(),
            "повторный diffuse дал другую форму логитов"
        );
        let max_diff = lg_cpu
            .iter()
            .zip(&lg2)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff == 0.0,
            "блок диффузии НЕ детерминирован на тех же входах: |Δ| max {max_diff}"
        );
        println!("DONE: дампы в /tmp/dump_*.bin, инварианты сошлись");
        Ok(())
    }
