//! Матрица окружения драфтера DFlash (бывший ignored-тест): таблицы таргета
//! (модельные | частичные) × бэкенд драфтера (модельный | свой). Проверяет
//! инварианты и печатает черновики.
//!
//! Запуск: QWEN_GGUF=…8B… DFLASH_GGUF=…drafter… \
//!   cargo run --release --features metal --example dflash_env_matrix
#![cfg(feature = "metal")]
use rust_inference::backend::Backend as _;
use rust_inference::dflash::load_drafter_gguf;
use rust_inference::metal::MetalBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
        let target = std::env::var("QWEN_GGUF").expect("QWEN_GGUF");
        let drafter_path = std::env::var("DFLASH_GGUF").expect("DFLASH_GGUF");
        let be_model = MetalBackend::new().unwrap();
        let be_own = MetalBackend::new().unwrap();
        let bytes = std::fs::read(&target).unwrap();
        let gf = rust_inference::gguf::parse(&bytes).unwrap();
        let wm = rust_inference::weights_map::WeightsMap::from_gguf(&gf).unwrap();
        let vocab = gf
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let cfg = rust_inference::qwen3::Qwen3Config::from_gguf(&gf, vocab).unwrap();
        let mut model = rust_inference::qwen3::Qwen3Model::load(&be_model, &wm, cfg).unwrap();

        // Фичи — как в generate: extract-слои драфтера на префилле.
        let ids: Vec<u32> = (0..13u32).map(|i| 100 + i).collect();
        let tlayers = [1usize, 9, 17, 25, 33];
        let (_lg, extract) = model.prefill_extract(&be_model, &ids, &tlayers).unwrap();

        let mut d_a = load_drafter_gguf(
            &be_model,
            std::path::Path::new(&drafter_path),
            model.token_embed_table().clone(),
            model.output_projection().clone(),
            vocab,
        )
        .unwrap();
        let dim = model.token_embed_table().shape()[1];
        let emb_t = wm.get("token_embd.weight").unwrap();
        let emb_f = rust_inference::weights_map::dequant_f32(emb_t);
        let emb_b = be_own
            .from_cpu_f32(&emb_f, vec![vocab, dim])
            .unwrap();
        let out_b = match wm.get("output.weight") {
            Some(ow) => rust_inference::qwen3::Qwen3Model::upload_weight_pub(
                &be_own,
                &wm,
                "output.weight",
                vec![ow.shape[0] as usize, ow.shape[1] as usize],
            )
            .unwrap(),
            None => emb_b.clone(),
        };
        let mut d_b = load_drafter_gguf(
            &be_own,
            std::path::Path::new(&drafter_path),
            emb_b,
            out_b,
            vocab,
        )
        .unwrap();

        // C: частичные таблицы на бэкенде МОДЕЛИ; D: таблицы модели на СВОЁМ.
        let emb_c = be_model.from_cpu_f32(&emb_f, vec![vocab, dim]).unwrap();
        let out_c = match wm.get("output.weight") {
            Some(ow) => rust_inference::qwen3::Qwen3Model::upload_weight_pub(
                &be_model,
                &wm,
                "output.weight",
                vec![ow.shape[0] as usize, ow.shape[1] as usize],
            )
            .unwrap(),
            None => emb_c.clone(),
        };
        let mut d_c = load_drafter_gguf(
            &be_model,
            std::path::Path::new(&drafter_path),
            emb_c,
            out_c,
            vocab,
        )
        .unwrap();
        let emb_d = rust_inference::qwen3::Qwen3Model::upload_weight_pub(
            &be_own,
            &wm,
            "token_embd.weight",
            vec![vocab, dim],
        )
        .unwrap();
        let out_d = match wm.get("output.weight") {
            Some(ow) => rust_inference::qwen3::Qwen3Model::upload_weight_pub(
                &be_own,
                &wm,
                "output.weight",
                vec![ow.shape[0] as usize, ow.shape[1] as usize],
            )
            .unwrap(),
            None => emb_d.clone(),
        };
        let mut d_d = load_drafter_gguf(
            &be_own,
            std::path::Path::new(&drafter_path),
            model.token_embed_table().clone(),
            model.output_projection().clone(),
            vocab,
        )
        .unwrap();
        let _ = (emb_d, out_d);

        let hist: Vec<u32> = ids.clone();
        // Инварианты (раньше это был только дамп): (1) ни одной нечисловой
        // вероятности ни в одной конфигурации; (2) первый черновик ОДИН и
        // тот же независимо от того, чьи таблицы таргета и чей бэкенд у
        // драфтера (это и проверяла матрица — расхождение = баг загрузки).
        let mut first_drafts: Vec<(String, u32)> = Vec::new();
        for (name, d, be) in [
            ("A таблицы+бэкенд модели", &mut d_a, &be_model as &dyn rust_inference::backend::Backend),
            ("B партиалы+свой бэкенд", &mut d_b, &be_own as &dyn rust_inference::backend::Backend),
            ("C партиалы+бэкенд модели", &mut d_c, &be_model as &dyn rust_inference::backend::Backend),
            ("D таблицы+свой бэкенд", &mut d_d, &be_own as &dyn rust_inference::backend::Backend),
        ] {
            rust_inference::spec::Drafter::after_extract(d, be, &extract, ids.len(), ids.len())
                .unwrap();
            let (t1, p1) = rust_inference::spec::Drafter::draft(d, be, &hist, 4);
            let mut h2 = hist.clone();
            h2.extend_from_slice(&t1);
            let (t2, p2) = rust_inference::spec::Drafter::draft(d, be, &h2, 4);
            let nf = |p: &[f32]| p.iter().filter(|v| !v.is_finite()).count();
            let nf1 = nf(&p1);
            let nf2 = nf(&p2);
            println!(
                "ENV_MATRIX {name}: draft1={t1:?} nonfinite={nf1} | draft2={t2:?} nonfinite={nf2}"
            );
            assert_eq!(nf1, 0, "{name}: нечисловые вероятности в первом черновике");
            assert_eq!(nf2, 0, "{name}: нечисловые вероятности во втором черновике");
            assert!(!t1.is_empty(), "{name}: пустой черновик");
            first_drafts.push((name.to_string(), t1[0]));
        }
        let (n0, t0) = first_drafts[0].clone();
        for (name, t) in &first_drafts[1..] {
            assert_eq!(
                *t, t0,
                "первый черновик зависит от конфигурации: {n0}={t0} vs {name}={t} \
                 (таблицы/бэкенд НЕ должны влиять на драфт)"
            );
        }
        println!("ENV_MATRIX OK: все конфигурации дали первый черновик {t0}");
        Ok(())
    }
