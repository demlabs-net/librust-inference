//! Safetensors: заголовок не должен «лгать» о размере тензора.
//!
//! `data_offsets[1]` (конец диапазона) раньше игнорировался: читатель брал
//! размер из shape/dtype и не сверял его с заголовком, поэтому неверный
//! заголовок приводил к молчаливому чтению ЧУЖИХ байтов (соседнего тензора)
//! вместо ошибки.

use rust_inference::weights::read_safetensors_tensor_f32;

fn write_st(dir: &std::path::Path, header: &str, payload: &[u8]) -> std::path::PathBuf {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(payload);
    let p = dir.join("model.safetensors");
    std::fs::write(&p, buf).unwrap();
    p
}

#[test]
fn declared_range_smaller_than_shape_is_rejected() {
    let dir = std::env::temp_dir().join(format!("st-guard-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // shape требует 4 f32 = 16 байт, а заголовок заявляет только 4 байта.
    let header = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,4]}}"#;
    let path = write_st(&dir, header, &[0u8; 16]);
    let res = read_safetensors_tensor_f32(&path, "w");
    assert!(
        res.is_err(),
        "ложный диапазон обязан быть отвергнут, получено {:?}",
        res.map(|(v, s)| (v.len(), s))
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn consistent_header_still_reads() {
    let dir = std::env::temp_dir().join(format!("st-ok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let header = r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut payload = Vec::new();
    for v in [1.5f32, -2.5] {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let path = write_st(&dir, header, &payload);
    let (vals, shape) = read_safetensors_tensor_f32(&path, "w").expect("корректный файл");
    assert_eq!(shape, vec![2]);
    assert_eq!(vals, vec![1.5, -2.5]);
    let _ = std::fs::remove_dir_all(&dir);
}
