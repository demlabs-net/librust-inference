//! Чтение контейнера `NINFER` (артефакт движка Infernix) — только ЧТЕНИЕ.
//!
//! Боевые веса Flash-Next на zeus лежат именно в этом формате: оригинального
//! GGUF на машине нет (в HF-кэше остались заглушки), а `.infernix` — не GGUF.
//! Формат снят с живого файла (проверено на IQ3_S-артефакте, 11.11.2026):
//!
//! ```text
//! 0..7    magic "NINFER\0"   (у файла-тома — "NINPRT\0", payload без манифеста)
//! 7       версия формата (3)
//! 8..16   длина JSON-манифеста (u64 LE)
//! 16..32  id тома (16 байт)
//! 32..    JSON-манифест: components, objects[], bindings, files[], metadata
//! base    payload первого тома; base = align_up(32 + json_len, 32)
//! ```
//!
//! `objects[]` — 920 тензоров и 4 ресурса: `id`, `shape`, `format`
//! (`ggml_iq3_s`, `bf16`, `fp32`, `q8_g32_fp16`, …), `layout` (раскладка
//! байтов внутри объекта), `offset`, `bytes`. Смещения ГЛОБАЛЬНЫЕ: объекты
//! продолжаются из контейнера в тома `NINPRT`, поэтому чтение объекта идёт
//! через `slice_ranges` и умеет пересекать границу тома (в живом файле такое
//! пересечение ровно одно).
//!
//! `bindings` связывает смысловые имена (`text/layers/0/attn_hc/norm`) с
//! объектами (`weight/000310`) — по ним загрузчик находит нужные тензоры.
//!
//! Здесь только разбор контейнера и байты объекта. Раскладки
//! (`ggml_tiles16_v1`, `row_split_k128_v1`, `ggml_expert_*_v1`, …) —
//! следующий шаг: каждая переносится в обычный ggml-порядок отдельно и
//! принимается по паритету текста с самим Infernix.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;

/// Magic контейнера с манифестом.
pub const MAGIC: &[u8; 7] = b"NINFER\0";
/// Magic файла-тома (продолжение payload).
pub const MAGIC_PART: &[u8; 7] = b"NINPRT\0";
/// Поддерживаемая версия формата.
pub const VERSION: u8 = 3;
/// Выравнивание начала payload в томе (контейнер и NINPRT-тома).
pub const PAYLOAD_ALIGN: u64 = 4096;

/// Один объект манифеста (тензор или ресурс).
#[derive(Debug, Clone)]
pub struct Object {
    pub id: String,
    pub kind: String,
    /// Пусто у ресурсов.
    pub shape: Vec<u64>,
    /// `ggml_iq3_s`, `bf16`, `fp32`, … (пусто у ресурсов).
    pub format: String,
    /// `contiguous_le_v1`, `ggml_tiles16_v1`, … (пусто у ресурсов).
    pub layout: String,
    pub offset: u64,
    pub bytes: u64,
}

impl Object {
    pub fn is_tensor(&self) -> bool {
        self.kind == "tensor"
    }

    /// Элементов по форме (0 у ресурсов).
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Кусок объекта в конкретном томе.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    pub volume: usize,
    /// Смещение В ФАЙЛЕ тома (не глобальное).
    pub file_offset: u64,
    pub len: u64,
}

/// Открытый контейнер: манифест + отображённые тома.
pub struct Ninfer {
    /// Тома: [0] — сам контейнер, дальше `NINPRT`.
    maps: Vec<Mmap>,
    /// Границы payload томов в глобальных смещениях: начало каждого тома.
    starts: Vec<u64>,
    /// Смещение payload внутри файла тома (у контейнера > 0, у `NINPRT` 0).
    bases: Vec<u64>,
    objects: Vec<Object>,
    by_id: HashMap<String, usize>,
    /// bindings: смысловое имя → индекс объекта (только «цельный» случай).
    named: HashMap<String, usize>,
    /// Сырые записи bindings: имя → `{object}` или `{parts:[{object,range}]}`.
    raw_bindings: HashMap<String, serde_json::Value>,
    /// Имя артефакта из манифеста.
    pub name: String,
}

impl std::fmt::Debug for Ninfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ninfer")
            .field("name", &self.name)
            .field("objects", &self.objects.len())
            .field("volumes", &self.maps.len())
            .field("named", &self.named.len())
            .finish()
    }
}

impl Ninfer {
    /// Открыть контейнер. Тома читаются из того же каталога (пути берутся из
    /// `files[]` манифеста).
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        if map.len() < 32 || &map[0..7] != MAGIC {
            return Err(format!("{}: не NINFER-контейнер", path.display()));
        }
        if map[7] != VERSION {
            return Err(format!(
                "{}: версия формата {} (ожидалась {VERSION})",
                path.display(),
                map[7]
            ));
        }
        let json_len = u64::from_le_bytes(map[8..16].try_into().unwrap()) as usize;
        let json_start = 32usize;
        let json_end = json_start
            .checked_add(json_len)
            .filter(|e| *e <= map.len())
            .ok_or_else(|| format!("{}: манифест за концом файла", path.display()))?;
        let manifest: serde_json::Value = serde_json::from_slice(&map[json_start..json_end])
            .map_err(|e| format!("манифест: {e}"))?;

        let name = manifest
            .get("metadata")
            .and_then(|m| m.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Объекты.
        let mut objects = Vec::new();
        let arr = manifest
            .get("objects")
            .and_then(|v| v.as_array())
            .ok_or("манифест: нет objects[]")?;
        for o in arr {
            let s = |k: &str| o.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let shape = o
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                .unwrap_or_default();
            objects.push(Object {
                id: s("id"),
                kind: s("kind"),
                shape,
                format: s("format"),
                layout: s("layout"),
                offset: o.get("offset").and_then(|v| v.as_u64()).unwrap_or(0),
                bytes: o.get("bytes").and_then(|v| v.as_u64()).unwrap_or(0),
            });
        }
        let mut by_id = HashMap::with_capacity(objects.len());
        for (i, o) in objects.iter().enumerate() {
            by_id.insert(o.id.clone(), i);
        }

        // Bindings: { "text/.../norm": { "object": "weight/000310" } } либо
        // { "text/.../query": { "parts": [{ "object": "...", "range": [a,b] }] } }
        let mut named = HashMap::new();
        let mut raw_bindings = HashMap::new();
        if let Some(b) = manifest.get("bindings").and_then(|v| v.as_object()) {
            for (k, v) in b {
                raw_bindings.insert(k.clone(), v.clone());
                if let Some(id) = v.get("object").and_then(|x| x.as_str())
                    && let Some(&idx) = by_id.get(id)
                {
                    named.insert(k.clone(), idx);
                }
            }
        }

        // Тома. Первый — контейнер: payload начинается после манифеста
        // (выравнивание до 32, как в живом файле), дальше `files[]`.
        let base0 = align_up(json_end as u64, PAYLOAD_ALIGN);
        let mut maps = vec![map];
        let mut starts = vec![0u64];
        let mut bases = vec![base0];
        let files = manifest
            .get("files")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        // files[0] описывает сам контейнер (path = null), остальные — тома.
        for entry in files.iter().skip(1) {
            let Some(rel) = entry.get("path").and_then(|v| v.as_str()) else {
                continue;
            };
            let part_path = path.parent().unwrap_or_else(|| Path::new(".")).join(rel);
            let f = std::fs::File::open(&part_path)
                .map_err(|e| format!("{}: {e}", part_path.display()))?;
            let m = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap: {e}"))?;
            if m.len() < 32 || &m[0..7] != MAGIC_PART {
                return Err(format!("{}: не NINPRT-том", part_path.display()));
            }
            // Payload тома выровнен до PAYLOAD_ALIGN (проверено на живом
            // файле: данные читаются только с 4096, со сдвигом 32 — мусор).
            let part_base = align_up(32, PAYLOAD_ALIGN);
            if (m.len() as u64) < part_base {
                return Err(format!("{}: том короче заголовка", part_path.display()));
            }
            let prev_end = {
                let sz = maps.last().unwrap().len() as u64;
                let b = *bases.last().unwrap();
                if sz < b {
                    return Err("контейнер: payload начинается за концом файла".into());
                }
                starts.last().copied().unwrap() + (sz - b)
            };
            starts.push(prev_end);
            bases.push(part_base);
            maps.push(m);
        }

        Ok(Self {
            maps,
            starts,
            bases,
            objects,
            by_id,
            named,
            raw_bindings,
            name,
        })
    }

    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    /// Объект по `bindings`-имени (`text/layers/0/attn_hc/norm`).
    pub fn object_by_name(&self, name: &str) -> Option<&Object> {
        self.named.get(name).map(|i| &self.objects[*i])
    }

    /// Объект по id (`weight/000310`).
    pub fn object(&self, id: &str) -> Option<&Object> {
        self.by_id.get(id).map(|i| &self.objects[*i])
    }

    /// Имена bindings, начинающиеся с префикса (для диагностики/карты).
    pub fn names_with_prefix<'a>(&'a self, prefix: &'a str) -> Vec<&'a str> {
        let mut v: Vec<&str> = self
            .named
            .keys()
            .filter(|k| k.starts_with(prefix))
            .map(|s| s.as_str())
            .collect();
        v.sort_unstable();
        v
    }

    /// Разложить объект на куски по томам (объект может пересекать границу).
    pub fn slice_ranges(&self, obj: &Object) -> Result<Vec<Range>, String> {
        let mut out = Vec::new();
        let mut left = obj.bytes;
        let mut pos = obj.offset;
        while left > 0 {
            let vol = self
                .starts
                .iter()
                .enumerate()
                .rev()
                .find(|(_, s)| **s <= pos)
                .map(|(i, _)| i)
                .ok_or_else(|| format!("{}: смещение {pos} до первого тома", obj.id))?;
            let vol_end = self.volume_end(vol);
            if pos >= vol_end {
                return Err(format!(
                    "{}: смещение {pos} вне тома {vol} (конец {vol_end})",
                    obj.id
                ));
            }
            let take = left.min(vol_end - pos);
            let file_offset = self.bases[vol] + (pos - self.starts[vol]);
            let map_len = self.maps[vol].len() as u64;
            if file_offset + take > map_len {
                return Err(format!(
                    "{}: кусок {file_offset}+{take} за концом тома {vol} ({map_len})",
                    obj.id
                ));
            }
            out.push(Range {
                volume: vol,
                file_offset,
                len: take,
            });
            pos += take;
            left -= take;
        }
        Ok(out)
    }

    /// Глобальный конец тома (payload).
    fn volume_end(&self, vol: usize) -> u64 {
        let len = self.maps[vol].len() as u64;
        let base = self.bases[vol];
        self.starts[vol] + len.saturating_sub(base)
    }

    /// Прочитать объект целиком (собирает куски по томам).
    pub fn read(&self, obj: &Object) -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(obj.bytes as usize);
        for r in self.slice_ranges(obj)? {
            let map = &self.maps[r.volume];
            let a = r.file_offset as usize;
            out.extend_from_slice(&map[a..a + r.len as usize]);
        }
        Ok(out)
    }

    /// Логический тензор по bindings-имени с учётом упаковки.
    ///
    /// Часть имён ссылается на объект целиком (`object`), часть — на
    /// БАЙТОВЫЙ ДИАПАЗОН внутри общего объекта (`parts: [{object, range}]`):
    /// так Infernix пакует q|k GDN, HC down|inject, роутер|shared_score.
    /// Возвращает куски `(объект, смещение-в-объекте, длина)` в порядке
    /// bindings — их склейка и есть тензор.
    pub fn parts(&self, name: &str) -> Result<Vec<(&Object, u64, u64)>, String> {
        let Some(v) = self.raw_bindings.get(name) else {
            return Err(format!("bindings: нет {name}"));
        };
        let mut out = Vec::new();
        if let Some(id) = v.get("object").and_then(|x| x.as_str()) {
            let obj = self
                .object(id)
                .ok_or_else(|| format!("bindings {name}: нет объекта {id}"))?;
            out.push((obj, 0, obj.bytes));
        }
        if let Some(arr) = v.get("parts").and_then(|x| x.as_array()) {
            for p in arr {
                let id = p
                    .get("object")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| format!("bindings {name}: part без object"))?;
                let obj = self
                    .object(id)
                    .ok_or_else(|| format!("bindings {name}: нет объекта {id}"))?;
                let r = p
                    .get("range")
                    .and_then(|x| x.as_array())
                    .ok_or_else(|| format!("bindings {name}: part без range"))?;
                if r.len() != 2 {
                    return Err(format!("bindings {name}: range не из двух чисел"));
                }
                let a = r[0].as_u64().unwrap_or(0);
                let b = r[1].as_u64().unwrap_or(0);
                if b < a || b > obj.bytes {
                    return Err(format!(
                        "bindings {name}: range [{a}, {b}] вне объекта {} ({} байт)",
                        obj.id, obj.bytes
                    ));
                }
                out.push((obj, a, b - a));
            }
        }
        if out.is_empty() {
            return Err(format!("bindings {name}: ни object, ни parts"));
        }
        Ok(out)
    }

    /// Прочитать логический тензор (склеивает части, при необходимости
    /// пересекая тома).
    pub fn read_name(&self, name: &str) -> Result<Vec<u8>, String> {
        let parts = self.parts(name)?;
        let total: u64 = parts.iter().map(|(_, _, n)| *n).sum();
        let mut out = Vec::with_capacity(total as usize);
        for (obj, off, len) in parts {
            let slice = Object {
                offset: obj.offset + off,
                bytes: len,
                ..obj.clone()
            };
            for r in self.slice_ranges(&slice)? {
                let map = &self.maps[r.volume];
                let a = r.file_offset as usize;
                out.extend_from_slice(&map[a..a + r.len as usize]);
            }
        }
        Ok(out)
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Синтетический контейнер: два тома, объект, пересекающий границу.
    /// Тест не зависит от реальных машин и файлов моделей.
    #[test]
    fn reads_across_volume_boundary() {
        let dir = std::env::temp_dir().join(format!("ninfer-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let container = dir.join("c.ninfer");
        let part = dir.join("c-00002.infernix");

        let json = serde_json::json!({
            "metadata": { "name": "test-model" },
            "objects": [
                // 8 байт «resource» в начале payload
                { "id": "resource/x", "kind": "resource", "offset": 0, "bytes": 8 },
                // объект, который начинается в томе 0 и продолжается в томе 1
                { "id": "weight/0", "kind": "tensor", "shape": [2, 3], "format": "bf16",
                  "layout": "contiguous_le_v1", "offset": 6, "bytes": 10 }
            ],
            "bindings": {
                "text/a": { "object": "weight/0" },
                // упаковка: два логических тензора в одном объекте
                "text/q": { "parts": [{ "object": "weight/0", "range": [0, 4] }] },
                "text/k": { "parts": [{ "object": "weight/0", "range": [4, 10] }] }
            },
            "files": [
                { "path": null, "payload_bytes": 16 },
                { "path": "c-00002.infernix", "payload_bytes": 16 }
            ]
        });
        let json_bytes = serde_json::to_vec(&json).unwrap();
        // заголовок: magic + версия + длина + 16 байт id, затем JSON
        let mut head = Vec::new();
        head.extend_from_slice(MAGIC);
        head.push(VERSION);
        head.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        head.extend_from_slice(&[0u8; 16]);
        head.extend_from_slice(&json_bytes);
        // payload тома 0: 16 байт, значение 0..16; заголовок выравнивается
        // до PAYLOAD_ALIGN (как в живом контейнере)
        while head.len() % PAYLOAD_ALIGN as usize != 0 {
            head.push(0);
        }
        let mut payload0: Vec<u8> = (0u8..16).collect();
        let mut file0 = head.clone();
        file0.append(&mut payload0);
        std::fs::write(&container, &file0).unwrap();

        // том 1: magic NINPRT + версия + нули до PAYLOAD_ALIGN, payload 16..32
        let mut file1 = Vec::new();
        file1.extend_from_slice(MAGIC_PART);
        file1.push(VERSION);
        file1.resize(PAYLOAD_ALIGN as usize, 0);
        file1.extend(16u8..32);
        std::fs::write(&part, &file1).unwrap();

        let n = Ninfer::open(&container).unwrap();
        assert_eq!(n.name, "test-model");
        let obj = n.object_by_name("text/a").unwrap().clone();
        assert_eq!(obj.numel(), 6);
        // объект: offset 6, длина 10 → байты 6..16 тома 0 (10 значений 6..15)
        assert_eq!(n.read(&obj).unwrap(), (6u8..16).collect::<Vec<u8>>());
        // parts: q = первые 4 байта объекта, k = остальные 6
        assert_eq!(n.read_name("text/q").unwrap(), vec![6u8, 7, 8, 9]);
        assert_eq!(
            n.read_name("text/k").unwrap(),
            vec![10u8, 11, 12, 13, 14, 15]
        );
        let ranges = n.slice_ranges(&obj).unwrap();
        assert_eq!(
            ranges.len(),
            1,
            "в этом раскладе объект не пересекает границу"
        );

        // Теперь сдвинем объект так, чтобы он пересёк границу томов.
        let mut crossing = obj.clone();
        crossing.offset = 12;
        crossing.bytes = 8; // 12..16 том 0, 16..20 том 1
        let ranges = n.slice_ranges(&crossing).unwrap();
        assert_eq!(ranges.len(), 2);
        assert_eq!(n.read(&crossing).unwrap(), (12u8..20).collect::<Vec<u8>>());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_foreign_file() {
        let dir = std::env::temp_dir().join(format!("ninfer-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.bin");
        std::fs::write(
            &p,
            b"NOPE\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        )
        .unwrap();
        assert!(Ninfer::open(&p).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
