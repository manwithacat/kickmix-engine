//! Plain-text plan files (one key per line) and a small JSON writer; no third-party crates.
//!
//! Plan file lines: `k <int>`, `groups 3,7,11,15;5,6,13,14;9,10,12`, `toffolis <int>`, `count <int>`,
//! `seed <int>`, `per <int>`, `design 8,9,11;2,6,10;...`, `rel 0x<hex>` (one per relation),
//! `term <a1> <h1> <a2> <h2> <s>` (decimal; h masks up to 2^85).

use std::path::Path;

use crate::bits::{B128, B525};
use crate::problem::Term;

#[derive(Clone, Debug, Default)]
pub struct PlanFile {
    pub k: usize,
    pub groups: Vec<Vec<usize>>,
    pub toffolis: Option<usize>,
    pub count: Option<usize>,
    pub seed: Option<u64>,
    pub per: Option<usize>,
    pub design: Vec<Vec<usize>>,
    pub relations: Vec<B525>,
    pub terms: Vec<Term>,
}

fn parse_groups(text: &str) -> Result<Vec<Vec<usize>>, String> {
    text.split(';')
        .filter(|g| !g.trim().is_empty())
        .map(|g| g.split(',').map(|x| x.trim().parse::<usize>().map_err(|e| format!("groups: {e}"))).collect())
        .collect()
}

pub fn read_plan(path: &Path) -> Result<PlanFile, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut pf = PlanFile::default();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let rest = rest.trim();
        let err = |e: String| format!("{}:{}: {e}", path.display(), ln + 1);
        match key {
            "k" => pf.k = rest.parse().map_err(|e| err(format!("{e}")))?,
            "groups" => pf.groups = parse_groups(rest).map_err(err)?,
            "design" => pf.design = parse_groups(rest).map_err(err)?,
            "toffolis" => pf.toffolis = Some(rest.parse().map_err(|e| err(format!("{e}")))?),
            "count" => pf.count = Some(rest.parse().map_err(|e| err(format!("{e}")))?),
            "seed" => pf.seed = Some(rest.parse().map_err(|e| err(format!("{e}")))?),
            "per" => pf.per = Some(rest.parse().map_err(|e| err(format!("{e}")))?),
            "rel" => pf.relations.push(B525::from_hex(rest).map_err(err)?),
            "term" => {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() != 5 {
                    return Err(err("term needs 5 fields".into()));
                }
                let a1: u16 = parts[0].parse().map_err(|e| err(format!("{e}")))?;
                let h1: u128 = parts[1].parse().map_err(|e| err(format!("{e}")))?;
                let a2: u16 = parts[2].parse().map_err(|e| err(format!("{e}")))?;
                let h2: u128 = parts[3].parse().map_err(|e| err(format!("{e}")))?;
                let s: u64 = parts[4].parse().map_err(|e| err(format!("{e}")))?;
                pf.terms.push(Term { a1, h1: B128::from_u128(h1), a2, h2: B128::from_u128(h2), s });
            }
            other => return Err(err(format!("unknown key {other}"))),
        }
    }
    if pf.groups.is_empty() {
        pf.groups = crate::problem::default_groups();
    }
    Ok(pf)
}

pub fn write_plan(path: &Path, pf: &PlanFile) -> Result<(), String> {
    let mut s = String::new();
    s.push_str(&format!("k {}\n", pf.k));
    let g: Vec<String> = pf.groups.iter().map(|g| g.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")).collect();
    s.push_str(&format!("groups {}\n", g.join(";")));
    if let Some(t) = pf.toffolis {
        s.push_str(&format!("toffolis {t}\n"));
    }
    for r in &pf.relations {
        s.push_str(&format!("rel {}\n", r.to_hex()));
    }
    for t in &pf.terms {
        let h1 = t.h1.0[0] as u128 | ((t.h1.0[1] as u128) << 64);
        let h2 = t.h2.0[0] as u128 | ((t.h2.0[1] as u128) << 64);
        s.push_str(&format!("term {} {} {} {} {}\n", t.a1, h1, t.a2, h2, t.s));
    }
    std::fs::write(path, s).map_err(|e| format!("{}: {e}", path.display()))
}

/// Tiny JSON builder.
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn obj() -> Json {
        Json::Obj(Vec::new())
    }

    pub fn push(mut self, key: &str, value: Json) -> Json {
        if let Json::Obj(ref mut v) = self {
            v.push((key.to_string(), value));
        }
        self
    }

    pub fn ints<I: IntoIterator<Item = usize>>(it: I) -> Json {
        Json::Arr(it.into_iter().map(|x| Json::Int(x as i64)).collect())
    }

    pub fn write(&self, out: &mut String, indent: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => out.push_str(&i.to_string()),
            Json::Float(f) => {
                if f.is_finite() {
                    out.push_str(&format!("{f}"))
                } else {
                    out.push_str("null")
                }
            }
            Json::Str(s) => {
                out.push('"');
                for c in s.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            Json::Arr(items) => {
                out.push('[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    it.write(out, indent);
                }
                out.push(']');
            }
            Json::Obj(items) => {
                out.push('{');
                for (i, (k, v)) in items.iter().enumerate() {
                    out.push_str(if i > 0 { ",\n" } else { "\n" });
                    out.push_str(&" ".repeat(indent + 1));
                    Json::Str(k.clone()).write(out, indent);
                    out.push_str(": ");
                    v.write(out, indent + 1);
                }
                out.push('\n');
                out.push_str(&" ".repeat(indent));
                out.push('}');
            }
        }
    }

    pub fn to_string_pretty(&self) -> String {
        let mut s = String::new();
        self.write(&mut s, 0);
        s.push('\n');
        s
    }
}
