#!/bin/bash
# Build the extension in place (no maturin needed): release cdylib with macOS dynamic Python symbol lookup.
set -e
cd "$(dirname "$0")"
cargo rustc --release --lib --offline -- -C link-arg=-undefined -C link-arg=dynamic_lookup
cp target/release/lib_engine.dylib python/kickmix_engine/_engine.abi3.so
shasum -a 256 python/kickmix_engine/_engine.abi3.so
