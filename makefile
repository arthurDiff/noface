run:
	cargo run
watch:
	cargo watch -x run
watch-check:
	cargo watch -x check
test:
	TEST_LOG=true cargo test | bunyan
coverage:
	cargo tarpaulin --ignore-tests
lint:
	cargo clippy -- -D warnings
usused-dep:
	cargo +nightly udeps
set-githook-path:
	git config core.hooksPath .githooks
