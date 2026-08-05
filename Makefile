.PHONY: install verify test run package

install:
	python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
	python -m pip install -r requirements-dev.txt

verify:
	python scripts/verify_assets.py

test: verify
	python -m pytest -q

run: verify
	python launch.py

package: test
	python scripts/package_release.py
