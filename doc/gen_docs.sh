#!/bin/bash
sphinx-apidoc --ext-autodoc -o doc ./active_matcher/
pushd doc
make html
