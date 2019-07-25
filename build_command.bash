#!/bin/bash

if [ -d dist ]; then
    echo "removig directory"
    rm dist -r
fi

python3 setup.py sdist bdist_wheel
