#!/bin/sh

flake8 .
mypy steerable_cnns/group/*.py
pytest steerable_cnns
