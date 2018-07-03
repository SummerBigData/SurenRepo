#!/bin/bash

echo "Wiping clean all runs, outputs, models, and weights"

find . -type f -name run\* -exec rm {} \;
find . -type f -name out\* -exec rm {} \;

rm -r weights/*
rm -r models/*
