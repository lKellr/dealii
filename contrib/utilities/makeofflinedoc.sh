#!/bin/bash
## ---------------------------------------------------------------------
##
## Copyright (C) 2012 - 2018 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the deal.II distribution.
##
## ---------------------------------------------------------------------

# This script downloads images referenced in the tutorial steps and patches
# the URLs to point to the local files. To be run in the doc/doxygen/deal.II
# directory of an installed deal.II documentation.

if [ ! -f Tutorial.html ]
then
  echo "Please run this script in the doc output directory (<install>/doc/doxygen/deal.II)"
  exit 1
fi

echo "Patching html files ..."
sed -i 's#"https\?://www.dealii.org/images/#"images/#g' step_*.html class*.html

echo "Downloading images (this will take a long time; press ctrl-c to cancel) ..."

for i in steps/developer \
         shape-functions/{DGPMonomial,DGPNonparametric,DGP}/P{1,2,3,4} \
         shape-functions/{hierarchical,lagrange}/Q{1,2,3,4}; do
  echo "  ... processing images/$i"
  mkdir -p images/"$i"
  (
    cd images/"$i"
    trap "echo \"(skipping)\"" SIGINT
    wget -q -nd -A svg,png,gif,webm -m -l 1 -np "https://www.dealii.org/images/$i"
    rm -f robots.txt*
  )
done

echo "all done!"
