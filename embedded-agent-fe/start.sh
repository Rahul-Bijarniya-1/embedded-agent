#!/bin/bash
# Start the frontend static server on http://localhost:3000
# (no --single: that flag rewrites every path to index.html once serve strips the .html extension,
#  which sent transactions.html, loan-details.html and credit.html to the Dashboard)
npx serve . --no-clipboard --no-port-switching --listen 3000
