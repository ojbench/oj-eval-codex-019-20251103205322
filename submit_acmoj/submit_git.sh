#!/bin/bash
set -e

# Get the git URL
GIT_URL=$(git remote get-url origin | sed 's|ojbench:.*@|https://|')

echo "Git URL: $GIT_URL"
echo "Submitting to ACMOJ..."

python3 submit_acmoj/acmoj_client.py submit --problem-id 2782 --language git --code-file "$GIT_URL"
