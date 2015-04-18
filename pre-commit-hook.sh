#!/usr/bin/env bash

# Stash uncommited changes to make sure the commited ones will work
git stash -q --keep-index
# Run tests
py.test
RETURN_CODE=$?
# Pop stashed changes
git stash pop -q

exit $RETURN_CODE