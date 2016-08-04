Before submitting your pull request, please check these hints!

- If you are not familiar with the github workflow, have a look:
  https://guides.github.com/introduction/flow/
  In particular, note that in order to update your pull request to include any
  changes we asked for, you just need to push to your branch again.
- If your pull request addresses a particular issue from our issue tracker,
  reference it in your pull request description on github (not the commit
  message) using the syntax `Closes #123` or `Fixes #123`.
  
Pull request check list:

- Install Lasagne in editable mode to be able to run tests locally:
  http://lasagne.readthedocs.io/en/latest/user/development.html#development-setup
- Make sure PEP8 is followed:
  `python -m pep8 lasagne/`
- Make sure the test suite runs through:
  `python -m py.test`
  (or, to only run tests that include the substring `foo` in their name:
  `python -m py.test -k foo`)
- At the end of the test run output, check if coverage is at 100%. If not (or
  not for the files you changed), you will need to add tests covering the code
  you added.
- It is fine to submit a PR without tests to get initial feedback on the
  implementation, but we cannot merge it without tests.
- If you added/changed any documentation, verify that it renders correctly:
  http://lasagne.readthedocs.io/en/latest/user/development.html#documentation
