# Repository instructions

## Publish shared-package changes

- An authorized change to a shared mouse package is not complete until the
  relevant checks pass and the change is committed and pushed to GitHub.
- Check the dependent packages' dependency declarations and install scripts.
  Publish to the branch or ref they actually install; mouse-experiment currently
  installs mouse-core from `main`. A local commit, local reinstall, or push to an
  unused branch does not make the fix available to those packages.
- Inspect the working tree and commit only the intended changes. Preserve
  unrelated work, and never force-push to publish a fix.
- Verify the published commit is present on the dependency's remote ref before
  reporting completion. When reinstalling a dependent package, use that GitHub
  dependency so the result matches its normal install workflow.
- If publishing fails or requires an unavailable permission, report the blocker
  and distinguish local changes from changes available on GitHub.
