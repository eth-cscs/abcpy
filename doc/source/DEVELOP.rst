Branching Scheme
================

We use the branching strategy described in this `blog post <http://nvie.com/posts/a-successful-git-branching-model>`_.


Deploy a new Release
====================

This documentation is mainly intended for the main developers. The deployment of
new releases is automated using Travis CI. However, there are still a few manual
steps required in order to deploy a new release. Assume we want to deploy the
new version `M.m.b':

1. Create a release branch `release-M.m.b`
2. Adapt `VERSION` file in the repos root directory: `echo M.m.b > VERSION`
3. Adapt `README.md` file: adapt links to correct version of `User Documentation` and `Reference`
4. Adapt `doc/source/DEVELOP.rst` file: to install correct version of ABCpy
5. Merge all desired feature branches into the release branch
6. Create a pull/ merge request: release branch -> master

After a successful merge:

7. Create tag vM.m.b (`git tag vM.m.b`)
8. Retag tag `stable` to the current version
9. Push the tag (`git push --tags`)
10. Create a release in GitHub

The new tag on master will signal Travis to deploy a new package to Pypi while
the GitHub release is just for user documentation.
