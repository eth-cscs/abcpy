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
2. Adapt `VERSION` file in the repos root directiory `echo M.m.b > VERSION`
3. Merge all desired feature branches into the release branch
4. Create a pull/ merge request: release branch -> master

After a successfull merge:

5. Create tag vM.m.b (`git tag vM.m.b`) and push the tag (`git push --tags`) 
6. Create a release in Github

The new tag on master will signal Travis to deploy a new package to Pypi while
the GitHub release is just for user documentation.
