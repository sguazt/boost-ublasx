#!/bin/bash

#
# Simple script to create a new release in a Git repository.
# 
# Increase the version number and update the Git repo with the new version
# number and a new annotated tag marking the new release.
#
# By default, this script reads version number from the `VERSION` file and
# increments the patch level (i.e., if `VERSION` contains `maj.min.pat`, this
# scripts will create a release for version `maj.min.(pat+1)`).
#
# Usage:
#  <script name> <release-type> [<version-file>]
# where:
# - <release-type>: the type of the release; one of 'major', 'minor' and
#   'patch'.
#   [default: 'patch']
# - <version-file>: the file containing the version number.
#   [default: 'VERSION']
#
#
# Marco Guazzone (marco.guazzone@gmail.com)
#

default_release_type=patch
default_version_file=VERSION


release_type=$default_release_type
version_file=$default_version_file

if [ $# -gt 0 ]; then
	release_type=$1

	if [ $# -gt 1 ]; then
		version_file=$2
	fi
fi

if [ ! -e "$version_file" ]; then
	echo "Version file '$version_file' not found. Created a new one for version '0.0.0'"
	echo "0.0.0" > $version_file
fi

version=$(cat $version_file)
IFS='.' read -r major_version minor_version patch_version <<<"$version"

case "$release_type" in
	major)
		major_version=$((major_version+1))
		minor_version=0
		patch_version=0
		;;
	minor)
		minor_version=$((minor_version+1))
		patch_version=0
		;;
	patch)
		patch_version=$((patch_version+1))
		;;
	*)
		echo "Usage: $0 <release-type> [<version-file>]"
		echo "where:"
		echo "- <release-type> is one of 'major', 'minor', 'patch'. [default: '$default_release_type']"
		echo "- <version-file> is the path to the file containing the version number. [default: '$default_version_file']"
		exit 1
		;;
esac

new_version="${major_version}.${minor_version}.${patch_version}"

cp -f $version_file $version_file.bak \
&& echo "$new_version" > $version_file \
&& git add $version_file \
&& git ci -m "Bumped version number to $new_version" \
&& git tag -a "v${new_version}" -m "Released version $new_version." \
&& git push --follow-tags \
&& rm -f "$version_file.bak"

if [ $? -ne 0 ]; then
	echo "Something went wrong! Please check your repo."
	exit 1
fi
