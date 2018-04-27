#!/bin/sh

usage="Usage: $0 ACTION VERSION [NAME]\n  ACTION = new, merge\n  VERSION = version number of the release\n  NAME = optional name instead of 'release' (i.e. 'hotfix')"

if [ $# -ne 2 ] && [ $# -ne 3 ]; then
    echo $usage
    exit 1
else
    action=$1
    version=$2
    if [ $# -eq 3 ]; then
        name=$3
    else
        name="release"
    fi
fi

if [ $version == "help" ] || [ $version == "-h" ] || [ $version == "--help" ]; then
    echo $usage
    exit 2
fi

if [ $name != "release" ] && [ $name != "hotfix" ]; then
    echo "Branch base name must be either 'release' or 'hotfix'"
    exit 5
fi

semver="[0-9]+\.[0-9]+\.[0-9]+[a-z]*"
if [[ ! $version =~ $semver ]]; then
    echo "Version number must follow semver (X.Y.Z)"
    exit 4
fi

fullname="$name-$version"


# Color code the output
BLK='\033[0;30m'
RED='\033[0;31m'
GRN='\033[0;32m'
ORG='\033[0;33m'
BLU='\033[0;34m'
PRP='\033[0;35m'
CYN='\033[0;36m'
GRY='\033[0;37m'
NC='\033[0m' # No color

if [ $action == "new" ]; then
    echo "${GRN}Making new release branch $fullname$NC" &&
    echo "${BLU}git checkout -b $fullname develop$NC" &&
    git checkout -b $fullname develop &&
    echo "${GRN}Now change the version number to $version and git commit with the message \"Bumped version number to $version\"$NC"
elif [ $action == "merge" ]; then
    echo "${GRN}Merging release branch $fullname to master$NC" &&
    echo "${BLU}git checkout master$NC" &&
    git checkout master &&
    echo "${BLU}git merge --no-ff $fullname$NC" &&
    git merge --no-ff $fullname &&
    echo "${BLU}git tag -a v$version -m \"Version $version\"$NC" &&
    git tag -a v$version -m "Version $version" &&
    echo "${BLU}git push$NC" &&
    git push &&
    echo "${BLU}git push origin v$version$NC" &&
    git push origin v$version &&
    echo "${GRN}Merging release branch $fullname to develop$NC" &&
    echo "${BLU}git checkout develop$NC" &&
    git checkout develop &&
    echo "${BLU}git merge --no-ff $fullname$NC" &&
    git merge --no-ff $fullname &&
    echo "${BLU}git branch -d $fullname$NC" &&
    git branch -d $fullname &&
    echo "${BLU}git push$NC" &&
    git push &&
    echo "${GRN}Version $version was successfully released!$NC"
else
    echo $usage
    exit 3
fi
