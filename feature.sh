#!/bin/sh

usage="Usage: $0 ACTION NAME\n  ACTION = new, merge, private\n  NAME = name of the feature"

if [ $# -ne 2 ]; then
    echo $usage
    exit 1
else
    action=$1
    name=$2
fi

if [ $name == "help" ] || [ $name == "-h" ] || [ $name == "--help" ]; then
    echo $usage
    exit 2
fi


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
    echo "${GRN}Making new feature branch $name$NC" &&
    echo "${BLU}git checkout -b $name develop$NC" &&
    git checkout -b $name develop &&
    echo "${BLU}git push origin $name$NC" &&
    git push origin $name &&
    echo "${GRN}Now add your new feature$NC"
elif [ $action == "private" ]; then
    echo "${GRN}Making new feature branch $name$NC" &&
    echo "${BLU}git checkout -b $name develop$NC" &&
    git checkout -b $name develop &&
    echo "${GRN}Now add your new feature$NC"
elif [ $action == "merge" ]; then
    echo "${GRN}Merging feature branch $name to develop$NC" &&
    echo "${BLU}git checkout develop$NC" &&
    git checkout develop &&
    echo "${BLU}git merge --no-ff $name$NC" &&
    git merge --no-ff $name &&
    echo "${BLU}git branch -d $name$NC" &&
    git branch -d $name &&
    echo "${BLU}git push -d origin $name$NC" &&
    git push -d origin $name &&
    echo "${BLU}git push$NC" &&
    git push &&
    echo "${GRN}Feature $name successfully added for next release!$NC"
else
    echo $usage
    exit 3
fi
