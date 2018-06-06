#!/usr/bin/env bash

# helper functions:

function lineprint {
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' =
}

function message {
    lineprint
    printf "$1\n"
    lineprint
}

current_action="IDLE"

function confirm_action {
    message "successfully finished action: $current_action"
}

function set_action {
    current_action="$1"
    message "$1"
}

function perform {
    "$@"
    local status=$?
    if [ $status -ne 0 ]
    then
        message "$current_action failed!"
    fi
    return $status
}

function perform_and_exit {
    perform "$@" || exit
}

# Downloading and unzipping dataset

D1_URL=https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
D1_ZIP=${D1_URL##*/}

D2_URL=https://raw.githubusercontent.com/GeorgeMcIntire/fake_real_news_dataset/master/fake_or_real_news.csv.zip
D2_ZIP=${D2_URL##*/}

P3_URL=https://raw.githubusercontent.com/SmartDataAnalytics/MA-INF-4222-NLP-Lab/master/2018_SoSe/exercises/script_dataset3.py
P3_SCRIPT=${P3_URL##*/}

set_action "checking whether unzip is installed"
# testing for unzip:
perform_and_exit unzip -v
confirm_action

set_action "downloading and unpacking $D1_URL if not already existing"

perform_and_exit mkdir -p ./data
perform_and_exit cd ./data/

if [ ! -e $D1_ZIP ];
then
    perform_and_exit curl $D1_URL --output ./$D1_ZIP
    perform_and_exit unzip $D1_ZIP
fi

confirm_action

set_action "downloading and unpacking $D2_URL if not already existing"

if [ ! -e $D2_ZIP ];
then
    perform_and_exit curl $D2_URL --output ./$D2_ZIP
    perform_and_exit unzip $D2_ZIP
fi

confirm_action

set_action "downloading Helper script: $P3_SCRIPT"

if [ ! -e $P3_SCRIPT ];
then
    perform_and_exit curl $P3_URL --output ./$P3_SCRIPT
fi

confirm_action
