#!/usr/bin/env bash

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

# toolset:---------------------------------------------------------------------

command 2> >(while read line; do echo -e "\e[01;31m$line\e[0m" >&2; done)

function lineprint {
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' =
}

function message {
    lineprint
    printf "$1\n"
    lineprint
}

function error_message {
    lineprint
    printf "$1\n" >&2
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
        error_message "$current_action failed!"
    fi
    return $status
}

function perform_and_exit {
    perform "$@" || exit 1
}

# -----------------------------------------------------------------------------

INPUT=$1
OUTPUT=$2
if [ $# -ne 2 ]
then
    error_message "Error: no input file given. Usage: $0 <Filepath> <outputfile>"
    exit 1
fi

set_action "processing all files in $INPUT and write to $OUTPUT"

perform_and_exit export elist=\"`head -c -1 "$SCRIPTPATH/emoji-list.txt" | tr '\n' ',' | sed 's/,/\",\"/g'`\"
perform_and_exit echo "filter by emoji list:"
perform_and_exit echo $elist | tr -d '"' | tr -d ','

#perform_and_exit find ./ -type f -name '*.bz2' -exec bzip2 -dc "{}" \; | jq ". | {id: .id, datetime: .created_at, person: .user.name, text: .text} | select(.text != null) | [select(.text | contains($elist))] | select(any)| unique_by(.id) | .[]" | tee /dev/tty > "$OUTPUT"
perform_and_exit find ./ -type f -name '*.bz2' -exec bzip2 -dc "{}" \; | jq ". | {id: .id, datetime: .created_at, person: .user.id, text: .text, lang: .lang, reply_to: .in_reply_to_status_id} | select(.text != null)" | grep --no-group-separator -Ff "$SCRIPTPATH/emoji-list.txt" -A 3 -B 4 | tee /dev/tty > "$OUTPUT"

# ↑ such obvious, much selfexplaining 💁😈

confirm_action
