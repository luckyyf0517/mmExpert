#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <keyword>"
    exit 1
fi

KEYWORD="$1"

ps -ef | grep "$KEYWORD" | grep -v grep | awk '{print $2}' | xargs kill -9