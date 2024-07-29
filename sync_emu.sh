#!/bin/bash

# Script to sync folders between local and remote with rsync

LOCAL_FOLDER="/media/blake/Files1/emu"
REMOTE_USER="bk639"
REMOTE_HOST="emu.physics.rutgers.edu"
REMOTE_FOLDER="/home/bk639/MorphologyMeasurements"
DRY_RUN=1
ACTION=""

for i in "$@"; do
case $i in
    --upload)
    ACTION="upload"
    shift # past argument=value
    ;;
    --download)
    ACTION="download"
    shift # past argument=value
    ;;
    --ignore-dry-run)
    DRY_RUN=0
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

if [ "$ACTION" == "" ]; then
    echo "You must specify --upload or --download."
    exit 1
fi

if [ $DRY_RUN -eq 1 ]; then
    echo "Performing a dry run. No changes will be made."
    OPTION="--dry-run"
else
    OPTION=""
fi

confirm() {
    # call with a prompt string or use a default
    read -r -p "${1:-Are you sure you want to proceed with the actual sync? [y/N]} " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            true
            ;;
        *)
            false
            ;;
    esac
}

execute_sync() {
    if [ "$ACTION" == "upload" ]; then
        rsync -avz $OPTION "$LOCAL_FOLDER/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_FOLDER"
    elif [ "$ACTION" == "download" ]; then
        rsync -avz $OPTION "$REMOTE_USER@$REMOTE_HOST:$REMOTE_FOLDER/" "$LOCAL_FOLDER"
    fi
}

# Perform the dry run if needed
if [ $DRY_RUN -eq 1 ]; then
    execute_sync
    if confirm ; then
        OPTION="" # Clear the dry run option for the actual sync
        execute_sync
    else
        echo "Sync canceled."
    fi
else
    execute_sync
fi
