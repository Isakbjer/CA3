
# Note that there cannot be space before or after the equal sign in variable assignment in BASH
USR_NAME="dat300-h24-3"
HOST_NAME="${USR_NAME}@filemanager.orion.nmbu.no"

# Note here that we have an extra slash at the end of the directory we are copying from, and we don't have it at the end of the directory we are copying to
CURRENT_DIR=$(pwd)
LOCAL_DIR="${CURRENT_DIR}/training_scripts/"
REMOTE_DIR="${HOST_NAME}:~/training_scripts"

rsync -avzP "$LOCAL_DIR" "$REMOTE_DIR"

# a: Archive mode (allows transfer of directories and not only files)
# v: Verbose mode (command will output more details about what it's doing, such as listing the files being transferred or skipped)
# z: Compress     (compresses the data before transferring it, which is faster)
# P: Progress     (shows the progress of the transfer, and allows you to resume a transfer if it's interrupted)
