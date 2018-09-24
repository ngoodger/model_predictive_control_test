#!/bin/bash
# Alias for running python black and flake8 through docker.
# ABSOLUTE_PROJECT_SRC_DIR should be set to your Project Source Directory. 

ABSOLUTE_PROJECT_SRC_DIR_DOCKER=""
ABSOLUTE_PROJECT_SRC_DIR_MOUNTS=""
CHECK_OPTION="--check"
# Generate the src paths for within docker.
for src_dir in $ABSOLUTE_PROJECT_SRC_DIR
do
    ABSOLUTE_PROJECT_SRC_DIR_DOCKER="$ABSOLUTE_PROJECT_SRC_DIR_DOCKER $src_dir"
done
# Generate the needed mounts to add the source directories.
for src_dir in $ABSOLUTE_PROJECT_SRC_DIR
do
    ABSOLUTE_PROJECT_SRC_DIR_MOUNTS="$ABSOLUTE_PROJECT_SRC_DIR_MOUNTS -v $src_dir:$src_dir"
done
function format_python ()
{
    docker run -e SRC="${ABSOLUTE_PROJECT_SRC_DIR_DOCKER}" ${ABSOLUTE_PROJECT_SRC_DIR_MOUNTS} python_black:latest
}
function format_python_check ()
{
    docker run -e SRC="${ABSOLUTE_PROJECT_SRC_DIR_DOCKER}" -e OPTIONS=$CHECK_OPTION ${ABSOLUTE_PROJECT_SRC_DIR_MOUNTS} python_black:latest
}
