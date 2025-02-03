#!/bin/bash
echo
echo '1. STOPPING APP DOCKER: stop prediction app to rebuild'
echo
docker compose stop

echo
echo '2. REBUILDING APP DOCKER: prediction app to rebuild'
echo
docker compose build 

echo
echo '3. STARTING APP DOCKER: prediction app available on port 8501'
echo
docker compose up &
