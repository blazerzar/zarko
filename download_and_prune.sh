#!/bin/bash

# Get the current date
current_date=$(date +%Y%m%d)

start_date="20220619"
location="maribor_letalisce"

# Loop through the dates
while [[ $start_date < $current_date ]]; do
    echo "Downloading photos for date: $start_date"

    wget -O ./temp/tmp.tar "https://apis-g.arso.gov.si/s3/webcam-arhiv/public/$location/$location"_"$start_date.tar?api-key=94ac9793-8fb8-47ea-83c8-765ea1852bf7"
    
    # Check if tar file exists
    if [ ! -f ./temp/tmp.tar ]; then
        echo "Tar file not found for date: $start_date"
        exit 0
    fi

    # Extract tar file
    tar -xf ./temp/tmp.tar -C ./temp 
    
    # Remove photos that do not end in "36.jpg"
    ls ./temp/ | grep -v "36.jpg" | xargs -I {} rm -f ./temp/{}

    # Move the photos to the data directory
    mv ./temp/* ./data/

    # Increment the date by 1 day
    start_date=$(date -d "$start_date + 10 day" +%Y%m%d)

    sleep 1

done
