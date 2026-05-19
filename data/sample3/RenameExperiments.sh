#!/bin/sh

for i in 0 1 2 3 4 5 6 7 8 9
do
    old="captura $i.csv"
    new="experiment$i.csv"

    if [ -f "$old" ]; then
        echo "Renaming: $old -> $new"
        mv "$old" "$new"
    else
        echo "File not found: $old"
    fi
done
