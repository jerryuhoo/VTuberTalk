find . | grep 'mp3' | nl -nrz -w3 -v1 | while read n f; do mv "$f" "$n.mp3"; done
