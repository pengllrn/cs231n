#！/bin/bash
git add *
git commit -m "$1"
git push origin master
git push github master