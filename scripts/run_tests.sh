#!/bin/sh

base_path=.
test_path="$base_path/libs/numeric/ublasx/test"

test_files=$(ls $test_path/*.cpp)

for f in $test_files; do
	f=$(basename $f .cpp)

	t="$test_path/$f"

	if [ -x $t ]; then
		echo -n "--- $f ==> "
		out=$($t 2>&1 | grep -i 'failed test')

		echo $out
	fi
done
